"""Compare GPS wandering RNN performance with and without Kalman filtering."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from realtime_sensor_lstm import build_features, latlon_to_xy, read_source
from train_parallel_sensor_lstm import GPS_FEATURES, SCENARIOS
from train_specialized_sensor_lstm import RobustScaler, best_threshold, make_sequences, metrics
from train_gps_rnn_wandering import BinaryRNN, predict, scenario_metrics


class SequenceDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[index], self.y[index]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(requested)


def kalman_filter_xy(
    x: np.ndarray,
    y: np.ndarray,
    dt: np.ndarray,
    measurement_noise_m: float,
    process_noise: float,
) -> tuple[np.ndarray, np.ndarray]:
    state = np.array([x[0], y[0], 0.0, 0.0], dtype=np.float64)
    covariance = np.eye(4, dtype=np.float64) * 10.0
    measurement_matrix = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float64)
    measurement_covariance = np.eye(2, dtype=np.float64) * (measurement_noise_m ** 2)
    identity = np.eye(4, dtype=np.float64)
    out_x = np.zeros_like(x, dtype=np.float64)
    out_y = np.zeros_like(y, dtype=np.float64)
    for index, (raw_x, raw_y, delta_t) in enumerate(zip(x, y, dt)):
        delta_t = float(np.clip(delta_t, 0.02, 5.0))
        transition = np.array(
            [
                [1.0, 0.0, delta_t, 0.0],
                [0.0, 1.0, 0.0, delta_t],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        process_covariance = np.diag(
            [
                process_noise * delta_t**2,
                process_noise * delta_t**2,
                process_noise * delta_t,
                process_noise * delta_t,
            ]
        )
        state = transition @ state
        covariance = transition @ covariance @ transition.T + process_covariance
        measurement = np.array([raw_x, raw_y], dtype=np.float64)
        innovation = measurement - measurement_matrix @ state
        innovation_covariance = measurement_matrix @ covariance @ measurement_matrix.T + measurement_covariance
        kalman_gain = covariance @ measurement_matrix.T @ np.linalg.inv(innovation_covariance)
        state = state + kalman_gain @ innovation
        covariance = (identity - kalman_gain @ measurement_matrix) @ covariance
        out_x[index] = state[0]
        out_y[index] = state[1]
    return out_x, out_y


def load_raw_frames(source: Path) -> list[pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    for sheet, label in SCENARIOS:
        frame = read_source(source, sheet_name=sheet).copy()
        frame["scenario"] = sheet
        frame["class_label"] = label
        frame["label"] = label
        frames.append(frame)
    return frames


def add_features(
    raw_frames: list[pd.DataFrame],
    use_kalman: bool,
    measurement_noise_m: float,
    process_noise: float,
) -> list[pd.DataFrame]:
    combined = pd.concat(raw_frames, ignore_index=True)
    origin_lat = float(combined["lat"].dropna().median())
    origin_lng = float(combined["lng"].dropna().median())
    featured_frames: list[pd.DataFrame] = []
    for frame in raw_frames:
        featured, _, _ = build_features(frame, origin_lat=origin_lat, origin_lng=origin_lng)
        if use_kalman:
            parts: list[pd.DataFrame] = []
            for _, group in featured.groupby("device", sort=False):
                group = group.sort_values("server_time").copy()
                lat = group["lat"].ffill().bfill().fillna(origin_lat).to_numpy(dtype=np.float64)
                lng = group["lng"].ffill().bfill().fillna(origin_lng).to_numpy(dtype=np.float64)
                x_raw, y_raw = latlon_to_xy(lat, lng, origin_lat, origin_lng)
                dt = group["dt_s"].to_numpy(dtype=np.float64)
                x_filtered, y_filtered = kalman_filter_xy(x_raw, y_raw, dt, measurement_noise_m, process_noise)
                dx = np.diff(x_filtered, prepend=x_filtered[0])
                dy = np.diff(y_filtered, prepend=y_filtered[0])
                speed = np.hypot(dx, dy) / np.clip(dt, 0.02, 30.0)
                group["x_m"] = x_filtered
                group["y_m"] = y_filtered
                group["dx_m"] = dx
                group["dy_m"] = dy
                group["speed_mps"] = np.clip(speed, 0.0, 25.0)
                parts.append(group)
            featured = pd.concat(parts, ignore_index=True)
        featured["scenario"] = frame["scenario"].iloc[0]
        featured["class_label"] = frame["class_label"].iloc[0]
        featured_frames.append(featured)
    return featured_frames


def scale_split(split: dict[str, Any]) -> tuple[dict[str, Any], RobustScaler]:
    scaler = RobustScaler.fit(split["x_train"].reshape(-1, split["x_train"].shape[-1]))
    out = dict(split)
    for key in ["x_train", "x_val", "x_test"]:
        x = split[key]
        out[key] = scaler.transform(x.reshape(-1, x.shape[-1])).reshape(x.shape)
    return out, scaler


def train_one(name: str, frames: list[pd.DataFrame], args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    split = make_sequences(frames, GPS_FEATURES, "wandering", args.sequence_length, args.train_ratio, args.validation_ratio)
    split, _ = scale_split(split)
    model = BinaryRNN(len(GPS_FEATURES), args.hidden_size, args.num_layers, args.dropout).to(device)
    pos = float(split["y_train"].sum())
    neg = float(len(split["y_train"]) - pos)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    loader = DataLoader(SequenceDataset(split["x_train"], split["y_train"]), batch_size=args.batch_size, shuffle=True)
    best_state: dict[str, torch.Tensor] | None = None
    best_f1 = -1.0
    history: list[dict[str, float]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        count = 0
        for sequences, labels in loader:
            sequences = sequences.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(sequences), labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.item()) * len(labels)
            count += len(labels)
        validation_scores = predict(model, split["x_val"], args.batch_size, device)
        _, validation_metrics = best_threshold(split["y_val"], validation_scores)
        history.append({"epoch": epoch, "loss": total / max(1, count), "validation_f1": validation_metrics["f1"]})
        print(f"{name} epoch={epoch:03d} loss={total / max(1, count):.6f} val_f1={validation_metrics['f1']:.4f}")
        if validation_metrics["f1"] > best_f1:
            best_f1 = validation_metrics["f1"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    validation_scores = predict(model, split["x_val"], args.batch_size, device)
    threshold, validation_metrics = best_threshold(split["y_val"], validation_scores)
    test_scores = predict(model, split["x_test"], args.batch_size, device)
    test_metrics = metrics(split["y_test"], test_scores, threshold)
    return {
        "name": name,
        "threshold": threshold,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "test_metrics_by_scenario": scenario_metrics(split["y_test"], test_scores, split["s_test"], threshold),
        "history": history,
        "split_sizes": {"train": len(split["y_train"]), "validation": len(split["y_val"]), "test": len(split["y_test"])},
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    raw = report["results"]["raw"]
    kalman = report["results"]["kalman"]
    raw_m = raw["test_metrics"]
    kalman_m = kalman["test_metrics"]
    delta = report["delta"]
    lines = [
        "# GPS Kalman Filter RNN 성능 비교",
        "",
        "## 실험 조건",
        "",
        f"- Source: `{report['source']}`",
        "- Model: GPS Wandering RNN",
        f"- Sequence length: `{report['sequence_length']}`",
        f"- Measurement noise: `{report['kalman']['measurement_noise_m']}` m",
        f"- Process noise: `{report['kalman']['process_noise']}`",
        "",
        "## 결과",
        "",
        "| Version | Accuracy | Precision | Recall | F1-score | Threshold |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        f"| Raw GPS | {raw_m['accuracy']:.4f} | {raw_m['precision']:.4f} | {raw_m['recall']:.4f} | {raw_m['f1']:.4f} | {raw['threshold']:.2f} |",
        f"| Kalman GPS | {kalman_m['accuracy']:.4f} | {kalman_m['precision']:.4f} | {kalman_m['recall']:.4f} | {kalman_m['f1']:.4f} | {kalman['threshold']:.2f} |",
        "",
        "## 변화량",
        "",
        f"- Accuracy: `{delta['accuracy']:+.4f}`",
        f"- Precision: `{delta['precision']:+.4f}`",
        f"- Recall: `{delta['recall']:+.4f}`",
        f"- F1-score: `{delta['f1']:+.4f}`",
        "",
        "## 해석",
        "",
        "칼만 필터는 GPS 좌표의 순간 튐을 줄여 `x_m`, `y_m`, `dx_m`, `dy_m`, `speed_mps`를 부드럽게 만듭니다.",
        "다만 배회 감지는 경로 이탈과 반복 이동의 형태도 중요하므로, 필터가 너무 강하면 실제 이동 변화까지 완화되어 성능이 내려갈 수 있습니다.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("../ICCAS_final_data.xlsx"))
    parser.add_argument("--report", type=Path, default=Path("../data/iccas_sensor_lstm/gps_kalman_rnn_comparison.json"))
    parser.add_argument("--markdown", type=Path, default=Path("docs/GPS_KALMAN_RNN_COMPARISON.md"))
    parser.add_argument("--sequence-length", type=int, default=16)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--validation-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--measurement-noise-m", type=float, default=5.0)
    parser.add_argument("--process-noise", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    raw_frames = load_raw_frames(args.source)
    raw_featured = add_features(raw_frames, False, args.measurement_noise_m, args.process_noise)
    kalman_featured = add_features(raw_frames, True, args.measurement_noise_m, args.process_noise)
    raw_result = train_one("raw_gps_rnn", raw_featured, args, device)
    set_seed(args.seed)
    kalman_result = train_one("kalman_gps_rnn", kalman_featured, args, device)
    raw_metrics = raw_result["test_metrics"]
    kalman_metrics = kalman_result["test_metrics"]
    delta = {
        key: float(kalman_metrics[key] - raw_metrics[key])
        for key in ["accuracy", "precision", "recall", "f1"]
    }
    report = {
        "source": str(args.source),
        "device": str(device),
        "sequence_length": args.sequence_length,
        "kalman": {
            "measurement_noise_m": args.measurement_noise_m,
            "process_noise": args.process_noise,
        },
        "results": {"raw": raw_result, "kalman": kalman_result},
        "delta": delta,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(report, args.markdown)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
