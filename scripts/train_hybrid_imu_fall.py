"""Retrain IMU fall LSTM and tune Impact/Rotation/Inactivity correction."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from train_sisfall_merged_imu_lstm import (
    FEATURE_COLUMNS,
    BinaryLSTM,
    SequenceDataset,
    grouped_metrics,
    load_merged_csv,
    make_sequences,
    metrics,
    predict,
    scale_split,
)


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
    if requested == "mps" and not torch.backends.mps.is_available():
        print("MPS was requested but is not available. Falling back to CPU.")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        print("CUDA was requested but is not available. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device(requested)


def best_threshold(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, dict[str, Any]]:
    best_t = 0.5
    best_m = metrics(y_true, scores, best_t)
    for threshold in np.linspace(0.05, 0.95, 91):
        current = metrics(y_true, scores, float(threshold))
        if (current["f1"], current["recall"], current["accuracy"]) > (
            best_m["f1"],
            best_m["recall"],
            best_m["accuracy"],
        ):
            best_t = float(threshold)
            best_m = current
    return best_t, best_m


def algorithm_scores(
    x_raw: np.ndarray,
    impact_g: float,
    freefall_g: float,
    gyro_dps: float,
    tilt_deg: float,
    still_accel_std: float,
    still_gyro_dps: float,
    post_samples: int,
) -> tuple[np.ndarray, dict[str, float]]:
    index = {name: FEATURE_COLUMNS.index(name) for name in FEATURE_COLUMNS}
    accel = x_raw[:, :, index["accel_norm"]]
    gyro = x_raw[:, :, index["gyro_norm"]]
    roll = x_raw[:, :, index["roll"]]
    pitch = x_raw[:, :, index["pitch"]]

    impact_peak = np.nanmax(accel, axis=1)
    freefall_min = np.nanmin(accel, axis=1)
    gyro_peak = np.nanmax(gyro, axis=1)
    tilt_change = np.nanmax(
        np.sqrt((roll - roll[:, :1]) ** 2 + (pitch - pitch[:, :1]) ** 2),
        axis=1,
    )
    impact_index = np.nanargmax(accel, axis=1)

    post_accel_std = np.zeros(len(x_raw), dtype=np.float32)
    post_gyro_mean = np.zeros(len(x_raw), dtype=np.float32)
    inactivity = np.zeros(len(x_raw), dtype=np.float32)
    for i, idx in enumerate(impact_index):
        start = int(idx) + 1
        end = min(x_raw.shape[1], start + post_samples)
        if start >= x_raw.shape[1]:
            segment_accel = accel[i, -max(2, post_samples) :]
            segment_gyro = gyro[i, -max(2, post_samples) :]
        else:
            segment_accel = accel[i, start:end]
            segment_gyro = gyro[i, start:end]
        post_accel_std[i] = float(np.nanstd(segment_accel)) if len(segment_accel) else 0.0
        post_gyro_mean[i] = float(np.nanmean(segment_gyro)) if len(segment_gyro) else 0.0
        inactivity[i] = float(post_accel_std[i] <= still_accel_std and post_gyro_mean[i] <= still_gyro_dps)

    impact_score = np.clip((impact_peak - impact_g) / max(impact_g, 1e-6), 0.0, 1.0)
    freefall_score = (freefall_min <= freefall_g).astype(np.float32)
    rotation_score = np.maximum(
        np.clip(gyro_peak / max(gyro_dps, 1e-6), 0.0, 1.0),
        np.clip(tilt_change / max(tilt_deg, 1e-6), 0.0, 1.0),
    )
    algorithm = (
        0.40 * impact_score
        + 0.20 * rotation_score
        + 0.25 * inactivity
        + 0.15 * np.maximum(freefall_score, np.clip(tilt_change / max(tilt_deg, 1e-6), 0.0, 1.0))
    )
    algorithm = np.clip(algorithm, 0.0, 1.0).astype(np.float32)
    diagnostics = {
        "impact_peak_mean": float(np.mean(impact_peak)),
        "impact_peak_p95": float(np.percentile(impact_peak, 95)),
        "gyro_peak_p95": float(np.percentile(gyro_peak, 95)),
        "tilt_change_p95": float(np.percentile(tilt_change, 95)),
        "inactivity_rate": float(np.mean(inactivity)),
    }
    return algorithm, diagnostics


def tune_hybrid(
    y_true: np.ndarray,
    lstm_scores: np.ndarray,
    algo_scores: np.ndarray,
) -> tuple[float, float, dict[str, Any]]:
    best_weight = 0.65
    best_threshold_value = 0.5
    best_metrics = metrics(y_true, 0.65 * lstm_scores + 0.35 * algo_scores, 0.5)
    for lstm_weight in np.linspace(0.0, 1.0, 21):
        final_scores = lstm_weight * lstm_scores + (1.0 - lstm_weight) * algo_scores
        threshold, current = best_threshold(y_true, final_scores)
        if (current["f1"], current["recall"], current["accuracy"]) > (
            best_metrics["f1"],
            best_metrics["recall"],
            best_metrics["accuracy"],
        ):
            best_weight = float(lstm_weight)
            best_threshold_value = float(threshold)
            best_metrics = current
    return best_weight, best_threshold_value, best_metrics


def train(args: argparse.Namespace) -> dict[str, Any]:
    set_seed(args.seed)
    frame = load_merged_csv(args.source)
    raw_split = make_sequences(
        frame,
        args.sequence_length,
        args.sequence_stride,
        args.train_ratio,
        args.validation_ratio,
        args.seed,
    )
    split, scaler = scale_split(raw_split)
    device = resolve_device(args.device)
    print(f"training_device={device}")
    print(
        "dataset="
        f"{args.source}, train={len(split['y_train'])}, validation={len(split['y_validation'])}, test={len(split['y_test'])}"
    )

    model = BinaryLSTM(
        len(FEATURE_COLUMNS),
        args.hidden_size,
        args.num_layers,
        args.dropout,
        args.bidirectional,
        args.pooling,
    ).to(device)
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
        validation_scores = predict(model, split["x_validation"], args.batch_size, device)
        _, validation_metrics = best_threshold(split["y_validation"], validation_scores)
        history.append({"epoch": epoch, "loss": total / max(1, count), "validation_f1": validation_metrics["f1"]})
        print(
            f"hybrid_imu_fall epoch={epoch:03d} loss={total / max(1, count):.6f} "
            f"val_f1={validation_metrics['f1']:.4f}"
        )
        if validation_metrics["f1"] > best_f1:
            best_f1 = validation_metrics["f1"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    validation_lstm = predict(model, split["x_validation"], args.batch_size, device)
    test_lstm = predict(model, split["x_test"], args.batch_size, device)
    lstm_threshold, validation_lstm_metrics = best_threshold(split["y_validation"], validation_lstm)
    test_lstm_metrics = metrics(split["y_test"], test_lstm, lstm_threshold)

    validation_algo, validation_algo_diag = algorithm_scores(
        raw_split["x_validation"],
        args.impact_g,
        args.freefall_g,
        args.gyro_dps,
        args.tilt_deg,
        args.still_accel_std,
        args.still_gyro_dps,
        args.post_samples,
    )
    test_algo, test_algo_diag = algorithm_scores(
        raw_split["x_test"],
        args.impact_g,
        args.freefall_g,
        args.gyro_dps,
        args.tilt_deg,
        args.still_accel_std,
        args.still_gyro_dps,
        args.post_samples,
    )
    algo_threshold, validation_algo_metrics = best_threshold(split["y_validation"], validation_algo)
    test_algo_metrics = metrics(split["y_test"], test_algo, algo_threshold)
    hybrid_lstm_weight, hybrid_threshold, validation_hybrid_metrics = tune_hybrid(
        split["y_validation"],
        validation_lstm,
        validation_algo,
    )
    test_hybrid = hybrid_lstm_weight * test_lstm + (1.0 - hybrid_lstm_weight) * test_algo
    test_hybrid_metrics = metrics(split["y_test"], test_hybrid, hybrid_threshold)

    args.model_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.model_dir / "iccas_final_hybrid_lstm_imu_fall.pt"
    metadata_path = args.model_dir / "iccas_final_hybrid_lstm_imu_fall.json"
    checkpoint = {
        "model_type": "iccas_hybrid_binary_imu_fall_lstm",
        "task": "imu_fall",
        "positive_label": "fall",
        "feature_columns": FEATURE_COLUMNS,
        "sequence_length": args.sequence_length,
        "sequence_stride": args.sequence_stride,
        "sample_ms": args.sample_ms,
        "sample_rate_hz": 1000.0 / args.sample_ms,
        "threshold": hybrid_threshold,
        "lstm_threshold": lstm_threshold,
        "algorithm_threshold": algo_threshold,
        "hybrid_lstm_weight": hybrid_lstm_weight,
        "hybrid_algorithm_weight": 1.0 - hybrid_lstm_weight,
        "fall_algorithm": {
            "impact_g": args.impact_g,
            "freefall_g": args.freefall_g,
            "gyro_dps": args.gyro_dps,
            "tilt_deg": args.tilt_deg,
            "still_accel_std": args.still_accel_std,
            "still_gyro_dps": args.still_gyro_dps,
            "post_samples": args.post_samples,
            "cooldown_ms": args.cooldown_ms,
        },
        "scaler_center": scaler.center,
        "scaler_scale": scaler.scale,
        "hidden_size": args.hidden_size,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "bidirectional": args.bidirectional,
        "pooling": args.pooling,
        "model_state": model.state_dict(),
    }
    torch.save(checkpoint, model_path)

    metadata = {key: value for key, value in checkpoint.items() if key not in {"model_state", "scaler_center", "scaler_scale"}}
    metadata["scaler_center"] = scaler.center.tolist()
    metadata["scaler_scale"] = scaler.scale.tolist()
    metadata["source"] = str(args.source)
    metadata["device"] = str(device)
    metadata["dataset_change_points"] = [
        "직접 취득 데이터는 ICCAS_final_data.xlsx에서 data/iccas_sensor_lstm/iccas_final_labeled.csv로 전처리됨",
        "IMU 낙상 재학습은 data/iccas_sensor_lstm/final_iccas_sisfall_imu_merged.csv를 사용",
        "SisFall은 GPS가 없으므로 IMU/Gyro 낙상 학습에만 사용",
        "새 직접 취득 데이터는 final_iccas_sisfall_imu_merged.csv에 같은 컬럼으로 병합하면 학습셋 변경 가능",
    ]
    metadata["split_method"] = "SisFall source-file/group hash split; ICCAS chronological split inside each scenario"
    metadata["split_sizes"] = {
        "train": int(len(split["y_train"])),
        "validation": int(len(split["y_validation"])),
        "test": int(len(split["y_test"])),
    }
    metadata["label_counts"] = {
        "train_positive": int(split["y_train"].sum()),
        "train_negative": int(len(split["y_train"]) - split["y_train"].sum()),
        "validation_positive": int(split["y_validation"].sum()),
        "validation_negative": int(len(split["y_validation"]) - split["y_validation"].sum()),
        "test_positive": int(split["y_test"].sum()),
        "test_negative": int(len(split["y_test"]) - split["y_test"].sum()),
    }
    metadata["validation_metrics"] = {
        "lstm_only": validation_lstm_metrics,
        "algorithm_only": validation_algo_metrics,
        "hybrid": validation_hybrid_metrics,
    }
    metadata["test_metrics"] = {
        "lstm_only": test_lstm_metrics,
        "algorithm_only": test_algo_metrics,
        "hybrid": test_hybrid_metrics,
    }
    metadata["test_metrics_by_dataset"] = grouped_metrics(
        split["y_test"], test_hybrid, split["meta_test"], hybrid_threshold, "source_dataset"
    )
    metadata["diagnostics"] = {
        "validation_algorithm": validation_algo_diag,
        "test_algorithm": test_algo_diag,
    }
    metadata["history"] = history
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    report = {
        "source": str(args.source),
        "model_path": str(model_path),
        "metadata_path": str(metadata_path),
        "report_path": str(args.report),
        "sample_ms": args.sample_ms,
        "sequence_length": args.sequence_length,
        "buffer_seconds": args.sample_ms * args.sequence_length / 1000.0,
        "fall_algorithm": metadata["fall_algorithm"],
        "split_sizes": metadata["split_sizes"],
        "label_counts": metadata["label_counts"],
        "validation_metrics": metadata["validation_metrics"],
        "test_metrics": metadata["test_metrics"],
        "test_metrics_by_dataset": metadata["test_metrics_by_dataset"],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(report, args.markdown)
    return report


def write_markdown(report: dict[str, Any], markdown_path: Path) -> None:
    test = report["test_metrics"]
    lines = [
        "# Hybrid IMU Fall 재학습 결과",
        "",
        "## 사용 데이터셋",
        "",
        f"- Source: `{report['source']}`",
        "- 데이터 구성: ICCAS 직접 취득 IMU 데이터 + SisFall IMU 낙상 데이터",
        "- GPS는 사용하지 않음. IMU/Gyro 낙상 탐지만 재학습함.",
        "",
        "## 직접 취득 임계값 반영",
        "",
        f"- SAMPLE_MS: `{report['sample_ms']}` ms",
        f"- IMU_BUF_N / sequence_length: `{report['sequence_length']}`",
        f"- Buffer seconds: `{report['buffer_seconds']:.2f}` s",
        f"- FALL_IMPACT_G: `{report['fall_algorithm']['impact_g']}` g",
        f"- FALL_FREE_G: `{report['fall_algorithm']['freefall_g']}` g",
        f"- FALL_COOLDOWN: `{report['fall_algorithm']['cooldown_ms']}` ms",
        "",
        "## 성능 비교",
        "",
        "| Method | Accuracy | Precision | Recall | F1-score | TP | FP | TN | FN |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, item in test.items():
        lines.append(
            f"| {name} | {item['accuracy']:.4f} | {item['precision']:.4f} | {item['recall']:.4f} | "
            f"{item['f1']:.4f} | {item['tp']} | {item['fp']} | {item['tn']} | {item['fn']} |"
        )
    lines.extend(
        [
            "",
            "## 모델 파일",
            "",
            f"- Model: `{report['model_path']}`",
            f"- Metadata: `{report['metadata_path']}`",
            f"- JSON report: `{report['report_path']}`",
            "",
            "## 해석",
            "",
            "- `lstm_only`는 LSTM 확률만 사용한 결과입니다.",
            "- `algorithm_only`는 impact/rotation/inactivity 알고리즘만 사용한 결과입니다.",
            "- `hybrid`는 LSTM 점수와 알고리즘 점수를 validation F1 기준으로 결합한 결과입니다.",
        ]
    )
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("../data/iccas_sensor_lstm/final_iccas_sisfall_imu_merged.csv"))
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--report", type=Path, default=Path("../data/iccas_sensor_lstm/hybrid_imu_fall_metrics.json"))
    parser.add_argument("--markdown", type=Path, default=Path("docs/HYBRID_IMU_FALL_RETRAINING_REPORT.md"))
    parser.add_argument("--sample-ms", type=int, default=40)
    parser.add_argument("--sequence-length", type=int, default=50)
    parser.add_argument("--sequence-stride", type=int, default=4)
    parser.add_argument("--impact-g", type=float, default=2.5)
    parser.add_argument("--freefall-g", type=float, default=0.6)
    parser.add_argument("--cooldown-ms", type=int, default=5000)
    parser.add_argument("--gyro-dps", type=float, default=250.0)
    parser.add_argument("--tilt-deg", type=float, default=45.0)
    parser.add_argument("--still-accel-std", type=float, default=0.35)
    parser.add_argument("--still-gyro-dps", type=float, default=80.0)
    parser.add_argument("--post-samples", type=int, default=25)
    parser.add_argument("--hidden-size", type=int, default=96)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--validation-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--bidirectional", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pooling", choices=["last", "mean", "attention"], default="attention")
    return parser.parse_args()


def main() -> None:
    report = train(parse_args())
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
