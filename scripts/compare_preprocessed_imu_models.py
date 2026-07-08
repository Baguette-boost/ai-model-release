"""Compare RNN, GRU, LSTM, and Transformer on the preprocessed IMU fall CSV."""

from __future__ import annotations

import argparse
import html
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


FEATURE_COLUMNS = [
    "roll",
    "pitch",
    "yaw",
    "ax",
    "ay",
    "az",
    "wx",
    "wy",
    "wz",
    "accel_norm",
    "gyro_norm",
    "dt_s",
]


class SequenceDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[index], self.y[index]


class RobustScaler:
    def __init__(self, center: np.ndarray, scale: np.ndarray) -> None:
        self.center = center.astype(np.float32)
        self.scale = scale.astype(np.float32)

    @classmethod
    def fit(cls, values: np.ndarray) -> "RobustScaler":
        center = np.nanmedian(values, axis=0)
        q25 = np.nanpercentile(values, 25, axis=0)
        q75 = np.nanpercentile(values, 75, axis=0)
        scale = q75 - q25
        std = np.nanstd(values, axis=0)
        scale = np.where(scale > 1e-6, scale, std)
        scale = np.where(scale > 1e-6, scale, 1.0)
        return cls(center, scale)

    def transform(self, values: np.ndarray) -> np.ndarray:
        scaled = (values.astype(np.float32) - self.center) / self.scale
        return np.clip(scaled, -12.0, 12.0).astype(np.float32)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512) -> None:
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class BinarySequenceModel(nn.Module):
    def __init__(
        self,
        architecture: str,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        transformer_heads: int,
    ) -> None:
        super().__init__()
        self.architecture = architecture
        if architecture in {"rnn", "gru", "lstm"}:
            recurrent_cls = {"rnn": nn.RNN, "gru": nn.GRU, "lstm": nn.LSTM}[architecture]
            self.encoder = recurrent_cls(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0.0,
                batch_first=True,
            )
            self.head = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, 1))
        elif architecture == "transformer":
            self.input_projection = nn.Linear(input_size, hidden_size)
            self.position = PositionalEncoding(hidden_size)
            layer = nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=transformer_heads,
                dim_feedforward=hidden_size * 2,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
            self.head = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, 1))
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.architecture in {"rnn", "gru", "lstm"}:
            output, _ = self.encoder(x)
            pooled = output[:, -1, :]
        else:
            output = self.encoder(self.position(self.input_projection(x)))
            pooled = output.mean(dim=1)
        return self.head(pooled).squeeze(-1)


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


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
        print("MPS requested but unavailable. Falling back to CPU.")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device(requested)


def metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    pred = scores >= threshold
    truth = y_true.astype(bool)
    tp = int((truth & pred).sum())
    fp = int((~truth & pred).sum())
    tn = int((~truth & ~pred).sum())
    fn = int((truth & ~pred).sum())
    accuracy = (tp + tn) / max(1, tp + fp + tn + fn)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "threshold": threshold,
    }


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


def load_preprocessed_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    frame = frame.copy()
    for column in FEATURE_COLUMNS + ["t_ms"]:
        if column not in frame.columns:
            raise SystemExit(f"Missing required column in preprocessed CSV: {column}")
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
    for column, default in {
        "device": "unknown-device",
        "source_dataset": "unknown",
        "source_file": "unknown-file",
        "source_activity": "unknown-activity",
    }.items():
        if column not in frame.columns:
            frame[column] = default
        frame[column] = frame[column].fillna(default).astype(str)
    if "fall_target" in frame.columns:
        frame["fall_target"] = pd.to_numeric(frame["fall_target"], errors="coerce").fillna(0.0).astype(np.float32)
    else:
        frame["fall_target"] = (frame["label"].astype(str) == "fall").astype(np.float32)
    frame["group_id"] = (
        frame["source_dataset"]
        + "::"
        + frame["source_file"]
        + "::"
        + frame["device"]
        + "::"
        + frame["source_activity"]
    )
    return frame


def split_position(position: float, train_ratio: float, validation_ratio: float) -> str:
    if position < train_ratio:
        return "train"
    if position < train_ratio + validation_ratio:
        return "validation"
    return "test"


def make_sequences(frame: pd.DataFrame, args: argparse.Namespace) -> dict[str, Any]:
    buckets: dict[str, list[Any]] = {
        "x_train": [],
        "y_train": [],
        "x_validation": [],
        "y_validation": [],
        "x_test": [],
        "y_test": [],
    }
    for _, group in frame.groupby("group_id", sort=False):
        group = group.sort_values("t_ms", kind="mergesort")
        values = group[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
        labels = group["fall_target"].to_numpy(dtype=np.float32)
        if len(values) < args.sequence_length:
            continue
        source_dataset = str(group["source_dataset"].iloc[0])
        for end in range(args.sequence_length, len(values) + 1, args.sequence_stride):
            start = end - args.sequence_length
            y_value = float(labels[start:end].max())
            position_split = split_position(end / len(values), args.train_ratio, args.validation_ratio)
            split = position_split if source_dataset == "ICCAS" else group["split"].iloc[0]
            buckets[f"x_{split}"].append(values[start:end])
            buckets[f"y_{split}"].append(y_value)
    out: dict[str, Any] = {}
    for split in ["train", "validation", "test"]:
        out[f"x_{split}"] = np.stack(buckets[f"x_{split}"]).astype(np.float32)
        out[f"y_{split}"] = np.asarray(buckets[f"y_{split}"], dtype=np.float32)
    return out


def assign_sisfall_group_splits(frame: pd.DataFrame, train_ratio: float, validation_ratio: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    group_ids = sorted(frame["group_id"].unique())
    shuffled = list(group_ids)
    rng.shuffle(shuffled)
    train_cut = int(len(shuffled) * train_ratio)
    validation_cut = int(len(shuffled) * (train_ratio + validation_ratio))
    split_by_group = {}
    for index, group_id in enumerate(shuffled):
        if index < train_cut:
            split_by_group[group_id] = "train"
        elif index < validation_cut:
            split_by_group[group_id] = "validation"
        else:
            split_by_group[group_id] = "test"
    frame = frame.copy()
    frame["split"] = frame["group_id"].map(split_by_group).fillna("train")
    return frame


def scale_split(split: dict[str, Any]) -> tuple[dict[str, Any], RobustScaler]:
    scaler = RobustScaler.fit(split["x_train"].reshape(-1, split["x_train"].shape[-1]))
    out = dict(split)
    for key in ["x_train", "x_validation", "x_test"]:
        x = split[key]
        out[key] = scaler.transform(x.reshape(-1, x.shape[-1])).reshape(x.shape)
    return out, scaler


def predict_scores(model: nn.Module, x: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    loader = DataLoader(SequenceDataset(x, np.zeros(len(x), dtype=np.float32)), batch_size=batch_size)
    chunks: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for sequences, _ in loader:
            logits = model(sequences.to(device))
            chunks.append(torch.sigmoid(logits).detach().cpu().numpy())
    return np.concatenate(chunks)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def measure_latency(model: nn.Module, x: np.ndarray, batch_size: int, device: torch.device, repeats: int) -> dict[str, float]:
    sample_count = min(len(x), max(batch_size, 1024))
    sample = torch.tensor(x[:sample_count], dtype=torch.float32, device=device)
    single = sample[:1]
    model.eval()
    with torch.no_grad():
        for _ in range(5):
            model(single)
            model(sample[:batch_size])
        synchronize(device)
        started = time.perf_counter()
        for _ in range(repeats):
            model(single)
        synchronize(device)
        single_ms = (time.perf_counter() - started) * 1000.0 / max(1, repeats)
        started = time.perf_counter()
        for _ in range(repeats):
            model(sample)
        synchronize(device)
        batch_ms = (time.perf_counter() - started) * 1000.0 / max(1, repeats)
    return {
        "single_sequence_ms": single_ms,
        "batch_size": float(sample_count),
        "batch_ms": batch_ms,
        "batch_per_sequence_ms": batch_ms / max(1, sample_count),
    }


def train_one(architecture: str, split: dict[str, Any], args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    set_seed(args.seed)
    model = BinarySequenceModel(
        architecture,
        len(FEATURE_COLUMNS),
        args.hidden_size,
        args.num_layers,
        args.dropout,
        args.transformer_heads,
    ).to(device)
    pos = float(split["y_train"].sum())
    neg = float(len(split["y_train"]) - pos)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    loader = DataLoader(SequenceDataset(split["x_train"], split["y_train"]), batch_size=args.batch_size, shuffle=True)
    best_state: dict[str, torch.Tensor] | None = None
    best_f1 = -1.0
    history: list[dict[str, float]] = []
    train_started = time.perf_counter()
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
        validation_scores = predict_scores(model, split["x_validation"], args.batch_size, device)
        _, validation_metrics = best_threshold(split["y_validation"], validation_scores)
        item = {
            "epoch": epoch,
            "loss": total / max(1, count),
            "validation_f1": float(validation_metrics["f1"]),
        }
        history.append(item)
        print(f"preprocessed_imu/{architecture} epoch={epoch:03d} loss={item['loss']:.6f} val_f1={item['validation_f1']:.4f}")
        if validation_metrics["f1"] > best_f1:
            best_f1 = float(validation_metrics["f1"])
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    synchronize(device)
    train_seconds = time.perf_counter() - train_started
    if best_state is not None:
        model.load_state_dict(best_state)
    validation_scores = predict_scores(model, split["x_validation"], args.batch_size, device)
    threshold, validation_metrics = best_threshold(split["y_validation"], validation_scores)
    test_scores = predict_scores(model, split["x_test"], args.batch_size, device)
    test_metrics = metrics(split["y_test"], test_scores, threshold)
    latency = measure_latency(model, split["x_test"], args.batch_size, device, args.latency_repeats)
    return {
        "task": "imu_fall_preprocessed",
        "architecture": architecture,
        "feature_columns": FEATURE_COLUMNS,
        "sequence_length": args.sequence_length,
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "train_seconds": train_seconds,
        "threshold": threshold,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "latency": latency,
        "history": history,
    }


def write_csv(report: dict[str, Any], path: Path) -> None:
    lines = [
        "task,architecture,accuracy,precision,recall,f1,threshold,train_seconds,single_sequence_ms,batch_per_sequence_ms,parameter_count"
    ]
    for result in report["results"]:
        test = result["test_metrics"]
        latency = result["latency"]
        lines.append(
            ",".join(
                [
                    result["task"],
                    result["architecture"],
                    f"{test['accuracy']:.6f}",
                    f"{test['precision']:.6f}",
                    f"{test['recall']:.6f}",
                    f"{test['f1']:.6f}",
                    f"{result['threshold']:.4f}",
                    f"{result['train_seconds']:.3f}",
                    f"{latency['single_sequence_ms']:.6f}",
                    f"{latency['batch_per_sequence_ms']:.6f}",
                    str(result["parameter_count"]),
                ]
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# 전처리 후 IMU 낙상 모델별 성능 비교",
        "",
        "## 실험 조건",
        "",
        f"- Source: `{report['source']}`",
        f"- Device: `{report['device']}`",
        f"- Sequence length: `{report['sequence_length']}`",
        f"- Sequence stride: `{report['sequence_stride']}`",
        f"- Epochs: `{report['epochs']}`",
        f"- Batch size: `{report['batch_size']}`",
        "- Feature: 전처리 완료 12 features, roll/pitch/yaw + accel + gyro + accel_norm + gyro_norm + dt_s",
        "- Split: SisFall group split, ICCAS chronological split",
        "",
        "## 성능 비교",
        "",
        "| Task | Model | Accuracy | Precision | Recall | F1-score | Single inference ms | Batch per sequence ms | Train sec | Params | Threshold |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for result in report["results"]:
        test = result["test_metrics"]
        latency = result["latency"]
        lines.append(
            f"| IMU 낙상 전처리 후 | {result['architecture'].upper()} | "
            f"{test['accuracy']:.4f} | {test['precision']:.4f} | {test['recall']:.4f} | {test['f1']:.4f} | "
            f"{latency['single_sequence_ms']:.3f} | {latency['batch_per_sequence_ms']:.4f} | "
            f"{result['train_seconds']:.1f} | {result['parameter_count']} | {result['threshold']:.2f} |"
        )
    best = max(report["results"], key=lambda item: item["test_metrics"]["f1"])
    lines += [
        "",
        "## 결론",
        "",
        f"- 전처리 후 기준 최고 F1 모델: `{best['architecture'].upper()}`",
        f"- 최고 F1-score: `{best['test_metrics']['f1']:.4f}`",
        f"- 해당 모델 Accuracy: `{best['test_metrics']['accuracy']:.4f}`",
        "- 낙상 감지는 Accuracy 단독보다 Precision, Recall, F1-score를 함께 보는 것이 맞습니다.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def svg_text(x: float, y: float, value: object, size: int = 22, color: str = "#272145", weight: int = 500, anchor: str = "start") -> str:
    return (
        f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" font-weight="{weight}" '
        f'text-anchor="{anchor}" font-family="-apple-system,BlinkMacSystemFont,Segoe UI,Noto Sans KR,sans-serif">{esc(value)}</text>'
    )


def svg_rect(x: float, y: float, w: float, h: float, fill: str, stroke: str = "none", rx: float = 0) -> str:
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}"/>'


def render_svg(report: dict[str, Any]) -> str:
    width = 1600
    height = 900
    metrics_order = ["accuracy", "precision", "recall", "f1"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1-score"]
    colors = {"rnn": "#5b55d9", "gru": "#8177ee", "lstm": "#a496ee", "transformer": "#d9cff8"}
    results = report["results"]
    chart_x = 145
    chart_y = 165
    chart_w = 1310
    chart_h = 505
    base_y = chart_y + chart_h
    group_w = chart_w / len(metrics_order)
    bar_w = 54
    bar_gap = 18
    group_bar_w = len(results) * bar_w + (len(results) - 1) * bar_gap
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        svg_rect(0, 0, width, height, "#fbfaff"),
        svg_text(105, 72, "IMU 낙상 감지 - 전처리 후 모델별 성능 비교", 36, "#272145", 850),
        svg_text(105, 113, "Preprocessed 12-feature input · Accuracy · Precision · Recall · F1-score", 18, "#6f6791", 650),
    ]
    for index, result in enumerate(results):
        x = 930 + index * 145
        arch = result["architecture"]
        parts += [svg_rect(x, 48, 18, 18, colors[arch], "none", 4), svg_text(x + 28, 64, arch.upper(), 18, "#514872", 700)]
    for i in range(6):
        value = i / 5
        y = base_y - chart_h * value
        parts += [
            f'<line x1="{chart_x}" y1="{y}" x2="{chart_x + chart_w}" y2="{y}" stroke="#e7e2f5" stroke-width="1"/>',
            svg_text(chart_x - 28, y + 6, f"{value:.1f}", 16, "#796f9c", 600, "end"),
        ]
    parts += [
        f'<line x1="{chart_x}" y1="{chart_y}" x2="{chart_x}" y2="{base_y}" stroke="#d6cfea" stroke-width="1.4"/>',
        f'<line x1="{chart_x}" y1="{base_y}" x2="{chart_x + chart_w}" y2="{base_y}" stroke="#d6cfea" stroke-width="1.4"/>',
    ]
    for metric_index, metric in enumerate(metrics_order):
        group_center = chart_x + group_w * metric_index + group_w / 2
        start_x = group_center - group_bar_w / 2
        for model_index, result in enumerate(results):
            arch = result["architecture"]
            value = result["test_metrics"][metric]
            x = start_x + model_index * (bar_w + bar_gap)
            bar_h = chart_h * value
            y = base_y - bar_h
            parts += [
                svg_rect(x, y, bar_w, bar_h, colors[arch], "none", 5),
                svg_text(x + bar_w / 2, y - 10, f"{value:.3f}", 15, "#514872", 700, "middle"),
            ]
        parts.append(svg_text(group_center, base_y + 48, metric_labels[metric_index], 19, "#514872", 780, "middle"))
    parts += [
        svg_rect(90, 735, 1420, 98, "#ffffff", "#eee9fb", 10),
        svg_text(125, 780, "단건 추론 (ms)", 19, "#514872", 850),
        svg_text(125, 820, "학습 시간 (s)", 19, "#514872", 850),
    ]
    metric_x = 355
    for index, result in enumerate(results):
        x = metric_x + index * 285
        arch = result["architecture"]
        parts += [
            svg_rect(x, 763, 16, 16, colors[arch], "none", 8),
            svg_text(x + 24, 778, f"{arch.upper()} {result['latency']['single_sequence_ms']:.3f}", 16, "#514872", 720),
            svg_text(x + 24, 819, f"{result['train_seconds']:.1f}", 16, "#514872", 720),
        ]
    best = max(results, key=lambda item: item["test_metrics"]["f1"])
    parts += [
        svg_text(90, 870, f"Source: {report['source']} · seq_len {report['sequence_length']} · stride {report['sequence_stride']} · epoch {report['epochs']}", 15, "#8a82a6", 600),
        svg_text(1510, 870, f"Best F1: {best['architecture'].upper()} {best['test_metrics']['f1']:.4f}", 17, "#5b55d9", 850, "end"),
        "</svg>",
    ]
    return "\n".join(parts)


def write_svg(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_svg(report), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("../data/iccas_sensor_lstm/imu_fall_preprocessed.csv"))
    parser.add_argument("--report", type=Path, default=Path("../data/iccas_sensor_lstm/imu_preprocessed_model_comparison.json"))
    parser.add_argument("--csv", type=Path, default=Path("../data/iccas_sensor_lstm/imu_preprocessed_model_comparison.csv"))
    parser.add_argument("--markdown", type=Path, default=Path("docs/IMU_PREPROCESSED_MODEL_COMPARISON.md"))
    parser.add_argument("--svg-output", type=Path, default=Path("assets/imu_preprocessed_model_performance_comparison.svg"))
    parser.add_argument("--sequence-length", type=int, default=50)
    parser.add_argument("--sequence-stride", type=int, default=4)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--validation-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--architectures", nargs="+", default=["rnn", "gru", "lstm", "transformer"])
    parser.add_argument("--transformer-heads", type=int, default=4)
    parser.add_argument("--latency-repeats", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    frame = load_preprocessed_csv(args.source)
    frame = assign_sisfall_group_splits(frame, args.train_ratio, args.validation_ratio, args.seed)
    raw_split = make_sequences(frame, args)
    split, _ = scale_split(raw_split)
    print(
        f"preprocessed split: train={len(split['y_train'])}, validation={len(split['y_validation'])}, "
        f"test={len(split['y_test'])}, test_positive={int(split['y_test'].sum())}, device={device}"
    )
    results = [train_one(architecture, split, args, device) for architecture in args.architectures]
    report = {
        "source": str(args.source),
        "device": str(device),
        "feature_columns": FEATURE_COLUMNS,
        "sequence_length": args.sequence_length,
        "sequence_stride": args.sequence_stride,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "hidden_size": args.hidden_size,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "split_sizes": {
            "train": int(len(split["y_train"])),
            "validation": int(len(split["y_validation"])),
            "test": int(len(split["y_test"])),
        },
        "label_counts": {
            "train_positive": int(split["y_train"].sum()),
            "train_negative": int(len(split["y_train"]) - split["y_train"].sum()),
            "validation_positive": int(split["y_validation"].sum()),
            "validation_negative": int(len(split["y_validation"]) - split["y_validation"].sum()),
            "test_positive": int(split["y_test"].sum()),
            "test_negative": int(len(split["y_test"]) - split["y_test"].sum()),
        },
        "results": results,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(report, args.csv)
    write_markdown(report, args.markdown)
    write_svg(report, args.svg_output)
    print(json.dumps({"report": str(args.report), "csv": str(args.csv), "markdown": str(args.markdown), "svg": str(args.svg_output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
