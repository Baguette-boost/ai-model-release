#!/usr/bin/env python3
"""Train an isolated CNN-LSTM experiment for IMU fall detection."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from compare_preprocessed_imu_models import (
    FEATURE_COLUMNS,
    SequenceDataset,
    assign_sisfall_group_splits,
    best_threshold,
    load_preprocessed_csv,
    make_sequences,
    measure_latency,
    metrics,
    predict_scores,
    resolve_device,
    scale_split,
    set_seed,
    synchronize,
)


class IMUCNNLSTM(nn.Module):
    """1D-CNN local feature extractor followed by LSTM temporal modeling."""

    def __init__(self, input_size: int, channels: int, hidden_size: int, dropout: float) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(input_size, channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(channels),
            nn.GELU(),
        )
        self.lstm = nn.LSTM(
            input_size=channels,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size * 2),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [batch, time, feature] -> [batch, feature, time] for Conv1d.
        features = self.cnn(x.transpose(1, 2)).transpose(1, 2)
        output, _ = self.lstm(features)
        pooled = output[:, -1, :]
        return self.head(pooled).squeeze(-1)


def train(args: argparse.Namespace) -> dict[str, Any]:
    set_seed(args.seed)
    args.experiment_dir.mkdir(parents=True, exist_ok=True)
    frame = load_preprocessed_csv(args.source)
    frame = assign_sisfall_group_splits(frame, args.train_ratio, args.validation_ratio, args.seed)
    raw_split = make_sequences(frame, args)
    split, scaler = scale_split(raw_split)
    device = resolve_device(args.device)
    print(f"experiment_dir={args.experiment_dir}", flush=True)
    print(f"training_device={device}", flush=True)
    print(
        "dataset="
        f"{args.source}, train={len(split['y_train'])}, validation={len(split['y_validation'])}, "
        f"test={len(split['y_test'])}",
        flush=True,
    )

    model = IMUCNNLSTM(len(FEATURE_COLUMNS), args.channels, args.hidden_size, args.dropout).to(device)
    pos = float(split["y_train"].sum())
    neg = float(len(split["y_train"]) - pos)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    loader = DataLoader(SequenceDataset(split["x_train"], split["y_train"]), batch_size=args.batch_size, shuffle=True)

    best_state: dict[str, torch.Tensor] | None = None
    best_f1 = -1.0
    history: list[dict[str, float]] = []
    started = time.perf_counter()
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
        print(f"imu_cnn_lstm epoch={epoch:03d} loss={item['loss']:.6f} val_f1={item['validation_f1']:.4f}", flush=True)
        if validation_metrics["f1"] > best_f1:
            best_f1 = float(validation_metrics["f1"])
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    synchronize(device)
    train_seconds = time.perf_counter() - started
    if best_state is not None:
        model.load_state_dict(best_state)

    validation_scores = predict_scores(model, split["x_validation"], args.batch_size, device)
    threshold, validation_metrics = best_threshold(split["y_validation"], validation_scores)
    test_scores = predict_scores(model, split["x_test"], args.batch_size, device)
    test_metrics = metrics(split["y_test"], test_scores, threshold)
    latency = measure_latency(model, split["x_test"], args.batch_size, device, args.latency_repeats)

    model_path = args.experiment_dir / "imu_fall_cnn_lstm.pt"
    metadata_path = args.experiment_dir / "imu_fall_cnn_lstm.json"
    metrics_path = args.experiment_dir / "metrics.json"
    csv_path = args.experiment_dir / "metrics.csv"
    markdown_path = args.experiment_dir / "README.md"

    checkpoint = {
        "model_type": "imu_fall_cnn_lstm",
        "task": "imu_fall",
        "feature_columns": FEATURE_COLUMNS,
        "sequence_length": args.sequence_length,
        "sequence_stride": args.sequence_stride,
        "channels": args.channels,
        "hidden_size": args.hidden_size,
        "dropout": args.dropout,
        "threshold": threshold,
        "scaler_center": scaler.center,
        "scaler_scale": scaler.scale,
        "model_state": model.state_dict(),
    }
    torch.save(checkpoint, model_path)

    metadata = {
        "model_type": "imu_fall_cnn_lstm",
        "task": "imu_fall",
        "source": str(args.source),
        "experiment_dir": str(args.experiment_dir),
        "device": str(device),
        "feature_columns": FEATURE_COLUMNS,
        "sequence_length": args.sequence_length,
        "sequence_stride": args.sequence_stride,
        "channels": args.channels,
        "hidden_size": args.hidden_size,
        "dropout": args.dropout,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "split_method": "SisFall group split, ICCAS chronological split",
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
        "threshold": threshold,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "latency": latency,
        "train_seconds": train_seconds,
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "history": history,
        "model_path": str(model_path),
        "metadata_path": str(metadata_path),
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    metrics_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    write_metrics_csv(metadata, csv_path)
    write_markdown(metadata, markdown_path)
    return metadata


def write_metrics_csv(report: dict[str, Any], path: Path) -> None:
    metric = report["test_metrics"]
    row = {
        "model": "CNN-LSTM",
        "accuracy": metric["accuracy"],
        "precision": metric["precision"],
        "recall": metric["recall"],
        "f1": metric["f1"],
        "threshold": report["threshold"],
        "tp": metric["tp"],
        "fp": metric["fp"],
        "tn": metric["tn"],
        "fn": metric["fn"],
        "single_sequence_ms": report["latency"]["single_sequence_ms"],
        "batch_per_sequence_ms": report["latency"]["batch_per_sequence_ms"],
        "train_seconds": report["train_seconds"],
        "parameter_count": report["parameter_count"],
    }
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


def write_markdown(report: dict[str, Any], path: Path) -> None:
    test = report["test_metrics"]
    latency = report["latency"]
    lines = [
        "# IMU Fall CNN-LSTM Experiment",
        "",
        "이 폴더는 기존 LSTM 최종 모델과 분리한 CNN-LSTM 단독 실험 결과입니다.",
        "",
        "## Experiment Setup",
        "",
        f"- Source: `{report['source']}`",
        f"- Device: `{report['device']}`",
        f"- Sequence length: `{report['sequence_length']}`",
        f"- Sequence stride: `{report['sequence_stride']}`",
        f"- Feature count: `{len(report['feature_columns'])}`",
        f"- Features: `{', '.join(report['feature_columns'])}`",
        f"- CNN channels: `{report['channels']}`",
        f"- LSTM hidden size: `{report['hidden_size']}`",
        f"- Epochs: `{report['epochs']}`",
        f"- Batch size: `{report['batch_size']}`",
        f"- Split: `{report['split_method']}`",
        "",
        "## Test Metrics",
        "",
        "| Model | Accuracy | Precision | Recall | F1-score | Threshold | TP | FP | TN | FN |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        f"| CNN-LSTM | {test['accuracy']:.4f} | {test['precision']:.4f} | {test['recall']:.4f} | "
        f"{test['f1']:.4f} | {report['threshold']:.2f} | {test['tp']} | {test['fp']} | {test['tn']} | {test['fn']} |",
        "",
        "## Speed",
        "",
        f"- Single sequence inference: `{latency['single_sequence_ms']:.4f} ms`",
        f"- Batch per sequence inference: `{latency['batch_per_sequence_ms']:.6f} ms`",
        f"- Training time: `{report['train_seconds']:.1f} s`",
        f"- Parameters: `{report['parameter_count']:,}`",
        "",
        "## Interpretation",
        "",
        "- CNN-LSTM은 CNN이 낙상 순간의 local impact/rotation pattern을 먼저 추출하고, LSTM이 충격 전후의 시간 흐름을 학습하는 구조입니다.",
        "- 기존 최종 LSTM(`iccas_final_hybrid_lstm_imu_fall`)은 별도 split/학습 파이프라인에서 F1-score 0.8677을 기록했습니다. 따라서 이 실험은 최종 모델 교체 여부를 판단하기 위한 독립 비교 자료로 사용합니다.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("../data/iccas_sensor_lstm/imu_fall_preprocessed.csv"))
    parser.add_argument("--experiment-dir", type=Path, default=Path("experiments/imu_fall_cnn_lstm"))
    parser.add_argument("--sequence-length", type=int, default=50)
    parser.add_argument("--sequence-stride", type=int, default=4)
    parser.add_argument("--channels", type=int, default=48)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--validation-ratio", type=float, default=0.15)
    parser.add_argument("--latency-repeats", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    return parser.parse_args()


def main() -> None:
    report = train(parse_args())
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
