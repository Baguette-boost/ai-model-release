#!/usr/bin/env python3
"""Compare IMU filter engineering variants across fall detection models."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from compare_preprocessed_imu_models import (
    BinarySequenceModel,
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
from train_imu_1d_cnn_experiment import IMUOneDCNN
from train_imu_cnn_lstm_experiment import IMUCNNLSTM


FILTER_COLUMNS = ["ax", "ay", "az", "wx", "wy", "wz"]


def recompute_physical_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["accel_norm"] = np.sqrt(out["ax"] ** 2 + out["ay"] ** 2 + out["az"] ** 2)
    out["gyro_norm"] = np.sqrt(out["wx"] ** 2 + out["wy"] ** 2 + out["wz"] ** 2)
    return out


def apply_median_filter(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    out = frame.copy()
    for column in FILTER_COLUMNS:
        out[column] = (
            out.groupby("group_id", sort=False)[column]
            .transform(lambda series: series.rolling(window=window, center=True, min_periods=1).median())
            .astype(np.float32)
        )
    return recompute_physical_features(out)


def apply_ema_filter(frame: pd.DataFrame, alpha: float) -> pd.DataFrame:
    out = frame.copy()
    for column in FILTER_COLUMNS:
        out[column] = (
            out.groupby("group_id", sort=False)[column]
            .transform(lambda series: series.ewm(alpha=alpha, adjust=False).mean())
            .astype(np.float32)
        )
    return recompute_physical_features(out)


def apply_filter_variant(frame: pd.DataFrame, variant: str, median_window: int, ema_alpha: float) -> pd.DataFrame:
    if variant == "baseline":
        return frame.copy()
    if variant == "median3":
        return apply_median_filter(frame, median_window)
    if variant == "ema030":
        return apply_ema_filter(frame, ema_alpha)
    if variant == "median3_ema030":
        return apply_ema_filter(apply_median_filter(frame, median_window), ema_alpha)
    raise ValueError(f"Unsupported filter variant: {variant}")


def build_model(model_name: str, args: argparse.Namespace) -> nn.Module:
    input_size = len(FEATURE_COLUMNS)
    if model_name == "lstm":
        return BinarySequenceModel(
            "lstm",
            input_size,
            args.hidden_size,
            args.num_layers,
            args.dropout,
            args.transformer_heads,
        )
    if model_name == "cnn1d":
        return IMUOneDCNN(input_size, args.cnn_channels, args.dropout)
    if model_name == "cnn_lstm":
        return IMUCNNLSTM(input_size, args.cnn_lstm_channels, args.hidden_size, args.dropout)
    raise ValueError(f"Unsupported model: {model_name}")


def train_model(
    model_name: str,
    split: dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
    filter_variant: str,
) -> dict[str, Any]:
    set_seed(args.seed)
    model = build_model(model_name, args).to(device)
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
        print(
            f"filter={filter_variant} model={model_name} epoch={epoch:03d} "
            f"loss={item['loss']:.6f} val_f1={item['validation_f1']:.4f}",
            flush=True,
        )
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
    return {
        "filter_variant": filter_variant,
        "model": model_name,
        "epochs": args.epochs,
        "threshold": threshold,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "latency": latency,
        "train_seconds": train_seconds,
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "history": history,
    }


def row_audit(
    variant: str,
    base_rows: int,
    filtered_frame: pd.DataFrame,
    raw_split: dict[str, Any],
    baseline_sequence_total: int | None,
) -> dict[str, Any]:
    rows_after = int(len(filtered_frame))
    sequence_counts = {
        "train": int(len(raw_split["y_train"])),
        "validation": int(len(raw_split["y_validation"])),
        "test": int(len(raw_split["y_test"])),
    }
    sequence_total = sum(sequence_counts.values())
    return {
        "filter_variant": variant,
        "sensor_rows_before": int(base_rows),
        "sensor_rows_after_filter": rows_after,
        "sensor_rows_delta": rows_after - int(base_rows),
        "sensor_rows_delta_rate": (rows_after - int(base_rows)) / max(1, int(base_rows)),
        "sequence_length": int(raw_split["x_train"].shape[1]),
        "sequence_stride": 4,
        "sequence_train": sequence_counts["train"],
        "sequence_validation": sequence_counts["validation"],
        "sequence_test": sequence_counts["test"],
        "sequence_total": sequence_total,
        "sequence_total_delta_vs_baseline": None if baseline_sequence_total is None else sequence_total - baseline_sequence_total,
    }


def write_csvs(report: dict[str, Any], out_dir: Path) -> None:
    metric_rows = []
    for item in report["results"]:
        test = item["test_metrics"]
        latency = item["latency"]
        metric_rows.append(
            {
                "filter_variant": item["filter_variant"],
                "model": item["model"],
                "accuracy": test["accuracy"],
                "precision": test["precision"],
                "recall": test["recall"],
                "f1": test["f1"],
                "threshold": item["threshold"],
                "tp": test["tp"],
                "fp": test["fp"],
                "tn": test["tn"],
                "fn": test["fn"],
                "single_sequence_ms": latency["single_sequence_ms"],
                "batch_per_sequence_ms": latency["batch_per_sequence_ms"],
                "train_seconds": item["train_seconds"],
                "parameter_count": item["parameter_count"],
            }
        )
    write_dict_csv(out_dir / "metrics.csv", metric_rows)
    write_dict_csv(out_dir / "row_changes.csv", report["row_changes"])


def write_dict_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(report: dict[str, Any], out_dir: Path) -> None:
    results = report["results"]
    row_changes = report["row_changes"]
    best = max(results, key=lambda item: item["test_metrics"]["f1"])
    lines = [
        "# IMU Filter Engineering Experiment",
        "",
        "Median filter, EMA low-pass filter, and their combination were compared across IMU fall detection models.",
        "",
        "## Setup",
        "",
        f"- Source: `{report['source']}`",
        f"- Device: `{report['device']}`",
        f"- Epochs: `{report['epochs']}`",
        f"- Sequence length: `{report['sequence_length']}`",
        f"- Sequence stride: `{report['sequence_stride']}`",
        f"- Median window: `{report['median_window']}`",
        f"- EMA alpha: `{report['ema_alpha']}`",
        "",
        "## Row Change Analysis",
        "",
        "| Filter | Sensor rows before | Sensor rows after | Row delta | Sequence total | Sequence delta |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in row_changes:
        lines.append(
            f"| {row['filter_variant']} | {row['sensor_rows_before']} | {row['sensor_rows_after_filter']} | "
            f"{row['sensor_rows_delta']} | {row['sequence_total']} | {row['sequence_total_delta_vs_baseline']} |"
        )
    lines.extend(
        [
            "",
            "해석:",
            "",
            "- Median/EMA filtering은 smoothing 방식이므로 sensor row를 삭제하지 않는다.",
            "- 따라서 `sensor_rows_delta`는 0이다.",
            "- LSTM sequence 수 역시 같은 sequence length/stride를 사용하므로 variant 간 동일하다.",
            "- 즉, 이번 비교의 성능 차이는 row 수 변화가 아니라 신호값 변화에서 발생한다.",
            "",
            "## Performance Metrics",
            "",
            "| Filter | Model | Accuracy | Precision | Recall | F1-score | Threshold | Single ms | Train sec |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for item in results:
        test = item["test_metrics"]
        lines.append(
            f"| {item['filter_variant']} | {item['model']} | {test['accuracy']:.4f} | "
            f"{test['precision']:.4f} | {test['recall']:.4f} | {test['f1']:.4f} | "
            f"{item['threshold']:.2f} | {item['latency']['single_sequence_ms']:.4f} | {item['train_seconds']:.1f} |"
        )
    lines.extend(
        [
            "",
            "## Best Result",
            "",
            f"- Best filter: `{best['filter_variant']}`",
            f"- Best model: `{best['model']}`",
            f"- Accuracy: `{best['test_metrics']['accuracy']:.4f}`",
            f"- Precision: `{best['test_metrics']['precision']:.4f}`",
            f"- Recall: `{best['test_metrics']['recall']:.4f}`",
            f"- F1-score: `{best['test_metrics']['f1']:.4f}`",
            "",
            "## Important Interpretation",
            "",
            "Filtering did not change the number of sensor rows. It only changed the signal values. "
            "This is important because the experiment isolates the effect of signal filtering from the effect of data size.",
        ]
    )
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    base_frame = load_preprocessed_csv(args.source)
    base_frame = assign_sisfall_group_splits(base_frame, args.train_ratio, args.validation_ratio, args.seed)
    base_rows = len(base_frame)
    print(f"device={device}", flush=True)
    print(f"source={args.source} rows={base_rows}", flush=True)

    results: list[dict[str, Any]] = []
    row_changes: list[dict[str, Any]] = []
    baseline_sequence_total: int | None = None
    for variant in args.filters:
        started = time.perf_counter()
        filtered_frame = apply_filter_variant(base_frame, variant, args.median_window, args.ema_alpha)
        raw_split = make_sequences(filtered_frame, args)
        if baseline_sequence_total is None:
            baseline_sequence_total = int(len(raw_split["y_train"]) + len(raw_split["y_validation"]) + len(raw_split["y_test"]))
        row_item = row_audit(variant, base_rows, filtered_frame, raw_split, baseline_sequence_total)
        row_changes.append(row_item)
        print(
            f"filter={variant} rows_delta={row_item['sensor_rows_delta']} "
            f"sequences={row_item['sequence_total']} prep_sec={time.perf_counter() - started:.1f}",
            flush=True,
        )
        split, _ = scale_split(raw_split)
        for model_name in args.models:
            results.append(train_model(model_name, split, args, device, variant))

    report = {
        "source": str(args.source),
        "out_dir": str(args.out_dir),
        "device": str(device),
        "epochs": args.epochs,
        "sequence_length": args.sequence_length,
        "sequence_stride": args.sequence_stride,
        "median_window": args.median_window,
        "ema_alpha": args.ema_alpha,
        "filters": args.filters,
        "models": args.models,
        "row_changes": row_changes,
        "results": results,
    }
    (args.out_dir / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csvs(report, args.out_dir)
    write_markdown(report, args.out_dir)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("../data/iccas_sensor_lstm/imu_fall_preprocessed.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("experiments/imu_filter_engineering"))
    parser.add_argument("--filters", nargs="+", default=["baseline", "median3", "ema030", "median3_ema030"])
    parser.add_argument("--models", nargs="+", default=["lstm", "cnn1d", "cnn_lstm"])
    parser.add_argument("--sequence-length", type=int, default=50)
    parser.add_argument("--sequence-stride", type=int, default=4)
    parser.add_argument("--median-window", type=int, default=3)
    parser.add_argument("--ema-alpha", type=float, default=0.30)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--transformer-heads", type=int, default=4)
    parser.add_argument("--cnn-channels", type=int, default=64)
    parser.add_argument("--cnn-lstm-channels", type=int, default=48)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--validation-ratio", type=float, default=0.15)
    parser.add_argument("--latency-repeats", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    return parser.parse_args()


def main() -> None:
    report = run(parse_args())
    best = max(report["results"], key=lambda item: item["test_metrics"]["f1"])
    print(
        "best="
        f"{best['filter_variant']}/{best['model']} "
        f"accuracy={best['test_metrics']['accuracy']:.4f} "
        f"precision={best['test_metrics']['precision']:.4f} "
        f"recall={best['test_metrics']['recall']:.4f} "
        f"f1={best['test_metrics']['f1']:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
