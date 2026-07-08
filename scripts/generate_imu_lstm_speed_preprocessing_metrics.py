"""Measure IMU Fall LSTM inference speed and render preprocessing comparison SVG."""

from __future__ import annotations

import argparse
import html
import json
import time
from pathlib import Path
from statistics import mean, median
from typing import Any

import numpy as np
import pandas as pd
import torch

from generate_imu_preprocessing_metrics_svg import derive_metrics, load_frame
from train_sisfall_merged_imu_lstm import BinaryLSTM, FEATURE_COLUMNS, load_merged_csv, make_sequences


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def text(x: float, y: float, value: object, size: int = 22, color: str = "#172033", weight: int = 500, anchor: str = "start") -> str:
    return (
        f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" '
        f'font-weight="{weight}" text-anchor="{anchor}" '
        f'font-family="-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif">{esc(value)}</text>'
    )


def rect(x: float, y: float, w: float, h: float, fill: str, stroke: str = "#d9dee8", rx: float = 8) -> str:
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}"/>'


def pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def ms(value: float) -> str:
    return f"{value:.4f} ms"


def num(value: float | int) -> str:
    if isinstance(value, float):
        return f"{value:,.3f}"
    return f"{value:,}"


def transform_with_checkpoint_scaler(x: np.ndarray, checkpoint: dict[str, Any]) -> np.ndarray:
    center = np.array(checkpoint["scaler_center"], dtype=np.float32)
    scale = np.array(checkpoint["scaler_scale"], dtype=np.float32)
    scaled = (x.astype(np.float32) - center) / scale
    return np.clip(scaled, -12.0, 12.0).astype(np.float32)


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[BinaryLSTM, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = BinaryLSTM(
        len(FEATURE_COLUMNS),
        int(checkpoint["hidden_size"]),
        int(checkpoint["num_layers"]),
        float(checkpoint["dropout"]),
        bool(checkpoint["bidirectional"]),
        str(checkpoint["pooling"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint


def predict_batch(model: BinaryLSTM, x: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            batch = torch.from_numpy(x[start : start + batch_size]).to(device)
            chunks.append(torch.sigmoid(model(batch)).detach().cpu().numpy())
    return np.concatenate(chunks)


def measure_speed(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device("cpu")
    model, checkpoint = load_model(args.model, device)

    t0 = time.perf_counter()
    frame = load_merged_csv(args.source)
    t1 = time.perf_counter()
    split = make_sequences(
        frame,
        int(checkpoint["sequence_length"]),
        int(checkpoint["sequence_stride"]),
        args.train_ratio,
        args.validation_ratio,
        args.seed,
    )
    t2 = time.perf_counter()
    x_test = transform_with_checkpoint_scaler(split["x_test"], checkpoint)
    t3 = time.perf_counter()

    with torch.no_grad():
        for _ in range(args.warmup):
            _ = model(torch.from_numpy(x_test[:1]).to(device))
        _ = predict_batch(model, x_test[: min(args.batch_size, len(x_test))], args.batch_size, device)

    batch_start = time.perf_counter()
    _ = predict_batch(model, x_test, args.batch_size, device)
    batch_end = time.perf_counter()
    batch_total_ms = (batch_end - batch_start) * 1000.0
    batch_ms_per_sequence = batch_total_ms / len(x_test)

    sample_count = min(args.realtime_samples, len(x_test))
    per_tensor_forward: list[float] = []
    with torch.no_grad():
        for sequence in x_test[:sample_count]:
            start = time.perf_counter()
            tensor = torch.from_numpy(sequence[None, :, :]).to(device)
            _ = torch.sigmoid(model(tensor)).item()
            end = time.perf_counter()
            per_tensor_forward.append((end - start) * 1000.0)

    prebuilt = [torch.from_numpy(sequence[None, :, :]).to(device) for sequence in x_test[:sample_count]]
    per_forward_only: list[float] = []
    with torch.no_grad():
        for tensor in prebuilt:
            start = time.perf_counter()
            _ = torch.sigmoid(model(tensor)).item()
            end = time.perf_counter()
            per_forward_only.append((end - start) * 1000.0)

    def p95(values: list[float]) -> float:
        return sorted(values)[max(0, int(len(values) * 0.95) - 1)]

    return {
        "device": str(device),
        "model": str(args.model),
        "source": str(args.source),
        "test_sequences": int(len(x_test)),
        "sequence_length": int(x_test.shape[1]),
        "feature_count": int(x_test.shape[2]),
        "batch_size": int(args.batch_size),
        "data_load_ms": (t1 - t0) * 1000.0,
        "sequence_build_ms": (t2 - t1) * 1000.0,
        "scale_test_ms": (t3 - t2) * 1000.0,
        "batch_total_ms": batch_total_ms,
        "batch_ms_per_sequence": batch_ms_per_sequence,
        "batch_sequences_per_sec": 1000.0 / batch_ms_per_sequence,
        "realtime_samples": sample_count,
        "realtime_tensor_forward_avg_ms": mean(per_tensor_forward),
        "realtime_tensor_forward_median_ms": median(per_tensor_forward),
        "realtime_tensor_forward_p95_ms": p95(per_tensor_forward),
        "realtime_forward_only_avg_ms": mean(per_forward_only),
        "realtime_forward_only_median_ms": median(per_forward_only),
        "realtime_forward_only_p95_ms": p95(per_forward_only),
        "threshold": float(checkpoint["threshold"]),
        "sample_ms": int(checkpoint["sample_ms"]),
        "window_seconds": int(checkpoint["sequence_length"]) * int(checkpoint["sample_ms"]) / 1000.0,
    }


def render_svg(before: dict[str, Any], after: dict[str, Any], speed: dict[str, Any]) -> str:
    width = 1700
    height = 1250
    before_ready = before["model_features_ready"] / before["model_features_total"]
    after_ready = after["model_features_ready"] / after["model_features_total"]
    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        rect(0, 0, width, height, "#f7f8fb", "none", 0),
        text(70, 82, "IMU Fall LSTM: Inference Speed & Preprocessing Metrics", 42, "#111827", 850),
        text(70, 122, "Measured CPU inference speed with before/after preprocessing readiness", 22, "#667085", 500),
        text(1630, 82, f"Window: {speed['sequence_length']} steps / {speed['window_seconds']:.1f}s", 18, "#667085", 700, "end"),
        text(1630, 112, f"Features: {speed['feature_count']} / Threshold: {speed['threshold']:.2f}", 18, "#667085", 700, "end"),
    ]

    cards = [
        ("Realtime LSTM", ms(speed["realtime_tensor_forward_avg_ms"]), "avg, tensor + forward", "#eff6ff"),
        ("Realtime p95", ms(speed["realtime_tensor_forward_p95_ms"]), "window 1-by-1", "#fef3c7"),
        ("Batch LSTM", ms(speed["batch_ms_per_sequence"]), f"batch size {speed['batch_size']}", "#ecfdf3"),
        ("Throughput", f"{speed['batch_sequences_per_sec']:,.0f}/s", "batch sequences/sec", "#f5f3ff"),
    ]
    for i, (title, value, note, fill) in enumerate(cards):
        x = 70 + i * 400
        parts += [
            rect(x, 175, 360, 165, fill),
            text(x + 24, 220, title, 22, "#667085", 750),
            text(x + 24, 278, value, 38, "#172033", 850),
            text(x + 24, 318, note, 17, "#667085", 550),
        ]

    parts += [
        rect(70, 390, 760, 330, "#ffffff"),
        text(100, 442, "Actual LSTM Speed Measurement", 29, "#172033", 850),
        text(100, 498, f"Device: {speed['device'].upper()}", 22, "#344054", 700),
        text(100, 542, f"Test sequences: {speed['test_sequences']:,}", 22, "#344054", 700),
        text(100, 586, f"Batch total: {speed['batch_total_ms']:.3f} ms", 22, "#344054", 700),
        text(100, 630, f"Realtime median: {speed['realtime_tensor_forward_median_ms']:.4f} ms", 22, "#344054", 700),
        text(100, 674, "Speed excludes server POST and map rendering.", 18, "#667085", 550),
        rect(870, 390, 760, 330, "#ffffff"),
        text(900, 442, "Before vs After Preprocessing", 29, "#172033", 850),
        text(900, 498, f"Rows: {before['rows']:,} -> {after['rows']:,}", 22, "#344054", 700),
        text(900, 542, f"Columns: {before['columns']} -> {after['columns']}", 22, "#344054", 700),
        text(900, 586, f"Model-ready features: {before['model_features_ready']}/12 -> {after['model_features_ready']}/12", 22, "#344054", 700),
        text(900, 630, f"Missing numeric rate: {pct(before['missing_numeric_rate'])} -> {pct(after['missing_numeric_rate'])}", 22, "#344054", 700),
        text(900, 674, "After adds accel_norm, gyro_norm, dt_s, and fall_target.", 18, "#667085", 550),
    ]

    parts += [
        rect(70, 770, 760, 285, "#ffffff"),
        text(100, 822, "Training Readiness Score", 29, "#172033", 850),
        text(120, 880, f"Before: {before['model_features_ready']}/{before['model_features_total']} features", 19, "#344054", 700),
        rect(120, 900, 560, 30, "#eef2f7", "none", 10),
        rect(120, 900, 560 * before_ready, 30, "#d97706", "none", 10),
        text(700, 925, pct(before_ready), 24, "#9a3412", 850),
        text(120, 980, f"After: {after['model_features_ready']}/{after['model_features_total']} features", 19, "#344054", 700),
        rect(120, 1000, 560, 30, "#eef2f7", "none", 10),
        rect(120, 1000, 560 * after_ready, 30, "#2563eb", "none", 10),
        text(700, 1025, pct(after_ready), 24, "#1d4ed8", 850),
        rect(870, 770, 760, 285, "#ffffff"),
        text(900, 822, "Signal Preservation", 29, "#172033", 850),
        text(900, 880, f"SVM mean: {before['svm_mean']:.3f}g -> {after['svm_mean']:.3f}g", 22, "#344054", 700),
        text(900, 930, f"SVM p99: {before['svm_p99']:.3f}g -> {after['svm_p99']:.3f}g", 22, "#344054", 700),
        text(900, 980, f"Gyro p95: {before['gyro_p95']:.1f} -> {after['gyro_p95']:.1f} dps", 22, "#344054", 700),
        text(900, 1025, "Preprocessing preserves raw signal magnitude while adding model-ready fields.", 17, "#667085", 550),
    ]

    parts += [
        text(70, 1145, f"Data preparation timing: load {speed['data_load_ms']:.1f} ms, build sequences {speed['sequence_build_ms']:.1f} ms, scale test {speed['scale_test_ms']:.1f} ms", 17, "#667085", 550),
        text(70, 1185, "Inference speed is measured after the 50-step ready window has already been prepared.", 17, "#667085", 550),
        text(1630, 1185, "Model: iccas_final_hybrid_lstm_imu_fall.pt", 17, "#667085", 550, "end"),
        "</svg>",
    ]
    return "\n".join(parts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("../data/iccas_sensor_lstm/final_iccas_sisfall_imu_merged.csv"))
    parser.add_argument("--before", type=Path, default=Path("../data/iccas_sensor_lstm/final_iccas_sisfall_imu_merged.csv"))
    parser.add_argument("--after", type=Path, default=Path("../data/iccas_sensor_lstm/imu_fall_preprocessed.csv"))
    parser.add_argument("--model", type=Path, default=Path("models/iccas_final_hybrid_lstm_imu_fall.pt"))
    parser.add_argument("--svg-output", type=Path, default=Path("assets/imu_lstm_speed_preprocessing_metrics.svg"))
    parser.add_argument("--summary", type=Path, default=Path("../data/iccas_sensor_lstm/imu_lstm_speed_preprocessing_metrics.json"))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--realtime-samples", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--validation-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    before = derive_metrics(load_frame(args.before), "before")
    after = derive_metrics(load_frame(args.after), "after")
    speed = measure_speed(args)
    args.svg_output.parent.mkdir(parents=True, exist_ok=True)
    args.svg_output.write_text(render_svg(before, after, speed), encoding="utf-8")
    summary = {"before": before, "after": after, "inference_speed": speed}
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"svg_output": str(args.svg_output), "summary": str(args.summary), **summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
