"""Generate an SVG comparing IMU data metrics before and after preprocessing."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


NUMERIC_COLUMNS = ["roll", "pitch", "yaw", "ax", "ay", "az", "wx", "wy", "wz", "t_ms"]
MODEL_FEATURES = ["roll", "pitch", "yaw", "ax", "ay", "az", "wx", "wy", "wz", "accel_norm", "gyro_norm", "dt_s"]


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


def num(value: float | int) -> str:
    if isinstance(value, float):
        return f"{value:,.3f}"
    return f"{value:,}"


def load_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    for column in NUMERIC_COLUMNS:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def derive_metrics(frame: pd.DataFrame, stage: str) -> dict[str, Any]:
    numeric_present = [column for column in NUMERIC_COLUMNS if column in frame.columns]
    model_present = [column for column in MODEL_FEATURES if column in frame.columns]
    missing_numeric = int(frame[numeric_present].isna().sum().sum()) if numeric_present else 0
    total_numeric = int(len(frame) * len(numeric_present)) if numeric_present else 0
    if "accel_norm" in frame.columns:
        svm = pd.to_numeric(frame["accel_norm"], errors="coerce")
    else:
        svm = np.sqrt(frame["ax"] ** 2 + frame["ay"] ** 2 + frame["az"] ** 2)
    if "gyro_norm" in frame.columns:
        gyro = pd.to_numeric(frame["gyro_norm"], errors="coerce")
    else:
        gyro = np.sqrt(frame["wx"] ** 2 + frame["wy"] ** 2 + frame["wz"] ** 2)
    return {
        "stage": stage,
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "model_features_ready": len(model_present),
        "model_features_total": len(MODEL_FEATURES),
        "missing_numeric": missing_numeric,
        "missing_numeric_rate": missing_numeric / max(1, total_numeric),
        "has_accel_norm": "accel_norm" in frame.columns,
        "has_gyro_norm": "gyro_norm" in frame.columns,
        "has_dt_s": "dt_s" in frame.columns,
        "has_fall_target": "fall_target" in frame.columns,
        "svm_mean": float(svm.mean()),
        "svm_p95": float(svm.quantile(0.95)),
        "svm_p99": float(svm.quantile(0.99)),
        "gyro_mean": float(gyro.mean()),
        "gyro_p95": float(gyro.quantile(0.95)),
        "label_counts": frame["label"].value_counts(dropna=False).to_dict() if "label" in frame.columns else {},
    }


def bar(x: float, y: float, w: float, h: float, value: float, color: str, label: str, max_value: float = 1.0) -> list[str]:
    fill_w = w * min(max(value / max_value, 0.0), 1.0)
    return [
        text(x, y - 10, label, 17, "#344054", 650),
        rect(x, y, w, h, "#eef2f7", "none", 10),
        rect(x, y, fill_w, h, color, "none", 10),
    ]


def render_svg(before: dict[str, Any], after: dict[str, Any], source_before: Path, source_after: Path) -> str:
    width = 1600
    height = 1150
    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        rect(0, 0, width, height, "#f7f8fb", "none", 0),
        text(70, 82, "IMU Preprocessing Metrics", 44, "#111827", 850),
        text(70, 122, "Before preprocessing vs After preprocessing for fall detection training data", 22, "#667085", 500),
        text(1530, 82, "SVM = sqrt(ax² + ay² + az²)", 18, "#667085", 700, "end"),
    ]

    cards = [
        ("Rows", before["rows"], after["rows"], "same sample count"),
        ("Columns", before["columns"], after["columns"], "derived features added"),
        ("Model-ready features", before["model_features_ready"], after["model_features_ready"], "12 features required"),
        ("Numeric missing rate", before["missing_numeric_rate"], after["missing_numeric_rate"], "lower is better"),
    ]
    for i, (title, b_val, a_val, note) in enumerate(cards):
        x = 70 + i * 370
        parts += [
            rect(x, 175, 340, 165, "#ffffff"),
            text(x + 24, 220, title, 22, "#667085", 700),
            text(x + 24, 270, pct(b_val) if "rate" in title.lower() else num(b_val), 34, "#d97706", 850),
            text(x + 180, 270, "before", 16, "#9a3412", 700),
            text(x + 24, 315, pct(a_val) if "rate" in title.lower() else num(a_val), 34, "#2563eb", 850),
            text(x + 180, 315, "after", 16, "#1d4ed8", 700),
            text(x + 24, 350, note, 15, "#667085", 500),
        ]

    parts += [
        rect(70, 395, 700, 315, "#ffffff"),
        text(100, 445, "Feature Readiness", 28, "#172033", 850),
    ]
    checks = [
        ("accel_norm / SVM", before["has_accel_norm"], after["has_accel_norm"]),
        ("gyro_norm", before["has_gyro_norm"], after["has_gyro_norm"]),
        ("dt_s", before["has_dt_s"], after["has_dt_s"]),
        ("fall_target", before["has_fall_target"], after["has_fall_target"]),
    ]
    for i, (name, b_ready, a_ready) in enumerate(checks):
        y = 495 + i * 48
        parts += [
            text(110, y, name, 20, "#344054", 700),
            text(420, y, "ready" if b_ready else "missing", 20, "#059669" if b_ready else "#dc2626", 800),
            text(590, y, "ready" if a_ready else "missing", 20, "#059669" if a_ready else "#dc2626", 800),
        ]
    parts += [
        text(420, 472, "Before", 17, "#667085", 700),
        text(590, 472, "After", 17, "#667085", 700),
    ]

    parts += [
        rect(830, 395, 700, 315, "#ffffff"),
        text(860, 445, "Signal Statistics", 28, "#172033", 850),
        text(860, 500, f"SVM mean: {before['svm_mean']:.3f}g -> {after['svm_mean']:.3f}g", 22, "#344054", 700),
        text(860, 548, f"SVM p95:  {before['svm_p95']:.3f}g -> {after['svm_p95']:.3f}g", 22, "#344054", 700),
        text(860, 596, f"SVM p99:  {before['svm_p99']:.3f}g -> {after['svm_p99']:.3f}g", 22, "#344054", 700),
        text(860, 644, f"Gyro mean: {before['gyro_mean']:.1f} -> {after['gyro_mean']:.1f} dps", 22, "#344054", 700),
        text(860, 686, "Signal values are preserved; preprocessing adds explicit model features.", 16, "#667085", 550),
    ]

    parts += [
        rect(70, 760, 1460, 270, "#ffffff"),
        text(100, 812, "Training Readiness Score", 30, "#172033", 850),
    ]
    before_ready = before["model_features_ready"] / before["model_features_total"]
    after_ready = after["model_features_ready"] / after["model_features_total"]
    parts += bar(120, 875, 580, 28, before_ready, "#d97706", f"Before: {before['model_features_ready']}/{before['model_features_total']} model features")
    parts += [text(720, 900, pct(before_ready), 24, "#9a3412", 850)]
    parts += bar(120, 955, 580, 28, after_ready, "#2563eb", f"After: {after['model_features_ready']}/{after['model_features_total']} model features")
    parts += [text(720, 980, pct(after_ready), 24, "#1d4ed8", 850)]

    parts += [
        text(860, 870, "Preprocessing result", 24, "#172033", 850),
        text(860, 912, "The after file contains a binary fall target and derived temporal features.", 19, "#344054", 600),
        text(860, 952, "It is directly usable for the IMU Fall LSTM input pipeline.", 19, "#344054", 600),
        text(70, 1092, f"Before: {source_before}", 16, "#667085", 500),
        text(1530, 1092, f"After: {source_after}", 16, "#667085", 500, "end"),
        "</svg>",
    ]
    return "\n".join(parts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, default=Path("../data/iccas_sensor_lstm/final_iccas_sisfall_imu_merged.csv"))
    parser.add_argument("--after", type=Path, default=Path("../data/iccas_sensor_lstm/imu_fall_preprocessed.csv"))
    parser.add_argument("--svg-output", type=Path, default=Path("assets/imu_preprocessing_before_after_metrics.svg"))
    parser.add_argument("--summary", type=Path, default=Path("../data/iccas_sensor_lstm/imu_preprocessing_before_after_metrics.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    before_frame = load_frame(args.before)
    after_frame = load_frame(args.after)
    before = derive_metrics(before_frame, "before")
    after = derive_metrics(after_frame, "after")
    args.svg_output.parent.mkdir(parents=True, exist_ok=True)
    args.svg_output.write_text(render_svg(before, after, args.before, args.after), encoding="utf-8")
    summary = {"before": before, "after": after}
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"svg_output": str(args.svg_output), "summary": str(args.summary), **summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
