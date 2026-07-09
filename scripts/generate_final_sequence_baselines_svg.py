"""Generate a SVG bar chart for final-data RNN, LSTM, and Transformer results."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASELINE_SOURCE = ROOT / "experiments" / "final_sequence_baselines" / "metrics.csv"
FINAL_LSTM_SOURCE = ROOT / "models" / "iccas_final_hybrid_lstm_imu_fall.json"
LSTM_SPEED_SOURCE = ROOT.parent / "data" / "iccas_sensor_lstm" / "imu_lstm_speed_preprocessing_metrics.json"
OUTPUT = ROOT / "assets" / "final_sequence_baselines_performance.svg"

WIDTH = 1500
HEIGHT = 900
MODELS = ["rnn", "lstm", "transformer"]
LABELS = {"rnn": "RNN", "lstm": "Final LSTM", "transformer": "Transformer"}
COLORS = {"rnn": "#B5773C", "lstm": "#7A6A58", "transformer": "#3E2B20"}
METRICS = ["accuracy", "precision", "recall", "f1"]
METRIC_LABELS = {
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1-score",
}


def esc(value: object) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def text(
    x: float,
    y: float,
    value: object,
    size: int = 20,
    color: str = "#3E2B20",
    weight: int = 600,
    anchor: str = "start",
) -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" '
        f'font-family="Inter, Pretendard, Arial, sans-serif" font-size="{size}" '
        f'font-weight="{weight}" fill="{color}">{esc(value)}</text>'
    )


def rect(x: float, y: float, width: float, height: float, fill: str, stroke: str = "none", rx: float = 0) -> str:
    return (
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" height="{height:.1f}" '
        f'rx="{rx:.1f}" fill="{fill}" stroke="{stroke}"/>'
    )


def line(x1: float, y1: float, x2: float, y2: float, stroke: str = "#DED3C4", width: float = 1.0) -> str:
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{stroke}" stroke-width="{width:.1f}"/>'


def baseline_rows() -> dict[str, dict[str, str]]:
    with BASELINE_SOURCE.open("r", encoding="utf-8", newline="") as file:
        rows = {row["architecture"]: row for row in csv.DictReader(file)}
    return rows


def final_lstm_row() -> dict[str, str]:
    checkpoint = json.loads(FINAL_LSTM_SOURCE.read_text(encoding="utf-8"))
    speed = json.loads(LSTM_SPEED_SOURCE.read_text(encoding="utf-8"))["inference_speed"]
    metrics = checkpoint["test_metrics"]["lstm_only"]
    return {
        "architecture": "lstm",
        "accuracy": str(metrics["accuracy"]),
        "precision": str(metrics["precision"]),
        "recall": str(metrics["recall"]),
        "f1": str(metrics["f1"]),
        "threshold": str(metrics.get("threshold", checkpoint.get("threshold", ""))),
        "forward_only_ms": str(speed["realtime_forward_only_median_ms"]),
        "tensor_create_plus_forward_ms": str(speed["realtime_tensor_forward_median_ms"]),
        "batch_per_sequence_ms": "",
        "window_seconds": str(speed["window_seconds"]),
        "train_seconds": "n/a",
        "best_epoch": "10",
        "epochs_trained": "15",
        "parameter_count": "139778",
    }


def selected_rows() -> list[dict[str, str]]:
    rows = baseline_rows()
    rows["lstm"] = final_lstm_row()
    return [rows[model] for model in MODELS]


def process_label(row: dict[str, str]) -> str:
    window_seconds = float(row["window_seconds"])
    process_ms = float(row["tensor_create_plus_forward_ms"])
    return f"{process_ms:.3f} ms / ~{window_seconds + process_ms / 1000.0:.3f}s incl. window"


def render_svg(rows: list[dict[str, str]]) -> str:
    chart_x = 110
    chart_y = 150
    chart_w = 1280
    chart_h = 455
    base_y = chart_y + chart_h
    metric_gap = 56
    group_w = (chart_w - metric_gap * (len(METRICS) - 1)) / len(METRICS)
    bar_gap = 18
    bar_w = (group_w - bar_gap * (len(rows) - 1)) / len(rows)
    best_f1 = max(rows, key=lambda row: float(row["f1"]))
    fastest = min(rows, key=lambda row: float(row["forward_only_ms"]))

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">',
        rect(0, 0, WIDTH, HEIGHT, "#FBF7F1"),
        text(90, 72, "IMU Fall Detection - Final Data Model Comparison", 38, "#3E2B20", 850),
        text(90, 112, "Same final merged dataset, 50-sample sequence, train-only robust scaling, CPU single-thread timing", 18, "#7A6A58", 650),
    ]

    legend_x = 825
    for index, row in enumerate(rows):
        model = row["architecture"]
        x = legend_x + index * 190
        parts.append(rect(x, 51, 18, 18, COLORS[model], rx=4))
        parts.append(text(x + 28, 66, LABELS[model], 16, "#3E2B20", 750))

    for tick in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        y = base_y - tick * chart_h
        parts.append(line(chart_x, y, chart_x + chart_w, y))
        parts.append(text(chart_x - 28, y + 6, f"{tick:.1f}", 15, "#7A6A58", 600, "end"))
    parts.append(line(chart_x, base_y, chart_x + chart_w, base_y, "#C8B89F", 1.5))

    for metric_index, metric in enumerate(METRICS):
        group_x = chart_x + metric_index * (group_w + metric_gap)
        for model_index, row in enumerate(rows):
            model = row["architecture"]
            value = float(row[metric])
            bar_h = value * chart_h
            x = group_x + model_index * (bar_w + bar_gap)
            y = base_y - bar_h
            parts.append(rect(x, y, bar_w, bar_h, COLORS[model], rx=5))
            parts.append(text(x + bar_w / 2, y - 10, f"{value:.3f}", 15, "#3E2B20", 800, "middle"))
        parts.append(text(group_x + group_w / 2, base_y + 44, METRIC_LABELS[metric], 19, "#3E2B20", 800, "middle"))

    panel_y = 690
    parts.append(rect(90, panel_y, 1320, 128, "#FFFFFF", "#E5D9CA", 10))
    parts.append(text(130, panel_y + 38, "Model", 18, "#3E2B20", 850))
    parts.append(text(130, panel_y + 75, "Inference", 18, "#3E2B20", 850))
    parts.append(text(130, panel_y + 111, "Train / epoch", 18, "#3E2B20", 850))

    item_x = 330
    col_w = 335
    for index, row in enumerate(rows):
        model = row["architecture"]
        x = item_x + index * col_w
        train_seconds = row["train_seconds"]
        train_label = "n/a" if train_seconds == "n/a" else f"{float(train_seconds):.1f}s"
        parts.append(rect(x, panel_y + 22, 16, 16, COLORS[model], rx=8))
        parts.append(text(x + 26, panel_y + 37, LABELS[model], 17, "#3E2B20", 850))
        parts.append(text(x + 26, panel_y + 74, process_label(row), 15, "#7A6A58", 700))
        parts.append(text(x + 26, panel_y + 110, f"{train_label} · best epoch {row['best_epoch']}", 15, "#7A6A58", 700))

    parts.append(
        text(
            90,
            862,
            f"Best F1: {LABELS[best_f1['architecture']]} {float(best_f1['f1']):.4f} | "
            f"Fastest forward: {LABELS[fastest['architecture']]} {float(fastest['forward_only_ms']):.3f} ms",
            20,
            "#3E2B20",
            850,
        )
    )
    parts.append(text(1410, 862, "Decision latency also includes 2.0s IMU collection window", 15, "#7A6A58", 600, "end"))
    parts.append("</svg>")
    return "\n".join(parts)


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(render_svg(selected_rows()), encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
