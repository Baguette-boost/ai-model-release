"""Generate an SVG chart for the extended IMU filter/model performance experiment."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "experiments" / "imu_filter_engineering_extended" / "metrics.csv"
OUTPUT = ROOT / "assets" / "imu_filter_extended_performance_comparison.svg"

WIDTH = 1600
HEIGHT = 940

MODEL_ORDER = ["rnn", "gru", "lstm", "transformer", "cnn1d", "cnn_lstm"]
MODEL_LABELS = {
    "rnn": "RNN",
    "gru": "GRU",
    "lstm": "LSTM",
    "transformer": "Transformer",
    "cnn1d": "1D-CNN",
    "cnn_lstm": "CNN-LSTM",
}
COLORS = {
    "rnn": "#5b5bd6",
    "gru": "#7e73ea",
    "lstm": "#a08df4",
    "transformer": "#c7b8f7",
    "cnn1d": "#55b7a8",
    "cnn_lstm": "#f28c6b",
}
METRICS = ["accuracy", "precision", "recall", "f1"]
METRIC_LABELS = {
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1-score",
}


def load_rows() -> list[dict[str, str]]:
    with SOURCE.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def best_rows_by_model(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    best: dict[str, dict[str, str]] = {}
    for row in rows:
        model = row["model"]
        if model not in best or float(row["f1"]) > float(best[model]["f1"]):
            best[model] = row
    return [best[model] for model in MODEL_ORDER if model in best]


def best_overall(rows: list[dict[str, str]]) -> dict[str, str]:
    return max(rows, key=lambda row: float(row["f1"]))


def text(x: float, y: float, value: str, size: int = 20, color: str = "#272145", weight: int = 600, anchor: str = "start") -> str:
    escaped = (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" '
        f'font-family="Inter, Pretendard, Arial, sans-serif" font-size="{size}" '
        f'font-weight="{weight}" fill="{color}">{escaped}</text>'
    )


def rect(x: float, y: float, width: float, height: float, fill: str, stroke: str = "none", rx: float = 0) -> str:
    return (
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" height="{height:.1f}" '
        f'rx="{rx:.1f}" fill="{fill}" stroke="{stroke}"/>'
    )


def line(x1: float, y1: float, x2: float, y2: float, stroke: str = "#e7e2f5", width: float = 1.0) -> str:
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{stroke}" stroke-width="{width:.1f}"/>'


def render_svg(rows: list[dict[str, str]]) -> str:
    best_by_model = best_rows_by_model(rows)
    best = best_overall(rows)
    chart_x = 105
    chart_y = 150
    chart_w = 1390
    chart_h = 465
    base_y = chart_y + chart_h
    max_value = 1.0
    metric_gap = 44
    group_w = (chart_w - metric_gap * (len(METRICS) - 1)) / len(METRICS)
    bar_gap = 10
    bar_w = (group_w - bar_gap * (len(best_by_model) - 1)) / len(best_by_model)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">',
        rect(0, 0, WIDTH, HEIGHT, "#fbfaff"),
        text(105, 74, "IMU Fall Detection - Extended Model Performance", 38, "#272145", 850),
        text(105, 114, "Best result per model after 30 max epochs, min 15 epochs, F1-based early stopping", 18, "#6f6791", 650),
    ]

    legend_x = 720
    for index, row in enumerate(best_by_model):
        model = row["model"]
        x = legend_x + index * 135
        parts.append(rect(x, 50, 16, 16, COLORS[model], rx=4))
        parts.append(text(x + 24, 64, MODEL_LABELS[model], 15, "#514872", 700))

    for tick in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        y = base_y - (tick / max_value) * chart_h
        parts.append(line(chart_x, y, chart_x + chart_w, y))
        parts.append(text(chart_x - 28, y + 6, f"{tick:.1f}", 15, "#796f9c", 600, "end"))

    parts.append(line(chart_x, base_y, chart_x + chart_w, base_y, "#d8d0ee", 1.4))

    for metric_index, metric in enumerate(METRICS):
        group_x = chart_x + metric_index * (group_w + metric_gap)
        for model_index, row in enumerate(best_by_model):
            model = row["model"]
            value = float(row[metric])
            h = value / max_value * chart_h
            x = group_x + model_index * (bar_w + bar_gap)
            y = base_y - h
            parts.append(rect(x, y, bar_w, h, COLORS[model], rx=5))
            parts.append(text(x + bar_w / 2, y - 10, f"{value:.3f}", 14, "#514872", 700, "middle"))
        parts.append(text(group_x + group_w / 2, base_y + 44, METRIC_LABELS[metric], 19, "#514872", 800, "middle"))

    panel_y = 700
    parts.append(rect(90, panel_y, 1420, 128, "#ffffff", "#eee9fb", 10))
    parts.append(text(125, panel_y + 42, "Best filter", 18, "#514872", 850))
    parts.append(text(125, panel_y + 84, "Inference ms", 18, "#514872", 850))
    parts.append(text(125, panel_y + 116, "Train sec", 18, "#514872", 850))

    item_x = 315
    col_w = 190
    for index, row in enumerate(best_by_model):
        model = row["model"]
        x = item_x + index * col_w
        parts.append(rect(x, panel_y + 25, 14, 14, COLORS[model], rx=7))
        parts.append(text(x + 22, panel_y + 39, MODEL_LABELS[model], 16, "#514872", 800))
        parts.append(text(x + 22, panel_y + 72, row["filter_variant"], 15, "#6f6791", 650))
        parts.append(text(x + 22, panel_y + 103, f"{float(row['single_sequence_ms']):.3f}", 15, "#6f6791", 650))
        parts.append(text(x + 22, panel_y + 135, f"{float(row['train_seconds']):.1f}", 15, "#6f6791", 650))

    best_label = MODEL_LABELS[best["model"]]
    parts.append(text(105, 878, f"Overall best: {best_label} + {best['filter_variant']} | F1 {float(best['f1']):.4f} | Accuracy {float(best['accuracy']):.4f} | Recall {float(best['recall']):.4f}", 19, "#272145", 850))
    parts.append(text(1495, 878, "Source: experiments/imu_filter_engineering_extended/metrics.csv", 15, "#8a82a6", 600, "end"))
    parts.append("</svg>")
    return "\n".join(parts)


def main() -> None:
    rows = load_rows()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(render_svg(rows), encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
