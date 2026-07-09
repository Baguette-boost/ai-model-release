"""Generate a brown-palette SVG for RNN, LSTM, and Transformer IMU performance."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXTENDED_SOURCE = ROOT / "experiments" / "imu_filter_engineering_extended" / "metrics.csv"
TRANSFORMER_CPU_SOURCE = ROOT / "experiments" / "imu_transformer_cpu_only" / "metrics.csv"
OUTPUT = ROOT / "assets" / "imu_filter_extended_performance_comparison_1.svg"

WIDTH = 1500
HEIGHT = 900
MODELS = ["rnn", "lstm", "transformer"]
MODEL_LABELS = {"rnn": "RNN", "lstm": "LSTM", "transformer": "Transformer"}
COLORS = {"rnn": "#B5773C", "lstm": "#7A6A58", "transformer": "#3E2B20"}
METRICS = ["accuracy", "precision", "recall", "f1"]
METRIC_LABELS = {
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1-score",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def best_for_model(rows: list[dict[str, str]], model: str) -> dict[str, str]:
    candidates = [row for row in rows if row["model"] == model]
    if not candidates:
        raise ValueError(f"No metrics found for model: {model}")
    return max(candidates, key=lambda row: float(row["f1"]))


def selected_rows() -> list[dict[str, str]]:
    extended = read_csv(EXTENDED_SOURCE)
    transformer_cpu = read_csv(TRANSFORMER_CPU_SOURCE)
    rows = [
        best_for_model(extended, "rnn"),
        best_for_model(extended, "lstm"),
        best_for_model(transformer_cpu, "transformer"),
    ]
    rows[2] = {**rows[2], "filter_variant": "baseline CPU-only"}
    return rows


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


def render_svg(rows: list[dict[str, str]]) -> str:
    chart_x = 110
    chart_y = 150
    chart_w = 1280
    chart_h = 450
    base_y = chart_y + chart_h
    metric_gap = 56
    group_w = (chart_w - metric_gap * (len(METRICS) - 1)) / len(METRICS)
    bar_gap = 18
    bar_w = (group_w - bar_gap * (len(rows) - 1)) / len(rows)
    best = max(rows, key=lambda row: float(row["f1"]))

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">',
        rect(0, 0, WIDTH, HEIGHT, "#FBF7F1"),
        text(90, 72, "IMU Fall Detection - RNN / LSTM / Transformer", 38, "#3E2B20", 850),
        text(90, 112, "Latest model metrics · CPU-based Transformer · Accuracy / Precision / Recall / F1-score", 18, "#7A6A58", 650),
    ]

    legend_x = 880
    for index, row in enumerate(rows):
        model = row["model"]
        x = legend_x + index * 170
        parts.append(rect(x, 51, 18, 18, COLORS[model], rx=4))
        parts.append(text(x + 28, 66, MODEL_LABELS[model], 16, "#3E2B20", 750))

    for tick in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        y = base_y - tick * chart_h
        parts.append(line(chart_x, y, chart_x + chart_w, y))
        parts.append(text(chart_x - 28, y + 6, f"{tick:.1f}", 15, "#7A6A58", 600, "end"))
    parts.append(line(chart_x, base_y, chart_x + chart_w, base_y, "#C8B89F", 1.5))

    for metric_index, metric in enumerate(METRICS):
        group_x = chart_x + metric_index * (group_w + metric_gap)
        for model_index, row in enumerate(rows):
            model = row["model"]
            value = float(row[metric])
            bar_h = value * chart_h
            x = group_x + model_index * (bar_w + bar_gap)
            y = base_y - bar_h
            parts.append(rect(x, y, bar_w, bar_h, COLORS[model], rx=5))
            parts.append(text(x + bar_w / 2, y - 10, f"{value:.3f}", 15, "#3E2B20", 800, "middle"))
        parts.append(text(group_x + group_w / 2, base_y + 44, METRIC_LABELS[metric], 19, "#3E2B20", 800, "middle"))

    panel_y = 690
    parts.append(rect(90, panel_y, 1320, 126, "#FFFFFF", "#E5D9CA", 10))
    parts.append(text(130, panel_y + 40, "Model", 18, "#3E2B20", 850))
    parts.append(text(130, panel_y + 78, "Best filter", 18, "#3E2B20", 850))
    parts.append(text(130, panel_y + 112, "Inference / train", 18, "#3E2B20", 850))

    item_x = 330
    col_w = 335
    for index, row in enumerate(rows):
        model = row["model"]
        x = item_x + index * col_w
        parts.append(rect(x, panel_y + 23, 16, 16, COLORS[model], rx=8))
        parts.append(text(x + 26, panel_y + 38, MODEL_LABELS[model], 17, "#3E2B20", 850))
        parts.append(text(x + 26, panel_y + 76, row["filter_variant"], 16, "#7A6A58", 700))
        parts.append(
            text(
                x + 26,
                panel_y + 111,
                f"{float(row['single_sequence_ms']):.3f} ms · {float(row['train_seconds']):.1f}s · best epoch {row['best_epoch']}",
                15,
                "#7A6A58",
                650,
            )
        )

    parts.append(
        text(
            90,
            862,
            f"Best in this view: {MODEL_LABELS[best['model']]} | F1 {float(best['f1']):.4f} | Recall {float(best['recall']):.4f} | Accuracy {float(best['accuracy']):.4f}",
            20,
            "#3E2B20",
            850,
        )
    )
    parts.append(text(1410, 862, "Sources: extended metrics + transformer CPU-only metrics", 15, "#7A6A58", 600, "end"))
    parts.append("</svg>")
    return "\n".join(parts)


def main() -> None:
    rows = selected_rows()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(render_svg(rows), encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
