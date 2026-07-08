"""Generate a presentation-ready IMU fall model performance comparison SVG."""

from __future__ import annotations

import html
from pathlib import Path


WIDTH = 1600
HEIGHT = 900
OUTPUT = Path("assets/imu_model_performance_comparison.svg")

MODELS = [
    {
        "name": "RNN",
        "color": "#5b55d9",
        "metrics": {"Accuracy": 0.9105, "Precision": 0.6016, "Recall": 0.5648, "F1-score": 0.5826},
        "single_ms": 0.091,
        "train_sec": 4.3,
    },
    {
        "name": "GRU",
        "color": "#8177ee",
        "metrics": {"Accuracy": 0.9110, "Precision": 0.6504, "Recall": 0.4230, "F1-score": 0.5126},
        "single_ms": 0.199,
        "train_sec": 10.3,
    },
    {
        "name": "LSTM",
        "color": "#a496ee",
        "metrics": {"Accuracy": 0.9521, "Precision": 0.7788, "Recall": 0.7922, "F1-score": 0.7855},
        "single_ms": 0.199,
        "train_sec": 12.6,
    },
    {
        "name": "Transformer",
        "color": "#d9cff8",
        "metrics": {"Accuracy": 0.9299, "Precision": 0.7863, "Recall": 0.5037, "F1-score": 0.6140},
        "single_ms": 0.278,
        "train_sec": 24.9,
    },
]

METRICS = ["Accuracy", "Precision", "Recall", "F1-score"]


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def text(
    x: float,
    y: float,
    value: object,
    size: int = 22,
    color: str = "#272145",
    weight: int = 500,
    anchor: str = "start",
) -> str:
    return (
        f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" '
        f'font-weight="{weight}" text-anchor="{anchor}" '
        f'font-family="-apple-system,BlinkMacSystemFont,Segoe UI,Noto Sans KR,sans-serif">{esc(value)}</text>'
    )


def rect(x: float, y: float, w: float, h: float, fill: str, stroke: str = "none", rx: float = 0) -> str:
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}"/>'


def line(x1: float, y1: float, x2: float, y2: float, color: str = "#e7e2f5", width: float = 1) -> str:
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{width}"/>'


def render_svg() -> str:
    chart_x = 145
    chart_y = 165
    chart_w = 1310
    chart_h = 505
    base_y = chart_y + chart_h
    group_w = chart_w / len(METRICS)
    bar_w = 54
    bar_gap = 18
    group_bar_w = len(MODELS) * bar_w + (len(MODELS) - 1) * bar_gap

    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">',
        rect(0, 0, WIDTH, HEIGHT, "#fbfaff"),
        rect(0, 0, WIDTH, HEIGHT, "#fbfaff"),
        text(105, 72, "IMU 낙상 감지 - 모델별 성능 비교", 36, "#272145", 850),
        text(105, 113, "Accuracy · Precision · Recall · F1-score (0-1)", 18, "#6f6791", 650),
    ]

    legend_x = 980
    for i, model in enumerate(MODELS):
        x = legend_x + i * 145
        parts += [
            rect(x, 48, 18, 18, model["color"], "none", 4),
            text(x + 28, 64, model["name"], 18, "#514872", 700),
        ]

    for i in range(6):
        value = i / 5
        y = base_y - chart_h * value
        parts += [
            line(chart_x, y, chart_x + chart_w, y),
            text(chart_x - 28, y + 6, f"{value:.1f}", 16, "#796f9c", 600, "end"),
        ]

    parts += [
        line(chart_x, chart_y, chart_x, base_y, "#d6cfea", 1.4),
        line(chart_x, base_y, chart_x + chart_w, base_y, "#d6cfea", 1.4),
    ]

    for metric_index, metric in enumerate(METRICS):
        group_center = chart_x + group_w * metric_index + group_w / 2
        start_x = group_center - group_bar_w / 2
        for model_index, model in enumerate(MODELS):
            value = model["metrics"][metric]
            x = start_x + model_index * (bar_w + bar_gap)
            bar_h = chart_h * value
            y = base_y - bar_h
            parts += [
                rect(x, y, bar_w, bar_h, model["color"], "none", 5),
                text(x + bar_w / 2, y - 10, f"{value:.3f}", 15, "#514872", 700, "middle"),
            ]
        parts.append(text(group_center, base_y + 48, metric, 19, "#514872", 780, "middle"))

    parts += [
        rect(90, 735, 1420, 98, "#ffffff", "#eee9fb", 10),
        text(125, 780, "단건 추론 (ms)", 19, "#514872", 850),
        text(125, 820, "학습 시간 (s)", 19, "#514872", 850),
    ]

    metric_x = 355
    for i, model in enumerate(MODELS):
        x = metric_x + i * 285
        parts += [
            rect(x, 763, 16, 16, model["color"], "none", 8),
            text(x + 24, 778, f'{model["name"]} {model["single_ms"]:.3f}', 16, "#514872", 720),
            text(x + 24, 819, f'{model["train_sec"]:.1f}', 16, "#514872", 720),
        ]

    parts += [
        text(90, 870, "Source: docs/MODEL_ARCHITECTURE_COMPARISON_RESULTS.md · 실험 조건: scenario별 chronological split, seq_len 16, epoch 15, CPU", 15, "#8a82a6", 600),
        text(1510, 870, "Best F1: LSTM 0.7855", 17, "#5b55d9", 850, "end"),
        "</svg>",
    ]
    return "\n".join(parts)


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(render_svg(), encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
