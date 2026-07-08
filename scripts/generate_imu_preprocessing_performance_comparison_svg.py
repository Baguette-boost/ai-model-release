"""Generate an IMU fall before/after preprocessing performance comparison SVG."""

from __future__ import annotations

import html
from pathlib import Path


WIDTH = 1600
HEIGHT = 900
OUTPUT = Path("assets/imu_preprocessing_performance_comparison.svg")

SERIES = [
    {
        "name": "Before preprocessing baseline",
        "short": "Before",
        "color": "#d97706",
        "metrics": {"Accuracy": 0.9621, "Precision": 0.8150, "Recall": 0.8509, "F1-score": 0.8325},
        "threshold": 0.82,
        "features": "9/12",
        "columns": 20,
        "source": "ICCAS final IMU Fall binary LSTM",
    },
    {
        "name": "After preprocessing + hybrid LSTM",
        "short": "After",
        "color": "#5b55d9",
        "metrics": {"Accuracy": 0.8799, "Precision": 0.8498, "Recall": 0.8863, "F1-score": 0.8677},
        "threshold": 0.35,
        "features": "12/12",
        "columns": 22,
        "source": "ICCAS + SisFall Hybrid IMU Fall LSTM",
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
    chart_h = 500
    base_y = chart_y + chart_h
    group_w = chart_w / len(METRICS)
    bar_w = 82
    bar_gap = 36
    group_bar_w = len(SERIES) * bar_w + bar_gap

    before_f1 = SERIES[0]["metrics"]["F1-score"]
    after_f1 = SERIES[1]["metrics"]["F1-score"]
    before_recall = SERIES[0]["metrics"]["Recall"]
    after_recall = SERIES[1]["metrics"]["Recall"]

    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">',
        rect(0, 0, WIDTH, HEIGHT, "#fbfaff"),
        text(105, 72, "IMU 낙상 감지 - 전처리 전/후 성능 비교", 36, "#272145", 850),
        text(105, 113, "Accuracy · Precision · Recall · F1-score (0-1)", 18, "#6f6791", 650),
    ]

    legend_x = 930
    for i, item in enumerate(SERIES):
        x = legend_x + i * 275
        parts += [
            rect(x, 47, 20, 20, item["color"], "none", 5),
            text(x + 30, 64, item["short"], 18, "#514872", 800),
            text(x + 30, 91, item["source"], 13, "#8a82a6", 600),
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
        for series_index, item in enumerate(SERIES):
            value = item["metrics"][metric]
            x = start_x + series_index * (bar_w + bar_gap)
            bar_h = chart_h * value
            y = base_y - bar_h
            parts += [
                rect(x, y, bar_w, bar_h, item["color"], "none", 5),
                text(x + bar_w / 2, y - 10, f"{value:.3f}", 16, "#514872", 750, "middle"),
            ]
        parts.append(text(group_center, base_y + 48, metric, 19, "#514872", 780, "middle"))

    parts += [
        rect(90, 735, 1420, 100, "#ffffff", "#eee9fb", 10),
        text(125, 780, "전처리 효과", 19, "#514872", 850),
        text(125, 820, "성능 변화", 19, "#514872", 850),
        rect(350, 760, 14, 14, SERIES[0]["color"], "none", 7),
        text(372, 775, f'Before features {SERIES[0]["features"]} · columns {SERIES[0]["columns"]} · threshold {SERIES[0]["threshold"]:.2f}', 16, "#514872", 720),
        rect(350, 800, 14, 14, SERIES[1]["color"], "none", 7),
        text(372, 815, f'After features {SERIES[1]["features"]} · columns {SERIES[1]["columns"]} · threshold {SERIES[1]["threshold"]:.2f}', 16, "#514872", 720),
        text(910, 775, f"Recall +{after_recall - before_recall:.4f}", 18, "#5b55d9", 850),
        text(1110, 775, f"F1-score +{after_f1 - before_f1:.4f}", 18, "#5b55d9", 850),
        text(910, 815, "Accuracy is lower because the after set includes harder ICCAS+SisFall evaluation.", 15, "#8a82a6", 650),
    ]

    parts += [
        text(90, 870, "Note: before/after values come from saved project reports and are not a strict same-split ablation.", 15, "#8a82a6", 600),
        text(1510, 870, "After F1: 0.8677", 17, "#5b55d9", 850, "end"),
        "</svg>",
    ]
    return "\n".join(parts)


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(render_svg(), encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
