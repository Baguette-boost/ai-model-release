"""Generate a presentation-ready IMU fall inference speed comparison SVG."""

from __future__ import annotations

import html
from pathlib import Path


WIDTH = 1600
HEIGHT = 900
OUTPUT = Path("assets/imu_inference_speed_model_comparison.svg")

MODELS = [
    {"name": "RNN", "color": "#5b55d9", "single_ms": 0.091, "batch_ms": 0.0040, "f1": 0.5826, "params": 13505},
    {"name": "GRU", "color": "#8177ee", "single_ms": 0.199, "batch_ms": 0.0124, "f1": 0.5126, "params": 40129},
    {"name": "LSTM", "color": "#a496ee", "single_ms": 0.199, "batch_ms": 0.0139, "f1": 0.7855, "params": 53441},
    {"name": "Transformer", "color": "#d9cff8", "single_ms": 0.278, "batch_ms": 0.0203, "f1": 0.6140, "params": 67969},
]


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


def speed_panel(
    x: float,
    y: float,
    width: float,
    title: str,
    key: str,
    max_value: float,
    unit: str,
    note: str,
) -> list[str]:
    parts = [
        rect(x, y, width, 250, "#ffffff", "#eee9fb", 10),
        text(x + 28, y + 47, title, 25, "#272145", 850),
        text(x + width - 28, y + 47, note, 15, "#8a82a6", 700, "end"),
    ]
    chart_x = x + 170
    chart_y = y + 82
    chart_w = width - 260
    row_h = 39
    for i in range(4):
        value = max_value * i / 3
        xx = chart_x + chart_w * i / 3
        parts += [
            line(xx, chart_y - 15, xx, chart_y + row_h * len(MODELS) + 8, "#f0ecfb"),
            text(xx, chart_y + row_h * len(MODELS) + 35, f"{value:.3f}", 13, "#8a82a6", 600, "middle"),
        ]
    for i, model in enumerate(MODELS):
        yy = chart_y + i * row_h
        value = float(model[key])
        bar_w = chart_w * value / max_value
        parts += [
            text(x + 34, yy + 22, model["name"], 17, "#514872", 800),
            rect(chart_x, yy, chart_w, 24, "#f2effb", "none", 8),
            rect(chart_x, yy, bar_w, 24, model["color"], "none", 8),
            text(chart_x + bar_w + 12, yy + 19, f"{value:.4f} {unit}" if value < 0.01 else f"{value:.3f} {unit}", 15, "#514872", 750),
        ]
    return parts


def render_svg() -> str:
    fastest = min(MODELS, key=lambda item: item["single_ms"])
    best_f1 = max(MODELS, key=lambda item: item["f1"])
    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">',
        rect(0, 0, WIDTH, HEIGHT, "#fbfaff"),
        text(105, 72, "IMU 낙상 감지 - 모델별 추론 속도 비교", 36, "#272145", 850),
        text(105, 113, "Single inference ms · Batch per sequence ms · Lower is better", 18, "#6f6791", 650),
    ]

    legend_x = 930
    for i, model in enumerate(MODELS):
        x = legend_x + i * 145
        parts += [
            rect(x, 48, 18, 18, model["color"], "none", 4),
            text(x + 28, 64, model["name"], 18, "#514872", 700),
        ]

    parts += speed_panel(90, 165, 1420, "실시간 단건 추론 속도", "single_ms", 0.30, "ms", "one sequence forward pass")
    parts += speed_panel(90, 455, 1420, "Batch 평가 속도", "batch_ms", 0.022, "ms/seq", "batch size 256")

    parts += [
        rect(90, 735, 1420, 98, "#ffffff", "#eee9fb", 10),
        text(125, 780, "해석", 19, "#514872", 850),
        text(260, 780, f"가장 빠른 단건 추론: {fastest['name']} {fastest['single_ms']:.3f} ms", 17, "#514872", 760),
        text(260, 820, f"최종 선택 관점: {best_f1['name']}은 {best_f1['single_ms']:.3f} ms로 충분히 빠르고 F1-score {best_f1['f1']:.4f}로 가장 높음", 17, "#5b55d9", 850),
        text(1050, 780, "Model params", 17, "#514872", 850),
    ]
    for i, model in enumerate(MODELS):
        x = 1175 + i * 78
        parts += [
            rect(x, 800 - model["params"] / 1000, 34, model["params"] / 1000, model["color"], "none", 4),
            text(x + 17, 823, model["name"][0] if model["name"] != "Transformer" else "T", 13, "#514872", 800, "middle"),
        ]

    parts += [
        text(90, 870, "Source: docs/MODEL_ARCHITECTURE_COMPARISON_RESULTS.md · Task: imu_fall · CPU · seq_len 16 · epoch 15", 15, "#8a82a6", 600),
        text(1510, 870, "Recommended: LSTM", 17, "#5b55d9", 850, "end"),
        "</svg>",
    ]
    return "\n".join(parts)


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(render_svg(), encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
