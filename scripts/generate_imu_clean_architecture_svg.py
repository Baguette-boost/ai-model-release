"""Generate a clean IMU Fall LSTM architecture SVG."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path


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


def line(x1: float, y1: float, x2: float, y2: float, color: str = "#98a2b3", width: int = 3) -> str:
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{width}" marker-end="url(#arrow)"/>'


def render_svg(meta: dict) -> str:
    width = 1800
    height = 980
    sequence_seconds = meta["sequence_length"] * meta["sample_ms"] / 1000
    f1 = meta["test_metrics"]["lstm_only"]["f1"] * 100
    realtime_ms = 1.1362

    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<defs>",
        '<marker id="arrow" markerWidth="12" markerHeight="12" refX="10" refY="6" orient="auto">',
        '<path d="M2,2 L10,6 L2,10 Z" fill="#98a2b3"/>',
        "</marker>",
        "</defs>",
        rect(0, 0, width, height, "#f8fafc", "none", 0),
        text(80, 86, "IMU Fall Detection Architecture", 46, "#111827", 850),
        text(80, 126, "Clean inference pipeline for the BiLSTM fall detector", 22, "#667085", 500),
        text(1720, 86, f"F1 {f1:.1f}%", 24, "#dc2626", 850, "end"),
        text(1720, 120, f"Realtime {realtime_ms:.2f} ms/window", 18, "#667085", 700, "end"),
    ]

    stages = [
        {
            "title": "1. IMU Stream",
            "main": "25 Hz sensor input",
            "note": "accel + gyro + posture",
            "fill": "#eff6ff",
        },
        {
            "title": "2. Window Buffer",
            "main": f"{meta['sequence_length']} steps",
            "note": f"{sequence_seconds:.1f}s rolling window",
            "fill": "#ecfdf3",
        },
        {
            "title": "3. Feature Layer",
            "main": "12 model features",
            "note": "SVM, gyro_norm, dt_s",
            "fill": "#fff7ed",
        },
        {
            "title": "4. BiLSTM + Attention",
            "main": "sequence classifier",
            "note": f"hidden {meta['hidden_size']}, {meta['num_layers']} layers",
            "fill": "#f5f3ff",
        },
        {
            "title": "5. Fall Decision",
            "main": f"score >= {meta['threshold']:.2f}",
            "note": "POST result to backend",
            "fill": "#fef2f2",
        },
    ]

    y = 230
    card_w = 300
    card_h = 170
    gap = 48
    for index, stage in enumerate(stages):
        x = 80 + index * (card_w + gap)
        parts += [
            rect(x, y, card_w, card_h, stage["fill"]),
            text(x + 24, y + 46, stage["title"], 23, "#172033", 850),
            text(x + 24, y + 96, stage["main"], 26, "#344054", 800),
            text(x + 24, y + 135, stage["note"], 18, "#667085", 550),
        ]
        if index < len(stages) - 1:
            parts.append(line(x + card_w, y + card_h / 2, x + card_w + gap - 10, y + card_h / 2))

    parts += [
        rect(80, 480, 520, 260, "#ffffff"),
        text(115, 535, "Input Tensor", 30, "#172033", 850),
        text(115, 590, f"shape = ({meta['sequence_length']}, {len(meta['feature_columns'])})", 25, "#2563eb", 850),
        text(115, 638, "roll, pitch, yaw", 19, "#344054", 600),
        text(115, 674, "ax, ay, az, wx, wy, wz", 19, "#344054", 600),
        text(115, 710, "accel_norm, gyro_norm, dt_s", 19, "#344054", 600),
        rect(640, 480, 520, 260, "#ffffff"),
        text(675, 535, "Model Block", 30, "#172033", 850),
        text(675, 590, "Bidirectional LSTM", 25, "#344054", 850),
        text(675, 638, "Attention pooling", 21, "#344054", 650),
        text(675, 674, "LayerNorm + Dropout + Linear", 21, "#344054", 650),
        text(675, 710, "Output: fall probability", 21, "#2563eb", 800),
        rect(1200, 480, 520, 260, "#ffffff"),
        text(1235, 535, "Safety Context", 30, "#172033", 850),
        text(1235, 590, "Impact / rotation / inactivity", 22, "#344054", 750),
        text(1235, 638, "Used as an auxiliary explanation", 20, "#667085", 600),
        text(1235, 674, "Current tuned weight:", 20, "#667085", 600),
        text(1235, 710, f"LSTM {meta['hybrid_lstm_weight']:.1f}, Algorithm {meta['hybrid_algorithm_weight']:.1f}", 22, "#dc2626", 850),
    ]

    parts += [
        rect(80, 810, 1640, 90, "#111827", "#111827"),
        text(120, 865, "Runtime summary", 24, "#ffffff", 850),
        text(370, 865, f"50-step ready window -> BiLSTM inference -> fall_score -> backend event", 22, "#e5e7eb", 600),
        text(1680, 865, "CPU measured", 18, "#cbd5e1", 700, "end"),
        "</svg>",
    ]
    return "\n".join(parts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=Path("models/iccas_final_hybrid_lstm_imu_fall.json"))
    parser.add_argument("--svg-output", type=Path, default=Path("assets/imu_fall_clean_architecture.svg"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    meta = json.loads(args.metadata.read_text(encoding="utf-8"))
    args.svg_output.parent.mkdir(parents=True, exist_ok=True)
    args.svg_output.write_text(render_svg(meta), encoding="utf-8")
    print(json.dumps({"svg_output": str(args.svg_output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
