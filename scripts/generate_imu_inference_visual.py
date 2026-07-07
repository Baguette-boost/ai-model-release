"""Generate an English IMU fall inference flow visualization."""

from __future__ import annotations

import argparse
import html
import json
import shutil
import subprocess
from pathlib import Path


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def text(x: float, y: float, value: object, size: int = 24, color: str = "#172033", weight: int = 500, anchor: str = "start") -> str:
    return (
        f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" '
        f'font-weight="{weight}" text-anchor="{anchor}" '
        f'font-family="-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif">{esc(value)}</text>'
    )


def rect(x: float, y: float, w: float, h: float, fill: str, stroke: str = "#d9dee8", rx: float = 8) -> str:
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}"/>'


def line(x1: float, y1: float, x2: float, y2: float, color: str = "#98a2b3", width: int = 3) -> str:
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{width}" marker-end="url(#arrow)"/>'


def metric(value: float) -> str:
    return f"{value * 100:.1f}%"


def render_svg(meta: dict) -> str:
    width = 1600
    height = 1050
    test = meta["test_metrics"]
    lstm = test["lstm_only"]
    algorithm = test["algorithm_only"]
    hybrid = test["hybrid"]
    fall_algorithm = meta["fall_algorithm"]
    sequence_seconds = meta["sequence_length"] * meta["sample_ms"] / 1000
    features = ", ".join(meta["feature_columns"])
    weights = f"LSTM {meta['hybrid_lstm_weight']:.1f} / Algorithm {meta['hybrid_algorithm_weight']:.1f}"

    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<defs>",
        '<marker id="arrow" markerWidth="12" markerHeight="12" refX="10" refY="6" orient="auto">',
        '<path d="M2,2 L10,6 L2,10 Z" fill="#98a2b3"/>',
        "</marker>",
        "</defs>",
        rect(0, 0, width, height, "#f7f8fb", "none", 0),
        text(70, 82, "IMU Fall Detection Inference Flow", 44, "#111827", 850),
        text(70, 122, "2-second IMU window -> BiLSTM + Attention -> fall probability -> backend result", 22, "#667085", 500),
        text(1530, 82, f"Device: {meta['device'].upper()}", 18, "#667085", 700, "end"),
        text(1530, 112, f"Threshold: {meta['threshold']:.2f}", 18, "#667085", 700, "end"),
    ]

    cards = [
        ("1. Sensor Stream", "25 Hz IMU samples", f"roll/pitch/yaw + accel + gyro", "#eff6ff"),
        ("2. Window Buffer", f"{meta['sequence_length']} steps x {len(meta['feature_columns'])} features", f"{sequence_seconds:.1f}s window, stride {meta['sequence_stride']}", "#ecfdf3"),
        ("3. Preprocessing", "Robust scaling", "accel_norm, gyro_norm, dt_s", "#fff7ed"),
        ("4. BiLSTM Inference", "Bidirectional LSTM + Attention", f"hidden {meta['hidden_size']}, layers {meta['num_layers']}", "#f5f3ff"),
    ]
    y = 190
    for i, (title, main, note, fill) in enumerate(cards):
        x = 70 + i * 380
        parts += [
            rect(x, y, 330, 150, fill),
            text(x + 24, y + 42, title, 23, "#172033", 800),
            text(x + 24, y + 84, main, 20, "#344054", 650),
            text(x + 24, y + 118, note, 17, "#667085", 500),
        ]
        if i < 3:
            parts.append(line(x + 330, y + 75, x + 372, y + 75))

    parts += [
        rect(70, 390, 700, 255, "#ffffff"),
        text(100, 440, "Input Features", 28, "#172033", 850),
        text(100, 486, features, 19, "#344054", 600),
        text(100, 536, "Inference Output", 25, "#172033", 800),
        text(100, 580, "fall_score = sigmoid(BiLSTM(window))", 20, "#2563eb", 700),
        text(100, 616, f"fall_detected = fall_score >= {meta['threshold']:.2f}", 20, "#dc2626", 800),
        rect(830, 390, 700, 255, "#ffffff"),
        text(860, 440, "Rule-based Safety Checks", 28, "#172033", 850),
        text(860, 486, f"Impact gate: accel_norm >= {fall_algorithm['impact_g']}g", 20, "#344054", 650),
        text(860, 524, f"Free-fall hint: accel_norm <= {fall_algorithm['freefall_g']}g", 20, "#344054", 650),
        text(860, 562, f"Rotation gate: gyro_norm >= {fall_algorithm['gyro_dps']} dps", 20, "#344054", 650),
        text(860, 600, f"Inactivity gate: still gyro <= {fall_algorithm['still_gyro_dps']} dps", 20, "#344054", 650),
    ]

    parts += [
        rect(70, 680, 455, 245, "#ffffff"),
        text(100, 730, "Final Test Metrics", 27, "#172033", 850),
        text(100, 780, f"Accuracy  {metric(lstm['accuracy'])}", 23, "#2563eb", 800),
        text(100, 820, f"Precision {metric(lstm['precision'])}", 23, "#059669", 800),
        text(100, 860, f"Recall    {metric(lstm['recall'])}", 23, "#d97706", 800),
        text(100, 900, f"F1-score  {metric(lstm['f1'])}", 23, "#dc2626", 800),
        rect(575, 680, 455, 245, "#ffffff"),
        text(605, 730, "Hybrid Weighting", 27, "#172033", 850),
        text(605, 780, weights, 23, "#344054", 800),
        text(605, 825, "Current tuned hybrid equals LSTM-only", 19, "#667085", 600),
        text(605, 863, "Algorithm-only has high recall but many false positives", 19, "#667085", 600),
        rect(1080, 680, 450, 245, "#ffffff"),
        text(1110, 730, "Method Comparison", 27, "#172033", 850),
        text(1110, 780, f"LSTM only      F1 {metric(lstm['f1'])}", 21, "#172033", 700),
        text(1110, 820, f"Algorithm only F1 {metric(algorithm['f1'])}", 21, "#172033", 700),
        text(1110, 860, f"Hybrid tuned   F1 {metric(hybrid['f1'])}", 21, "#172033", 700),
    ]

    parts += [
        text(70, 988, "Model file: models/iccas_final_hybrid_lstm_imu_fall.pt", 18, "#667085", 600),
        text(1530, 988, "Source: ICCAS IMU + SisFall IMU merged dataset", 18, "#667085", 600, "end"),
        "</svg>",
    ]
    return "\n".join(parts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=Path("models/iccas_final_hybrid_lstm_imu_fall.json"))
    parser.add_argument("--svg-output", type=Path, default=Path("assets/imu_fall_inference_flow.svg"))
    parser.add_argument("--png-output", type=Path, default=Path("assets/imu_fall_inference_flow.png"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    meta = json.loads(args.metadata.read_text(encoding="utf-8"))
    args.svg_output.parent.mkdir(parents=True, exist_ok=True)
    args.svg_output.write_text(render_svg(meta), encoding="utf-8")
    args.png_output.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            ["sips", "-s", "format", "png", str(args.svg_output), "--out", str(args.png_output)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        subprocess.run(["qlmanage", "-t", "-s", "2000", "-o", str(args.png_output.parent), str(args.svg_output)], check=True)
        quicklook_output = args.png_output.parent / f"{args.svg_output.name}.png"
        if not quicklook_output.exists():
            raise SystemExit("PNG conversion failed")
        shutil.copyfile(quicklook_output, args.png_output)
    print(json.dumps({"svg_output": str(args.svg_output), "png_output": str(args.png_output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
