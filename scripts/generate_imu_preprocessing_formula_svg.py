"""Generate an SVG for IMU fall preprocessing formulas."""

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


def mono(x: float, y: float, value: object, size: int = 20, color: str = "#172033", weight: int = 650) -> str:
    return (
        f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" '
        f'font-weight="{weight}" text-anchor="start" '
        f'font-family="SFMono-Regular,Consolas,Liberation Mono,Menlo,monospace">{esc(value)}</text>'
    )


def rect(x: float, y: float, w: float, h: float, fill: str, stroke: str = "#d9dee8", rx: float = 8) -> str:
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}"/>'


def formula_card(x: float, y: float, w: float, h: float, title: str, formulas: list[str], note: str, fill: str) -> list[str]:
    parts = [rect(x, y, w, h, fill), text(x + 26, y + 44, title, 25, "#172033", 850)]
    line_y = y + 86
    for formula in formulas:
        parts.append(mono(x + 26, line_y, formula, 20, "#111827", 700))
        line_y += 36
    if note:
        parts.append(text(x + 26, y + h - 26, note, 16, "#667085", 550))
    return parts


def render_svg() -> str:
    width = 1800
    height = 1320
    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        rect(0, 0, width, height, "#f8fafc", "none", 0),
        text(80, 82, "IMU Fall Detection Preprocessing Formulas", 44, "#111827", 850),
        text(80, 122, "Formula summary for 25 Hz IMU data, 2-second windows, and BiLSTM inference", 22, "#667085", 500),
        text(1720, 82, "X_window = [50, 12]", 22, "#2563eb", 850, "end"),
        text(1720, 116, "50 samples x 40 ms = 2.0 s", 18, "#667085", 700, "end"),
    ]

    parts += formula_card(
        80,
        180,
        520,
        230,
        "1. Acceleration Magnitude / SVM",
        [
            "accel_norm_t = sqrt(ax_t^2 + ay_t^2 + az_t^2)",
            "svm_t        = sqrt(ax_t^2 + ay_t^2 + az_t^2)",
            "svm_g == accel_norm",
        ],
        "Used to detect impact intensity.",
        "#eff6ff",
    )
    parts += formula_card(
        640,
        180,
        520,
        230,
        "2. Gyroscope Magnitude",
        [
            "gyro_norm_t = sqrt(wx_t^2 + wy_t^2 + wz_t^2)",
            "unit: deg/s",
            "gyro_peak = max(gyro_norm_t)",
        ],
        "Used to detect fast body rotation.",
        "#ecfdf3",
    )
    parts += formula_card(
        1200,
        180,
        520,
        230,
        "3. Time Delta",
        [
            "dt_ms_t = t_ms_t - t_ms_(t-1)",
            "dt_ms_t = clip(dt_ms_t, 0, 1000)",
            "dt_s_t  = dt_ms_t / 1000",
        ],
        "Normal 25 Hz sampling gives dt_s ~= 0.04.",
        "#fff7ed",
    )

    parts += formula_card(
        80,
        460,
        520,
        280,
        "4. Binary Fall Target",
        [
            "fall_target_t = 1, if label_t == \"fall\"",
            "fall_target_t = 0, otherwise",
            "y_window = max(fall_target_start ... end)",
        ],
        "A window is positive if any sample inside it is fall.",
        "#fef2f2",
    )
    parts += formula_card(
        640,
        460,
        520,
        280,
        "5. LSTM Feature Vector",
        [
            "x_t = [roll, pitch, yaw,",
            "       ax, ay, az, wx, wy, wz,",
            "       accel_norm, gyro_norm, dt_s]",
            "X_window = [50 timesteps, 12 features]",
        ],
        "This tensor is the direct BiLSTM input.",
        "#f5f3ff",
    )
    parts += formula_card(
        1200,
        460,
        520,
        280,
        "6. Robust Scaling",
        [
            "center_j = median(feature_j)",
            "scale_j  = Q3(feature_j) - Q1(feature_j)",
            "z_(t,j) = (x_(t,j) - center_j) / scale_j",
            "z_(t,j) = clip(z_(t,j), -12, 12)",
        ],
        "Scaler statistics come from the training split.",
        "#eef2ff",
    )

    parts += formula_card(
        80,
        790,
        800,
        330,
        "7. Physical Fall Context Features",
        [
            "impact_peak  = max(accel_norm_t)",
            "freefall_min = min(accel_norm_t)",
            "tilt_change_t = sqrt((roll_t-roll_0)^2 + (pitch_t-pitch_0)^2)",
            "tilt_change = max(tilt_change_t)",
            "post_accel_std = std(accel_norm after impact)",
            "post_gyro_mean = mean(gyro_norm after impact)",
        ],
        "These are auxiliary context features; current final checkpoint uses LSTM-only decision.",
        "#ffffff",
    )
    parts += formula_card(
        920,
        790,
        800,
        330,
        "8. Algorithm Score & Final Decision",
        [
            "impact_score = clip((impact_peak - 2.5) / 2.5, 0, 1)",
            "rotation_score = max(clip(gyro_peak/250,0,1),",
            "                     clip(tilt_change/45,0,1))",
            "algorithm_score = 0.40*impact + 0.20*rotation",
            "                + 0.25*inactivity + 0.15*freefall_or_tilt",
            "fall_detected = LSTM(X_window) >= 0.35",
        ],
        "Tuned hybrid weight: LSTM 1.0 / Algorithm 0.0.",
        "#ffffff",
    )

    parts += [
        rect(80, 1170, 1640, 70, "#111827", "#111827"),
        text(120, 1213, "Code references", 22, "#ffffff", 850),
        text(350, 1213, "export_imu_preprocessed_csv.py  |  train_sisfall_merged_imu_lstm.py  |  train_hybrid_imu_fall.py", 19, "#e5e7eb", 600),
        "</svg>",
    ]
    return "\n".join(parts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--svg-output", type=Path, default=Path("assets/imu_fall_preprocessing_formulas.svg"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.svg_output.parent.mkdir(parents=True, exist_ok=True)
    args.svg_output.write_text(render_svg(), encoding="utf-8")
    print(json.dumps({"svg_output": str(args.svg_output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
