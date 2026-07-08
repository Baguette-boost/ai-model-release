"""Generate a mathematical SVG summary for IMU fall preprocessing."""

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


def math_text(x: float, y: float, value: object, size: int = 23, color: str = "#111827", weight: int = 650) -> str:
    return (
        f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" '
        f'font-weight="{weight}" text-anchor="start" '
        f'font-family="Times New Roman, STIX Two Text, Cambria Math, serif">{esc(value)}</text>'
    )


def rect(x: float, y: float, w: float, h: float, fill: str, stroke: str = "#d9dee8", rx: float = 8) -> str:
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}"/>'


def card(x: float, y: float, w: float, h: float, title: str, formulas: list[str], note: str, fill: str) -> list[str]:
    parts = [rect(x, y, w, h, fill), text(x + 28, y + 48, title, 25, "#172033", 850)]
    yy = y + 96
    for formula in formulas:
        parts.append(math_text(x + 28, yy, formula))
        yy += 45
    if note:
        parts.append(text(x + 28, y + h - 28, note, 16, "#667085", 550))
    return parts


def render_svg() -> str:
    width = 1900
    height = 1390
    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        rect(0, 0, width, height, "#f8fafc", "none", 0),
        text(80, 82, "Mathematical Formulas for IMU Fall Preprocessing", 42, "#111827", 850),
        text(80, 122, "Vector notation for SVM, gyroscope magnitude, temporal windowing, scaling, and fall decision", 22, "#667085", 500),
        text(1820, 82, "25 Hz, 50 samples", 20, "#667085", 800, "end"),
        text(1820, 114, "T = 2.0 s", 20, "#2563eb", 850, "end"),
    ]

    parts += card(
        80,
        175,
        560,
        255,
        "1. IMU Sample Vector",
        [
            "a_t = [a_{x,t}, a_{y,t}, a_{z,t}]^T",
            "ω_t = [ω_{x,t}, ω_{y,t}, ω_{z,t}]^T",
            "r_t = [roll_t, pitch_t, yaw_t]^T",
        ],
        "a_t uses g units, ω_t uses deg/s.",
        "#eff6ff",
    )
    parts += card(
        670,
        175,
        560,
        255,
        "2. Vector Magnitudes",
        [
            "SVM_t = ||a_t||_2 = √(a_{x,t}² + a_{y,t}² + a_{z,t}²)",
            "G_t = ||ω_t||_2 = √(ω_{x,t}² + ω_{y,t}² + ω_{z,t}²)",
            "accel_norm_t ≡ SVM_t,   gyro_norm_t ≡ G_t",
        ],
        "These two norms are the core derived IMU features.",
        "#ecfdf3",
    )
    parts += card(
        1260,
        175,
        560,
        255,
        "3. Time Interval",
        [
            "Δt_t = clip(t_t − t_{t−1}, 0, 1000) / 1000",
            "Δt_t ≈ 0.04 s",
            "t = 1, 2, …, N",
        ],
        "The first sample in each group uses Δt_0 = 0.",
        "#fff7ed",
    )

    parts += card(
        80,
        480,
        560,
        300,
        "4. Window Tensor",
        [
            "x_t = [r_t, a_t, ω_t, SVM_t, G_t, Δt_t]",
            "X_k = [x_{k−49}, …, x_k] ∈ R^{50×12}",
            "y_k = max(label_{k−49}, …, label_k)",
            "label_t = 1{event_t = fall}",
        ],
        "A window is positive if any sample in the window is fall.",
        "#f5f3ff",
    )
    parts += card(
        670,
        480,
        560,
        300,
        "5. Robust Scaling",
        [
            "c_j = median(x_{:,j})",
            "s_j = Q_{0.75}(x_{:,j}) − Q_{0.25}(x_{:,j})",
            "z_{t,j} = clip((x_{t,j} − c_j) / s_j, −12, 12)",
        ],
        "c_j and s_j are fitted only on the training split.",
        "#eef2ff",
    )
    parts += card(
        1260,
        480,
        560,
        300,
        "6. BiLSTM Fall Probability",
        [
            "h_k = BiLSTM(Z_k)",
            "α_t = softmax(W_a h_t)",
            "p_k = σ(W_o Σ_t α_t h_t + b_o)",
            "p_k ∈ [0, 1]",
        ],
        "p_k is the LSTM fall probability for one 2-second window.",
        "#fef2f2",
    )

    parts += card(
        80,
        830,
        850,
        380,
        "7. Physical Context Scores",
        [
            "I_k = max_{t∈W_k} SVM_t",
            "F_k = min_{t∈W_k} SVM_t",
            "R_k = max_{t∈W_k} G_t",
            "Θ_k = max_{t∈W_k} √((roll_t−roll_0)² + (pitch_t−pitch_0)²)",
            "post_accel_std = std(SVM_t | t > argmax SVM)",
            "post_gyro_mean = mean(G_t | t > argmax SVM)",
        ],
        "These variables describe impact, free-fall, rotation, posture change, and inactivity.",
        "#ffffff",
    )
    parts += card(
        970,
        830,
        850,
        380,
        "8. Final Decision Rule",
        [
            "A_k = 0.40·clip((I_k−2.5)/2.5,0,1)",
            "    + 0.20·max(clip(R_k/250,0,1), clip(Θ_k/45,0,1))",
            "    + 0.25·1{post_accel_std≤0.35 ∧ post_gyro_mean≤80}",
            "    + 0.15·max(1{F_k≤0.6}, clip(Θ_k/45,0,1))",
            "score_k = 1.0·p_k + 0.0·A_k",
            "fall_detected_k = 1{score_k ≥ 0.35}",
        ],
        "Current tuned checkpoint uses LSTM-only final score; A_k is auxiliary context.",
        "#ffffff",
    )

    parts += [
        rect(80, 1260, 1740, 70, "#111827", "#111827"),
        text(120, 1303, "Model input", 22, "#ffffff", 850),
        math_text(300, 1304, "Z_k ∈ R^{50×12}  →  BiLSTM + Attention  →  p_k  →  fall_detected_k", 24, "#e5e7eb", 700),
        "</svg>",
    ]
    return "\n".join(parts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--svg-output", type=Path, default=Path("assets/imu_fall_preprocessing_math_formulas.svg"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.svg_output.parent.mkdir(parents=True, exist_ok=True)
    args.svg_output.write_text(render_svg(), encoding="utf-8")
    print(json.dumps({"svg_output": str(args.svg_output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
