"""Generate a clean paper-style SVG for IMU preprocessing equations."""

from __future__ import annotations

import html
from pathlib import Path


WIDTH = 1800
HEIGHT = 1320
OUTPUT = Path("assets/imu_preprocessing_paper_formulas.svg")


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def text(
    x: float,
    y: float,
    value: object,
    size: int = 22,
    color: str = "#1f2937",
    weight: int = 500,
    anchor: str = "start",
    family: str = "-apple-system,BlinkMacSystemFont,Segoe UI,Noto Sans KR,sans-serif",
) -> str:
    return (
        f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" '
        f'font-weight="{weight}" text-anchor="{anchor}" font-family="{family}">{esc(value)}</text>'
    )


def math_text(x: float, y: float, value: object, size: int = 26, color: str = "#111827", anchor: str = "start") -> str:
    return text(x, y, value, size, color, 520, anchor, "Times New Roman, STIX Two Text, Cambria Math, serif")


def rect(x: float, y: float, w: float, h: float, fill: str, stroke: str = "#d7dce7", rx: float = 0) -> str:
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}"/>'


def eq_block(x: float, y: float, w: float, h: float, title: str, lines: list[str], eq_no: str, note: str = "") -> list[str]:
    parts = [
        rect(x, y, w, h, "#ffffff", "#d9dee8", 6),
        text(x + 28, y + 42, title, 22, "#111827", 800),
    ]
    yy = y + 88
    for line in lines:
        parts.append(math_text(x + 36, yy, line, 26))
        yy += 43
    parts.append(math_text(x + w - 46, y + h - 35, f"({eq_no})", 25, "#374151", "end"))
    if note:
        parts.append(text(x + 28, y + h - 22, note, 14, "#6b7280", 500))
    return parts


def render_svg() -> str:
    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">',
        rect(0, 0, WIDTH, HEIGHT, "#f8fafc", "none"),
        text(80, 76, "IMU Fall Detection Preprocessing Equations", 42, "#111827", 850),
        text(80, 118, "Paper-style mathematical notation for 25 Hz IMU/Gyro fall detection", 21, "#667085", 500),
        text(1720, 76, "L = 50 samples", 20, "#475467", 750, "end"),
        text(1720, 108, "T ≈ 2.0 s", 20, "#2563eb", 850, "end"),
    ]

    parts += eq_block(
        80,
        165,
        520,
        190,
        "Sample Vectors",
        [
            "r_t = [φ_t, θ_t, ψ_t]^T",
            "a_t = [a_{x,t}, a_{y,t}, a_{z,t}]^T",
            "ω_t = [ω_{x,t}, ω_{y,t}, ω_{z,t}]^T",
        ],
        "1",
        "r: attitude, a: acceleration, ω: angular velocity",
    )
    parts += eq_block(
        640,
        165,
        520,
        190,
        "Magnitude Features",
        [
            "s_t = ||a_t||_2 = √(a_x,t² + a_y,t² + a_z,t²)",
            "g_t = ||ω_t||_2 = √(ω_x,t² + ω_y,t² + ω_z,t²)",
        ],
        "2",
        "s_t is SVM / accel_norm, g_t is gyro_norm",
    )
    parts += eq_block(
        1200,
        165,
        520,
        190,
        "Temporal Feature",
        [
            "Δt_t = clip(m_t − m_{t−1}, 0, 1000) / 1000",
            "Δt_0 = 0,     Δt_t ≈ 0.04 s",
        ],
        "3",
        "m_t is timestamp in milliseconds",
    )

    parts += eq_block(
        80,
        400,
        800,
        220,
        "LSTM Feature Vector",
        [
            "x_t = [ r_t, a_t, ω_t, s_t, g_t, Δt_t ]^T ∈ R^{12}",
            "X_k = [x_{k−L+1}, ..., x_k]^T ∈ R^{50×12}",
        ],
        "4",
        "The sequence window contains 50 consecutive IMU samples.",
    )
    parts += eq_block(
        920,
        400,
        800,
        220,
        "Sequence Label",
        [
            "y_t = 1{label_t = fall}",
            "Y_k = max_{i∈{k−L+1,...,k}} y_i",
        ],
        "5",
        "A window is positive if any time step in the window is fall.",
    )

    parts += eq_block(
        80,
        665,
        800,
        260,
        "Robust Scaling",
        [
            "c_j = median(x_:,j)",
            "q_j = Q_0.75(x_:,j) − Q_0.25(x_:,j)",
            "σ_j = q_j if q_j>ε, else std(x_:,j), else 1",
            "z_t,j = clip((x_t,j − c_j) / σ_j, −12, 12)",
        ],
        "6",
        "c_j and σ_j are estimated only from the training split.",
    )
    parts += eq_block(
        920,
        665,
        800,
        260,
        "Physical Context Features",
        [
            "I_k = max_{t∈W_k} s_t,     F_k = min_{t∈W_k} s_t",
            "R_k = max_{t∈W_k} g_t",
            "Θ_k = max √((φ_t−φ_0)² + (θ_t−θ_0)²)",
            "t* = argmax_{t∈W_k} s_t",
        ],
        "7",
        "Impact, free-fall, rotation, posture change, and impact index.",
    )

    parts += eq_block(
        80,
        970,
        800,
        190,
        "Post-impact Inactivity",
        [
            "P_k = {t | t* < t ≤ t* + 25}",
            "A_k^std = std(s_t | t∈P_k)",
            "G_k^mean = mean(g_t | t∈P_k)",
        ],
        "8",
        "Used to distinguish fall impact from ordinary motion.",
    )
    parts += eq_block(
        920,
        970,
        800,
        190,
        "Model Input and Decision",
        [
            "Z_k ∈ R^{50×12}  →  BiLSTM + Attention",
            "p_k = σ(W_o h_k + b_o)",
            "fall_k = 1{p_k ≥ τ},     τ = 0.35",
        ],
        "9",
        "The tuned final checkpoint uses the LSTM probability as primary score.",
    )

    parts += [
        rect(80, 1210, 1640, 56, "#111827", "#111827", 4),
        text(120, 1246, "Summary", 20, "#ffffff", 850),
        math_text(260, 1247, "raw IMU → {s_t, g_t, Δt_t} → X_k ∈ R^{50×12} → robust scaling → Z_k → BiLSTM fall probability", 24, "#e5e7eb"),
        "</svg>",
    ]
    return "\n".join(parts)


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(render_svg(), encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
