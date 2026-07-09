#!/usr/bin/env python3
"""Generate an SVG explaining IMU performance improvement causes."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets" / "imu_performance_improvement_cause.svg"


COLORS = {
    "bg": "#F8FAFC",
    "panel": "#FFFFFF",
    "ink": "#172033",
    "muted": "#667085",
    "line": "#D8DEE9",
    "blue": "#3B82F6",
    "green": "#10B981",
    "violet": "#8B5CF6",
    "amber": "#F59E0B",
    "rose": "#F43F5E",
    "cyan": "#06B6D4",
    "soft_blue": "#DBEAFE",
    "soft_green": "#D1FAE5",
    "soft_violet": "#EDE9FE",
    "soft_amber": "#FEF3C7",
    "soft_rose": "#FFE4E6",
}


def text(x: int, y: int, value: str, size: int = 18, weight: int = 400, color: str = "ink", anchor: str = "start") -> str:
    return (
        f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" '
        f'fill="{COLORS.get(color, color)}" text-anchor="{anchor}">{value}</text>'
    )


def rect(x: int, y: int, w: int, h: int, fill: str = "panel", stroke: str = "line", rx: int = 10) -> str:
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
        f'fill="{COLORS.get(fill, fill)}" stroke="{COLORS.get(stroke, stroke)}"/>'
    )


def line(x1: int, y1: int, x2: int, y2: int, color: str = "line", width: int = 2) -> str:
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{COLORS[color]}" stroke-width="{width}"/>'


def arrow(x1: int, y1: int, x2: int, y2: int, color: str = "muted") -> str:
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{COLORS[color]}" stroke-width="2.5" marker-end="url(#arrow)"/>'


def wrap_text(x: int, y: int, value: str, width: int, size: int = 16, line_height: int = 24, color: str = "muted") -> str:
    words = value.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        if len(candidate) > width:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return "\n".join(text(x, y + i * line_height, ln, size=size, color=color) for i, ln in enumerate(lines))


def metric_bar(x: int, y: int, label: str, before: float, after: float, color: str) -> str:
    max_w = 245
    b_w = int(max_w * before)
    a_w = int(max_w * after)
    delta = after - before
    return "\n".join(
        [
            text(x, y, label, 17, 800),
            text(x + 128, y, f"{before:.4f}", 14, 700, "muted"),
            rect(x + 200, y - 16, max_w, 16, "#EEF2F7", "#EEF2F7", rx=5),
            rect(x + 200, y - 16, b_w, 16, "#CBD5E1", "#CBD5E1", rx=5),
            text(x + 128, y + 29, f"{after:.4f}", 14, 700, color),
            rect(x + 200, y + 13, max_w, 16, "#EEF2F7", "#EEF2F7", rx=5),
            rect(x + 200, y + 13, a_w, 16, color, color, rx=5),
            text(x + 468, y + 15, f"{delta:+.4f}", 15, 800, "green" if delta >= 0 else "rose"),
        ]
    )


def feature_card(x: int, y: int, title: str, formula: str, desc: str, color: str) -> str:
    return "\n".join(
        [
            rect(x, y, 340, 150),
            rect(x, y, 8, 150, color, color, rx=4),
            text(x + 28, y + 40, title, 20, 800),
            text(x + 28, y + 78, formula, 18, 800, color),
            wrap_text(x + 28, y + 112, desc, 32, 15, 20),
        ]
    )


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="1050" viewBox="0 0 1600 1050">
  <defs>
    <marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth">
      <path d="M0,0 L0,6 L9,3 z" fill="{COLORS["muted"]}"/>
    </marker>
    <filter id="shadow" x="-20%" y="-20%" width="140%" height="140%">
      <feDropShadow dx="0" dy="8" stdDeviation="10" flood-color="#0F172A" flood-opacity="0.08"/>
    </filter>
  </defs>
  <rect width="1600" height="1050" fill="{COLORS["bg"]}"/>
  <g font-family="Apple SD Gothic Neo, Noto Sans KR, Pretendard, Arial, sans-serif">
    {text(70, 76, "IMU Fall Detection: Performance Improvement Cause Analysis", 34, 850)}
    {text(70, 112, "데이터 표현 방식 변화 → 물리 기반 feature 추가 → 시간 문맥 학습 → Recall/F1 중심 평가", 18, 500, "muted")}
    {line(70, 142, 1530, 142)}

    <g filter="url(#shadow)">{rect(70, 175, 690, 265)}</g>
    {text(105, 225, "1. Data Shape Change", 25, 850)}
    {text(105, 265, "전처리 전", 19, 800, "muted")}
    {text(315, 265, "전처리 후", 19, 800, "blue")}
    {line(105, 285, 715, 285)}
    {text(105, 325, "Rows", 17, 700)}
    {text(315, 325, "349,560", 17, 800, "muted")}
    {text(515, 325, "349,560", 17, 800, "blue")}
    {text(105, 365, "Columns", 17, 700)}
    {text(315, 365, "20", 17, 800, "muted")}
    {text(515, 365, "22", 17, 800, "blue")}
    {text(105, 405, "Model-ready features", 17, 700)}
    {text(315, 405, "9 / 12", 17, 800, "muted")}
    {text(515, 405, "12 / 12", 17, 800, "blue")}
    {wrap_text(810, 230, "핵심: row 수는 그대로이고, 데이터 양을 임의로 늘린 것이 아니다. 변화의 본질은 원본 IMU 값을 모델이 해석하기 쉬운 12차원 feature로 확장한 것이다.", 58, 19, 30, "ink")}

    {arrow(780, 305, 925, 305)}
    <g filter="url(#shadow)">{rect(960, 175, 570, 265)}</g>
    {text(1000, 225, "2. Feature Engineering", 25, 850)}
    {text(1000, 272, "Original 9 features", 18, 800, "muted")}
    {text(1000, 307, "roll pitch yaw / ax ay az / wx wy wz", 18, 700)}
    {text(1000, 357, "+ 3 physical features", 18, 800, "blue")}
    {text(1000, 392, "accel_norm, gyro_norm, dt_s", 20, 850, "blue")}

    {feature_card(70, 485, "accel_norm", "sqrt(ax²+ay²+az²)", "센서 방향과 무관한 전체 충격 크기", "blue")}
    {feature_card(430, 485, "gyro_norm", "sqrt(wx²+wy²+wz²)", "낙상 순간의 전체 회전량", "green")}
    {feature_card(790, 485, "dt_s", "(tᵢ - tᵢ₋₁) / 1000", "ICCAS 25Hz와 SisFall 50Hz 차이 보정", "amber")}
    {feature_card(1150, 485, "Robust Scaling", "(x - median) / IQR", "이상치와 데이터셋 분포 차이에 강함", "violet")}

    <g filter="url(#shadow)">{rect(70, 685, 705, 285)}</g>
    {text(105, 735, "3. Before vs After Metrics", 25, 850)}
    {text(255, 774, "Before", 14, 800, "muted")}
    {text(255, 803, "After", 14, 800, "blue")}
    {metric_bar(105, 835, "RNN F1", 0.6022, 0.6584, "blue")}
    {metric_bar(105, 905, "Transformer F1", 0.6663, 0.7273, "violet")}
    {text(105, 955, "Recall improved in 3 of 4 models. F1 improved in 2 of 4 models.", 16, 800, "ink")}

    <g filter="url(#shadow)">{rect(825, 685, 705, 285)}</g>
    {text(860, 735, "4. Final LSTM Interpretation", 25, 850)}
    {text(860, 785, "Input shape", 17, 700, "muted")}
    {text(1040, 785, "50 timesteps × 12 features", 20, 850, "blue")}
    {text(860, 830, "Final metrics", 17, 700, "muted")}
    {text(1040, 830, "Accuracy 0.8799 / Recall 0.8863 / F1 0.8677", 20, 850, "green")}
    {text(860, 875, "Threshold", 17, 700, "muted")}
    {text(1040, 875, "0.35", 20, 850, "amber")}
    {wrap_text(860, 925, "정확한 결론: 전처리가 모든 모델을 무조건 향상시킨 것은 아니다. 그러나 물리 기반 feature와 50-step window를 통해 낙상 전후의 충격, 회전, 시간 간격을 LSTM이 학습할 수 있게 만들었다.", 64, 17, 25, "ink")}

    {text(70, 1015, "Final claim: 성능 개선의 핵심은 단순 모델 변경이 아니라 데이터 표현 방식의 개선이다.", 20, 850, "rose")}
  </g>
</svg>
'''
    OUT.write_text(svg, encoding="utf-8")
    print(OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
