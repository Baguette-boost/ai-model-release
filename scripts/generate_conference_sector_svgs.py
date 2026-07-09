#!/usr/bin/env python3
"""Generate conference-ready SVG panels for the final IMU fall project."""

from __future__ import annotations

import csv
import html
import json
from collections import Counter
from pathlib import Path
from textwrap import wrap


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
ASSET_DIR = ROOT / "assets" / "conference_sectors"
DOC_DIR = ROOT / "docs"
FINAL_MODEL = ROOT / "models" / "iccas_final_hybrid_lstm_imu_fall.json"
ICCAS_MODEL = ROOT / "models" / "iccas_final_lstm_imu_fall.json"
SPEED_JSON = PROJECT / "data" / "iccas_sensor_lstm" / "imu_lstm_speed_preprocessing_metrics.json"
PREPROCESS_EFFECT_MD = ROOT / "docs" / "IMU_PREPROCESSING_EFFECT_COMPARISON.md"
DATA_CSV = PROJECT / "data" / "iccas_sensor_lstm" / "imu_fall_preprocessed.csv"


PALETTE = {
    "ink": "#172033",
    "muted": "#667085",
    "line": "#D8DEE9",
    "bg": "#F8FAFC",
    "panel": "#FFFFFF",
    "blue": "#3B82F6",
    "green": "#10B981",
    "violet": "#8B5CF6",
    "amber": "#F59E0B",
    "rose": "#F43F5E",
    "cyan": "#06B6D4",
}


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def text(x: float, y: float, value: object, size: int = 18, weight: int = 400,
         color: str = "ink", anchor: str = "start") -> str:
    return (
        f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" '
        f'fill="{PALETTE.get(color, color)}" text-anchor="{anchor}">{esc(value)}</text>'
    )


def multi_text(x: float, y: float, value: str, width: int = 48, size: int = 16,
               line_height: int = 24, color: str = "muted", weight: int = 400) -> str:
    lines = []
    for i, line in enumerate(wrap(value, width=width, break_long_words=False)):
        lines.append(text(x, y + i * line_height, line, size=size, weight=weight, color=color))
    return "\n".join(lines)


def rect(x: float, y: float, w: float, h: float, fill: str = "panel",
         stroke: str = "line", rx: int = 8, opacity: float = 1.0) -> str:
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
        f'fill="{PALETTE.get(fill, fill)}" stroke="{PALETTE.get(stroke, stroke)}" '
        f'opacity="{opacity}"/>'
    )


def line(x1: float, y1: float, x2: float, y2: float, color: str = "line", width: float = 2) -> str:
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{PALETTE.get(color, color)}" stroke-width="{width}"/>'


def arrow(x1: float, y1: float, x2: float, y2: float, color: str = "muted") -> str:
    return (
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{PALETTE[color]}" stroke-width="2" marker-end="url(#arrow)"/>'
    )


def svg_page(title: str, subtitle: str, body: str, height: int = 900) -> str:
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="1400" height="{height}" viewBox="0 0 1400 {height}">
  <defs>
    <marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth">
      <path d="M0,0 L0,6 L9,3 z" fill="{PALETTE["muted"]}"/>
    </marker>
    <filter id="shadow" x="-20%" y="-20%" width="140%" height="140%">
      <feDropShadow dx="0" dy="8" stdDeviation="10" flood-color="#0F172A" flood-opacity="0.08"/>
    </filter>
  </defs>
  <rect width="1400" height="{height}" fill="{PALETTE["bg"]}"/>
  <g font-family="Apple SD Gothic Neo, Noto Sans KR, Pretendard, Arial, sans-serif">
    {text(64, 72, title, 34, 800)}
    {text(64, 108, subtitle, 18, 400, "muted")}
    {line(64, 136, 1336, 136)}
    {body}
  </g>
</svg>
'''


def progress_bar(x: int, y: int, w: int, h: int, value: float, color: str, label: str) -> str:
    filled = max(0, min(w, w * value))
    return "\n".join([
        text(x, y - 10, label, 15, 600, "ink"),
        rect(x, y, w, h, "#EEF2F7", "#EEF2F7", rx=6),
        rect(x, y, filled, h, color, color, rx=6),
        text(x + w + 14, y + h - 3, f"{value:.4f}", 15, 700, "ink"),
    ])


def metric_card(x: int, y: int, title: str, value: str, caption: str, color: str) -> str:
    return "\n".join([
        f'<g filter="url(#shadow)">',
        rect(x, y, 290, 150),
        f'</g>',
        rect(x, y, 8, 150, color, color, rx=4),
        text(x + 28, y + 42, title, 18, 700, "ink"),
        text(x + 28, y + 88, value, 32, 800, color),
        multi_text(x + 28, y + 120, caption, width=26, size=14, line_height=18),
    ])


def count_dataset_rows(limit: int | None = None) -> dict:
    source_counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    fall_counts: Counter[str] = Counter()
    rows = 0
    with DATA_CSV.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows += 1
            source_counts[row.get("source_dataset", "unknown")] += 1
            label_counts[row.get("label", "unknown")] += 1
            fall_counts["fall" if row.get("fall_target") == "1" else "non-fall"] += 1
            if limit and rows >= limit:
                break
    return {
        "rows": rows,
        "source_counts": dict(source_counts),
        "label_counts": dict(label_counts),
        "fall_counts": dict(fall_counts),
    }


def sector_01_data(data_counts: dict) -> str:
    rows = data_counts["rows"]
    source = data_counts["source_counts"]
    labels = data_counts["label_counts"]
    fall = data_counts["fall_counts"]
    total = sum(source.values())
    iccas = source.get("ICCAS", 0)
    sisfall = source.get("SisFall", 0)
    bars = []
    for i, (name, count, color) in enumerate([
        ("ICCAS direct", iccas, "blue"),
        ("SisFall open", sisfall, "green"),
        ("Fall target", fall.get("fall", 0), "rose"),
        ("Non-fall target", fall.get("non-fall", 0), "cyan"),
    ]):
        bars.append(progress_bar(760, 235 + i * 82, 420, 26, count / total, color, f"{name}: {count:,} rows"))
    label_items = sorted(labels.items(), key=lambda kv: kv[1], reverse=True)[:6]
    table = [text(86, 245, "Top labels", 22, 800)]
    for i, (label, count) in enumerate(label_items):
        table.append(text(96, 292 + i * 46, label, 18, 600, "ink"))
        table.append(text(370, 292 + i * 46, f"{count:,}", 18, 700, "blue", "end"))
        table.append(line(96, 308 + i * 46, 382, 308 + i * 46))
    body = "\n".join([
        metric_card(82, 170, "Total IMU Rows", f"{rows:,}", "전처리 완료 CSV 기준", "blue"),
        metric_card(402, 170, "Data Sources", "2", "ICCAS 직접 취득 + SisFall 공개 데이터", "green"),
        rect(82, 380, 610, 420),
        "\n".join(table),
        rect(720, 170, 600, 430),
        text(760, 215, "Source / target composition", 22, 800),
        "\n".join(bars),
        rect(720, 635, 600, 165),
        text(760, 682, "학술적 해석", 22, 800),
        multi_text(760, 724, "ICCAS는 실제 장비 환경의 normal baseline을 제공하고, SisFall은 다양한 낙상 positive pattern을 제공한다. 두 데이터를 병합해 낙상 패턴과 실제 환경 오탐 요인을 함께 학습했다.", width=62, size=18, line_height=28),
    ])
    return svg_page("Sector 01. IMU Dataset Construction", "직접 취득 ICCAS 데이터와 공개 SisFall 데이터를 병합한 최종 IMU 낙상 데이터", body)


def sector_02_distribution() -> str:
    rows = [
        ("normal accel p50", "1.0029g", "1.0096g", "0.0067g"),
        ("fall accel p50", "1.0041g", "1.0010g", "0.0031g"),
        ("normal accel p95", "1.4684g", "1.7078g", "tail 차이"),
        ("fall accel p95", "1.2840g", "1.3671g", "tail 차이"),
        ("fall gyro p95", "177.9467", "116.6851", "도메인 차이"),
        ("dt_s p50", "0.0400s", "0.0200s", "sampling 차이"),
    ]
    table = [
        rect(78, 180, 1245, 455),
        text(112, 232, "Numeric compatibility check", 24, 800),
        text(112, 286, "Feature", 17, 800, "muted"),
        text(508, 286, "ICCAS", 17, 800, "muted"),
        text(748, 286, "SisFall", 17, 800, "muted"),
        text(998, 286, "Interpretation", 17, 800, "muted"),
        line(112, 306, 1270, 306),
    ]
    for i, row in enumerate(rows):
        y = 350 + i * 50
        table.extend([
            text(112, y, row[0], 18, 600),
            text(508, y, row[1], 18, 700, "blue"),
            text(748, y, row[2], 18, 700, "green"),
            text(998, y, row[3], 18, 600, "ink"),
            line(112, y + 18, 1270, y + 18),
        ])
    body = "\n".join([
        "\n".join(table),
        rect(78, 675, 390, 130),
        text(112, 720, "결론 1", 20, 800, "blue"),
        multi_text(112, 758, "가속도 중심값은 두 데이터 모두 1g 근처로 유사하다.", width=34, size=17),
        rect(505, 675, 390, 130),
        text(539, 720, "결론 2", 20, 800, "amber"),
        multi_text(539, 758, "자이로와 tail 분포는 차이가 있어 robust scaling이 필요하다.", width=34, size=17),
        rect(930, 675, 390, 130),
        text(964, 720, "결론 3", 20, 800, "green"),
        multi_text(964, 758, "샘플링 간격 차이를 dt_s feature로 모델 입력에 포함했다.", width=34, size=17),
    ])
    return svg_page("Sector 02. ICCAS vs SisFall Distribution", "오픈 데이터와 직접 취득 데이터의 수치적 호환성 및 차이 분석", body)


def sector_03_preprocessing() -> str:
    formulas = [
        ("Acceleration magnitude", "a_norm = sqrt(ax^2 + ay^2 + az^2)", "전체 충격 크기"),
        ("Gyroscope magnitude", "g_norm = sqrt(wx^2 + wy^2 + wz^2)", "전체 회전량"),
        ("Time interval", "dt_s = (t_i - t_{i-1}) / 1000", "샘플링 차이 보정"),
        ("Robust scaling", "x' = (x - median) / IQR", "이상치에 강한 정규화"),
        ("Windowing", "X_k in R^(50 x 12)", "2초 시계열 패턴"),
    ]
    cards = []
    for i, (name, eq, desc) in enumerate(formulas):
        y = 190 + i * 116
        cards.extend([
            rect(90, y, 560, 84),
            text(124, y + 32, name, 18, 800, "ink"),
            text(124, y + 62, eq, 18, 700, "violet"),
            text(720, y + 44, desc, 22, 800, ["blue", "green", "amber", "rose", "cyan"][i]),
            arrow(660, y + 42, 700, y + 42),
        ])
    body = "\n".join([
        rect(78, 165, 1244, 640),
        text(112, 220, "Input features after preprocessing", 24, 800),
        "\n".join(cards),
        text(720, 720, "핵심 어필", 24, 800),
        multi_text(720, 764, "단순 센서값을 그대로 학습한 것이 아니라, 낙상에서 중요한 충격 크기, 회전량, 시간 간격을 물리 기반 feature로 구성하여 LSTM이 시간적 패턴을 학습하도록 만들었다.", width=54, size=18, line_height=28),
    ])
    return svg_page("Sector 03. IMU Preprocessing", "원본 9개 IMU feature에 물리 기반 feature 3개를 추가한 12차원 입력 구성", body)


def sector_04_training(model: dict) -> str:
    hist = model["history"]
    points = []
    chart_x, chart_y, chart_w, chart_h = 112, 260, 760, 300
    max_loss = max(h["loss"] for h in hist)
    min_loss = min(h["loss"] for h in hist)
    for h in hist:
        px = chart_x + (h["epoch"] - 1) / (len(hist) - 1) * chart_w
        py = chart_y + chart_h - (h["loss"] - min_loss) / (max_loss - min_loss) * chart_h
        points.append((px, py))
    poly = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    f1_points = []
    for h in hist:
        px = chart_x + (h["epoch"] - 1) / (len(hist) - 1) * chart_w
        py = chart_y + chart_h - h["validation_f1"] * chart_h
        f1_points.append((px, py))
    f1_poly = " ".join(f"{x:.1f},{y:.1f}" for x, y in f1_points)
    body = "\n".join([
        metric_card(930, 180, "Sequence", "50 x 12", "50 timesteps, 12 IMU features", "blue"),
        metric_card(930, 360, "Model", "Bi-LSTM", "hidden 64, layers 2, attention pooling", "violet"),
        metric_card(930, 540, "Training", "15 epochs", "best validation F1 observed at epoch 10", "green"),
        rect(78, 180, 840, 500),
        text(112, 232, "Training curve", 24, 800),
        line(chart_x, chart_y, chart_x, chart_y + chart_h),
        line(chart_x, chart_y + chart_h, chart_x + chart_w, chart_y + chart_h),
        f'<polyline points="{poly}" fill="none" stroke="{PALETTE["rose"]}" stroke-width="4"/>',
        f'<polyline points="{f1_poly}" fill="none" stroke="{PALETTE["blue"]}" stroke-width="4"/>',
        text(140, 620, "Loss", 18, 800, "rose"),
        text(230, 620, "Validation F1", 18, 800, "blue"),
        text(112, 714, "Training method", 22, 800),
        multi_text(112, 756, "Group/chronological split로 train, validation, test를 분리하고, 각 window 안에 fall sample이 하나라도 있으면 positive sequence로 라벨링했다.", width=82, size=18, line_height=28),
    ])
    return svg_page("Sector 04. LSTM Training Method", "50-step sliding window 기반 IMU 낙상 시계열 학습", body)


def bar_group(x: int, y: int, title: str, metrics: list[tuple[str, float, str]], max_w: int = 420) -> str:
    parts = [text(x, y, title, 22, 800)]
    for i, (name, value, color) in enumerate(metrics):
        parts.append(progress_bar(x, y + 52 + i * 66, max_w, 24, value, color, name))
    return "\n".join(parts)


def sector_05_performance(model: dict, speed: dict) -> str:
    test = model["test_metrics"]["lstm_only"]
    iccas = model["test_metrics_by_dataset"]["ICCAS"]
    sisfall = model["test_metrics_by_dataset"]["SisFall"]
    metrics = [
        ("Accuracy", test["accuracy"], "blue"),
        ("Precision", test["precision"], "green"),
        ("Recall", test["recall"], "rose"),
        ("F1-score", test["f1"], "violet"),
    ]
    body = "\n".join([
        metric_card(88, 176, "Final F1-score", f'{test["f1"]:.4f}', "낙상 탐지 균형 지표", "violet"),
        metric_card(408, 176, "Recall", f'{test["recall"]:.4f}', "실제 낙상 탐지율", "rose"),
        metric_card(728, 176, "Threshold", f'{test["threshold"]:.2f}', "LSTM probability threshold", "amber"),
        metric_card(1048, 176, "Realtime p95", f'{speed["realtime_tensor_forward_p95_ms"]:.3f} ms', "1 window tensor+forward", "green"),
        rect(78, 370, 600, 390),
        bar_group(112, 426, "Overall test metrics", metrics),
        rect(720, 370, 600, 390),
        text(760, 426, "Dataset-level test F1", 22, 800),
        progress_bar(760, 486, 420, 24, iccas["f1"], "blue", f'ICCAS F1: {iccas["f1"]:.4f}'),
        progress_bar(760, 552, 420, 24, sisfall["f1"], "green", f'SisFall F1: {sisfall["f1"]:.4f}'),
        text(760, 650, "Confusion matrix", 20, 800),
        text(760, 694, f'TP {test["tp"]:,}   FP {test["fp"]:,}   TN {test["tn"]:,}   FN {test["fn"]:,}', 20, 700, "ink"),
    ])
    return svg_page("Sector 05. Final IMU Fall Performance", "최종 적용 모델: 전처리 feature 기반 LSTM 낙상 감지", body)


def sector_06_preprocess_effect() -> str:
    rows = [
        ("RNN", 0.6022, 0.6584, 0.6935, 0.8333, "blue"),
        ("GRU", 0.6898, 0.6878, 0.8695, 0.8590, "green"),
        ("LSTM", 0.6882, 0.6746, 0.8054, 0.8228, "violet"),
        ("Transformer", 0.6663, 0.7273, 0.7762, 0.7832, "amber"),
    ]
    parts = [rect(78, 170, 1245, 610), text(112, 228, "Effect of preprocessing features", 24, 800)]
    for i, (name, f1_b, f1_a, r_b, r_a, color) in enumerate(rows):
        y = 300 + i * 105
        parts.append(text(112, y, name, 20, 800))
        parts.append(progress_bar(285, y - 20, 260, 22, f1_b, "#D7DCE7", f"F1 before {f1_b:.4f}"))
        parts.append(progress_bar(620, y - 20, 260, 22, f1_a, color, f"F1 after {f1_a:.4f}"))
        delta = f1_a - f1_b
        parts.append(text(990, y, f"Delta {delta:+.4f}", 19, 800, "green" if delta >= 0 else "rose"))
        parts.append(text(112, y + 48, f"Recall: {r_b:.4f} -> {r_a:.4f}", 16, 700, "muted"))
    parts.extend([
        text(112, 720, "정확한 해석", 22, 800),
        multi_text(112, 758, "전처리는 모든 모델에서 항상 성능을 올리지는 않는다. 다만 RNN/Transformer의 F1이 상승했고, 4개 중 3개 모델에서 Recall이 개선되어 낙상 미탐 감소 관점의 효과를 확인했다.", width=95, size=18, line_height=28),
    ])
    return svg_page("Sector 06. Preprocessing Effect", "전처리 feature 추가 전/후 모델별 성능 비교", "\n".join(parts))


def sector_07_system(model: dict, speed: dict) -> str:
    stages = [
        ("ESP32 IMU", "25 Hz sampling\\nax ay az wx wy wz\\nroll pitch yaw", "blue"),
        ("Preprocessing", "accel_norm\\ngyro_norm\\ndt_s + scaling", "green"),
        ("LSTM Window", "50 steps = 2.0 s\\n12 features", "violet"),
        ("Inference", f'p95 {speed["realtime_tensor_forward_p95_ms"]:.3f} ms\\nthreshold {model["threshold"]:.2f}', "amber"),
        ("Server Result", "fall / normal\\nrisk level\\nAPI response", "rose"),
    ]
    parts = []
    for i, (name, desc, color) in enumerate(stages):
        x = 85 + i * 255
        parts.append(rect(x, 250, 215, 180))
        parts.append(text(x + 24, 300, name, 21, 800, color))
        for j, ln in enumerate(desc.split("\\n")):
            parts.append(text(x + 24, 344 + j * 28, ln, 17, 600, "ink"))
        if i < len(stages) - 1:
            parts.append(arrow(x + 222, 340, x + 252, 340))
    parts.extend([
        rect(95, 535, 560, 190),
        text(130, 585, "Why LSTM?", 24, 800),
        multi_text(130, 630, "낙상은 단일 peak가 아니라 충격, 회전, 자세 변화, 이후 정지 상태가 순서대로 나타나는 시계열 이벤트다. 따라서 window 기반 LSTM이 threshold 단독 방식보다 설명력이 높다.", width=52, size=18, line_height=28),
        rect(720, 535, 585, 190),
        text(755, 585, "Final scope", 24, 800),
        multi_text(755, 630, "최종 학술대회 범위는 GPS를 제외한 IMU 낙상 감지다. 표현은 '하이브리드'가 아니라 '전처리 feature 기반 LSTM 낙상 감지 모델'이 정확하다.", width=52, size=18, line_height=28),
    ])
    return svg_page("Sector 07. Realtime Inference Architecture", "실시간 IMU 입력부터 서버 결과 출력까지의 최종 시스템 구조", "\n".join(parts))


def sector_08_professor_feedback() -> str:
    items = [
        ("Scope", "GPS 제외, IMU 낙상 감지로 범위를 명확히 축소", "blue"),
        ("Model naming", "Hybrid 대신 preprocessing-feature LSTM으로 표현", "violet"),
        ("Metrics", "Accuracy보다 Recall/F1 중심으로 해석", "rose"),
        ("Data claim", "ICCAS와 SisFall은 유사점과 차이가 모두 있으므로 과장 금지", "amber"),
        ("Preprocessing", "성능 보장 표현보다 물리 의미 강화로 설명", "green"),
    ]
    parts = [rect(78, 170, 1245, 610)]
    for i, (k, v, color) in enumerate(items):
        y = 235 + i * 95
        parts.append(rect(118, y - 42, 12, 62, color, color, rx=4))
        parts.append(text(150, y, k, 23, 800, color))
        parts.append(text(390, y, v, 22, 700, "ink"))
        parts.append(line(150, y + 28, 1260, y + 28))
    parts.extend([
        text(118, 735, "최종 발표 문장", 24, 800),
        multi_text(118, 778, "본 연구는 IMU 센서 기반 낙상 감지에 집중하고, 직접 취득 데이터와 공개 낙상 데이터를 병합해 50-step LSTM을 학습하였다. 최종 모델은 Recall 0.8863, F1-score 0.8677을 기록했으며, 낙상 미탐을 줄이는 것을 핵심 평가 기준으로 삼았다.", width=105, size=18, line_height=28),
    ])
    return svg_page("Sector 08. Professor-Level Final Feedback", "학술대회 발표에서 과장 없이 방어 가능한 핵심 피드백", "\n".join(parts))


def write_catalog(files: list[Path]) -> None:
    lines = [
        "# Conference SVG Sector Pack",
        "",
        "GPS를 제외한 최종 IMU 낙상 감지 범위에 맞춰 학술대회 발표/포스터용 SVG를 섹터별로 정리했다.",
        "",
        "## 생성 명령",
        "",
        "```bash",
        "cd /Volumes/Hub_1T/ICCAS/ai-model-release",
        "../.venv/bin/python scripts/generate_conference_sector_svgs.py",
        "```",
        "",
        "## SVG 파일",
        "",
    ]
    for path in files:
        lines.append(f"- `{path.relative_to(ROOT)}`")
    lines.extend([
        "",
        "## 최종 표현 원칙",
        "",
        "- 최종 범위는 GPS가 아니라 IMU 낙상 감지다.",
        "- 모델 명칭은 `전처리 feature 기반 LSTM 낙상 감지 모델`로 설명한다.",
        "- 핵심 성능은 Accuracy보다 Recall과 F1-score 중심으로 제시한다.",
        "- 전처리는 성능을 항상 올리는 마법이 아니라 IMU 신호의 물리적 의미를 강화하는 과정으로 설명한다.",
    ])
    (DOC_DIR / "CONFERENCE_SVG_SECTOR_PACK.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    DOC_DIR.mkdir(parents=True, exist_ok=True)
    final_model = load_json(FINAL_MODEL)
    speed = load_json(SPEED_JSON)["inference_speed"]
    data_counts = count_dataset_rows()
    files = [
        ("sector_01_dataset_construction.svg", sector_01_data(data_counts)),
        ("sector_02_iccas_sisfall_distribution.svg", sector_02_distribution()),
        ("sector_03_imu_preprocessing.svg", sector_03_preprocessing()),
        ("sector_04_lstm_training_method.svg", sector_04_training(final_model)),
        ("sector_05_final_performance.svg", sector_05_performance(final_model, speed)),
        ("sector_06_preprocessing_effect.svg", sector_06_preprocess_effect()),
        ("sector_07_realtime_inference_architecture.svg", sector_07_system(final_model, speed)),
        ("sector_08_professor_final_feedback.svg", sector_08_professor_feedback()),
    ]
    written = []
    for filename, content in files:
        out = ASSET_DIR / filename
        out.write_text(content, encoding="utf-8")
        written.append(out)
    write_catalog(written)
    for path in written:
        print(path.relative_to(ROOT))
    print("docs/CONFERENCE_SVG_SECTOR_PACK.md")


if __name__ == "__main__":
    main()
