"""Generate full ICCAS + SisFall dataset distribution docs and SVG."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any

import pandas as pd


PALETTE = ["#5b55d9", "#8177ee", "#a496ee", "#d9cff8", "#2563eb", "#059669", "#d97706", "#dc2626"]


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


def pct(value: int, total: int) -> str:
    return f"{(value / max(total, 1)) * 100:.2f}%"


def derive(source: Path) -> dict[str, Any]:
    frame = pd.read_csv(source, low_memory=False)
    label_counts = {str(k): int(v) for k, v in frame["label"].value_counts(dropna=False).to_dict().items()}
    source_counts = {str(k): int(v) for k, v in frame["source_dataset"].value_counts(dropna=False).to_dict().items()}
    activity_counts = {str(k): int(v) for k, v in frame["source_activity"].value_counts(dropna=False).sort_index().to_dict().items()}
    subject_counts = {str(k): int(v) for k, v in frame["source_subject"].value_counts(dropna=False).sort_index().to_dict().items()}
    trial_counts = {str(k): int(v) for k, v in frame["source_trial"].value_counts(dropna=False).sort_index().to_dict().items()}
    return {
        "source": str(source),
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "source_dataset": source_counts,
        "label": label_counts,
        "binary_fall": {
            "fall": int((frame["label"].astype(str) == "fall").sum()),
            "non_fall": int((frame["label"].astype(str) != "fall").sum()),
        },
        "activity": activity_counts,
        "subject": subject_counts,
        "trial": trial_counts,
    }


def table(title: str, data: dict[str, int], total: int, description: dict[str, str] | None = None) -> list[str]:
    lines = [f"## {title}", "", "| Item | Description | Rows | Ratio |", "| --- | --- | ---: | ---: |"]
    for key, value in data.items():
        desc = description.get(key, "") if description else ""
        lines.append(f"| {key} | {desc} | {value:,} | {pct(value, total)} |")
    lines.append(f"| Total |  | {sum(data.values()):,} | {pct(sum(data.values()), total)} |")
    lines.append("")
    return lines


def render_markdown(report: dict[str, Any]) -> str:
    rows = report["rows"]
    label_desc = {
        "normal": "SisFall D01-D19 normal ADL",
        "fall": "ICCAS fall + SisFall F01-F15 fall",
        "wandering": "ICCAS GPS/IMU wandering scenario",
        "walk": "ICCAS walking scenario",
        "idle": "ICCAS idle scenario",
        "sit": "ICCAS sitting scenario",
    }
    source_desc = {
        "SisFall": "External public IMU fall dataset",
        "ICCAS": "Directly collected project dataset",
    }
    activity_desc = {
        "walk": "ICCAS walk",
        "wandering": "ICCAS wandering",
        "fall": "ICCAS fall",
        "idle": "ICCAS idle",
        "sit": "ICCAS sit",
    }
    for idx in range(1, 20):
        activity_desc[f"D{idx:02d}"] = "SisFall normal ADL"
    for idx in range(1, 16):
        activity_desc[f"F{idx:02d}"] = "SisFall fall"

    lines = [
        "# 전체 데이터셋 분포",
        "",
        "## 기준 파일",
        "",
        "```text",
        report["source"],
        "```",
        "",
        "ICCAS 직접 취득 데이터와 SisFall IMU 낙상 데이터를 병합한 최종 IMU 낙상 학습용 CSV 기준입니다.",
        "",
        "## 전체 요약",
        "",
        "| 항목 | 값 |",
        "| --- | ---: |",
        f"| 전체 row | {rows:,} |",
        f"| 전체 column | {report['columns']:,} |",
        f"| source_dataset 종류 | {len(report['source_dataset'])} |",
        f"| label 종류 | {len(report['label'])} |",
        f"| activity 종류 | {len(report['activity'])} |",
        f"| subject 종류 | {len(report['subject'])} |",
        f"| trial 종류 | {len(report['trial'])} |",
        "",
    ]
    lines += table("Source Dataset 분포", report["source_dataset"], rows, source_desc)
    lines += table("Label 분포", report["label"], rows, label_desc)
    lines += table("Binary Fall 분포", report["binary_fall"], rows, {"fall": "positive", "non_fall": "negative"})
    lines += table("Activity 전체 분포", report["activity"], rows, activity_desc)
    lines += table("Subject 분포", report["subject"], rows)
    lines += table("Trial 분포", report["trial"], rows)
    lines += [
        "## 학습 적용 해석",
        "",
        "- SisFall은 GPS가 없으므로 IMU/Gyro 낙상 탐지 학습에만 사용합니다.",
        "- ICCAS는 walk, wandering, fall, idle, sit 시나리오를 포함합니다.",
        "- LSTM 입력은 `roll`, `pitch`, `yaw`, `ax`, `ay`, `az`, `wx`, `wy`, `wz`, `accel_norm`, `gyro_norm`, `dt_s`의 12개 feature입니다.",
        "- 최종 IMU 낙상 모델은 50 samples, 25 Hz 기준 약 2초 window를 사용합니다.",
        "",
    ]
    return "\n".join(lines)


def horizontal_chart(x: float, y: float, w: float, h: float, title: str, data: dict[str, int], total: int) -> list[str]:
    parts = [rect(x, y, w, h, "#ffffff", "#eee9fb", 10), text(x + 28, y + 46, title, 25, "#272145", 850)]
    max_value = max(data.values()) if data else 1
    row_h = min(38, (h - 88) / max(1, len(data)))
    start_y = y + 76
    for i, (name, value) in enumerate(data.items()):
        yy = start_y + i * row_h
        bar_x = x + 190
        bar_w = w - 390
        fill_w = bar_w * value / max_value
        color = PALETTE[i % len(PALETTE)]
        parts += [
            text(x + 28, yy + 18, name, 15, "#514872", 760),
            rect(bar_x, yy, bar_w, 21, "#f2effb", "none", 7),
            rect(bar_x, yy, fill_w, 21, color, "none", 7),
            text(x + w - 28, yy + 17, f"{value:,} ({pct(value, total)})", 14, "#6f6791", 650, "end"),
        ]
    return parts


def compact_activity_chart(x: float, y: float, w: float, h: float, data: dict[str, int], total: int) -> list[str]:
    parts = [rect(x, y, w, h, "#ffffff", "#eee9fb", 10), text(x + 28, y + 46, "Activity 전체 분포", 25, "#272145", 850)]
    items = list(data.items())
    col_w = (w - 70) / 2
    row_h = 27
    max_value = max(data.values()) if data else 1
    for i, (name, value) in enumerate(items):
        col = i // 20
        row = i % 20
        xx = x + 28 + col * col_w
        yy = y + 82 + row * row_h
        bar_x = xx + 86
        bar_w = col_w - 245
        color = "#dc2626" if name.startswith("F") or name == "fall" else "#5b55d9"
        if name in {"walk", "wandering", "idle", "sit"}:
            color = "#059669"
        parts += [
            text(xx, yy + 15, name, 13, "#514872", 750),
            rect(bar_x, yy, bar_w, 16, "#f2effb", "none", 6),
            rect(bar_x, yy, bar_w * value / max_value, 16, color, "none", 6),
            text(xx + col_w - 18, yy + 14, f"{value:,}", 12, "#6f6791", 650, "end"),
        ]
    parts.append(text(x + 28, y + h - 24, f"Total activities: {len(data)} · Total rows: {total:,}", 15, "#8a82a6", 650))
    return parts


def render_svg(report: dict[str, Any]) -> str:
    width = 1800
    height = 1700
    rows = report["rows"]
    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        rect(0, 0, width, height, "#fbfaff"),
        text(80, 76, "전체 데이터셋 분포 - ICCAS + SisFall", 42, "#272145", 850),
        text(80, 118, "Source · Label · Binary Fall · Activity · Subject · Trial", 21, "#6f6791", 650),
        text(1720, 76, f"{rows:,} rows", 24, "#5b55d9", 850, "end"),
        text(1720, 108, f"{report['columns']} columns", 18, "#8a82a6", 700, "end"),
    ]
    cards = [
        ("Source", len(report["source_dataset"])),
        ("Labels", len(report["label"])),
        ("Activities", len(report["activity"])),
        ("Subjects", len(report["subject"])),
    ]
    for i, (name, value) in enumerate(cards):
        xx = 80 + i * 420
        parts += [
            rect(xx, 165, 370, 120, "#ffffff", "#eee9fb", 10),
            text(xx + 28, 210, name, 21, "#6f6791", 750),
            text(xx + 28, 262, value, 38, "#272145", 850),
        ]
    parts += horizontal_chart(80, 330, 800, 250, "Source Dataset", report["source_dataset"], rows)
    parts += horizontal_chart(920, 330, 800, 330, "Label 분포", report["label"], rows)
    parts += horizontal_chart(80, 625, 800, 210, "Binary Fall", report["binary_fall"], rows)
    parts += horizontal_chart(920, 705, 800, 210, "Subject 분포", report["subject"], rows)
    parts += horizontal_chart(80, 880, 800, 270, "Trial 분포", report["trial"], rows)
    parts += compact_activity_chart(920, 960, 800, 620, report["activity"], rows)
    parts += [
        rect(80, 1205, 800, 255, "#111827", "#111827", 10),
        text(120, 1254, "학습 적용 요약", 24, "#ffffff", 850),
        text(120, 1300, "ICCAS: walk / wandering / fall / idle / sit", 19, "#e5e7eb", 650),
        text(120, 1340, "SisFall: D01-D19 normal ADL, F01-F15 fall", 19, "#e5e7eb", 650),
        text(120, 1380, "IMU LSTM input: 50 samples x 12 features", 19, "#e5e7eb", 650),
        text(120, 1420, "GPS is not available in SisFall, so SisFall is used only for IMU/Gyro fall training.", 17, "#cbd5e1", 550),
        text(80, 1668, f"Source: {report['source']}", 15, "#8a82a6", 600),
        "</svg>",
    ]
    return "\n".join(parts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("../data/iccas_sensor_lstm/final_iccas_sisfall_imu_merged.csv"))
    parser.add_argument("--summary", type=Path, default=Path("../data/iccas_sensor_lstm/all_dataset_distribution_summary.json"))
    parser.add_argument("--markdown", type=Path, default=Path("docs/ALL_DATASET_DISTRIBUTION.md"))
    parser.add_argument("--svg-output", type=Path, default=Path("assets/all_dataset_distribution.svg"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = derive(args.source)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(render_markdown(report), encoding="utf-8")
    args.svg_output.parent.mkdir(parents=True, exist_ok=True)
    args.svg_output.write_text(render_svg(report), encoding="utf-8")
    print(json.dumps({"summary": str(args.summary), "markdown": str(args.markdown), "svg_output": str(args.svg_output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
