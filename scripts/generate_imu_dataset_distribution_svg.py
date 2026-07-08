"""Generate an SVG visualization of the IMU fall dataset distribution."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any

import pandas as pd


PALETTE = ["#2563eb", "#dc2626", "#059669", "#d97706", "#7c3aed", "#0891b2", "#be123c", "#4b5563"]


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


def pct(value: float, total: float) -> str:
    return f"{(value / max(total, 1)) * 100:.1f}%"


def bar_chart(x: float, y: float, w: float, h: float, title: str, data: dict[str, int], horizontal: bool = True) -> list[str]:
    total = sum(data.values())
    max_value = max(data.values()) if data else 1
    parts = [rect(x, y, w, h, "#ffffff"), text(x + 28, y + 48, title, 28, "#172033", 850)]
    if horizontal:
        start_y = y + 92
        row_h = min(48, (h - 120) / max(1, len(data)))
        for i, (label, value) in enumerate(data.items()):
            yy = start_y + i * row_h
            bar_w = w - 290
            bw = bar_w * value / max_value
            color = PALETTE[i % len(PALETTE)]
            parts += [
                text(x + 32, yy + 20, label, 17, "#344054", 700),
                rect(x + 190, yy, bar_w, 24, "#eef2f7", "none", 10),
                rect(x + 190, yy, bw, 24, color, "none", 10),
                text(x + w - 38, yy + 20, f"{value:,} ({pct(value, total)})", 16, "#667085", 650, "end"),
            ]
    return parts


def stacked_bar(x: float, y: float, w: float, h: float, title: str, rows: list[tuple[str, int, int]]) -> list[str]:
    parts = [rect(x, y, w, h, "#ffffff"), text(x + 28, y + 48, title, 28, "#172033", 850)]
    parts += [
        rect(x + 36, y + 75, 18, 18, "#dc2626", "none", 4),
        text(x + 62, y + 90, "fall / positive", 16, "#344054", 700),
        rect(x + 220, y + 75, 18, 18, "#2563eb", "none", 4),
        text(x + 246, y + 90, "non-fall / negative", 16, "#344054", 700),
    ]
    yy = y + 130
    for name, positive, negative in rows:
        total = positive + negative
        positive_w = (w - 240) * positive / max(total, 1)
        negative_w = (w - 240) * negative / max(total, 1)
        parts += [
            text(x + 36, yy + 20, name, 20, "#172033", 800),
            rect(x + 160, yy, w - 240, 28, "#eef2f7", "none", 10),
            rect(x + 160, yy, positive_w, 28, "#dc2626", "none", 10),
            rect(x + 160 + positive_w, yy, negative_w, 28, "#2563eb", "none", 10),
            text(x + w - 36, yy + 21, f"{positive:,} / {negative:,}", 17, "#667085", 700, "end"),
            text(x + 160, yy + 55, f"positive {pct(positive, total)}, total {total:,}", 15, "#667085", 550),
        ]
        yy += 86
    return parts


def derive_distribution(source: Path, metadata: Path) -> dict[str, Any]:
    frame = pd.read_csv(source, low_memory=False)
    meta = json.loads(metadata.read_text(encoding="utf-8"))
    label_counts = {str(k): int(v) for k, v in frame["label"].value_counts(dropna=False).to_dict().items()}
    source_counts = {str(k): int(v) for k, v in frame["source_dataset"].value_counts(dropna=False).to_dict().items()}
    fall_count = int((frame["label"].astype(str) == "fall").sum())
    non_fall_count = int(len(frame) - fall_count)
    activity_counts = {str(k): int(v) for k, v in frame["source_activity"].value_counts(dropna=False).head(8).to_dict().items()}
    split_rows = [
        ("Train", int(meta["label_counts"]["train_positive"]), int(meta["label_counts"]["train_negative"])),
        ("Validation", int(meta["label_counts"]["validation_positive"]), int(meta["label_counts"]["validation_negative"])),
        ("Test", int(meta["label_counts"]["test_positive"]), int(meta["label_counts"]["test_negative"])),
    ]
    return {
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "label_counts": label_counts,
        "source_counts": source_counts,
        "fall_binary": {"fall": fall_count, "non_fall": non_fall_count},
        "activity_top": activity_counts,
        "split_rows": split_rows,
        "model_sequence_counts": meta["split_sizes"],
        "test_metrics_by_dataset": meta["test_metrics_by_dataset"],
    }


def render_svg(report: dict[str, Any], source: Path) -> str:
    width = 1800
    height = 1320
    rows = report["rows"]
    fall = report["fall_binary"]["fall"]
    non_fall = report["fall_binary"]["non_fall"]
    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        rect(0, 0, width, height, "#f8fafc", "none", 0),
        text(80, 82, "IMU Fall Dataset Distribution", 44, "#111827", 850),
        text(80, 122, "ICCAS direct IMU data + SisFall IMU fall dataset", 22, "#667085", 500),
        text(1720, 82, f"{rows:,} samples", 24, "#2563eb", 850, "end"),
        text(1720, 116, "for IMU/Gyro fall detection", 18, "#667085", 700, "end"),
    ]

    cards = [
        ("Total samples", f"{rows:,}", "raw IMU rows"),
        ("Columns", f"{report['columns']}", "merged source columns"),
        ("Fall samples", f"{fall:,}", pct(fall, rows)),
        ("Non-fall samples", f"{non_fall:,}", pct(non_fall, rows)),
    ]
    for i, (title, value, note) in enumerate(cards):
        x = 80 + i * 415
        parts += [
            rect(x, 175, 370, 145, "#ffffff"),
            text(x + 24, 218, title, 21, "#667085", 700),
            text(x + 24, 274, value, 38, "#172033", 850),
            text(x + 24, 306, note, 16, "#667085", 550),
        ]

    parts += bar_chart(80, 370, 800, 390, "Label Distribution", report["label_counts"])
    parts += bar_chart(920, 370, 800, 270, "Source Dataset Distribution", report["source_counts"])
    parts += stacked_bar(920, 680, 800, 330, "Train / Validation / Test Sequence Labels", report["split_rows"])
    parts += bar_chart(80, 800, 800, 330, "Top Source Activities", report["activity_top"])

    metrics = report["test_metrics_by_dataset"]
    parts += [
        rect(80, 1170, 1640, 90, "#111827", "#111827"),
        text(120, 1225, "Dataset-level test F1", 24, "#ffffff", 850),
        text(430, 1225, f"ICCAS {metrics['ICCAS']['f1'] * 100:.1f}%  |  SisFall {metrics['SisFall']['f1'] * 100:.1f}%", 24, "#e5e7eb", 750),
        text(1720, 1225, f"Source: {source}", 16, "#cbd5e1", 550, "end"),
        "</svg>",
    ]
    return "\n".join(parts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("../data/iccas_sensor_lstm/final_iccas_sisfall_imu_merged.csv"))
    parser.add_argument("--metadata", type=Path, default=Path("models/iccas_final_hybrid_lstm_imu_fall.json"))
    parser.add_argument("--svg-output", type=Path, default=Path("assets/imu_dataset_distribution.svg"))
    parser.add_argument("--summary", type=Path, default=Path("../data/iccas_sensor_lstm/imu_dataset_distribution_summary.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = derive_distribution(args.source, args.metadata)
    args.svg_output.parent.mkdir(parents=True, exist_ok=True)
    args.svg_output.write_text(render_svg(report, args.source), encoding="utf-8")
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"svg_output": str(args.svg_output), "summary": str(args.summary), **report}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
