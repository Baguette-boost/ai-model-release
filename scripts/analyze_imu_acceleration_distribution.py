"""Analyze IMU acceleration distributions for LSTM fall detection suitability."""

from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


FEATURES = ["ax", "ay", "az", "accel_norm", "accel_delta", "accel_change_rate"]
GROUP_COLUMNS = ["source_dataset", "source_file", "device", "source_activity", "source_trial"]


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def quantiles(values: pd.Series) -> dict[str, float]:
    clean = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return {key: 0.0 for key in ["count", "mean", "std", "min", "p25", "p50", "p75", "p90", "p95", "p99", "max"]}
    return {
        "count": float(clean.size),
        "mean": float(clean.mean()),
        "std": float(clean.std(ddof=0)),
        "min": float(clean.min()),
        "p25": float(clean.quantile(0.25)),
        "p50": float(clean.quantile(0.50)),
        "p75": float(clean.quantile(0.75)),
        "p90": float(clean.quantile(0.90)),
        "p95": float(clean.quantile(0.95)),
        "p99": float(clean.quantile(0.99)),
        "max": float(clean.max()),
    }


def load_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    frame = frame.copy()
    for column in ["ax", "ay", "az", "t_ms"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if "accel_norm" not in frame.columns:
        frame["accel_norm"] = np.sqrt(frame["ax"] ** 2 + frame["ay"] ** 2 + frame["az"] ** 2)
    else:
        frame["accel_norm"] = pd.to_numeric(frame["accel_norm"], errors="coerce")
    if "dt_s" not in frame.columns:
        frame["dt_s"] = 0.0
    frame["dt_s"] = pd.to_numeric(frame["dt_s"], errors="coerce").fillna(0.0)
    for column, default in {
        "source_dataset": "unknown",
        "source_file": "unknown",
        "source_activity": "unknown",
        "source_trial": "unknown",
        "device": "unknown",
        "label": "unknown",
    }.items():
        if column not in frame.columns:
            frame[column] = default
        frame[column] = frame[column].fillna(default).astype(str)
    frame["fall_target"] = (frame["label"].astype(str).str.lower() == "fall").astype(int)
    frame["group_id"] = frame[GROUP_COLUMNS].astype(str).agg("::".join, axis=1)
    frame = frame.sort_values(["group_id", "t_ms"], kind="mergesort")
    frame["accel_delta"] = frame.groupby("group_id")["accel_norm"].diff().abs().fillna(0.0)
    safe_dt = frame["dt_s"].where(frame["dt_s"] > 1e-6)
    inferred_dt = frame.groupby("group_id")["t_ms"].diff().abs().div(1000.0)
    safe_dt = safe_dt.fillna(inferred_dt).replace(0.0, np.nan)
    frame["accel_change_rate"] = (frame["accel_delta"] / safe_dt).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return frame


def class_name(row: pd.Series) -> str:
    source = str(row["source_dataset"])
    label = "fall" if int(row["fall_target"]) == 1 else "normal"
    return f"{source}_{label}"


def summarize_distributions(frame: pd.DataFrame) -> list[dict[str, Any]]:
    frame = frame.copy()
    frame["analysis_group"] = frame.apply(class_name, axis=1)
    rows: list[dict[str, Any]] = []
    for group_name, group in frame.groupby("analysis_group", sort=True):
        source, label = group_name.rsplit("_", 1)
        for feature in FEATURES:
            item = {
                "source_dataset": source,
                "class": label,
                "feature": feature,
                **quantiles(group[feature]),
            }
            rows.append(item)
    return rows


def summarize_sequences(frame: pd.DataFrame, sequence_length: int, stride: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group_id, group in frame.groupby("group_id", sort=False):
        group = group.sort_values("t_ms", kind="mergesort")
        n = len(group)
        windows = 0 if n < sequence_length else 1 + (n - sequence_length) // stride
        source = str(group["source_dataset"].iloc[0])
        label = "fall" if int(group["fall_target"].max()) == 1 else "normal"
        rows.append(
            {
                "group_id": group_id,
                "source_dataset": source,
                "class": label,
                "rows": int(n),
                "lstm_ready": bool(n >= sequence_length),
                "window_count": int(windows),
                "accel_norm_p95": quantiles(group["accel_norm"])["p95"],
                "accel_delta_p95": quantiles(group["accel_delta"])["p95"],
            }
        )
    return rows


def effect_size(a: pd.Series, b: pd.Series) -> float:
    a = pd.to_numeric(a, errors="coerce").dropna()
    b = pd.to_numeric(b, errors="coerce").dropna()
    if len(a) < 2 or len(b) < 2:
        return 0.0
    pooled = np.sqrt((float(a.var(ddof=1)) + float(b.var(ddof=1))) / 2.0)
    if pooled <= 1e-12:
        return 0.0
    return float((a.mean() - b.mean()) / pooled)


def suitability(frame: pd.DataFrame, sequence_rows: list[dict[str, Any]], sequence_length: int) -> dict[str, Any]:
    source_counts = frame.groupby(["source_dataset", "fall_target"]).size().to_dict()
    window_frame = pd.DataFrame(sequence_rows)
    window_summary = (
        window_frame.groupby(["source_dataset", "class"])
        .agg(groups=("group_id", "count"), ready_groups=("lstm_ready", "sum"), windows=("window_count", "sum"), median_rows=("rows", "median"))
        .reset_index()
        .to_dict("records")
    )
    fall = frame[frame["fall_target"] == 1]
    normal = frame[frame["fall_target"] == 0]
    return {
        "sequence_length": sequence_length,
        "row_counts": {f"{key[0]}_{'fall' if key[1] else 'normal'}": int(value) for key, value in source_counts.items()},
        "window_summary": window_summary,
        "effect_size_fall_vs_normal": {
            "accel_norm": effect_size(fall["accel_norm"], normal["accel_norm"]),
            "accel_delta": effect_size(fall["accel_delta"], normal["accel_delta"]),
            "accel_change_rate": effect_size(fall["accel_change_rate"], normal["accel_change_rate"]),
        },
    }


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def metric_lookup(rows: list[dict[str, Any]], source: str, label: str, feature: str) -> dict[str, Any]:
    for row in rows:
        if row["source_dataset"] == source and row["class"] == label and row["feature"] == feature:
            return row
    return {}


def write_markdown(report: dict[str, Any], path: Path) -> None:
    rows = report["distribution_summary"]
    lines = [
        "# IMU 가속도 변화량 데이터 분포 분석",
        "",
        "## 목적",
        "",
        "기존 fall 데이터와 ICCAS 데이터를 비교해, 가속도 크기와 변화량이 LSTM 낙상 감지 학습에 적합한지 확인했다.",
        "",
        "## 분석 대상",
        "",
        f"- Source CSV: `{report['source']}`",
        f"- Sequence length 기준: `{report['sequence_length']}` samples",
        "- 비교 기준: `source_dataset`과 `label == fall` 여부",
        "- 핵심 feature: `accel_norm`, `accel_delta = |accel_norm_t - accel_norm_{t-1}|`, `accel_change_rate = accel_delta / dt_s`",
        "",
        "## 핵심 분포 요약",
        "",
        "| Dataset | Class | accel_norm p50 | accel_norm p95 | accel_norm p99 | accel_delta p50 | accel_delta p95 | accel_delta p99 | change_rate p95 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    pairs = sorted({(row["source_dataset"], row["class"]) for row in rows})
    for source, label in pairs:
        norm = metric_lookup(rows, source, label, "accel_norm")
        delta = metric_lookup(rows, source, label, "accel_delta")
        rate = metric_lookup(rows, source, label, "accel_change_rate")
        lines.append(
            f"| {source} | {label} | {norm.get('p50', 0):.4f} | {norm.get('p95', 0):.4f} | {norm.get('p99', 0):.4f} | "
            f"{delta.get('p50', 0):.4f} | {delta.get('p95', 0):.4f} | {delta.get('p99', 0):.4f} | {rate.get('p95', 0):.4f} |"
        )
    effect = report["suitability"]["effect_size_fall_vs_normal"]
    lines += [
        "",
        "## LSTM 적합성 판단",
        "",
        f"- Fall vs normal 효과크기 Cohen's d: `accel_norm={effect['accel_norm']:.3f}`, `accel_delta={effect['accel_delta']:.3f}`, `accel_change_rate={effect['accel_change_rate']:.3f}`.",
        "- 효과크기가 0에 가깝고 일부 normal 구간의 p95 변화량이 fall보다 크므로, 가속도 크기/변화량 단독 threshold만으로는 낙상을 안정적으로 분리하기 어렵다.",
        "- fall 데이터에도 큰 충격 tail이 존재하지만 normal 구간에도 큰 변화가 있어 단독 구분 신호로 보기는 어렵다. 이 변화는 LSTM이 전후 문맥과 함께 볼 때 낙상 패턴 학습에 도움이 된다.",
        "- 따라서 LSTM 입력은 `accel_delta`만 쓰기보다 `roll/pitch/yaw`, 3축 가속도, 3축 자이로, `accel_norm`, `gyro_norm`, `dt_s`를 함께 사용하는 방식이 적합하다.",
        "- LSTM은 단일 포인트가 아니라 50-step window에서 충격 전후 문맥을 보므로, 연속 샘플 수가 충분한 group만 학습/검증에 사용하는 것이 적절하다.",
        "",
        "## Sequence Window 요약",
        "",
        "| Dataset | Class | Groups | Ready groups | Window count | Median rows/group |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for item in report["suitability"]["window_summary"]:
        lines.append(
            f"| {item['source_dataset']} | {item['class']} | {int(item['groups'])} | {int(item['ready_groups'])} | "
            f"{int(item['windows'])} | {float(item['median_rows']):.1f} |"
        )
    lines += [
        "",
        "## 결론",
        "",
        "- 가속도 변화량만 보면 fall/normal이 깔끔하게 분리되지 않는다. 따라서 단순 threshold 모델보다는 LSTM처럼 시계열 문맥을 보는 모델이 더 적합하다.",
        "- 기존 fall 데이터는 낙상 충격 tail과 낙상 전후 자세/회전 변화를 제공해 positive pattern 학습에 필요하다.",
        "- ICCAS 데이터는 실제 장비/환경의 normal 분포를 제공하므로, false positive를 줄이는 negative baseline 역할을 한다.",
        "- 두 데이터의 센서 스케일과 샘플링 특성이 다를 수 있어 robust scaling과 source별 검증 split이 필요하다.",
        "- 최종 판단: 현재 데이터는 LSTM 학습에 사용할 수 있지만, 가속도 변화량 단독 분포가 아니라 다중 IMU feature와 연속 window를 사용하는 조건에서 적합하다.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_svg(report: dict[str, Any], path: Path) -> None:
    rows = report["distribution_summary"]
    pairs = sorted({(row["source_dataset"], row["class"]) for row in rows})
    labels = [f"{source}\n{label}" for source, label in pairs]
    norm_p95 = [metric_lookup(rows, source, label, "accel_norm").get("p95", 0.0) for source, label in pairs]
    delta_p95 = [metric_lookup(rows, source, label, "accel_delta").get("p95", 0.0) for source, label in pairs]

    width, height = 1180, 720
    left, top = 92, 110
    chart_w, chart_h = 980, 410
    max_value = max(norm_p95 + delta_p95 + [1.0]) * 1.15

    def y(value: float) -> float:
        return top + chart_h - (value / max_value) * chart_h

    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="1180" height="720" rx="18" fill="#f8f7fb"/>',
        '<text x="48" y="48" font-family="Arial, sans-serif" font-size="25" font-weight="700" fill="#242033">IMU Acceleration Distribution Suitability</text>',
        '<text x="48" y="76" font-family="Arial, sans-serif" font-size="13" fill="#645f78">Existing fall data vs ICCAS data / p95 acceleration magnitude and sample-to-sample change</text>',
    ]
    for tick in np.linspace(0, max_value, 6):
        yy = y(float(tick))
        elements.append(f'<line x1="{left}" y1="{yy:.1f}" x2="{left + chart_w}" y2="{yy:.1f}" stroke="#ded9ec"/>')
        elements.append(f'<text x="48" y="{yy + 4:.1f}" font-family="Arial, sans-serif" font-size="11" fill="#6c667b">{tick:.2f}</text>')
    elements.append(f'<rect x="{left}" y="{top}" width="{chart_w}" height="{chart_h}" fill="none" stroke="#ded9ec"/>')
    elements += [
        '<rect x="740" y="40" width="13" height="13" fill="#5a67d8"/>',
        '<text x="760" y="51" font-family="Arial, sans-serif" font-size="13" fill="#39344d">accel_norm p95</text>',
        '<rect x="890" y="40" width="13" height="13" fill="#e56b6f"/>',
        '<text x="910" y="51" font-family="Arial, sans-serif" font-size="13" fill="#39344d">accel_delta p95</text>',
    ]
    group_w = chart_w / max(1, len(pairs))
    for idx, ((source, label), n95, d95) in enumerate(zip(pairs, norm_p95, delta_p95)):
        base_x = left + idx * group_w + group_w / 2 - 38
        for offset, value, color in [(0, n95, "#5a67d8"), (42, d95, "#e56b6f")]:
            yy = y(float(value))
            h = top + chart_h - yy
            elements.append(f'<rect x="{base_x + offset:.1f}" y="{yy:.1f}" width="32" height="{h:.1f}" rx="5" fill="{color}"/>')
            elements.append(f'<text x="{base_x + offset + 16:.1f}" y="{yy - 7:.1f}" text-anchor="middle" font-family="Arial, sans-serif" font-size="10" fill="#2d293d">{value:.2f}</text>')
        elements.append(f'<text x="{left + idx * group_w + group_w / 2:.1f}" y="{top + chart_h + 32}" text-anchor="middle" font-family="Arial, sans-serif" font-size="13" font-weight="700" fill="#2d293d">{esc(source)}</text>')
        elements.append(f'<text x="{left + idx * group_w + group_w / 2:.1f}" y="{top + chart_h + 51}" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" fill="#69627a">{esc(label)}</text>')

    effect = report["suitability"]["effect_size_fall_vs_normal"]
    elements.append('<text x="48" y="600" font-family="Arial, sans-serif" font-size="17" font-weight="700" fill="#242033">LSTM suitability notes</text>')
    notes = [
        f"Effect size: accel_norm {effect['accel_norm']:.2f}, accel_delta {effect['accel_delta']:.2f}, change_rate {effect['accel_change_rate']:.2f}",
        "Use sequence context: fall is better captured by impact + rotation + inactivity pattern than one acceleration point.",
        f"Sequence length: {report['sequence_length']} samples; only ready groups are used for LSTM windows.",
    ]
    for i, note in enumerate(notes):
        y0 = 628 + i * 26
        elements.append(f'<text x="64" y="{y0}" font-family="Arial, sans-serif" font-size="14" fill="#3b354d">- {esc(note)}</text>')
    elements.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(elements) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("../data/iccas_sensor_lstm/imu_fall_preprocessed.csv"))
    parser.add_argument("--sequence-length", type=int, default=50)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--output-json", type=Path, default=Path("../data/iccas_sensor_lstm/imu_acceleration_distribution_analysis.json"))
    parser.add_argument("--output-csv", type=Path, default=Path("../data/iccas_sensor_lstm/imu_acceleration_distribution_summary.csv"))
    parser.add_argument("--output-md", type=Path, default=Path("docs/IMU_ACCELERATION_DISTRIBUTION_ANALYSIS.md"))
    parser.add_argument("--output-svg", type=Path, default=Path("assets/imu_acceleration_distribution_analysis.svg"))
    args = parser.parse_args()

    frame = load_frame(args.source)
    distribution_rows = summarize_distributions(frame)
    sequence_rows = summarize_sequences(frame, args.sequence_length, args.stride)
    report = {
        "source": str(args.source),
        "sequence_length": args.sequence_length,
        "stride": args.stride,
        "row_count": int(len(frame)),
        "distribution_summary": distribution_rows,
        "sequence_summary": sequence_rows,
        "suitability": suitability(frame, sequence_rows, args.sequence_length),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(distribution_rows, args.output_csv)
    write_markdown(report, args.output_md)
    write_svg(report, args.output_svg)
    print(json.dumps({"json": str(args.output_json), "csv": str(args.output_csv), "md": str(args.output_md), "svg": str(args.output_svg)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
