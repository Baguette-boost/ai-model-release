"""Compare IMU fall models before and after preprocessing feature expansion."""

from __future__ import annotations

import argparse
import csv
import html
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from compare_preprocessed_imu_models import (
    BinarySequenceModel,
    RobustScaler,
    SequenceDataset,
    assign_sisfall_group_splits,
    best_threshold,
    load_preprocessed_csv,
    measure_latency,
    metrics,
    predict_scores,
    resolve_device,
    scale_split,
    set_seed,
    split_position,
    synchronize,
)


RAW_FEATURE_COLUMNS = ["roll", "pitch", "yaw", "ax", "ay", "az", "wx", "wy", "wz"]
PREPROCESSED_FEATURE_COLUMNS = RAW_FEATURE_COLUMNS + ["accel_norm", "gyro_norm", "dt_s"]


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def make_sequences(frame: pd.DataFrame, args: argparse.Namespace, feature_columns: list[str]) -> dict[str, Any]:
    buckets: dict[str, list[Any]] = {
        "x_train": [],
        "y_train": [],
        "x_validation": [],
        "y_validation": [],
        "x_test": [],
        "y_test": [],
    }
    for _, group in frame.groupby("group_id", sort=False):
        group = group.sort_values("t_ms", kind="mergesort")
        values = group[feature_columns].to_numpy(dtype=np.float32)
        labels = group["fall_target"].to_numpy(dtype=np.float32)
        if len(values) < args.sequence_length:
            continue
        source_dataset = str(group["source_dataset"].iloc[0])
        for end in range(args.sequence_length, len(values) + 1, args.sequence_stride):
            start = end - args.sequence_length
            y_value = float(labels[start:end].max())
            position_split = split_position(end / len(values), args.train_ratio, args.validation_ratio)
            split = position_split if source_dataset == "ICCAS" else group["split"].iloc[0]
            buckets[f"x_{split}"].append(values[start:end])
            buckets[f"y_{split}"].append(y_value)

    out: dict[str, Any] = {}
    for split in ["train", "validation", "test"]:
        if not buckets[f"x_{split}"]:
            raise SystemExit(f"No {split} sequences were created.")
        out[f"x_{split}"] = np.stack(buckets[f"x_{split}"]).astype(np.float32)
        out[f"y_{split}"] = np.asarray(buckets[f"y_{split}"], dtype=np.float32)
    return out


def train_model(
    architecture: str,
    split: dict[str, Any],
    input_size: int,
    args: argparse.Namespace,
    device: torch.device,
    label: str,
) -> dict[str, Any]:
    set_seed(args.seed)
    model = BinarySequenceModel(
        architecture,
        input_size,
        args.hidden_size,
        args.num_layers,
        args.dropout,
        args.transformer_heads,
    ).to(device)
    pos = float(split["y_train"].sum())
    neg = float(len(split["y_train"]) - pos)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    loader = DataLoader(SequenceDataset(split["x_train"], split["y_train"]), batch_size=args.batch_size, shuffle=True)

    best_state: dict[str, torch.Tensor] | None = None
    best_f1 = -1.0
    history: list[dict[str, float]] = []
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        count = 0
        for sequences, labels in loader:
            sequences = sequences.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(sequences), labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.item()) * len(labels)
            count += len(labels)
        validation_scores = predict_scores(model, split["x_validation"], args.batch_size, device)
        _, validation_metrics = best_threshold(split["y_validation"], validation_scores)
        item = {
            "epoch": float(epoch),
            "loss": total / max(1, count),
            "validation_f1": float(validation_metrics["f1"]),
        }
        history.append(item)
        print(f"{label}/{architecture} epoch={epoch:03d} loss={item['loss']:.6f} val_f1={item['validation_f1']:.4f}")
        if validation_metrics["f1"] > best_f1:
            best_f1 = float(validation_metrics["f1"])
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    synchronize(device)
    train_seconds = time.perf_counter() - started
    if best_state is not None:
        model.load_state_dict(best_state)

    validation_scores = predict_scores(model, split["x_validation"], args.batch_size, device)
    threshold, validation_metrics = best_threshold(split["y_validation"], validation_scores)
    test_scores = predict_scores(model, split["x_test"], args.batch_size, device)
    test_metrics = metrics(split["y_test"], test_scores, threshold)
    latency = measure_latency(model, split["x_test"], args.batch_size, device, args.latency_repeats)
    return {
        "architecture": architecture,
        "threshold": threshold,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "latency": latency,
        "train_seconds": train_seconds,
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "history": history,
    }


def metric_delta(before: dict[str, float], after: dict[str, float], key: str) -> float:
    return float(after[key] - before[key])


def write_csv(report: dict[str, Any], path: Path) -> None:
    rows = []
    for result in report["results"]:
        before = result["raw"]["test_metrics"]
        after = result["preprocessed"]["test_metrics"]
        rows.append(
            {
                "model": result["architecture"].upper(),
                "raw_accuracy": before["accuracy"],
                "preprocessed_accuracy": after["accuracy"],
                "delta_accuracy": metric_delta(before, after, "accuracy"),
                "raw_precision": before["precision"],
                "preprocessed_precision": after["precision"],
                "delta_precision": metric_delta(before, after, "precision"),
                "raw_recall": before["recall"],
                "preprocessed_recall": after["recall"],
                "delta_recall": metric_delta(before, after, "recall"),
                "raw_f1": before["f1"],
                "preprocessed_f1": after["f1"],
                "delta_f1": metric_delta(before, after, "f1"),
                "raw_single_ms": result["raw"]["latency"]["single_sequence_ms"],
                "preprocessed_single_ms": result["preprocessed"]["latency"]["single_sequence_ms"],
                "raw_train_seconds": result["raw"]["train_seconds"],
                "preprocessed_train_seconds": result["preprocessed"]["train_seconds"],
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# IMU 낙상 감지 전처리 적용 전/후 성능 비교",
        "",
        "## 핵심 정리",
        "",
        "- 이 문서에서는 `하이브리드`라는 표현을 사용하지 않고, 모델 입력 전 `전처리 feature 추가`의 효과만 비교한다.",
        "- 전처리 전 입력: `roll, pitch, yaw, ax, ay, az, wx, wy, wz` 9개 feature.",
        "- 전처리 후 입력: 원본 9개 feature에 `accel_norm, gyro_norm, dt_s` 3개 feature를 추가한 12개 feature.",
        "- 같은 CSV, 같은 sequence length, 같은 train/validation/test split 조건에서 비교했다.",
        "",
        "## 성능 비교",
        "",
        "| Model | Accuracy Before | Accuracy After | Delta | Precision Before | Precision After | Delta | Recall Before | Recall After | Delta | F1 Before | F1 After | Delta |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    improved = {"accuracy": 0, "precision": 0, "recall": 0, "f1": 0}
    for result in report["results"]:
        before = result["raw"]["test_metrics"]
        after = result["preprocessed"]["test_metrics"]
        for key in improved:
            if after[key] > before[key]:
                improved[key] += 1
        lines.append(
            f"| {result['architecture'].upper()} | "
            f"{before['accuracy']:.4f} | {after['accuracy']:.4f} | {after['accuracy'] - before['accuracy']:+.4f} | "
            f"{before['precision']:.4f} | {after['precision']:.4f} | {after['precision'] - before['precision']:+.4f} | "
            f"{before['recall']:.4f} | {after['recall']:.4f} | {after['recall'] - before['recall']:+.4f} | "
            f"{before['f1']:.4f} | {after['f1']:.4f} | {after['f1'] - before['f1']:+.4f} |"
        )
    best_after = max(report["results"], key=lambda item: item["preprocessed"]["test_metrics"]["f1"])
    lines += [
        "",
        "## 발표용 문장",
        "",
        f"- 전처리 적용 후 F1-score 기준 최고 모델은 `{best_after['architecture'].upper()}`이며, F1-score는 `{best_after['preprocessed']['test_metrics']['f1']:.4f}`이다.",
        f"- 4개 모델 중 Recall이 개선된 모델은 `{improved['recall']}/4`, F1-score가 개선된 모델은 `{improved['f1']}/4`이다.",
        "- 낙상 감지는 정상/낙상 불균형이 크기 때문에 Accuracy만 보면 성능을 과대평가할 수 있다. 따라서 발표에서는 Recall과 F1-score를 함께 제시하는 것이 적절하다.",
        "",
        "## 추론 속도",
        "",
        "| Model | Before ms/sequence | After ms/sequence | Delta | Before train sec | After train sec |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for result in report["results"]:
        raw_latency = result["raw"]["latency"]["single_sequence_ms"]
        pre_latency = result["preprocessed"]["latency"]["single_sequence_ms"]
        lines.append(
            f"| {result['architecture'].upper()} | {raw_latency:.3f} | {pre_latency:.3f} | {pre_latency - raw_latency:+.3f} | "
            f"{result['raw']['train_seconds']:.1f} | {result['preprocessed']['train_seconds']:.1f} |"
        )
    lines += [
        "",
        "## 주의점",
        "",
        "- 전처리는 모델이 보기 쉬운 물리량을 추가하는 과정이지, 규칙 기반 알고리즘을 모델 결과와 섞는 과정이 아니다.",
        "- 모든 지표가 항상 동시에 오르는 것은 아니다. 실제 서비스 판단은 낙상 미탐을 줄이기 위해 Recall/F1 중심으로 보는 것이 더 타당하다.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_svg(report: dict[str, Any], path: Path) -> None:
    width = 1180
    height = 720
    margin_left = 86
    chart_top = 92
    chart_height = 408
    chart_width = 980
    group_gap = 44
    bar_width = 34
    max_y = 1.0
    models = [result["architecture"].upper() for result in report["results"]]
    colors = {
        "raw_f1": "#7c8a97",
        "pre_f1": "#6658e8",
        "raw_recall": "#b3bac2",
        "pre_recall": "#9b8cf5",
    }

    def y(value: float) -> float:
        return chart_top + chart_height - (value / max_y) * chart_height

    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="1180" height="720" rx="18" fill="#f7f5fc"/>',
        '<text x="50" y="46" font-family="Arial, sans-serif" font-size="25" font-weight="700" fill="#25223a">IMU Fall Detection - Preprocessing Effect</text>',
        '<text x="50" y="74" font-family="Arial, sans-serif" font-size="13" fill="#655f78">Before: raw 9-axis IMU / After: raw IMU + accel_norm + gyro_norm + dt_s</text>',
    ]
    for tick in np.linspace(0.0, 1.0, 6):
        yy = y(float(tick))
        elements.append(f'<line x1="{margin_left}" y1="{yy:.1f}" x2="{margin_left + chart_width}" y2="{yy:.1f}" stroke="#ded9ee" stroke-width="1"/>')
        elements.append(f'<text x="48" y="{yy + 4:.1f}" font-family="Arial, sans-serif" font-size="11" fill="#6d6880">{tick:.1f}</text>')

    elements += [
        f'<rect x="{margin_left}" y="{chart_top}" width="{chart_width}" height="{chart_height}" fill="none" stroke="#ded9ee"/>',
        '<rect x="690" y="36" width="12" height="12" fill="#7c8a97"/>',
        '<text x="708" y="46" font-family="Arial, sans-serif" font-size="12" fill="#403a58">F1 before</text>',
        '<rect x="790" y="36" width="12" height="12" fill="#6658e8"/>',
        '<text x="808" y="46" font-family="Arial, sans-serif" font-size="12" fill="#403a58">F1 after</text>',
        '<rect x="890" y="36" width="12" height="12" fill="#b3bac2"/>',
        '<text x="908" y="46" font-family="Arial, sans-serif" font-size="12" fill="#403a58">Recall before</text>',
        '<rect x="1016" y="36" width="12" height="12" fill="#9b8cf5"/>',
        '<text x="1034" y="46" font-family="Arial, sans-serif" font-size="12" fill="#403a58">Recall after</text>',
    ]

    group_width = (chart_width - group_gap * (len(models) - 1)) / len(models)
    for index, result in enumerate(report["results"]):
        group_x = margin_left + index * (group_width + group_gap) + 18
        before = result["raw"]["test_metrics"]
        after = result["preprocessed"]["test_metrics"]
        values = [
            ("raw_f1", before["f1"]),
            ("pre_f1", after["f1"]),
            ("raw_recall", before["recall"]),
            ("pre_recall", after["recall"]),
        ]
        for bar_index, (key, value) in enumerate(values):
            x = group_x + bar_index * (bar_width + 9)
            yy = y(float(value))
            bar_height = chart_top + chart_height - yy
            elements.append(f'<rect x="{x:.1f}" y="{yy:.1f}" width="{bar_width}" height="{bar_height:.1f}" rx="5" fill="{colors[key]}"/>')
            elements.append(f'<text x="{x + bar_width / 2:.1f}" y="{yy - 7:.1f}" text-anchor="middle" font-family="Arial, sans-serif" font-size="10" fill="#28233d">{value:.3f}</text>')
        delta_f1 = after["f1"] - before["f1"]
        delta_color = "#239a61" if delta_f1 >= 0 else "#cf4b54"
        elements.append(f'<text x="{group_x + 78:.1f}" y="{chart_top + chart_height + 32}" text-anchor="middle" font-family="Arial, sans-serif" font-size="15" font-weight="700" fill="#25223a">{esc(result["architecture"].upper())}</text>')
        elements.append(f'<text x="{group_x + 78:.1f}" y="{chart_top + chart_height + 53}" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" fill="{delta_color}">F1 delta {delta_f1:+.3f}</text>')

    table_top = 574
    elements.append('<text x="50" y="548" font-family="Arial, sans-serif" font-size="17" font-weight="700" fill="#25223a">Presentation-safe summary</text>')
    summary = []
    for result in report["results"]:
        before = result["raw"]["test_metrics"]
        after = result["preprocessed"]["test_metrics"]
        summary.append(
            f'{result["architecture"].upper()}: F1 {before["f1"]:.3f}->{after["f1"]:.3f}, Recall {before["recall"]:.3f}->{after["recall"]:.3f}'
        )
    for row, text in enumerate(summary):
        x = 50 + (row % 2) * 540
        yy = table_top + (row // 2) * 48
        elements.append(f'<rect x="{x}" y="{yy - 24}" width="500" height="36" rx="8" fill="#ffffff" stroke="#e1dced"/>')
        elements.append(f'<text x="{x + 16}" y="{yy}" font-family="Arial, sans-serif" font-size="14" fill="#38314d">{esc(text)}</text>')

    elements.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(elements) + "\n", encoding="utf-8")


def load_existing_preprocessed_results(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    by_arch: dict[str, Any] = {}
    for result in data.get("results", []):
        arch = str(result["architecture"])
        copied = dict(result)
        copied["test_metrics"] = result["test_metrics"]["model_only"]
        copied["validation_metrics"] = result["validation_metrics"]["model_only"]
        copied["threshold"] = result["model_threshold"]
        by_arch[arch] = copied
    return {"source_report": str(path), "by_architecture": by_arch}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=Path("../data/iccas_sensor_lstm/imu_fall_preprocessed.csv"))
    parser.add_argument("--existing-preprocessed-json", type=Path, default=Path("../data/iccas_sensor_lstm/imu_preprocessed_model_comparison.json"))
    parser.add_argument("--output-json", type=Path, default=Path("../data/iccas_sensor_lstm/imu_preprocessing_effect_comparison.json"))
    parser.add_argument("--output-csv", type=Path, default=Path("../data/iccas_sensor_lstm/imu_preprocessing_effect_comparison.csv"))
    parser.add_argument("--output-md", type=Path, default=Path("docs/IMU_PREPROCESSING_EFFECT_COMPARISON.md"))
    parser.add_argument("--output-svg", type=Path, default=Path("assets/imu_preprocessing_effect_comparison.svg"))
    parser.add_argument("--architectures", nargs="+", default=["rnn", "gru", "lstm", "transformer"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--sequence-length", type=int, default=50)
    parser.add_argument("--sequence-stride", type=int, default=25)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--validation-ratio", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=384)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--transformer-heads", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--latency-repeats", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--retrain-preprocessed", action="store_true")
    args = parser.parse_args()

    device = resolve_device(args.device)
    frame = load_preprocessed_csv(args.csv)
    frame = assign_sisfall_group_splits(frame, args.train_ratio, args.validation_ratio, args.seed)

    raw_split, raw_scaler = scale_split(make_sequences(frame, args, RAW_FEATURE_COLUMNS))
    pre_split, pre_scaler = scale_split(make_sequences(frame, args, PREPROCESSED_FEATURE_COLUMNS))
    existing_preprocessed = None if args.retrain_preprocessed else load_existing_preprocessed_results(args.existing_preprocessed_json)

    results = []
    for architecture in args.architectures:
        raw_result = train_model(architecture, raw_split, len(RAW_FEATURE_COLUMNS), args, device, "raw_imu")
        if existing_preprocessed and architecture in existing_preprocessed["by_architecture"]:
            preprocessed_result = existing_preprocessed["by_architecture"][architecture]
            preprocessed_result["source"] = "existing_preprocessed_model_comparison_json"
        else:
            preprocessed_result = train_model(
                architecture,
                pre_split,
                len(PREPROCESSED_FEATURE_COLUMNS),
                args,
                device,
                "preprocessed_imu",
            )
        results.append(
            {
                "architecture": architecture,
                "raw": raw_result,
                "preprocessed": preprocessed_result,
            }
        )

    report = {
        "task": "imu_fall_preprocessing_effect",
        "source": str(args.csv),
        "device": str(device),
        "sequence_length": args.sequence_length,
        "sequence_stride": args.sequence_stride,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "raw_feature_columns": RAW_FEATURE_COLUMNS,
        "preprocessed_feature_columns": PREPROCESSED_FEATURE_COLUMNS,
        "raw_scaler": {"center": raw_scaler.center.tolist(), "scale": raw_scaler.scale.tolist()},
        "preprocessed_scaler": {"center": pre_scaler.center.tolist(), "scale": pre_scaler.scale.tolist()},
        "results": results,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(report, args.output_csv)
    write_markdown(report, args.output_md)
    write_svg(report, args.output_svg)
    print(json.dumps({"json": str(args.output_json), "csv": str(args.output_csv), "md": str(args.output_md), "svg": str(args.output_svg)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
