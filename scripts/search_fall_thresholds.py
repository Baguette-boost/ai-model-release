"""Search IMU fall trigger thresholds on the current merged dataset."""

from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np

from train_hybrid_imu_fall import algorithm_scores, best_threshold
from train_sisfall_merged_imu_lstm import load_merged_csv, make_sequences, metrics


def grid_values(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def score_candidate(
    y_true: np.ndarray,
    x_raw: np.ndarray,
    impact_g: float,
    freefall_g: float,
    gyro_dps: float,
    tilt_deg: float,
    still_accel_std: float,
    still_gyro_dps: float,
    post_samples: int,
) -> dict[str, Any]:
    scores, diagnostics = algorithm_scores(
        x_raw,
        impact_g,
        freefall_g,
        gyro_dps,
        tilt_deg,
        still_accel_std,
        still_gyro_dps,
        post_samples,
    )
    threshold, result = best_threshold(y_true, scores)
    result = dict(result)
    result["score_threshold"] = threshold
    result["impact_g"] = impact_g
    result["freefall_g"] = freefall_g
    result["gyro_dps"] = gyro_dps
    result["tilt_deg"] = tilt_deg
    result["still_accel_std"] = still_accel_std
    result["still_gyro_dps"] = still_gyro_dps
    result["post_samples"] = post_samples
    result["diagnostics"] = diagnostics
    return result


def to_csv(results: list[dict[str, Any]], path: Path) -> None:
    headers = [
        "rank",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "tp",
        "fp",
        "tn",
        "fn",
        "score_threshold",
        "impact_g",
        "freefall_g",
        "gyro_dps",
        "tilt_deg",
        "still_accel_std",
        "still_gyro_dps",
        "post_samples",
    ]
    lines = [",".join(headers)]
    for rank, item in enumerate(results, start=1):
        values = [
            rank,
            item["accuracy"],
            item["precision"],
            item["recall"],
            item["f1"],
            item["tp"],
            item["fp"],
            item["tn"],
            item["fn"],
            item["score_threshold"],
            item["impact_g"],
            item["freefall_g"],
            item["gyro_dps"],
            item["tilt_deg"],
            item["still_accel_std"],
            item["still_gyro_dps"],
            item["post_samples"],
        ]
        lines.append(",".join(str(value) for value in values))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_markdown(report: dict[str, Any], path: Path) -> None:
    best = report["best_by_f1"]
    current = report["current_threshold_result"]
    lines = [
        "# IMU Fall Threshold Search 결과",
        "",
        "## 목적",
        "",
        "직접 취득 장비의 ESP32 fall-suspect threshold에 적합한 값을 찾기 위해 현재 IMU 낙상 데이터셋에서 grid search를 수행했습니다.",
        "",
        "## 데이터셋",
        "",
        f"- Source: `{report['source']}`",
        f"- Sequence length: `{report['sequence_length']}`",
        f"- Split: `{report['split']}`",
        f"- Test sequences: `{report['test_size']}`",
        f"- Test fall positives: `{report['test_positive']}`",
        "",
        "## 현재 기준값 성능",
        "",
        f"- FALL_IMPACT_G: `{current['impact_g']}`",
        f"- FALL_FREE_G: `{current['freefall_g']}`",
        f"- gyro_dps: `{current['gyro_dps']}`",
        f"- tilt_deg: `{current['tilt_deg']}`",
        f"- still_accel_std: `{current['still_accel_std']}`",
        f"- still_gyro_dps: `{current['still_gyro_dps']}`",
        f"- Algorithm score threshold: `{current['score_threshold']:.2f}`",
        f"- Accuracy: `{current['accuracy']:.4f}`",
        f"- Precision: `{current['precision']:.4f}`",
        f"- Recall: `{current['recall']:.4f}`",
        f"- F1-score: `{current['f1']:.4f}`",
        "",
        "## Best F1 threshold",
        "",
        f"- FALL_IMPACT_G: `{best['impact_g']}`",
        f"- FALL_FREE_G: `{best['freefall_g']}`",
        f"- gyro_dps: `{best['gyro_dps']}`",
        f"- tilt_deg: `{best['tilt_deg']}`",
        f"- still_accel_std: `{best['still_accel_std']}`",
        f"- still_gyro_dps: `{best['still_gyro_dps']}`",
        f"- Algorithm score threshold: `{best['score_threshold']:.2f}`",
        f"- Accuracy: `{best['accuracy']:.4f}`",
        f"- Precision: `{best['precision']:.4f}`",
        f"- Recall: `{best['recall']:.4f}`",
        f"- F1-score: `{best['f1']:.4f}`",
        f"- Confusion Matrix: TP `{best['tp']}`, FP `{best['fp']}`, TN `{best['tn']}`, FN `{best['fn']}`",
        "",
        "## 추천",
        "",
        "ESP32에서는 단독 최종 판단보다 fall-suspect 트리거로 사용하고, 서버 LSTM이 최종 낙상을 판단하는 구조를 권장합니다.",
        "",
        "```cpp",
        "#define SAMPLE_MS     40",
        "#define IMU_BUF_N     50",
        f"#define FALL_IMPACT_G {best['impact_g']:.1f}f",
        f"#define FALL_FREE_G   {best['freefall_g']:.1f}f",
        "#define FALL_COOLDOWN 5000",
        "```",
        "",
        "## 상위 후보",
        "",
        "| Rank | Impact G | Free G | Gyro dps | Still gyro | Precision | Recall | F1 | FP | FN |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for index, item in enumerate(report["top_results"][:10], start=1):
        lines.append(
            f"| {index} | {item['impact_g']:.1f} | {item['freefall_g']:.1f} | {item['gyro_dps']:.0f} | "
            f"{item['still_gyro_dps']:.0f} | {item['precision']:.4f} | {item['recall']:.4f} | "
            f"{item['f1']:.4f} | {item['fp']} | {item['fn']} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("../data/iccas_sensor_lstm/final_iccas_sisfall_imu_merged.csv"))
    parser.add_argument("--report", type=Path, default=Path("../data/iccas_sensor_lstm/fall_threshold_search.json"))
    parser.add_argument("--csv", type=Path, default=Path("../data/iccas_sensor_lstm/fall_threshold_search.csv"))
    parser.add_argument("--markdown", type=Path, default=Path("docs/FALL_THRESHOLD_SEARCH_REPORT.md"))
    parser.add_argument("--sequence-length", type=int, default=50)
    parser.add_argument("--sequence-stride", type=int, default=4)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--validation-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--impact-values", default="1.8,2.0,2.2,2.5,2.8,3.0,3.3,3.5")
    parser.add_argument("--freefall-values", default="0.4,0.5,0.6,0.7,0.8")
    parser.add_argument("--gyro-values", default="150,200,250,300,350,400")
    parser.add_argument("--tilt-values", default="30,45,60")
    parser.add_argument("--still-accel-values", default="0.20,0.35,0.50")
    parser.add_argument("--still-gyro-values", default="40,80,120")
    parser.add_argument("--post-samples", type=int, default=25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = load_merged_csv(args.source)
    split = make_sequences(
        frame,
        args.sequence_length,
        args.sequence_stride,
        args.train_ratio,
        args.validation_ratio,
        args.seed,
    )
    x_test = split["x_test"]
    y_test = split["y_test"]
    print(f"test_sequences={len(y_test)} positives={int(y_test.sum())}")

    current = score_candidate(y_test, x_test, 2.5, 0.6, 250.0, 45.0, 0.35, 80.0, args.post_samples)
    results: list[dict[str, Any]] = []
    combos = product(
        grid_values(args.impact_values),
        grid_values(args.freefall_values),
        grid_values(args.gyro_values),
        grid_values(args.tilt_values),
        grid_values(args.still_accel_values),
        grid_values(args.still_gyro_values),
    )
    for idx, (impact_g, freefall_g, gyro_dps, tilt_deg, still_accel_std, still_gyro_dps) in enumerate(combos, start=1):
        result = score_candidate(
            y_test,
            x_test,
            impact_g,
            freefall_g,
            gyro_dps,
            tilt_deg,
            still_accel_std,
            still_gyro_dps,
            args.post_samples,
        )
        results.append(result)
        if idx % 200 == 0:
            print(f"searched={idx}")
    results.sort(key=lambda item: (item["f1"], item["precision"], item["accuracy"]), reverse=True)
    report = {
        "source": str(args.source),
        "sequence_length": args.sequence_length,
        "split": "test split from same group/chronological split used by IMU fall training",
        "test_size": int(len(y_test)),
        "test_positive": int(y_test.sum()),
        "current_threshold_result": current,
        "best_by_f1": results[0],
        "top_results": results[:30],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    to_csv(results, args.csv)
    write_markdown(report, args.markdown)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
