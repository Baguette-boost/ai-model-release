"""Export IMU Sensor Vector Magnitude (SVM) values to CSV."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


BASE_COLUMNS = [
    "server_time",
    "device",
    "label",
    "t_ms",
    "ax",
    "ay",
    "az",
    "svm_g",
    "wx",
    "wy",
    "wz",
    "gyro_norm",
    "fall_target",
    "source_dataset",
    "source_file",
    "source_subject",
    "source_activity",
    "source_trial",
]


def export_svm(source: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = pd.read_csv(source, low_memory=False)
    frame = frame.copy()
    for column in ["ax", "ay", "az", "wx", "wy", "wz", "t_ms"]:
        if column not in frame.columns:
            frame[column] = 0.0
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
    for column, default in {
        "server_time": "",
        "device": "unknown-device",
        "label": "normal",
        "source_dataset": "unknown",
        "source_file": "unknown-file",
        "source_subject": "unknown-subject",
        "source_activity": "unknown-activity",
        "source_trial": "unknown-trial",
    }.items():
        if column not in frame.columns:
            frame[column] = default
        frame[column] = frame[column].fillna(default).astype(str)

    frame["svm_g"] = np.sqrt(frame["ax"] ** 2 + frame["ay"] ** 2 + frame["az"] ** 2)
    if "gyro_norm" not in frame.columns:
        frame["gyro_norm"] = np.sqrt(frame["wx"] ** 2 + frame["wy"] ** 2 + frame["wz"] ** 2)
    if "fall_target" not in frame.columns:
        frame["fall_target"] = (frame["label"].astype(str) == "fall").astype(int)
    output = frame[[column for column in BASE_COLUMNS if column in frame.columns]].copy()
    summary = {
        "source": str(source),
        "rows": int(len(output)),
        "columns": list(output.columns),
        "formula": "svm_g = sqrt(ax^2 + ay^2 + az^2)",
        "svm_g": {
            "mean": float(output["svm_g"].mean()),
            "std": float(output["svm_g"].std()),
            "min": float(output["svm_g"].min()),
            "p50": float(output["svm_g"].quantile(0.50)),
            "p95": float(output["svm_g"].quantile(0.95)),
            "p99": float(output["svm_g"].quantile(0.99)),
            "max": float(output["svm_g"].max()),
        },
        "label_counts": output["label"].value_counts(dropna=False).to_dict(),
        "fall_target_counts": output["fall_target"].value_counts(dropna=False).to_dict(),
        "notes": [
            "svm_g is the same physical feature as accel_norm.",
            "This file is for IMU fall threshold analysis and visualization.",
        ],
    }
    return output, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("../data/iccas_sensor_lstm/imu_fall_preprocessed.csv"))
    parser.add_argument("--output", type=Path, default=Path("../data/iccas_sensor_lstm/imu_fall_svm.csv"))
    parser.add_argument("--summary", type=Path, default=Path("../data/iccas_sensor_lstm/imu_fall_svm_summary.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output, summary = export_svm(args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "summary": str(args.summary), **summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
