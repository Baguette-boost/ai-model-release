"""Export the IMU/Gyro fall preprocessing result to CSV."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


IMU_COLUMNS = [
    "server_time",
    "device",
    "label",
    "t_ms",
    "roll",
    "pitch",
    "yaw",
    "ax",
    "ay",
    "az",
    "wx",
    "wy",
    "wz",
    "accel_norm",
    "gyro_norm",
    "dt_s",
    "source_dataset",
    "source_file",
    "source_subject",
    "source_activity",
    "source_trial",
    "fall_target",
]


def preprocess_imu(source: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = pd.read_csv(source, low_memory=False)
    frame = frame.copy()
    for column in ["roll", "pitch", "yaw", "ax", "ay", "az", "wx", "wy", "wz", "t_ms"]:
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

    frame["accel_norm"] = np.sqrt(frame["ax"] ** 2 + frame["ay"] ** 2 + frame["az"] ** 2)
    frame["gyro_norm"] = np.sqrt(frame["wx"] ** 2 + frame["wy"] ** 2 + frame["wz"] ** 2)
    frame["fall_target"] = (frame["label"].astype(str) == "fall").astype(int)
    group_id = (
        frame["source_dataset"]
        + "::"
        + frame["source_file"]
        + "::"
        + frame["device"]
        + "::"
        + frame["source_activity"]
    )
    frame["dt_s"] = frame.groupby(group_id, sort=False)["t_ms"].diff().fillna(0.0).clip(lower=0.0, upper=1000.0) / 1000.0
    output = frame[[column for column in IMU_COLUMNS if column in frame.columns]].copy()
    summary = {
        "source": str(source),
        "rows": int(len(output)),
        "columns": list(output.columns),
        "label_counts": output["label"].value_counts(dropna=False).to_dict(),
        "source_dataset_counts": output["source_dataset"].value_counts(dropna=False).to_dict(),
        "fall_target_counts": output["fall_target"].value_counts(dropna=False).to_dict(),
        "feature_columns": [
            "roll",
            "pitch",
            "yaw",
            "ax",
            "ay",
            "az",
            "wx",
            "wy",
            "wz",
            "accel_norm",
            "gyro_norm",
            "dt_s",
        ],
        "notes": [
            "GPS columns are intentionally excluded.",
            "This CSV is for IMU/Gyro fall detection preprocessing.",
            "fall_target is 1 when label == fall, otherwise 0.",
        ],
    }
    return output, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("../data/iccas_sensor_lstm/final_iccas_sisfall_imu_merged.csv"))
    parser.add_argument("--output", type=Path, default=Path("../data/iccas_sensor_lstm/imu_fall_preprocessed.csv"))
    parser.add_argument("--summary", type=Path, default=Path("../data/iccas_sensor_lstm/imu_fall_preprocessed_summary.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output, summary = preprocess_imu(args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "summary": str(args.summary), **summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
