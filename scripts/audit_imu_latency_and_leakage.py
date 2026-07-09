"""Audit IMU sequence split leakage risk and CPU latency measurement assumptions."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from compare_preprocessed_imu_models import (
    BinarySequenceModel,
    FEATURE_COLUMNS,
    assign_sisfall_group_splits,
    load_preprocessed_csv,
)
from train_sisfall_merged_imu_lstm import BinaryLSTM


def split_position(position: float, train_ratio: float, validation_ratio: float) -> str:
    if position < train_ratio:
        return "train"
    if position < train_ratio + validation_ratio:
        return "validation"
    return "test"


def sequence_ranges(frame: pd.DataFrame, args: argparse.Namespace) -> list[dict[str, Any]]:
    ranges: list[dict[str, Any]] = []
    for group_id, group in frame.groupby("group_id", sort=False):
        group = group.sort_values("t_ms", kind="mergesort").reset_index(drop=False)
        if len(group) < args.sequence_length:
            continue
        source_dataset = str(group["source_dataset"].iloc[0])
        for end in range(args.sequence_length, len(group) + 1, args.sequence_stride):
            start = end - args.sequence_length
            split = split_position(end / len(group), args.train_ratio, args.validation_ratio)
            if source_dataset != "ICCAS":
                split = str(group["split"].iloc[0])
            ranges.append(
                {
                    "source_dataset": source_dataset,
                    "group_id": str(group_id),
                    "split": split,
                    "start": start,
                    "end": end,
                }
            )
    return ranges


def leakage_audit(ranges: list[dict[str, Any]]) -> dict[str, Any]:
    by_group: dict[str, list[dict[str, Any]]] = {}
    for item in ranges:
        by_group.setdefault(item["group_id"], []).append(item)

    shared_groups = 0
    boundary_overlap_groups = 0
    overlapping_pairs = 0
    overlap_samples_total = 0
    exact_duplicate_ranges = 0
    dataset_summary: dict[str, dict[str, int]] = {}

    for group_items in by_group.values():
        splits = {item["split"] for item in group_items}
        dataset = group_items[0]["source_dataset"]
        summary = dataset_summary.setdefault(
            dataset,
            {
                "groups": 0,
                "groups_with_multiple_splits": 0,
                "sequences": 0,
                "cross_split_overlapping_pairs": 0,
                "cross_split_overlap_samples": 0,
                "exact_duplicate_ranges": 0,
            },
        )
        summary["groups"] += 1
        summary["sequences"] += len(group_items)
        if len(splits) > 1:
            shared_groups += 1
            summary["groups_with_multiple_splits"] += 1

        for i, left in enumerate(group_items):
            for right in group_items[i + 1 :]:
                if left["split"] == right["split"]:
                    continue
                overlap = max(0, min(left["end"], right["end"]) - max(left["start"], right["start"]))
                if overlap <= 0:
                    continue
                overlapping_pairs += 1
                overlap_samples_total += overlap
                summary["cross_split_overlapping_pairs"] += 1
                summary["cross_split_overlap_samples"] += overlap
                if left["start"] == right["start"] and left["end"] == right["end"]:
                    exact_duplicate_ranges += 1
                    summary["exact_duplicate_ranges"] += 1
    boundary_overlap_groups = sum(
        1
        for group_items in by_group.values()
        if any(
            max(0, min(left["end"], right["end"]) - max(left["start"], right["start"])) > 0
            for i, left in enumerate(group_items)
            for right in group_items[i + 1 :]
            if left["split"] != right["split"]
        )
    )
    return {
        "groups_total": len(by_group),
        "groups_with_multiple_splits": shared_groups,
        "groups_with_cross_split_window_overlap": boundary_overlap_groups,
        "cross_split_overlapping_window_pairs": overlapping_pairs,
        "cross_split_overlap_samples_total": overlap_samples_total,
        "exact_duplicate_window_ranges": exact_duplicate_ranges,
        "by_dataset": dataset_summary,
    }


def time_model(model_name: str, repeats: int, sequence_length: int, feature_count: int, hidden_size: int, num_layers: int, dropout: float, transformer_heads: int) -> dict[str, float]:
    torch.set_num_threads(1)
    device = torch.device("cpu")
    model = BinarySequenceModel(model_name, feature_count, hidden_size, num_layers, dropout, transformer_heads).to(device)
    model.eval()
    np_sample = np.random.default_rng(42).normal(size=(1, sequence_length, feature_count)).astype(np.float32)
    tensor = torch.tensor(np_sample, dtype=torch.float32, device=device)
    with torch.no_grad():
        for _ in range(20):
            model(tensor)

        started = time.perf_counter()
        for _ in range(repeats):
            model(tensor)
        forward_ms = (time.perf_counter() - started) * 1000.0 / repeats

        started = time.perf_counter()
        for _ in range(repeats):
            current = torch.tensor(np_sample, dtype=torch.float32, device=device)
            model(current)
        tensor_plus_forward_ms = (time.perf_counter() - started) * 1000.0 / repeats

    return {
        "forward_only_ms": forward_ms,
        "tensor_create_plus_forward_ms": tensor_plus_forward_ms,
        "parameter_count": float(sum(parameter.numel() for parameter in model.parameters())),
    }


def time_final_lstm(checkpoint_path: Path, repeats: int) -> dict[str, float]:
    torch.set_num_threads(1)
    device = torch.device("cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = BinaryLSTM(
        len(FEATURE_COLUMNS),
        int(checkpoint["hidden_size"]),
        int(checkpoint["num_layers"]),
        float(checkpoint["dropout"]),
        bool(checkpoint["bidirectional"]),
        str(checkpoint["pooling"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    np_sample = np.random.default_rng(42).normal(
        size=(1, int(checkpoint["sequence_length"]), len(FEATURE_COLUMNS))
    ).astype(np.float32)
    tensor = torch.tensor(np_sample, dtype=torch.float32, device=device)
    with torch.no_grad():
        for _ in range(50):
            model(tensor)

        started = time.perf_counter()
        for _ in range(repeats):
            model(tensor)
        forward_ms = (time.perf_counter() - started) * 1000.0 / repeats

        started = time.perf_counter()
        for _ in range(repeats):
            current = torch.tensor(np_sample, dtype=torch.float32, device=device)
            model(current)
        tensor_plus_forward_ms = (time.perf_counter() - started) * 1000.0 / repeats

    return {
        "forward_only_ms": forward_ms,
        "tensor_create_plus_forward_ms": tensor_plus_forward_ms,
        "parameter_count": float(sum(parameter.numel() for parameter in model.parameters())),
        "window_seconds": int(checkpoint["sequence_length"]) * int(checkpoint["sample_ms"]) / 1000.0,
        "architecture": "2-layer bidirectional LSTM with attention pooling",
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    frame = load_preprocessed_csv(args.source)
    frame = assign_sisfall_group_splits(frame, args.train_ratio, args.validation_ratio, args.seed)
    ranges = sequence_ranges(frame, args)
    latency = {
        "rnn": time_model("rnn", args.repeats, args.sequence_length, len(FEATURE_COLUMNS), args.hidden_size, args.num_layers, args.dropout, args.transformer_heads),
        "transformer": time_model("transformer", args.repeats, args.sequence_length, len(FEATURE_COLUMNS), args.hidden_size, args.num_layers, args.dropout, args.transformer_heads),
        "final_lstm_trained_checkpoint": time_final_lstm(args.final_lstm_model, args.repeats),
    }
    report = {
        "source": str(args.source),
        "sequence_length": args.sequence_length,
        "sequence_stride": args.sequence_stride,
        "latency_note": "Latency uses randomly initialized models with identical architecture shape; it audits operation cost, not trained accuracy.",
        "leakage_audit": leakage_audit(ranges),
        "latency_audit_cpu_single_thread": latency,
        "interpretation": [
            "Exact duplicate windows across splits were checked separately from overlapping raw samples.",
            "SisFall uses group-level splits; ICCAS chronological windows can share boundary samples because sliding windows overlap near split boundaries.",
            "Forward-only latency excludes sensor acquisition, scaling, API, and database time.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("../data/iccas_sensor_lstm/imu_fall_preprocessed.csv"))
    parser.add_argument("--final-lstm-model", type=Path, default=Path("models/iccas_final_hybrid_lstm_imu_fall.pt"))
    parser.add_argument("--output", type=Path, default=Path("experiments/imu_latency_leakage_audit.json"))
    parser.add_argument("--sequence-length", type=int, default=50)
    parser.add_argument("--sequence-stride", type=int, default=4)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--validation-ratio", type=float, default=0.15)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--transformer-heads", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    report = run(parse_args())
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
