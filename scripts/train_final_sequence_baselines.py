"""Train RNN and Transformer baselines with the final IMU fall LSTM protocol."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parent))

from train_sisfall_merged_imu_lstm import (  # noqa: E402
    FEATURE_COLUMNS,
    SequenceDataset,
    best_threshold,
    grouped_metrics,
    load_merged_csv,
    make_sequences,
    metrics,
    predict,
    resolve_device,
    scale_split,
    set_seed,
)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512) -> None:
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class FinalSequenceBaseline(nn.Module):
    def __init__(
        self,
        architecture: str,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        transformer_heads: int,
    ) -> None:
        super().__init__()
        self.architecture = architecture
        if architecture == "rnn":
            self.encoder = nn.RNN(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0.0,
                batch_first=True,
            )
            self.head = nn.Sequential(
                nn.LayerNorm(hidden_size),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, 1),
            )
        elif architecture == "transformer":
            self.input_projection = nn.Linear(input_size, hidden_size)
            self.position = PositionalEncoding(hidden_size)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=transformer_heads,
                dim_feedforward=hidden_size * 2,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.head = nn.Sequential(
                nn.LayerNorm(hidden_size),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, 1),
            )
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.architecture == "rnn":
            output, _ = self.encoder(x)
            pooled = output[:, -1, :]
        else:
            encoded = self.encoder(self.position(self.input_projection(x)))
            pooled = encoded.mean(dim=1)
        return self.head(pooled).squeeze(-1)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def measure_latency(model: nn.Module, x: np.ndarray, batch_size: int, device: torch.device, repeats: int) -> dict[str, float]:
    sample_count = min(len(x), max(batch_size, 1024))
    np_single = x[:1].astype(np.float32)
    single = torch.tensor(np_single, dtype=torch.float32, device=device)
    batch = torch.tensor(x[:sample_count], dtype=torch.float32, device=device)
    model.eval()
    with torch.no_grad():
        for _ in range(20):
            model(single)
            model(batch[: min(batch_size, sample_count)])
        synchronize(device)

        started = time.perf_counter()
        for _ in range(repeats):
            model(single)
        synchronize(device)
        forward_only_ms = (time.perf_counter() - started) * 1000.0 / max(1, repeats)

        started = time.perf_counter()
        for _ in range(repeats):
            current = torch.tensor(np_single, dtype=torch.float32, device=device)
            model(current)
        synchronize(device)
        tensor_create_plus_forward_ms = (time.perf_counter() - started) * 1000.0 / max(1, repeats)

        started = time.perf_counter()
        for _ in range(repeats):
            model(batch)
        synchronize(device)
        batch_ms = (time.perf_counter() - started) * 1000.0 / max(1, repeats)

    return {
        "forward_only_ms": forward_only_ms,
        "tensor_create_plus_forward_ms": tensor_create_plus_forward_ms,
        "batch_size": float(sample_count),
        "batch_ms": batch_ms,
        "batch_per_sequence_ms": batch_ms / max(1, sample_count),
        "window_seconds": x.shape[1] * 0.04,
    }


def train_one(
    architecture: str,
    split: dict[str, Any],
    scaler: Any,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    set_seed(args.seed)
    model = FinalSequenceBaseline(
        architecture,
        len(FEATURE_COLUMNS),
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
    best_epoch = 0
    best_f1 = -1.0
    stale_epochs = 0
    history: list[dict[str, float]] = []
    train_started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        count = 0
        for sequences, labels in loader:
            sequences = sequences.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(sequences), labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.item()) * len(labels)
            count += len(labels)

        validation_scores = predict(model, split["x_validation"], args.batch_size, device)
        _, validation_metrics = best_threshold(split["y_validation"], validation_scores)
        epoch_item = {
            "epoch": float(epoch),
            "loss": total_loss / max(1, count),
            "validation_f1": float(validation_metrics["f1"]),
            "validation_accuracy": float(validation_metrics["accuracy"]),
        }
        history.append(epoch_item)
        print(
            f"final_imu/{architecture} epoch={epoch:03d} "
            f"loss={epoch_item['loss']:.6f} val_f1={epoch_item['validation_f1']:.4f}"
        )

        if validation_metrics["f1"] > best_f1:
            best_f1 = float(validation_metrics["f1"])
            best_epoch = epoch
            stale_epochs = 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale_epochs += 1
        if epoch >= args.min_epochs and stale_epochs >= args.early_stopping_patience:
            print(f"final_imu/{architecture} early_stop epoch={epoch:03d} best_epoch={best_epoch:03d}")
            break

    synchronize(device)
    train_seconds = time.perf_counter() - train_started
    if best_state is not None:
        model.load_state_dict(best_state)

    validation_scores = predict(model, split["x_validation"], args.batch_size, device)
    threshold, validation_metrics = best_threshold(split["y_validation"], validation_scores)
    test_scores = predict(model, split["x_test"], args.batch_size, device)
    test_metrics = metrics(split["y_test"], test_scores, threshold)
    latency = measure_latency(model, split["x_test"], args.batch_size, device, args.latency_repeats)

    model_path = args.out_dir / f"final_{architecture}_imu_fall.pt"
    metadata_path = args.out_dir / f"final_{architecture}_imu_fall.json"
    checkpoint = {
        "model_type": f"final_binary_imu_fall_{architecture}",
        "task": "imu_fall",
        "positive_label": "fall",
        "architecture": architecture,
        "feature_columns": FEATURE_COLUMNS,
        "sequence_length": args.sequence_length,
        "sequence_stride": args.sequence_stride,
        "sample_ms": 40,
        "sample_rate_hz": 25,
        "threshold": threshold,
        "scaler_center": scaler.center,
        "scaler_scale": scaler.scale,
        "hidden_size": args.hidden_size,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "transformer_heads": args.transformer_heads,
        "model_state": model.state_dict(),
    }
    torch.save(checkpoint, model_path)

    metadata = {key: value for key, value in checkpoint.items() if key not in {"model_state", "scaler_center", "scaler_scale"}}
    metadata.update(
        {
            "source": str(args.source),
            "device": str(device),
            "split_method": "Same as final LSTM: SisFall source-file/group hash split; ICCAS chronological split inside each scenario",
            "split_sizes": {
                "train": int(len(split["y_train"])),
                "validation": int(len(split["y_validation"])),
                "test": int(len(split["y_test"])),
            },
            "label_counts": {
                "train_positive": int(split["y_train"].sum()),
                "train_negative": int(len(split["y_train"]) - split["y_train"].sum()),
                "validation_positive": int(split["y_validation"].sum()),
                "validation_negative": int(len(split["y_validation"]) - split["y_validation"].sum()),
                "test_positive": int(split["y_test"].sum()),
                "test_negative": int(len(split["y_test"]) - split["y_test"].sum()),
            },
            "scaler_center": scaler.center.tolist(),
            "scaler_scale": scaler.scale.tolist(),
            "best_epoch": best_epoch,
            "epochs_trained": len(history),
            "train_seconds": train_seconds,
            "validation_metrics": validation_metrics,
            "test_metrics": test_metrics,
            "test_metrics_by_dataset": grouped_metrics(split["y_test"], test_scores, split["meta_test"], threshold, "source_dataset"),
            "test_metrics_by_activity": grouped_metrics(split["y_test"], test_scores, split["meta_test"], threshold, "source_activity"),
            "latency_cpu_or_requested_device": latency,
            "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
            "history": history,
            "note": "This is a fair final-data baseline trained with the same input CSV, feature columns, sequence settings, split policy, and scaler policy as the final LSTM.",
        }
    )
    metadata_path.write_text(json.dumps(json_ready(metadata), ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "architecture": architecture,
        "model_path": str(model_path),
        "metadata_path": str(metadata_path),
        "best_epoch": best_epoch,
        "epochs_trained": len(history),
        "train_seconds": train_seconds,
        "threshold": threshold,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "test_metrics_by_dataset": metadata["test_metrics_by_dataset"],
        "latency": latency,
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
    }


def write_csv(results: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "architecture",
                "accuracy",
                "precision",
                "recall",
                "f1",
                "threshold",
                "forward_only_ms",
                "tensor_create_plus_forward_ms",
                "batch_per_sequence_ms",
                "window_seconds",
                "train_seconds",
                "best_epoch",
                "epochs_trained",
                "parameter_count",
            ]
        )
        for result in results:
            test = result["test_metrics"]
            latency = result["latency"]
            writer.writerow(
                [
                    result["architecture"],
                    f"{test['accuracy']:.6f}",
                    f"{test['precision']:.6f}",
                    f"{test['recall']:.6f}",
                    f"{test['f1']:.6f}",
                    f"{result['threshold']:.4f}",
                    f"{latency['forward_only_ms']:.6f}",
                    f"{latency['tensor_create_plus_forward_ms']:.6f}",
                    f"{latency['batch_per_sequence_ms']:.6f}",
                    f"{latency['window_seconds']:.3f}",
                    f"{result['train_seconds']:.3f}",
                    result["best_epoch"],
                    result["epochs_trained"],
                    result["parameter_count"],
                ]
            )


def write_readme(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Final IMU RNN and Transformer Baselines",
        "",
        "This experiment retrains RNN and Transformer models with the same data protocol used by the final IMU LSTM.",
        "",
        "## Protocol",
        "",
        f"- Source CSV: `{report['source']}`",
        f"- Device: `{report['device']}`",
        f"- Features: `{', '.join(FEATURE_COLUMNS)}`",
        f"- Sequence length: `{report['sequence_length']}` samples = `2.0 seconds` at 25 Hz",
        f"- Sequence stride: `{report['sequence_stride']}`",
        "- Split: SisFall group/hash split and ICCAS chronological split, matching the final LSTM script",
        "- Scaling: Robust median/IQR scaler fitted on train only, then clipped to [-12, 12]",
        "- Early stopping: minimum 15 epochs, then stop after validation F1 does not improve",
        "",
        "## Test Results",
        "",
        "| Model | Accuracy | Precision | Recall | F1-score | Forward ms | Process ms | Train sec | Best epoch |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for result in report["results"]:
        test = result["test_metrics"]
        latency = result["latency"]
        lines.append(
            f"| {result['architecture'].upper()} | {test['accuracy']:.4f} | {test['precision']:.4f} | "
            f"{test['recall']:.4f} | {test['f1']:.4f} | {latency['forward_only_ms']:.3f} | "
            f"{latency['tensor_create_plus_forward_ms']:.3f} | {result['train_seconds']:.1f} | {result['best_epoch']} |"
        )
    lines += [
        "",
        "Forward ms is model execution only for one 50-sample sequence. Process ms includes NumPy-to-tensor creation plus model execution, but not the 2-second sensor acquisition window, API, or database time.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.device == "cpu":
        torch.set_num_threads(args.cpu_threads)
    set_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    frame = load_merged_csv(args.source)
    split = make_sequences(frame, args.sequence_length, args.sequence_stride, args.train_ratio, args.validation_ratio, args.seed)
    split, scaler = scale_split(split)
    device = resolve_device(args.device)
    print(f"training_device={device}")
    results = [train_one(model_name, split, scaler, args, device) for model_name in args.models]
    report = {
        "source": str(args.source),
        "device": str(device),
        "sequence_length": args.sequence_length,
        "sequence_stride": args.sequence_stride,
        "train_ratio": args.train_ratio,
        "validation_ratio": args.validation_ratio,
        "seed": args.seed,
        "results": results,
    }
    (args.out_dir / "metrics.json").write_text(json.dumps(json_ready(report), ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(results, args.out_dir / "metrics.csv")
    write_readme(report, args.out_dir / "README.md")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("../data/iccas_sensor_lstm/final_iccas_sisfall_imu_merged.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("experiments/final_sequence_baselines"))
    parser.add_argument("--models", nargs="+", choices=["rnn", "transformer"], default=["rnn", "transformer"])
    parser.add_argument("--sequence-length", type=int, default=50)
    parser.add_argument("--sequence-stride", type=int, default=4)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--transformer-heads", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--min-epochs", type=int, default=15)
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--validation-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="cpu")
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--latency-repeats", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    report = run(parse_args())
    print(json.dumps(json_ready(report), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
