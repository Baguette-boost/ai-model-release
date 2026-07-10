"""Train V3 TCN and CNN-GRU IMU fall models in an isolated folder."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "scripts"))

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


class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int) -> None:
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp_size == 0:
            return x
        return x[:, :, : -self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.network = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.network(x) + self.downsample(x))


class TCNFallModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, kernel_size: int, dropout: float) -> None:
        super().__init__()
        channels = [hidden_size] * num_layers
        blocks = []
        in_channels = input_size
        for index, out_channels in enumerate(channels):
            blocks.append(TemporalBlock(in_channels, out_channels, kernel_size, dilation=2**index, dropout=dropout))
            in_channels = out_channels
        self.encoder = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x.transpose(1, 2)).transpose(1, 2)
        pooled = encoded.mean(dim=1)
        return self.head(pooled).squeeze(-1)


class CNNGRUFallModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, kernel_size: int, dropout: float) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.cnn = nn.Sequential(
            nn.Conv1d(input_size, hidden_size, kernel_size, padding=padding),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_size, hidden_size, kernel_size, padding=padding),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.gru = nn.GRU(
            input_size=hidden_size,
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.cnn(x.transpose(1, 2)).transpose(1, 2)
        output, _ = self.gru(features)
        pooled = output[:, -1, :]
        return self.head(pooled).squeeze(-1)


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


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def build_model(name: str, args: argparse.Namespace) -> nn.Module:
    if name == "tcn":
        return TCNFallModel(len(FEATURE_COLUMNS), args.hidden_size, args.num_layers, args.kernel_size, args.dropout)
    if name == "cnn_gru":
        return CNNGRUFallModel(len(FEATURE_COLUMNS), args.hidden_size, args.num_layers, args.kernel_size, args.dropout)
    raise ValueError(f"Unsupported model: {name}")


def measure_latency(model: nn.Module, x: np.ndarray, batch_size: int, device: torch.device, repeats: int) -> dict[str, float]:
    sample_count = min(len(x), max(batch_size, 1024))
    np_single = x[:1].astype(np.float32)
    single = torch.tensor(np_single, dtype=torch.float32, device=device)
    batch = torch.tensor(x[:sample_count], dtype=torch.float32, device=device)
    model.eval()
    with torch.no_grad():
        for _ in range(30):
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
        tensor_forward_ms = (time.perf_counter() - started) * 1000.0 / max(1, repeats)

        started = time.perf_counter()
        for _ in range(repeats):
            model(batch)
        synchronize(device)
        batch_ms = (time.perf_counter() - started) * 1000.0 / max(1, repeats)

    return {
        "forward_only_ms": forward_only_ms,
        "tensor_create_plus_forward_ms": tensor_forward_ms,
        "batch_size": float(sample_count),
        "batch_ms": batch_ms,
        "batch_per_sequence_ms": batch_ms / max(1, sample_count),
        "window_seconds": x.shape[1] * 0.04,
    }


def train_one(name: str, split: dict[str, Any], scaler: Any, args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    print(f"V3/{name} start_training", flush=True)
    set_seed(args.seed)
    model = build_model(name, args).to(device)
    pos = float(split["y_train"].sum())
    neg = float(len(split["y_train"]) - pos)
    base_pos_weight = neg / max(pos, 1.0)
    effective_pos_weight = base_pos_weight * args.pos_weight_multiplier
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([effective_pos_weight], dtype=torch.float32, device=device)
    )
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
        history.append(
            {
                "epoch": float(epoch),
                "loss": total_loss / max(1, count),
                "validation_f1": float(validation_metrics["f1"]),
                "validation_accuracy": float(validation_metrics["accuracy"]),
            }
        )
        print(f"V3/{name} epoch={epoch:03d} loss={history[-1]['loss']:.6f} val_f1={history[-1]['validation_f1']:.4f}", flush=True)

        if validation_metrics["f1"] > best_f1:
            best_f1 = float(validation_metrics["f1"])
            best_epoch = epoch
            stale_epochs = 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale_epochs += 1
        if epoch >= args.min_epochs and stale_epochs >= args.early_stopping_patience:
            print(f"V3/{name} early_stop epoch={epoch:03d} best_epoch={best_epoch:03d}", flush=True)
            break

    synchronize(device)
    train_seconds = time.perf_counter() - train_started
    if best_state is not None:
        model.load_state_dict(best_state)

    validation_scores = predict(model, split["x_validation"], args.batch_size, device)
    threshold, validation_metrics = best_threshold(split["y_validation"], validation_scores)
    print(f"V3/{name} evaluating_test", flush=True)
    test_scores = predict(model, split["x_test"], args.batch_size, device)
    test_metrics = metrics(split["y_test"], test_scores, threshold)
    print(f"V3/{name} measuring_latency repeats={args.latency_repeats}", flush=True)
    latency = measure_latency(model, split["x_test"], args.batch_size, device, args.latency_repeats)
    parameter_count = int(sum(parameter.numel() for parameter in model.parameters()))

    model_path = args.model_dir / f"v3_{name}_imu_fall.pt"
    metadata_path = args.model_dir / f"v3_{name}_imu_fall.json"
    checkpoint = {
        "model_type": f"v3_binary_imu_fall_{name}",
        "task": "imu_fall",
        "positive_label": "fall",
        "architecture": name,
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
        "kernel_size": args.kernel_size,
        "dropout": args.dropout,
        "base_pos_weight": base_pos_weight,
        "pos_weight_multiplier": args.pos_weight_multiplier,
        "effective_pos_weight": effective_pos_weight,
        "model_state": model.state_dict(),
    }
    torch.save(checkpoint, model_path)

    metadata = {key: value for key, value in checkpoint.items() if key not in {"model_state", "scaler_center", "scaler_scale"}}
    metadata.update(
        {
            "source": str(args.source),
            "device": str(device),
            "split_method": "Same final protocol: SisFall group/hash split; ICCAS chronological split inside each scenario",
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
            "latency": latency,
            "parameter_count": parameter_count,
            "history": history,
        }
    )
    metadata_path.write_text(json.dumps(json_ready(metadata), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"V3/{name} saved_model={model_path}", flush=True)

    return {
        "architecture": name,
        "model_path": str(model_path),
        "metadata_path": str(metadata_path),
        "best_epoch": best_epoch,
        "epochs_trained": len(history),
        "train_seconds": train_seconds,
            "threshold": threshold,
            "base_pos_weight": base_pos_weight,
            "pos_weight_multiplier": args.pos_weight_multiplier,
            "effective_pos_weight": effective_pos_weight,
            "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "test_metrics_by_dataset": metadata["test_metrics_by_dataset"],
        "latency": latency,
        "parameter_count": parameter_count,
    }


def read_reference_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    final_baseline = root / "experiments" / "final_sequence_baselines" / "metrics.csv"
    if final_baseline.exists():
        with final_baseline.open("r", encoding="utf-8", newline="") as file:
            for row in csv.DictReader(file):
                rows.append(
                    {
                        "architecture": row["architecture"],
                        "accuracy": float(row["accuracy"]),
                        "precision": float(row["precision"]),
                        "recall": float(row["recall"]),
                        "f1": float(row["f1"]),
                        "forward_only_ms": float(row["forward_only_ms"]),
                        "tensor_create_plus_forward_ms": float(row["tensor_create_plus_forward_ms"]),
                        "train_seconds": float(row["train_seconds"]),
                        "best_epoch": int(row["best_epoch"]),
                        "source": "previous_final_baseline",
                    }
                )

    final_lstm = root / "models" / "iccas_final_hybrid_lstm_imu_fall.json"
    final_speed = root.parent / "data" / "iccas_sensor_lstm" / "imu_lstm_speed_preprocessing_metrics.json"
    if final_lstm.exists() and final_speed.exists():
        checkpoint = json.loads(final_lstm.read_text(encoding="utf-8"))
        speed = json.loads(final_speed.read_text(encoding="utf-8"))["inference_speed"]
        item = checkpoint["test_metrics"]["lstm_only"]
        rows.append(
            {
                "architecture": "final_lstm",
                "accuracy": float(item["accuracy"]),
                "precision": float(item["precision"]),
                "recall": float(item["recall"]),
                "f1": float(item["f1"]),
                "forward_only_ms": float(speed["realtime_forward_only_median_ms"]),
                "tensor_create_plus_forward_ms": float(speed["realtime_tensor_forward_median_ms"]),
                "train_seconds": None,
                "best_epoch": 10,
                "source": "previous_final_lstm",
            }
        )
    return rows


def write_metrics_csv(report: dict[str, Any], path: Path) -> None:
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
        for result in report["results"]:
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


def write_single_result_csv(result: dict[str, Any], path: Path) -> None:
    test = result["test_metrics"]
    latency = result["latency"]
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


def write_comparison_csv(report: dict[str, Any], path: Path) -> None:
    rows = []
    for item in report["reference_results"]:
        rows.append(item)
    for result in report["results"]:
        test = result["test_metrics"]
        latency = result["latency"]
        rows.append(
            {
                "architecture": result["architecture"],
                "accuracy": test["accuracy"],
                "precision": test["precision"],
                "recall": test["recall"],
                "f1": test["f1"],
                "forward_only_ms": latency["forward_only_ms"],
                "tensor_create_plus_forward_ms": latency["tensor_create_plus_forward_ms"],
                "train_seconds": result["train_seconds"],
                "best_epoch": result["best_epoch"],
                "source": "V3_new_training",
            }
        )

    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "architecture",
                "accuracy",
                "precision",
                "recall",
                "f1",
                "forward_only_ms",
                "tensor_create_plus_forward_ms",
                "train_seconds",
                "best_epoch",
                "source",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_readme(report: dict[str, Any], path: Path) -> None:
    rows = []
    for result in report["results"]:
        test = result["test_metrics"]
        latency = result["latency"]
        rows.append(
            f"| {result['architecture'].upper()} | {test['accuracy']:.4f} | {test['precision']:.4f} | "
            f"{test['recall']:.4f} | {test['f1']:.4f} | {latency['forward_only_ms']:.3f} | "
            f"{latency['tensor_create_plus_forward_ms']:.3f} | {result['train_seconds']:.1f} | {result['best_epoch']} |"
        )
    text = [
        "# V3 IMU Fall Detection Experiments",
        "",
        "V3 is isolated from the earlier mixed experiment folders. It contains its own dataset copy, model checkpoints, metrics, and comparison files.",
        "",
        "## Dataset",
        "",
        f"- CSV: `{report['source']}`",
        f"- Rows are read from the V3 dataset copy.",
        f"- Features: `{', '.join(FEATURE_COLUMNS)}`",
        "- Preprocessing: accel_norm, gyro_norm, dt_s, 50-sample windows, train-only robust median/IQR scaling, clipping to [-12, 12].",
        "",
        "## New V3 Results",
        "",
        "| Model | Accuracy | Precision | Recall | F1-score | Forward ms | Process ms | Train sec | Best epoch |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        *rows,
        "",
        "Forward ms is neural-network execution for one 2-second IMU sequence. Process ms includes tensor creation plus neural-network execution, not sensor collection, API, or database time.",
        "",
        "## Files",
        "",
        "- `dataset/final_iccas_sisfall_imu_merged.csv`: V3 dataset copy",
        "- `models/v3_tcn_imu_fall.pt`: TCN checkpoint",
        "- `models/v3_cnn_gru_imu_fall.pt`: CNN-GRU checkpoint",
        "- `results/v3_metrics.csv`: TCN/CNN-GRU metrics",
        "- `results/v3_comparison_all_models.csv`: previous final RNN/LSTM/Transformer plus new V3 TCN/CNN-GRU",
    ]
    path.write_text("\n".join(text), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.device == "cpu":
        torch.set_num_threads(args.cpu_threads)
    set_seed(args.seed)
    args.model_dir.mkdir(parents=True, exist_ok=True)
    args.result_dir.mkdir(parents=True, exist_ok=True)

    frame = load_merged_csv(args.source)
    split = make_sequences(frame, args.sequence_length, args.sequence_stride, args.train_ratio, args.validation_ratio, args.seed)
    split, scaler = scale_split(split)
    device = resolve_device(args.device)
    print(f"V3 training_device={device}", flush=True)
    results = []
    for model_name in args.models:
        result = train_one(model_name, split, scaler, args, device)
        results.append(result)
        single_json = args.result_dir / f"v3_{model_name}_metrics.json"
        single_csv = args.result_dir / f"v3_{model_name}_metrics.csv"
        single_json.write_text(json.dumps(json_ready(result), ensure_ascii=False, indent=2), encoding="utf-8")
        write_single_result_csv(result, single_csv)
        print(f"V3/{model_name} saved_metrics={single_csv}", flush=True)
    report = {
        "source": str(args.source),
        "device": str(device),
        "sequence_length": args.sequence_length,
        "sequence_stride": args.sequence_stride,
        "train_ratio": args.train_ratio,
        "validation_ratio": args.validation_ratio,
        "seed": args.seed,
        "results": results,
        "reference_results": read_reference_rows(ROOT),
    }
    (args.result_dir / "v3_metrics.json").write_text(json.dumps(json_ready(report), ensure_ascii=False, indent=2), encoding="utf-8")
    write_metrics_csv(report, args.result_dir / "v3_metrics.csv")
    write_comparison_csv(report, args.result_dir / "v3_comparison_all_models.csv")
    write_readme(report, ROOT / "V3" / "README.md")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "V3" / "dataset" / "final_iccas_sisfall_imu_merged.csv")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "V3" / "models")
    parser.add_argument("--result-dir", type=Path, default=ROOT / "V3" / "results")
    parser.add_argument("--models", nargs="+", choices=["tcn", "cnn_gru"], default=["tcn", "cnn_gru"])
    parser.add_argument("--sequence-length", type=int, default=50)
    parser.add_argument("--sequence-stride", type=int, default=4)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--kernel-size", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--min-epochs", type=int, default=15)
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--pos-weight-multiplier", type=float, default=1.0)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--validation-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="cpu")
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--latency-repeats", type=int, default=200)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(json_ready(run(parse_args())), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
