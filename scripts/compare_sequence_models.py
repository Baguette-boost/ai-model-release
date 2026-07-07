"""Train and compare RNN, GRU, LSTM, and Transformer binary sensor models."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from train_parallel_sensor_lstm import GPS_FEATURES, IMU_GYRO_FEATURES
from train_specialized_sensor_lstm import (
    best_threshold,
    load_frames,
    make_sequences,
    metrics,
    scale_split,
)


class SequenceDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[index], self.y[index]


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


class BinarySequenceModel(nn.Module):
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
        if architecture in {"rnn", "gru", "lstm"}:
            recurrent_cls = {"rnn": nn.RNN, "gru": nn.GRU, "lstm": nn.LSTM}[architecture]
            self.encoder = recurrent_cls(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0.0,
                batch_first=True,
            )
            self.head = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, 1))
        elif architecture == "transformer":
            self.input_projection = nn.Linear(input_size, hidden_size)
            self.position = PositionalEncoding(hidden_size)
            layer = nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=transformer_heads,
                dim_feedforward=hidden_size * 2,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
            self.head = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, 1))
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.architecture in {"rnn", "gru", "lstm"}:
            output, _ = self.encoder(x)
            pooled = output[:, -1, :]
        else:
            projected = self.position(self.input_projection(x))
            output = self.encoder(projected)
            pooled = output.mean(dim=1)
        return self.head(pooled).squeeze(-1)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(requested)


def predict_scores(model: nn.Module, x: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    loader = DataLoader(SequenceDataset(x, np.zeros(len(x), dtype=np.float32)), batch_size=batch_size)
    chunks: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for sequences, _ in loader:
            logits = model(sequences.to(device))
            chunks.append(torch.sigmoid(logits).detach().cpu().numpy())
    return np.concatenate(chunks)


def measure_latency(
    model: nn.Module,
    x: np.ndarray,
    batch_size: int,
    device: torch.device,
    warmup: int,
    repeats: int,
) -> dict[str, float]:
    sample_count = min(len(x), max(batch_size, 1024))
    sample = torch.tensor(x[:sample_count], dtype=torch.float32, device=device)
    single = sample[:1]
    model.eval()

    def synchronize() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps":
            torch.mps.synchronize()

    with torch.no_grad():
        for _ in range(warmup):
            model(sample[:batch_size])
            model(single)
        synchronize()

        start = time.perf_counter()
        for _ in range(repeats):
            model(single)
        synchronize()
        single_total = time.perf_counter() - start

        start = time.perf_counter()
        for _ in range(repeats):
            model(sample)
        synchronize()
        batch_total = time.perf_counter() - start

    single_ms = single_total * 1000.0 / max(1, repeats)
    batch_ms = batch_total * 1000.0 / max(1, repeats)
    per_sequence_ms = batch_ms / max(1, sample_count)
    return {
        "single_sequence_ms": single_ms,
        "batch_size": float(sample_count),
        "batch_ms": batch_ms,
        "batch_per_sequence_ms": per_sequence_ms,
    }


def parameter_count(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


def train_and_evaluate(
    architecture: str,
    task_name: str,
    positive_label: str,
    feature_columns: list[str],
    split: dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    set_seed(args.seed)
    model = BinarySequenceModel(
        architecture=architecture,
        input_size=len(feature_columns),
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        transformer_heads=args.transformer_heads,
    ).to(device)
    pos = float(split["y_train"].sum())
    neg = float(len(split["y_train"]) - pos)
    pos_weight = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    train_loader = DataLoader(
        SequenceDataset(split["x_train"], split["y_train"]),
        batch_size=args.batch_size,
        shuffle=True,
    )

    best_state: dict[str, torch.Tensor] | None = None
    best_validation_f1 = -1.0
    history: list[dict[str, float]] = []
    train_start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        item_count = 0
        for sequences, labels in train_loader:
            sequences = sequences.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(sequences), labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.item()) * len(labels)
            item_count += len(labels)

        val_scores = predict_scores(model, split["x_val"], args.batch_size, device)
        _, val_metrics = best_threshold(split["y_val"], val_scores)
        epoch_result = {
            "epoch": float(epoch),
            "loss": total_loss / max(1, item_count),
            "validation_f1": float(val_metrics["f1"]),
            "validation_accuracy": float(val_metrics["accuracy"]),
        }
        history.append(epoch_result)
        print(
            f"{task_name}/{architecture} epoch={epoch:03d} "
            f"loss={epoch_result['loss']:.6f} val_acc={epoch_result['validation_accuracy']:.4f} "
            f"val_f1={epoch_result['validation_f1']:.4f}"
        )
        if val_metrics["f1"] > best_validation_f1:
            best_validation_f1 = float(val_metrics["f1"])
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()
    train_seconds = time.perf_counter() - train_start

    if best_state is not None:
        model.load_state_dict(best_state)
    val_scores = predict_scores(model, split["x_val"], args.batch_size, device)
    threshold, validation_metrics = best_threshold(split["y_val"], val_scores)
    test_scores = predict_scores(model, split["x_test"], args.batch_size, device)
    test_metrics = metrics(split["y_test"], test_scores, threshold)
    latency = measure_latency(model, split["x_test"], args.batch_size, device, args.latency_warmup, args.latency_repeats)
    return {
        "task": task_name,
        "architecture": architecture,
        "positive_label": positive_label,
        "feature_count": len(feature_columns),
        "sequence_length": args.sequence_length,
        "parameter_count": parameter_count(model),
        "train_seconds": train_seconds,
        "threshold": threshold,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "latency": latency,
        "history": history,
    }


def make_task_split(
    frames: list[Any],
    task_name: str,
    positive_label: str,
    feature_columns: list[str],
    args: argparse.Namespace,
) -> dict[str, Any]:
    split = make_sequences(
        frames,
        feature_columns,
        positive_label,
        args.sequence_length,
        args.train_ratio,
        args.validation_ratio,
    )
    split, _ = scale_split(split)
    print(
        f"{task_name} split: train={len(split['y_train'])}, "
        f"validation={len(split['y_val'])}, test={len(split['y_test'])}, "
        f"positive_test={int(split['y_test'].sum())}"
    )
    return split


def write_csv(results: list[dict[str, Any]], csv_path: Path) -> None:
    lines = [
        "task,architecture,accuracy,precision,recall,f1,threshold,train_seconds,"
        "single_sequence_ms,batch_per_sequence_ms,parameter_count"
    ]
    for result in results:
        test = result["test_metrics"]
        latency = result["latency"]
        lines.append(
            ",".join(
                [
                    result["task"],
                    result["architecture"],
                    f"{test['accuracy']:.6f}",
                    f"{test['precision']:.6f}",
                    f"{test['recall']:.6f}",
                    f"{test['f1']:.6f}",
                    f"{result['threshold']:.4f}",
                    f"{result['train_seconds']:.3f}",
                    f"{latency['single_sequence_ms']:.6f}",
                    f"{latency['batch_per_sequence_ms']:.6f}",
                    str(result["parameter_count"]),
                ]
            )
        )
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_markdown(report: dict[str, Any], markdown_path: Path) -> None:
    results = report["results"]
    lines = [
        "# RNN, GRU, LSTM, Transformer 실제 학습 비교 결과",
        "",
        "## 실험 조건",
        "",
        f"- Source: `{report['source']}`",
        f"- Device: `{report['device']}`",
        f"- Sequence length: `{report['sequence_length']}`",
        f"- Epochs: `{report['epochs']}`",
        f"- Batch size: `{report['batch_size']}`",
        f"- Split: scenario별 chronological split, train {report['train_ratio']:.2f}, validation {report['validation_ratio']:.2f}, test remainder",
        "",
        "## 전체 비교",
        "",
        "| Task | Model | Accuracy | Precision | Recall | F1-score | Single inference ms | Batch per sequence ms | Train sec | Params |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for result in results:
        test = result["test_metrics"]
        latency = result["latency"]
        lines.append(
            f"| {result['task']} | {result['architecture'].upper()} | "
            f"{test['accuracy']:.4f} | {test['precision']:.4f} | {test['recall']:.4f} | {test['f1']:.4f} | "
            f"{latency['single_sequence_ms']:.3f} | {latency['batch_per_sequence_ms']:.4f} | "
            f"{result['train_seconds']:.1f} | {result['parameter_count']} |"
        )

    lines.extend(
        [
            "",
            "## Task별 최고 F1",
            "",
        ]
    )
    for task in sorted({item["task"] for item in results}):
        task_results = [item for item in results if item["task"] == task]
        best = max(task_results, key=lambda item: item["test_metrics"]["f1"])
        lines.append(
            f"- {task}: `{best['architecture'].upper()}` "
            f"F1 {best['test_metrics']['f1']:.4f}, "
            f"Accuracy {best['test_metrics']['accuracy']:.4f}, "
            f"single inference {best['latency']['single_sequence_ms']:.3f} ms"
        )

    lines.extend(
        [
            "",
            "## 해석",
            "",
            "- Accuracy는 전체 정답률이고, F1-score는 위험 클래스의 Precision과 Recall 균형을 나타냅니다.",
            "- 낙상과 배회 감지는 위험 상황을 놓치지 않는 것이 중요하므로 Recall과 F1-score를 함께 봐야 합니다.",
            "- Single inference ms는 실시간으로 센서 포인트가 들어왔을 때 한 시퀀스를 추론하는 시간입니다.",
            "- Batch per sequence ms는 많은 시퀀스를 한꺼번에 평가할 때 시퀀스 1개당 평균 처리 시간입니다.",
            "",
        ]
    )
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("../ICCAS_final_data.xlsx"))
    parser.add_argument("--report", type=Path, default=Path("../data/iccas_sensor_lstm/model_architecture_comparison.json"))
    parser.add_argument("--csv", type=Path, default=Path("../data/iccas_sensor_lstm/model_architecture_comparison.csv"))
    parser.add_argument("--markdown", type=Path, default=Path("docs/MODEL_ARCHITECTURE_COMPARISON_RESULTS.md"))
    parser.add_argument("--sequence-length", type=int, default=16)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--validation-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--architectures", nargs="+", default=["rnn", "gru", "lstm", "transformer"])
    parser.add_argument("--tasks", nargs="+", choices=["gps_wandering", "imu_fall"], default=["gps_wandering", "imu_fall"])
    parser.add_argument("--transformer-heads", type=int, default=4)
    parser.add_argument("--latency-warmup", type=int, default=10)
    parser.add_argument("--latency-repeats", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    print(f"Using device: {device}")
    frames = load_frames(args.source)
    task_configs = {
        "gps_wandering": ("wandering", GPS_FEATURES),
        "imu_fall": ("fall", IMU_GYRO_FEATURES),
    }
    splits: dict[str, dict[str, Any]] = {}
    for task_name in args.tasks:
        positive_label, feature_columns = task_configs[task_name]
        splits[task_name] = make_task_split(frames, task_name, positive_label, feature_columns, args)

    results: list[dict[str, Any]] = []
    for task_name in args.tasks:
        positive_label, feature_columns = task_configs[task_name]
        for architecture in args.architectures:
            result = train_and_evaluate(
                architecture,
                task_name,
                positive_label,
                feature_columns,
                splits[task_name],
                args,
                device,
            )
            results.append(result)

    report = {
        "source": str(args.source),
        "device": str(device),
        "sequence_length": args.sequence_length,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "hidden_size": args.hidden_size,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "train_ratio": args.train_ratio,
        "validation_ratio": args.validation_ratio,
        "results": results,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(results, args.csv)
    write_markdown(report, args.markdown)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
