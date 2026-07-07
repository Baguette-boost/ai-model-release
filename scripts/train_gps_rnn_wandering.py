"""Train the final GPS wandering RNN model."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from train_parallel_sensor_lstm import GPS_FEATURES
from train_specialized_sensor_lstm import best_threshold, load_frames, make_sequences, metrics, scale_split


class SequenceDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[index], self.y[index]


class BinaryRNN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.rnn(x)
        return self.head(output[:, -1, :]).squeeze(-1)


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


def predict(model: nn.Module, x: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    loader = DataLoader(SequenceDataset(x, np.zeros(len(x), dtype=np.float32)), batch_size=batch_size)
    chunks: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for sequences, _ in loader:
            chunks.append(torch.sigmoid(model(sequences.to(device))).detach().cpu().numpy())
    return np.concatenate(chunks)


def scenario_metrics(y_true: np.ndarray, scores: np.ndarray, scenarios: list[str], threshold: float) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for scenario in sorted(set(scenarios)):
        idx = np.array([item == scenario for item in scenarios], dtype=bool)
        out[scenario] = metrics(y_true[idx], scores[idx], threshold)
    return out


def train(args: argparse.Namespace) -> dict[str, Any]:
    set_seed(args.seed)
    frames = load_frames(args.source)
    split = make_sequences(frames, GPS_FEATURES, "wandering", args.sequence_length, args.train_ratio, args.validation_ratio)
    split, scaler = scale_split(split)
    device = resolve_device(args.device)
    print(f"training_device={device}")
    model = BinaryRNN(len(GPS_FEATURES), args.hidden_size, args.num_layers, args.dropout).to(device)
    pos = float(split["y_train"].sum())
    neg = float(len(split["y_train"]) - pos)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    loader = DataLoader(SequenceDataset(split["x_train"], split["y_train"]), batch_size=args.batch_size, shuffle=True)
    best_state: dict[str, torch.Tensor] | None = None
    best_f1 = -1.0
    history: list[dict[str, float]] = []
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
        validation_scores = predict(model, split["x_val"], args.batch_size, device)
        _, validation_metrics = best_threshold(split["y_val"], validation_scores)
        history.append({"epoch": epoch, "loss": total / max(1, count), "validation_f1": validation_metrics["f1"]})
        print(f"gps_rnn epoch={epoch:03d} loss={total / max(1, count):.6f} val_f1={validation_metrics['f1']:.4f}")
        if validation_metrics["f1"] > best_f1:
            best_f1 = validation_metrics["f1"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    validation_scores = predict(model, split["x_val"], args.batch_size, device)
    threshold, validation_metrics = best_threshold(split["y_val"], validation_scores)
    test_scores = predict(model, split["x_test"], args.batch_size, device)
    test_metrics = metrics(split["y_test"], test_scores, threshold)
    by_scenario = scenario_metrics(split["y_test"], test_scores, split["s_test"], threshold)

    args.model_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.model_dir / "iccas_final_rnn_gps_wandering.pt"
    metadata_path = args.model_dir / "iccas_final_rnn_gps_wandering.json"
    checkpoint = {
        "model_type": "iccas_specialized_binary_rnn",
        "task": "gps_wandering",
        "positive_label": "wandering",
        "feature_columns": GPS_FEATURES,
        "sequence_length": args.sequence_length,
        "threshold": threshold,
        "scaler_center": scaler.center,
        "scaler_scale": scaler.scale,
        "hidden_size": args.hidden_size,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "model_state": model.state_dict(),
    }
    torch.save(checkpoint, model_path)
    metadata = {key: value for key, value in checkpoint.items() if key not in {"model_state", "scaler_center", "scaler_scale"}}
    metadata["scaler_center"] = scaler.center.tolist()
    metadata["scaler_scale"] = scaler.scale.tolist()
    metadata["source"] = str(args.source)
    metadata["device"] = str(device)
    metadata["split_sizes"] = {"train": len(split["y_train"]), "validation": len(split["y_val"]), "test": len(split["y_test"])}
    metadata["validation_metrics"] = validation_metrics
    metadata["test_metrics"] = test_metrics
    metadata["test_metrics_by_scenario"] = by_scenario
    metadata["history"] = history
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "model_path": str(model_path),
        "metadata_path": str(metadata_path),
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "split_sizes": metadata["split_sizes"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("../ICCAS_final_data.xlsx"))
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
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
    return parser.parse_args()


def main() -> None:
    print(json.dumps(train(parse_args()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
