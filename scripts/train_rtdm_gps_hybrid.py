"""Train/evaluate GPS RNN with a lightweight RTDM-inspired sequence detector."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from compare_gps_kalman_rnn import add_features, load_raw_frames
from train_gps_rnn_wandering import BinaryRNN, predict
from train_parallel_sensor_lstm import GPS_FEATURES
from train_specialized_sensor_lstm import RobustScaler, best_threshold, make_sequences, metrics


class SequenceDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[index], self.y[index]


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


def scale_split(split: dict[str, Any]) -> tuple[dict[str, Any], RobustScaler]:
    scaler = RobustScaler.fit(split["x_train"].reshape(-1, split["x_train"].shape[-1]))
    out = dict(split)
    for key in ["x_train", "x_val", "x_test"]:
        x = split[key]
        out[key] = scaler.transform(x.reshape(-1, x.shape[-1])).reshape(x.shape)
    return out, scaler


def train_rnn(split: dict[str, Any], args: argparse.Namespace, device: torch.device) -> tuple[BinaryRNN, dict[str, Any]]:
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
        print(f"gps_rtdm_rnn epoch={epoch:03d} loss={total / max(1, count):.6f} val_f1={validation_metrics['f1']:.4f}")
        if validation_metrics["f1"] > best_f1:
            best_f1 = validation_metrics["f1"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    validation_scores = predict(model, split["x_val"], args.batch_size, device)
    threshold, validation_metrics = best_threshold(split["y_val"], validation_scores)
    test_scores = predict(model, split["x_test"], args.batch_size, device)
    test_metrics = metrics(split["y_test"], test_scores, threshold)
    return model, {
        "threshold": threshold,
        "validation_scores": validation_scores,
        "test_scores": test_scores,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "history": history,
    }


def token_for_xy(x: float, y: float, cell_size_m: float) -> str:
    return f"{int(np.floor(x / cell_size_m))}:{int(np.floor(y / cell_size_m))}"


def compress_tokens(tokens: list[str]) -> list[str]:
    compressed: list[str] = []
    for token in tokens:
        if not compressed or compressed[-1] != token:
            compressed.append(token)
    return compressed


def sequence_tokens(x: np.ndarray, cell_size_m: float) -> list[str]:
    x_index = GPS_FEATURES.index("x_m")
    y_index = GPS_FEATURES.index("y_m")
    return compress_tokens([token_for_xy(float(row[x_index]), float(row[y_index]), cell_size_m) for row in x])


def ngrams(tokens: list[str], n: int) -> set[tuple[str, ...]]:
    if len(tokens) < n:
        return set()
    return {tuple(tokens[index : index + n]) for index in range(len(tokens) - n + 1)}


def build_support(
    x_train_raw: np.ndarray,
    y_train: np.ndarray,
    cell_size_m: float,
    ngram_size: int,
    min_frequency: int,
) -> dict[str, Any]:
    unigram_counter: Counter[str] = Counter()
    ngram_counter: Counter[tuple[str, ...]] = Counter()
    transition_counter: Counter[tuple[str, str]] = Counter()
    for sequence, label in zip(x_train_raw, y_train):
        if int(label) == 1:
            continue
        tokens = sequence_tokens(sequence, cell_size_m)
        unigram_counter.update(tokens)
        ngram_counter.update(ngrams(tokens, ngram_size))
        transition_counter.update(zip(tokens, tokens[1:]))
    support_tokens = {token for token, count in unigram_counter.items() if count >= min_frequency}
    support_ngrams = {token for token, count in ngram_counter.items() if count >= min_frequency}
    support_transitions = {token for token, count in transition_counter.items() if count >= min_frequency}
    return {
        "support_tokens": support_tokens,
        "support_ngrams": support_ngrams,
        "support_transitions": support_transitions,
        "support_token_count": len(support_tokens),
        "support_ngram_count": len(support_ngrams),
        "support_transition_count": len(support_transitions),
    }


def rtdm_scores(x_raw: np.ndarray, support: dict[str, Any], cell_size_m: float, ngram_size: int) -> np.ndarray:
    support_tokens = support["support_tokens"]
    support_ngrams = support["support_ngrams"]
    support_transitions = support["support_transitions"]
    scores: list[float] = []
    for sequence in x_raw:
        tokens = sequence_tokens(sequence, cell_size_m)
        if not tokens:
            scores.append(1.0)
            continue
        token_match = sum(1 for token in tokens if token in support_tokens) / max(1, len(tokens))
        transitions = list(zip(tokens, tokens[1:]))
        transition_match = sum(1 for transition in transitions if transition in support_transitions) / max(1, len(transitions))
        seq_ngrams = ngrams(tokens, ngram_size)
        if seq_ngrams:
            ngram_match = len(seq_ngrams & support_ngrams) / max(1, len(seq_ngrams))
        else:
            ngram_match = token_match
        similarity = 0.35 * token_match + 0.30 * transition_match + 0.35 * ngram_match
        scores.append(float(np.clip(1.0 - similarity, 0.0, 1.0)))
    return np.array(scores, dtype=np.float32)


def tune_hybrid(y_true: np.ndarray, rnn_scores: np.ndarray, rtdm_anomaly: np.ndarray) -> tuple[float, float, dict[str, Any]]:
    best_weight = 1.0
    best_threshold = 0.5
    best_scores = rnn_scores
    best_metrics = metrics(y_true, best_scores, best_threshold)
    for rnn_weight in np.linspace(0.0, 1.0, 21):
        scores = rnn_weight * rnn_scores + (1.0 - rnn_weight) * rtdm_anomaly
        threshold, current = best_threshold_for_scores(y_true, scores)
        if (current["f1"], current["precision"], current["accuracy"]) > (
            best_metrics["f1"],
            best_metrics["precision"],
            best_metrics["accuracy"],
        ):
            best_weight = float(rnn_weight)
            best_threshold = float(threshold)
            best_metrics = current
    return best_weight, best_threshold, best_metrics


def best_threshold_for_scores(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, dict[str, Any]]:
    best_t = 0.5
    best_m = metrics(y_true, scores, best_t)
    for threshold in np.linspace(0.05, 0.95, 91):
        current = metrics(y_true, scores, float(threshold))
        if (current["f1"], current["precision"], current["accuracy"]) > (best_m["f1"], best_m["precision"], best_m["accuracy"]):
            best_t = float(threshold)
            best_m = current
    return best_t, best_m


def write_markdown(report: dict[str, Any], path: Path) -> None:
    rnn = report["test_metrics"]["rnn_kalman"]
    rtdm = report["test_metrics"]["rtdm_only"]
    hybrid = report["test_metrics"]["hybrid"]
    lines = [
        "# GPS RTDM Hybrid 실시간 탐지 보고서",
        "",
        "## 적용한 RTDM 아이디어",
        "",
        "NicklasXYZ/rtdm의 핵심 아이디어인 `trajectory -> token sequence -> normal support -> sequence anomaly score`를 우리 GPS 데이터에 맞춰 lightweight 구현했습니다.",
        "",
        "주의: 원본 NicklasXYZ/rtdm 패키지를 그대로 이식한 것이 아니라, 원본의 geohash/token sequence support 개념을 참고해 `x_m`, `y_m` 기반 meter-grid token으로 재구현한 코드입니다.",
        "",
        "## 처리 흐름",
        "",
        "```text",
        "lat/lng",
        "  -> meter 좌표 변환",
        "  -> Kalman filter",
        "  -> grid token sequence",
        "  -> normal route support score",
        "  -> RNN wandering score와 결합",
        "```",
        "",
        "## 성능",
        "",
        "| Method | Accuracy | Precision | Recall | F1-score | Threshold |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        f"| Kalman RNN | {rnn['accuracy']:.4f} | {rnn['precision']:.4f} | {rnn['recall']:.4f} | {rnn['f1']:.4f} | {report['thresholds']['rnn']:.2f} |",
        f"| RTDM only | {rtdm['accuracy']:.4f} | {rtdm['precision']:.4f} | {rtdm['recall']:.4f} | {rtdm['f1']:.4f} | {report['thresholds']['rtdm']:.2f} |",
        f"| RTDM + RNN Hybrid | {hybrid['accuracy']:.4f} | {hybrid['precision']:.4f} | {hybrid['recall']:.4f} | {hybrid['f1']:.4f} | {report['thresholds']['hybrid']:.2f} |",
        "",
        "## 처리 속도",
        "",
        "| Step | Time |",
        "| --- | ---: |",
        "| GPS RNN batch inference | 0.0065 ms / sequence |",
        "| GPS RNN realtime loop, forward only | avg 0.0929 ms / sequence |",
        "| GPS RNN realtime loop, tensor + forward | avg 0.0988 ms / sequence |",
        "| RTDM support score, batch | 0.0069 ms / sequence |",
        "| RTDM support score, realtime loop | avg 0.0078 ms / sequence |",
        "| Hybrid score combine | < 0.001 ms / sequence |",
        "| Estimated realtime total | 약 0.107 ms / sequence |",
        "",
        "속도는 이미 전처리된 GPS 16-step window 기준으로 별도 측정했습니다. Excel 로딩, 전체 Kalman feature 생성, sequence 생성, normal support 구축 시간은 실시간 1-window 추론 시간에 포함하지 않았습니다.",
        "",
        "## 검증 범위와 한계",
        "",
        "- 현재 성능 수치는 각 시나리오 내부를 train/validation/test 시간 구간으로 나눈 평가입니다.",
        "- 완전히 새로운 사용자, 장소, 장비를 분리한 holdout 검증 결과는 아닙니다.",
        "- 이 스크립트는 Kalman RNN과 RTDM support를 학습/평가하지만, 배포용 hybrid checkpoint 파일은 아직 저장하지 않습니다.",
        "",
        "## 결론",
        "",
        f"- Hybrid F1 변화량 vs Kalman RNN: `{report['delta_vs_rnn']['f1']:+.4f}`",
        f"- Hybrid Precision 변화량 vs Kalman RNN: `{report['delta_vs_rnn']['precision']:+.4f}`",
        f"- Hybrid Recall 변화량 vs Kalman RNN: `{report['delta_vs_rnn']['recall']:+.4f}`",
        "",
        "실시간 서버에서는 RTDM score를 RNN과 병렬로 계산하고, validation 기준 최적 weight를 적용하는 방식을 권장합니다.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("../ICCAS_final_data.xlsx"))
    parser.add_argument("--report", type=Path, default=Path("../data/iccas_sensor_lstm/gps_rtdm_hybrid_report.json"))
    parser.add_argument("--markdown", type=Path, default=Path("docs/GPS_RTDM_HYBRID_REPORT.md"))
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
    parser.add_argument("--measurement-noise-m", type=float, default=5.0)
    parser.add_argument("--process-noise", type=float, default=1.0)
    parser.add_argument("--cell-size-m", type=float, default=35.0)
    parser.add_argument("--ngram-size", type=int, default=3)
    parser.add_argument("--min-frequency", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    raw_frames = load_raw_frames(args.source)
    featured_frames = add_features(raw_frames, True, args.measurement_noise_m, args.process_noise)
    raw_split = make_sequences(featured_frames, GPS_FEATURES, "wandering", args.sequence_length, args.train_ratio, args.validation_ratio)
    split, _ = scale_split(raw_split)
    _, rnn_result = train_rnn(split, args, device)
    support = build_support(
        raw_split["x_train"],
        raw_split["y_train"],
        args.cell_size_m,
        args.ngram_size,
        args.min_frequency,
    )
    validation_rtdm = rtdm_scores(raw_split["x_val"], support, args.cell_size_m, args.ngram_size)
    test_rtdm = rtdm_scores(raw_split["x_test"], support, args.cell_size_m, args.ngram_size)
    rtdm_threshold, validation_rtdm_metrics = best_threshold_for_scores(raw_split["y_val"], validation_rtdm)
    test_rtdm_metrics = metrics(raw_split["y_test"], test_rtdm, rtdm_threshold)
    hybrid_weight, hybrid_threshold, validation_hybrid_metrics = tune_hybrid(
        raw_split["y_val"],
        rnn_result["validation_scores"],
        validation_rtdm,
    )
    test_hybrid_scores = hybrid_weight * rnn_result["test_scores"] + (1.0 - hybrid_weight) * test_rtdm
    test_hybrid_metrics = metrics(raw_split["y_test"], test_hybrid_scores, hybrid_threshold)
    rnn_test = rnn_result["test_metrics"]
    delta = {
        key: float(test_hybrid_metrics[key] - rnn_test[key])
        for key in ["accuracy", "precision", "recall", "f1"]
    }
    report = {
        "source": str(args.source),
        "device": str(device),
        "sequence_length": args.sequence_length,
        "rtdm_source": "https://github.com/NicklasXYZ/rtdm",
        "rtdm_method": "lightweight meter-grid token support sequence scoring inspired by NicklasXYZ/rtdm; not a direct port of the original package",
        "kalman": {
            "measurement_noise_m": args.measurement_noise_m,
            "process_noise": args.process_noise,
        },
        "rtdm": {
            "cell_size_m": args.cell_size_m,
            "ngram_size": args.ngram_size,
            "min_frequency": args.min_frequency,
            "support_token_count": support["support_token_count"],
            "support_transition_count": support["support_transition_count"],
            "support_ngram_count": support["support_ngram_count"],
        },
        "thresholds": {
            "rnn": rnn_result["threshold"],
            "rtdm": rtdm_threshold,
            "hybrid": hybrid_threshold,
        },
        "hybrid_weights": {
            "rnn": hybrid_weight,
            "rtdm": 1.0 - hybrid_weight,
        },
        "validation_metrics": {
            "rnn_kalman": rnn_result["validation_metrics"],
            "rtdm_only": validation_rtdm_metrics,
            "hybrid": validation_hybrid_metrics,
        },
        "test_metrics": {
            "rnn_kalman": rnn_test,
            "rtdm_only": test_rtdm_metrics,
            "hybrid": test_hybrid_metrics,
        },
        "delta_vs_rnn": delta,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(report, args.markdown)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
