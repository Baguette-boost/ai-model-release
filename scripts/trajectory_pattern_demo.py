"""Demo: early Martino-Saltzman trajectory inference before LSTM is ready.

Input JSON format:
[
  {"time": "10:00:00", "lat": 36.6220, "lng": 127.4288},
  ...
]
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


EARTH_RADIUS_M = 6_371_008.8
GPS_FEATURES = ["x_m", "y_m", "dx_m", "dy_m", "speed_mps", "dt_s", "gps_valid"]


class BinaryLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(x)
        return self.head(output[:, -1, :]).squeeze(-1)


class RobustScaler:
    def __init__(self, center: list[float], scale: list[float]) -> None:
        self.center = np.asarray(center, dtype=np.float32)
        self.scale = np.asarray(scale, dtype=np.float32)

    def transform(self, values: np.ndarray) -> np.ndarray:
        scaled = (values.astype(np.float32) - self.center) / self.scale
        return np.clip(scaled, -12.0, 12.0).astype(np.float32)


def latlon_to_xy(lat: float, lng: float, origin_lat: float, origin_lng: float) -> tuple[float, float]:
    x = EARTH_RADIUS_M * math.radians(lng - origin_lng) * math.cos(math.radians(origin_lat))
    y = EARTH_RADIUS_M * math.radians(lat - origin_lat)
    return x, y


def distance_m(a: dict[str, Any], b: dict[str, Any]) -> float:
    x1, y1 = latlon_to_xy(float(a["lat"]), float(a["lng"]), float(a["lat"]), float(a["lng"]))
    x2, y2 = latlon_to_xy(float(b["lat"]), float(b["lng"]), float(a["lat"]), float(a["lng"]))
    return math.hypot(x2 - x1, y2 - y1)


def parse_time(value: str) -> datetime:
    return datetime.strptime(value, "%H:%M:%S")


def clean_points(points: list[dict[str, Any]], max_speed_mps: float) -> list[dict[str, Any]]:
    cleaned: list[dict[str, Any]] = []
    for point in points:
        try:
            current = {
                "time": str(point["time"]),
                "lat": float(point["lat"]),
                "lng": float(point["lng"]),
            }
        except (KeyError, TypeError, ValueError):
            continue
        if not (-90 <= current["lat"] <= 90 and -180 <= current["lng"] <= 180):
            continue
        if cleaned:
            dt = max((parse_time(current["time"]) - parse_time(cleaned[-1]["time"])).total_seconds(), 1.0)
            speed = distance_m(cleaned[-1], current) / dt
            if speed > max_speed_mps:
                continue
        cleaned.append(current)
    return cleaned


def heading(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.atan2(b[1] - a[1], b[0] - a[0])


def angle_diff(a: float, b: float) -> float:
    value = (b - a + math.pi) % (2 * math.pi) - math.pi
    return value


def segments_intersect(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float], d: tuple[float, float]) -> bool:
    def orient(p: tuple[float, float], q: tuple[float, float], r: tuple[float, float]) -> float:
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

    return orient(a, b, c) * orient(a, b, d) < 0 and orient(c, d, a) * orient(c, d, b) < 0


def heading_entropy(turns: list[float]) -> float:
    if not turns:
        return 0.0
    bins = [0] * 8
    for turn in turns:
        index = min(7, int(((turn + math.pi) / (2 * math.pi)) * 8))
        bins[index] += 1
    total = sum(bins)
    entropy = 0.0
    for count in bins:
        if count:
            p = count / total
            entropy -= p * math.log(p, 2)
    return entropy / 3.0


def trajectory_features(points: list[dict[str, Any]]) -> dict[str, Any]:
    if len(points) < 2:
        return {
            "distance_factor": 0.0,
            "total_distance_m": 0.0,
            "self_intersections": 0,
            "max_repeated_node_visits": 0,
            "uturn_count": 0,
            "angle_entropy": 0.0,
            "signed_turn_sum_deg": 0.0,
        }

    origin = points[0]
    xy = [latlon_to_xy(p["lat"], p["lng"], origin["lat"], origin["lng"]) for p in points]
    distances = [math.hypot(xy[i][0] - xy[i - 1][0], xy[i][1] - xy[i - 1][1]) for i in range(1, len(xy))]
    total = sum(distances)
    direct = math.hypot(xy[-1][0] - xy[0][0], xy[-1][1] - xy[0][1])
    df = direct / total if total > 0 else 0.0

    headings = [heading(xy[i - 1], xy[i]) for i in range(1, len(xy)) if distances[i - 1] > 0.5]
    turns = [angle_diff(headings[i - 1], headings[i]) for i in range(1, len(headings))]
    abs_turns = [abs(turn) for turn in turns]
    uturn_count = sum(1 for turn in abs_turns if math.radians(140) <= turn <= math.radians(220))
    signed_turn_sum_deg = math.degrees(sum(turns))

    intersections = 0
    for i in range(len(xy) - 1):
        for j in range(i + 2, len(xy) - 1):
            if j == i + 1:
                continue
            if segments_intersect(xy[i], xy[i + 1], xy[j], xy[j + 1]):
                intersections += 1

    max_visits = 0
    for i, center in enumerate(xy):
        visits = sum(1 for item in xy if math.hypot(center[0] - item[0], center[1] - item[1]) <= 20.0)
        max_visits = max(max_visits, visits)

    return {
        "distance_factor": df,
        "total_distance_m": total,
        "self_intersections": intersections,
        "max_repeated_node_visits": max_visits,
        "uturn_count": uturn_count,
        "angle_entropy": heading_entropy(turns),
        "signed_turn_sum_deg": signed_turn_sum_deg,
    }


def classify_pattern(points: list[dict[str, Any]]) -> dict[str, Any]:
    features = trajectory_features(points)
    df = features["distance_factor"]
    total = features["total_distance_m"]
    intersections = features["self_intersections"]
    revisits = features["max_repeated_node_visits"]
    uturns = features["uturn_count"]
    entropy = features["angle_entropy"]
    turn_sum = abs(features["signed_turn_sum_deg"])

    if len(points) < 3 or total < 5:
        pattern = "Direct"
        confidence = 0.35
        reason = "좌표 수가 부족해 초기 정상 이동 후보로만 판단했습니다."
    elif df >= 0.75 and entropy < 0.45 and intersections == 0:
        pattern = "Direct"
        confidence = min(0.95, 0.65 + df * 0.25)
        reason = "직선거리 대비 총 이동거리가 유사하고 방향 변화가 작습니다."
    elif df <= 0.6 and intersections >= 3:
        pattern = "Random"
        confidence = min(0.95, 0.68 + min(intersections, 6) * 0.035 + entropy * 0.15)
        reason = "거리 효율이 낮고 자기 경로 교차가 여러 번 발생했습니다."
    elif uturns >= 2 and revisits >= 2 and entropy < 0.75:
        pattern = "Pacing"
        confidence = min(0.95, 0.55 + 0.1 * uturns + 0.03 * revisits)
        reason = "180도에 가까운 방향 전환과 동일 지점 반복 방문이 나타났습니다."
    elif df <= 0.25 and revisits >= 2 and 250 <= turn_sum <= 520:
        pattern = "Lapping"
        confidence = min(0.95, 0.65 + (1 - df) * 0.2)
        reason = "시작점과 끝점이 가깝고 누적 회전각이 원형 순환 패턴에 가깝습니다."
    elif df <= 0.55 and (entropy >= 0.62 or intersections >= 2):
        pattern = "Random"
        confidence = min(0.95, 0.58 + entropy * 0.25 + min(intersections, 4) * 0.03)
        reason = "거리 효율이 낮고 방향 전환 불규칙성 또는 자기 교차가 큽니다."
    else:
        pattern = "Direct"
        confidence = 0.55
        reason = "뚜렷한 왕복, 순환, 무작위 교차 조건이 충분하지 않습니다."

    return {
        "classified_pattern": pattern,
        "confidence_score": round(confidence, 3),
        "extracted_features": {
            "distance_factor": round(df, 4),
            "total_distance_m": round(total, 2),
            "self_intersections": int(intersections),
            "max_repeated_node_visits": int(revisits),
        },
        "reasoning": reason,
        "_debug": {
            "uturn_count": int(uturns),
            "angle_entropy": round(entropy, 4),
            "signed_turn_sum_deg": round(features["signed_turn_sum_deg"], 2),
        },
    }


def load_lstm(model_path: Path, device_name: str) -> tuple[dict[str, Any], BinaryLSTM, RobustScaler, torch.device]:
    try:
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(model_path, map_location="cpu")
    device = torch.device("mps" if device_name == "auto" and torch.backends.mps.is_available() else device_name)
    if str(device) == "auto":
        device = torch.device("cpu")
    model = BinaryLSTM(
        input_size=len(checkpoint["feature_columns"]),
        hidden_size=int(checkpoint["hidden_size"]),
        num_layers=int(checkpoint["num_layers"]),
        dropout=float(checkpoint["dropout"]),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    scaler = RobustScaler(checkpoint["scaler_center"], checkpoint["scaler_scale"])
    return checkpoint, model, scaler, device


def make_lstm_features(points: list[dict[str, Any]], checkpoint: dict[str, Any]) -> np.ndarray:
    origin_lat = points[0]["lat"]
    origin_lng = points[0]["lng"]
    values: list[list[float]] = []
    previous: dict[str, Any] | None = None
    for point in points:
        x, y = latlon_to_xy(point["lat"], point["lng"], origin_lat, origin_lng)
        if previous is None:
            dx = dy = 0.0
            dt = 1.0
        else:
            dx = x - previous["x"]
            dy = y - previous["y"]
            dt = max((parse_time(point["time"]) - parse_time(previous["time"])).total_seconds(), 1.0)
        speed = math.hypot(dx, dy) / dt
        row = {
            "x_m": x,
            "y_m": y,
            "dx_m": dx,
            "dy_m": dy,
            "speed_mps": min(speed, 25.0),
            "dt_s": dt,
            "gps_valid": 1.0,
        }
        values.append([row[column] for column in checkpoint["feature_columns"]])
        previous = {"x": x, "y": y, "time": point["time"]}
    return np.asarray(values, dtype=np.float32)


def lstm_predict(
    points: list[dict[str, Any]],
    checkpoint: dict[str, Any],
    model: BinaryLSTM,
    scaler: RobustScaler,
    device: torch.device,
) -> dict[str, Any]:
    sequence_length = int(checkpoint["sequence_length"])
    if len(points) < sequence_length:
        return {"lstm_ready": False, "lstm_wandering_probability": None, "lstm_wandering_detected": False}
    features = make_lstm_features(points[-sequence_length:], checkpoint)
    scaled = scaler.transform(features.reshape(-1, features.shape[-1])).reshape(features.shape)
    tensor = torch.tensor(scaled[None, :, :], dtype=torch.float32, device=device)
    with torch.no_grad():
        probability = float(torch.sigmoid(model(tensor)).cpu().numpy()[0])
    threshold = float(checkpoint.get("threshold", 0.5))
    return {
        "lstm_ready": True,
        "lstm_wandering_probability": round(probability, 4),
        "lstm_wandering_detected": bool(probability >= threshold),
        "lstm_threshold": threshold,
    }


def merge_results(algorithm: dict[str, Any], lstm: dict[str, Any]) -> dict[str, Any]:
    pattern = algorithm["classified_pattern"]
    algorithm_abnormal = pattern in {"Pacing", "Lapping", "Random"}
    if not lstm["lstm_ready"]:
        return {
            "final_detection": "early_pattern_warning" if algorithm_abnormal else "pending_normal",
            "risk_level": "medium" if algorithm_abnormal else "low",
            "final_confidence": algorithm["confidence_score"],
        }
    if lstm["lstm_wandering_detected"] and algorithm_abnormal:
        return {"final_detection": "wandering", "risk_level": "high", "final_confidence": 0.92}
    if lstm["lstm_wandering_detected"] and not algorithm_abnormal:
        return {"final_detection": "suspicious", "risk_level": "medium", "final_confidence": 0.72}
    if not lstm["lstm_wandering_detected"] and algorithm_abnormal:
        return {"final_detection": "early_pattern_warning", "risk_level": "medium", "final_confidence": 0.68}
    return {"final_detection": "normal", "risk_level": "low", "final_confidence": 0.88}


def sample_points(kind: str) -> list[dict[str, Any]]:
    base_lat = 36.622
    base_lng = 127.4288
    meter_lat = 1 / 111_320
    meter_lng = 1 / (111_320 * math.cos(math.radians(base_lat)))
    offsets: list[tuple[float, float]]
    if kind == "direct":
        offsets = [(i * 8, i * 1) for i in range(20)]
    elif kind == "lapping":
        offsets = [(35 * math.cos(i * 2 * math.pi / 20), 35 * math.sin(i * 2 * math.pi / 20)) for i in range(21)]
    elif kind == "random":
        offsets = [(0, 0), (25, 5), (-10, 35), (35, -15), (-20, -25), (28, 30), (-30, 10), (20, -35), (0, 0), (40, 5), (-15, 25), (15, -5), (-35, -10), (10, 40), (0, 0)]
    else:
        offsets = [(0, 0), (18, 0), (36, 0), (18, 0), (0, 0), (18, 0), (36, 0), (18, 0), (0, 0), (18, 0), (36, 0), (18, 0), (0, 0), (18, 0), (36, 0), (18, 0), (0, 0)]
    start = datetime.strptime("10:00:00", "%H:%M:%S")
    out = []
    for index, (east_m, north_m) in enumerate(offsets):
        out.append(
            {
                "time": (start + timedelta(seconds=index * 5)).strftime("%H:%M:%S"),
                "lat": base_lat + north_m * meter_lat,
                "lng": base_lng + east_m * meter_lng,
            }
        )
    return out


def run_demo(args: argparse.Namespace) -> dict[str, Any]:
    if args.input:
        points = json.loads(args.input.read_text(encoding="utf-8"))
    else:
        points = sample_points(args.sample)
    cleaned = clean_points(points, args.max_speed_mps)
    checkpoint, model, scaler, device = load_lstm(args.model, args.device)
    timeline = []
    for index in range(1, len(cleaned) + 1):
        window = cleaned[:index]
        algorithm = classify_pattern(window)
        algorithm.pop("_debug", None)
        lstm = lstm_predict(window, checkpoint, model, scaler, device)
        merged = merge_results(algorithm, lstm)
        timeline.append(
            {
                "index": index,
                "time": window[-1]["time"],
                "algorithm_ready": index >= 3,
                **algorithm,
                **lstm,
                **merged,
            }
        )
    summary = timeline[-1] if timeline else {}
    return {
        "demo": "martino_saltzman_before_lstm",
        "sample": args.sample if not args.input else str(args.input),
        "model": str(args.model),
        "cleaned_points": len(cleaned),
        "lstm_sequence_length": int(checkpoint["sequence_length"]),
        "summary": summary,
        "timeline": timeline,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="JSON file containing GPS points.")
    parser.add_argument("--sample", choices=["direct", "pacing", "lapping", "random"], default="pacing")
    parser.add_argument("--model", type=Path, default=Path("models/iccas_final_lstm_gps_wandering.pt"))
    parser.add_argument("--output", type=Path, default=Path("../data/iccas_sensor_lstm/trajectory_pattern_demo.json"))
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--max-speed-mps", type=float, default=8.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_demo(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "summary": result["summary"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
