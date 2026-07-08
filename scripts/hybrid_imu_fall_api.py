"""FastAPI service for the ICCAS Hybrid IMU Fall LSTM checkpoint."""

from __future__ import annotations

import os
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from torch import nn


DEFAULT_MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/iccas_final_hybrid_lstm_imu_fall.pt"))
MAX_DEVICE_BUFFERS = int(os.getenv("MAX_DEVICE_BUFFERS", "256"))


class BinaryLSTM(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        pooling: str,
    ) -> None:
        super().__init__()
        self.bidirectional = bidirectional
        self.pooling = pooling
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=bidirectional,
        )
        output_size = hidden_size * (2 if bidirectional else 1)
        self.attention = nn.Linear(output_size, 1)
        self.head = nn.Sequential(
            nn.LayerNorm(output_size),
            nn.Dropout(dropout),
            nn.Linear(output_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(x)
        if self.pooling == "attention":
            weights = torch.softmax(self.attention(output), dim=1)
            pooled = (output * weights).sum(dim=1)
        elif self.pooling == "mean":
            pooled = output.mean(dim=1)
        else:
            pooled = output[:, -1, :]
        return self.head(pooled).squeeze(-1)


class ImuSample(BaseModel):
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    ax: float = 0.0
    ay: float = 0.0
    az: float = 0.0
    wx: float = 0.0
    wy: float = 0.0
    wz: float = 0.0
    t_ms: float | None = None
    timestamp: str | None = None
    device: str = "default"
    label: str | None = None
    lat: float | None = None
    lng: float | None = None


class WindowRequest(BaseModel):
    device: str = "default"
    samples: list[ImuSample] = Field(..., min_length=1)


class PointRequest(ImuSample):
    pass


class HybridImuFallService:
    def __init__(self, model_path: Path) -> None:
        if not model_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
        self.model_path = model_path
        self.device = torch.device("cpu")
        # The checkpoint is produced by this project and contains numpy scaler
        # arrays in addition to tensors, so PyTorch 2.6+ needs weights_only=False.
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.feature_columns: list[str] = list(checkpoint["feature_columns"])
        self.sequence_length = int(checkpoint["sequence_length"])
        self.sample_ms = int(checkpoint.get("sample_ms", 40))
        self.threshold = float(checkpoint["threshold"])
        self.lstm_threshold = float(checkpoint.get("lstm_threshold", self.threshold))
        self.algorithm_threshold = float(checkpoint.get("algorithm_threshold", 0.08))
        self.hybrid_lstm_weight = float(checkpoint.get("hybrid_lstm_weight", 1.0))
        self.hybrid_algorithm_weight = float(checkpoint.get("hybrid_algorithm_weight", 0.0))
        self.fall_algorithm = dict(checkpoint.get("fall_algorithm", {}))
        self.center = np.asarray(checkpoint["scaler_center"], dtype=np.float32)
        self.scale = np.asarray(checkpoint["scaler_scale"], dtype=np.float32)
        self.model = BinaryLSTM(
            len(self.feature_columns),
            int(checkpoint["hidden_size"]),
            int(checkpoint["num_layers"]),
            float(checkpoint["dropout"]),
            bool(checkpoint["bidirectional"]),
            str(checkpoint["pooling"]),
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()
        self.buffers: dict[str, deque[ImuSample]] = {}

    def sample_to_features(self, samples: list[ImuSample]) -> np.ndarray:
        rows: list[list[float]] = []
        previous_t_ms: float | None = None
        for index, sample in enumerate(samples):
            accel_norm = float(np.sqrt(sample.ax**2 + sample.ay**2 + sample.az**2))
            gyro_norm = float(np.sqrt(sample.wx**2 + sample.wy**2 + sample.wz**2))
            if sample.t_ms is None:
                dt_s = 0.0 if index == 0 else self.sample_ms / 1000.0
            elif previous_t_ms is None:
                dt_s = 0.0
            else:
                dt_s = float(np.clip(sample.t_ms - previous_t_ms, 0.0, 1000.0) / 1000.0)
            if sample.t_ms is not None:
                previous_t_ms = sample.t_ms
            row = {
                "roll": sample.roll,
                "pitch": sample.pitch,
                "yaw": sample.yaw,
                "ax": sample.ax,
                "ay": sample.ay,
                "az": sample.az,
                "wx": sample.wx,
                "wy": sample.wy,
                "wz": sample.wz,
                "accel_norm": accel_norm,
                "gyro_norm": gyro_norm,
                "dt_s": dt_s,
            }
            rows.append([float(row[column]) for column in self.feature_columns])
        return np.asarray(rows, dtype=np.float32)

    def scale_features(self, values: np.ndarray) -> np.ndarray:
        scaled = (values.astype(np.float32) - self.center) / self.scale
        return np.clip(scaled, -12.0, 12.0).astype(np.float32)

    def algorithm_score(self, values: np.ndarray) -> dict[str, Any]:
        index = {name: self.feature_columns.index(name) for name in self.feature_columns}
        accel = values[:, index["accel_norm"]]
        gyro = values[:, index["gyro_norm"]]
        roll = values[:, index["roll"]]
        pitch = values[:, index["pitch"]]
        impact_peak = float(np.nanmax(accel))
        freefall_min = float(np.nanmin(accel))
        gyro_peak = float(np.nanmax(gyro))
        tilt_change = float(np.nanmax(np.sqrt((roll - roll[0]) ** 2 + (pitch - pitch[0]) ** 2)))
        impact_index = int(np.nanargmax(accel))
        post_samples = int(self.fall_algorithm.get("post_samples", 25))
        start = impact_index + 1
        end = min(len(values), start + post_samples)
        post_accel = accel[start:end] if start < len(values) else accel[-max(2, post_samples) :]
        post_gyro = gyro[start:end] if start < len(values) else gyro[-max(2, post_samples) :]
        post_accel_std = float(np.nanstd(post_accel)) if len(post_accel) else 0.0
        post_gyro_mean = float(np.nanmean(post_gyro)) if len(post_gyro) else 0.0
        impact_g = float(self.fall_algorithm.get("impact_g", 2.5))
        freefall_g = float(self.fall_algorithm.get("freefall_g", 0.6))
        gyro_dps = float(self.fall_algorithm.get("gyro_dps", 250.0))
        tilt_deg = float(self.fall_algorithm.get("tilt_deg", 45.0))
        still_accel_std = float(self.fall_algorithm.get("still_accel_std", 0.35))
        still_gyro_dps = float(self.fall_algorithm.get("still_gyro_dps", 80.0))
        impact_score = float(np.clip((impact_peak - impact_g) / max(impact_g, 1e-6), 0.0, 1.0))
        freefall_score = float(freefall_min <= freefall_g)
        gyro_score = float(np.clip(gyro_peak / max(gyro_dps, 1e-6), 0.0, 1.0))
        tilt_score = float(np.clip(tilt_change / max(tilt_deg, 1e-6), 0.0, 1.0))
        rotation_score = max(gyro_score, tilt_score)
        inactivity = float(post_accel_std <= still_accel_std and post_gyro_mean <= still_gyro_dps)
        score = (
            0.40 * impact_score
            + 0.20 * rotation_score
            + 0.25 * inactivity
            + 0.15 * max(freefall_score, tilt_score)
        )
        return {
            "algorithm_score": float(np.clip(score, 0.0, 1.0)),
            "impact_peak_g": impact_peak,
            "freefall_min_g": freefall_min,
            "gyro_peak_dps": gyro_peak,
            "tilt_change_deg": tilt_change,
            "post_accel_std": post_accel_std,
            "post_gyro_mean": post_gyro_mean,
            "post_fall_inactivity": bool(inactivity),
        }

    def predict_samples(self, samples: list[ImuSample]) -> dict[str, Any]:
        if len(samples) < self.sequence_length:
            return {
                "ready": False,
                "required_samples": self.sequence_length,
                "received_samples": len(samples),
                "fall_detected": False,
                "risk_level": "warming_up",
                "detection_type": "insufficient_window",
            }
        window = samples[-self.sequence_length :]
        raw_features = self.sample_to_features(window)
        scaled = self.scale_features(raw_features)
        started = time.perf_counter()
        with torch.inference_mode():
            logits = self.model(torch.from_numpy(scaled[None, :, :]).to(self.device))
            lstm_score = float(torch.sigmoid(logits).cpu().numpy()[0])
        inference_ms = (time.perf_counter() - started) * 1000.0
        algo = self.algorithm_score(raw_features)
        hybrid_score = (
            self.hybrid_lstm_weight * lstm_score
            + self.hybrid_algorithm_weight * float(algo["algorithm_score"])
        )
        fall_detected = bool(hybrid_score >= self.threshold)
        return {
            "ready": True,
            "required_samples": self.sequence_length,
            "received_samples": len(samples),
            "fall_detected": fall_detected,
            "risk_level": "high" if fall_detected else "low",
            "detection_type": "fall" if fall_detected else "normal",
            "hybrid_score": round(float(hybrid_score), 6),
            "lstm_score": round(lstm_score, 6),
            "algorithm_score": round(float(algo["algorithm_score"]), 6),
            "threshold": self.threshold,
            "lstm_threshold": self.lstm_threshold,
            "algorithm_threshold": self.algorithm_threshold,
            "hybrid_lstm_weight": self.hybrid_lstm_weight,
            "hybrid_algorithm_weight": self.hybrid_algorithm_weight,
            "inference_ms": round(inference_ms, 4),
            "event": "fall" if fall_detected else "",
            **algo,
        }

    def append_point(self, sample: ImuSample) -> dict[str, Any]:
        if len(self.buffers) >= MAX_DEVICE_BUFFERS and sample.device not in self.buffers:
            oldest = next(iter(self.buffers))
            self.buffers.pop(oldest, None)
        buffer = self.buffers.setdefault(sample.device, deque(maxlen=self.sequence_length))
        buffer.append(sample)
        result = self.predict_samples(list(buffer))
        result.update(
            {
                "device": sample.device,
                "timestamp": sample.timestamp,
                "lat": sample.lat,
                "lng": sample.lng,
            }
        )
        return result


app = FastAPI(title="ICCAS Hybrid IMU Fall LSTM API", version="1.0.0")
service = HybridImuFallService(DEFAULT_MODEL_PATH)


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "model_path": str(service.model_path),
        "sequence_length": service.sequence_length,
        "sample_rate_hz": 1000.0 / service.sample_ms,
        "threshold": service.threshold,
    }


@app.get("/model-info")
def model_info() -> dict[str, Any]:
    return {
        "model_path": str(service.model_path),
        "feature_columns": service.feature_columns,
        "sequence_length": service.sequence_length,
        "sample_ms": service.sample_ms,
        "threshold": service.threshold,
        "hybrid_lstm_weight": service.hybrid_lstm_weight,
        "hybrid_algorithm_weight": service.hybrid_algorithm_weight,
        "fall_algorithm": service.fall_algorithm,
    }


@app.post("/predict-point")
def predict_point(sample: PointRequest) -> dict[str, Any]:
    return service.append_point(sample)


@app.post("/predict-window")
def predict_window(request: WindowRequest) -> dict[str, Any]:
    if len(request.samples) < service.sequence_length:
        raise HTTPException(status_code=422, detail=f"At least {service.sequence_length} samples are required.")
    result = service.predict_samples(request.samples)
    result["device"] = request.device
    return result


@app.post("/reset-buffer/{device}")
def reset_buffer(device: str) -> dict[str, Any]:
    existed = device in service.buffers
    service.buffers.pop(device, None)
    return {"ok": True, "device": device, "cleared": existed}
