"""Production-ready FastAPI server for the final TCN IMU fall model."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from torch import nn


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = Path(os.getenv("MODEL_PATH", ROOT / "models" / "v3_tcn_imu_fall.pt"))
BACKEND_RESULT_URL = os.getenv("BACKEND_RESULT_URL", "").strip()
MAX_DEVICE_BUFFERS = int(os.getenv("MAX_DEVICE_BUFFERS", "256"))
HTTP_TIMEOUT_SECONDS = float(os.getenv("HTTP_TIMEOUT_SECONDS", "2.0"))


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
        blocks = []
        in_channels = input_size
        for index in range(num_layers):
            blocks.append(TemporalBlock(in_channels, hidden_size, kernel_size, dilation=2**index, dropout=dropout))
            in_channels = hidden_size
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
    personId: str | int | None = None
    lat: float | None = None
    lng: float | None = None


class WindowRequest(BaseModel):
    device: str = "default"
    personId: str | int | None = None
    samples: list[ImuSample] = Field(..., min_length=1)
    forward_to_backend: bool = True


class PointRequest(ImuSample):
    forward_to_backend: bool = True


class TCNFallService:
    def __init__(self, model_path: Path) -> None:
        if not model_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
        self.model_path = model_path
        self.device = torch.device("cpu")
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.feature_columns: list[str] = list(checkpoint["feature_columns"])
        self.sequence_length = int(checkpoint["sequence_length"])
        self.sequence_stride = int(checkpoint.get("sequence_stride", 4))
        self.sample_ms = int(checkpoint.get("sample_ms", 40))
        self.sample_rate_hz = float(checkpoint.get("sample_rate_hz", 1000.0 / self.sample_ms))
        self.threshold = float(checkpoint["threshold"])
        self.center = np.asarray(checkpoint["scaler_center"], dtype=np.float32)
        self.scale = np.asarray(checkpoint["scaler_scale"], dtype=np.float32)
        self.model = TCNFallModel(
            len(self.feature_columns),
            int(checkpoint["hidden_size"]),
            int(checkpoint["num_layers"]),
            int(checkpoint["kernel_size"]),
            float(checkpoint["dropout"]),
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

    def predict_samples(self, samples: list[ImuSample]) -> dict[str, Any]:
        if len(samples) < self.sequence_length:
            return {
                "ready": False,
                "required_samples": self.sequence_length,
                "received_samples": len(samples),
                "fall_detected": False,
                "alarm_active": False,
                "risk_level": "warming_up",
                "detection_type": "insufficient_window",
                "event": "",
            }

        window = samples[-self.sequence_length :]
        raw_features = self.sample_to_features(window)
        scaled = self.scale_features(raw_features)
        started = time.perf_counter()
        with torch.inference_mode():
            logits = self.model(torch.from_numpy(scaled[None, :, :]).to(self.device))
            score = float(torch.sigmoid(logits).cpu().numpy()[0])
        inference_ms = (time.perf_counter() - started) * 1000.0

        fall_detected = bool(score >= self.threshold)
        accel_norm = raw_features[:, self.feature_columns.index("accel_norm")]
        gyro_norm = raw_features[:, self.feature_columns.index("gyro_norm")]
        return {
            "ready": True,
            "required_samples": self.sequence_length,
            "received_samples": len(samples),
            "fall_detected": fall_detected,
            "alarm_active": fall_detected,
            "risk_level": "high" if fall_detected else "low",
            "detection_type": "fall" if fall_detected else "normal",
            "event": "fall" if fall_detected else "",
            "label": "fall" if fall_detected else "normal",
            "model_type": "tcn",
            "model_score": round(score, 6),
            "threshold": round(self.threshold, 6),
            "inference_ms": round(inference_ms, 4),
            "window_seconds": round(self.sequence_length * self.sample_ms / 1000.0, 4),
            "accel_peak_g": round(float(np.nanmax(accel_norm)), 6),
            "gyro_peak_dps": round(float(np.nanmax(gyro_norm)), 6),
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
                "personId": sample.personId,
                "timestamp": sample.timestamp,
                "lat": sample.lat,
                "lng": sample.lng,
            }
        )
        return result


def post_json(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    if not url:
        return {"forwarded": False, "reason": "BACKEND_RESULT_URL is empty"}
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT_SECONDS) as response:
            body = response.read().decode("utf-8", errors="replace")
            return {
                "forwarded": True,
                "status": response.status,
                "body": body[:500],
                "url": url,
            }
    except urllib.error.URLError as exc:
        return {
            "forwarded": False,
            "url": url,
            "error": str(exc),
        }


app = FastAPI(title="ICCAS Final TCN IMU Fall Server", version="1.0.0")
service = TCNFallService(DEFAULT_MODEL_PATH)


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "model_type": "tcn",
        "model_path": str(service.model_path),
        "sequence_length": service.sequence_length,
        "sample_rate_hz": service.sample_rate_hz,
        "threshold": service.threshold,
        "backend_result_url": BACKEND_RESULT_URL or None,
    }


@app.get("/model-info")
def model_info() -> dict[str, Any]:
    return {
        "model_type": "tcn",
        "feature_columns": service.feature_columns,
        "sequence_length": service.sequence_length,
        "sequence_stride": service.sequence_stride,
        "sample_ms": service.sample_ms,
        "sample_rate_hz": service.sample_rate_hz,
        "threshold": service.threshold,
    }


@app.post("/predict-point")
def predict_point(sample: PointRequest) -> dict[str, Any]:
    result = service.append_point(sample)
    if sample.forward_to_backend and result["ready"]:
        result["backend_delivery"] = post_json(BACKEND_RESULT_URL, result)
    return result


@app.post("/predict-window")
def predict_window(request: WindowRequest) -> dict[str, Any]:
    if len(request.samples) < service.sequence_length:
        raise HTTPException(status_code=422, detail=f"At least {service.sequence_length} samples are required.")
    result = service.predict_samples(request.samples)
    last = request.samples[-1]
    result.update(
        {
            "device": request.device,
            "personId": request.personId if request.personId is not None else last.personId,
            "timestamp": last.timestamp,
            "lat": last.lat,
            "lng": last.lng,
        }
    )
    if request.forward_to_backend:
        result["backend_delivery"] = post_json(BACKEND_RESULT_URL, result)
    return result


@app.post("/api/sensor-result")
def sensor_result_compatible(sample: PointRequest) -> dict[str, Any]:
    """Compatibility endpoint for backend/sensor clients that POST sensor data."""
    return predict_point(sample)


@app.post("/reset-buffer/{device}")
def reset_buffer(device: str) -> dict[str, Any]:
    existed = device in service.buffers
    service.buffers.pop(device, None)
    return {"ok": True, "device": device, "cleared": existed}
