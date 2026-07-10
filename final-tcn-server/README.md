# Final TCN IMU Fall Server

This directory is the final deployable TCN server package.
It is separated from previous experiments so the production server does not mix with V1/V2/V3 training files.

## Included Files

- `app.py`: FastAPI server that loads the TCN model and runs realtime inference.
- `models/v3_tcn_imu_fall.pt`: selected TCN checkpoint.
- `requirements.txt`: Python packages.
- `run_server.sh`: local server launcher.
- `scripts/send_demo_window.py`: demo client for a 50-sample IMU window.

## Model

- Model: TCN
- Task: IMU fall detection
- Input window: 50 IMU samples
- Sampling rate: 25 Hz
- Decision window: 2.0 seconds
- Threshold: 0.46
- Test performance: Accuracy 0.9132, Precision 0.9548, Recall 0.8446, F1-score 0.8963

## Run Server

Use the existing project virtual environment:

```bash
cd /Volumes/Hub_1T/ICCAS/ai-model-release/final-tcn-server
../../.venv/bin/python -m uvicorn app:app --host 127.0.0.1 --port 8010
```

Or:

```bash
cd /Volumes/Hub_1T/ICCAS/ai-model-release/final-tcn-server
PATH=/Volumes/Hub_1T/ICCAS/.venv/bin:$PATH ./run_server.sh
```

## Health Check

```bash
curl http://127.0.0.1:8010/health
```

Expected:

```json
{
  "ok": true,
  "model_type": "tcn",
  "sequence_length": 50,
  "sample_rate_hz": 25.0,
  "threshold": 0.46
}
```

## Realtime Point Inference

Send one IMU point at a time.
The server keeps a 50-sample buffer per `device`.
Before 50 samples arrive, the response has `ready: false`.
After 50 samples, the response includes `fall_detected`, `model_score`, and `event`.

Endpoint:

```text
POST /predict-point
```

Compatible endpoint:

```text
POST /api/sensor-result
```

Example:

```bash
curl -X POST http://127.0.0.1:8010/predict-point \
  -H "Content-Type: application/json" \
  -d '{
    "device": "esp32-1",
    "personId": "user-1",
    "t_ms": 0,
    "roll": 0.0,
    "pitch": 0.0,
    "yaw": 0.0,
    "ax": 0.02,
    "ay": -0.86,
    "az": -0.06,
    "wx": -2.0,
    "wy": 1.1,
    "wz": -0.4,
    "lat": 36.621853,
    "lng": 127.426337,
    "forward_to_backend": false
  }'
```

## Window Inference

Send at least 50 samples at once:

```text
POST /predict-window
```

Demo:

```bash
cd /Volumes/Hub_1T/ICCAS/ai-model-release/final-tcn-server
../../.venv/bin/python scripts/send_demo_window.py
```

## Forward Result To Backend

Set `BACKEND_RESULT_URL` when running the server:

```bash
cd /Volumes/Hub_1T/ICCAS/ai-model-release/final-tcn-server
BACKEND_RESULT_URL=http://127.0.0.1:8000/api/sensor-result \
../../.venv/bin/python -m uvicorn app:app --host 127.0.0.1 --port 8010
```

When `forward_to_backend` is true and the 50-sample window is ready, the TCN server POSTs the inference result to `BACKEND_RESULT_URL`.

## Output Fields

Important fields:

- `ready`: whether 50 samples are available.
- `fall_detected`: final TCN fall decision.
- `alarm_active`: same boolean as `fall_detected`, useful for backend alarm logic.
- `model_score`: TCN sigmoid score.
- `threshold`: model decision threshold.
- `inference_ms`: model inference time for the request.
- `window_seconds`: 2.0 seconds for 50 samples at 25 Hz.
- `backend_delivery`: backend POST result, only when forwarding is enabled.
