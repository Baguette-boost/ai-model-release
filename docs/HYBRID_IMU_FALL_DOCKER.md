# Hybrid IMU Fall LSTM Docker 실행 방법

이 컨테이너는 `models/iccas_final_hybrid_lstm_imu_fall.pt`를 로드해서 IMU/Gyro 낙상 감지 API를 제공합니다.

## 빌드

```bash
cd /Volumes/Hub_1T/ICCAS/ai-model-release
docker build -t iccas-hybrid-imu-fall:latest .
```

또는 compose 사용:

```bash
docker compose up --build
```

## 실행

```bash
docker run --rm -p 8001:8001 iccas-hybrid-imu-fall:latest
```

실행 후 확인:

```bash
curl http://127.0.0.1:8001/health
curl http://127.0.0.1:8001/model-info
```

## 실시간 센서 포인트 추론

ESP32처럼 샘플이 1개씩 들어오는 경우 `POST /predict-point`를 사용합니다.
모델은 내부적으로 device별 50-sample ring buffer를 유지합니다.

```bash
curl -X POST http://127.0.0.1:8001/predict-point \
  -H "Content-Type: application/json" \
  -d '{
    "device": "esp32-1",
    "timestamp": "2026-07-08T15:30:00",
    "t_ms": 0,
    "roll": 0,
    "pitch": 0,
    "yaw": 0,
    "ax": 0.02,
    "ay": -0.86,
    "az": -0.07,
    "wx": -1.9,
    "wy": 1.1,
    "wz": -0.4
  }'
```

처음 49개 샘플까지는 아래처럼 반환됩니다.

```json
{
  "ready": false,
  "required_samples": 50,
  "received_samples": 1,
  "fall_detected": false,
  "risk_level": "warming_up"
}
```

50개 샘플이 쌓이면 LSTM 추론 결과가 반환됩니다.

```json
{
  "ready": true,
  "fall_detected": false,
  "risk_level": "low",
  "detection_type": "normal",
  "hybrid_score": 0.123456,
  "lstm_score": 0.123456,
  "algorithm_score": 0.25,
  "threshold": 0.35,
  "inference_ms": 1.1
}
```

## 50개 window 직접 추론

이미 50개 샘플을 모아서 서버에 보내는 구조라면 `POST /predict-window`를 사용합니다.

```json
{
  "device": "esp32-1",
  "samples": [
    {
      "t_ms": 0,
      "roll": 0,
      "pitch": 0,
      "yaw": 0,
      "ax": 0.02,
      "ay": -0.86,
      "az": -0.07,
      "wx": -1.9,
      "wy": 1.1,
      "wz": -0.4
    }
  ]
}
```

`samples` 배열은 최소 50개가 필요합니다.

## API 목록

| Method | Path | 설명 |
| --- | --- | --- |
| GET | `/health` | 서버/모델 로딩 확인 |
| GET | `/model-info` | feature, threshold, hybrid weight 확인 |
| POST | `/predict-point` | 센서 1개 입력, device별 buffer 누적 |
| POST | `/predict-window` | 50개 이상 window 직접 추론 |
| POST | `/reset-buffer/{device}` | 특정 device buffer 초기화 |

## 현재 모델 설정

| 항목 | 값 |
| --- | ---: |
| model | `models/iccas_final_hybrid_lstm_imu_fall.pt` |
| sequence length | 50 |
| sample rate | 25 Hz |
| window length | 약 2초 |
| threshold | 0.35 |
| hybrid LSTM weight | 1.0 |
| hybrid algorithm weight | 0.0 |

현재 튜닝 결과에서는 hybrid 구조가 남아 있지만 최종 score는 LSTM 확률을 100% 사용합니다. 물리 기반 algorithm score는 응답에 함께 제공되어 설명/디버깅용으로 사용할 수 있습니다.
