# 최종 구조: GPS RNN + IMU Fall LSTM

## 1. 최종 모델 선택

| Sensor | Task | Final Model | Model File |
| --- | --- | --- | --- |
| GPS | 배회 감지 | RNN | `models/iccas_final_rnn_gps_wandering.pt` |
| IMU/Gyro | 낙상 감지 | LSTM | `models/iccas_final_hybrid_lstm_imu_fall.pt` |

## 2. Requirements

```text
numpy>=1.24,<3
pandas>=2.0,<3
torch>=2.0
```

설치:

```bash
cd /Volumes/Hub_1T/ICCAS/ai-model-release
python3 -m venv ../.venv
../.venv/bin/python -m pip install --upgrade pip
../.venv/bin/python -m pip install -r requirements.txt
```

## 3. GPS 전처리

GPS 원본 컬럼:

```text
lat, lng, server_time, device, label
```

RNN 입력 feature:

```text
x_m, y_m, dx_m, dy_m, speed_mps, dt_s, gps_valid
```

전처리 흐름:

```text
lat/lng
  ↓
기준 좌표 origin_lat/origin_lng 기준 meter 좌표 변환
  ↓
x_m, y_m 생성
  ↓
이전 시점 대비 dx_m, dy_m 생성
  ↓
dt_s와 speed_mps 계산
  ↓
최근 16개 시점 x 7개 feature를 RNN에 입력
```

GPS RNN 입력 shape:

```text
[sequence_length=16, feature_count=7]
```

GT label:

```text
label == wandering -> 1
그 외 label -> 0
```

## 4. IMU 낙상 전처리

IMU 원본 컬럼:

```text
roll, pitch, yaw,
ax, ay, az,
wx, wy, wz,
t_ms, label
```

LSTM 입력 feature:

```text
roll, pitch, yaw,
ax, ay, az,
wx, wy, wz,
accel_norm, gyro_norm, dt_s
```

전처리 계산:

```text
accel_norm = sqrt(ax^2 + ay^2 + az^2)
gyro_norm  = sqrt(wx^2 + wy^2 + wz^2)
dt_s       = 이전 t_ms와의 차이 / 1000
fall_target = 1 if label == fall else 0
```

IMU LSTM 입력 shape:

```text
[sequence_length=50, feature_count=12]
```

이 값은 직접 취득 장비 기준과 맞습니다.

```text
SAMPLE_MS = 40 ms
IMU_BUF_N = 50
25 Hz x 2초 buffer
```

## 5. IMU 전처리 CSV

생성 파일:

```text
data/iccas_sensor_lstm/imu_fall_preprocessed.csv
data/iccas_sensor_lstm/imu_fall_preprocessed_summary.json
```

생성 명령어:

```bash
cd /Volumes/Hub_1T/ICCAS/ai-model-release
../.venv/bin/python scripts/export_imu_preprocessed_csv.py \
  --source ../data/iccas_sensor_lstm/final_iccas_sisfall_imu_merged.csv \
  --output ../data/iccas_sensor_lstm/imu_fall_preprocessed.csv \
  --summary ../data/iccas_sensor_lstm/imu_fall_preprocessed_summary.json
```

## 6. 재학습 명령어

GPS RNN:

```bash
cd /Volumes/Hub_1T/ICCAS/ai-model-release
../.venv/bin/python scripts/train_gps_rnn_wandering.py \
  --source ../ICCAS_final_data.xlsx \
  --epochs 15 \
  --batch-size 256 \
  --device auto
```

IMU Fall LSTM:

```bash
cd /Volumes/Hub_1T/ICCAS/ai-model-release
../.venv/bin/python scripts/train_hybrid_imu_fall.py \
  --source ../data/iccas_sensor_lstm/final_iccas_sisfall_imu_merged.csv \
  --epochs 10 \
  --hidden-size 64 \
  --batch-size 256 \
  --device auto
```

## 7. 서버 적용

ESP32:

```text
GPS는 계속 lat/lng 전송
IMU는 svm > 2.5g일 때 최근 2초 buffer 전송
```

AI 서버:

```text
GPS RNN -> wandering_detected
IMU Fall LSTM -> fall_detected
```

최종 서버 payload:

```json
{
  "lat": 36.621853,
  "lng": 127.426337,
  "wandering_detected": false,
  "fall_detected": true,
  "risk_level": "high",
  "detection_type": "fall"
}
```
