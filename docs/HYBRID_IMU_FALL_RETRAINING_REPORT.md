# Hybrid IMU Fall 재학습 결과

## 사용 데이터셋

- Source: `../data/iccas_sensor_lstm/final_iccas_sisfall_imu_merged.csv`
- 데이터 구성: ICCAS 직접 취득 IMU 데이터 + SisFall IMU 낙상 데이터
- GPS는 사용하지 않음. IMU/Gyro 낙상 탐지만 재학습함.

## 직접 취득 임계값 반영

- SAMPLE_MS: `40` ms
- IMU_BUF_N / sequence_length: `50`
- Buffer seconds: `2.00` s
- FALL_IMPACT_G: `2.5` g
- FALL_FREE_G: `0.6` g
- FALL_COOLDOWN: `5000` ms

## 성능 비교

| Method | Accuracy | Precision | Recall | F1-score | TP | FP | TN | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lstm_only | 0.8799 | 0.8498 | 0.8863 | 0.8677 | 4147 | 733 | 5120 | 532 |
| algorithm_only | 0.4445 | 0.4443 | 0.9981 | 0.6148 | 4670 | 5842 | 11 | 9 |
| hybrid | 0.8799 | 0.8498 | 0.8863 | 0.8677 | 4147 | 733 | 5120 | 532 |

## 모델 파일

- Model: `models/iccas_final_hybrid_lstm_imu_fall.pt`
- Metadata: `models/iccas_final_hybrid_lstm_imu_fall.json`
- JSON report: `../data/iccas_sensor_lstm/hybrid_imu_fall_metrics.json`

## 해석

- `lstm_only`는 LSTM 확률만 사용한 결과입니다.
- `algorithm_only`는 impact/rotation/inactivity 알고리즘만 사용한 결과입니다.
- `hybrid`는 LSTM 점수와 알고리즘 점수를 validation F1 기준으로 결합한 결과입니다.