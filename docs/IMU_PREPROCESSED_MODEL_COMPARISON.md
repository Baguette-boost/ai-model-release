# 전처리 후 IMU 낙상 모델별 성능 비교

## 실험 조건

- Source: `../data/iccas_sensor_lstm/imu_fall_preprocessed.csv`
- Device: `cpu`
- Sequence length: `50`
- Sequence stride: `4`
- Epochs: `15`
- Batch size: `256`
- Feature: 전처리 완료 12 features, roll/pitch/yaw + accel + gyro + accel_norm + gyro_norm + dt_s
- Split: SisFall group split, ICCAS chronological split

## Hybrid 성능 비교

| Task | Model | Accuracy | Precision | Recall | F1-score | Model weight | Algorithm weight | Threshold | Single inference ms | Train sec |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| IMU 낙상 전처리 후 | RNN | 0.7189 | 0.5787 | 0.7807 | 0.6647 | 1.00 | 0.00 | 0.83 | 0.236 | 36.6 |
| IMU 낙상 전처리 후 | GRU | 0.7721 | 0.6400 | 0.8260 | 0.7212 | 0.85 | 0.15 | 0.79 | 0.564 | 100.0 |
| IMU 낙상 전처리 후 | LSTM | 0.7638 | 0.6178 | 0.8871 | 0.7283 | 1.00 | 0.00 | 0.84 | 0.565 | 124.7 |
| IMU 낙상 전처리 후 | TRANSFORMER | 0.7813 | 0.6397 | 0.8867 | 0.7432 | 0.75 | 0.25 | 0.63 | 0.301 | 169.8 |

## Model Only vs Hybrid

| Model | Method | Accuracy | Precision | Recall | F1-score | Threshold |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| RNN | model_only | 0.7189 | 0.5787 | 0.7807 | 0.6647 | 0.83 |
| RNN | algorithm_only | 0.3569 | 0.3569 | 0.9994 | 0.5259 | 0.07 |
| RNN | hybrid | 0.7189 | 0.5787 | 0.7807 | 0.6647 | 0.83 |
| GRU | model_only | 0.7727 | 0.6408 | 0.8262 | 0.7218 | 0.88 |
| GRU | algorithm_only | 0.3569 | 0.3569 | 0.9994 | 0.5259 | 0.07 |
| GRU | hybrid | 0.7721 | 0.6400 | 0.8260 | 0.7212 | 0.79 |
| LSTM | model_only | 0.7638 | 0.6178 | 0.8871 | 0.7283 | 0.84 |
| LSTM | algorithm_only | 0.3569 | 0.3569 | 0.9994 | 0.5259 | 0.07 |
| LSTM | hybrid | 0.7638 | 0.6178 | 0.8871 | 0.7283 | 0.84 |
| TRANSFORMER | model_only | 0.7820 | 0.6398 | 0.8903 | 0.7446 | 0.73 |
| TRANSFORMER | algorithm_only | 0.3569 | 0.3569 | 0.9994 | 0.5259 | 0.07 |
| TRANSFORMER | hybrid | 0.7813 | 0.6397 | 0.8867 | 0.7432 | 0.63 |

## 결론

- 전처리 후 Hybrid 기준 최고 F1 모델: `TRANSFORMER`
- 최고 Hybrid F1-score: `0.7432`
- 해당 모델 Accuracy: `0.7813`
- 선택된 결합 가중치: model `0.75`, algorithm `0.25`
- 낙상 감지는 Accuracy 단독보다 Precision, Recall, F1-score를 함께 보는 것이 맞습니다.
