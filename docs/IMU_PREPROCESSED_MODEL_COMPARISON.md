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

## 성능 비교

| Task | Model | Accuracy | Precision | Recall | F1-score | Single inference ms | Batch per sequence ms | Train sec | Params | Threshold |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| IMU 낙상 전처리 후 | RNN | 0.7189 | 0.5787 | 0.7807 | 0.6647 | 0.230 | 0.0121 | 32.4 | 13505 | 0.83 |
| IMU 낙상 전처리 후 | GRU | 0.7727 | 0.6408 | 0.8262 | 0.7218 | 0.572 | 0.0362 | 96.1 | 40129 | 0.88 |
| IMU 낙상 전처리 후 | LSTM | 0.7638 | 0.6178 | 0.8871 | 0.7283 | 0.563 | 0.0423 | 122.7 | 53441 | 0.84 |
| IMU 낙상 전처리 후 | TRANSFORMER | 0.7820 | 0.6398 | 0.8903 | 0.7446 | 0.305 | 0.0374 | 171.1 | 67969 | 0.73 |

## 결론

- 전처리 후 기준 최고 F1 모델: `TRANSFORMER`
- 최고 F1-score: `0.7446`
- 해당 모델 Accuracy: `0.7820`
- 낙상 감지는 Accuracy 단독보다 Precision, Recall, F1-score를 함께 보는 것이 맞습니다.
