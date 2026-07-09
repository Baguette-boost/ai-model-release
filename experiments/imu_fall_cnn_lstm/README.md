# IMU Fall CNN-LSTM Experiment

이 폴더는 기존 LSTM 최종 모델과 분리한 CNN-LSTM 단독 실험 결과입니다.

## Experiment Setup

- Source: `../data/iccas_sensor_lstm/imu_fall_preprocessed.csv`
- Device: `cpu`
- Sequence length: `50`
- Sequence stride: `4`
- Feature count: `12`
- Features: `roll, pitch, yaw, ax, ay, az, wx, wy, wz, accel_norm, gyro_norm, dt_s`
- CNN channels: `48`
- LSTM hidden size: `64`
- Epochs: `15`
- Batch size: `256`
- Split: `SisFall group split, ICCAS chronological split`

## Test Metrics

| Model | Accuracy | Precision | Recall | F1-score | Threshold | TP | FP | TN | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CNN-LSTM | 0.8277 | 0.7051 | 0.8890 | 0.7864 | 0.41 | 4629 | 1936 | 7446 | 578 |

## Speed

- Single sequence inference: `0.7048 ms`
- Batch per sequence inference: `0.086807 ms`
- Training time: `221.6 s`
- Parameters: `77,025`

## Interpretation

- CNN-LSTM은 CNN이 낙상 순간의 local impact/rotation pattern을 먼저 추출하고, LSTM이 충격 전후의 시간 흐름을 학습하는 구조입니다.
- 기존 최종 LSTM(`iccas_final_hybrid_lstm_imu_fall`)은 별도 split/학습 파이프라인에서 F1-score 0.8677을 기록했습니다. 따라서 이 실험은 최종 모델 교체 여부를 판단하기 위한 독립 비교 자료로 사용합니다.
