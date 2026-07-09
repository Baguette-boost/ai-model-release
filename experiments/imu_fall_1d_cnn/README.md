# IMU Fall 1D-CNN Experiment

이 폴더는 기존 LSTM 최종 모델과 분리한 1D-CNN 단독 실험 결과입니다.

## Experiment Setup

- Source: `../data/iccas_sensor_lstm/imu_fall_preprocessed.csv`
- Device: `cpu`
- Sequence length: `50`
- Sequence stride: `4`
- Feature count: `12`
- Features: `roll, pitch, yaw, ax, ay, az, wx, wy, wz, accel_norm, gyro_norm, dt_s`
- Epochs: `15`
- Batch size: `256`
- Split: `SisFall group split, ICCAS chronological split`

## Test Metrics

| Model | Accuracy | Precision | Recall | F1-score | Threshold | TP | FP | TN | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1D-CNN | 0.8312 | 0.7261 | 0.8464 | 0.7817 | 0.90 | 4407 | 1662 | 7720 | 800 |

## Speed

- Single sequence inference: `0.4230 ms`
- Batch per sequence inference: `0.208043 ms`
- Training time: `380.5 s`
- Parameters: `107,777`

## Interpretation

- 1D-CNN은 LSTM처럼 긴 순서를 순환적으로 기억하지는 않지만, 낙상 순간의 짧은 충격/회전 local pattern을 빠르게 잡는 데 적합합니다.
- 최종 적용 모델을 바꾸기보다는 LSTM과 비교하는 보조 실험 또는 CNN-LSTM 확장 근거로 사용하는 것이 좋습니다.
- 기존 최종 LSTM(`iccas_final_hybrid_lstm_imu_fall`)은 별도 split/학습 파이프라인에서 F1-score 0.8677을 기록했습니다. 따라서 이 1D-CNN 결과는 최종 모델 교체 근거라기보다, CNN 계열이 local impact pattern을 학습할 수 있음을 보여주는 분리 실험으로 해석하는 것이 정확합니다.
