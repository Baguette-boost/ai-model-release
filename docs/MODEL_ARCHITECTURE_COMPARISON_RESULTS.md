# RNN, GRU, LSTM, Transformer 실제 학습 비교 결과

## 실험 조건

- Source: `../ICCAS_final_data.xlsx`
- Device: `cpu`
- Sequence length: `16`
- Epochs: `15`
- Batch size: `256`
- Split: scenario별 chronological split, train 0.70, validation 0.15, test remainder

## 전체 비교

| Task | Model | Accuracy | Precision | Recall | F1-score | Single inference ms | Batch per sequence ms | Train sec | Params |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gps_wandering | RNN | 0.9156 | 0.8593 | 0.9941 | 0.9218 | 0.092 | 0.0042 | 4.1 | 13185 |
| gps_wandering | GRU | 0.8823 | 0.8773 | 0.8891 | 0.8832 | 0.202 | 0.0125 | 10.7 | 39169 |
| gps_wandering | LSTM | 0.8215 | 0.8499 | 0.7810 | 0.8140 | 0.198 | 0.0140 | 12.9 | 52161 |
| gps_wandering | TRANSFORMER | 0.8039 | 0.8168 | 0.7837 | 0.7999 | 0.283 | 0.0195 | 25.0 | 67649 |
| imu_fall | RNN | 0.9105 | 0.6016 | 0.5648 | 0.5826 | 0.091 | 0.0040 | 4.3 | 13505 |
| imu_fall | GRU | 0.9110 | 0.6504 | 0.4230 | 0.5126 | 0.199 | 0.0124 | 10.3 | 40129 |
| imu_fall | LSTM | 0.9521 | 0.7788 | 0.7922 | 0.7855 | 0.199 | 0.0139 | 12.6 | 53441 |
| imu_fall | TRANSFORMER | 0.9299 | 0.7863 | 0.5037 | 0.6140 | 0.278 | 0.0203 | 24.9 | 67969 |

## Task별 최고 F1

- gps_wandering: `RNN` F1 0.9218, Accuracy 0.9156, single inference 0.092 ms
- imu_fall: `LSTM` F1 0.7855, Accuracy 0.9521, single inference 0.199 ms

## 해석

- Accuracy는 전체 정답률이고, F1-score는 위험 클래스의 Precision과 Recall 균형을 나타냅니다.
- 낙상과 배회 감지는 위험 상황을 놓치지 않는 것이 중요하므로 Recall과 F1-score를 함께 봐야 합니다.
- Single inference ms는 실시간으로 센서 포인트가 들어왔을 때 한 시퀀스를 추론하는 시간입니다.
- Batch per sequence ms는 많은 시퀀스를 한꺼번에 평가할 때 시퀀스 1개당 평균 처리 시간입니다.
