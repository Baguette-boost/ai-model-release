# IMU Fall Threshold Search 결과

## 목적

직접 취득 장비의 ESP32 fall-suspect threshold에 적합한 값을 찾기 위해 현재 IMU 낙상 데이터셋에서 grid search를 수행했습니다.

## 데이터셋

- Source: `../data/iccas_sensor_lstm/final_iccas_sisfall_imu_merged.csv`
- Sequence length: `50`
- Split: `test split from same group/chronological split used by IMU fall training`
- Test sequences: `10532`
- Test fall positives: `4679`

## 현재 기준값 성능

- FALL_IMPACT_G: `2.5`
- FALL_FREE_G: `0.6`
- gyro_dps: `250.0`
- tilt_deg: `45.0`
- still_accel_std: `0.35`
- still_gyro_dps: `80.0`
- Algorithm score threshold: `0.05`
- Accuracy: `0.4443`
- Precision: `0.4443`
- Recall: `1.0000`
- F1-score: `0.6152`

## Best F1 threshold

- FALL_IMPACT_G: `2.0`
- FALL_FREE_G: `0.7`
- gyro_dps: `300.0`
- tilt_deg: `45.0`
- still_accel_std: `0.35`
- still_gyro_dps: `80.0`
- Algorithm score threshold: `0.07`
- Accuracy: `0.4457`
- Precision: `0.4448`
- Recall: `0.9985`
- F1-score: `0.6155`
- Confusion Matrix: TP `4672`, FP `5831`, TN `22`, FN `7`

## 추천

ESP32에서는 단독 최종 판단보다 fall-suspect 트리거로 사용하고, 서버 LSTM이 최종 낙상을 판단하는 구조를 권장합니다.

```cpp
#define SAMPLE_MS     40
#define IMU_BUF_N     50
#define FALL_IMPACT_G 2.0f
#define FALL_FREE_G   0.7f
#define FALL_COOLDOWN 5000
```

## 상위 후보

| Rank | Impact G | Free G | Gyro dps | Still gyro | Precision | Recall | F1 | FP | FN |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2.0 | 0.7 | 300 | 80 | 0.4448 | 0.9985 | 0.6155 | 5831 | 7 |
| 2 | 2.0 | 0.5 | 300 | 80 | 0.4448 | 0.9983 | 0.6154 | 5831 | 8 |
| 3 | 2.0 | 0.6 | 300 | 80 | 0.4448 | 0.9983 | 0.6154 | 5831 | 8 |
| 4 | 2.2 | 0.7 | 300 | 80 | 0.4448 | 0.9983 | 0.6154 | 5831 | 8 |
| 5 | 2.0 | 0.7 | 200 | 80 | 0.4448 | 0.9981 | 0.6153 | 5830 | 9 |
| 6 | 2.0 | 0.5 | 200 | 80 | 0.4444 | 0.9998 | 0.6153 | 5849 | 1 |
| 7 | 2.0 | 0.6 | 200 | 80 | 0.4444 | 0.9998 | 0.6153 | 5849 | 1 |
| 8 | 2.2 | 0.5 | 300 | 80 | 0.4447 | 0.9981 | 0.6153 | 5831 | 9 |
| 9 | 2.2 | 0.6 | 300 | 80 | 0.4447 | 0.9981 | 0.6153 | 5831 | 9 |
| 10 | 3.0 | 0.5 | 200 | 80 | 0.4443 | 1.0000 | 0.6153 | 5852 | 0 |