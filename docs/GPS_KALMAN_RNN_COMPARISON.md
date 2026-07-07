# GPS Kalman Filter RNN 성능 비교

## 실험 조건

- Source: `../ICCAS_final_data.xlsx`
- Model: GPS Wandering RNN
- Sequence length: `16`
- Measurement noise: `5.0` m
- Process noise: `1.0`

## 결과

| Version | Accuracy | Precision | Recall | F1-score | Threshold |
| --- | ---: | ---: | ---: | ---: | ---: |
| Raw GPS | 0.9156 | 0.8593 | 0.9941 | 0.9218 | 0.76 |
| Kalman GPS | 0.9494 | 0.9337 | 0.9676 | 0.9503 | 0.45 |

## 변화량

- Accuracy: `+0.0338`
- Precision: `+0.0744`
- Recall: `-0.0265`
- F1-score: `+0.0286`

## 해석

칼만 필터는 GPS 좌표의 순간 튐을 줄여 `x_m`, `y_m`, `dx_m`, `dy_m`, `speed_mps`를 부드럽게 만듭니다.
다만 배회 감지는 경로 이탈과 반복 이동의 형태도 중요하므로, 필터가 너무 강하면 실제 이동 변화까지 완화되어 성능이 내려갈 수 있습니다.