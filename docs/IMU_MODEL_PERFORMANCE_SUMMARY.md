# IMU 낙상 감지 모델 성능 지표 종합

## 핵심 결론

최종 적용 모델은 `전처리 feature 기반 LSTM`이다.

```text
Model file : models/iccas_final_hybrid_lstm_imu_fall.pt
Input      : 50 timesteps x 12 features
Threshold  : 0.35
```

이름에는 과거 실험명으로 `hybrid`가 남아 있지만, 실제 최종 weight는 LSTM 점수만 사용한다.

```text
hybrid_lstm_weight      = 1.0
hybrid_algorithm_weight = 0.0
```

따라서 발표에서는 `Hybrid LSTM`이 아니라 **전처리 feature 기반 LSTM 낙상 감지 모델**이라고 설명하는 것이 정확하다.

## 1. 최종 적용 모델 성능

| Model | Accuracy | Precision | Recall | F1-score | Threshold | TP | FP | TN | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Final LSTM | 0.8799 | 0.8498 | 0.8863 | 0.8677 | 0.35 | 4147 | 733 | 5120 | 532 |

해석:

- Recall `0.8863`: 실제 낙상 중 약 88.6%를 탐지했다.
- Precision `0.8498`: 낙상으로 판단한 것 중 약 85.0%가 실제 낙상이었다.
- F1-score `0.8677`: 낙상 탐지와 오탐 억제 사이의 균형이 가장 좋았다.

## 2. 최종 LSTM의 데이터셋별 성능

| Dataset | Accuracy | Precision | Recall | F1-score | TP | FP | TN | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ICCAS | 0.9881 | 0.9259 | 0.9709 | 0.9479 | 100 | 8 | 817 | 3 |
| SisFall | 0.8694 | 0.8481 | 0.8844 | 0.8659 | 4047 | 725 | 4303 | 529 |

해석:

- ICCAS 직접 취득 데이터에서는 F1-score가 `0.9479`로 높았다.
- SisFall에서는 F1-score가 `0.8659`로 전체 성능과 유사했다.
- 직접 취득 데이터와 오픈 데이터가 완전히 같은 분포는 아니므로, 데이터셋별 성능을 함께 제시하는 것이 좋다.

## 3. 1D-CNN / CNN-LSTM 분리 실험

아래 두 모델은 기존 LSTM 최종 모델과 분리된 실험 폴더에서 학습했다.

```text
experiments/imu_fall_1d_cnn/
experiments/imu_fall_cnn_lstm/
```

| Model | Accuracy | Precision | Recall | F1-score | Threshold | Single inference ms | Train sec | Params |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1D-CNN | 0.8312 | 0.7261 | 0.8464 | 0.7817 | 0.90 | 0.4230 | 380.5 | 107,777 |
| CNN-LSTM | 0.8277 | 0.7051 | 0.8890 | 0.7864 | 0.41 | 0.7048 | 221.6 | 77,025 |

해석:

- CNN-LSTM은 1D-CNN보다 F1-score가 약간 높다.
- CNN-LSTM은 Recall `0.8890`으로 낙상 탐지율은 높지만, Precision이 낮아 오탐이 많다.
- 최종 LSTM의 F1-score `0.8677`보다 낮으므로, 최종 모델 교체 근거로 보기는 어렵다.
- CNN 계열은 `향후 CNN-LSTM 고도화 가능성` 또는 `local impact pattern 비교 실험`으로 설명하는 것이 적절하다.

## 4. 전처리 feature 추가 전/후 비교 실험

전처리 전 입력:

```text
roll, pitch, yaw, ax, ay, az, wx, wy, wz
```

전처리 후 입력:

```text
roll, pitch, yaw, ax, ay, az, wx, wy, wz,
accel_norm, gyro_norm, dt_s
```

| Model | Accuracy Before | Accuracy After | Precision Before | Precision After | Recall Before | Recall After | F1 Before | F1 After |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| RNN | 0.6709 | 0.6893 | 0.5322 | 0.5441 | 0.6935 | 0.8333 | 0.6022 | 0.6584 |
| GRU | 0.7190 | 0.7198 | 0.5716 | 0.5735 | 0.8695 | 0.8590 | 0.6898 | 0.6878 |
| LSTM | 0.7379 | 0.7148 | 0.6009 | 0.5717 | 0.8054 | 0.8228 | 0.6882 | 0.6746 |
| Transformer | 0.7207 | 0.7889 | 0.5837 | 0.6788 | 0.7762 | 0.7832 | 0.6663 | 0.7273 |

정확한 해석:

- F1-score는 RNN과 Transformer에서 상승했다.
- Recall은 RNN, LSTM, Transformer에서 상승했다.
- GRU와 LSTM의 F1-score는 이 비교 실험에서는 소폭 하락했다.
- 따라서 `전처리 후 모든 모델 성능이 좋아졌다`라고 말하면 안 된다.

발표용 표현:

> 전처리 feature 추가는 모든 모델에서 일괄적인 F1-score 향상을 보장하지는 않았지만, 일부 모델에서 F1과 Recall 개선을 확인했다. 특히 낙상 감지에서 중요한 Recall은 4개 모델 중 3개 모델에서 개선되었다.

## 5. 모델별 최종 비교 표

주의: 아래 표는 모든 모델이 완전히 동일한 split에서 학습된 것은 아니므로, 최종 적용 모델 선정용 참고표로 사용한다.

| Category | Model | Accuracy | Precision | Recall | F1-score | 비고 |
|---|---|---:|---:|---:|---:|---|
| Final model | LSTM | 0.8799 | 0.8498 | 0.8863 | 0.8677 | 최종 적용 모델 |
| Isolated experiment | 1D-CNN | 0.8312 | 0.7261 | 0.8464 | 0.7817 | local impact pattern 실험 |
| Isolated experiment | CNN-LSTM | 0.8277 | 0.7051 | 0.8890 | 0.7864 | Recall 높지만 Precision 낮음 |
| Preprocessing comparison | RNN after preprocessing | 0.6893 | 0.5441 | 0.8333 | 0.6584 | 비교 실험 |
| Preprocessing comparison | GRU after preprocessing | 0.7198 | 0.5735 | 0.8590 | 0.6878 | 비교 실험 |
| Preprocessing comparison | LSTM after preprocessing | 0.7148 | 0.5717 | 0.8228 | 0.6746 | 비교 실험 |
| Preprocessing comparison | Transformer after preprocessing | 0.7889 | 0.6788 | 0.7832 | 0.7273 | 비교 실험 |

## 6. 최종 선정 이유

최종 모델로 LSTM을 선택한 이유:

1. F1-score가 가장 높다.
2. Recall도 높아 낙상 미탐을 줄이는 데 유리하다.
3. 50-step IMU sequence에서 낙상 전후의 시간 패턴을 학습할 수 있다.
4. 1D-CNN과 CNN-LSTM보다 Precision/F1 균형이 좋다.
5. 최종 서버/데모 적용 모델로 이미 정리되어 있다.

최종 발표 문장:

> 여러 모델을 비교한 결과, 최종 적용 모델은 전처리 feature 기반 LSTM으로 선정하였다. 1D-CNN과 CNN-LSTM은 낙상 충격 패턴을 학습할 수 있었지만, Precision과 F1-score 측면에서 최종 LSTM보다 낮았다. 최종 LSTM은 테스트 기준 Accuracy 0.8799, Recall 0.8863, F1-score 0.8677을 기록하여 낙상 탐지와 오탐 억제 사이에서 가장 안정적인 성능을 보였다.
