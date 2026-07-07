# RNN, GRU, LSTM, Transformer 비교 분석 자료

## 1. 비교 목적

본 프로젝트는 GPS와 IMU 센서 시계열 데이터를 사용해 배회 감지와 낙상 감지를 수행합니다. 따라서 모델 비교는 단순히 정확도만 보는 것이 아니라, 아래 기준을 함께 봐야 합니다.

```text
1. 짧은 시계열에서 실시간 추론이 가능한가
2. GPS 이동 경로처럼 시간 순서가 중요한 데이터를 잘 처리하는가
3. IMU 낙상처럼 순간적인 충격 패턴을 잘 잡는가
4. 데이터가 많지 않아도 안정적으로 학습되는가
5. 모바일/서버 환경에서 처리 비용이 적절한가
6. 백엔드로 실시간 결과를 보내기에 지연 시간이 낮은가
```

현재 최종 시스템은 `GPS 배회 감지`와 `IMU 낙상 감지`를 분리했고, 서버 적용 모델은 LSTM 기반으로 구성했습니다.

```text
GPS Wandering:
models/iccas_final_lstm_gps_wandering.pt

IMU Fall:
models/iccas_final_sisfall_lstm_imu_fall.pt
```

## 2. 모델별 핵심 개념

| Model | 핵심 구조 | 장점 | 단점 | 본 프로젝트 적합도 |
| --- | --- | --- | --- | --- |
| Vanilla RNN | 이전 hidden state를 다음 시점으로 전달 | 구조가 단순하고 빠름 | 긴 시퀀스에서 기울기 소실 문제가 큼 | 낮음 |
| GRU | update/reset gate로 기억 제어 | LSTM보다 가볍고 빠름 | 복잡한 장기 패턴 표현력은 LSTM보다 약할 수 있음 | 중간~높음 |
| LSTM | input/forget/output gate와 cell state 사용 | 장단기 패턴을 안정적으로 학습 | GRU보다 연산량이 조금 큼 | 높음 |
| Transformer | self-attention으로 전체 시점을 동시에 참조 | 긴 시퀀스와 대규모 데이터에서 강함 | 데이터와 연산량이 많이 필요하고 실시간 소형 입력에는 과할 수 있음 | 조건부 적합 |

## 3. 센서 데이터 관점 비교

### GPS 배회 감지

GPS 배회 감지는 좌표 하나만으로 판단하기 어렵고, 시간에 따른 이동 방향, 거리, 반복 방문, 경로 이탈 흐름을 봐야 합니다.

| Model | GPS 배회 감지 관점 |
| --- | --- |
| RNN | 짧은 경로에서는 동작 가능하지만, 반복 경로와 장기 이동 흐름을 안정적으로 기억하기 어렵습니다. |
| GRU | 경량 모델로 실시간성이 좋고, 짧은 배회 패턴에는 충분히 사용할 수 있습니다. |
| LSTM | 이동 경로의 누적 흐름을 기억하기 좋아 현재 데이터 규모와 실시간 서버 구조에 적합합니다. |
| Transformer | 충분한 GPS 데이터가 많으면 경로 전체 관계를 잘 볼 수 있지만, 현재 규모에서는 과적합 위험과 연산 비용이 있습니다. |

### IMU 낙상 감지

IMU 낙상 감지는 가속도와 자이로의 순간 변화가 중요합니다. 낙상은 짧은 시간에 충격, 자세 변화, 정지 또는 회복 움직임이 이어지는 패턴입니다.

| Model | IMU 낙상 감지 관점 |
| --- | --- |
| RNN | 순간 패턴은 일부 잡을 수 있지만 안정성이 낮습니다. |
| GRU | 빠르고 가벼워 실시간 낙상 감지 후보로 좋습니다. |
| LSTM | 충격 전후 흐름을 함께 보기 좋아 현재 최종 낙상 모델에 적합합니다. |
| Transformer | 대규모 착용자 데이터가 있으면 성능 개선 가능성이 있으나, 현재 데이터에서는 튜닝 부담이 큽니다. |

## 4. 성능 비교 시 봐야 하는 지표

정확도만 보면 안 됩니다. 낙상과 배회는 정상 데이터가 더 많거나 특정 클래스가 적을 수 있어 Accuracy가 높아도 실제 위험 상황을 놓칠 수 있습니다.

| Metric | 의미 | 본 프로젝트에서 중요한 이유 |
| --- | --- | --- |
| Accuracy | 전체 중 맞춘 비율 | 전체적인 기준이지만 단독 사용은 위험합니다. |
| Precision | 위험이라고 예측한 것 중 실제 위험 비율 | 오탐을 줄이는 데 중요합니다. |
| Recall | 실제 위험 중 잡아낸 비율 | 낙상/배회를 놓치지 않는 데 가장 중요합니다. |
| F1-score | Precision과 Recall의 균형 | 모델 비교의 대표 지표로 사용하기 좋습니다. |
| Latency | 한 번 추론하는 데 걸리는 시간 | 실시간 서버 전송에 중요합니다. |
| Model Size | 모델 파일 크기 | 배포와 모바일/엣지 확장 시 중요합니다. |

발표에서는 아래처럼 말하면 좋습니다.

```text
이 프로젝트에서는 위험 상황을 놓치는 것이 더 큰 문제이므로 Recall과 F1-score를 중요하게 봤습니다. 다만 보호자 알림 서비스에서는 오탐이 너무 많아도 신뢰도가 떨어지기 때문에 Precision도 함께 확인했습니다.
```

## 5. 현재 LSTM 최종 성능 기준

현재 프로젝트에서 실제로 학습 완료된 최종 LSTM 성능은 아래와 같습니다.

| Task | Model | Accuracy | Precision | Recall | F1-score |
| --- | --- | ---: | ---: | ---: | ---: |
| GPS Wandering, 전체 sheet 기준 | LSTM | 0.9115 | 0.8586 | 0.9854 | 0.9177 |
| GPS Wandering, 서버 task 기준 | LSTM | 0.9544 | 0.9368 | 0.9854 | 0.9605 |
| GPS Wandering, moving route only | LSTM | 0.9878 | 0.9945 | 0.9854 | 0.9899 |
| IMU Fall, ICCAS + SisFall | BiLSTM + Attention | 0.8814 | 0.8879 | 0.8392 | 0.8629 |

주의할 점:

```text
위 표는 현재 실제 학습된 LSTM 계열 모델의 결과입니다. RNN, GRU, Transformer의 수치는 동일 데이터, 동일 split, 동일 sequence length로 별도 학습해야 공정하게 비교할 수 있습니다.
```

동일 조건으로 RNN, GRU, LSTM, Transformer를 새로 학습한 비교 결과는 아래 문서에 정리했습니다.

```text
docs/MODEL_ARCHITECTURE_COMPARISON_RESULTS.md
```

비교 실험 실행 명령어:

```bash
python scripts/compare_sequence_models.py \
  --source ../ICCAS_final_data.xlsx \
  --epochs 15 \
  --batch-size 256 \
  --device auto
```

## 6. 공정한 비교 실험 설계

모델 비교를 하려면 아래 조건을 반드시 동일하게 맞춰야 합니다.

| 조건 | 고정 값 또는 원칙 |
| --- | --- |
| 데이터 | ICCAS_final_data.xlsx 기반 전처리 데이터 |
| 데이터 분할 | 같은 train/validation/test split 사용 |
| 입력 feature | GPS 모델은 GPS feature만, IMU 모델은 IMU/Gyro feature만 사용 |
| sequence length | GPS 16, IMU Fall 32 기준 |
| batch size | 동일 batch size 사용 |
| epoch | 동일 epoch 또는 early stopping 기준 사용 |
| threshold | validation set에서 F1이 가장 높은 threshold 선택 |
| 평가 지표 | Accuracy, Precision, Recall, F1, confusion matrix, latency |

추천 비교 표 형식:

| Task | Model | Accuracy | Precision | Recall | F1 | Latency | Model Size | 비고 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| GPS Wandering | RNN | 측정 필요 | 측정 필요 | 측정 필요 | 측정 필요 | 측정 필요 | 측정 필요 | 기준 모델 |
| GPS Wandering | GRU | 측정 필요 | 측정 필요 | 측정 필요 | 측정 필요 | 측정 필요 | 측정 필요 | 경량 후보 |
| GPS Wandering | LSTM | 0.9544 | 0.9368 | 0.9854 | 0.9605 | 측정 필요 | 측정 필요 | 현재 서버 적용 |
| GPS Wandering | Transformer | 측정 필요 | 측정 필요 | 측정 필요 | 측정 필요 | 측정 필요 | 측정 필요 | 데이터 증가 시 후보 |
| IMU Fall | RNN | 측정 필요 | 측정 필요 | 측정 필요 | 측정 필요 | 측정 필요 | 측정 필요 | 기준 모델 |
| IMU Fall | GRU | 측정 필요 | 측정 필요 | 측정 필요 | 측정 필요 | 측정 필요 | 측정 필요 | 경량 후보 |
| IMU Fall | BiLSTM + Attention | 0.8814 | 0.8879 | 0.8392 | 0.8629 | 측정 필요 | 측정 필요 | 현재 서버 적용 |
| IMU Fall | Transformer | 측정 필요 | 측정 필요 | 측정 필요 | 측정 필요 | 측정 필요 | 측정 필요 | 대규모 데이터 후보 |

## 7. 예상 비교 결과 해석

실제 학습 전 예상되는 경향은 아래와 같습니다.

| Model | 예상 경향 |
| --- | --- |
| RNN | 가장 빠르지만 F1이 낮을 가능성이 높습니다. 발표에서는 baseline으로 쓰기 좋습니다. |
| GRU | LSTM과 비슷한 성능이 나오면서 더 가벼울 수 있습니다. 서버 비용을 줄이는 후보입니다. |
| LSTM | 현재 데이터 규모와 실시간 시계열 문제에서 가장 균형 잡힌 선택입니다. |
| Transformer | 데이터가 충분히 커지면 좋아질 수 있지만, 현재는 과적합과 지연 시간 문제가 생길 수 있습니다. |

## 8. 왜 현재는 LSTM을 선택했는가

현재 시스템에서 LSTM을 선택한 이유는 아래와 같습니다.

```text
1. GPS와 IMU 모두 시간 순서가 중요한 센서 데이터입니다.
2. RNN보다 장기 의존성을 안정적으로 처리합니다.
3. Transformer보다 적은 데이터와 낮은 연산량으로 학습 가능합니다.
4. 실시간 백엔드 전송 구조에서 추론 지연이 작습니다.
5. GPS 배회와 IMU 낙상 모두에서 sequence 기반 판단이 필요합니다.
```

발표 멘트:

```text
Transformer는 최신 모델이지만 항상 최선은 아닙니다. 본 프로젝트는 실시간 센서 시계열, 제한된 데이터, 빠른 서버 응답이 중요하기 때문에 LSTM이 성능과 안정성의 균형이 가장 좋았습니다.
```

## 9. 발표용 슬라이드 구성

### Slide 1. 비교 목적

```text
GPS와 IMU는 시간 순서가 중요한 센서 데이터이므로 시계열 모델 비교가 필요합니다.
비교 대상은 RNN, GRU, LSTM, Transformer입니다.
```

### Slide 2. 모델별 특징

| Model | 특징 | 장점 | 한계 |
| --- | --- | --- | --- |
| RNN | 기본 순환 구조 | 빠름 | 장기 기억 약함 |
| GRU | 2개 gate 구조 | 가볍고 안정적 | LSTM보다 표현력 제한 가능 |
| LSTM | 3개 gate + cell state | 장단기 패턴 안정적 | GRU보다 약간 무거움 |
| Transformer | self-attention | 긴 시퀀스 강함 | 데이터/연산량 요구 큼 |

### Slide 3. 프로젝트 적용 관점

```text
GPS 배회: 경로 누적 흐름과 반복 이동 패턴이 중요
IMU 낙상: 충격 전후의 짧은 시간 변화가 중요
실시간 서버: 낮은 latency와 안정적인 추론이 중요
```

### Slide 4. 현재 선택

```text
최종 적용 모델은 LSTM 계열입니다.
GPS는 LSTM으로 배회 여부를 판단하고, IMU는 BiLSTM + Attention으로 낙상 여부를 판단합니다.
```

### Slide 5. 결론

```text
RNN은 baseline, GRU는 경량화 후보, LSTM은 현재 최적 선택, Transformer는 데이터가 더 많아졌을 때 확장 후보입니다.
```

## 10. 최종 결론

현재 프로젝트에서는 LSTM이 가장 현실적인 선택입니다. 이유는 GPS와 IMU가 모두 시간 순서가 중요한 데이터이고, 현재 데이터 규모에서 Transformer보다 안정적으로 학습되며, RNN보다 장기 패턴을 잘 기억하기 때문입니다.

다음 단계로는 같은 데이터 split에서 RNN, GRU, Transformer를 실제 학습시켜 F1-score와 latency를 함께 비교하면 됩니다. 이때 최종 선택 기준은 단순 Accuracy가 아니라 `F1-score`, `Recall`, `Latency`, `Model Size`를 함께 보는 방식이 적절합니다.
