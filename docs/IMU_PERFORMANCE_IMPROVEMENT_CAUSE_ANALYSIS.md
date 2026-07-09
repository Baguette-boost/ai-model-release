# IMU 낙상 감지 성능 변화 원인 분석

## 목적

본 문서는 IMU 낙상 감지 모델의 성능이 왜 변화했는지 설명하기 위해, 전처리 전/후 데이터가 어떻게 달라졌는지와 모델별 성능 변화가 어떤 의미를 갖는지 정리한다.

중요한 점은 다음과 같다.

- 전처리는 모든 모델의 모든 지표를 항상 향상시키지는 않았다.
- 전처리의 핵심 효과는 `원본 센서값을 모델이 해석하기 쉬운 물리 기반 feature로 확장`한 것이다.
- 최종 적용 모델은 전처리된 12차원 IMU feature를 사용하는 LSTM이다.

## 1. 데이터가 어떻게 바뀌었는가

전처리 전 입력은 원본 IMU 9개 feature였다.

```text
roll, pitch, yaw,
ax, ay, az,
wx, wy, wz
```

전처리 후에는 다음 3개 feature를 추가했다.

```text
accel_norm = sqrt(ax^2 + ay^2 + az^2)
gyro_norm  = sqrt(wx^2 + wy^2 + wz^2)
dt_s       = (t_i - t_{i-1}) / 1000
```

따라서 모델 입력은 9개에서 12개로 확장되었다.

| 항목 | 전처리 전 | 전처리 후 |
|---|---:|---:|
| Rows | 349,560 | 349,560 |
| Columns | 20 | 22 |
| Model-ready features | 9 / 12 | 12 / 12 |
| Missing numeric rate | 0.0 | 0.0 |
| accel_norm | 없음 | 있음 |
| gyro_norm | 없음 | 있음 |
| dt_s | 없음 | 있음 |
| fall_target | 없음 | 있음 |

해석:

- row 수는 변하지 않았다. 즉, 데이터를 임의로 늘린 것이 아니다.
- 성능 변화의 원인은 데이터 양 증가가 아니라 입력 feature 의미 강화에 있다.
- `accel_norm`, `gyro_norm`, `dt_s`, `fall_target`이 추가되며 모델이 낙상 판단에 필요한 물리량과 라벨을 더 명확하게 사용할 수 있게 되었다.

## 2. 왜 accel_norm이 필요한가

원본 가속도는 축별 값이다.

```text
ax, ay, az
```

하지만 낙상은 특정 축 하나만으로 나타나지 않는다. 사용자의 착용 방향, 기기 회전, 넘어지는 방향에 따라 큰 충격이 x/y/z 중 어느 축에 나타날지 달라진다.

따라서 3축 가속도를 하나의 크기로 합친다.

```text
accel_norm = sqrt(ax^2 + ay^2 + az^2)
```

이 값은 방향과 무관하게 전체 충격 크기를 나타낸다.

예를 들어 같은 충격이라도 센서 방향이 바뀌면 `ax`, `ay`, `az` 분포는 달라질 수 있다. 그러나 `accel_norm`은 전체 충격 크기를 유지하므로 낙상 패턴을 더 안정적으로 표현한다.

## 3. 왜 gyro_norm이 필요한가

낙상은 단순히 충격만 발생하는 것이 아니라 몸의 회전, 자세 변화가 함께 발생한다.

원본 자이로는 축별 회전 속도다.

```text
wx, wy, wz
```

이를 전체 회전량으로 변환한다.

```text
gyro_norm = sqrt(wx^2 + wy^2 + wz^2)
```

이 값은 낙상 순간의 급격한 회전 또는 자세 변화 가능성을 모델이 보기 쉽게 만든다.

## 4. 왜 dt_s가 필요한가

ICCAS와 SisFall은 샘플링 간격이 다르다.

| Dataset | dt_s p50 |
|---|---:|
| ICCAS normal | 0.0400s |
| ICCAS fall | 0.0400s |
| SisFall normal | 0.0200s |
| SisFall fall | 0.0200s |

즉, ICCAS는 약 25Hz, SisFall은 약 50Hz이다.

같은 50-step window라도 실제 시간 길이가 다를 수 있으므로, `dt_s`를 feature로 넣어 모델이 샘플링 간격 차이를 함께 학습하도록 했다.

## 5. 직접 취득 데이터와 오픈 데이터는 얼마나 비슷한가

ICCAS 직접 취득 데이터와 SisFall 오픈 데이터의 가속도 크기 중앙값은 매우 유사하다.

| Class | ICCAS accel_norm p50 | SisFall accel_norm p50 | Difference |
|---|---:|---:|---:|
| normal | 1.0029 | 1.0096 | 0.0067g |
| fall | 1.0041 | 1.0010 | 0.0031g |

해석:

- 두 데이터 모두 중심값이 약 1g 근처다.
- 기본 가속도 스케일은 크게 다르지 않다.
- 따라서 병합 학습의 기본 조건은 만족한다고 볼 수 있다.

다만 완전히 같은 분포는 아니다.

| 항목 | 관찰 결과 | 처리 방식 |
|---|---|---|
| accel_norm 중심값 | 차이 작음 | 병합 학습 가능 |
| accel_norm p95 | 일부 차이 있음 | scaling 필요 |
| accel_delta | normal/fall tail 차이 있음 | threshold 단독 부적합 |
| gyro_norm | 데이터셋 간 차이 큼 | robust scaling 필요 |
| dt_s | 25Hz vs 50Hz 차이 | dt_s feature 추가 |

## 6. threshold 단독 판단이 어려운 이유

가속도 변화량만 보면 fall과 normal이 깔끔하게 분리되지 않았다.

| Dataset | Class | accel_norm p95 | accel_delta p95 | change_rate p95 |
|---|---|---:|---:|---:|
| ICCAS | fall | 1.2840 | 0.2123 | 5.2102 |
| ICCAS | normal | 1.4684 | 0.3243 | 7.7447 |
| SisFall | fall | 1.3671 | 0.2887 | 14.4338 |
| SisFall | normal | 1.7078 | 0.4784 | 23.9183 |

Fall vs normal 효과크기:

```text
accel_norm         Cohen's d = -0.064
accel_delta        Cohen's d = -0.124
accel_change_rate  Cohen's d = -0.096
```

해석:

- 효과크기가 0에 가깝다.
- 정상 동작에서도 큰 가속도 변화가 발생한다.
- 따라서 `accel_norm > 특정 임계값` 같은 단순 규칙만으로는 낙상을 안정적으로 구분하기 어렵다.

이 때문에 최종 모델은 단일 시점이 아니라 50-step window를 사용한다.

```text
충격 발생 -> 회전 변화 -> 자세 변화 -> 정지 상태
```

이 흐름을 LSTM이 시간 문맥으로 학습하도록 만든 것이다.

## 7. 전처리 전/후 모델 성능 변화

같은 CSV, 같은 sequence length, 같은 train/validation/test split 조건에서 9개 feature와 12개 feature를 비교했다.

| Model | F1 Before | F1 After | Delta | Recall Before | Recall After | Delta |
|---|---:|---:|---:|---:|---:|---:|
| RNN | 0.6022 | 0.6584 | +0.0562 | 0.6935 | 0.8333 | +0.1399 |
| GRU | 0.6898 | 0.6878 | -0.0020 | 0.8695 | 0.8590 | -0.0105 |
| LSTM | 0.6882 | 0.6746 | -0.0136 | 0.8054 | 0.8228 | +0.0175 |
| Transformer | 0.6663 | 0.7273 | +0.0609 | 0.7762 | 0.7832 | +0.0070 |

정확한 해석:

- F1-score는 RNN과 Transformer에서 상승했다.
- Recall은 RNN, LSTM, Transformer에서 상승했다.
- GRU와 LSTM의 F1은 이 비교 실험에서는 소폭 하락했다.
- 따라서 `전처리 후 모든 모델 성능이 좋아졌다`라고 말하면 안 된다.

발표에서는 다음처럼 말하는 것이 정확하다.

> 전처리 feature 추가는 모든 모델에서 일괄적으로 F1-score를 상승시키지는 않았지만, 일부 모델에서 F1과 Recall 개선을 확인했다. 특히 낙상 감지에서 중요한 Recall은 4개 모델 중 3개 모델에서 개선되었다. 이는 `accel_norm`, `gyro_norm`, `dt_s`가 낙상 미탐을 줄이는 데 유용한 신호를 제공할 수 있음을 의미한다.

## 8. 최종 LSTM 성능 향상 원인을 어떻게 설명할 것인가

최종 적용 모델은 전처리된 12차원 feature 기반 LSTM이다.

```text
Input shape = [50 timesteps, 12 features]
Threshold   = 0.35
```

최종 테스트 성능:

| Metric | Value |
|---|---:|
| Accuracy | 0.8799 |
| Precision | 0.8498 |
| Recall | 0.8863 |
| F1-score | 0.8677 |

성능의 주요 원인은 다음과 같이 정리할 수 있다.

1. 데이터 병합
   - SisFall은 다양한 낙상 positive pattern을 제공했다.
   - ICCAS는 실제 장비 환경의 normal baseline을 제공했다.

2. 물리 기반 feature 추가
   - `accel_norm`: 방향과 무관한 전체 충격 크기
   - `gyro_norm`: 전체 회전량
   - `dt_s`: 샘플링 간격 보정

3. Robust Scaling
   - 센서 이상치와 데이터셋 간 분포 차이에 덜 민감하게 만들었다.

4. 50-step sliding window
   - 단일 시점 threshold가 아니라 낙상 전후의 시간 문맥을 학습하게 했다.

5. Recall 중심 threshold 선택
   - 낙상 감지에서는 false negative가 위험하므로, Accuracy보다 Recall/F1을 중심으로 threshold를 선택했다.

## 9. 성능 향상 원인에 대한 최종 표현

가장 정확한 표현:

> 최종 모델의 성능은 단순히 모델 구조만으로 향상된 것이 아니라, 데이터 병합, 물리 기반 feature engineering, robust scaling, 50-step sliding window 구성이 함께 작용한 결과이다. 특히 `accel_norm`, `gyro_norm`, `dt_s`를 추가함으로써 원본 축별 센서값만으로는 직접 드러나지 않던 전체 충격 크기, 회전량, 샘플링 간격 정보를 LSTM이 학습할 수 있게 되었다.

주의해야 할 표현:

```text
틀림: 전처리를 하니까 모든 모델 성능이 올라갔다.
정확: 전처리는 모든 모델에서 일괄적인 F1 향상을 보장하지는 않았지만,
      낙상 판단에 필요한 물리적 신호를 명확히 만들어 일부 모델의 Recall/F1 개선에 기여했다.
```

최종 발표 문장:

> 성능 개선의 핵심 원인은 단순 모델 변경이 아니라 데이터 표현 방식의 개선이다. 원본 IMU 9개 feature에 전체 충격 크기, 전체 회전량, 샘플링 간격을 나타내는 3개 feature를 추가하여 총 12차원 입력을 구성했다. 또한 ICCAS와 SisFall의 센서 분포 차이를 고려해 robust scaling을 적용하고, 50-step window로 낙상 전후의 시간적 문맥을 학습하도록 했다. 그 결과 최종 LSTM은 테스트 기준 Recall 0.8863, F1-score 0.8677을 기록했다.
