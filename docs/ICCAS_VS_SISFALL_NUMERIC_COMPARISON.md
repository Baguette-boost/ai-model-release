# ICCAS 직접 취득 데이터 vs SisFall 오픈 데이터 수치 비교

## 목적

직접 취득한 ICCAS IMU 데이터와 오픈 데이터인 SisFall IMU 데이터가 LSTM 낙상 감지 학습에 함께 사용될 수 있는지 수치적으로 확인했다.

비교 기준:

```text
ICCAS   = 직접 취득 데이터
SisFall = 오픈 낙상 데이터
normal  = fall이 아닌 활동
fall    = 낙상 활동
```

## 핵심 결론

두 데이터는 **가속도 크기의 중심값은 매우 유사**하다. 따라서 기본적인 IMU 가속도 스케일은 크게 어긋나지 않는다.

다만 **샘플링 간격, 자이로 크기, 변화량 분포는 차이가 있다.** 그래서 두 데이터를 단순히 같은 분포라고 말하면 안 된다. 대신 robust scaling과 50-step LSTM window를 적용해, 절대값 차이보다 `시간적 변화 패턴`을 학습하도록 만든 것이 타당하다.

## 1. 가속도 크기 중심값은 차이가 작음

`accel_norm = sqrt(ax^2 + ay^2 + az^2)` 기준으로 보면 중앙값 차이가 매우 작다.

| Class | ICCAS accel_norm p50 | SisFall accel_norm p50 | Difference |
|---|---:|---:|---:|
| normal | 1.0029 | 1.0096 | -0.0067 |
| fall | 1.0041 | 1.0010 | +0.0031 |

해석:

- normal 중앙값 차이: 약 `0.0067g`
- fall 중앙값 차이: 약 `0.0031g`
- 두 데이터 모두 중심값이 약 `1g` 근처에 있어, 중력 기준 가속도 스케일은 유사하다.

발표용 표현:

> ICCAS 직접 취득 데이터와 SisFall 오픈 데이터의 가속도 크기 중앙값은 모두 1g 근처로 나타났다. normal 기준 차이는 0.0067g, fall 기준 차이는 0.0031g 수준으로, 기본 가속도 스케일은 크게 어긋나지 않는다.

## 2. 고분위 구간은 차이가 있음

중앙값은 비슷하지만 p95에서는 차이가 있다.

| Class | ICCAS accel_norm p95 | SisFall accel_norm p95 | ICCAS/SisFall |
|---|---:|---:|---:|
| normal | 1.4684 | 1.7078 | 0.8598 |
| fall | 1.2840 | 1.3671 | 0.9392 |

해석:

- SisFall이 p95 기준으로 더 큰 가속도 tail을 가진다.
- 이는 오픈 데이터의 낙상/활동 실험 강도, 센서 부착 위치, 샘플링 특성이 직접 취득 데이터와 다르기 때문일 수 있다.
- 따라서 단순 threshold보다 scaling + sequence learning이 필요하다.

## 3. 가속도 변화량은 완전히 같은 분포가 아님

`accel_delta = |accel_norm_t - accel_norm_{t-1}|` 기준:

| Class | ICCAS accel_delta p50 | SisFall accel_delta p50 | ICCAS accel_delta p95 | SisFall accel_delta p95 |
|---|---:|---:|---:|---:|
| normal | 0.0899 | 0.0156 | 0.3243 | 0.4784 |
| fall | 0.0162 | 0.0135 | 0.2123 | 0.2887 |

해석:

- fall 중앙값은 ICCAS `0.0162`, SisFall `0.0135`로 비슷하다.
- normal 변화량은 ICCAS가 더 크게 나타난다. 직접 취득 데이터에서 걷기/배회/장비 흔들림이 반영된 것으로 볼 수 있다.
- p95에서는 SisFall이 더 큰 tail을 가진다.

이 결과는 오히려 실제 환경성을 보여준다.

> 정상 활동에서도 큰 가속도 변화가 발생하므로, 가속도 변화량 하나만으로 낙상을 판단하면 오탐이 생길 수 있다. 따라서 LSTM이 50-step window에서 충격 전후의 문맥을 학습하는 방식이 필요하다.

## 4. 자이로 분포는 차이가 큼

`gyro_norm = sqrt(wx^2 + wy^2 + wz^2)` 기준:

| Class | ICCAS gyro_norm p50 | SisFall gyro_norm p50 | ICCAS gyro_norm p95 | SisFall gyro_norm p95 |
|---|---:|---:|---:|---:|
| normal | 48.2768 | 14.7802 | 112.9700 | 130.3051 |
| fall | 25.4464 | 6.6681 | 177.9467 | 116.6851 |

해석:

- 자이로 중앙값은 ICCAS가 SisFall보다 크다.
- fall p95에서는 ICCAS가 `177.9467`, SisFall이 `116.6851`로 ICCAS가 더 크다.
- 장비 부착 위치, 센서 방향, 실제 이동 환경 차이가 반영된 것으로 볼 수 있다.

따라서 자이로는 raw 값 그대로 threshold로 쓰기보다 robust scaling 후 LSTM feature로 사용하는 것이 타당하다.

## 5. 샘플링 간격 차이

| Class | ICCAS dt_s p50 | SisFall dt_s p50 |
|---|---:|---:|
| normal | 0.0400s | 0.0200s |
| fall | 0.0400s | 0.0200s |

해석:

- ICCAS는 대략 `25Hz`
- SisFall은 대략 `50Hz`
- 샘플링 간격이 2배 차이 나므로, `dt_s`를 feature로 넣은 것은 적절하다.

발표용 표현:

> 두 데이터의 샘플링 간격이 달라 같은 50-step window라도 실제 시간 길이가 다를 수 있다. 이를 보정하기 위해 `dt_s`를 LSTM 입력 feature에 포함했다.

## 최종 판단

| 항목 | 차이 수준 | 판단 |
|---|---|---|
| accel_norm 중심값 | 작음 | 병합 학습에 유리 |
| accel_norm p95 | 중간 차이 | scaling 필요 |
| accel_delta | 차이 있음 | threshold 단독 부적합 |
| gyro_norm | 차이 큼 | robust scaling 필요 |
| dt_s | 차이 있음 | feature로 포함 필요 |
| sequence window | 충분함 | LSTM 학습 적합 |

## 발표용 최종 문장

> 오픈 데이터인 SisFall과 직접 취득한 ICCAS 데이터의 가속도 크기 중앙값은 모두 1g 근처로 나타나 기본 IMU 스케일은 유사했다. normal 기준 accel_norm 중앙값 차이는 0.0067g, fall 기준 차이는 0.0031g로 작았다.

> 반면 자이로 크기, 가속도 변화량의 tail, 샘플링 간격은 차이가 있었다. 따라서 두 데이터를 단순히 같은 분포로 가정하지 않고, robust scaling과 `dt_s` feature를 적용한 뒤 50-step LSTM window로 학습했다.

> 이 점에서 ICCAS 데이터는 실제 장비 환경의 normal baseline을 제공하고, SisFall은 다양한 낙상 positive pattern을 제공한다. 두 데이터를 병합하면 낙상 패턴과 실제 환경 오탐 요인을 함께 학습할 수 있다.
