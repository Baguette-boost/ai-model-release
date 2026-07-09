# IMU 낙상 데이터가 LSTM에 적합한 이유

## 핵심 결론

IMU 낙상 데이터는 LSTM 학습에 적합하다. 단, `가속도 변화량이 크면 낙상`처럼 단일 threshold로 분리되는 데이터이기 때문은 아니다. 낙상은 짧은 시간 안에서 `충격`, `회전`, `자세 변화`, `충격 이후 안정/비활동`이 순서대로 나타나는 시계열 이벤트이기 때문에 LSTM처럼 연속 window를 보는 모델에 더 적합하다.

## 논문/리서치 근거

### 1. 낙상은 시간적 순서가 중요한 이벤트

웨어러블 낙상 감지 연구들은 accelerometer와 gyroscope의 시간 신호를 사용한다. CareFall 연구는 스마트워치의 accelerometer, gyroscope time signal을 사용하며, threshold 방식보다 machine learning 방식이 accuracy, sensitivity, specificity에서 더 좋은 결과를 보였다고 정리한다.

Source: https://arxiv.org/abs/2307.05275

### 2. 단일 가속도만 쓰면 오탐이 커질 수 있음

최근 multi-modal fall detection 연구는 single-modality acceleration data만 사용하는 방식은 false alarm 문제가 커질 수 있다고 설명한다. 그래서 accelerometer뿐 아니라 gyroscope, temporal weighting, multi-scale feature를 함께 사용하는 방향이 제안된다.

Source: https://arxiv.org/abs/2603.22313

### 3. LSTM은 낙상 검출에서 recall 중심 목적에 적합

낙상 감지는 낙상을 놓치지 않는 것이 중요하므로 recall을 우선해야 한다. LSTM 기반 낙상 연구도 false negative를 줄이기 위해 recall을 우선하는 설계를 강조한다.

Source: https://arxiv.org/abs/2309.07154

### 4. IMU fall 데이터는 짧은 fixed window에서 특징이 나타남

웨어러블 낙상 데이터는 짧은 window 안에서 impact signature가 나타난다. 최신 연구에서도 fall-discriminative feature는 짧은 고정 길이 window 내 국소적인 충격 패턴으로 설명된다.

Source: https://arxiv.org/abs/2605.20275

## LSTM에 적합한 데이터 분포 조건

LSTM에 적합한 IMU 낙상 데이터는 아래 특성을 가져야 한다.

| 조건 | 의미 | 우리 데이터 상태 |
|---|---|---|
| 연속 시계열 | 샘플이 시간 순서대로 충분히 이어져야 함 | 적합. 50-step window 생성 가능 |
| 다축 IMU | 3축 가속도와 3축 자이로가 있어야 함 | 적합. ax/ay/az, wx/wy/wz 사용 |
| 충격 tail | fall 구간에 순간적으로 큰 accel/gyro 변화가 있어야 함 | 일부 존재 |
| normal overlap | normal 활동에도 큰 변화가 있어야 현실적임 | 존재. 단일 threshold는 부적합 |
| window label | window 안의 fall 여부를 이진 라벨로 만들 수 있어야 함 | 가능 |
| source 다양성 | fall positive와 실제 장비 normal baseline이 같이 있어야 함 | SisFall + ICCAS 병합 |

## 현재 데이터 분석 결과와 연결

현재 분석 파일:

```text
docs/IMU_ACCELERATION_DISTRIBUTION_ANALYSIS.md
assets/imu_acceleration_distribution_analysis.svg
```

가속도 변화량 p95를 보면 fall과 normal이 완전히 분리되지 않는다.

```text
ICCAS fall   accel_delta p95 = 0.2123
ICCAS normal accel_delta p95 = 0.3243
SisFall fall accel_delta p95 = 0.2887
SisFall normal accel_delta p95 = 0.4784
```

이 결과는 오히려 LSTM 적용 근거가 된다. 단일 가속도 threshold로는 normal 활동의 빠른 움직임과 낙상 충격을 구분하기 어렵기 때문에, LSTM이 50-step window에서 전후 문맥을 함께 학습해야 한다.

## 현재 최종 모델 입력

최종 IMU 낙상 LSTM은 12차원 feature를 사용한다.

```text
roll, pitch, yaw,
ax, ay, az,
wx, wy, wz,
accel_norm, gyro_norm, dt_s
```

전처리 feature:

```text
accel_norm = sqrt(ax^2 + ay^2 + az^2)
gyro_norm  = sqrt(wx^2 + wy^2 + wz^2)
dt_s       = (t_ms[i] - t_ms[i-1]) / 1000
```

입력 window:

```text
X = [50 timesteps, 12 features]
y = 1 if any fall sample exists in the window else 0
```

## 발표용 어필 문장

> IMU 낙상 데이터는 단일 가속도 임계값으로 fall/normal이 완전히 분리되는 분포는 아니었다. 그러나 이는 실제 환경과 유사한 데이터 특성이다. 정상 활동에서도 큰 가속도 변화가 발생할 수 있기 때문에, 본 연구는 3축 가속도, 3축 자이로, 자세 변화, 가속도 크기, 자이로 크기, 시간 간격을 50-step 시계열 window로 구성하여 LSTM이 낙상 전후의 동적 패턴을 학습하도록 설계했다.

> 따라서 현재 데이터는 LSTM 학습에 적합하다. 특히 SisFall은 낙상 positive pattern을 제공하고, ICCAS 직접 취득 데이터는 실제 장비 환경의 normal baseline을 제공해 false positive를 줄이는 데 기여한다.

## 주의해서 말해야 할 부분

- `가속도 변화량만으로 낙상이 잘 분리된다`라고 말하면 안 된다.
- `가속도 변화량 단독으로는 부족해서 LSTM 시계열 학습이 필요하다`라고 말하는 것이 정확하다.
- Accuracy만 강조하기보다 Recall과 F1-score를 함께 제시해야 한다.
