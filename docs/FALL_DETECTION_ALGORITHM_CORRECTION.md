# IMU 낙상 감지 F1-score 보정 방안

## 1. 현재 문제

현재 IMU 낙상 감지는 LSTM 기반으로 동작하지만, 비교 실험에서 F1-score가 GPS 배회 감지보다 낮게 나왔습니다.

주요 이유:

```text
1. 낙상 데이터 수가 정상/보행 데이터보다 적음
2. 앉기, 급정지, 손 흔들림, 기기 흔들림이 낙상 충격과 비슷하게 보일 수 있음
3. 낙상은 순간 이벤트라 window 위치에 따라 학습 난이도가 커짐
4. 낙상 직후의 정지 상태까지 보지 않으면 오탐이 증가함
```

따라서 LSTM 단독 판단보다, IMU 낙상 물리 패턴 알고리즘을 함께 사용하는 하이브리드 구조가 적합합니다.

## 2. 낙상 패턴의 대표 알고리즘

IMU 기반 낙상은 보통 아래 4단계 패턴을 봅니다.

```text
1. Free-fall 또는 급격한 자세 변화
2. 큰 충격 impact
3. roll/pitch/yaw 또는 gyro 변화
4. 낙상 후 일정 시간 낮은 움직임, 즉 inactivity
```

사용할 feature:

```text
accel_norm = sqrt(ax^2 + ay^2 + az^2)
gyro_norm  = sqrt(wx^2 + wy^2 + wz^2)
tilt_delta = 이전 자세와 현재 자세의 roll/pitch 변화량
post_motion = 충격 이후 1~3초간 accel_norm/gyro_norm 평균
```

## 3. 보정 알고리즘 제안

### 3-1. Impact Gate

낙상 후보는 먼저 강한 충격이 있어야 합니다.

```text
impact_score = max(accel_norm in window)

if impact_score >= impact_threshold:
    impact_gate = true
else:
    impact_gate = false
```

권장 시작값:

```text
impact_threshold = train 데이터의 accel_norm 상위 95~98 percentile
```

역할:

```text
LSTM이 fall이라고 예측해도 충격이 전혀 없으면 fall 확률을 낮춤
강한 충격이 있으면 fall 후보로 올림
```

### 3-2. Rotation Gate

낙상은 단순 충격뿐 아니라 자세 변화나 회전량이 동반되는 경우가 많습니다.

```text
rotation_score = max(gyro_norm in window)
tilt_delta = max(abs(roll - roll_start), abs(pitch - pitch_start))

if rotation_score >= gyro_threshold or tilt_delta >= tilt_threshold:
    rotation_gate = true
else:
    rotation_gate = false
```

권장 시작값:

```text
gyro_threshold = train 데이터의 gyro_norm 상위 90~95 percentile
tilt_threshold = 35~60 degrees
```

역할:

```text
기기 흔들림과 실제 몸의 자세 변화가 있는 낙상을 구분
```

### 3-3. Post-fall Inactivity Gate

낙상 후에는 보통 1~3초 정도 움직임이 줄어드는 구간이 나타납니다. 사용자가 바로 일어나거나 걸으면 낙상 가능성을 낮춰야 합니다.

```text
after_impact_window = impact 이후 1~3초
post_accel_std = std(accel_norm after impact)
post_gyro_mean = mean(gyro_norm after impact)

if post_accel_std <= still_accel_threshold and post_gyro_mean <= still_gyro_threshold:
    inactivity_gate = true
else:
    inactivity_gate = false
```

역할:

```text
앉기, 점프, 손 흔들림, 잠깐 충격 후 정상 이동을 낙상으로 오탐하는 문제를 줄임
```

## 4. LSTM 점수 보정 공식

LSTM의 fall probability에 물리 알고리즘 점수를 결합합니다.

```text
lstm_score = LSTM fall probability

algorithm_score =
  0.40 * impact_gate
  + 0.25 * rotation_gate
  + 0.25 * inactivity_gate
  + 0.10 * posture_change_score

final_score =
  0.65 * lstm_score
  + 0.35 * algorithm_score
```

최종 판단:

```text
if final_score >= threshold:
    fall_detected = true
else:
    fall_detected = false
```

처음 권장 threshold:

```text
threshold = validation set에서 F1-score가 가장 높은 값으로 선택
```

## 5. 오탐 감소 규칙

아래 조건은 fall 가능성을 낮춥니다.

```text
1. 충격 이후 2초 이내 다시 정상 보행 speed/gyro 패턴이 나타남
2. impact는 있지만 roll/pitch 변화가 거의 없음
3. gyro_norm만 높고 accel_norm 충격이 낮음
4. sit 라벨과 비슷하게 천천히 자세가 변함
```

예시:

```text
if impact_gate and not rotation_gate and not inactivity_gate:
    detection_type = "impact_only"
    fall_detected = false 또는 risk_level = medium

if lstm_score >= 0.8 and impact_gate and inactivity_gate:
    detection_type = "fall"
    risk_level = high
```

## 6. 미탐 감소 규칙

낙상을 놓치지 않기 위해 아래 조건은 fall 가능성을 높입니다.

```text
1. impact가 매우 크고 이후 정지 상태가 있음
2. LSTM score가 중간 이상이고 gyro_norm이 크게 증가함
3. roll/pitch 변화가 크고 이후 움직임이 거의 없음
```

예시:

```text
if impact_score >= very_high_impact_threshold and inactivity_gate:
    fall_detected = true

if lstm_score >= 0.45 and impact_gate and rotation_gate and inactivity_gate:
    fall_detected = true
```

## 7. 기대 효과

| 문제 | 보정 방법 | 기대 효과 |
| --- | --- | --- |
| 앉기/급정지 오탐 | post-fall inactivity와 자세 변화 확인 | Precision 개선 |
| 약한 낙상 미탐 | LSTM score + impact gate 결합 | Recall 개선 |
| 기기 흔들림 오탐 | accel_norm과 gyro_norm 동시 확인 | FP 감소 |
| threshold 고정 문제 | validation F1 기준 threshold 선택 | F1-score 개선 |

## 8. 서버 적용 구조

```text
IMU JSON 수신
  ↓
roll, pitch, yaw, ax, ay, az, wx, wy, wz 누적
  ↓
accel_norm, gyro_norm, tilt_delta 계산
  ↓
낙상 물리 알고리즘 점수 계산
  ↓
LSTM fall probability 계산
  ↓
final_score = LSTM + 알고리즘 보정
  ↓
fall_detected, risk_level, detection_type 백엔드 전송
```

서버로 보낼 추천 필드:

```json
{
  "fall_detected": true,
  "fall_score": 0.87,
  "lstm_score": 0.79,
  "fall_algorithm_score": 0.95,
  "impact_score": 18.4,
  "gyro_score": 540.2,
  "post_fall_inactivity": true,
  "detection_type": "fall",
  "risk_level": "high"
}
```

## 9. 발표용 설명

```text
IMU 낙상 감지는 단순히 LSTM 확률만 사용하는 것이 아니라, 낙상 물리 패턴을 함께 반영해 보정할 수 있습니다. 충격 크기, 회전량, 자세 변화, 낙상 후 정지 상태를 gate로 계산하고, 이를 LSTM 점수와 결합하면 오탐과 미탐을 동시에 줄일 수 있습니다.
```

## 10. 최종 추천

현재 프로젝트에는 아래 방식이 가장 적합합니다.

```text
1. LSTM은 낙상 전후의 시간 흐름을 학습
2. 알고리즘은 충격/회전/정지 같은 물리 조건을 확인
3. validation set에서 final_score threshold를 다시 튜닝
4. Precision, Recall, F1-score를 다시 측정
```

즉, `IMU Fall LSTM + Impact/Rotation/Inactivity Algorithm` 구조로 바꾸는 것이 F1-score 개선에 가장 현실적입니다.
