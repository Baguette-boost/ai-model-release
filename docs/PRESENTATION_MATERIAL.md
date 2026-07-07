# ICCAS AI 모델 발표 자료

## 1. 발표 제목

```text
GPS·IMU 센서 기반 LSTM 실시간 배회 감지 및 낙상 감지 시스템
```

발표 핵심 문장:

```text
본 프로젝트는 GPS와 IMU 센서 데이터를 분리 학습하여 사용자의 이동 경로 이상 행동과 낙상 위험을 실시간으로 탐지하고, 탐지 결과를 백엔드와 프론트 지도 화면으로 전달하는 AI 시스템입니다.
```

## 2. 문제 정의

고령자나 보호 대상자의 실시간 위치와 움직임을 기반으로 아래 상황을 자동 감지하는 것이 목표입니다.

```text
1. 평소 경로를 벗어난 배회 행동
2. 강한 충격과 자세 변화가 동반되는 낙상 행동
3. 감지 결과를 서버로 전달하여 지도 화면에서 확인
```

발표 멘트:

```text
단순히 GPS 위치만 보는 것이 아니라, GPS는 동선 이상 여부를 판단하고 IMU는 낙상 움직임을 판단하도록 역할을 분리했습니다. 이렇게 해야 GPS 흔들림과 낙상 충격을 하나의 모델에 섞지 않고 더 안정적으로 감지할 수 있습니다.
```

## 3. 전체 시스템 구조

```text
센서 데이터 수집
  ↓
GPS 서버: 배회 감지
  ↓
IMU 서버: 낙상 감지
  ↓
결과 병합 서버
  ↓
프론트 지도 화면 실시간 표시
```

적용 모델:

```text
GPS 배회 감지:
models/iccas_final_lstm_gps_wandering.pt

IMU 낙상 감지:
models/iccas_final_sisfall_lstm_imu_fall.pt
```

발표 멘트:

```text
최종 구조는 GPS와 IMU를 병렬 서버로 분리하는 방식입니다. GPS 모델은 동선 이탈과 배회를 판단하고, IMU 모델은 낙상 여부를 판단합니다. 마지막 서버에서는 두 결과를 합쳐 최종 위험 수준을 결정합니다.
```

## 4. 사용 데이터

최종 학습 데이터:

```text
ICCAS_final_data.xlsx
```

Sheet별 라벨 매핑:

| Sheet | Label | Rows |
| --- | --- | ---: |
| 추가된 학교 데이터 | walk | 5300 |
| 집 앞 walk | walk | 1603 |
| 학교 walk | walk | 1100 |
| 새로운 경로에서 배회 | wandering | 3801 |
| 새로운 경로 1.2km, 새로운 경로 600m | wandering | 8552 |
| 떨어졌다가 앞으로 걸어가고 떨어졌다 앞으로 걸어감 | fall | 1251 |
| 제자리에서 떨어짐1,2 | fall | 1500 |
| sit | sit | 651 |
| idle | idle | 1001 |

전처리 후 라벨 분포:

| Label | Rows |
| --- | ---: |
| walk | 8003 |
| wandering | 12353 |
| fall | 2751 |
| sit | 651 |
| idle | 1001 |

발표 멘트:

```text
최종 데이터에는 새 GPS 데이터인 추가된 학교 데이터가 포함되었습니다. 이 데이터는 정상 보행 동선인 walk로 라벨링했고, 새로운 경로 관련 sheet는 배회 상황인 wandering으로 재라벨링했습니다.
```

## 5. 전처리 방식

입력 컬럼:

```text
server_time, device, label, t_ms,
roll, pitch, yaw,
ax, ay, az,
wx, wy, wz,
lat, lng
```

GPS feature:

```text
x_m, y_m, dx_m, dy_m, speed_mps, dt_s, gps_valid
```

IMU/Gyro feature:

```text
roll, pitch, yaw,
ax, ay, az,
wx, wy, wz,
accel_norm, gyro_norm, dt_s
```

생성 파일:

```text
data/iccas_sensor_lstm/iccas_final_labeled.csv
data/iccas_sensor_lstm/iccas_final_lstm_features.csv
data/iccas_sensor_lstm/iccas_final_preprocess_summary.json
```

발표 멘트:

```text
위도와 경도는 meter 단위 좌표로 변환했고, 연속 위치 변화량과 속도를 계산했습니다. IMU는 가속도 크기와 자이로 크기를 추가해 낙상 충격을 더 잘 표현하도록 했습니다.
```

## 6. 모델 설계

5-class 실험 모델:

```text
walk, wandering, fall, sit, idle
```

서버 적용 최종 모델:

```text
GPS Wandering Binary LSTM
IMU Fall Binary LSTM
```

발표 멘트:

```text
처음에는 5개 클래스를 한 번에 분류하는 모델을 만들었지만, 실제 서버 적용에서는 GPS와 IMU의 역할이 다르기 때문에 binary 모델을 최종 구조로 선택했습니다. GPS는 wandering 여부만 판단하고, IMU는 fall 여부만 판단합니다.
```

## 6-1. RNN, GRU, LSTM, Transformer 비교

시계열 모델 비교 요약:

| Model | 장점 | 한계 | 본 프로젝트에서의 역할 |
| --- | --- | --- | --- |
| RNN | 구조가 단순하고 빠름 | 긴 시퀀스 기억이 약함 | baseline 비교용 |
| GRU | LSTM보다 가볍고 빠름 | 복잡한 장기 패턴은 LSTM보다 약할 수 있음 | 경량화 후보 |
| LSTM | 장단기 시계열 패턴을 안정적으로 학습 | GRU보다 연산량이 조금 큼 | 현재 최종 선택 |
| Transformer | 긴 시퀀스와 대규모 데이터에 강함 | 데이터와 연산량이 많이 필요함 | 데이터 증가 후 확장 후보 |

비교할 때 중요한 지표:

```text
Accuracy, Precision, Recall, F1-score,
추론 latency, 모델 크기, 실시간 서버 적용 가능성
```

발표 멘트:

```text
Transformer가 최신 구조이기는 하지만, 현재 프로젝트는 실시간 센서 데이터와 제한된 학습 데이터가 핵심 조건입니다. 그래서 RNN은 baseline, GRU는 경량화 후보, Transformer는 향후 확장 후보로 두고, 현재 서버 적용에는 성능과 안정성의 균형이 좋은 LSTM을 선택했습니다.
```

자세한 비교 자료:

```text
docs/MODEL_COMPARISON_ANALYSIS.md
```

## 7. 최종 성능 지표

GPS Wandering, 전체 sheet 기준:

| Metric | Value |
| --- | ---: |
| Accuracy | 0.9115 |
| Precision | 0.8586 |
| Recall | 0.9854 |
| F1-score | 0.9177 |

GPS Wandering, 서버 task 기준:

| Metric | Value |
| --- | ---: |
| Accuracy | 0.9544 |
| Precision | 0.9368 |
| Recall | 0.9854 |
| F1-score | 0.9605 |

IMU Fall, ICCAS + SisFall 기준:

| Metric | Value |
| --- | ---: |
| Accuracy | 0.8814 |
| Precision | 0.8879 |
| Recall | 0.8392 |
| F1-score | 0.8629 |

발표 멘트:

```text
GPS 모델은 전체 sheet 기준 F1이 0.9177이고, 실제 서버에서 GPS가 담당해야 하는 task 기준으로는 F1이 0.9605입니다. 낙상 모델은 ICCAS 낙상 데이터와 SisFall 데이터를 병합해 F1 0.8629를 확보했습니다.
```

## 8. 성능 시각화 자료

발표에 바로 사용할 이미지:

```text
assets/final_model_performance_dashboard.png
```

이미지에 포함된 내용:

```text
GPS 서버 F1: 96.0%
GPS 이동경로 F1: 99.0%
IMU 낙상 F1: 86.3%
학습 데이터: 24,759 rows
모델별 Accuracy / Precision / Recall / F1 비교
라벨 분포
Confusion Matrix
서버 적용 구조
```

발표 멘트:

```text
이 시각화 자료는 최종 모델 성능을 한 장으로 요약한 것입니다. GPS 모델은 배회 감지에 특화되어 있고, IMU 모델은 낙상 감지에 특화되어 있다는 점을 한눈에 보여줍니다.
```

## 9. 백엔드 및 프론트 연동

AI 추론 결과는 백엔드로 전송됩니다.

```text
POST /api/sensor-result
GET  /api/sensor-result/latest
GET  /api/sensor-result/history?limit=5000
```

프론트 화면에서는 아래 정보를 사용합니다.

```text
lat, lng
fall_detected
wandering_detected
risk_level
detection_type
```

발표 멘트:

```text
AI 모델은 단독으로 끝나는 것이 아니라, 추론 결과를 백엔드로 전달합니다. 프론트는 백엔드의 최신 결과와 히스토리를 조회해 지도 위에 실시간 상태를 표시합니다.
```

## 10. 사용자의 동선을 추가 학습할 수 있는가?

가능합니다. 현재 구조는 새로운 사용자의 GPS 동선을 추가로 학습할 수 있습니다.

가능한 방식:

```text
1. 새로운 사용자의 정상 이동 경로를 walk 데이터로 추가
2. 새로운 위험 경로나 이탈 경로를 wandering 데이터로 추가
3. sheet 또는 CSV 단위로 기존 전처리 파이프라인에 넣기
4. GPS Wandering 모델 재학습
5. 새 모델을 서버에 배포
```

필요한 데이터 컬럼:

```text
server_time, device, lat, lng
```

있으면 좋은 컬럼:

```text
label, t_ms, roll, pitch, yaw, ax, ay, az, wx, wy, wz
```

권장 데이터량:

| 목적 | 최소 | 권장 |
| --- | ---: | ---: |
| 특정 사용자 정상 동선 등록 | 경로당 3회 이상 | 경로당 5~10회 이상 |
| 배회/이탈 경로 학습 | 유형당 2~3회 이상 | 유형당 5회 이상 |
| 실시간 서비스 안정화 | 1~2일 | 1~2주 |

발표 멘트:

```text
사용자의 동선은 추가 학습이 가능합니다. 다만 실시간으로 들어오는 데이터를 즉시 모델에 반영하는 online learning 방식보다는, 일정 기간 정상 동선을 수집한 뒤 batch retraining 방식으로 모델을 갱신하는 것이 안전합니다.
```

## 10-1. LSTM 준비 전 초기 궤적 추론 데모

LSTM은 `sequence_length`만큼 GPS 포인트가 쌓인 뒤부터 추론할 수 있습니다. 현재 GPS Wandering 모델은 16개 포인트가 쌓이면 LSTM 추론이 시작됩니다.

그 전에는 Martino-Saltzman 기반 규칙 알고리즘을 먼저 사용합니다.

```text
GPS 포인트 3개 이상:
DF, heading angle, node revisit, self-intersection 계산

GPS 포인트 5~8개 이상:
Direct / Pacing / Lapping / Random 초기 분류

GPS 포인트 16개 이상:
LSTM Wandering 추론과 알고리즘 결과 병합
```

Pacing 샘플 데모 결과:

| Index | Algorithm | LSTM Ready | Final Detection | Risk |
| ---: | --- | --- | --- | --- |
| 3 | Direct | false | pending_normal | low |
| 8 | Pacing | false | early_pattern_warning | medium |
| 15 | Pacing | false | early_pattern_warning | medium |
| 16 | Pacing | true | wandering | high |
| 17 | Pacing | true | wandering | high |

실행 명령어:

```bash
python scripts/trajectory_pattern_demo.py \
  --sample pacing \
  --model models/iccas_final_lstm_gps_wandering.pt \
  --output ../data/iccas_sensor_lstm/trajectory_pattern_demo_pacing.json
```

발표 멘트:

```text
LSTM이 시퀀스를 기다리는 동안 시스템이 아무 판단도 하지 않는 것이 아니라, 규칙 기반 공간 궤적 분석으로 먼저 early warning을 발생시킵니다. 이후 LSTM이 준비되면 두 결과를 병합해 최종 위험도를 높입니다.
```

## 11. 동선 추가 학습의 권장 구조

권장 방식:

```text
사용자별 정상 경로 데이터 수집
  ↓
전처리 및 route feature 생성
  ↓
기존 ICCAS_final_data와 병합
  ↓
GPS Wandering 모델 재학습
  ↓
성능 검증
  ↓
모델 배포
```

주의할 점:

```text
1. GPS 오차가 큰 실내/건물 주변 데이터는 별도 표시
2. 등하교, 병원, 산책 등 경로 목적을 구분하면 성능이 좋아짐
3. 정상 동선만 추가하면 모델이 너무 관대해질 수 있으므로 이탈 예시도 함께 필요
4. 사용자별 모델과 공통 모델을 분리할 수 있음
```

추천 설계:

```text
공통 모델: 전체 사용자 기본 배회 감지
사용자별 route profile: 개인의 자주 가는 경로 보정
최종 판단: 공통 LSTM 점수 + 사용자별 경로 이탈 거리
```

발표 멘트:

```text
가장 현실적인 구조는 공통 LSTM 모델과 사용자별 동선 프로파일을 함께 사용하는 것입니다. 공통 모델은 기본적인 배회 패턴을 잡고, 사용자별 프로파일은 개인의 자주 가는 경로를 반영해 오탐을 줄입니다.
```

## 12. 향후 개선 방향

```text
1. 사용자별 정상 동선 자동 누적
2. 보호자가 정상 경로를 승인하는 라벨링 기능
3. GPS 음영 구간 보정
4. 낙상 직후 일정 시간 GPS 배회 알림 우선순위 낮춤
5. 더 많은 실제 착용자 데이터로 재학습
```

발표 마무리 멘트:

```text
본 프로젝트는 GPS와 IMU를 분리해 각각의 센서가 잘 판단할 수 있는 문제를 맡기는 구조입니다. 현재 모델은 실시간 서버 전송과 지도 시각화까지 연결되어 있으며, 이후 사용자별 동선을 추가 학습하면 개인화된 배회 감지 시스템으로 확장할 수 있습니다.
```

## 발표용 한 줄 요약

```text
GPS는 사용자의 동선 이탈을 판단하고, IMU는 낙상 움직임을 판단하며, 두 결과를 서버에서 병합해 실시간 지도 기반 보호 시스템을 구현했습니다.
```
