# IMU 가속도 변화량 데이터 분포 분석

## 목적

기존 fall 데이터와 ICCAS 데이터를 비교해, 가속도 크기와 변화량이 LSTM 낙상 감지 학습에 적합한지 확인했다.

## 분석 대상

- Source CSV: `../data/iccas_sensor_lstm/imu_fall_preprocessed.csv`
- Sequence length 기준: `50` samples
- 비교 기준: `source_dataset`과 `label == fall` 여부
- 핵심 feature: `accel_norm`, `accel_delta = |accel_norm_t - accel_norm_{t-1}|`, `accel_change_rate = accel_delta / dt_s`

## 핵심 분포 요약

| Dataset | Class | accel_norm p50 | accel_norm p95 | accel_norm p99 | accel_delta p50 | accel_delta p95 | accel_delta p99 | change_rate p95 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ICCAS | fall | 1.0041 | 1.2840 | 1.7374 | 0.0162 | 0.2123 | 0.4183 | 5.2102 |
| ICCAS | normal | 1.0029 | 1.4684 | 1.6416 | 0.0899 | 0.3243 | 0.5412 | 7.7447 |
| SisFall | fall | 1.0010 | 1.3671 | 2.2341 | 0.0135 | 0.2887 | 0.8006 | 14.4338 |
| SisFall | normal | 1.0096 | 1.7078 | 2.7465 | 0.0156 | 0.4784 | 1.0925 | 23.9183 |

## LSTM 적합성 판단

- Fall vs normal 효과크기 Cohen's d: `accel_norm=-0.064`, `accel_delta=-0.124`, `accel_change_rate=-0.096`.
- 효과크기가 0에 가깝고 일부 normal 구간의 p95 변화량이 fall보다 크므로, 가속도 크기/변화량 단독 threshold만으로는 낙상을 안정적으로 분리하기 어렵다.
- fall 데이터에도 큰 충격 tail이 존재하지만 normal 구간에도 큰 변화가 있어 단독 구분 신호로 보기는 어렵다. 이 변화는 LSTM이 전후 문맥과 함께 볼 때 낙상 패턴 학습에 도움이 된다.
- 따라서 LSTM 입력은 `accel_delta`만 쓰기보다 `roll/pitch/yaw`, 3축 가속도, 3축 자이로, `accel_norm`, `gyro_norm`, `dt_s`를 함께 사용하는 방식이 적합하다.
- LSTM은 단일 포인트가 아니라 50-step window에서 충격 전후 문맥을 보므로, 연속 샘플 수가 충분한 group만 학습/검증에 사용하는 것이 적절하다.

## Sequence Window 요약

| Dataset | Class | Groups | Ready groups | Window count | Median rows/group |
| --- | --- | ---: | ---: | ---: | ---: |
| ICCAS | fall | 1 | 1 | 676 | 2751.0 |
| ICCAS | normal | 4 | 4 | 5454 | 4502.0 |
| SisFall | fall | 200 | 200 | 35200 | 750.0 |
| SisFall | normal | 200 | 200 | 41320 | 600.0 |

## 결론

- 가속도 변화량만 보면 fall/normal이 깔끔하게 분리되지 않는다. 따라서 단순 threshold 모델보다는 LSTM처럼 시계열 문맥을 보는 모델이 더 적합하다.
- 기존 fall 데이터는 낙상 충격 tail과 낙상 전후 자세/회전 변화를 제공해 positive pattern 학습에 필요하다.
- ICCAS 데이터는 실제 장비/환경의 normal 분포를 제공하므로, false positive를 줄이는 negative baseline 역할을 한다.
- 두 데이터의 센서 스케일과 샘플링 특성이 다를 수 있어 robust scaling과 source별 검증 split이 필요하다.
- 최종 판단: 현재 데이터는 LSTM 학습에 사용할 수 있지만, 가속도 변화량 단독 분포가 아니라 다중 IMU feature와 연속 window를 사용하는 조건에서 적합하다.
