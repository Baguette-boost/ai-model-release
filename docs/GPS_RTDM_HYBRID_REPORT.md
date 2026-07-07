# GPS RTDM Hybrid 실시간 탐지 보고서

## 적용한 RTDM 아이디어

NicklasXYZ/rtdm의 핵심 아이디어인 `trajectory -> token sequence -> normal support -> sequence anomaly score`를 우리 GPS 데이터에 맞춰 lightweight 구현했습니다.

주의: 원본 NicklasXYZ/rtdm 패키지를 그대로 이식한 것이 아니라, 원본의 geohash/token sequence support 개념을 참고해 `x_m`, `y_m` 기반 meter-grid token으로 재구현한 코드입니다.

## 처리 흐름

```text
lat/lng
  -> meter 좌표 변환
  -> Kalman filter
  -> grid token sequence
  -> normal route support score
  -> RNN wandering score와 결합
```

## 성능

| Method | Accuracy | Precision | Recall | F1-score | Threshold |
| --- | ---: | ---: | ---: | ---: | ---: |
| Kalman RNN | 0.9494 | 0.9337 | 0.9676 | 0.9503 | 0.45 |
| RTDM only | 0.7752 | 0.6925 | 0.9903 | 0.8150 | 0.51 |
| RTDM + RNN Hybrid | 0.9662 | 0.9380 | 0.9984 | 0.9673 | 0.41 |

## 처리 속도

| Step | Time |
| --- | ---: |
| GPS RNN batch inference | 0.0065 ms / sequence |
| GPS RNN realtime loop, forward only | avg 0.0929 ms / sequence |
| GPS RNN realtime loop, tensor + forward | avg 0.0988 ms / sequence |
| RTDM support score, batch | 0.0069 ms / sequence |
| RTDM support score, realtime loop | avg 0.0078 ms / sequence |
| Hybrid score combine | < 0.001 ms / sequence |
| Estimated realtime total | 약 0.107 ms / sequence |

테스트 기준 RTDM은 이미 전처리된 3,697개 GPS window를 25.682 ms에 처리했습니다. 실시간처럼 window 1개씩 함수 호출하는 방식으로는 평균 0.0078 ms, median 0.0076 ms, p95 0.0085 ms가 측정되었습니다.

RNN은 동일한 3,697개 GPS window를 batch size 256으로 한 번에 처리하면 24.156 ms, 즉 0.0065 ms/sequence로 측정되었습니다. 다만 실제 서버처럼 window 1개씩 호출하면 tensor 변환 포함 평균 0.0988 ms, median 0.0941 ms, p95 0.1121 ms입니다.

주의: 위 속도는 Excel 파일 로딩, 전체 데이터셋 Kalman feature 생성, support 구축 시간을 제외한 "실시간 ready window 1개에 대한 알고리즘 추론" 시간입니다. 초기 준비 시간은 Excel 로딩 798.716 ms, 전체 Kalman feature 생성 246.164 ms, sequence 생성 10.401 ms, normal support 구축 49.390 ms로 별도 측정되었습니다.

## 검증 범위와 한계

- 현재 성능 수치는 각 시나리오 내부를 train/validation/test 시간 구간으로 나눈 평가입니다.
- 완전히 새로운 사용자, 장소, 장비를 분리한 holdout 검증 결과는 아닙니다.
- 이 스크립트는 Kalman RNN과 RTDM support를 학습/평가하지만, 배포용 hybrid checkpoint 파일은 아직 저장하지 않습니다.

## 결론

- Hybrid F1 변화량 vs Kalman RNN: `+0.0169`
- Hybrid Precision 변화량 vs Kalman RNN: `+0.0043`
- Hybrid Recall 변화량 vs Kalman RNN: `+0.0308`

실시간 서버에서는 RTDM score를 RNN과 병렬로 계산하고, validation 기준 최적 weight를 적용하는 방식을 권장합니다.
