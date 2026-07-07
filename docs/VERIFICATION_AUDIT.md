# 검증 감사 보고서

## 결론

지금까지 만든 GPS 관련 결과는 실행 로그와 생성 파일 기준으로 대부분 재현 가능한 값입니다. 다만 아래 내용은 정확히 구분해서 말해야 합니다.

- `RTDM`은 NicklasXYZ/rtdm 원본 패키지를 그대로 이식한 것이 아닙니다.
- 현재 구현은 원본의 token sequence support 개념을 참고한 `meter-grid token` 기반 lightweight 재구현입니다.
- `RTDM + RNN Hybrid F1 0.9673`은 학습/평가 스크립트가 실행 중에 만든 in-memory 모델 기준입니다.
- 현재 Git에 저장된 배포 모델 `models/iccas_final_rnn_gps_wandering.pt`는 기존 GPS RNN 모델이며, hybrid checkpoint 파일은 아직 저장하지 않습니다.
- 성능 수치는 각 시나리오 내부를 시간 순서로 train/validation/test 분할한 결과입니다. 완전히 새로운 사용자, 장소, 장비 holdout 검증은 아닙니다.

## 검증된 성능 수치

Source: `../ICCAS_final_data.xlsx`

| Method | Accuracy | Precision | Recall | F1-score | Threshold |
| --- | ---: | ---: | ---: | ---: | ---: |
| Raw GPS RNN | 0.9156 | 0.8593 | 0.9941 | 0.9218 | 0.76 |
| Kalman GPS RNN | 0.9494 | 0.9337 | 0.9676 | 0.9503 | 0.45 |
| RTDM only | 0.7752 | 0.6925 | 0.9903 | 0.8150 | 0.51 |
| RTDM + RNN Hybrid | 0.9662 | 0.9380 | 0.9984 | 0.9673 | 0.41 |

## 검증된 처리 속도

측정 환경: 현재 Mac Python 환경, CPU, GPS 16-step window, feature 7개.

| Method | Condition | Time |
| --- | --- | ---: |
| RTDM only | realtime loop, window 1개씩 호출 | avg 0.0078 ms / sequence |
| RNN only | realtime loop, tensor 변환 + forward | avg 0.0988 ms / sequence |
| Hybrid | RTDM + RNN + score combine 추정 | 약 0.107 ms / sequence |

주의: 위 속도는 전처리 완료 후 ready window 1개에 대한 순수 추론 시간입니다. Excel 로딩, 전체 Kalman feature 생성, sequence 생성, normal support 구축, 서버 POST, 지도 렌더링 시간은 포함하지 않습니다.

## 초기 준비 시간

| Step | Time |
| --- | ---: |
| Excel 로딩 | 798.716 ms |
| 전체 Kalman feature 생성 | 246.164 ms |
| sequence 생성 | 10.401 ms |
| normal support 구축 | 49.390 ms |

## 구현 확인

- RTDM token: `x_m`, `y_m`를 35m grid cell로 변환.
- RTDM support: 정상 라벨 train window에서 token, transition, n-gram support를 구축.
- RTDM score: token match, transition match, n-gram match를 합산해 anomaly score 계산.
- Hybrid score: validation 기준으로 RNN score 0.6, RTDM anomaly score 0.4 조합이 선택됨.

## 남은 정확도 리스크

- 데이터 수가 작고 동일 시나리오 내부 분할이라 실제 현장 일반화 성능은 더 낮아질 수 있습니다.
- GPS 배회 라벨은 시나리오 단위 라벨이므로, window 내부의 세밀한 순간 라벨까지 완벽히 반영한다고 보기 어렵습니다.
- RTDM only는 Recall이 높지만 Precision이 낮아 오탐이 많습니다. 단독 최종 판단보다는 빠른 1차 감지나 RNN 보정용으로 쓰는 것이 맞습니다.
- 배포용으로 쓰려면 Kalman 파라미터, scaler, RNN checkpoint, RTDM support set, threshold, hybrid weight를 하나의 artifact로 저장하는 단계가 추가로 필요합니다.
