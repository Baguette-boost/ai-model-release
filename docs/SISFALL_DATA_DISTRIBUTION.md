# SisFall 데이터 분포 정리

## 기준 파일

```text
../data/iccas_sensor_lstm/final_iccas_sisfall_imu_merged.csv
```

아래 표는 병합 CSV에서 `source_dataset == "SisFall"`인 행만 필터링해 계산했습니다.
SisFall은 GPS가 없으므로 최종 시스템에서는 IMU/Gyro 낙상 탐지 학습에만 사용합니다.

## 전체 요약

| 항목 | 값 |
| --- | ---: |
| 전체 row | 324,801 |
| 전체 column | 20 |
| source file 수 | 400 |
| subject 수 | 3 |
| label 종류 | normal, fall |
| activity 종류 | D01-D19, F01-F15 |

## Label 분포

| Label | 의미 | Rows | 비율 |
| --- | --- | ---: | ---: |
| normal | 일상 동작, ADL, Dxx | 174,800 | 53.82% |
| fall | 낙상 동작, Fxx | 150,001 | 46.18% |
| Total |  | 324,801 | 100.00% |

## Subject 분포

| Subject | Rows | 비율 |
| --- | ---: | ---: |
| SA01 | 123,000 | 37.87% |
| SA02 | 123,001 | 37.87% |
| SA03 | 78,800 | 24.26% |
| Total | 324,801 | 100.00% |

## Trial 분포

| Trial | Rows |
| --- | ---: |
| 01 | 94,000 |
| 02 | 58,000 |
| 03 | 58,000 |
| 04 | 57,401 |
| 05 | 57,400 |
| Total | 324,801 |

## Activity 분포

### Normal Activity, D01-D19

| Activity | Label | Rows |
| --- | --- | ---: |
| D01 | normal | 9,000 |
| D02 | normal | 9,000 |
| D03 | normal | 9,000 |
| D04 | normal | 9,000 |
| D05 | normal | 18,750 |
| D06 | normal | 18,750 |
| D07 | normal | 9,000 |
| D08 | normal | 9,000 |
| D09 | normal | 9,000 |
| D10 | normal | 9,000 |
| D11 | normal | 9,000 |
| D12 | normal | 7,800 |
| D13 | normal | 6,000 |
| D14 | normal | 6,000 |
| D15 | normal | 6,000 |
| D16 | normal | 6,000 |
| D17 | normal | 12,500 |
| D18 | normal | 6,000 |
| D19 | normal | 6,000 |
| Total | normal | 174,800 |

### Fall Activity, F01-F15

| Activity | Label | Rows |
| --- | --- | ---: |
| F01 | fall | 11,250 |
| F02 | fall | 11,250 |
| F03 | fall | 11,251 |
| F04 | fall | 11,250 |
| F05 | fall | 11,250 |
| F06 | fall | 11,250 |
| F07 | fall | 11,250 |
| F08 | fall | 11,250 |
| F09 | fall | 11,250 |
| F10 | fall | 11,250 |
| F11 | fall | 7,500 |
| F12 | fall | 7,500 |
| F13 | fall | 7,500 |
| F14 | fall | 7,500 |
| F15 | fall | 7,500 |
| Total | fall | 150,001 |

## 학습 적용 방식

| 구분 | 적용 내용 |
| --- | --- |
| 입력 센서 | `roll`, `pitch`, `yaw`, `ax`, `ay`, `az`, `wx`, `wy`, `wz` |
| 추가 전처리 feature | `accel_norm`, `gyro_norm`, `dt_s` |
| LSTM 입력 window | 50 samples, 25 Hz 기준 약 2초 |
| Positive label | `fall` |
| Negative label | `normal` |
| GPS 사용 여부 | 사용하지 않음 |

## 해석

- SisFall은 현재 병합 데이터의 대부분을 차지하는 외부 IMU 낙상 데이터입니다.
- 정상 데이터가 53.82%, 낙상 데이터가 46.18%라서 낙상 binary 학습에는 비교적 균형이 좋습니다.
- 다만 subject가 `SA01`, `SA02`, `SA03` 세 명으로 제한되어 있어, 실제 현장 일반화를 위해서는 직접 취득한 ICCAS 낙상/비낙상 데이터가 더 필요합니다.
