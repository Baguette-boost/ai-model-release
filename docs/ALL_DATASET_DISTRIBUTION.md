# 전체 데이터셋 분포

## 기준 파일

```text
../data/iccas_sensor_lstm/final_iccas_sisfall_imu_merged.csv
```

ICCAS 직접 취득 데이터와 SisFall IMU 낙상 데이터를 병합한 최종 IMU 낙상 학습용 CSV 기준입니다.

## 전체 요약

| 항목 | 값 |
| --- | ---: |
| 전체 row | 349,560 |
| 전체 column | 20 |
| source_dataset 종류 | 2 |
| label 종류 | 6 |
| activity 종류 | 39 |
| subject 종류 | 4 |
| trial 종류 | 6 |

## Source Dataset 분포

| Item | Description | Rows | Ratio |
| --- | --- | ---: | ---: |
| SisFall | External public IMU fall dataset | 324,801 | 92.92% |
| ICCAS | Directly collected project dataset | 24,759 | 7.08% |
| Total |  | 349,560 | 100.00% |

## Label 분포

| Item | Description | Rows | Ratio |
| --- | --- | ---: | ---: |
| normal | SisFall D01-D19 normal ADL | 174,800 | 50.01% |
| fall | ICCAS fall + SisFall F01-F15 fall | 152,752 | 43.70% |
| wandering | ICCAS GPS/IMU wandering scenario | 12,353 | 3.53% |
| walk | ICCAS walking scenario | 8,003 | 2.29% |
| idle | ICCAS idle scenario | 1,001 | 0.29% |
| sit | ICCAS sitting scenario | 651 | 0.19% |
| Total |  | 349,560 | 100.00% |

## Binary Fall 분포

| Item | Description | Rows | Ratio |
| --- | --- | ---: | ---: |
| fall | positive | 152,752 | 43.70% |
| non_fall | negative | 196,808 | 56.30% |
| Total |  | 349,560 | 100.00% |

## Activity 전체 분포

| Item | Description | Rows | Ratio |
| --- | --- | ---: | ---: |
| D01 | SisFall normal ADL | 9,000 | 2.57% |
| D02 | SisFall normal ADL | 9,000 | 2.57% |
| D03 | SisFall normal ADL | 9,000 | 2.57% |
| D04 | SisFall normal ADL | 9,000 | 2.57% |
| D05 | SisFall normal ADL | 18,750 | 5.36% |
| D06 | SisFall normal ADL | 18,750 | 5.36% |
| D07 | SisFall normal ADL | 9,000 | 2.57% |
| D08 | SisFall normal ADL | 9,000 | 2.57% |
| D09 | SisFall normal ADL | 9,000 | 2.57% |
| D10 | SisFall normal ADL | 9,000 | 2.57% |
| D11 | SisFall normal ADL | 9,000 | 2.57% |
| D12 | SisFall normal ADL | 7,800 | 2.23% |
| D13 | SisFall normal ADL | 6,000 | 1.72% |
| D14 | SisFall normal ADL | 6,000 | 1.72% |
| D15 | SisFall normal ADL | 6,000 | 1.72% |
| D16 | SisFall normal ADL | 6,000 | 1.72% |
| D17 | SisFall normal ADL | 12,500 | 3.58% |
| D18 | SisFall normal ADL | 6,000 | 1.72% |
| D19 | SisFall normal ADL | 6,000 | 1.72% |
| F01 | SisFall fall | 11,250 | 3.22% |
| F02 | SisFall fall | 11,250 | 3.22% |
| F03 | SisFall fall | 11,251 | 3.22% |
| F04 | SisFall fall | 11,250 | 3.22% |
| F05 | SisFall fall | 11,250 | 3.22% |
| F06 | SisFall fall | 11,250 | 3.22% |
| F07 | SisFall fall | 11,250 | 3.22% |
| F08 | SisFall fall | 11,250 | 3.22% |
| F09 | SisFall fall | 11,250 | 3.22% |
| F10 | SisFall fall | 11,250 | 3.22% |
| F11 | SisFall fall | 7,500 | 2.15% |
| F12 | SisFall fall | 7,500 | 2.15% |
| F13 | SisFall fall | 7,500 | 2.15% |
| F14 | SisFall fall | 7,500 | 2.15% |
| F15 | SisFall fall | 7,500 | 2.15% |
| fall | ICCAS fall | 2,751 | 0.79% |
| idle | ICCAS idle | 1,001 | 0.29% |
| sit | ICCAS sit | 651 | 0.19% |
| walk | ICCAS walk | 8,003 | 2.29% |
| wandering | ICCAS wandering | 12,353 | 3.53% |
| Total |  | 349,560 | 100.00% |

## Subject 분포

| Item | Description | Rows | Ratio |
| --- | --- | ---: | ---: |
| SA01 |  | 123,000 | 35.19% |
| SA02 |  | 123,001 | 35.19% |
| SA03 |  | 78,800 | 22.54% |
| local |  | 24,759 | 7.08% |
| Total |  | 349,560 | 100.00% |

## Trial 분포

| Item | Description | Rows | Ratio |
| --- | --- | ---: | ---: |
| 01 |  | 94,000 | 26.89% |
| 02 |  | 58,000 | 16.59% |
| 03 |  | 58,000 | 16.59% |
| 04 |  | 57,401 | 16.42% |
| 05 |  | 57,400 | 16.42% |
| local |  | 24,759 | 7.08% |
| Total |  | 349,560 | 100.00% |

## 학습 적용 해석

- SisFall은 GPS가 없으므로 IMU/Gyro 낙상 탐지 학습에만 사용합니다.
- ICCAS는 walk, wandering, fall, idle, sit 시나리오를 포함합니다.
- LSTM 입력은 `roll`, `pitch`, `yaw`, `ax`, `ay`, `az`, `wx`, `wy`, `wz`, `accel_norm`, `gyro_norm`, `dt_s`의 12개 feature입니다.
- 최종 IMU 낙상 모델은 50 samples, 25 Hz 기준 약 2초 window를 사용합니다.
