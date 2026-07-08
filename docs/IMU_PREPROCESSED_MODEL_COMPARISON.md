# 전처리 후 IMU 낙상 모델 성능 요약

이 문서는 과거 실험명을 정리한 호환용 요약입니다. 발표와 공유 자료에서는 `전처리 적용 전/후 비교`라는 표현을 사용합니다.

## 사용한 전처리

- 전처리 전 입력: `roll, pitch, yaw, ax, ay, az, wx, wy, wz`
- 전처리 후 입력: 위 9개 feature에 `accel_norm, gyro_norm, dt_s` 추가
- 목적: 모델이 충격 크기, 회전 크기, 샘플 간 시간 간격을 직접 학습할 수 있게 만드는 것

## 최종 비교 문서

- 상세 표: `docs/IMU_PREPROCESSING_EFFECT_COMPARISON.md`
- 시각화: `assets/imu_preprocessing_effect_comparison.svg`
- 재현 스크립트: `scripts/compare_imu_preprocessing_effect.py`

## 발표용 핵심 문장

- 모든 모델에 동일한 IMU 전처리를 적용해 비교했다.
- F1-score는 RNN과 Transformer에서 상승했고, Recall은 4개 모델 중 3개 모델에서 상승했다.
- 전처리는 규칙 기반 판단을 섞는 것이 아니라, LSTM/RNN/GRU/Transformer가 학습할 입력 feature를 더 물리적으로 명확하게 만드는 단계이다.
