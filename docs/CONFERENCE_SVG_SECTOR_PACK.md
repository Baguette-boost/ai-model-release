# Conference SVG Sector Pack

GPS를 제외한 최종 IMU 낙상 감지 범위에 맞춰 학술대회 발표/포스터용 SVG를 섹터별로 정리했다.

## 생성 명령

```bash
cd /Volumes/Hub_1T/ICCAS/ai-model-release
../.venv/bin/python scripts/generate_conference_sector_svgs.py
```

## SVG 파일

- `assets/conference_sectors/sector_01_dataset_construction.svg`
- `assets/conference_sectors/sector_02_iccas_sisfall_distribution.svg`
- `assets/conference_sectors/sector_03_imu_preprocessing.svg`
- `assets/conference_sectors/sector_04_lstm_training_method.svg`
- `assets/conference_sectors/sector_05_final_performance.svg`
- `assets/conference_sectors/sector_06_preprocessing_effect.svg`
- `assets/conference_sectors/sector_07_realtime_inference_architecture.svg`
- `assets/conference_sectors/sector_08_professor_final_feedback.svg`

## 최종 표현 원칙

- 최종 범위는 GPS가 아니라 IMU 낙상 감지다.
- 모델 명칭은 `전처리 feature 기반 LSTM 낙상 감지 모델`로 설명한다.
- 핵심 성능은 Accuracy보다 Recall과 F1-score 중심으로 제시한다.
- 전처리는 성능을 항상 올리는 마법이 아니라 IMU 신호의 물리적 의미를 강화하는 과정으로 설명한다.
