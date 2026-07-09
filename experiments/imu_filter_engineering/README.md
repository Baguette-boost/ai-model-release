# IMU Filter Engineering Experiment

Median filter, EMA low-pass filter, and their combination were compared across IMU fall detection models.

## Setup

- Source: `../data/iccas_sensor_lstm/imu_fall_preprocessed.csv`
- Device: `cpu`
- Epochs: `8`
- Sequence length: `50`
- Sequence stride: `4`
- Median window: `3`
- EMA alpha: `0.3`

## Row Change Analysis

| Filter | Sensor rows before | Sensor rows after | Row delta | Sequence total | Sequence delta |
|---|---:|---:|---:|---:|---:|
| baseline | 349560 | 349560 | 0 | 82650 | 0 |
| median3 | 349560 | 349560 | 0 | 82650 | 0 |
| ema030 | 349560 | 349560 | 0 | 82650 | 0 |
| median3_ema030 | 349560 | 349560 | 0 | 82650 | 0 |

해석:

- Median/EMA filtering은 smoothing 방식이므로 sensor row를 삭제하지 않는다.
- 따라서 `sensor_rows_delta`는 0이다.
- LSTM sequence 수 역시 같은 sequence length/stride를 사용하므로 variant 간 동일하다.
- 즉, 이번 비교의 성능 차이는 row 수 변화가 아니라 신호값 변화에서 발생한다.

## Performance Metrics

| Filter | Model | Accuracy | Precision | Recall | F1-score | Threshold | Single ms | Train sec |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | lstm | 0.7301 | 0.5836 | 0.8506 | 0.6922 | 0.72 | 0.5709 | 65.7 |
| baseline | cnn1d | 0.8312 | 0.7261 | 0.8464 | 0.7817 | 0.90 | 0.3101 | 180.8 |
| baseline | cnn_lstm | 0.8277 | 0.7051 | 0.8890 | 0.7864 | 0.41 | 0.7022 | 119.1 |
| median3 | lstm | 0.7540 | 0.6096 | 0.8642 | 0.7149 | 0.81 | 0.5615 | 63.4 |
| median3 | cnn1d | 0.8014 | 0.6669 | 0.8859 | 0.7610 | 0.51 | 0.2995 | 182.4 |
| median3 | cnn_lstm | 0.7872 | 0.6563 | 0.8481 | 0.7399 | 0.44 | 0.7197 | 118.5 |
| ema030 | lstm | 0.7613 | 0.6288 | 0.8085 | 0.7074 | 0.74 | 0.5592 | 64.2 |
| ema030 | cnn1d | 0.8059 | 0.6738 | 0.8844 | 0.7649 | 0.67 | 0.3225 | 181.4 |
| ema030 | cnn_lstm | 0.7794 | 0.6444 | 0.8521 | 0.7339 | 0.65 | 0.6979 | 121.0 |
| median3_ema030 | lstm | 0.7297 | 0.5957 | 0.7548 | 0.6659 | 0.81 | 0.5678 | 64.1 |
| median3_ema030 | cnn1d | 0.7888 | 0.6580 | 0.8500 | 0.7418 | 0.74 | 0.3190 | 178.1 |
| median3_ema030 | cnn_lstm | 0.7767 | 0.6289 | 0.9130 | 0.7448 | 0.40 | 0.7045 | 117.6 |

## Best Result

- Best filter: `baseline`
- Best model: `cnn_lstm`
- Accuracy: `0.8277`
- Precision: `0.7051`
- Recall: `0.8890`
- F1-score: `0.7864`

## Important Interpretation

Filtering did not change the number of sensor rows. It only changed the signal values. This is important because the experiment isolates the effect of signal filtering from the effect of data size.
