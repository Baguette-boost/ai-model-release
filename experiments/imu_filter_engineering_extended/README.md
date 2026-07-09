# IMU Filter Engineering Experiment

Median filter, EMA low-pass filter, and their combination were compared across IMU fall detection models.

## Setup

- Source: `../data/iccas_sensor_lstm/imu_fall_preprocessed.csv`
- Device: `cpu`
- Max epochs: `30`
- Min epochs before early stopping: `15`
- Early stopping patience: `5`
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

| Filter | Model | Epochs trained | Best epoch | Accuracy | Precision | Recall | F1-score | Threshold | Single ms | Train sec |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | rnn | 23 | 18 | 0.7326 | 0.5937 | 0.7943 | 0.6795 | 0.63 | 0.2272 | 43.2 |
| baseline | gru | 28 | 23 | 0.7951 | 0.6630 | 0.8658 | 0.7510 | 0.79 | 0.5679 | 137.7 |
| baseline | lstm | 17 | 12 | 0.7676 | 0.6326 | 0.8320 | 0.7187 | 0.74 | 0.5769 | 101.4 |
| baseline | transformer | 24 | 19 | 0.8131 | 0.6895 | 0.8667 | 0.7680 | 0.90 | 0.2927 | 271.5 |
| baseline | cnn1d | 17 | 12 | 0.8345 | 0.7164 | 0.8875 | 0.7928 | 0.84 | 0.3473 | 387.3 |
| baseline | cnn_lstm | 16 | 11 | 0.7869 | 0.6687 | 0.7985 | 0.7279 | 0.81 | 0.7114 | 201.9 |
| median3 | rnn | 16 | 11 | 0.6988 | 0.5601 | 0.7281 | 0.6331 | 0.69 | 0.2213 | 29.6 |
| median3 | gru | 23 | 18 | 0.7801 | 0.6389 | 0.8830 | 0.7414 | 0.64 | 0.5434 | 113.0 |
| median3 | lstm | 15 | 10 | 0.7556 | 0.6231 | 0.7974 | 0.6996 | 0.83 | 0.5589 | 89.6 |
| median3 | transformer | 17 | 12 | 0.8006 | 0.6765 | 0.8458 | 0.7517 | 0.83 | 0.2888 | 193.2 |
| median3 | cnn1d | 22 | 17 | 0.7935 | 0.6652 | 0.8483 | 0.7457 | 0.88 | 0.3286 | 503.4 |
| median3 | cnn_lstm | 16 | 11 | 0.8029 | 0.6674 | 0.8928 | 0.7638 | 0.40 | 0.7117 | 203.7 |
| ema030 | rnn | 20 | 15 | 0.7151 | 0.5692 | 0.8293 | 0.6751 | 0.46 | 0.2322 | 36.9 |
| ema030 | gru | 16 | 11 | 0.7675 | 0.6278 | 0.8560 | 0.7244 | 0.78 | 0.5559 | 78.8 |
| ema030 | lstm | 15 | 10 | 0.7686 | 0.6317 | 0.8431 | 0.7223 | 0.77 | 0.5585 | 89.6 |
| ema030 | transformer | 16 | 11 | 0.7559 | 0.6087 | 0.8853 | 0.7214 | 0.55 | 0.2911 | 181.8 |
| ema030 | cnn1d | 18 | 13 | 0.7946 | 0.6554 | 0.8949 | 0.7567 | 0.71 | 0.3239 | 410.2 |
| ema030 | cnn_lstm | 16 | 11 | 0.7973 | 0.6590 | 0.8953 | 0.7592 | 0.58 | 0.7123 | 203.8 |
| median3_ema030 | rnn | 15 | 10 | 0.7316 | 0.6016 | 0.7342 | 0.6613 | 0.67 | 0.2394 | 28.3 |
| median3_ema030 | gru | 16 | 11 | 0.7683 | 0.6275 | 0.8635 | 0.7268 | 0.75 | 0.5662 | 79.6 |
| median3_ema030 | lstm | 15 | 10 | 0.7722 | 0.6476 | 0.7939 | 0.7133 | 0.84 | 0.5736 | 90.4 |
| median3_ema030 | transformer | 17 | 12 | 0.7933 | 0.6554 | 0.8877 | 0.7541 | 0.81 | 0.3018 | 194.6 |
| median3_ema030 | cnn1d | 18 | 13 | 0.7881 | 0.6501 | 0.8800 | 0.7478 | 0.73 | 0.3162 | 414.4 |
| median3_ema030 | cnn_lstm | 28 | 23 | 0.8402 | 0.7274 | 0.8830 | 0.7977 | 0.71 | 0.7031 | 350.5 |

## Best Result

- Best filter: `median3_ema030`
- Best model: `cnn_lstm`
- Accuracy: `0.8402`
- Precision: `0.7274`
- Recall: `0.8830`
- F1-score: `0.7977`

## Important Interpretation

Filtering did not change the number of sensor rows. It only changed the signal values. This is important because the experiment isolates the effect of signal filtering from the effect of data size.
