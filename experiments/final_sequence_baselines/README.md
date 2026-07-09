# Final IMU RNN and Transformer Baselines

This experiment retrains RNN and Transformer models with the same data protocol used by the final IMU LSTM.

## Protocol

- Source CSV: `../data/iccas_sensor_lstm/final_iccas_sisfall_imu_merged.csv`
- Device: `cpu`
- Features: `roll, pitch, yaw, ax, ay, az, wx, wy, wz, accel_norm, gyro_norm, dt_s`
- Sequence length: `50` samples = `2.0 seconds` at 25 Hz
- Sequence stride: `4`
- Split: SisFall group/hash split and ICCAS chronological split, matching the final LSTM script
- Scaling: Robust median/IQR scaler fitted on train only, then clipped to [-12, 12]
- Early stopping: minimum 15 epochs, then stop after validation F1 does not improve

## Test Results

| Model | Accuracy | Precision | Recall | F1-score | Forward ms | Process ms | Train sec | Best epoch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RNN | 0.8029 | 0.7971 | 0.7463 | 0.7709 | 0.236 | 0.238 | 41.9 | 14 |
| TRANSFORMER | 0.8828 | 0.9225 | 0.8038 | 0.8591 | 0.187 | 0.189 | 265.1 | 9 |

Forward ms is model execution only for one 50-sample sequence. Process ms includes NumPy-to-tensor creation plus model execution, but not the 2-second sensor acquisition window, API, or database time.

## Final LSTM Reference

The current production candidate remains `models/iccas_final_hybrid_lstm_imu_fall.pt`. It uses the same final merged data, 50-sample windows, robust scaling, and CPU evaluation protocol.

| Model | Accuracy | Precision | Recall | F1-score | Forward ms | Process ms | Best epoch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FINAL LSTM | 0.8799 | 0.8498 | 0.8863 | 0.8677 | 1.120 | 1.123 | 10 |

The Transformer is slightly higher in accuracy and precision, but the final LSTM has the best F1-score and recall among the final-data models, so LSTM is still the safer fall-detection choice when missed falls are more important than false alarms.
