# IMU Fall Preprocessing Formulas

## 1. Raw Input

The IMU fall detector receives one sample every 40 ms.

```text
sample_rate = 25 Hz
sample_ms = 40 ms
sequence_length = 50
window_seconds = 50 x 40 ms = 2.0 s
```

Raw columns:

```text
roll, pitch, yaw,
ax, ay, az,
wx, wy, wz,
t_ms, label
```

Units used in the current dataset:

```text
ax, ay, az: g
wx, wy, wz: deg/s
roll, pitch, yaw: deg
t_ms: ms
```

## 2. Sensor Vector Magnitude

Acceleration magnitude is calculated from the 3-axis accelerometer.

```text
accel_norm_t = sqrt(ax_t^2 + ay_t^2 + az_t^2)
```

In ESP32-side notation this is the same as SVM:

```text
svm_t = sqrt(ax_t^2 + ay_t^2 + az_t^2)
```

The current training CSV uses the column name:

```text
accel_norm
```

The exported SVM CSV uses:

```text
svm_g
```

These two values represent the same physical quantity.

## 3. Gyroscope Magnitude

Angular velocity magnitude is calculated from the 3-axis gyroscope.

```text
gyro_norm_t = sqrt(wx_t^2 + wy_t^2 + wz_t^2)
```

This value is used to detect fast rotation during a fall-like event.

## 4. Time Delta

For each device/group, the sample interval is calculated from `t_ms`.

```text
dt_ms_t = t_ms_t - t_ms_(t-1)
dt_ms_t = clip(dt_ms_t, 0, 1000)
dt_s_t  = dt_ms_t / 1000
```

For the first sample in each group:

```text
dt_s_0 = 0
```

In normal 25 Hz data:

```text
dt_s ~= 0.04
```

## 5. Fall Target Label

The binary target used for LSTM training is:

```text
fall_target_t = 1, if label_t == "fall"
fall_target_t = 0, otherwise
```

For a sequence window, the sequence label is positive if any sample inside the window is fall:

```text
y_window = max(fall_target_start ... fall_target_end)
```

## 6. LSTM Feature Vector

Each time step is converted into a 12-dimensional feature vector.

```text
x_t = [
  roll_t,
  pitch_t,
  yaw_t,
  ax_t,
  ay_t,
  az_t,
  wx_t,
  wy_t,
  wz_t,
  accel_norm_t,
  gyro_norm_t,
  dt_s_t
]
```

The model input shape is:

```text
X_window = [50, 12]
```

## 7. Robust Scaling

Before training/inference, each feature is robust-scaled using training-set statistics.

```text
center_j = median(feature_j)
scale_j  = Q3(feature_j) - Q1(feature_j)
```

If the interquartile range is too small, the standard deviation is used. If that is also too small, scale is set to 1.

```text
scale_j = IQR_j, if IQR_j > 1e-6
scale_j = std_j, if IQR_j <= 1e-6 and std_j > 1e-6
scale_j = 1, otherwise
```

Scaled feature:

```text
z_(t,j) = (x_(t,j) - center_j) / scale_j
z_(t,j) = clip(z_(t,j), -12, 12)
```

## 8. Physical Fall Algorithm Features

The LSTM is the final selected model, but the project also computes physical fall features for analysis and safety context.

For one 50-step window:

```text
impact_peak = max(accel_norm_t)
freefall_min = min(accel_norm_t)
gyro_peak = max(gyro_norm_t)
```

Posture change from the first sample:

```text
tilt_change_t = sqrt((roll_t - roll_0)^2 + (pitch_t - pitch_0)^2)
tilt_change = max(tilt_change_t)
```

Impact index:

```text
impact_index = argmax(accel_norm_t)
```

Post-impact segment:

```text
post_window = samples after impact_index, up to post_samples
post_samples = 25
```

Post-fall inactivity features:

```text
post_accel_std = std(accel_norm in post_window)
post_gyro_mean = mean(gyro_norm in post_window)
```

Inactivity gate:

```text
inactivity = 1,
  if post_accel_std <= 0.35 and post_gyro_mean <= 80

inactivity = 0,
  otherwise
```

## 9. Physical Algorithm Score

Current thresholds:

```text
impact_g = 2.5
freefall_g = 0.6
gyro_dps = 250
tilt_deg = 45
still_accel_std = 0.35
still_gyro_dps = 80
```

Impact score:

```text
impact_score = clip((impact_peak - impact_g) / impact_g, 0, 1)
```

Free-fall score:

```text
freefall_score = 1, if freefall_min <= freefall_g
freefall_score = 0, otherwise
```

Rotation score:

```text
gyro_score = clip(gyro_peak / gyro_dps, 0, 1)
tilt_score = clip(tilt_change / tilt_deg, 0, 1)
rotation_score = max(gyro_score, tilt_score)
```

Algorithm score:

```text
algorithm_score =
  0.40 * impact_score
  + 0.20 * rotation_score
  + 0.25 * inactivity
  + 0.15 * max(freefall_score, tilt_score)
```

Final clipping:

```text
algorithm_score = clip(algorithm_score, 0, 1)
```

## 10. Final Inference Rule

The tuned final model currently selected the LSTM score only:

```text
hybrid_lstm_weight = 1.0
hybrid_algorithm_weight = 0.0
```

Therefore the current final score is:

```text
fall_score = LSTM(X_window)
```

Final decision:

```text
fall_detected = true,  if fall_score >= 0.35
fall_detected = false, otherwise
```

The algorithm score is still useful as an auxiliary explanation, but it is not improving the final validation F1 in the current trained checkpoint.

## 11. Code References

Preprocessing:

```text
scripts/export_imu_preprocessed_csv.py
scripts/train_sisfall_merged_imu_lstm.py
```

Physical fall score:

```text
scripts/train_hybrid_imu_fall.py
```

SVM export:

```text
scripts/export_imu_svm_csv.py
```
