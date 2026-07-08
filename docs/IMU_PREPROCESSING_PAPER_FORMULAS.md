# IMU Fall Detection Preprocessing Equations

## Notation

Let one IMU sample at time step \(t\) be composed of attitude, acceleration, angular velocity, and timestamp:

\[
\mathbf{r}_t =
\begin{bmatrix}
\phi_t & \theta_t & \psi_t
\end{bmatrix}^{\top},
\quad
\mathbf{a}_t =
\begin{bmatrix}
a_{x,t} & a_{y,t} & a_{z,t}
\end{bmatrix}^{\top},
\quad
\boldsymbol{\omega}_t =
\begin{bmatrix}
\omega_{x,t} & \omega_{y,t} & \omega_{z,t}
\end{bmatrix}^{\top}.
\]

Here, \(\mathbf{r}_t\) denotes roll, pitch, and yaw in degrees, \(\mathbf{a}_t\) denotes acceleration in \(g\), and \(\boldsymbol{\omega}_t\) denotes angular velocity in degrees per second.

## Magnitude Features

The acceleration vector magnitude, also referred to as signal vector magnitude, is computed as:

\[
s_t = \left\lVert \mathbf{a}_t \right\rVert_2
    = \sqrt{a_{x,t}^{2} + a_{y,t}^{2} + a_{z,t}^{2}}.
\tag{1}
\]

The gyroscope magnitude is computed as:

\[
g_t = \left\lVert \boldsymbol{\omega}_t \right\rVert_2
    = \sqrt{\omega_{x,t}^{2} + \omega_{y,t}^{2} + \omega_{z,t}^{2}}.
\tag{2}
\]

In the implementation, \(s_t\) is stored as `accel_norm` or `svm_g`, and \(g_t\) is stored as `gyro_norm`.

## Temporal Feature

The time interval between adjacent samples is derived from the timestamp \(m_t\) in milliseconds:

\[
\Delta t_t =
\frac{\operatorname{clip}(m_t - m_{t-1}, 0, 1000)}{1000}.
\tag{3}
\]

For the first sample in a sequence, \(\Delta t_0 = 0\). In the current dataset, the nominal sampling period is \(40\) ms, corresponding to \(25\) Hz.

## LSTM Input Vector

Each time step is converted into a 12-dimensional feature vector:

\[
\mathbf{x}_t =
\begin{bmatrix}
\mathbf{r}_t &
\mathbf{a}_t &
\boldsymbol{\omega}_t &
s_t &
g_t &
\Delta t_t
\end{bmatrix}^{\top}
\in \mathbb{R}^{12}.
\tag{4}
\]

A fixed-length input window is then constructed using \(L=50\) consecutive samples:

\[
\mathbf{X}_k =
\begin{bmatrix}
\mathbf{x}_{k-L+1}, \mathbf{x}_{k-L+2}, \ldots, \mathbf{x}_{k}
\end{bmatrix}^{\top}
\in \mathbb{R}^{50 \times 12}.
\tag{5}
\]

At 25 Hz, this window corresponds to approximately 2 seconds of IMU motion.

## Sequence Label

The binary fall target at time \(t\) is defined as:

\[
y_t =
\begin{cases}
1, & \text{if the activity label is fall},\\
0, & \text{otherwise}.
\end{cases}
\tag{6}
\]

The sequence-level label is positive if at least one sample within the window is a fall sample:

\[
Y_k = \max_{i \in \{k-L+1,\ldots,k\}} y_i.
\tag{7}
\]

## Robust Normalization

For each feature dimension \(j\), the center and scale are computed from the training set:

\[
c_j = \operatorname{median}(x_{:,j}),
\quad
q_j = Q_{0.75}(x_{:,j}) - Q_{0.25}(x_{:,j}).
\tag{8}
\]

The robust scale is selected as:

\[
\sigma_j =
\begin{cases}
q_j, & q_j > \epsilon,\\
\operatorname{std}(x_{:,j}), & q_j \le \epsilon \ \land \ \operatorname{std}(x_{:,j}) > \epsilon,\\
1, & \text{otherwise},
\end{cases}
\quad \epsilon = 10^{-6}.
\tag{9}
\]

The normalized feature is:

\[
z_{t,j}
= \operatorname{clip}
\left(
\frac{x_{t,j} - c_j}{\sigma_j},
-12,
12
\right).
\tag{10}
\]

The final LSTM input is:

\[
\mathbf{Z}_k \in \mathbb{R}^{50 \times 12}.
\tag{11}
\]

## Physical Fall Context Features

For a window \(W_k = \{k-L+1,\ldots,k\}\), impact, free-fall, and rotation-related features are:

\[
I_k = \max_{t \in W_k} s_t,
\quad
F_k = \min_{t \in W_k} s_t,
\quad
R_k = \max_{t \in W_k} g_t.
\tag{12}
\]

The posture change is:

\[
\Theta_k =
\max_{t \in W_k}
\sqrt{
(\phi_t-\phi_{k-L+1})^2
+
(\theta_t-\theta_{k-L+1})^2
}.
\tag{13}
\]

Let \(t^\ast = \arg\max_{t \in W_k} s_t\) be the impact index. The post-impact inactivity statistics are:

\[
P_k = \{t \mid t^\ast < t \le t^\ast + 25\},
\tag{14}
\]

\[
A^{std}_k = \operatorname{std}\left(s_t \mid t \in P_k\right),
\quad
G^{mean}_k = \operatorname{mean}\left(g_t \mid t \in P_k\right).
\tag{15}
\]

These physical features are used as auxiliary fall-context indicators. The final tuned checkpoint currently uses the LSTM probability as the primary decision score.
