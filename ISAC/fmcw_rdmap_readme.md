# FMCW Radar Simulation: In-Depth Guide to Accurate Range-Doppler Mapping

This document provides a **complete, mathematically grounded guide** to building a correct **FMCW radar simulation** in Python that yields accurate **range and velocity** detection. The final solution resolves common issues like range/velocity mismatch and aliasing using signal theory and system design best practices.

---

## ✅ Goals

* Simulate realistic FMCW radar signal returns for a moving target
* Build a 2D **Range-Doppler Map (RDM)** using FFT
* **Ensure alignment** between ground truth and detected target

---

## 1. FMCW Radar Basics

### 1.1 Transmitted Chirp

FMCW radar transmits a chirp:

$$
  s_{\text{tx}}(t) = \exp\left(j 2\pi \left[f_c t + \frac{S}{2} t^2\right]\right)
$$

* $f_c$: Carrier frequency (e.g., 77 GHz)
* $S = \frac{B}{T}$: Chirp slope (Hz/s)
* $B$: Bandwidth (Hz)
* $T$: Chirp duration (s)

### 1.2 Received Signal

The received signal is delayed and Doppler shifted:

$$
  s_{\text{rx}}(t) = \exp\left(j 2\pi \left[f_c (t - \tau) + \frac{S}{2}(t - \tau)^2\right]\right)
$$

Where:

* $\tau = \frac{2R}{c}$: Round-trip delay
* $R$: Range to target
* $c$: Speed of light

### 1.3 Beat Signal

The receiver multiplies $s_{\text{tx}}(t) \cdot s^*_{\text{rx}}(t)$, yielding a beat signal:

$$
  s_{\text{beat}}(t) \approx \exp\left(j 2\pi \left[f_b t + f_d m T\right]\right)
$$

Where:

* $f_b = \frac{2 R S}{c}$: Beat frequency (Hz) proportional to range
* $f_d = \frac{2 v}{\lambda}$: Doppler frequency (Hz) proportional to velocity
* $m$: Chirp index
* $T$: Chirp repetition interval

---

## 2. Key Design Parameters

| Parameter         | Symbol     | Example Value        |
| ----------------- | ---------- | -------------------- |
| Carrier Frequency | $f_c$      | 77 GHz               |
| Bandwidth         | $B$        | 150 MHz              |
| Chirp Duration    | $T$        | 20 us (optimized)    |
| Max Range         | $R_{\max}$ | 100 m                |
| Max Velocity      | $v_{\max}$ | 30 m/s (desired)     |
| Sampling Rate     | $f_s$      | Computed (see below) |

### 2.1 Range Resolution

$$
\Delta R = \frac{c}{2B}
$$

### 2.2 Max Unambiguous Range

To avoid range aliasing:

$$
  f_s \geq \frac{4 B R_{\max}}{T c}
$$

### 2.3 Max Unambiguous Velocity

$$
  v_{\max} = \frac{\lambda}{4 T}
$$

Ensure that ground truth $v$ stays within this bound to prevent Doppler aliasing.

---

## 3. Final Corrected Simulation Steps

### Step 1: Compute Sampling Rate

```python
fs = ceil((4 * B * R_max) / (T * c))
```

### Step 2: Generate Beat Signal

Analytically model the beat signal over 2D grid:

```python
s[n, m] = exp(j*2*pi*(f_b * t + f_d * m*T))
```

* Use `np.exp` and broadcasting to generate a matrix of shape `[N_chirps, N_samples]`

### Step 3: Apply Hann Window + AWGN

* Use `np.hanning(N_samples)` to reduce spectral leakage
* Add complex Gaussian noise based on target SNR (in dB)

### Step 4: 2D FFT with Zero-Padding

```python
range_fft = fft(beat, n=zero_pad, axis=1)
doppler_fft = fftshift(fft(range_fft[:, :zero_pad//2], axis=0), axes=0)
```

### Step 5: Compute Correct Range Axis

$$
  \Delta R = \frac{c f_s}{2 S N_{\text{fft}}}
$$

```python
range_axis = np.arange(zero_pad // 2) * (c * fs) / (2 * slope * zero_pad)
```

### Step 6: Doppler Axis

```python
velocity_axis = fftshift(fftfreq(N_chirps, d=T)) * lambda_c / 2
```

### Step 7: Detection and Alignment

* Find peak in RDM:

  ```python
  i_peak, j_peak = np.unravel_index(np.argmax(RDM), RDM.shape)
  ```
* Detected:

  ```python
  R_det = range_axis[j_peak]
  v_det = velocity_axis[i_peak]
  ```
* Ground truth bin:

  ```python
  j_gt = argmin(abs(range_axis - R_true))
  i_gt = argmin(abs(velocity_axis - v_true))
  ```

---

## 4. Key Plots

### Range-Doppler Map

* Show `RDM` via `imshow`
* Mark:

  * Ground truth as red dashed circle at (range\_axis\[j\_gt], velocity\_axis\[i\_gt])
  * Detected as white "x" at (R\_det, v\_det)

---

## 5. Faced Problems and Solutions

### ❌ Problem 1: Detected Range Far from Ground Truth

* **Cause**: Wrong range bin calculation after FFT zero-padding
* **Fix**: Use $\Delta R = \frac{c f_s}{2 S N_{\text{fft}}}$ to build `range_axis`

### ❌ Problem 2: Velocity Accurate but Range Always \~3 m

* **Cause**: Range aliasing due to undersampling (beat frequency exceeds Nyquist)
* **Fix**: Increase `fs` based on $R_{\max}$

### ❌ Problem 3: Detected Velocity \~+12 m/s when Ground Truth is −27 m/s

* **Cause**: Doppler aliasing (true velocity exceeds unambiguous bound)
* **Fix**: Shorten chirp duration `T_chirp` and restrict target $v$ to within $v_{\max} = \frac{\lambda}{4T}$

### ❌ Problem 4: Ground Truth Marker Doesn’t Align With RDM Peak

* **Cause**: Scatter plotted with physical values, not FFT bin centers
* **Fix**: Align using closest FFT bin:

  ```python
  j_gt = np.argmin(abs(range_axis - R_true))
  i_gt = np.argmin(abs(velocity_axis - v_true))
  ```

---

## ✅ Conclusion

With proper signal modeling, sampling, FFT bin mapping, and aliasing constraints, your FMCW simulation will produce accurate, physically aligned range-Doppler maps.

This simulation now models:

* Analytical signal generation
* Noise injection
* Windowing and FFT leakage
* Range and Doppler bin spacing

**And eliminates:**

* Range wraparound
* Velocity aliasing
* FFT bin misalignment

Ready to be extended to multiple targets, CFAR, or moving scenes.

---

**Next Steps:**

* [ ] Add multiple targets
* [ ] Add CFAR detector
* [ ] Export `.npy`, `.csv`
* [ ] Animate moving vehicles
