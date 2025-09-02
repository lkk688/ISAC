import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c

class FMCWRadarSimCompensated:
    def __init__(self, fc=77e9, B=200e6, T_chirp=40e-6,
                 N_samples=1024, N_chirps=128, SNR_dB=40):
        self.fc = fc
        self.B = B
        self.T_chirp = T_chirp
        self.N_samples = N_samples
        self.N_chirps = N_chirps
        self.fs = 4 * B
        self.slope = B / T_chirp
        self.lambda_c = c / fc
        self.SNR_dB = SNR_dB

        self.t_fast = np.arange(N_samples) / self.fs
        self.t_slow = np.arange(N_chirps) * T_chirp

        # Ground truth (randomized)
        self.R_true = np.random.uniform(10, 90)
        self.v_true = np.random.uniform(-25, 25)

        # Axes
        self.range_axis = np.arange(N_samples // 2) * (c / (2 * B))
        self.velocity_axis = np.fft.fftshift(np.fft.fftfreq(N_chirps, d=T_chirp)) * self.lambda_c / 2

    def simulate(self):
        tx = np.exp(1j * 2 * np.pi * (self.fc * self.t_fast +
                                      0.5 * self.slope * self.t_fast ** 2))
        beat = np.zeros((self.N_chirps, self.N_samples), dtype=complex)

        for m, t in enumerate(self.t_slow):
            Rm = self.R_true + self.v_true * t
            tau = 2 * Rm / c
            rx = np.exp(1j * 2 * np.pi * (self.fc * (self.t_fast - tau) +
                                          0.5 * self.slope * (self.t_fast - tau) ** 2))
            beat[m] = tx * np.conj(rx)

        # Add complex AWGN
        power = np.mean(np.abs(beat)**2)
        snr_lin = 10 ** (self.SNR_dB / 10)
        noise = (np.random.randn(*beat.shape) + 1j * np.random.randn(*beat.shape)) * np.sqrt(power / (2 * snr_lin))
        beat += noise

        # 2D FFT
        range_fft = np.fft.fft(beat, axis=1)[:, :self.N_samples // 2]
        rd_cube = np.fft.fftshift(np.fft.fft(range_fft, axis=0), axes=0)
        self.RDM = 20 * np.log10(np.abs(rd_cube) + 1e-6)

        # Find peak
        i, j = np.unravel_index(np.argmax(self.RDM), self.RDM.shape)

        # FFT bin frequencies
        fb = j * (self.fs / self.N_samples)
        fd = (i - self.N_chirps // 2) * (1 / self.T_chirp / self.N_chirps)

        # Apply compensation for Doppler shift
        R_est = (c / (2 * self.slope)) * (fb - fd)
        v_est = self.velocity_axis[i]

        self.detected = {
            "range": R_est,
            "velocity": v_est,
            "fb": fb,
            "fd": fd
        }

    def report(self):
        print(f"🎯 Ground Truth: Range = {self.R_true:.2f} m, Velocity = {self.v_true:.2f} m/s")
        print(f"✅ Detected    : Range = {self.detected['range']:.2f} m, Velocity = {self.detected['velocity']:.2f} m/s")
        print(f"📏 Errors      : ΔR = {abs(self.R_true - self.detected['range']):.2f} m, "
              f"Δv = {abs(self.v_true - self.detected['velocity']):.2f} m/s")

    def plot(self, out_dir="output"):
        import os
        os.makedirs(out_dir, exist_ok=True)

        # -- Image extent fix for imshow --
        dr = self.range_axis[1] - self.range_axis[0]
        dv = self.velocity_axis[1] - self.velocity_axis[0]
        extent = [self.range_axis[0] - dr / 2, self.range_axis[-1] + dr / 2,
                self.velocity_axis[0] - dv / 2, self.velocity_axis[-1] + dv / 2]

        # -- Find peak FFT bin indices --
        i_peak, j_peak = np.unravel_index(np.argmax(self.RDM), self.RDM.shape)

        # -- Find ground truth FFT bin indices --
        j_gt = np.argmin(np.abs(self.range_axis - self.R_true))
        i_gt = np.argmin(np.abs(self.velocity_axis - self.v_true))

        # ========================================
        # ✅ Figure 1: Plot with FFT peak detected
        # ========================================
        plt.figure(figsize=(10, 6))
        plt.imshow(self.RDM, extent=extent, origin='lower', cmap='viridis', aspect='auto')
        plt.colorbar(label="Magnitude (dB)")
        plt.xlabel("Range (m)")
        plt.ylabel("Velocity (m/s)")
        plt.title("RDM Peak Index as Detection")

        # RDM peak bin (detected)
        plt.scatter(self.range_axis[j_peak], self.velocity_axis[i_peak],
                    color='white', marker='x', s=100, label='Detected (FFT Peak)')

        # Ground truth bin
        plt.scatter(self.range_axis[j_gt], self.velocity_axis[i_gt],
                    facecolors='none', edgecolors='red', s=120, linestyle='--', label='Ground Truth')

        plt.legend()
        plt.tight_layout()
        path1 = os.path.join(out_dir, "rd_map_detected_peak.png")
        plt.savefig(path1)
        plt.close()
        print(f"🖼️ Saved: {path1}")

        # ========================================
        # ✅ Figure 2: Plot with GT FFT bin only
        # ========================================
        plt.figure(figsize=(10, 6))
        plt.imshow(self.RDM, extent=extent, origin='lower', cmap='viridis', aspect='auto')
        plt.colorbar(label="Magnitude (dB)")
        plt.xlabel("Range (m)")
        plt.ylabel("Velocity (m/s)")
        plt.title("Ground Truth Bin Location Only")

        # Ground truth FFT bin
        plt.scatter(self.range_axis[j_gt], self.velocity_axis[i_gt],
                    facecolors='none', edgecolors='red', s=120, linestyle='--', label='Ground Truth (FFT Bin)')

        plt.legend()
        plt.tight_layout()
        path2 = os.path.join(out_dir, "rd_map_ground_truth_bin.png")
        plt.savefig(path2)
        plt.close()
        print(f"🖼️ Saved: {path2}")

    def plot2(self, out_dir="output"):
        os.makedirs(out_dir, exist_ok=True)
        plt.figure(figsize=(10, 6))
        plt.imshow(self.RDM,
                   extent=[self.range_axis[0], self.range_axis[-1],
                           self.velocity_axis[0], self.velocity_axis[-1]],
                   aspect='auto', origin='lower', cmap='viridis')
        plt.colorbar(label="Magnitude (dB)")
        plt.xlabel("Range (m)")
        plt.ylabel("Velocity (m/s)")
        plt.title("Range-Doppler Map")

        # Mark ground truth
        plt.scatter(self.R_true, self.v_true, facecolors='none', edgecolors='red',
                    s=100, linestyle='--', label='Ground Truth')
        # Mark detected
        plt.scatter(self.detected['range'], self.detected['velocity'],
                    color='white', marker='x', label='Detected')

        plt.legend()
        path = os.path.join(out_dir, "rd_map.png")
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        print(f"🖼️ RD map saved to: {path}")


if __name__ == "__main__":
    sim = FMCWRadarSimCompensated()
    sim.simulate()
    sim.report()
    sim.plot()