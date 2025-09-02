import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c


class FMCWRadarSimulation:
    def __init__(self,
                 fc=77e9,
                 B=150e6,
                 T_chirp=20e-6,          # ⬅️ Shorter chirp to support high velocity
                 N_samples=1024,
                 N_chirps=128,
                 R_max=100,
                 SNR_dB=40,
                 zero_pad_factor=8):

        self.fc = fc
        self.B = B
        self.T = T_chirp
        self.Ns = N_samples
        self.Nc = N_chirps
        self.lambda_c = c / fc
        self.slope = B / T_chirp
        self.SNR_dB = SNR_dB
        self.zero_pad = zero_pad_factor * N_samples

        # ✅ Sampling rate from max range (anti-aliasing)
        self.fs = int(np.ceil((4 * B * R_max) / (T_chirp * c)))

        # ✅ Doppler max velocity
        self.v_max = self.lambda_c / (4 * self.T)

        # Time vectors
        self.t_fast = np.arange(self.Ns) / self.fs
        self.t_slow = np.arange(self.Nc) * self.T

        # ✅ Ground truth within valid range
        self.R_true = np.random.uniform(10, R_max - 10)
        self.v_true = np.random.uniform(-self.v_max + 1, self.v_max - 1)

        # ✅ Range axis using correct beat-to-range conversion
        range_res = (c * self.fs) / (2 * self.slope * self.zero_pad)
        self.range_axis = np.arange(self.zero_pad // 2) * range_res

        # ✅ Velocity axis from Doppler FFT
        self.velocity_axis = np.fft.fftshift(np.fft.fftfreq(self.Nc, d=self.T)) * self.lambda_c / 2

    def simulate(self):
        fb = 2 * self.R_true * self.slope / c
        fd = 2 * self.v_true / self.lambda_c

        # Analytic beat signal
        beat = np.exp(1j * 2 * np.pi * (
            fb * self.t_fast[None, :] + fd * self.t_slow[:, None]
        ))

        # Hann window
        window = np.hanning(self.Ns)
        beat *= window[None, :]

        # AWGN
        power = np.mean(np.abs(beat)**2)
        snr_linear = 10 ** (self.SNR_dB / 10)
        noise = (np.random.randn(*beat.shape) + 1j * np.random.randn(*beat.shape)) * np.sqrt(power / (2 * snr_linear))
        beat += noise

        # FFTs
        range_fft = np.fft.fft(beat, n=self.zero_pad, axis=1)
        range_fft = range_fft[:, :self.zero_pad // 2]
        doppler_fft = np.fft.fftshift(np.fft.fft(range_fft, axis=0), axes=0)

        self.RDM = 20 * np.log10(np.abs(doppler_fft) + 1e-6)

        # Peak detection
        self.i_peak, self.j_peak = np.unravel_index(np.argmax(self.RDM), self.RDM.shape)
        self.R_det = self.range_axis[self.j_peak]
        self.v_det = self.velocity_axis[self.i_peak]

    def report(self):
        print(f"🎯 Ground Truth : Range = {self.R_true:.2f} m, Velocity = {self.v_true:.2f} m/s")
        print(f"✅ Detected     : Range = {self.R_det:.2f} m, Velocity = {self.v_det:.2f} m/s")
        print(f"📏 Errors       : ΔRange = {abs(self.R_true - self.R_det):.2f} m, "
              f"ΔVelocity = {abs(self.v_true - self.v_det):.2f} m/s")

    def plot(self, out_dir="output"):
        os.makedirs(out_dir, exist_ok=True)

        j_gt = np.argmin(np.abs(self.range_axis - self.R_true))
        i_gt = np.argmin(np.abs(self.velocity_axis - self.v_true))

        dr = self.range_axis[1] - self.range_axis[0]
        dv = self.velocity_axis[1] - self.velocity_axis[0]
        extent = [self.range_axis[0] - dr/2, self.range_axis[-1] + dr/2,
                  self.velocity_axis[0] - dv/2, self.velocity_axis[-1] + dv/2]

        plt.figure(figsize=(10, 6))
        plt.imshow(self.RDM, extent=extent, origin='lower', cmap='viridis', aspect='auto')
        plt.colorbar(label="Magnitude (dB)")
        plt.xlabel("Range (m)")
        plt.ylabel("Velocity (m/s)")
        plt.title("Range-Doppler Map (Aliasing-Free in Range & Velocity)")

        plt.scatter(self.R_det, self.v_det, color='white', marker='x', s=100, label='Detected')
        plt.scatter(self.range_axis[j_gt], self.velocity_axis[i_gt],
                    facecolors='none', edgecolors='red', s=120, linestyle='--', label='Ground Truth')

        plt.legend()
        path = os.path.join(out_dir, "rd_map_final_no_alias.png")
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        print(f"🖼️ Saved RD map to: {path}")


if __name__ == "__main__":
    sim = FMCWRadarSimulation()
    sim.simulate()
    sim.report()
    sim.plot()