# Modulated FMCW-OFDM ISAC Simulation

import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c
#from scipy.signal import hann
from scipy.signal.windows import hann #or use np.hanning
from numpy.fft import fft, ifft, fftshift

class ISACSimulator:
    def __init__(self,
                 fc=77e9,
                 B=150e6,
                 T_chirp=20e-6,
                 N_samples=1024,
                 N_chirps=128,
                 R_max=100,
                 SNR_dB=30,
                 zero_pad_factor=8,
                 N_subcarriers=64,
                 CP_len=16):

        self.fc = fc
        self.B = B
        self.T = T_chirp
        self.Ns = N_samples
        self.Nc = N_chirps
        self.lambda_c = c / fc
        self.slope = B / T_chirp
        self.SNR_dB = SNR_dB
        self.zero_pad = zero_pad_factor * N_samples

        self.fs = int(np.ceil((4 * B * R_max) / (T_chirp * c)))
        self.v_max = self.lambda_c / (4 * self.T)

        self.t_fast = np.arange(self.Ns) / self.fs
        self.t_slow = np.arange(self.Nc) * self.T

        self.R_true = np.random.uniform(10, R_max - 10)
        self.v_true = np.random.uniform(-self.v_max + 1, self.v_max - 1)

        self.range_res = (c * self.fs) / (2 * self.slope * self.zero_pad)
        self.range_axis = np.arange(self.zero_pad // 2) * self.range_res
        self.velocity_axis = fftshift(np.fft.fftfreq(self.Nc, d=self.T)) * self.lambda_c / 2

        # OFDM config
        self.N_sub = N_subcarriers
        self.CP_len = CP_len
        self.N_sym = self.Nc
        self.OFDM_len = self.N_sub + self.CP_len

        self.bits = np.random.randint(0, 2, (self.N_sym, self.N_sub * 2))
        self.qpsk = self.bits[:, 0::2] + 1j * self.bits[:, 1::2]
        self.qpsk = (1 - 2 * self.qpsk.real) + 1j * (1 - 2 * self.qpsk.imag)

    def generate_ofdm(self):
        ofdm_symbols = []
        for sym in self.qpsk:
            freq = np.zeros(self.N_sub, dtype=complex)
            freq[:len(sym)] = sym
            time = ifft(freq)
            time_cp = np.concatenate([time[-self.CP_len:], time])
            ofdm_symbols.append(time_cp)
        return np.stack(ofdm_symbols)  # shape: [N_sym, OFDM_len]

    def simulate(self):
        fb = 2 * self.R_true * self.slope / c
        fd = 2 * self.v_true / self.lambda_c

        ofdm_bb = self.generate_ofdm()
        L = min(self.Ns, self.OFDM_len)
        ofdm_bb = ofdm_bb[:, :L]

        phase = 2 * np.pi * (self.fc * self.t_fast[:L] + 0.5 * self.slope * self.t_fast[:L]**2)
        carrier = np.exp(1j * phase)

        tx = ofdm_bb * carrier[None, :]

        # Received signal: apply delay and Doppler
        rx = np.zeros_like(tx, dtype=complex)
        for m, t in enumerate(self.t_slow):
            tau = 2 * (self.R_true + self.v_true * t) / c
            delay_samples = int(round(tau * self.fs))
            if delay_samples < L:
                delayed = np.roll(tx[m], delay_samples)
                doppler = np.exp(1j * 2 * np.pi * fd * t)
                rx[m] = delayed * doppler

        # Add AWGN
        power = np.mean(np.abs(rx)**2)
        snr_lin = 10 ** (self.SNR_dB / 10)
        noise = (np.random.randn(*rx.shape) + 1j * np.random.randn(*rx.shape)) * np.sqrt(power / (2 * snr_lin))
        rx += noise

        # Dechirp (downconvert to baseband)
        dechirped = rx * np.exp(-1j * phase)[None, :]

        # Radar processing
        window = hann(L)
        beat = dechirped * window[None, :]

        range_fft = np.fft.fft(beat, n=self.zero_pad, axis=1)
        range_fft = range_fft[:, :self.zero_pad // 2]
        doppler_fft = fftshift(fft(range_fft, axis=0), axes=0)
        self.RDM = 20 * np.log10(np.abs(doppler_fft) + 1e-6)

        i, j = np.unravel_index(np.argmax(self.RDM), self.RDM.shape)
        self.R_det = self.range_axis[j]
        self.v_det = self.velocity_axis[i]

        # OFDM demodulation
        rx_demod = []
        for sym in dechirped:
            sym = sym[self.CP_len:self.CP_len + self.N_sub]
            rx_sym = fft(sym)
            rx_demod.append(rx_sym)
        self.rx_qpsk = np.stack(rx_demod)

    def evaluate_ofdm(self):
        tx = self.qpsk[:, :self.rx_qpsk.shape[1]]
        rx = self.rx_qpsk

        real_bits_tx = tx.real < 0
        real_bits_rx = rx.real < 0
        imag_bits_tx = tx.imag < 0
        imag_bits_rx = rx.imag < 0

        bit_errors = (real_bits_tx != real_bits_rx) + (imag_bits_tx != imag_bits_rx)
        total_bits = tx.size * 2  # QPSK: 2 bits per symbol

        ber = np.sum(bit_errors) / total_bits
        return ber

    def report(self):
        print(f"🎯 Ground Truth: Range = {self.R_true:.2f} m, Velocity = {self.v_true:.2f} m/s")
        print(f"✅ Detected    : Range = {self.R_det:.2f} m, Velocity = {self.v_det:.2f} m/s")
        print(f"📏 Errors      : ΔR = {abs(self.R_true - self.R_det):.2f} m, Δv = {abs(self.v_true - self.v_det):.2f} m/s")
        print(f"📶 OFDM BER    : {self.evaluate_ofdm():.4f}")

    def plot_rdm(self):
        plt.imshow(self.RDM, extent=[self.range_axis[0], self.range_axis[-1],
                                     self.velocity_axis[0], self.velocity_axis[-1]],
                   origin='lower', aspect='auto', cmap='viridis')
        plt.colorbar(label='Magnitude (dB)')
        plt.xlabel('Range (m)')
        plt.ylabel('Velocity (m/s)')
        plt.title('Range-Doppler Map (ISAC)')
        plt.scatter(self.R_det, self.v_det, color='white', marker='x', label='Detected')
        plt.legend()
        plt.tight_layout()
        #plt.show()
        plt.savefig("test_isaac.pdf")


if __name__ == "__main__":
    sim = ISACSimulator()
    sim.simulate()
    sim.report()
    sim.plot_rdm()
