import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

class RawBoostAugmentation:
    """
    SwarVed AI RawBoost Data Augmentation with Mobile OPUS/AAC Codec Adaptation (Tak et al., 2022 + Mobile Codec).
    Applies direct 1D raw waveform signal processing:
    1. Algo 1: Linear & Non-linear Convolutive Noise (Sinc/Harmonic Filter Response)
    2. Algo 2: Impulsive Signal-Dependent Additive Noise (Transmission Clicks/Dropouts)
    3. Algo 3: Stationary Additive Gaussian Noise (Background Telephony Static)
    4. Algo 4: Combined Convolutive + Additive Telephony Noise
    5. Algo 5: OPUS/AAC Mobile Codec Simulation (4-7kHz Band Cutoff + Predictive Quantization Noise)
    6. Algo 6: Real-World Mic/Room Noise (Pink Noise + Simple Room Echo + Breath Artifacts)
    7. Algo 7: Combined Mic/Room Noise + OPUS Codec (Real Phone Call Simulation)
    """
    def __init__(self, sample_rate=16000, algo_types=[1, 2, 3, 4, 5]):
        self.sample_rate = sample_rate
        self.algo_types = algo_types

    def apply_convolutive_noise(self, x, num_harmonics=5):
        n = x.shape[-1]
        t = torch.linspace(0, 1, n, device=x.device)
        filter_mask = torch.ones(n, device=x.device)
        for h in range(1, num_harmonics + 1):
            freq = torch.tensor(random.uniform(100.0, 4000.0), device=x.device)
            gain = torch.tensor(random.uniform(0.1, 0.4), device=x.device)
            filter_mask += gain * torch.sin(2 * np.pi * freq * t * h)
        
        y = x * (filter_mask / filter_mask.max())
        y = torch.tanh(y * 1.1)
        return y

    def apply_impulsive_noise(self, x, p=0.005):
        mask = torch.rand_like(x) < p
        impulses = (torch.rand_like(x) * 2.0 - 1.0) * x.abs().max() * 0.8
        y = x.clone()
        y[mask] = impulses[mask]
        return y

    def apply_additive_noise(self, x, snr_db_range=(15, 35)):
        snr_db = random.uniform(snr_db_range[0], snr_db_range[1])
        signal_power = torch.mean(x ** 2)
        if signal_power == 0:
            return x
        noise_power = signal_power / (10 ** (snr_db / 10.0))
        noise = torch.randn_like(x) * torch.sqrt(noise_power)
        return x + noise

    def apply_opus_codec_simulation(self, x):
        """
        Simulates OPUS / AAC / AMR lossy mobile codec compression:
        1. Low-pass spectral cutoff (4kHz - 7.5kHz cutoff)
        2. Frame-predictive quantization noise
        """
        n = x.shape[-1]
        # FFT low-pass filter to simulate OPUS 4kHz - 7.5kHz bandwidth limits
        x_fft = torch.fft.rfft(x)
        cutoff_bin = int(random.uniform(0.5, 0.9) * x_fft.shape[-1])
        
        # Smooth roll-off filter
        filter_curve = torch.ones_like(x_fft.real)
        filter_curve[cutoff_bin:] *= torch.exp(-torch.linspace(0, 3, x_fft.shape[-1] - cutoff_bin, device=x.device))
        
        x_filtered = torch.fft.irfft(x_fft * filter_curve, n=n)
        
        # Add 16kbps OPUS frame quantization noise (small uniform step noise)
        q_step = random.uniform(0.005, 0.02)
        x_quantized = torch.round(x_filtered / q_step) * q_step
        
        return x_quantized

    def apply_room_mic_noise(self, x, snr_db_range=(12, 30)):
        """
        Simulates real-world microphone & room environment noise:
        1. Pink noise (1/f spectral shape) - realistic mic background hiss
        2. Simple first-order room echo (single reflection ~100-250ms)
        3. Occasional breath artifacts (short low-frequency burst)
        """
        n = x.shape[-1]
        sr = 16000

        # Pink noise: generate white, then apply 1/f spectral shaping
        white = torch.randn(n, device=x.device)
        freqs = torch.fft.rfftfreq(n, d=1.0 / sr, device=x.device)
        freqs[0] = 1.0  # avoid div-by-zero at DC
        pink_filter = 1.0 / torch.sqrt(freqs)
        pink_filter /= pink_filter.mean()
        pink = torch.fft.irfft(torch.fft.rfft(white) * pink_filter, n=n)
        pink = pink / (pink.abs().max() + 1e-8)

        # Scale to desired SNR
        sig_power = torch.mean(x ** 2)
        if sig_power < 1e-10:
            return x
        snr_db = random.uniform(snr_db_range[0], snr_db_range[1])
        noise_power = sig_power / (10 ** (snr_db / 10.0))
        noise = pink * torch.sqrt(noise_power)

        # Simple room echo: single reflection at 80-250 ms delay
        echo_delay_ms = random.uniform(80, 250)
        echo_delay = int(echo_delay_ms * sr / 1000.0)
        echo_gain = random.uniform(0.05, 0.20)
        echo = torch.zeros_like(x)
        if echo_delay < n:
            echo[echo_delay:] = x[:n - echo_delay] * echo_gain

        # Occasional breath artifact: 20% chance, short low-freq burst
        if random.random() < 0.20:
            breath_len = int(random.uniform(0.05, 0.15) * sr)  # 50-150ms
            breath_start = random.randint(0, max(0, n - breath_len - 1))
            breath = torch.randn(breath_len, device=x.device)
            # Low-pass the breath (keep only < 500 Hz)
            breath_fft = torch.fft.rfft(breath)
            cutoff = int(500 * breath_len / sr)
            breath_fft[cutoff:] = 0
            breath = torch.fft.irfft(breath_fft, n=breath_len)
            breath = breath / (breath.abs().max() + 1e-8) * random.uniform(0.01, 0.06)
            noise[breath_start: breath_start + breath_len] += breath

        return x + noise + echo

    def __call__(self, waveform):
        orig_shape = waveform.shape
        x = waveform.squeeze()
        
        # Apply RawBoost augmentation with 50% probability during training
        if random.random() > 0.5:
            return waveform

        algo = random.choice(self.algo_types) if self.algo_types else 0
        if algo == 1:
            x = self.apply_convolutive_noise(x)
        elif algo == 2:
            x = self.apply_impulsive_noise(x)
        elif algo == 3:
            x = self.apply_additive_noise(x)
        elif algo == 4:
            x = self.apply_convolutive_noise(x)
            x = self.apply_additive_noise(x)
        elif algo == 5:
            x = self.apply_opus_codec_simulation(x)
        elif algo == 6:
            x = self.apply_room_mic_noise(x)
        elif algo == 7:
            x = self.apply_room_mic_noise(x)
            x = self.apply_opus_codec_simulation(x)

        return x.view(orig_shape)

if __name__ == "__main__":
    rawboost = RawBoostAugmentation()
    dummy_audio = torch.randn(48000)
    augmented = rawboost(dummy_audio)
    print(f"✅ OPUS-Upgraded RawBoost Test: Input shape {dummy_audio.shape} -> Augmented shape {augmented.shape}")
