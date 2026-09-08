import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

class LFCCExtractor(nn.Module):
    """
    Linear Frequency Cepstral Coefficients (LFCC) Feature Extractor.
    Extracts 60-channel static LFCC + Delta + Delta-Delta (180-dim representation)
    to capture vocoder phase discontinuities and high-frequency spectral artifacts.
    """
    def __init__(self, sample_rate=16000, n_fft=512, win_length=400, hop_length=160, n_lfcc=60, n_filter=128):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_lfcc = n_lfcc

        # Linear Filterbank matrix
        linear_filterbank = self._create_linear_filterbank(sample_rate, n_fft, n_filter)
        self.register_buffer("filterbank", linear_filterbank)

        # DCT Basis matrix
        dct_matrix = self._create_dct_matrix(n_lfcc, n_filter)
        self.register_buffer("dct_matrix", dct_matrix)

        # Delta Kernel (fixed shape (60, 1, 5) for ONNX export)
        kernel = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0]) / 10.0
        kernel = kernel.view(1, 1, 5).repeat(n_lfcc, 1, 1)
        self.register_buffer("delta_kernel", kernel)

    def _create_linear_filterbank(self, sample_rate, n_fft, n_filter):
        num_fft_bins = n_fft // 2 + 1
        fft_freqs = torch.linspace(0, sample_rate / 2.0, num_fft_bins)
        filter_freqs = torch.linspace(0, sample_rate / 2.0, n_filter + 2)

        filterbank = torch.zeros(num_fft_bins, n_filter)
        for i in range(n_filter):
            f_m_minus = filter_freqs[i]
            f_m = filter_freqs[i + 1]
            f_m_plus = filter_freqs[i + 2]

            for j in range(num_fft_bins):
                f = fft_freqs[j]
                if f_m_minus <= f <= f_m:
                    filterbank[j, i] = (f - f_m_minus) / (f_m - f_m_minus + 1e-6)
                elif f_m <= f <= f_m_plus:
                    filterbank[j, i] = (f_m_plus - f) / (f_m_plus - f + 1e-6)

        return filterbank

    def _create_dct_matrix(self, n_lfcc, n_filter):
        n = torch.arange(n_filter).float()
        k = torch.arange(n_lfcc).float().unsqueeze(1)
        dct = torch.cos(torch.pi / n_filter * (n + 0.5) * k)
        dct[0] *= 1.0 / torch.sqrt(torch.tensor(2.0))
        dct *= torch.sqrt(torch.tensor(2.0 / n_filter))
        return dct

    def compute_deltas(self, x):
        # x: (B, C, T)
        return F.conv1d(x, self.delta_kernel, padding=2, groups=x.size(1))

    def forward(self, wav):
        # wav: (B, 48000)
        window = torch.hann_window(self.win_length, device=wav.device)
        stft = torch.stft(
            wav,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            return_complex=False
        )
        spectrogram = stft.pow(2).sum(-1) # (B, Freq, T)

        # Apply Linear Filterbank
        # spectrogram: (B, Freq, T), filterbank: (Freq, n_filter)
        linear_spec = torch.matmul(spectrogram.transpose(1, 2), self.filterbank).transpose(1, 2)
        log_spec = torch.log(linear_spec + 1e-6)

        # Apply DCT to get LFCC
        lfcc = torch.matmul(self.dct_matrix, log_spec) # (B, 60, T)

        # Compute Delta & Delta-Delta
        delta = self.compute_deltas(lfcc)
        delta2 = self.compute_deltas(delta)

        # Concatenate static + delta + delta2 -> (B, 180, T)
        lfcc_full = torch.cat([lfcc, delta, delta2], dim=1)
        return lfcc_full

if __name__ == "__main__":
    extractor = LFCCExtractor()
    dummy_wav = torch.randn(2, 48000)
    feats = extractor(dummy_wav)
    print(f"✅ LFCC Extractor Test: Input {dummy_wav.shape} -> LFCC Features {feats.shape}")
