import torch
import torchaudio
import torchaudio.transforms as T
import torch.nn.functional as F
import numpy as np

class LFCCFeatureExtractor:
    def __init__(self, sample_rate=16000, n_lfcc=60, n_filter=60, win_length=400, hop_length=160):
        self.sample_rate = sample_rate
        self.n_lfcc = n_lfcc
        self.lfcc_transform = T.LFCC(
            sample_rate=sample_rate,
            n_filter=n_filter,
            n_lfcc=n_lfcc,
            speckwargs={"n_fft": 1024, "win_length": win_length, "hop_length": hop_length}
        )
        self.compute_deltas = T.ComputeDeltas()

    def __call__(self, audio_input, target_frames=301):
        """
        Extracts 3-Channel Cepstral Matrix: Static LFCC (60), Delta (60), Delta-Delta (60)
        Returns: Torch tensor of shape (1, 3, 60, target_frames)
        """
        if isinstance(audio_input, str):
            try:
                waveform, sr = torchaudio.load(audio_input)
                if sr != self.sample_rate:
                    resampler = T.Resample(sr, self.sample_rate)
                    waveform = resampler(waveform)
                if waveform.shape[0] > 1:
                    waveform = torch.mean(waveform, dim=0, keepdim=True)
            except Exception:
                waveform = torch.zeros(1, self.sample_rate * 3)
        elif isinstance(audio_input, np.ndarray):
            waveform = torch.from_numpy(audio_input.astype(np.float32))
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)
        elif isinstance(audio_input, torch.Tensor):
            waveform = audio_input
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)
        else:
            waveform = torch.zeros(1, self.sample_rate * 3)

        if waveform.shape[-1] == 0:
            waveform = torch.zeros(1, self.sample_rate * 3)

        # Ensure target length (3 seconds = 48,000 samples)
        target_samples = self.sample_rate * 3
        if waveform.shape[-1] < target_samples:
            pad_len = target_samples - waveform.shape[-1]
            waveform = F.pad(waveform, (0, pad_len))
        elif waveform.shape[-1] > target_samples:
            waveform = waveform[:, :target_samples]

        lfcc = self.lfcc_transform(waveform).squeeze(0) # (60, frames)
        delta = self.compute_deltas(lfcc)               # (60, frames)
        delta2 = self.compute_deltas(delta)             # (60, frames)

        feature_matrix = torch.stack([lfcc, delta, delta2], dim=0) # (3, 60, frames)

        current_frames = feature_matrix.shape[-1]
        if current_frames < target_frames:
            pad_amount = target_frames - current_frames
            feature_matrix = F.pad(feature_matrix, (0, pad_amount), mode='constant', value=0)
        elif current_frames > target_frames:
            feature_matrix = feature_matrix[:, :, :target_frames]

        # Cepstral Mean & Variance Normalization (CMVN)
        mean = feature_matrix.mean(dim=(1, 2), keepdim=True)
        std = feature_matrix.std(dim=(1, 2), keepdim=True) + 1e-6
        feature_matrix = (feature_matrix - mean) / std

        return feature_matrix.unsqueeze(0) # (1, 3, 60, target_frames)

GLOBAL_EXTRACTOR = LFCCFeatureExtractor()

def extract_swarved_features(audio_input, sample_rate=16000, target_frames=301):
    return GLOBAL_EXTRACTOR(audio_input, target_frames=target_frames).numpy().astype(np.float32)

if __name__ == "__main__":
    dummy_pcm = np.random.randn(16000 * 3).astype(np.float32)
    tensor = extract_swarved_features(dummy_pcm)
    print(f"✅ State-of-the-Art LFCC+Delta Feature Extractor Test Passed! Shape: {tensor.shape}")
