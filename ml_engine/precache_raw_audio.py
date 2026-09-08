import os
import glob
import random
import time
import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T

def precompute_raw_audio_cache(data_dir="ml_engine/data", max_samples_per_class=3400):
    print("⚡ Pre-loading 16kHz Raw Audio Tensors into RAM Cache...", flush=True)

    asvspoof_dir = os.path.join(data_dir, "asvspoof")
    meta_path = os.path.join(asvspoof_dir, "keys", "LA", "CM", "trial_metadata.txt")

    flac_files = glob.glob(os.path.join(asvspoof_dir, "**", "*.flac"), recursive=True)
    flac_map = {os.path.splitext(os.path.basename(f))[0]: f for f in flac_files}

    real_samples, synth_samples = [], []
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 6:
                    file_id = parts[1]
                    label_str = parts[5].lower()
                    if file_id in flac_map:
                        if label_str == "bonafide":
                            real_samples.append((flac_map[file_id], 0))
                        elif label_str == "spoof":
                            synth_samples.append((flac_map[file_id], 1))

    random.seed(42)
    random.shuffle(real_samples)
    random.shuffle(synth_samples)

    target_count = min(len(real_samples), len(synth_samples), max_samples_per_class)
    selected_samples = real_samples[:target_count] + synth_samples[:target_count]
    random.shuffle(selected_samples)

    print(f"   Found {len(selected_samples)} audio files ({target_count} Real, {target_count} Deepfake).", flush=True)

    waveforms = []
    labels = []
    t0 = time.time()

    for idx, (path, label) in enumerate(selected_samples):
        try:
            wav, sr = torchaudio.load(path)
            if sr != 16000:
                resampler = T.Resample(sr, 16000)
                wav = resampler(wav)
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)

            target_len = 48000 # 3 seconds @ 16kHz
            if wav.shape[-1] < target_len:
                pad = target_len - wav.shape[-1]
                wav = F.pad(wav, (0, pad))
            else:
                wav = wav[:, :target_len]

            waveforms.append(wav.squeeze(0))
            labels.append(label)
        except Exception:
            pass

        if (idx + 1) % 1000 == 0 or (idx + 1) == len(selected_samples):
            print(f"   Pre-loaded {idx+1}/{len(selected_samples)} audio files into RAM ({time.time()-t0:.1f}s)", flush=True)

    x_tensor = torch.stack(waveforms) # (N, 48000)
    y_tensor = torch.tensor(labels, dtype=torch.long)

    cache_path = os.path.join(data_dir, "raw_audio_dataset.pt")
    torch.save({"x": x_tensor, "y": y_tensor}, cache_path)
    size_mb = (x_tensor.element_size() * x_tensor.nelement()) / 1e6
    print(f"🎉 Raw Audio RAM Cache Saved to {cache_path}! Shape: {x_tensor.shape}, Size: {size_mb:.1f} MB", flush=True)

if __name__ == "__main__":
    precompute_raw_audio_cache()
