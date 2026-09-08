import os, glob, random, time
import torch
import torchaudio
import torchaudio.transforms as T

def precompute_wav2vec2_cache(data_dir="ml_engine/data", max_samples_per_class=6000):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"⚡ Pre-computing Literature-Grade Wav2Vec2 Feature Cache on Device: {device}...", flush=True)

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

    print(f"   Found {len(real_samples)} Bonafide (Real) and {len(synth_samples)} Spoof (Synthetic) audio files.", flush=True)
    random.seed(42)
    random.shuffle(real_samples)
    random.shuffle(synth_samples)

    target_count = min(len(real_samples), len(synth_samples), max_samples_per_class)
    selected_samples = real_samples[:target_count] + synth_samples[:target_count]
    random.shuffle(selected_samples)

    print(f"   Loading Wav2Vec2 Base Backbone...", flush=True)
    bundle = torchaudio.pipelines.WAV2VEC2_BASE
    wav2vec2 = bundle.get_model().to(device)
    wav2vec2.eval()

    cached_x = []
    cached_y = []

    print(f"   Extracting 768-dim contextual embeddings for {len(selected_samples)} samples on GPU...", flush=True)
    batch_size = 64
    t0 = time.time()

    for i in range(0, len(selected_samples), batch_size):
        batch_items = selected_samples[i:i+batch_size]
        waveforms = []
        labels = []
        for path, lbl in batch_items:
            try:
                wav, sr = torchaudio.load(path)
                if sr != 16000:
                    resampler = T.Resample(sr, 16000)
                    wav = resampler(wav)
                if wav.shape[0] > 1:
                    wav = wav.mean(dim=0, keepdim=True)
                
                target_len = 48000 # 3 seconds
                if wav.shape[-1] < target_len:
                    pad = target_len - wav.shape[-1]
                    wav = torch.nn.functional.pad(wav, (0, pad))
                else:
                    wav = wav[:, :target_len]
                waveforms.append(wav.squeeze(0))
                labels.append(lbl)
            except Exception:
                pass

        if len(waveforms) == 0:
            continue

        wav_batch = torch.stack(waveforms).to(device)
        lbl_batch = torch.tensor(labels, dtype=torch.long)

        with torch.no_grad():
            feats, _ = wav2vec2(wav_batch) # (B, 149, 768)

        cached_x.append(feats.cpu().half()) # Store in float16 to conserve RAM/Disk
        cached_y.append(lbl_batch)

        if (i // batch_size + 1) % 10 == 0 or (i + batch_size) >= len(selected_samples):
            processed = min(i + batch_size, len(selected_samples))
            print(f"   Progress: {processed}/{len(selected_samples)} samples ({processed/len(selected_samples)*100:.1f}%) in {time.time()-t0:.1f}s", flush=True)

    x_tensor = torch.cat(cached_x, dim=0) # (12000, 149, 768)
    y_tensor = torch.cat(cached_y, dim=0)

    cache_path = os.path.join(data_dir, "wav2vec2_cached_dataset.pt")
    torch.save({"x": x_tensor, "y": y_tensor}, cache_path)
    print(f"🎉 Wav2Vec2 Feature Cache Saved to {cache_path}! Tensor Shape: {x_tensor.shape}, Memory: {x_tensor.element_size() * x_tensor.nelement() / 1e6:.1f} MB", flush=True)

if __name__ == "__main__":
    precompute_wav2vec2_cache(max_samples_per_class=6000)
