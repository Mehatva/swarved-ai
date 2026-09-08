import os
import glob
import random
import torch
from concurrent.futures import ProcessPoolExecutor
from extract_features import GLOBAL_EXTRACTOR

def process_sample(item):
    path, label = item
    try:
        feat = GLOBAL_EXTRACTOR(path, target_frames=301) # (1, 3, 60, 301)
        return feat.squeeze(0), torch.tensor(label, dtype=torch.long)
    except Exception as e:
        return None

def build_feature_cache(data_dir="ml_engine/data", max_samples_per_class=6000):
    print("⚡ Pre-computing and Caching 12,000 High-Resolution Audio Feature Tensors...")
    asvspoof_dir = os.path.join(data_dir, "asvspoof")
    meta_path = os.path.join(asvspoof_dir, "keys", "LA", "CM", "trial_metadata.txt")

    flac_files = glob.glob(os.path.join(asvspoof_dir, "**", "*.flac"), recursive=True)
    flac_map = {os.path.splitext(os.path.basename(f))[0]: f for f in flac_files}

    real_samples = []
    synth_samples = []

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

    print(f"   Found {len(real_samples)} Bonafide (Real) and {len(synth_samples)} Spoof (Synthetic) FLAC audio files.")
    
    random.seed(42)
    random.shuffle(real_samples)
    random.shuffle(synth_samples)

    target_count = min(len(real_samples), len(synth_samples), max_samples_per_class)
    selected_samples = real_samples[:target_count] + synth_samples[:target_count]
    random.shuffle(selected_samples)

    print(f"   Extracting 3-Channel LFCC+Delta features for {len(selected_samples)} samples across 8 CPU worker processes...")
    
    cached_x = []
    cached_y = []

    with ProcessPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(process_sample, selected_samples, chunksize=50))

    for res in results:
        if res is not None:
            cached_x.append(res[0])
            cached_y.append(res[1])

    x_tensor = torch.stack(cached_x)
    y_tensor = torch.stack(cached_y)

    cache_path = os.path.join(data_dir, "cached_dataset.pt")
    os.makedirs(data_dir, exist_ok=True)
    torch.save({"x": x_tensor, "y": y_tensor}, cache_path)
    print(f"🎉 12,000 Sample Feature Cache Saved to {cache_path}! Tensor Shape: {x_tensor.shape}")

if __name__ == "__main__":
    build_feature_cache(max_samples_per_class=6000)
