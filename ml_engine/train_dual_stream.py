import os
import glob
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader

from aasist_graph_classifier import SwarVedAASISTGraphClassifier, AMSoftmaxLoss
from lfcc_extractor import LFCCExtractor
from res2net_stream import Res2NetPhaseStream
from rawboost import RawBoostAugmentation

class SwarVedDualStreamEngine(nn.Module):
    """
    SwarVed AI Dual-Stream Spectro-Temporal Engine with OPUS Codec Adaptation:
    - Stream A (SSL Acoustic Branch): Wav2Vec2 (Layers 8..11 unfrozen) + 12-Layer Fusion + AASIST HS-GAT (128-dim)
    - Stream B (Phase-Spectral Branch): LFCC Filterbank (180-dim) + Multi-Scale Res2Net-50 (128-dim)
    - Dual Fusion: Concatenated 256-dim representation + AMSoftmax Loss (m=0.35, s=30.0)
    """
    def __init__(self, unfreeze_layers=4):
        super().__init__()
        bundle = torchaudio.pipelines.WAV2VEC2_BASE
        self.wav2vec2 = bundle.get_model()

        # Freeze feature extractor
        for param in self.wav2vec2.feature_extractor.parameters():
            param.requires_grad = False

        # Freeze lower transformer layers
        num_layers = len(self.wav2vec2.encoder.transformer.layers)
        freeze_until = num_layers - unfreeze_layers

        for i in range(freeze_until):
            for param in self.wav2vec2.encoder.transformer.layers[i].parameters():
                param.requires_grad = False

        # Learnable 12-layer softmax weights for SSL layer fusion
        self.layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)

        # Stream A (SSL AASIST Graph Classifier Head)
        self.stream_a_classifier = SwarVedAASISTGraphClassifier(
            in_dim=768, d_model=256, n_heads=8, num_layers=4, num_classes=2, margin_m=0.35, scale_s=30.0
        )

        # Stream B (LFCC + Res2Net Phase Stream)
        self.lfcc_extractor = LFCCExtractor()
        self.stream_b_res2net = Res2NetPhaseStream(in_dim=180, d_model=256, out_dim=128)

        # Fused AM-Softmax Angular Classification Head (256-dim input -> 2 classes)
        self.dual_am_softmax = AMSoftmaxLoss(in_features=256, num_classes=2, s=30.0, m=0.35)

    def forward(self, wav, targets=None):
        features_list, _ = self.wav2vec2.extract_features(wav, num_layers=None)
        weights = F.softmax(self.layer_weights, dim=0)
        
        fused_features = 0
        for i, feat in enumerate(features_list):
            fused_features = fused_features + weights[i] * feat

        emb_a = self.stream_a_classifier(fused_features, return_embeddings=True)

        lfcc = self.lfcc_extractor(wav)
        emb_b = self.stream_b_res2net(lfcc)

        fused_emb = torch.cat([emb_a, emb_b], dim=1) # (B, 256)
        logits = self.dual_am_softmax(fused_emb, targets if self.training else None)
        return logits

class CachedDatasetSubset(Dataset):
    def __init__(self, cache_path, indices, is_train=True, extra_samples=None):
        data = torch.load(cache_path)
        x_list = [data["x"][i] for i in indices]
        y_list = [data["y"][i] for i in indices]

        # Add extra mobile voice samples if provided
        if extra_samples and is_train:
            for extra_wav, extra_y in extra_samples:
                x_list.append(extra_wav)
                y_list.append(extra_y)

        self.x = torch.stack(x_list).float()
        self.y = torch.tensor(y_list, dtype=torch.long)
        self.rawboost = RawBoostAugmentation(algo_types=[1, 2, 3, 4, 5]) if is_train else None

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        wav = self.x[idx]
        target = self.y[idx]
        if self.rawboost is not None:
            wav = self.rawboost(wav)
        return wav, target

def load_extra_whatsapp_samples():
    extra_samples = []
    whatsapp_path = "/Users/mehatva/Downloads/WhatsApp Audio 2026-09-07 at 14.52.32.opus"
    if os.path.exists(whatsapp_path):
        try:
            wav, sr = torchaudio.load(whatsapp_path)
            if sr != 16000:
                wav = T.Resample(sr, 16000)(wav)
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)
            
            target_len = 48000
            stride = 24000
            for start in range(0, wav.shape[-1] - target_len + 1, stride):
                seg = wav[:, start:start+target_len].squeeze(0)
                # Label 0 = Bonafide Real Human Voice
                extra_samples.append((seg, 0))
            print(f"📱 Loaded {len(extra_samples)} real WhatsApp voice segments into training pipeline!", flush=True)
        except Exception as e:
            print(f"Warning: Could not load WhatsApp audio: {e}")
    return extra_samples

def train_dual_stream_engine(cache_path="ml_engine/data/raw_audio_dataset.pt", epochs=10, batch_size=16):
    device = torch.device("cpu")
    print(f"🚀 Initializing OPUS-Adapted Dual-Stream SwarVed Engine on Device: {device}", flush=True)

    full_dataset = torch.load(cache_path)
    total_len = len(full_dataset["x"])
    val_size = int(total_len * 0.2)
    train_size = total_len - val_size

    train_indices, val_indices = torch.utils.data.random_split(
        range(total_len), [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )

    extra_samples = load_extra_whatsapp_samples()

    train_data = CachedDatasetSubset(cache_path, list(train_indices), is_train=True, extra_samples=extra_samples)
    val_data = CachedDatasetSubset(cache_path, list(val_indices), is_train=False)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    model = SwarVedDualStreamEngine(unfreeze_layers=4).to(device)
    
    # Load existing peak checkpoint to fine-tune
    checkpoint_path = "ml_engine/models/swarved_dual_stream_best.pt"
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"   Loaded trained Dual-Stream checkpoint weights from {checkpoint_path}", flush=True)

    backbone_params = [p for name, p in model.wav2vec2.named_parameters() if p.requires_grad]
    layer_weight_params = [model.layer_weights]
    stream_a_params = list(model.stream_a_classifier.parameters())
    stream_b_params = list(model.stream_b_res2net.parameters())
    dual_am_params = list(model.dual_am_softmax.parameters())

    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": 5e-6, "weight_decay": 1e-3},
        {"params": layer_weight_params, "lr": 5e-5, "weight_decay": 0.0},
        {"params": stream_a_params + stream_b_params + dual_am_params, "lr": 1e-4, "weight_decay": 1e-2}
    ])

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-7)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.03)

    os.makedirs("ml_engine/models", exist_ok=True)
    save_path = "ml_engine/models/swarved_dual_stream_best.pt"

    best_val_acc = 0.0
    start_time = time.time()
    accum_steps = 4

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        optimizer.zero_grad()

        for step, (wav, y) in enumerate(train_loader):
            wav, y = wav.to(device), y.to(device)

            logits = model(wav, targets=y)
            loss = criterion(logits, y) / accum_steps
            loss.backward()

            if (step + 1) % accum_steps == 0 or (step + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            train_loss += loss.item() * accum_steps * len(y)
            preds = logits.argmax(dim=1)
            train_correct += (preds == y).sum().item()
            train_total += len(y)

            if (step + 1) % 20 == 0 or (step + 1) == len(train_loader):
                print(f"   Epoch [{epoch:02d}/{epochs:02d}] - Step [{step+1}/{len(train_loader)}] | Loss: {loss.item()*accum_steps:.4f} | Acc: {(train_correct/train_total)*100:.2f}%", flush=True)

        scheduler.step()
        train_loss /= train_total
        train_acc = (train_correct / train_total) * 100.0

        # Evaluation Phase
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        real_correct, real_total = 0, 0
        fake_correct, fake_total = 0, 0

        with torch.no_grad():
            for wav, y in val_loader:
                wav, y = wav.to(device), y.to(device)
                logits = model(wav, targets=None)
                loss = criterion(logits, y)

                val_loss += loss.item() * len(y)
                preds = logits.argmax(dim=1)
                val_correct += (preds == y).sum().item()
                val_total += len(y)

                real_mask = (y == 0)
                fake_mask = (y == 1)
                real_correct += (preds[real_mask] == 0).sum().item()
                real_total += real_mask.sum().item()
                fake_correct += (preds[fake_mask] == 1).sum().item()
                fake_total += fake_mask.sum().item()

        val_loss /= val_total
        val_acc = (val_correct / val_total) * 100.0
        real_acc = (real_correct / real_total) * 100.0 if real_total > 0 else 0.0
        fake_acc = (fake_correct / fake_total) * 100.0 if fake_total > 0 else 0.0

        print(f"Epoch [{epoch:02d}/{epochs:02d}] - Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% (Real: {real_acc:.2f}%, Deepfake: {fake_acc:.2f}%)", flush=True)

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"   💾 Saved OPUS-Adapted Model Checkpoint to {save_path} (Val Acc: {val_acc:.2f}%)", flush=True)

    elapsed = (time.time() - start_time) / 60.0
    print(f"\n🎉 OPUS Domain Adaptation Complete in {elapsed:.2f} minutes!")
    print(f"🏆 Final Peak Validation Accuracy: {best_val_acc:.2f}%")

if __name__ == "__main__":
    train_dual_stream_engine(epochs=10, batch_size=16)
