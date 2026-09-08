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
from aasist_graph_classifier import SwarVedAASISTGraphClassifier

class CachedRawAudioDataset(Dataset):
    def __init__(self, cache_path="ml_engine/data/raw_audio_dataset.pt"):
        data = torch.load(cache_path)
        self.x = data["x"].float() # (N, 48000)
        self.y = data["y"]
        print(f"⚡ Loaded {len(self.x)} Raw Audio Waveform Tensors directly from RAM ({cache_path})", flush=True)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

class SwarVedEndToEndEngine(nn.Module):
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

        print(f"🔒 Wav2Vec2 Feature Extractor & Layers 0..{freeze_until-1} Frozen.", flush=True)
        print(f"🔓 Wav2Vec2 Layers {freeze_until}..{num_layers-1} Unfrozen for End-to-End Fine-Tuning.", flush=True)

        self.classifier_head = SwarVedAASISTGraphClassifier(
            in_dim=768, d_model=256, n_heads=8, num_layers=4, num_classes=2, margin_m=0.35, scale_s=30.0
        )

    def forward(self, wav, targets=None):
        # wav: (B, 48000)
        features, _ = self.wav2vec2(wav) # (B, 149, 768)
        logits = self.classifier_head(features, targets=targets, use_am_softmax=False)
        return logits

def train_end_to_end_engine(cache_path="ml_engine/data/raw_audio_dataset.pt", epochs=15, batch_size=16):
    # Use CPU for stable multi-layer Wav2Vec2 backpropagation without MPS driver stalls
    device = torch.device("cpu")
    print(f"🚀 Initializing End-to-End SwarVed AI Protection Engine on Device: {device}", flush=True)

    dataset = CachedRawAudioDataset(cache_path=cache_path)
    val_size = int(len(dataset) * 0.2)
    train_size = len(dataset) - val_size

    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = SwarVedEndToEndEngine(unfreeze_layers=4).to(device)

    # Differential Learning Rates
    backbone_params = [p for name, p in model.wav2vec2.named_parameters() if p.requires_grad]
    head_params = list(model.classifier_head.parameters())

    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": 1e-5, "weight_decay": 1e-3},
        {"params": head_params, "lr": 3e-4, "weight_decay": 1e-2}
    ])

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-7)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    os.makedirs("ml_engine/models", exist_ok=True)
    save_path = "ml_engine/models/swarved_e2e_best.pt"

    best_val_acc = 0.0
    start_time = time.time()
    accum_steps = 4 # Effective batch size = 16 * 4 = 64

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

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.classifier_head.state_dict(), "ml_engine/models/aasist_best.pt")
            torch.save(model.state_dict(), save_path)
            print(f"   💾 Saved Peak Model Checkpoint to {save_path} (Val Acc: {val_acc:.2f}%)", flush=True)


    elapsed = (time.time() - start_time) / 60.0
    print(f"\n🎉 End-to-End Fine-Tuning Complete in {elapsed:.2f} minutes!")
    print(f"🏆 Peak Validation Accuracy Reached: {best_val_acc:.2f}%")

if __name__ == "__main__":
    train_end_to_end_engine(epochs=15, batch_size=16)


