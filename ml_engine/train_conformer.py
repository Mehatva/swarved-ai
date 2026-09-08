import os
import time
import torch
import torch.nn as nn
import torchaudio.transforms as T
from dataset_prep import get_dataloaders
from conformer_classifier import SwarVedConformerV2

class SpecAugment(nn.Module):
    def __init__(self, freq_mask_param=8, time_mask_param=16):
        super().__init__()
        self.freq_mask = T.FrequencyMasking(freq_mask_param=freq_mask_param)
        self.time_mask = T.TimeMasking(time_mask_param=time_mask_param)

    def forward(self, x):
        if self.training:
            for b in range(x.shape[0]):
                x[b] = self.freq_mask(x[b])
                x[b] = self.time_mask(x[b])
        return x

def train_model(cache_path="ml_engine/data/cached_dataset.pt", epochs=30, batch_size=64, lr=5e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"🚀 Initializing High-Speed SwarVedConformerV2 Training on Device: {device}", flush=True)

    if not os.path.exists(cache_path):
        print(f"⚠️ Cache file {cache_path} not found! Building 12,000 sample cache first...", flush=True)
        from precache_features import build_feature_cache
        build_feature_cache(max_samples_per_class=6000)

    train_loader, val_loader = get_dataloaders(cache_path=cache_path, batch_size=batch_size)

    model = SwarVedConformerV2(in_channels=3, n_lfcc=60, d_model=128, n_heads=4, num_layers=4, num_classes=2).to(device)
    spec_aug = SpecAugment(freq_mask_param=8, time_mask_param=16).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    os.makedirs("ml_engine/models", exist_ok=True)
    save_path = "ml_engine/models/conformer_best.pt"

    # Pre-training Baseline Evaluation
    model.eval()
    val_correct = 0
    val_total = 0
    val_loss = 0.0
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            val_loss += loss.item() * batch_x.size(0)
            preds = logits.argmax(dim=1)
            val_correct += (preds == batch_y).sum().item()
            val_total += batch_y.size(0)

    baseline_val_acc = val_correct / val_total if val_total > 0 else 0.0
    print(f"   📊 Initial Random Checkpoint Validation Accuracy: {baseline_val_acc*100:.2f}%\n", flush=True)

    best_val_acc = 0.0

    start_time = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            batch_x = spec_aug(batch_x)

            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch_x.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)

        scheduler.step()
        epoch_loss = running_loss / total if total > 0 else 0.0
        epoch_acc = correct / total if total > 0 else 0.0

        # Validation Phase
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                logits = model(batch_x)
                loss = criterion(logits, batch_y)

                val_loss += loss.item() * batch_x.size(0)
                preds = logits.argmax(dim=1)
                val_correct += (preds == batch_y).sum().item()
                val_total += batch_y.size(0)

        val_acc = val_correct / val_total if val_total > 0 else 0.0
        val_loss_avg = val_loss / val_total if val_total > 0 else 0.0

        print(f"Epoch [{epoch:02d}/{epochs:02d}] - Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc*100:.2f}% | Val Loss: {val_loss_avg:.4f} | Val Acc: {val_acc*100:.2f}%", flush=True)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"   💾 New Peak Model Checkpoint Saved to {save_path} (Val Acc: {val_acc*100:.2f}%)", flush=True)

    total_duration = time.time() - start_time
    print(f"\n🎉 Training Complete in {total_duration/60:.2f} minutes!", flush=True)
    print(f"🏆 Peak True Validation Accuracy Reached: {best_val_acc*100:.2f}%", flush=True)

if __name__ == "__main__":
    train_model(epochs=30, batch_size=64, lr=5e-4)
