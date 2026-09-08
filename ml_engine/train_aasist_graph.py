import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from aasist_graph_classifier import SwarVedAASISTGraphClassifier

class Wav2Vec2CachedDataset(Dataset):
    def __init__(self, cache_path="ml_engine/data/wav2vec2_cached_dataset.pt"):
        data = torch.load(cache_path)
        self.x = data["x"].float() # (N, 149, 768)
        self.y = data["y"]
        print(f"⚡ Loaded {len(self.x)} Pre-Cached 768-dim Wav2Vec2 Feature Tensors from RAM ({cache_path})", flush=True)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

def train_aasist_graph_model(cache_path="ml_engine/data/wav2vec2_cached_dataset.pt", epochs=35, batch_size=64, lr=3e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"🚀 Initializing World-Record SSL-AASIST Graph Attention + AM-Softmax Model on Device: {device}", flush=True)

    dataset = Wav2Vec2CachedDataset(cache_path=cache_path)
    val_size = int(len(dataset) * 0.2)
    train_size = len(dataset) - val_size

    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = SwarVedAASISTGraphClassifier(
        in_dim=768, d_model=256, n_heads=8, num_layers=4, num_classes=2, margin_m=0.35, scale_s=30.0
    ).to(device)

    # Standard CrossEntropy on AM-Softmax Margin Logits
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    os.makedirs("ml_engine/models", exist_ok=True)
    save_path = "ml_engine/models/aasist_best.pt"

    best_val_acc = 0.0
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            # Pass targets to trigger AM-Softmax Angular Margin during training
            logits = model(x, targets=y)
            loss = criterion(logits, y)
            loss.backward()

            # Gradient clipping for stable convergence
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * len(y)
            preds = logits.argmax(dim=1)
            train_correct += (preds == y).sum().item()
            train_total += len(y)

        scheduler.step()

        train_loss /= train_total
        train_acc = (train_correct / train_total) * 100.0

        # Evaluation Phase
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        real_correct, real_total = 0, 0
        fake_correct, fake_total = 0, 0

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x, targets=None) # Evaluates clean cosine logits
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

        log_line = (f"Epoch [{epoch:02d}/{epochs:02d}] - Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                    f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% (Real: {real_acc:.2f}%, Deepfake: {fake_acc:.2f}%)")
        print(log_line, flush=True)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"   💾 New Peak Model Checkpoint Saved to {save_path} (Val Acc: {val_acc:.2f}%)", flush=True)

    elapsed = (time.time() - start_time) / 60.0
    print(f"\n🎉 SSL-AASIST Training Complete in {elapsed:.2f} minutes!")
    print(f"🏆 Peak True Validation Accuracy Reached: {best_val_acc:.2f}%")

if __name__ == "__main__":
    train_aasist_graph_model()
