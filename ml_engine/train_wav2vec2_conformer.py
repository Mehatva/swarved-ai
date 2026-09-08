import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

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

class SwarVedWav2Vec2ConformerHead(nn.Module):
    """
    Literature-grade Conformer Classifier Head on top of pre-trained Wav2Vec2 768-dim embeddings.
    """
    def __init__(self, in_dim=768, d_model=256, n_heads=8, num_layers=4, num_classes=2):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU()
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True
        )
        self.conformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.depthwise_conv = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=17, padding=8, groups=d_model),
            nn.BatchNorm1d(d_model),
            nn.SiLU(),
            nn.Conv1d(d_model, d_model, kernel_size=1),
            nn.Dropout(0.1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(d_model * 2, 128),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x, return_probabilities=False):
        # x: (B, 149, 768)
        out = self.proj(x) # (B, 149, 256)

        out_enc = self.conformer_encoder(out)
        out_conv = self.depthwise_conv(out_enc.transpose(1, 2)).transpose(1, 2)
        out = out_enc + out_conv # Residual connection

        # Global Statistics Pooling (Mean + Std)
        mean = out.mean(dim=1)
        std = out.std(dim=1) + 1e-6
        stats = torch.cat([mean, std], dim=1) # (B, 512)

        logits = self.classifier(stats)
        if return_probabilities:
            return F.softmax(logits, dim=-1)
        return logits

def train_wav2vec2_conformer(cache_path="ml_engine/data/wav2vec2_cached_dataset.pt", epochs=30, batch_size=64, lr=3e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"🚀 Initializing Literature-Grade Wav2Vec2 + Conformer Fine-Tuning on Device: {device}", flush=True)

    dataset = Wav2Vec2CachedDataset(cache_path=cache_path)
    val_size = int(len(dataset) * 0.2)
    train_size = len(dataset) - val_size

    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = SwarVedWav2Vec2ConformerHead(in_dim=768, d_model=256, n_heads=8, num_layers=4, num_classes=2).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    os.makedirs("ml_engine/models", exist_ok=True)
    save_path = "ml_engine/models/conformer_best.pt"

    best_val_acc = 0.0
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)

            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * bx.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == by).sum().item()
            total += by.size(0)

        scheduler.step()
        train_acc = correct / total if total > 0 else 0.0
        train_loss = running_loss / total if total > 0 else 0.0

        # Eval
        model.eval()
        v_correct, v_total, v_loss = 0, 0, 0.0
        bf_correct, bf_total = 0, 0
        sp_correct, sp_total = 0, 0

        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                logits = model(bx)
                loss = criterion(logits, by)
                v_loss += loss.item() * bx.size(0)
                preds = logits.argmax(dim=1)
                v_correct += (preds == by).sum().item()
                v_total += by.size(0)

                for p, target in zip(preds, by):
                    if target.item() == 0:
                        bf_total += 1
                        if p.item() == 0: bf_correct += 1
                    else:
                        sp_total += 1
                        if p.item() == 1: sp_correct += 1

        val_acc = v_correct / v_total if v_total > 0 else 0.0
        val_loss_avg = v_loss / v_total if v_total > 0 else 0.0
        bf_acc = bf_correct / bf_total if bf_total > 0 else 0.0
        sp_acc = sp_correct / sp_total if sp_total > 0 else 0.0

        print(f"Epoch [{epoch:02d}/{epochs:02d}] - Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | Val Loss: {val_loss_avg:.4f} | Val Acc: {val_acc*100:.2f}% (Real: {bf_acc*100:.2f}%, Deepfake: {sp_acc*100:.2f}%)", flush=True)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"   💾 New Peak Model Checkpoint Saved to {save_path} (Val Acc: {val_acc*100:.2f}%)", flush=True)

    duration = time.time() - start_time
    print(f"\n🎉 Training Complete in {duration/60:.2f} minutes!", flush=True)
    print(f"🏆 Peak True Validation Accuracy Reached: {best_val_acc*100:.2f}%", flush=True)

if __name__ == "__main__":
    train_wav2vec2_conformer(epochs=30, batch_size=64, lr=3e-4)
