"""
SwarVed AI — Noise-Robust Head Fine-Tuning
==========================================
Goal: Make the model robust to real-world microphone & room noise WITHOUT
      re-training the expensive Wav2Vec2 backbone.

Strategy:
  - Freeze Wav2Vec2 + AASIST (Stream A) entirely.
  - Fine-tune only Stream B (LFCC-Res2Net) + dual_am_softmax fusion head.
  - Apply the new Algo 6 & 7 (room mic noise + codec) augmentations
    at a HIGHER rate (80% prob instead of 50%) to force noise robustness.
  - Run only 5 epochs — fast even on CPU (~8-15 minutes).

Usage:
    cd /Users/mehatva/Desktop/Projects/Hackathons/SIH_2026/swarved-ai
    python3 ml_engine/finetune_noise_robust.py
"""

import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from train_dual_stream import SwarVedDualStreamEngine
from rawboost import RawBoostAugmentation

# ─── Config ─────────────────────────────────────────────────────────────────
CACHE_PATH      = "ml_engine/data/raw_audio_dataset.pt"
CHECKPOINT_IN   = "ml_engine/models/swarved_dual_stream_best.pt"
CHECKPOINT_OUT  = "ml_engine/models/swarved_dual_stream_noise_robust.pt"
EPOCHS          = 5
BATCH_SIZE      = 8     # smaller batch — only head layers, still fast
LR_HEAD         = 5e-4  # Stream B + fusion head learning rate
NOISE_AUG_PROB  = 0.80  # 80 % of samples get mic/room noise
VAL_FRAC        = 0.20
SEED            = 42
# ────────────────────────────────────────────────────────────────────────────


class NoiseRobustDataset(Dataset):
    """
    Loads raw-audio cache and applies ONLY the new real-world noise augmentation
    (Algos 6 & 7) during training so the head learns noise invariance.
    """
    def __init__(self, cache_path, indices, is_train=True):
        data = torch.load(cache_path)
        self.x = torch.stack([data["x"][i] for i in indices]).float()
        self.y = torch.tensor([data["y"][i] for i in indices], dtype=torch.long)
        self.is_train = is_train
        # Use only real-world mic noise algos during this fine-tune pass
        self.noise_aug = RawBoostAugmentation(algo_types=[6, 7])

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        wav = self.x[idx]
        target = self.y[idx]
        if self.is_train and torch.rand(1).item() < NOISE_AUG_PROB:
            wav = self.noise_aug(wav)
        return wav, target


def freeze_stream_a(model: SwarVedDualStreamEngine):
    """Freeze Wav2Vec2 backbone + AASIST head (Stream A). Only Stream B + fusion train."""
    # Freeze all Wav2Vec2 layers
    for p in model.wav2vec2.parameters():
        p.requires_grad = False
    # Freeze layer fusion weights
    model.layer_weights.requires_grad = False
    # Freeze Stream A AASIST classifier
    for p in model.stream_a_classifier.parameters():
        p.requires_grad = False

    # Verify what remains trainable
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen    = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"   🔒 Frozen   : {frozen/1e6:.1f} M params  (Wav2Vec2 + AASIST Stream A)")
    print(f"   🔥 Trainable: {trainable/1e6:.2f} M params  (LFCC-Res2Net Stream B + Fusion Head)")


def run_noise_robust_finetuning():
    print("\n" + "="*65)
    print("🛡️  SWARVED AI: NOISE-ROBUST HEAD FINE-TUNING")
    print("="*65)
    print(f"📁 Cache      : {CACHE_PATH}")
    print(f"📦 Checkpoint : {CHECKPOINT_IN}")
    print(f"💾 Output     : {CHECKPOINT_OUT}")
    print(f"🔢 Epochs     : {EPOCHS}  |  Batch: {BATCH_SIZE}  |  Noise Aug: {NOISE_AUG_PROB*100:.0f}%")
    print("="*65)

    if not os.path.exists(CACHE_PATH):
        print(f"❌ Dataset cache not found at {CACHE_PATH}. Run precache_raw_audio.py first.")
        return

    if not os.path.exists(CHECKPOINT_IN):
        print(f"❌ Checkpoint not found at {CHECKPOINT_IN}.")
        return

    # ── Data splits ────────────────────────────────────────────────────────
    device = torch.device("cpu")
    full_data = torch.load(CACHE_PATH)
    total_len = len(full_data["x"])
    val_size  = int(total_len * VAL_FRAC)
    train_size = total_len - val_size

    train_idx, val_idx = torch.utils.data.random_split(
        range(total_len), [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED)
    )

    train_ds = NoiseRobustDataset(CACHE_PATH, list(train_idx), is_train=True)
    val_ds   = NoiseRobustDataset(CACHE_PATH, list(val_idx),   is_train=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"✅ Loaded {len(train_ds)} train / {len(val_ds)} val samples")

    # ── Model ─────────────────────────────────────────────────────────────
    model = SwarVedDualStreamEngine(unfreeze_layers=4).to(device)
    ckpt = torch.load(CHECKPOINT_IN, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt, strict=False)
    print(f"✅ Loaded checkpoint: {CHECKPOINT_IN}")

    freeze_stream_a(model)

    # Only optimize the unfrozen parts
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=LR_HEAD, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.03)

    os.makedirs("ml_engine/models", exist_ok=True)
    best_val_acc = 0.0
    start_t = time.time()

    for epoch in range(1, EPOCHS + 1):
        # ── Train ──────────────────────────────────────────────────────────
        model.train()
        tr_loss, tr_correct, tr_total = 0.0, 0, 0
        for step, (wav, y) in enumerate(train_loader):
            wav, y = wav.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(wav, targets=y)
            loss   = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()

            tr_loss    += loss.item() * len(y)
            preds       = logits.argmax(dim=1)
            tr_correct += (preds == y).sum().item()
            tr_total   += len(y)

            if (step + 1) % 30 == 0 or (step + 1) == len(train_loader):
                print(f"   Ep [{epoch}/{EPOCHS}] Step [{step+1}/{len(train_loader)}] "
                      f"Loss: {loss.item():.4f} | Acc: {(tr_correct/tr_total)*100:.2f}%", flush=True)

        scheduler.step()
        tr_loss /= tr_total
        tr_acc   = (tr_correct / tr_total) * 100.0

        # ── Validate ───────────────────────────────────────────────────────
        model.eval()
        vl_loss, vl_correct, vl_total = 0.0, 0, 0
        real_ok, real_tot = 0, 0
        fake_ok, fake_tot = 0, 0

        with torch.no_grad():
            for wav, y in val_loader:
                wav, y = wav.to(device), y.to(device)
                logits = model(wav, targets=None)
                loss   = criterion(logits, y)
                vl_loss    += loss.item() * len(y)
                preds       = logits.argmax(dim=1)
                vl_correct += (preds == y).sum().item()
                vl_total   += len(y)
                real_mask   = (y == 0)
                fake_mask   = (y == 1)
                real_ok    += (preds[real_mask] == 0).sum().item()
                real_tot   += real_mask.sum().item()
                fake_ok    += (preds[fake_mask] == 1).sum().item()
                fake_tot   += fake_mask.sum().item()

        vl_loss /= vl_total
        vl_acc   = (vl_correct / vl_total) * 100.0
        real_acc = (real_ok / real_tot * 100.0) if real_tot > 0 else 0.0
        fake_acc = (fake_ok / fake_tot * 100.0) if fake_tot > 0 else 0.0

        print(f"Epoch [{epoch}/{EPOCHS}] — Train: {tr_acc:.2f}%  |  "
              f"Val: {vl_acc:.2f}%  (Real: {real_acc:.2f}%, Deepfake: {fake_acc:.2f}%)")

        if vl_acc >= best_val_acc:
            best_val_acc = vl_acc
            torch.save(model.state_dict(), CHECKPOINT_OUT)
            print(f"   💾 Saved noise-robust checkpoint → {CHECKPOINT_OUT}  (Val Acc: {vl_acc:.2f}%)")

    elapsed = (time.time() - start_t) / 60.0
    print(f"\n🎉 Noise-Robust Fine-Tuning Complete in {elapsed:.1f} minutes!")
    print(f"🏆 Best Val Accuracy: {best_val_acc:.2f}%")
    print(f"📦 Model saved to   : {CHECKPOINT_OUT}")
    print("\nNext step → export to ONNX:")
    print("   python3 ml_engine/export_e2e_onnx.py  (update checkpoint path if needed)")


if __name__ == "__main__":
    run_noise_robust_finetuning()
