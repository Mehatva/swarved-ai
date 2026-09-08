import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from train_aasist_graph import Wav2Vec2CachedDataset
from aasist_graph_classifier import SwarVedAASISTGraphClassifier
import numpy as np

def evaluate_aasist_checkpoint(checkpoint_path="ml_engine/models/aasist_best.pt", cache_path="ml_engine/data/wav2vec2_cached_dataset.pt"):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"🔍 Evaluating World-Record SSL-AASIST Checkpoint: {checkpoint_path} on device {device}")

    dataset = Wav2Vec2CachedDataset(cache_path=cache_path)
    val_size = int(len(dataset) * 0.2)
    train_size = len(dataset) - val_size

    _, val_ds = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )

    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

    model = SwarVedAASISTGraphClassifier(
        in_dim=768, d_model=256, n_heads=8, num_layers=4, num_classes=2, margin_m=0.35, scale_s=30.0
    ).to(device)

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x, targets=None)
            preds = logits.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    total = len(all_targets)
    correct = (all_preds == all_targets).sum()
    overall_acc = (correct / total) * 100.0

    real_mask = (all_targets == 0)
    fake_mask = (all_targets == 1)

    real_correct = (all_preds[real_mask] == 0).sum()
    real_acc = (real_correct / real_mask.sum()) * 100.0

    fake_correct = (all_preds[fake_mask] == 1).sum()
    fake_acc = (fake_correct / fake_mask.sum()) * 100.0

    frr = ((all_preds[real_mask] == 1).sum() / real_mask.sum()) * 100.0
    far = ((all_preds[fake_mask] == 0).sum() / fake_mask.sum()) * 100.0

    print("\n" + "="*55)
    print("🏆 EVALUATION RESULTS SUMMARY: WORLD-RECORD SSL-AASIST ENGINE")
    print("="*55)
    print(f"🎯 Overall True Validation Accuracy : {overall_acc:.2f}% ({correct}/{total})")
    print(f"🎙️  Bonafide Real Voice Accuracy   : {real_acc:.2f}% ({real_correct}/{real_mask.sum()})")
    print(f"🤖 Deepfake Voice Accuracy         : {fake_acc:.2f}% ({fake_correct}/{fake_mask.sum()})")
    print(f"⚠️  False Rejection Rate (FRR)      : {frr:.2f}%")
    print(f"⚠️  False Acceptance Rate (FAR)     : {far:.2f}%")
    print("="*55)

if __name__ == "__main__":
    evaluate_aasist_checkpoint()
