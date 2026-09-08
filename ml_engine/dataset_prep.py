import os
import torch
from torch.utils.data import Dataset, DataLoader

class SwarVedCachedDataset(Dataset):
    def __init__(self, cache_path="ml_engine/data/cached_dataset.pt"):
        data = torch.load(cache_path)
        self.x = data["x"]
        self.y = data["y"]
        print(f"⚡ Loaded {len(self.x)} Pre-Cached Feature Tensors from RAM ({cache_path})")

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

def get_dataloaders(cache_path="ml_engine/data/cached_dataset.pt", batch_size=64, val_split=0.2):
    dataset = SwarVedCachedDataset(cache_path=cache_path)
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size

    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=True)

    return train_loader, val_loader

if __name__ == "__main__":
    train_loader, val_loader = get_dataloaders()
    for batch_x, batch_y in train_loader:
        print(f"✅ Fast Cached DataLoader Test Passed! Shape: {batch_x.shape}")
        break
