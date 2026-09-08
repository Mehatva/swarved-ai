import torch
import torch.nn as nn
import torch.nn.functional as F

class SwarVedConformerV2(nn.Module):
    """
    Literature-grade Conformer Architecture for Audio Anti-Spoofing.
    Input: (B, 3, 60, T) representing 3-channel LFCC (Static, Delta, Delta-Delta)
    Returns: Unnormalized Logits (B, 2) during training for CrossEntropyLoss.
    """
    def __init__(self, in_channels=3, n_lfcc=60, d_model=128, n_heads=4, num_layers=4, num_classes=2):
        super().__init__()
        # 2D Conv Stem for spatial-spectral downsampling
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1))
        self.bn3 = nn.BatchNorm2d(128)

        # Downsampled Frequency Dim: 60 // 8 = 8 (handling odd padding: 8 bins)
        freq_out = 8
        self.proj = nn.Linear(128 * freq_out, d_model)

        # Transformer Encoder Blocks
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Depthwise Separable Conv Refinement Module
        self.depthwise_conv = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=17, padding=8, groups=d_model),
            nn.BatchNorm1d(d_model),
            nn.SiLU(),
            nn.Conv1d(d_model, d_model, kernel_size=1),
            nn.Dropout(0.1)
        )

        # Global Statistics Pooling & Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(d_model * 2, 64),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x, return_probabilities=False):
        if x.dim() == 3:
            x = x.unsqueeze(1)
            if x.shape[1] == 1:
                x = x.repeat(1, 3, 1, 1)

        out = F.silu(self.bn1(self.conv1(x))) # (B, 32, 30, T/2)
        out = F.silu(self.bn2(self.conv2(out))) # (B, 64, 15, T/4)
        out = F.silu(self.bn3(self.conv3(out))) # (B, 128, 8, T/4)

        B, C, F_dim, T_dim = out.shape
        out = out.permute(0, 3, 1, 2).reshape(B, T_dim, C * F_dim) # (B, T_dim, 128*8)
        out = self.proj(out) # (B, T_dim, d_model)

        # Transformer Encoder
        out_enc = self.encoder(out)

        # Depthwise Conv Refinement
        out_conv = self.depthwise_conv(out_enc.transpose(1, 2)).transpose(1, 2)
        out = out_enc + out_conv # Residual connection

        # Global Statistics Pooling (Mean + Std across time)
        mean = out.mean(dim=1)
        std = out.std(dim=1) + 1e-6
        stats = torch.cat([mean, std], dim=1) # (B, d_model * 2)

        logits = self.classifier(stats)
        if return_probabilities:
            return F.softmax(logits, dim=-1)
        return logits

# Alias for backward compatibility
SwarVedConformer = SwarVedConformerV2

if __name__ == "__main__":
    model = SwarVedConformerV2()
    dummy_input = torch.randn(2, 3, 60, 301)
    logits = model(dummy_input)
    print(f"✅ Fast SwarVedConformerV2 Model Test Passed! Logits Shape: {logits.shape}")
