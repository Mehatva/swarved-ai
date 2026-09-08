import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SELayer(nn.Module):
    """Squeeze-and-Excitation Channel Attention Block"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (B, C, T)
        b, c, _ = x.shape
        w = self.fc(x).view(b, c, 1)
        return x * w

class Res2NetBlock(nn.Module):
    """Multi-Scale Res2Net Bottleneck Block for Spectro-Temporal Filtering"""
    def __init__(self, in_channels, out_channels, scale=4):
        super().__init__()
        self.scale = scale
        width = out_channels // scale
        self.width = width

        self.conv1 = nn.Conv1d(in_channels, width * scale, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(width * scale)

        convs = []
        bns = []
        for i in range(scale - 1):
            convs.append(nn.Conv1d(width, width, kernel_size=3, padding=1))
            bns.append(nn.BatchNorm1d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        self.conv3 = nn.Conv1d(width * scale, out_channels, kernel_size=1)
        self.bn3 = nn.BatchNorm1d(out_channels)
        self.se = SELayer(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.scale - 1):
            if i == 0:
                sp = spx[i]
            else:
                sp = spx[i] + sp
            sp = self.convs[i](sp)
            sp = self.relu(self.bns[i](sp))
            if i == 0:
                out_conv = sp
            else:
                out_conv = torch.cat((out_conv, sp), 1)

        out_conv = torch.cat((out_conv, spx[self.scale - 1]), 1)
        out = self.conv3(out_conv)
        out = self.bn3(out)
        out = self.se(out)

        out = out + residual
        return self.relu(out)

class Res2NetPhaseStream(nn.Module):
    """
    Stream B: Phase-Spectral Feature Stream.
    Processes 180-dim LFCC spectrograms using a multi-scale Res2Net-50 architecture
    to extract 128-dim embeddings capturing vocoder phase discontinuities.
    """
    def __init__(self, in_dim=180, d_model=256, out_dim=128):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Conv1d(in_dim, d_model // 2, kernel_size=7, padding=3, stride=1),
            nn.BatchNorm1d(d_model // 2),
            nn.ReLU(inplace=True),
            nn.Conv1d(d_model // 2, d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model),
            nn.ReLU(inplace=True)
        )

        self.layer1 = Res2NetBlock(d_model, d_model, scale=4)
        self.layer2 = Res2NetBlock(d_model, d_model, scale=4)

        # Global Statistics Pooling (Mean + Std = 512-dim)
        self.embedding_head = nn.Sequential(
            nn.Linear(d_model * 2, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, out_dim)
        )

    def forward(self, lfcc):
        # lfcc: (B, 180, T)
        x = self.input_proj(lfcc)
        x = self.layer1(x)
        x = self.layer2(x)

        mean = x.mean(dim=2)
        std = x.std(dim=2) + 1e-6
        stats = torch.cat([mean, std], dim=1) # (B, 512)

        embedding = self.embedding_head(stats) # (B, 128)
        return embedding

if __name__ == "__main__":
    stream_b = Res2NetPhaseStream()
    dummy_lfcc = torch.randn(2, 180, 301)
    emb = stream_b(dummy_lfcc)
    print(f"✅ Res2Net Stream B Test: LFCC Input {dummy_lfcc.shape} -> Phase Embedding {emb.shape}")
