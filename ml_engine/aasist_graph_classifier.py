import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class AMSoftmaxLoss(nn.Module):
    """
    Additive Margin Softmax Loss (AM-Softmax)
    Forces explicit angular margin 'm' between Real Voice (Class 0) and Deepfake Voice (Class 1).
    """
    def __init__(self, in_features, num_classes=2, s=30.0, m=0.35):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, features, targets=None):
        # features: (B, D)
        # Normalize weights and features to unit sphere
        w_norm = F.normalize(self.weight, p=2, dim=1)
        x_norm = F.normalize(features, p=2, dim=1)

        # Cosine similarity matrix: cos(theta)
        cosine = F.linear(x_norm, w_norm) # (B, num_classes)

        if targets is None or not self.training:
            # During evaluation, return scaled cosine logits
            return cosine * self.s

        # Apply Additive Margin to target class logits
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, targets.view(-1, 1).long(), 1.0)

        # cos(theta) - m for correct class
        output = self.s * (cosine - one_hot * self.m)
        return output

class GraphAttentionLayer(nn.Module):
    """
    Heterogeneous Graph Attention Layer (GAT) for Spectro-Temporal Feature Graphs.
    """
    def __init__(self, in_dim=256, out_dim=256, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads

        self.w_q = nn.Linear(in_dim, out_dim, bias=False)
        self.w_k = nn.Linear(in_dim, out_dim, bias=False)
        self.w_v = nn.Linear(in_dim, out_dim, bias=False)

        self.attn_drop = nn.Dropout(0.1)
        self.proj = nn.Linear(out_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x):
        # x: (B, T, D)
        B, T, D = x.shape
        Q = self.w_q(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.w_k(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.w_v(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(scores, dim=-1)
        attn = self.attn_drop(attn)

        context = torch.matmul(attn, V).transpose(1, 2).contiguous().view(B, T, D)
        out = self.norm(x + self.proj(context))
        return out

class SwarVedAASISTGraphClassifier(nn.Module):
    """
    World-Record Classifying Engine:
    Combines 768-dim Wav2Vec2/WavLM representations, Transformer Conformer encoder,
    Graph Attention (HS-GAT), Global Statistics Pooling, and Unified AM-Softmax Margin Loss.
    """
    def __init__(self, in_dim=768, d_model=256, n_heads=8, num_layers=4, num_classes=2, margin_m=0.35, scale_s=30.0):
        super().__init__()
        self.input_proj = nn.Sequential(
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
        self.conformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.depthwise_conv = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=17, padding=8, groups=d_model),
            nn.BatchNorm1d(d_model),
            nn.SiLU(),
            nn.Conv1d(d_model, d_model, kernel_size=1),
            nn.Dropout(0.1)
        )

        self.gat = GraphAttentionLayer(in_dim=d_model, out_dim=d_model, num_heads=4)

        self.embedding_head = nn.Sequential(
            nn.Linear(d_model * 2, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128)
        )

        self.am_softmax = AMSoftmaxLoss(in_features=128, num_classes=num_classes, s=scale_s, m=margin_m)

    def forward(self, x, targets=None, use_am_softmax=True, return_embeddings=False):
        # x: (B, T, 768)
        out = self.input_proj(x)
        out_enc = self.conformer(out)
        out_conv = self.depthwise_conv(out_enc.transpose(1, 2)).transpose(1, 2)
        out_combined = out_enc + out_conv

        out_graph = self.gat(out_combined)

        mean = out_graph.mean(dim=1)
        std = out_graph.std(dim=1) + 1e-6
        stats = torch.cat([mean, std], dim=1) # (B, 512)

        embeddings = self.embedding_head(stats) # (B, 128)

        if return_embeddings:
            return embeddings

        return self.am_softmax(embeddings, targets if self.training else None)
