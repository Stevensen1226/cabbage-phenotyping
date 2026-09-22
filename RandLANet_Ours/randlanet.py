"""RandLA-Net 纯语义分割 (简化版)."""
import torch, torch.nn as nn, torch.nn.functional as F, numpy as np
from scipy.spatial import cKDTree


class LocalEnc(nn.Module):
    """KNN局部编码 + 注意力池化."""
    def __init__(self, d_in, d_out, k=16):
        super().__init__()
        self.k = k
        self.mlp = nn.Sequential(
            nn.Linear(d_in + 3, d_out), nn.ReLU(), nn.Linear(d_out, d_out))

    def forward(self, xyz, feat):
        B, N, D = feat.shape
        idx = torch.zeros(B, N, self.k, dtype=torch.long, device=xyz.device)
        for b in range(B):
            _, nn = cKDTree(xyz[b].cpu().numpy()).query(xyz[b].cpu().numpy(), k=self.k)
            idx[b] = torch.from_numpy(nn).long().to(xyz.device)
        bidx = torch.arange(B, device=xyz.device).view(B, 1, 1).expand(-1, N, self.k)
        nxyz = xyz[bidx, idx]
        rel = nxyz - xyz.unsqueeze(2)
        nfeat = feat[bidx, idx]
        cat = torch.cat([rel, nfeat], dim=-1)
        B2, N2, K2, C2 = cat.shape
        enc = self.mlp(cat.view(B2 * N2 * K2, C2)).view(B2, N2, K2, -1)
        return torch.max(enc, dim=2)[0]  # max pooling over neighbors


class RandLANet(nn.Module):
    def __init__(self, d_in=6, num_classes=2, d_feat=32, k=16):
        super().__init__()
        self.emb = nn.Linear(d_in, d_feat)
        self.enc1 = LocalEnc(d_feat, d_feat * 2, k)
        self.enc2 = LocalEnc(d_feat * 2, d_feat * 4, k)
        self.enc3 = LocalEnc(d_feat * 4, d_feat * 8, k)
        self.cls = nn.Sequential(
            nn.Linear(d_feat * 8, d_feat * 4), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(d_feat * 4, num_classes))

    def forward(self, xyz, feat):
        f = self.emb(torch.cat([xyz, feat], dim=-1))
        f = self.enc1(xyz, f)
        f = self.enc2(xyz, f)
        f = self.enc3(xyz, f)
        B, N, C = f.shape
        return self.cls(f.view(B * N, C)).view(B, N, -1)
