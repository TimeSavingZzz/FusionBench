"""Restormer U-Net: MDTA self-attention + GDFN on skip connections.

Prediction: self-attention on skip-connected encoder+decoder features provides
implicit cross-modal alignment, making dedicated fusion modules largely redundant.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import UNetSkeleton


class MDTA(nn.Module):
    """Multi-Dconv Head Transposed Attention (from Restormer, CVPR 2022)."""
    def __init__(self, dim, num_heads=4, bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, 3, 1, 1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, 1, bias=bias)

    def forward(self, x, x_size=None):
        B, N, C = x.shape
        H = W = int(N ** 0.5)
        x = x.transpose(1, 2).view(B, C, H, W)
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        q = q.view(B, self.num_heads, C // self.num_heads, N)
        k = k.view(B, self.num_heads, C // self.num_heads, N)
        v = v.view(B, self.num_heads, C // self.num_heads, N)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v).view(B, C, H, W)
        return (self.project_out(out).flatten(2).transpose(1, 2) + x.flatten(2).transpose(1, 2))


class GDFN(nn.Module):
    """Gated-Dconv Feed-Forward Network (from Restormer)."""
    def __init__(self, dim, ffn_expansion_factor=2.66, bias=False):
        super().__init__()
        hidden = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden * 2, 1, bias=bias)
        self.dwconv = nn.Conv2d(hidden * 2, hidden * 2, 3, 1, 1, groups=hidden * 2, bias=bias)
        self.project_out = nn.Conv2d(hidden, dim, 1, bias=bias)

    def forward(self, x, x_size=None):
        B, N, C = x.shape
        H = W = int(N ** 0.5)
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        return (self.project_out(x).flatten(2).transpose(1, 2))


class TransformerBlock(nn.Module):
    """MDTA + GDFN block."""
    def __init__(self, dim, num_heads=4, ffn_expansion_factor=2.66, bias=False):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MDTA(dim, num_heads, bias)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = GDFN(dim, ffn_expansion_factor, bias)

    def forward(self, x, x_size=None):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class RestormerUNet(UNetSkeleton):
    """Restormer U-Net: identical skeleton to CNN U-Net but TransformerBlock blocks."""
    def _make_level(self, dim, num_blocks):
        num_heads = max(1, dim // 48)
        return nn.Sequential(*[TransformerBlock(dim, num_heads) for _ in range(num_blocks)])
