"""CNN U-Net: ResBlock-based, NO self-attention on skip connections.

Prediction: fusion modules should provide significant gains because the backbone
has no built-in mechanism for cross-modal interaction between encoder and decoder features.
"""
import math
import torch.nn as nn
from .base import UNetSkeleton


class ResBlock(nn.Module):
    """Standard 2-conv residual block. Same hidden dim as MDTA/GDFN for fair comparison."""
    def __init__(self, dim, expansion_factor=2.66, bias=False):
        super().__init__()
        hidden = int(dim * expansion_factor)
        self.norm1 = nn.LayerNorm(dim)
        self.conv1 = nn.Conv2d(dim, hidden, 3, 1, 1, bias=bias)
        self.act = nn.GELU()
        self.norm2 = nn.LayerNorm(hidden)
        self.conv2 = nn.Conv2d(hidden, dim, 3, 1, 1, bias=bias)

    def forward(self, x, x_size=None):
        B, N, C = x.shape
        H = W = int(math.sqrt(N))
        shortcut = x
        x = self.norm1(x).transpose(1, 2).view(B, C, H, W)
        x = self.act(self.conv1(x))
        x = x.flatten(2).transpose(1, 2)
        x = self.norm2(x).transpose(1, 2).view(B, -1, H, W)
        x = self.conv2(x).flatten(2).transpose(1, 2)
        return x + shortcut


class CNNUNet(UNetSkeleton):
    """CNN U-Net: identical skeleton to Restormer but ResBlock instead of MDTA+GDFN."""
    def _make_level(self, dim, num_blocks):
        return nn.Sequential(*[ResBlock(dim) for _ in range(num_blocks)])
