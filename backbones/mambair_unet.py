"""MambaIR U-Net (SSM-based): VSSBlock with 4-scan SS2D + Channel Attention.

Prediction: SSM provides sequential token mixing (more than CNN, less than
self-attention). Fusion modules should show intermediate benefit — more than
Restormer (where fusion is redundant) but less than CNN (where fusion is essential).

Uses a pure-PyTorch 4-directional scan SS2D to avoid mamba-ssm dependency.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import UNetSkeleton


# ==============================================================================
# Selective Scan 2D — Pure PyTorch (4 directions, no mamba-ssm dependency)
# ==============================================================================
class SS2D(nn.Module):
    """Simplified 2D Selective Scan using 4 Conv1d directions.

    The original Mamba SS2D uses a hardware-optimized selective_scan CUDA kernel.
    This pure-PyTorch version uses 4 Conv1d passes (L->R, R->L, T->B, B->T)
    followed by summation. It preserves the key property: sequential token mixing
    via causal 1D convolutions — more expressive than static convolutions but
    less expressive than full pairwise self-attention.
    """
    def __init__(self, dim, d_state=16, bias=False):
        super().__init__()
        self.dim = dim
        self.d_state = d_state

        # Input projection: x -> (x, z)
        self.in_proj_x = nn.Linear(dim, dim * 2, bias=bias)
        self.in_proj_z = nn.Linear(dim, dim * 2, bias=bias)

        # 4 scan directions as 1D depthwise convolutions
        self.scan_convs = nn.ModuleList([
            nn.Conv1d(dim, dim, kernel_size=4, padding=3, groups=dim, bias=bias)
            for _ in range(4)
        ])

        # Output projection
        self.out_proj = nn.Linear(dim, dim, bias=bias)

    def forward(self, x):
        """x: (B, N, C)"""
        B, N, C = x.shape
        H = W = int(N ** 0.5)

        # Input projections
        x_proj = self.in_proj_x(x)
        z_proj = self.in_proj_z(x)
        x1, x2 = x_proj.chunk(2, dim=-1)
        z1, z2 = z_proj.chunk(2, dim=-1)
        x_in = x1 * F.silu(z1)  # gating
        x_skip = x2 * F.silu(z2)

        # Reshape to 2D
        x_2d = x_in.view(B, H, W, C).permute(0, 3, 1, 2)  # (B, C, H, W)

        # 4-directional scans via Conv1d
        scans = []
        # Direction 1: Left → Right (flatten H*W as row-major)
        scan1 = x_2d.flatten(2)  # (B, C, H*W)
        scans.append(self.scan_convs[0](scan1))

        # Direction 2: Right → Left
        scan2 = x_2d.flip(-1).flatten(2)
        scans.append(self.scan_convs[1](scan2).flip(-1))

        # Direction 3: Top → Bottom (transpose H<->W)
        scan3 = x_2d.transpose(2, 3).flatten(2)
        scans.append(self.scan_convs[2](scan3).view(B, C, W, H).transpose(2, 3).flatten(2))

        # Direction 4: Bottom → Top
        x_flip = x_2d.flip(-2)
        scan4 = x_flip.transpose(2, 3).flatten(2)
        scans.append(self.scan_convs[3](scan4).view(B, C, W, H).flip(-2).transpose(2, 3).flatten(2))

        # Sum all scan directions
        out = sum(scans) / 4.0  # (B, C, H*W)
        out = out.transpose(1, 2)  # (B, H*W, C)

        # Gating + residual
        out = out * x_skip + x
        return self.out_proj(out)


# ==============================================================================
# Channel Attention Block (from MambaIR)
# ==============================================================================
class CAB(nn.Module):
    """Channel Attention Block: SE-like with GELU activation."""
    def __init__(self, dim, reduction=4, bias=False):
        super().__init__()
        hidden = max(1, dim // reduction)
        self.body = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, hidden, 1, bias=bias),
            nn.GELU(),
            nn.Conv2d(hidden, dim, 1, bias=bias),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """x: (B, C, H, W)"""
        return x * self.body(x)


# ==============================================================================
# VSSBlock — Vision State Space Block (from MambaIR v1, used in MambaIRUNet)
# ==============================================================================
class VSSBlock(nn.Module):
    """SS2D + CAB with residual connections."""
    def __init__(self, dim, d_state=16, bias=False):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.ss2d = SS2D(dim, d_state, bias)
        self.norm2 = nn.LayerNorm(dim)
        self.cab = CAB(dim)
        self.skip_scale = nn.Parameter(torch.ones(1))
        self.skip_scale2 = nn.Parameter(torch.ones(1))

    def forward(self, x, x_size=None):
        """x: (B, N, C)"""
        B, N, C = x.shape
        H = W = int(N ** 0.5)
        shortcut = x
        x = self.norm1(x)
        x = self.ss2d(x)
        x = x * self.skip_scale + shortcut

        shortcut2 = x
        x = self.norm2(x)
        x_2d = x.transpose(1, 2).view(B, C, H, W)
        x_2d = self.cab(x_2d)
        x = x_2d.flatten(2).transpose(1, 2)
        x = x * self.skip_scale2 + shortcut2
        return x


# ==============================================================================
# MambaIRUNetAdapter — MambaIR U-Net using UNetSkeleton
# ==============================================================================
class MambaIRUNetAdapter(UNetSkeleton):
    """MambaIR U-Net: SSM-based VSSBlocks on the standard UNetSkeleton."""
    def _make_level(self, dim, num_blocks):
        return nn.Sequential(*[VSSBlock(dim) for _ in range(num_blocks)])
