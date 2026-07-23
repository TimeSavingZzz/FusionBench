"""Shared ShadowEncoder for all backbones.

Extracts multi-scale shadow features from grayscale shadow map.
Output channels: dim (H), dim*2 (H/2), dim*4 (H/4).
s2 feeds D2 fusion, s3 feeds D3 fusion.
"""
import torch.nn as nn


class ShadowEncoder(nn.Module):
    def __init__(self, dim=48):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, dim, 3, 1, 1),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(dim, dim * 2, 3, 2, 1),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(dim * 2, dim * 4, 3, 2, 1),
            nn.ReLU(inplace=True),
        )

    def forward(self, gray):
        """gray: [B, 1, H, W] -> s1, s2, s3 at H, H/2, H/4 resolutions."""
        s1 = self.conv1(gray)
        s2 = self.conv2(s1)
        s3 = self.conv3(s2)
        return s1, s2, s3
