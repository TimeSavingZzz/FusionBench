"""Five fusion strategies for skip-connection feature fusion."""

import torch
import torch.nn as nn


# ==============================================================================
# (a) Concat — simple concatenation + 1x1 conv
# ==============================================================================
class ConcatFusion(nn.Module):
    def __init__(self, feat_ch, shadow_ch, bias=False):
        super().__init__()
        self.fuse = nn.Conv2d(feat_ch + shadow_ch, feat_ch, 1, bias=bias)

    def forward(self, decoder_feat, shadow_feat):
        return self.fuse(torch.cat([decoder_feat, shadow_feat], dim=1))


# ==============================================================================
# (b) Cross-Attention — Q=decoder, K/V=shadow, zero-init gamma residual
# ==============================================================================
class CrossAttnFusion(nn.Module):
    def __init__(self, feat_ch, shadow_ch, num_heads=4, bias=False):
        super().__init__()
        self.shadow_proj = nn.Conv2d(shadow_ch, feat_ch, 1, bias=bias)
        self.norm_q = nn.LayerNorm(feat_ch)
        self.norm_kv = nn.LayerNorm(feat_ch)
        self.cross_attn = nn.MultiheadAttention(feat_ch, num_heads, batch_first=True)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, decoder_feat, shadow_feat):
        B, C, H, W = decoder_feat.shape
        shadow = self.shadow_proj(shadow_feat)
        # Flatten to (B, HW, C)
        q = self.norm_q(decoder_feat.flatten(2).transpose(1, 2))
        kv = self.norm_kv(shadow.flatten(2).transpose(1, 2))
        out, _ = self.cross_attn(q, kv, kv)
        out = out.transpose(1, 2).view(B, C, H, W)
        return decoder_feat + out * self.gamma


# ==============================================================================
# (c) FiLM — channel-wise modulation: feat * (1+tanh(gamma)) + beta
# ==============================================================================
class FiLMFusion(nn.Module):
    def __init__(self, feat_ch, shadow_ch, bias=False):
        super().__init__()
        self.shadow_proj = nn.Conv2d(shadow_ch, feat_ch, 1, bias=bias)
        self.gamma_conv = nn.Conv2d(feat_ch, feat_ch, 1, bias=bias)
        self.beta_conv = nn.Conv2d(feat_ch, feat_ch, 1, bias=bias)

    def forward(self, decoder_feat, shadow_feat):
        s = self.shadow_proj(shadow_feat)
        gamma = torch.tanh(self.gamma_conv(s))
        beta = self.beta_conv(s)
        return decoder_feat * (1 + gamma) + beta


# ==============================================================================
# (d) Gated — sigmoid-gated convex combination: F*g + S*(1-g)
# ==============================================================================
class GatedFusion(nn.Module):
    def __init__(self, feat_ch, shadow_ch, reduction=4, bias=False):
        super().__init__()
        self.shadow_proj = nn.Conv2d(shadow_ch, feat_ch, 1, bias=bias)
        hidden = max(1, feat_ch // reduction)
        self.gate_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(feat_ch * 2, hidden, 1, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, feat_ch, 1, bias=bias),
            nn.Sigmoid(),
        )

    def forward(self, decoder_feat, shadow_feat):
        s = self.shadow_proj(shadow_feat)
        gate = self.gate_net(torch.cat([decoder_feat, s], dim=1))
        return decoder_feat * gate + s * (1 - gate)


# ==============================================================================
# (e) Large — Concat with wider backbone dim
# ==============================================================================
# Large is implemented as a model variant with dim=64, not a separate module.
# The fusion mechanism itself is identical to ConcatFusion.


# ==============================================================================
# No Fusion (baseline)
# ==============================================================================
class NoFusion(nn.Module):
    def forward(self, decoder_feat, shadow_feat):
        return decoder_feat


# ==============================================================================
# Fusion factory
# ==============================================================================
def build_fusion(name, feat_ch, shadow_ch, **kwargs):
    registry = {
        'none': NoFusion,
        'concat': ConcatFusion,
        'cross_attn': CrossAttnFusion,
        'film': FiLMFusion,
        'gated': GatedFusion,
    }
    if name not in registry:
        raise ValueError(f"Unknown fusion: {name}. Choose from {list(registry.keys())}")
    return registry[name](feat_ch, shadow_ch, **kwargs)
