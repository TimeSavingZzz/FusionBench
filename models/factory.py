"""Model factory: assemble backbone + shadow encoder + fusion strategies."""

import torch
import torch.nn as nn
from backbones.cnn_unet import CNNUNet
from backbones.restormer_unet import RestormerUNet
from fusion.shadow_encoder import ShadowEncoder
from fusion.modules import build_fusion


class FusionModel(nn.Module):
    """Wrapper: any U-Net backbone + shadow encoder + fusion at D2 & D3.

    The forward method replicates the U-Net encoder-decoder path explicitly
    so that fusion modules can be inserted at D3 and D2. This avoids the
    need for "interruptible" U-Net abstractions.
    """

    def __init__(self, backbone, shadow_encoder, fusion_d3, fusion_d2):
        super().__init__()
        self.backbone = backbone
        self.shadow_encoder = shadow_encoder
        self.fusion_d3 = fusion_d3
        self.fusion_d2 = fusion_d2

    def forward(self, gray, inp):
        gray = gray[:, :1, :, :] if gray.shape[1] > 1 else gray
        inp = self.backbone.check_image_size(inp)
        gray = self.backbone.check_image_size(gray)
        return self._full_forward(gray, inp)

    def _full_forward(self, gray, inp):
        B, C, H, W = inp.shape
        s1, s2, s3 = self.shadow_encoder(gray)

        x = self.backbone.patch_embed(inp)

        # --- Encoder ---
        enc1 = self.backbone.encoder_level1(x.flatten(2).transpose(1, 2))
        enc1_out = enc1.transpose(1, 2).view(B, -1, H, W)

        enc2_in = self.backbone.down1_2(enc1_out)
        _, _, H2, W2 = enc2_in.shape
        enc2 = self.backbone.encoder_level2(enc2_in.flatten(2).transpose(1, 2))
        enc2_out = enc2.transpose(1, 2).view(B, -1, H2, W2)

        enc3_in = self.backbone.down2_3(enc2_out)
        _, _, H3, W3 = enc3_in.shape
        enc3 = self.backbone.encoder_level3(enc3_in.flatten(2).transpose(1, 2))
        enc3_out = enc3.transpose(1, 2).view(B, -1, H3, W3)

        enc4_in = self.backbone.down3_4(enc3_out)
        _, _, H4, W4 = enc4_in.shape
        latent = self.backbone.bottleneck(enc4_in.flatten(2).transpose(1, 2))
        latent_out = latent.transpose(1, 2).view(B, -1, H4, W4)

        # --- Decoder with fusion ---
        dec3_up = self.backbone.up4_3(latent_out)
        dec3_cat = torch.cat([dec3_up, enc3_out], 1)
        dec3 = self.backbone.decoder_level3(
            self.backbone.reduce_chan_level3(dec3_cat).flatten(2).transpose(1, 2))
        dec3_out = dec3.transpose(1, 2).view(B, -1, H3, W3)
        dec3_out = self.fusion_d3(dec3_out, s3)       # *** FUSION D3 ***

        dec2_up = self.backbone.up3_2(dec3_out)
        dec2_cat = torch.cat([dec2_up, enc2_out], 1)
        dec2 = self.backbone.decoder_level2(
            self.backbone.reduce_chan_level2(dec2_cat).flatten(2).transpose(1, 2))
        dec2_out = dec2.transpose(1, 2).view(B, -1, H2, W2)
        dec2_out = self.fusion_d2(dec2_out, s2)       # *** FUSION D2 ***

        dec1_up = self.backbone.up2_1(dec2_out)
        dec1_cat = torch.cat([dec1_up, enc1_out], 1)
        dec1 = self.backbone.decoder_level1(dec1_cat.flatten(2).transpose(1, 2))
        dec1_out = dec1.transpose(1, 2).view(B, -1, H, W)

        # --- Refinement + Output ---
        refined = self.backbone.refinement(dec1_out.flatten(2).transpose(1, 2))
        refined = refined.transpose(1, 2).view(refined.shape[0], -1, H, W)
        return self.backbone.output(refined) + inp


def build_model(backbone_name, fusion_name, dim=48, **kwargs):
    """Build a complete model from backbone + fusion strategy.

    Args:
        backbone_name: 'cnn', 'restormer', 'mamba'
        fusion_name: 'none', 'concat', 'cross_attn', 'film', 'gated', 'large'
        dim: base channel dimension

    Returns:
        FusionModel
    """
    # 'large' means Concat fusion with dim=64
    actual_fusion = fusion_name
    if fusion_name == 'large':
        dim = 64
        actual_fusion = 'concat'

    # Backbone
    if backbone_name == 'cnn':
        backbone = CNNUNet(dim=dim)
    elif backbone_name == 'restormer':
        backbone = RestormerUNet(dim=dim)
    elif backbone_name == 'mamba':
        from backbones.mambair_unet import MambaIRUNetAdapter
        backbone = MambaIRUNetAdapter(dim=dim)
    else:
        raise ValueError(f"Unknown backbone: {backbone_name}")

    # Shadow encoder (dim must match backbone)
    shadow_enc = ShadowEncoder(dim=dim)

    # Fusion modules
    fusion_d3 = build_fusion(actual_fusion, dim * 4, dim * 4, **kwargs)
    fusion_d2 = build_fusion(actual_fusion, dim * 2, dim * 2, **kwargs)

    return FusionModel(backbone, shadow_enc, fusion_d3, fusion_d2)
