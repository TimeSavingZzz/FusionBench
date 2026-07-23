"""Abstract U-Net interface for cross-paradigm fusion comparison.

All three backbones (CNN, Transformer, SSM) share the identical U-Net skeleton:
    Encoder: L1(48, H) -> L2(96, H/2) -> L3(192, H/4) -> Bottleneck(384, H/8)
    Decoder: L3(192, H/4) -> L2(96, H/2) -> L1(96, H) -> Refinement
    Skip connections at every level via concatenation + 1x1 conv reduction.

The ONLY variable is the block type filling each level:
    CNN: ResBlock (Conv+ReLU, no attention)
    Transformer: MDTA + GDFN (self-attention on skip concat)
    SSM: VSSBlock (SS2D 4-scan + CAB)

This ensures any observed difference in fusion strategy effectiveness
is attributable to the block's built-in cross-modal interaction capability.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class UNetSkeleton(nn.Module):
    """Shared U-Net skeleton. Subclasses override _make_block() to supply their block type."""

    def __init__(self, dim=48, num_blocks=(4, 6, 6, 8), num_refinement=4,
                 inp_channels=3, out_channels=3, bias=False):
        super().__init__()
        self.dim = dim
        self.num_blocks = num_blocks

        # Shallow feature extraction
        self.patch_embed = nn.Conv2d(inp_channels, dim, 3, 1, 1, bias=bias)

        # Encoder
        self.encoder_level1 = self._make_level(dim, num_blocks[0])
        self.down1_2 = self._make_down(dim, dim * 2, bias)
        self.encoder_level2 = self._make_level(dim * 2, num_blocks[1])
        self.down2_3 = self._make_down(dim * 2, dim * 4, bias)
        self.encoder_level3 = self._make_level(dim * 4, num_blocks[2])
        self.down3_4 = self._make_down(dim * 4, dim * 8, bias)
        self.bottleneck = self._make_level(dim * 8, num_blocks[3])

        # Decoder
        self.up4_3 = self._make_up(dim * 8, dim * 4, bias)
        self.reduce_chan_level3 = nn.Conv2d(dim * 8, dim * 4, 1, bias=bias)
        self.decoder_level3 = self._make_level(dim * 4, num_blocks[2])
        self.up3_2 = self._make_up(dim * 4, dim * 2, bias)
        self.reduce_chan_level2 = nn.Conv2d(dim * 4, dim * 2, 1, bias=bias)
        self.decoder_level2 = self._make_level(dim * 2, num_blocks[1])
        self.up2_1 = self._make_up(dim * 2, dim, bias)
        # D1 operates at dim*2 because skip concat 48+48=96, no reduce_chan
        self.decoder_level1 = self._make_level(dim * 2, num_blocks[0])
        self.refinement = self._make_level(dim * 2, num_refinement)

        # Output
        self.output = nn.Conv2d(dim * 2, out_channels, 3, 1, 1, bias=bias)

    def _make_down(self, in_ch, out_ch, bias):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=bias),
            nn.PixelUnshuffle(2),
        )

    def _make_up(self, in_ch, out_ch, bias):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch * 4, 3, 1, 1, bias=bias),
            nn.PixelShuffle(2),
        )

    def _make_level(self, dim, num_blocks):
        """Override in subclass to supply architecture-specific blocks."""
        raise NotImplementedError

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (64 - h % 64) % 64
        mod_pad_w = (64 - w % 64) % 64
        if mod_pad_h > 0 or mod_pad_w > 0:
            x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def _forward_sequence(self, x, return_encoder=False):
        """Core U-Net forward. Returns decoder outputs.
        If return_encoder=True, also returns encoder outputs for analysis."""
        B, C, H, W = x.shape

        enc1 = self.encoder_level1(x.flatten(2).transpose(1, 2))
        enc1_out = enc1.transpose(1, 2).view(B, -1, H, W)

        enc2_in = self.down1_2(enc1_out)
        _, _, H2, W2 = enc2_in.shape
        enc2 = self.encoder_level2(enc2_in.flatten(2).transpose(1, 2))
        enc2_out = enc2.transpose(1, 2).view(B, -1, H2, W2)

        enc3_in = self.down2_3(enc2_out)
        _, _, H3, W3 = enc3_in.shape
        enc3 = self.encoder_level3(enc3_in.flatten(2).transpose(1, 2))
        enc3_out = enc3.transpose(1, 2).view(B, -1, H3, W3)

        enc4_in = self.down3_4(enc3_out)
        _, _, H4, W4 = enc4_in.shape
        latent = self.bottleneck(enc4_in.flatten(2).transpose(1, 2))
        latent_out = latent.transpose(1, 2).view(B, -1, H4, W4)

        dec3_up = self.up4_3(latent_out)
        dec3_cat = torch.cat([dec3_up, enc3_out], dim=1)
        dec3_in = self.reduce_chan_level3(dec3_cat)
        dec3 = self.decoder_level3(dec3_in.flatten(2).transpose(1, 2))
        dec3_out = dec3.transpose(1, 2).view(B, -1, H3, W3)

        dec2_up = self.up3_2(dec3_out)
        dec2_cat = torch.cat([dec2_up, enc2_out], dim=1)
        dec2_in = self.reduce_chan_level2(dec2_cat)
        dec2 = self.decoder_level2(dec2_in.flatten(2).transpose(1, 2))
        dec2_out = dec2.transpose(1, 2).view(B, -1, H2, W2)

        dec1_up = self.up2_1(dec2_out)
        dec1_cat = torch.cat([dec1_up, enc1_out], 1)
        dec1 = self.decoder_level1(dec1_cat.flatten(2).transpose(1, 2))
        dec1_out = dec1.transpose(1, 2).view(B, -1, H, W)

        if return_encoder:
            return dec3_out, dec2_out, dec1_out, (enc1_out, enc2_out, enc3_out)
        return dec3_out, dec2_out, dec1_out

    def forward(self, inp):
        inp = self.check_image_size(inp)
        x = self.patch_embed(inp)
        _, _, dec1_out = self._forward_sequence(x)
        refined = self.refinement(dec1_out.flatten(2).transpose(1, 2))
        refined = refined.transpose(1, 2).view(refined.shape[0], -1, dec1_out.shape[-2], dec1_out.shape[-1])
        return self.output(refined) + inp
