"""
ASF (Adaptive Stability-aware Fusion) module and extended model variants.
Extends Restormer and NAFNet from comparison_models with new fusion strategies.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .comparison_models import (
    NAFNet, Restormer, ShadowEncoder,
    SGCF, SGFM, SGGF, SGCA,
)


class ASF(nn.Module):
    """Adaptive Stability-aware Fusion."""
    def __init__(self, feat_ch, shadow_ch, reduction=4):
        super().__init__()
        self.shadow_proj = nn.Conv2d(shadow_ch, feat_ch, 1, bias=True)
        self.gamma_conv = nn.Conv2d(feat_ch, feat_ch, 1, bias=True)
        self.beta_conv = nn.Conv2d(feat_ch, feat_ch, 1, bias=True)
        self.gate_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(feat_ch * 2, feat_ch // reduction, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_ch // reduction, feat_ch, 1, bias=True),
            nn.Sigmoid()
        )
        self.alpha_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(feat_ch * 2, feat_ch // reduction, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_ch // reduction, 1, 1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, feat, shadow_feat):
        s = self.shadow_proj(shadow_feat)
        gamma = torch.tanh(self.gamma_conv(s))
        beta = self.beta_conv(s)
        f_mod = feat * (1 + gamma) + beta
        gate = self.gate_net(torch.cat([feat, s], dim=1))
        f_gate = feat * gate + s * (1 - gate)
        alpha = self.alpha_net(torch.cat([feat, s], dim=1))
        return alpha * f_mod + (1 - alpha) * f_gate


class ShadowGuidedNAFNet_FiLM(NAFNet):
    """NAFNet + ShadowEncoder + SGFM (FiLM modulation)."""
    def __init__(self, img_channel=3, width=16, middle_blk_num=1,
                 enc_blk_nums=None, dec_blk_nums=None):
        super().__init__(img_channel=img_channel, width=width,
                         middle_blk_num=middle_blk_num,
                         enc_blk_nums=enc_blk_nums, dec_blk_nums=dec_blk_nums)
        self.shadow_encoder = ShadowEncoder(width=width)
        self.sgfm_dec1 = SGFM(width * 4, width * 4)
        self.sgfm_dec2 = SGFM(width * 2, width * 2)

    def forward(self, gray, inp):
        B, C, H, W = inp.shape
        inp_padded = self.check_image_size(inp)
        gray_padded = F.interpolate(gray, size=inp_padded.shape[2:],
                                     mode='bilinear', align_corners=False) if gray.shape[2:] != inp_padded.shape[2:] else gray
        s1, s2, s3 = self.shadow_encoder(gray_padded)
        x = self.intro(inp_padded)
        encs = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)
        x = self.middle_blks(x)
        for i, (decoder, up, enc_skip) in enumerate(zip(self.decoders, self.ups, encs[::-1])):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)
            if i == 1:
                x = self.sgfm_dec1(x, s3)
            elif i == 2:
                x = self.sgfm_dec2(x, s2)
        x = self.ending(x)
        x = x + inp_padded
        return x[:, :, :H, :W]


class ShadowGuidedNAFNet_Gated(NAFNet):
    def __init__(self, img_channel=3, width=16, middle_blk_num=1,
                 enc_blk_nums=None, dec_blk_nums=None):
        super().__init__(img_channel=img_channel, width=width,
                         middle_blk_num=middle_blk_num,
                         enc_blk_nums=enc_blk_nums, dec_blk_nums=dec_blk_nums)
        self.shadow_encoder = ShadowEncoder(width=width)
        self.sggf_dec1 = SGGF(width * 4, width * 4)
        self.sggf_dec2 = SGGF(width * 2, width * 2)

    def forward(self, gray, inp):
        B, C, H, W = inp.shape
        inp_padded = self.check_image_size(inp)
        gray_padded = F.interpolate(gray, size=inp_padded.shape[2:],
                                     mode='bilinear', align_corners=False) if gray.shape[2:] != inp_padded.shape[2:] else gray
        s1, s2, s3 = self.shadow_encoder(gray_padded)
        x = self.intro(inp_padded)
        encs = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)
        x = self.middle_blks(x)
        for i, (decoder, up, enc_skip) in enumerate(zip(self.decoders, self.ups, encs[::-1])):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)
            if i == 1:
                x = self.sggf_dec1(x, s3)
            elif i == 2:
                x = self.sggf_dec2(x, s2)
        x = self.ending(x)
        x = x + inp_padded
        return x[:, :, :H, :W]


class ShadowGuidedNAFNet_CrossAttn(NAFNet):
    def __init__(self, img_channel=3, width=16, middle_blk_num=1,
                 enc_blk_nums=None, dec_blk_nums=None, cross_heads=4):
        super().__init__(img_channel=img_channel, width=width,
                         middle_blk_num=middle_blk_num,
                         enc_blk_nums=enc_blk_nums, dec_blk_nums=dec_blk_nums)
        self.shadow_encoder = ShadowEncoder(width=width)
        self.sgcf_dec1 = SGCF(width * 4, width * 4, num_heads=cross_heads)
        self.sgcf_dec2 = SGCF(width * 2, width * 2, num_heads=cross_heads)

    def forward(self, gray, inp):
        B, C, H, W = inp.shape
        inp_padded = self.check_image_size(inp)
        gray_padded = F.interpolate(gray, size=inp_padded.shape[2:],
                                     mode='bilinear', align_corners=False) if gray.shape[2:] != inp_padded.shape[2:] else gray
        s1, s2, s3 = self.shadow_encoder(gray_padded)
        x = self.intro(inp_padded)
        encs = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)
        x = self.middle_blks(x)
        for i, (decoder, up, enc_skip) in enumerate(zip(self.decoders, self.ups, encs[::-1])):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)
            if i == 1:
                x = self.sgcf_dec1(x, s3)
            elif i == 2:
                x = self.sgcf_dec2(x, s2)
        x = self.ending(x)
        x = x + inp_padded
        return x[:, :, :H, :W]


class ShadowGuidedNAFNet_ASF(NAFNet):
    def __init__(self, img_channel=3, width=16, middle_blk_num=1,
                 enc_blk_nums=None, dec_blk_nums=None):
        super().__init__(img_channel=img_channel, width=width,
                         middle_blk_num=middle_blk_num,
                         enc_blk_nums=enc_blk_nums, dec_blk_nums=dec_blk_nums)
        self.shadow_encoder = ShadowEncoder(width=width)
        self.asf_dec1 = ASF(width * 4, width * 4)
        self.asf_dec2 = ASF(width * 2, width * 2)

    def forward(self, gray, inp):
        B, C, H, W = inp.shape
        inp_padded = self.check_image_size(inp)
        gray_padded = F.interpolate(gray, size=inp_padded.shape[2:],
                                     mode='bilinear', align_corners=False) if gray.shape[2:] != inp_padded.shape[2:] else gray
        s1, s2, s3 = self.shadow_encoder(gray_padded)
        x = self.intro(inp_padded)
        encs = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)
        x = self.middle_blks(x)
        for i, (decoder, up, enc_skip) in enumerate(zip(self.decoders, self.ups, encs[::-1])):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)
            if i == 1:
                x = self.asf_dec1(x, s3)
            elif i == 2:
                x = self.asf_dec2(x, s2)
        x = self.ending(x)
        x = x + inp_padded
        return x[:, :, :H, :W]


class ShadowGuidedNAFNet_Large(NAFNet):
    def __init__(self, img_channel=3, width=32, middle_blk_num=1,
                 enc_blk_nums=None, dec_blk_nums=None):
        if enc_blk_nums is None:
            enc_blk_nums = [1, 1, 1, 8]
        if dec_blk_nums is None:
            dec_blk_nums = [1, 1, 1, 1]
        super().__init__(img_channel=img_channel, width=width,
                         middle_blk_num=middle_blk_num,
                         enc_blk_nums=enc_blk_nums, dec_blk_nums=dec_blk_nums)
        self.shadow_encoder = ShadowEncoder(width=width)
        self.fuse_dec1 = nn.Conv2d(width * 4 + width * 4, width * 4, 1, bias=True)
        self.fuse_dec2 = nn.Conv2d(width * 2 + width * 2, width * 2, 1, bias=True)

    def forward(self, gray, inp):
        B, C, H, W = inp.shape
        inp_padded = self.check_image_size(inp)
        gray_padded = F.interpolate(gray, size=inp_padded.shape[2:],
                                     mode='bilinear', align_corners=False) if gray.shape[2:] != inp_padded.shape[2:] else gray
        s1, s2, s3 = self.shadow_encoder(gray_padded)
        x = self.intro(inp_padded)
        encs = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)
        x = self.middle_blks(x)
        for i, (decoder, up, enc_skip) in enumerate(zip(self.decoders, self.ups, encs[::-1])):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)
            if i == 1:
                x = self.fuse_dec1(torch.cat([x, s3], dim=1))
            elif i == 2:
                x = self.fuse_dec2(torch.cat([x, s2], dim=1))
        x = self.ending(x)
        x = x + inp_padded
        return x[:, :, :H, :W]


class ShadowGuidedRestormer_ASF(Restormer):
    def __init__(self, inp_channels=3, out_channels=3, dim=48,
                 num_blocks=None, num_refinement_blocks=4,
                 heads=None, ffn_expansion_factor=2.66,
                 bias=False, LayerNorm_type='WithBias'):
        super().__init__(inp_channels=inp_channels, out_channels=out_channels, dim=dim,
                         num_blocks=num_blocks, num_refinement_blocks=num_refinement_blocks,
                         heads=heads, ffn_expansion_factor=ffn_expansion_factor,
                         bias=bias, LayerNorm_type=LayerNorm_type)
        self.shadow_encoder = ShadowEncoder(width=dim)
        self.asf_dec3 = ASF(dim * 4, dim * 4)
        self.asf_dec2 = ASF(dim * 2, dim * 2)

    def forward(self, gray, inp):
        B, C, H, W = inp.shape
        if gray.shape[2:] != (H, W):
            gray = F.interpolate(gray, size=(H, W), mode='bilinear', align_corners=False)
        s1, s2, s3 = self.shadow_encoder(gray)
        inp_enc_level1 = self.patch_embed(inp)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)
        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)
        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)
        inp_enc_level4 = self.down3_4(out_enc_level3)
        latent = self.latent(inp_enc_level4)
        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)
        out_dec_level3 = self.asf_dec3(out_dec_level3, s3)
        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)
        out_dec_level2 = self.asf_dec2(out_dec_level2, s2)
        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)
        out_dec_level1 = self.refinement(out_dec_level1)
        out_dec_level1 = self.output(out_dec_level1) + inp
        return out_dec_level1


__all__ = [
    'ASF',
    'ShadowGuidedNAFNet_FiLM',
    'ShadowGuidedNAFNet_Gated',
    'ShadowGuidedNAFNet_CrossAttn',
    'ShadowGuidedNAFNet_ASF',
    'ShadowGuidedNAFNet_Large',
    'ShadowGuidedRestormer_ASF',
]
