"""
对比实验模型集合: BEDSR-Generator, UNet, NAFNet, Restormer
模型签名统一: forward(self, x) -> x: [B,3,H,W] shadow -> [B,3,H,W] restored
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numbers


# ============================================================
# BEDSR-Generator: 基于 BEDSR-Net 的 U-Net 编码器-解码器
# 原文: BEDSR-Net: A Deep Shadow Removal Network from a Single Document Image (CVPR 2020)
# 简化: 输入 3 通道代替原文的 7 通道 (跳过背景估计阶段)
# ============================================================

class ConvBlock(nn.Module):
    """Conv2d + optional BatchNorm + optional Activation"""
    def __init__(self, in_c, out_c, kernel_size=4, stride=2, padding=1,
                 before=None, after=None, transposed=False):
        super().__init__()
        if transposed:
            self.conv = nn.ConvTranspose2d(in_c, out_c, kernel_size, stride, padding)
        else:
            self.conv = nn.Conv2d(in_c, out_c, kernel_size, stride, padding)
        self.before = before
        self.after = after
        if after == 'BN':
            self.norm = nn.BatchNorm2d(out_c)
        else:
            self.norm = None

    def forward(self, x):
        if self.before == 'ReLU':
            x = F.relu(x, inplace=True)
        elif self.before == 'LReLU':
            x = F.leaky_relu(x, 0.2, inplace=True)
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.after == 'Tanh':
            x = torch.tanh(x)
        elif self.after == 'sigmoid':
            x = torch.sigmoid(x)
        return x


class BEDSRGenerator(nn.Module):
    """BEDSR-Net Generator (simplified). Input: [B,3,H,W], Output: [B,3,H,W]"""
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        # Encoder: 64->128->256->512->512->512->512 (all stride-2 except first)
        self.enc0 = nn.Conv2d(in_channels, 64, 3, 1, 1)       # H, W
        self.enc1 = ConvBlock(64, 128, before='LReLU', after='BN')      # H/2, W/2
        self.enc2 = ConvBlock(128, 256, before='LReLU', after='BN')     # H/4, W/4
        self.enc3 = ConvBlock(256, 512, before='LReLU', after='BN')     # H/8, W/8
        self.enc4a = ConvBlock(512, 512, before='LReLU', after='BN')    # H/16, W/16
        self.enc4b = ConvBlock(512, 512, before='LReLU', after='BN')    # H/32, W/32
        self.enc4c = ConvBlock(512, 512, before='LReLU')                # H/64, W/64

        # Decoder (transposed conv doubles spatial dims)
        self.dec6 = ConvBlock(512, 512, before='ReLU', after='BN', transposed=True)   # H/32, W/32
        self.dec7a = ConvBlock(1024, 512, before='ReLU', after='BN', transposed=True) # H/16, W/16
        self.dec7b = ConvBlock(1024, 512, before='ReLU', after='BN', transposed=True) # H/8, W/8
        self.dec8 = ConvBlock(1024, 256, before='ReLU', after='BN', transposed=True)  # H/4, W/4
        self.dec9 = ConvBlock(512, 128, before='ReLU', after='BN', transposed=True)   # H/2, W/2
        self.dec10 = ConvBlock(256, 64, before='ReLU', after='BN', transposed=True)   # H, W
        self.dec11 = nn.Sequential(
            nn.Conv2d(128, out_channels, 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, x):
        e0 = self.enc0(x)       # B,64,H,W
        e1 = self.enc1(e0)      # B,128,H/2,W/2
        e2 = self.enc2(e1)      # B,256,H/4,W/4
        e3 = self.enc3(e2)      # B,512,H/8,W/8
        e4a = self.enc4a(e3)    # B,512,H/16,W/16
        e4b = self.enc4b(e4a)   # B,512,H/32,W/32
        e4c = self.enc4c(e4b)   # B,512,H/64,W/64

        d6 = self.dec6(e4c)     # B,512,H/32,W/32
        d6 = F.interpolate(d6, size=e4b.shape[2:], mode='bilinear', align_corners=True)

        d7a = self.dec7a(torch.cat([d6, e4b], dim=1))  # B,512,H/16,W/16
        d7a = F.interpolate(d7a, size=e4a.shape[2:], mode='bilinear', align_corners=True)

        d7b = self.dec7b(torch.cat([d7a, e4a], dim=1))  # B,512,H/8,W/8
        d7b = F.interpolate(d7b, size=e3.shape[2:], mode='bilinear', align_corners=True)

        d8 = self.dec8(torch.cat([d7b, e3], dim=1))     # B,256,H/4,W/4
        d9 = self.dec9(torch.cat([d8, e2], dim=1))      # B,128,H/2,W/2
        d10 = self.dec10(torch.cat([d9, e1], dim=1))    # B,64,H,W
        d10 = F.interpolate(d10, size=e0.shape[2:], mode='bilinear', align_corners=True)

        out = self.dec11(torch.cat([d10, e0], dim=1))   # B,3,H,W
        return (out + 1) / 2


# ============================================================
# UNet: 标准 U-Net 基线 (轻量版)
# ============================================================

class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class Down(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_c, out_c)

    def forward(self, x):
        return self.conv(self.pool(x))


class UNetUp(nn.Module):
    def __init__(self, in_c, skip_c, out_c):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_c, in_c, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_c + skip_c, out_c)

    def forward(self, x, skip):
        x = self.up(x)
        # Pad to match skip size
        diff_y = skip.size(2) - x.size(2)
        diff_x = skip.size(3) - x.size(3)
        if diff_y > 0 or diff_x > 0:
            x = F.pad(x, [diff_x // 2, diff_x - diff_x // 2,
                          diff_y // 2, diff_y - diff_y // 2])
        return self.conv(torch.cat([skip, x], dim=1))


class UNet(nn.Module):
    """Standard U-Net for shadow removal baseline.
    Args: base=32 → [32,64,128,256,256]
    """
    def __init__(self, in_channels=3, out_channels=3, base=32):
        super().__init__()
        self.inc = DoubleConv(in_channels, base)           # 3 -> 32
        self.down1 = Down(base, base * 2)                   # 32 -> 64
        self.down2 = Down(base * 2, base * 4)               # 64 -> 128
        self.down3 = Down(base * 4, base * 8)               # 128 -> 256
        self.down4 = Down(base * 8, base * 8)               # 256 -> 256

        self.up1 = UNetUp(base * 8, base * 8, base * 4)     # 256+256 -> 128
        self.up2 = UNetUp(base * 4, base * 4, base * 2)     # 128+128 -> 64
        self.up3 = UNetUp(base * 2, base * 2, base)         # 64+64 -> 32
        self.up4 = UNetUp(base, base, base)                 # 32+32 -> 32
        self.outc = nn.Conv2d(base, out_channels, 1)

    def forward(self, x):
        x1 = self.inc(x)        # base
        x2 = self.down1(x1)     # base*2
        x3 = self.down2(x2)     # base*4
        x4 = self.down3(x3)     # base*8
        x5 = self.down4(x4)     # base*8

        x = self.up1(x5, x4)    # base*4
        x = self.up2(x, x3)     # base*2
        x = self.up3(x, x2)     # base
        x = self.up4(x, x1)     # base
        return torch.sigmoid(self.outc(x))


# ============================================================
# NAFNet: Simple Baselines for Image Restoration (ECCV 2022)
# 使用 lightweight 配置 (width=16) 适合 2080 Ti 训练
# ============================================================

class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps
        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)
        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(dim=0), None


class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(c, dw_channel, 1, bias=True)
        self.conv2 = nn.Conv2d(dw_channel, dw_channel, 3, 1, 1, groups=dw_channel, bias=True)
        self.conv3 = nn.Conv2d(dw_channel // 2, c, 1, bias=True)

        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_channel // 2, dw_channel // 2, 1, bias=True),
        )
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(c, ffn_channel, 1, bias=True)
        self.conv5 = nn.Conv2d(ffn_channel // 2, c, 1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = self.norm1(inp)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)
        x = self.dropout1(x)
        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)
        x = self.dropout2(x)
        return y + x * self.gamma


class NAFNet(nn.Module):
    """NAFNet for shadow removal. lightweight config: width=16.
    Standard config (width=32) has ~68M params; we use width=16 → ~17M.
    """
    def __init__(self, img_channel=3, width=16, middle_blk_num=1,
                 enc_blk_nums=None, dec_blk_nums=None):
        super().__init__()
        if enc_blk_nums is None:
            enc_blk_nums = [1, 1, 1, 8]
        if dec_blk_nums is None:
            dec_blk_nums = [1, 1, 1, 1]

        self.intro = nn.Conv2d(img_channel, width, 3, 1, 1, bias=True)
        self.ending = nn.Conv2d(width, img_channel, 3, 1, 1, bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))
            self.downs.append(nn.Conv2d(chan, 2 * chan, 2, 2))
            chan = chan * 2

        self.middle_blks = nn.Sequential(*[NAFBlock(chan) for _ in range(middle_blk_num)])

        for num in dec_blk_nums:
            self.ups.append(nn.Sequential(
                nn.Conv2d(chan, chan * 2, 1, bias=False),
                nn.PixelShuffle(2)
            ))
            chan = chan // 2
            self.decoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp):
        B, C, H, W = inp.shape
        inp_padded = self.check_image_size(inp)
        x = self.intro(inp_padded)
        encs = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)
        x = self.middle_blks(x)
        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)
        x = self.ending(x)
        x = x + inp_padded
        return x[:, :, :H, :W]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        return F.pad(x, (0, mod_pad_w, 0, mod_pad_h))


# ============================================================
# ShadowGuidedNAFNet: NAFNet + shadow-mask-guided channel attention
# ============================================================

class ShadowEncoder(nn.Module):
    """轻量 shadow 特征提取器: 从灰度阴影图提取多尺度特征."""
    def __init__(self, width=16):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, width, 3, 1, 1, bias=True),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(width, width * 2, 3, 2, 1, bias=True),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(width * 2, width * 4, 3, 2, 1, bias=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, gray):
        f1 = self.conv1(gray)   # H,   width
        f2 = self.conv2(f1)      # H/2, width*2
        f3 = self.conv3(f2)      # H/4, width*4
        return f1, f2, f3


class SGCA(nn.Module):
    """Shadow-Guided Channel Attention: 用 shadow feature 调制 decoder feature."""
    def __init__(self, feat_ch, shadow_ch):
        super().__init__()
        self.shadow_proj = nn.Conv2d(shadow_ch, feat_ch, 1, bias=True)
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(feat_ch, feat_ch, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, feat, shadow_feat):
        s = self.shadow_proj(shadow_feat)
        attn = self.attn(feat + s)
        return feat * attn


class ShadowGuidedNAFNet(NAFNet):
    """NAFNet + shadow-guided channel attention in decoder.

    forward(gray, inp) where gray=[B,1,H,W] shadow mask, inp=[B,3,H,W] shadow image.
    Shadow features from gray are injected into decoder levels 1 and 2 via SGCA.
    """

    def __init__(self, img_channel=3, width=16, middle_blk_num=1,
                 enc_blk_nums=None, dec_blk_nums=None):
        super().__init__(img_channel=img_channel, width=width,
                         middle_blk_num=middle_blk_num,
                         enc_blk_nums=enc_blk_nums, dec_blk_nums=dec_blk_nums)
        self.shadow_encoder = ShadowEncoder(width=width)
        # decoder[1] at H/4: 64ch=width*4, decoder[2] at H/2: 32ch=width*2
        self.sgca_dec1 = SGCA(width * 4, width * 4)   # H/4, dec[1]
        self.sgca_dec2 = SGCA(width * 2, width * 2)    # H/2, dec[2]

    def forward(self, gray, inp):
        B, C, H, W = inp.shape
        inp_padded = self.check_image_size(inp)
        gray_padded = F.interpolate(gray, size=inp_padded.shape[2:],
                                     mode='bilinear', align_corners=False) if gray.shape[2:] != inp_padded.shape[2:] else gray

        # Shadow features
        s1, s2, s3 = self.shadow_encoder(gray_padded)

        # NAFNet encoder
        x = self.intro(inp_padded)
        encs = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        # NAFNet bottleneck
        x = self.middle_blks(x)

        for i, (decoder, up, enc_skip) in enumerate(zip(self.decoders, self.ups, encs[::-1])):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)
            if i == 1:   # H/4, width*4 ch
                x = self.sgca_dec1(x, s3)
            elif i == 2: # H/2, width*2 ch
                x = self.sgca_dec2(x, s2)

        x = self.ending(x)
        x = x + inp_padded
        return x[:, :, :H, :W]


class ShadowGuidedNAFNet_NoSGCA(NAFNet):
    """Ablation: NAFNet + ShadowEncoder, but SGCA replaced with concat+1x1 fusion."""

    def __init__(self, img_channel=3, width=16, middle_blk_num=1,
                 enc_blk_nums=None, dec_blk_nums=None):
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


class ShadowGuidedNAFNet_Concat(NAFNet):
    """Ablation: plain NAFNet with 4-channel input (RGB + gray shadow)."""

    def __init__(self, img_channel=4, width=16, middle_blk_num=1,
                 enc_blk_nums=None, dec_blk_nums=None):
        super().__init__(img_channel=img_channel, width=width,
                         middle_blk_num=middle_blk_num,
                         enc_blk_nums=enc_blk_nums, dec_blk_nums=dec_blk_nums)

    def forward(self, gray, inp):
        B, C, H, W = inp.shape
        gray_resized = F.interpolate(gray, size=(H, W),
                                      mode='bilinear', align_corners=False) if gray.shape[2:] != (H, W) else gray
        x = torch.cat([inp, gray_resized], dim=1)
        out = super().forward(x)
        return out[:, :3, :, :]  # strip gray channel from output


# 标准配置 dim=48, num_blocks=[4,6,6,8], heads=[1,2,4,8] → ~26M
# ============================================================

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class RestormerLayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type='WithBias'):
        super().__init__()
        self.body = WithBias_LayerNorm(dim) if LayerNorm_type == 'WithBias' else BiasFree_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.body(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        return x


class MDTA(nn.Module):
    """Multi-DConv Head Transposed Self-Attention"""
    def __init__(self, dim, num_heads, bias):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, 3, 1, 1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, 1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = attn @ v
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        return self.project_out(out)


class GDFN(nn.Module):
    """Gated-Dconv Feed-Forward Network"""
    def __init__(self, dim, ffn_expansion_factor, bias):
        super().__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features * 2, 1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, 3, 1, 1,
                                 groups=hidden_features * 2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, 1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        return self.project_out(x)


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super().__init__()
        self.norm1 = RestormerLayerNorm(dim, LayerNorm_type)
        self.attn = MDTA(dim, num_heads, bias)
        self.norm2 = RestormerLayerNorm(dim, LayerNorm_type)
        self.ffn = GDFN(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super().__init__()
        self.proj = nn.Conv2d(in_c, embed_dim, 3, 1, 1, bias=bias)

    def forward(self, x):
        return self.proj(x)


class RestormerDownsample(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat // 2, 3, 1, 1, bias=False),
            nn.PixelUnshuffle(2)
        )

    def forward(self, x):
        return self.body(x)


class RestormerUpsample(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat * 2, 3, 1, 1, bias=False),
            nn.PixelShuffle(2)
        )

    def forward(self, x):
        return self.body(x)


class Restormer(nn.Module):
    """Restormer for shadow removal. Standard config: dim=48, ~26M params."""
    def __init__(self, inp_channels=3, out_channels=3, dim=48,
                 num_blocks=None, num_refinement_blocks=4,
                 heads=None, ffn_expansion_factor=2.66,
                 bias=False, LayerNorm_type='WithBias'):
        super().__init__()
        if num_blocks is None:
            num_blocks = [4, 6, 6, 8]
        if heads is None:
            heads = [1, 2, 4, 8]

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for _ in range(num_blocks[0])])

        self.down1_2 = RestormerDownsample(dim)
        self.encoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for _ in range(num_blocks[1])])

        self.down2_3 = RestormerDownsample(int(dim * 2))
        self.encoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 4), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for _ in range(num_blocks[2])])

        self.down3_4 = RestormerDownsample(int(dim * 4))
        self.latent = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 8), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for _ in range(num_blocks[3])])

        self.up4_3 = RestormerUpsample(int(dim * 8))
        self.reduce_chan_level3 = nn.Conv2d(int(dim * 8), int(dim * 4), 1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 4), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for _ in range(num_blocks[2])])

        self.up3_2 = RestormerUpsample(int(dim * 4))
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 4), int(dim * 2), 1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for _ in range(num_blocks[1])])

        self.up2_1 = RestormerUpsample(int(dim * 2))
        self.decoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for _ in range(num_blocks[0])])

        self.refinement = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for _ in range(num_refinement_blocks)])

        self.output = nn.Conv2d(int(dim * 2), out_channels, 3, 1, 1, bias=bias)

    def forward(self, inp_img):
        inp_enc_level1 = self.patch_embed(inp_img)
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

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        out_dec_level1 = self.refinement(out_dec_level1)
        out_dec_level1 = self.output(out_dec_level1) + inp_img

        return out_dec_level1


# ============================================================
# ShadowGuidedRestormer: Restormer + ShadowEncoder + SGCA
# ============================================================

class ShadowGuidedRestormer(Restormer):
    """Restormer + ShadowEncoder + SGCA shadow-guided document restoration.

    Shadow features from grayscale input are injected into decoder levels
    2 and 3 via channel attention (SGCA), analogous to ShadowGuidedNAFNet.

    forward(gray, inp) where gray=[B,1,H,W], inp=[B,3,H,W].
    """

    def __init__(self, inp_channels=3, out_channels=3, dim=48,
                 num_blocks=None, num_refinement_blocks=4,
                 heads=None, ffn_expansion_factor=2.66,
                 bias=False, LayerNorm_type='WithBias'):
        super().__init__(inp_channels=inp_channels, out_channels=out_channels, dim=dim,
                         num_blocks=num_blocks, num_refinement_blocks=num_refinement_blocks,
                         heads=heads, ffn_expansion_factor=ffn_expansion_factor,
                         bias=bias, LayerNorm_type=LayerNorm_type)
        self.shadow_encoder = ShadowEncoder(width=dim)
        self.sgca_dec3 = SGCA(dim * 4, dim * 4)
        self.sgca_dec2 = SGCA(dim * 2, dim * 2)

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
        out_dec_level3 = self.sgca_dec3(out_dec_level3, s3)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)
        out_dec_level2 = self.sgca_dec2(out_dec_level2, s2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        out_dec_level1 = self.refinement(out_dec_level1)
        out_dec_level1 = self.output(out_dec_level1) + inp

        return out_dec_level1


class ShadowGuidedRestormer_NoSGCA(Restormer):
    """Ablation: Restormer + ShadowEncoder, SGCA replaced with concat+1x1 fusion."""

    def __init__(self, inp_channels=3, out_channels=3, dim=48,
                 num_blocks=None, num_refinement_blocks=4,
                 heads=None, ffn_expansion_factor=2.66,
                 bias=False, LayerNorm_type='WithBias'):
        super().__init__(inp_channels=inp_channels, out_channels=out_channels, dim=dim,
                         num_blocks=num_blocks, num_refinement_blocks=num_refinement_blocks,
                         heads=heads, ffn_expansion_factor=ffn_expansion_factor,
                         bias=bias, LayerNorm_type=LayerNorm_type)
        self.shadow_encoder = ShadowEncoder(width=dim)
        self.fuse_dec3 = nn.Conv2d(dim * 4 + dim * 4, dim * 4, 1, bias=True)
        self.fuse_dec2 = nn.Conv2d(dim * 2 + dim * 2, dim * 2, 1, bias=True)

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
        out_dec_level3 = self.fuse_dec3(torch.cat([out_dec_level3, s3], dim=1))

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)
        out_dec_level2 = self.fuse_dec2(torch.cat([out_dec_level2, s2], dim=1))

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        out_dec_level1 = self.refinement(out_dec_level1)
        out_dec_level1 = self.output(out_dec_level1) + inp

        return out_dec_level1


class ShadowGuidedRestormer_Concat(Restormer):
    """Ablation: plain Restormer with 4-channel input (RGB + gray shadow)."""

    def __init__(self, inp_channels=4, out_channels=4, dim=48,
                 num_blocks=None, num_refinement_blocks=4,
                 heads=None, ffn_expansion_factor=2.66,
                 bias=False, LayerNorm_type='WithBias'):
        super().__init__(inp_channels=inp_channels, out_channels=out_channels, dim=dim,
                         num_blocks=num_blocks, num_refinement_blocks=num_refinement_blocks,
                         heads=heads, ffn_expansion_factor=ffn_expansion_factor,
                         bias=bias, LayerNorm_type=LayerNorm_type)

    def forward(self, gray, inp):
        B, C, H, W = inp.shape
        if gray.shape[2:] != (H, W):
            gray = F.interpolate(gray, size=(H, W), mode='bilinear', align_corners=False)
        x = torch.cat([inp, gray], dim=1)
        out = super().forward(x)
        return out[:, :3, :, :]  # strip gray channel from output



# ============================================================
# Transformer-Compatible Shadow-Guided Fusion Modules
# SGCF: Cross-Attention Fusion (Q=decoder, K/V=shadow)
# SGFM: FiLM-style Feature Modulation
# ============================================================

class SGCF(nn.Module):
    def __init__(self, feat_ch, shadow_ch, num_heads=4):
        super().__init__()
        self.shadow_proj = nn.Conv2d(shadow_ch, feat_ch, 1, bias=True)
        self.norm_q = nn.LayerNorm(feat_ch)
        self.norm_kv = nn.LayerNorm(feat_ch)
        self.cross_attn = nn.MultiheadAttention(feat_ch, num_heads, batch_first=True)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, feat, shadow_feat):
        s = self.shadow_proj(shadow_feat)
        B, C, H, W = feat.shape
        q = feat.flatten(2).transpose(1, 2)
        kv = s.flatten(2).transpose(1, 2)
        q = self.norm_q(q)
        kv = self.norm_kv(kv)
        out, _ = self.cross_attn(q, kv, kv)
        out = out.transpose(1, 2).view(B, C, H, W)
        return feat + out * self.gamma


# ============================================================
# SGGF: Shadow-Guided Gated Fusion
# Gate = sigmoid(GAP([feat, shadow])) — bounded [0,1], no gradient explosion.
# More stable than FiLM (tanh can saturate), lighter than CrossAttn.
# ============================================================
class SGGF(nn.Module):
    """Per-channel gated fusion: feat * gate + shadow_proj * (1-gate).

    Shadow features are projected and concatenated with decoder features,
    then a squeeze-excitation-style gate (GAP + FC + sigmoid) selects
    per-channel how much shadow information to admit.
    """
    def __init__(self, feat_ch, shadow_ch, reduction=4):
        super().__init__()
        self.shadow_proj = nn.Conv2d(shadow_ch, feat_ch, 1, bias=True)
        self.gate_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(feat_ch * 2, feat_ch // reduction, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_ch // reduction, feat_ch, 1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, feat, shadow_feat):
        s = self.shadow_proj(shadow_feat)
        gate = self.gate_net(torch.cat([feat, s], dim=1))
        return feat * gate + s * (1 - gate)


class SGFM(nn.Module):
    def __init__(self, feat_ch, shadow_ch):
        super().__init__()
        self.shadow_proj = nn.Conv2d(shadow_ch, feat_ch, 1, bias=True)
        self.gamma_conv = nn.Conv2d(feat_ch, feat_ch, 1, bias=True)
        self.beta_conv = nn.Conv2d(feat_ch, feat_ch, 1, bias=True)

    def forward(self, feat, shadow_feat):
        s = self.shadow_proj(shadow_feat)
        gamma = torch.tanh(self.gamma_conv(s))
        beta = self.beta_conv(s)
        return feat * (1 + gamma) + beta
# ============================================================
# ShadowGuidedRestormer_CrossAttn: Restormer + ShadowEncoder + SGCF
# ============================================================

class ShadowGuidedRestormer_CrossAttn(Restormer):

    def __init__(self, inp_channels=3, out_channels=3, dim=48,
                 num_blocks=None, num_refinement_blocks=4,
                 heads=None, ffn_expansion_factor=2.66,
                 bias=False, LayerNorm_type='WithBias', cross_heads=4):
        super().__init__(inp_channels=inp_channels, out_channels=out_channels, dim=dim,
                         num_blocks=num_blocks, num_refinement_blocks=num_refinement_blocks,
                         heads=heads, ffn_expansion_factor=ffn_expansion_factor,
                         bias=bias, LayerNorm_type=LayerNorm_type)
        self.shadow_encoder = ShadowEncoder(width=dim)
        self.sgcf_dec3 = SGCF(dim * 4, dim * 4, num_heads=cross_heads)
        self.sgcf_dec2 = SGCF(dim * 2, dim * 2, num_heads=cross_heads)

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
        out_dec_level3 = self.sgcf_dec3(out_dec_level3, s3)
        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)
        out_dec_level2 = self.sgcf_dec2(out_dec_level2, s2)
        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)
        out_dec_level1 = self.refinement(out_dec_level1)
        out_dec_level1 = self.output(out_dec_level1) + inp
        return out_dec_level1


# ============================================================
# ShadowGuidedRestormer_FiLM: Restormer + ShadowEncoder + SGFM
# ============================================================

class ShadowGuidedRestormer_FiLM(Restormer):

    def __init__(self, inp_channels=3, out_channels=3, dim=48,
                 num_blocks=None, num_refinement_blocks=4,
                 heads=None, ffn_expansion_factor=2.66,
                 bias=False, LayerNorm_type='WithBias'):
        super().__init__(inp_channels=inp_channels, out_channels=out_channels, dim=dim,
                         num_blocks=num_blocks, num_refinement_blocks=num_refinement_blocks,
                         heads=heads, ffn_expansion_factor=ffn_expansion_factor,
                         bias=bias, LayerNorm_type=LayerNorm_type)
        self.shadow_encoder = ShadowEncoder(width=dim)
        self.sgfm_dec3 = SGFM(dim * 4, dim * 4)
        self.sgfm_dec2 = SGFM(dim * 2, dim * 2)

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
        out_dec_level3 = self.sgfm_dec3(out_dec_level3, s3)
        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)
        out_dec_level2 = self.sgfm_dec2(out_dec_level2, s2)
        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)
        out_dec_level1 = self.refinement(out_dec_level1)
        out_dec_level1 = self.output(out_dec_level1) + inp
        return out_dec_level1


# ============================================================
# ShadowGuidedRestormer_Large: dim=64 + ShadowEncoder + concat
# ============================================================

# ============================================================
# ShadowGuidedRestormer_Gated: Restormer + ShadowEncoder + SGGF
# Standard dim=48, multi-scale gated fusion at decoder levels 2 and 3.
# ============================================================
class ShadowGuidedRestormer_Gated(Restormer):
    """Shadow-guided gated fusion with sigmoid gates.

    Uses SGGF at decoder bottleneck and mid-level for stable,
    per-channel blending of shadow features. No tanh → no NaN risk.
    """
    def __init__(self, inp_channels=3, out_channels=3, dim=48,
                 num_blocks=None, num_refinement_blocks=4,
                 heads=None, ffn_expansion_factor=2.66,
                 bias=False, LayerNorm_type='WithBias'):
        super().__init__(inp_channels=inp_channels, out_channels=out_channels, dim=dim,
                         num_blocks=num_blocks, num_refinement_blocks=num_refinement_blocks,
                         heads=heads, ffn_expansion_factor=ffn_expansion_factor,
                         bias=bias, LayerNorm_type=LayerNorm_type)
        self.shadow_encoder = ShadowEncoder(width=dim)
        self.sggf_dec3 = SGGF(dim * 4, dim * 4)
        self.sggf_dec2 = SGGF(dim * 2, dim * 2)

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
        out_dec_level3 = self.sggf_dec3(out_dec_level3, s3)
        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)
        out_dec_level2 = self.sggf_dec2(out_dec_level2, s2)
        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)
        out_dec_level1 = self.refinement(out_dec_level1)
        out_dec_level1 = self.output(out_dec_level1) + inp
        return out_dec_level1


# ============================================================
# ShadowGuidedRestormer_GatedLarge: dim=64 + ShadowEncoder + SGGF
# ============================================================
class ShadowGuidedRestormer_GatedLarge(Restormer):
    """Large (dim=64) variant with gated fusion.

    Same SGGF mechanism as the standard Gated model, but with
    wider feature channels throughout the Restormer backbone.
    """
    def __init__(self, inp_channels=3, out_channels=3, dim=64,
                 num_blocks=None, num_refinement_blocks=4,
                 heads=None, ffn_expansion_factor=2.66,
                 bias=False, LayerNorm_type='WithBias'):
        super().__init__(inp_channels=inp_channels, out_channels=out_channels, dim=dim,
                         num_blocks=num_blocks, num_refinement_blocks=num_refinement_blocks,
                         heads=heads, ffn_expansion_factor=ffn_expansion_factor,
                         bias=bias, LayerNorm_type=LayerNorm_type)
        self.shadow_encoder = ShadowEncoder(width=dim)
        self.sggf_dec3 = SGGF(dim * 4, dim * 4)
        self.sggf_dec2 = SGGF(dim * 2, dim * 2)

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
        out_dec_level3 = self.sggf_dec3(out_dec_level3, s3)
        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)
        out_dec_level2 = self.sggf_dec2(out_dec_level2, s2)
        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)
        out_dec_level1 = self.refinement(out_dec_level1)
        out_dec_level1 = self.output(out_dec_level1) + inp
        return out_dec_level1


class ShadowGuidedRestormer_Large(Restormer):

    def __init__(self, inp_channels=3, out_channels=3, dim=64,
                 num_blocks=None, num_refinement_blocks=4,
                 heads=None, ffn_expansion_factor=2.66,
                 bias=False, LayerNorm_type='WithBias'):
        super().__init__(inp_channels=inp_channels, out_channels=out_channels, dim=dim,
                         num_blocks=num_blocks, num_refinement_blocks=num_refinement_blocks,
                         heads=heads, ffn_expansion_factor=ffn_expansion_factor,
                         bias=bias, LayerNorm_type=LayerNorm_type)
        self.shadow_encoder = ShadowEncoder(width=dim)
        self.fuse_dec3 = nn.Conv2d(dim * 4 + dim * 4, dim * 4, 1, bias=True)
        self.fuse_dec2 = nn.Conv2d(dim * 2 + dim * 2, dim * 2, 1, bias=True)

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
        out_dec_level3 = self.fuse_dec3(torch.cat([out_dec_level3, s3], dim=1))
        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)
        out_dec_level2 = self.fuse_dec2(torch.cat([out_dec_level2, s2], dim=1))
        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)
        out_dec_level1 = self.refinement(out_dec_level1)
        out_dec_level1 = self.output(out_dec_level1) + inp
        return out_dec_level1


# ============================================================
# 测试
# ============================================================
if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    for name, model_cls in [
        ('BEDSRGenerator', BEDSRGenerator),
        ('UNet', UNet),
        ('NAFNet', NAFNet),
        ('Restormer', Restormer),
    ]:
        m = model_cls().to(device)
        x = torch.randn(2, 3, 384, 384).to(device)
        y = m(x)
        p = sum(p.numel() for p in m.parameters())
        print(f"{name}: {x.shape} -> {y.shape}, params: {p:,}")

# ============================================================
# SGGF Ablation: No Shadow Encoder (gate on decoder features only)
# ============================================================
class ShadowGuidedRestormer_Gated_NoShadow(Restormer):
    """Ablation: SGGF without shadow encoder. Gate computed from decoder features alone."""
    def __init__(self, inp_channels=3, out_channels=3, dim=48,
                 num_blocks=None, num_refinement_blocks=4,
                 heads=None, ffn_expansion_factor=2.66,
                 bias=False, LayerNorm_type='WithBias'):
        super().__init__(inp_channels=inp_channels, out_channels=out_channels, dim=dim,
                         num_blocks=num_blocks, num_refinement_blocks=num_refinement_blocks,
                         heads=heads, ffn_expansion_factor=ffn_expansion_factor,
                         bias=bias, LayerNorm_type=LayerNorm_type)
        self.shadow_encoder = ShadowEncoder(width=dim)
        self.sggf_dec3 = SGGF(dim * 4, dim * 4)
        self.sggf_dec2 = SGGF(dim * 2, dim * 2)

    def forward(self, gray, inp):
        B, C, H, W = inp.shape
        if gray.shape[2:] != (H, W):
            gray = F.interpolate(gray, size=(H, W), mode='bilinear', align_corners=False)
        s1, s2, s3 = self.shadow_encoder(gray)
        # Replace shadow features with zeros for ablation
        s2 = torch.zeros_like(s2)
        s3 = torch.zeros_like(s3)
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
        out_dec_level3 = self.sggf_dec3(out_dec_level3, s3)
        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)
        out_dec_level2 = self.sggf_dec2(out_dec_level2, s2)
        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)
        out_dec_level1 = self.refinement(out_dec_level1)
        out_dec_level1 = self.output(out_dec_level1) + inp
        return out_dec_level1


# ============================================================
# SGGF Ablation: Dec3-only fusion (single-scale)
# ============================================================
class ShadowGuidedRestormer_Gated_Dec3Only(Restormer):
    """Ablation: SGGF fusion only at decoder level 3 (bottleneck)."""
    def __init__(self, inp_channels=3, out_channels=3, dim=48,
                 num_blocks=None, num_refinement_blocks=4,
                 heads=None, ffn_expansion_factor=2.66,
                 bias=False, LayerNorm_type='WithBias'):
        super().__init__(inp_channels=inp_channels, out_channels=out_channels, dim=dim,
                         num_blocks=num_blocks, num_refinement_blocks=num_refinement_blocks,
                         heads=heads, ffn_expansion_factor=ffn_expansion_factor,
                         bias=bias, LayerNorm_type=LayerNorm_type)
        self.shadow_encoder = ShadowEncoder(width=dim)
        self.sggf_dec3 = SGGF(dim * 4, dim * 4)

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
        out_dec_level3 = self.sggf_dec3(out_dec_level3, s3)
        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)
        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)
        out_dec_level1 = self.refinement(out_dec_level1)
        out_dec_level1 = self.output(out_dec_level1) + inp
        return out_dec_level1
