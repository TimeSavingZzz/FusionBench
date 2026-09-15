"""
TextAware ShaDocFormer v2: 可学习文字感知文档阴影去除

改进 (vs v1):
1. LDTD (Lightweight Differentiable Text Detector) 替代 MSER — 端到端可学习
2. 5层 SFT 多尺度文字条件特征调制 — 替代单点乘法 gate
3. Adaptive Hierarchical Loss (L1 + VGG + Edge + DTRM + SSIM) — 替代固定权重 MSE

用法:
    model = TextAwareModel()
    res, text_map = model(bin_x, x, return_text_map=True)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.mask import ThresholdFormer
from models.refine import Refiner, SFTLayer

# EasyOCR reader singleton
_easyocr_reader = None


def _get_easyocr_reader(gpu=True):
    global _easyocr_reader
    if _easyocr_reader is None:
        import easyocr
        _easyocr_reader = easyocr.Reader(['en'], gpu=gpu, verbose=False)
    return _easyocr_reader


# ---------------------------------------------------------------------------
# LDTD: Lightweight Differentiable Text Detector
# ---------------------------------------------------------------------------

class LDTD(nn.Module):
    """可学习轻量文字检测器.

    从 encoder level-1 features [B, 32, H/2, W/2] 预测文字概率图 [B, 1, H/2, W/2].
    参数量 ~15K.
    """

    def __init__(self, in_channels=32):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 8, 3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
        )
        self.conv4 = nn.Conv2d(8, 1, 3, padding=1)

    def forward(self, x):
        f = self.conv1(x)
        f = self.conv2(f)
        f = self.conv3(f)
        return torch.sigmoid(self.conv4(f))


# ---------------------------------------------------------------------------
# VGGPerceptualLoss
# ---------------------------------------------------------------------------

class VGGPerceptualLoss(nn.Module):
    """VGG16 感知损失 (relu3_3), 冻结预训练权重."""

    def __init__(self):
        super().__init__()
        from torchvision import models
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        self.vgg = nn.Sequential(*list(vgg.features.children())[:16])
        self.vgg.eval()
        for p in self.vgg.parameters():
            p.requires_grad = False
        self._vgg_device = None

    def forward(self, pred, target):
        if self._vgg_device != pred.device:
            self.vgg = self.vgg.to(pred.device)
            self._vgg_device = pred.device
        with torch.cuda.amp.autocast(enabled=False):
            p = (pred.float() - 0.5) * 2.0
            t = (target.float() - 0.5) * 2.0
            return F.l1_loss(self.vgg(p), self.vgg(t))


# ---------------------------------------------------------------------------
# TextAwareRefiner v2
# ---------------------------------------------------------------------------

class TextAwareRefiner(Refiner):
    """TextAware Refiner: supports v1 (MSER+gate, no SFT/LDTD) and v2 (LDTD+5-SFT)."""

    def __init__(self, variant='v2', pretrained_ldtd_path=None):
        super().__init__()
        self.variant = variant
        dim = 16

        if variant == 'v2':
            # 5 个 SFT 注入层
            self.sft_enc1 = SFTLayer(dim * 2)
            self.sft_enc2 = SFTLayer(dim * 4)
            self.sft_bottleneck = SFTLayer(dim * 8)
            self.sft_dec2 = SFTLayer(dim * 2)
            self.sft_dec1 = SFTLayer(dim)
            # LDTD 文字检测器
            self.text_detector = LDTD(in_channels=dim * 2)
            if pretrained_ldtd_path is not None:
                self._load_pretrained_ldtd(pretrained_ldtd_path)

    def _extract_text_attention(self, x):
        """从 encoder level-1 features 提取文字注意力图 (v2 only)."""
        with torch.no_grad():
            inp_enc = self.patch_embed(x)
        enc1_feat = self.down_1(inp_enc)
        text_map = self.text_detector(enc1_feat)
        return text_map

    def forward(self, x, text_attention=None, return_text_map=False):
        if self.variant == 'v1':
            # v1: 使用外部 MSER text_attention, 无 LDTD, 无 SFT
            restored = super().forward(x, text_attention=None)
            if return_text_map:
                return restored, None
            return restored

        # v2: LDTD + 5-SFT
        if text_attention is None:
            text_attention = self._extract_text_attention(x)
        restored = super().forward(x, text_attention=text_attention)
        if return_text_map:
            return restored, text_attention
        return restored

    def _load_pretrained_ldtd(self, path):
        ckpt = torch.load(path, map_location='cpu')
        self.text_detector.load_state_dict(ckpt)


# ---------------------------------------------------------------------------
# TextAwareModel v2
# ---------------------------------------------------------------------------

class TextAwareModel(nn.Module):
    """TextAware ShaDocFormer — supports v1 (MSER+gate) and v2 (LDTD+5-SFT)."""

    def __init__(self, num_trans_blocks=3, variant='v2', pretrained_ldtd_path=None):
        super().__init__()
        self.variant = variant
        self.mask = ThresholdFormer(num_trans_blocks=num_trans_blocks)
        self.refine = TextAwareRefiner(variant=variant, pretrained_ldtd_path=pretrained_ldtd_path)

    def forward(self, bin_x, x, text_attention=None, return_text_map=False):
        """
        Args:
            bin_x: [B, 1, H, W] 灰度阴影图
            x: [B, 3, H, W] RGB 阴影图
            text_attention: [B, 1, H/2, W/2] 可选 MSER pseudo-label (仅用于 DTRM loss 监督)
            return_text_map: 若 True, 返回 (restored, text_map)
        """
        mask = self.mask(bin_x)
        x_res = torch.cat((x, mask), dim=1)
        return self.refine(x_res, text_attention=text_attention,
                           return_text_map=return_text_map)


# ---------------------------------------------------------------------------
# TextAwareLoss v2: Adaptive Hierarchical Loss
# ---------------------------------------------------------------------------

class TextAwareLoss(nn.Module):
    """v2 自适应分层文本感知损失.

    组件:
        L_text: L1 + VGG perceptual (自适应权重, 基于文字占比)
        L_bg:   L1 only
        L_edge: Sobel 梯度边缘损失 (文字边界)
        L_dtrm: BCE 损失监督文字检测器
        L_ssim: 全局 SSIM
    """

    def __init__(self, ssim_weight=0.3, vgg_weight=0.1, edge_weight=0.5,
                 dtrm_weight=0.5, fixed_adaptive_weight=-1.0):
        super().__init__()
        self.ssim_weight = ssim_weight
        self.vgg_weight = vgg_weight
        self.edge_weight = edge_weight
        self.dtrm_weight = dtrm_weight
        self.fixed_adaptive_weight = fixed_adaptive_weight
        self.vgg_loss = VGGPerceptualLoss()

        # Sobel kernels for edge detection
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                                dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                                dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def forward(self, pred, target, text_attention, text_pred=None, text_target_map=None):
        """
        Args:
            pred: [B, 3, H, W] 恢复图像
            target: [B, 3, H, W] GT
            text_attention: [B, 1, H, W] 文字概率图 (用于损失加权)
            text_pred: [B, 1, H/2, W/2] LDTD 输出 (用于 DTRM BCE 监督)
            text_target_map: [B, 1, H/2, W/2] MSER pseudo-label (DTRM 目标)
        """
        # --- 自适应文字权重: 文字占比越大, 权重越高 ---
        if self.fixed_adaptive_weight > 0:
            adaptive_text_w = self.fixed_adaptive_weight
        else:
            text_ratio = text_attention.mean()
            adaptive_text_w = 1.0 + 3.0 * text_ratio  # [1.0, 4.0]

        # 对齐 text_attention 到 pred 尺寸
        if text_attention.shape[2:] != pred.shape[2:]:
            text_mask = F.interpolate(text_attention, size=pred.shape[2:],
                                      mode='bilinear', align_corners=False)
        else:
            text_mask = text_attention

        # --- L_text: L1 on text regions + VGG perceptual ---
        text_area = text_mask.sum().clamp(min=100.0)
        loss_text_l1 = (F.l1_loss(pred, target, reduction='none')
                        * text_mask.expand_as(pred)).sum() / text_area
        loss_text_l1 = loss_text_l1 * adaptive_text_w

        text_pred_vgg = pred * text_mask.expand_as(pred)
        text_target_vgg = target * text_mask.expand_as(pred)
        loss_text_vgg = self.vgg_loss(text_pred_vgg, text_target_vgg)
        loss_text = loss_text_l1 + self.vgg_weight * loss_text_vgg

        # --- L_bg: L1 on background ---
        bg_mask = (1.0 - text_mask).clamp(min=0.0)
        bg_area = bg_mask.sum().clamp(min=100.0)
        loss_bg = (F.l1_loss(pred, target, reduction='none')
                   * bg_mask.expand_as(pred)).sum() / bg_area

        # --- L_edge: gradient loss on text boundaries ---
        loss_edge = self._edge_loss(pred, target, text_mask)

        # --- L_ssim: global SSIM ---
        loss_ssim = 1 - self._ssim(pred, target)

        # --- L_dtrm: BCE for text detector ---
        loss_dtrm = torch.tensor(0.0, device=pred.device)
        if text_pred is not None and text_target_map is not None:
            if text_pred.shape[2:] != text_target_map.shape[2:]:
                tgt = F.interpolate(text_target_map, size=text_pred.shape[2:],
                                    mode='bilinear', align_corners=False)
            else:
                tgt = text_target_map
            pos_weight = torch.tensor([50.0], device=pred.device)
            loss_dtrm = F.binary_cross_entropy_with_logits(
                torch.logit(text_pred.clamp(1e-6, 1-1e-6)), tgt,
                pos_weight=pos_weight)

        total = (loss_text + loss_bg +
                 self.edge_weight * loss_edge +
                 self.ssim_weight * loss_ssim +
                 self.dtrm_weight * loss_dtrm)

        return total, {
            'loss_text': loss_text.item(),
            'loss_bg': loss_bg.item(),
            'loss_edge': loss_edge.item(),
            'loss_vgg': loss_text_vgg.item(),
            'loss_ssim': loss_ssim.item(),
            'loss_dtrm': loss_dtrm.item(),
            'adaptive_w': adaptive_text_w.item() if isinstance(adaptive_text_w, torch.Tensor) else adaptive_text_w,
            'total': total.item(),
        }

    def _edge_loss(self, pred, target, text_mask):
        """Sobel 梯度边缘损失 — 让文字边缘更清晰."""
        b = text_mask.shape[0]
        edge_maps = []
        for i in range(b):
            m = text_mask[i:i+1]  # [1, 1, H, W]
            ex = F.conv2d(m, self.sobel_x, padding=1)
            ey = F.conv2d(m, self.sobel_y, padding=1)
            edge = torch.sqrt(ex ** 2 + ey ** 2 + 1e-8)
            edge = edge / edge.max().clamp(min=1e-8)
            edge_maps.append(edge)
        edge_map = torch.cat(edge_maps, dim=0).clamp(0, 1)

        edge_area = edge_map.sum().clamp(min=100.0)
        return (F.l1_loss(pred, target, reduction='none')
                * edge_map.expand_as(pred)).sum() / edge_area

    @staticmethod
    def _ssim(img1, img2, window_size=11):
        from torchmetrics.functional import structural_similarity_index_measure
        return structural_similarity_index_measure(img1, img2, data_range=1)


# ---------------------------------------------------------------------------
# OCR evaluation utility
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# TextDetector v1 (keep for backward compatibility with train_compare_models)
# ---------------------------------------------------------------------------

class TextDetector:
    """v1 MSER 文字检测器 (向后兼容). v2 请使用 TextAwareModel 内置 LDTD."""

    def __init__(self, method='opencv', device='cuda'):
        self.method = method
        self.device = device
        self._detector = None
        if method == 'opencv':
            self._init_opencv_mser()

    def _init_opencv_mser(self):
        import cv2
        try:
            self._mser = cv2.MSER_create(
                _delta=5, _min_area=30, _max_area=5000,
                _max_variation=0.25, _min_diversity=0.2,
                _max_evolution=200, _area_threshold=1.01,
                _min_margin=0.003, _edge_blur_size=5)
        except TypeError:
            self._mser = cv2.MSER_create()

    def _detect_mser(self, img_np):
        import cv2
        if img_np.max() <= 1.0:
            img_np = (img_np * 255).astype('uint8')
        if len(img_np.shape) == 3:
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_np
        regions, _ = self._mser.detectRegions(gray)
        mask = torch.zeros(gray.shape, dtype=torch.float32)
        if regions is not None:
            for region in regions:
                if len(region) > 0:
                    hull = cv2.convexHull(region.reshape(-1, 1, 2))
                    cv2.fillConvexPoly(mask.numpy(), hull, 10.0)
        mask = torch.clamp(mask / 10.0, 0.0, 1.0)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 5))
        mask_np = cv2.dilate(mask.numpy(), kernel, iterations=2)
        mask_np = cv2.GaussianBlur(mask_np, (21, 21), 10)
        return torch.from_numpy(mask_np).float()

    def detect(self, img):
        if isinstance(img, torch.Tensor):
            if img.dim() == 4:
                img = img[0]
            img_np = img.permute(1, 2, 0).cpu().numpy()
            if img_np.max() <= 1.0:
                img_np = (img_np * 255).astype('uint8')
        if self.method == 'opencv':
            mask = self._detect_mser(img_np)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        return mask

    @staticmethod
    def batch_detect(detector, imgs, device='cpu'):
        batch_size = imgs.shape[0]
        text_attentions = []
        for i in range(batch_size):
            attn = detector.detect(imgs[i])
            text_attentions.append(attn.unsqueeze(0))
        return torch.stack(text_attentions).to(device)


def compute_ocr_accuracy(original_img, restored_img, gt_img):
    """计算 OCR 识别准确率提升."""
    import re
    try:
        reader = _get_easyocr_reader(gpu=True)
        results_orig = reader.readtext(
            (original_img.permute(1, 2, 0).cpu().numpy() * 255).astype('uint8'), detail=0)
        results_restored = reader.readtext(
            (restored_img.permute(1, 2, 0).cpu().numpy() * 255).astype('uint8'), detail=0)
        results_gt = reader.readtext(
            (gt_img.permute(1, 2, 0).cpu().numpy() * 255).astype('uint8'), detail=0)

        original_chars = len(re.findall(r'\w', ' '.join(results_orig)))
        restored_chars = len(re.findall(r'\w', ' '.join(results_restored)))
        gt_chars = len(re.findall(r'\w', ' '.join(results_gt)))

        return {
            'original_ocr_chars': original_chars,
            'restored_ocr_chars': restored_chars,
            'gt_ocr_chars': gt_chars,
            'ocr_recovery_rate': restored_chars / max(gt_chars, 1),
        }
    except ImportError:
        return None
