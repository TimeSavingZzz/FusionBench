"""
统一对比实验脚本 v2: 支持多模型、多数据集、断点续训
用法: python train_compare_models.py --model all --epochs 200 --dataset rdd
      python train_compare_models.py --model textaware --epochs 300 --dataset sd7k
      python train_compare_models.py --model restormer --resume ./xxx/textaware_latest.pth
"""
import os, sys, time, warnings, argparse, json, glob as glob_m
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchmetrics.functional import peak_signal_noise_ratio, structural_similarity_index_measure
from torchmetrics.functional.regression import mean_squared_error
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import Model as ShaDocFormer
from models.text_aware_model import TextAwareModel, TextDetector, TextAwareLoss
from models.comparison_models import (BEDSRGenerator, UNet, NAFNet, Restormer, ShadowGuidedNAFNet, ShadowGuidedNAFNet_NoSGCA, ShadowGuidedNAFNet_Concat, ShadowGuidedRestormer, ShadowGuidedRestormer_NoSGCA, ShadowGuidedRestormer_Concat,
    ShadowGuidedRestormer_CrossAttn, ShadowGuidedRestormer_FiLM, ShadowGuidedRestormer_Large,
    ShadowGuidedRestormer_Gated, ShadowGuidedRestormer_GatedLarge,
    ShadowGuidedRestormer_Gated_NoShadow, ShadowGuidedRestormer_Gated_Dec3Only)
from models.fusion_models import (ShadowGuidedNAFNet_FiLM, ShadowGuidedNAFNet_Gated,
    ShadowGuidedNAFNet_CrossAttn, ShadowGuidedNAFNet_ASF, ShadowGuidedNAFNet_Large,
    ShadowGuidedRestormer_ASF)
from config.config import Config
from data.data_RGB import get_data
from data.dataset_lol import LOLDataReader
from data.dataset_rain100h import Rain100HDataReader
from utils import seed_everything

# ---------------------------------------------------------------------------
# Dataset paths
# ---------------------------------------------------------------------------
DATASET_CONFIGS = {
    'rdd': {
        'train_dir': './dataset/RDD/train/',
        'val_dir': './dataset/RDD/test/',
        'input': 'img',
        'target': 'back_gt',
    },
    'sd7k': {
        'train_dir': './dataset/SD7K/train/',
        'val_dir': './dataset/SD7K/test/',
        'input': 'input',
        'target': 'target',
    },
    'synthetic': {
        'train_dir': './dataset/Synthetic/train/',
        'val_dir': './dataset/Synthetic/test/',
        'input': 'input',
        'target': 'target',
    },
    'lol': {
        'train_dir': './dataset/LOL-v1/train/',
        'val_dir': './dataset/LOL-v1/test/',
        'input': 'input',
        'target': 'target',
    },
    'rain100h': {
        'train_dir': './dataset/Rain100H/train/',
        'val_dir': './dataset/Rain100H/test/',
        'input': 'input',
        'target': 'target',
    },
}

# ---------------------------------------------------------------------------
# DocDeshadower lazy loader (isolated import)
# ---------------------------------------------------------------------------
DocDeshadowerModel = None


def _load_docdeshadower():
    global DocDeshadowerModel
    if DocDeshadowerModel is not None:
        return DocDeshadowerModel
    ddr_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'DocDeshadower'))
    if not os.path.isdir(ddr_root):
        raise FileNotFoundError(f"DocDeshadower not found at {ddr_root}")
    # Snapshot current models module
    saved_models = sys.modules.pop('models', None)
    saved_sub = {k: sys.modules.pop(k) for k in list(sys.modules) if k.startswith('models.')}
    sys.path.insert(0, ddr_root)
    try:
        from models.model import Model as _M
        DocDeshadowerModel = _M
    finally:
        sys.path.pop(0)
        if saved_models is not None:
            sys.modules['models'] = saved_models
        for k, v in saved_sub.items():
            sys.modules[k] = v
    return DocDeshadowerModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def build_dataloader(dataset_cfg, split='train', shuffle=True, batch_size=4, img_size=384,
                     text_detector=False):
    data_dir = dataset_cfg['train_dir'] if split == 'train' else dataset_cfg['val_dir']
    if 'lol' in data_dir.lower():
        ds = LOLDataReader(data_dir, inp=dataset_cfg['input'], tar=dataset_cfg['target'],
                           mode='train' if split == 'train' else 'val',
                           img_options={'h': img_size, 'w': img_size})
    elif 'rain100h' in data_dir.lower():
        ds = Rain100HDataReader(data_dir, inp=dataset_cfg['input'], tar=dataset_cfg['target'],
                                mode='train' if split == 'train' else 'val',
                                img_options={'h': img_size, 'w': img_size})
    else:
        ds = get_data(data_dir, dataset_cfg['input'], dataset_cfg['target'],
                      mode='train' if split == 'train' else 'val',
                      img_options={'h': img_size, 'w': img_size},
                      text_detector=text_detector)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=8, pin_memory=True, persistent_workers=True,
                      prefetch_factor=4)


def build_model(name, device, model_variant='v1'):
    if name == 'baseline':
        return ShaDocFormer().to(device), 'baseline'
    elif name == 'textaware':
        m = TextAwareModel(variant=model_variant).to(device)
        return m, 'textaware_v2' if model_variant == 'v2' else 'textaware'
    elif name == 'docdeshadower':
        m = _load_docdeshadower()
        return m().to(device), 'simple'
    elif name == 'bedsr':
        return BEDSRGenerator().to(device), 'simple'
    elif name == 'unet':
        return UNet().to(device), 'simple'
    elif name == 'nafnet':
        return NAFNet().to(device), 'simple'
    elif name == 'shadow_guided':
        return ShadowGuidedNAFNet().to(device), 'shadow_guided'
    elif name == 'shadow_guided_no_sgca':
        return ShadowGuidedNAFNet_NoSGCA().to(device), 'shadow_guided'
    elif name == 'shadow_guided_nafnet_nosgca':
        return ShadowGuidedNAFNet_NoSGCA().to(device), 'shadow_guided'
    elif name == 'shadow_guided_nafnet_concat':
        return ShadowGuidedNAFNet_Concat().to(device), 'shadow_guided'
    elif name == 'shadow_guided_concat':
        return ShadowGuidedNAFNet_Concat().to(device), 'shadow_guided'
    elif name == 'shadow_guided_nafnet_film':
        return ShadowGuidedNAFNet_FiLM().to(device), 'shadow_guided'
    elif name == 'shadow_guided_nafnet_gated':
        return ShadowGuidedNAFNet_Gated().to(device), 'shadow_guided'
    elif name == 'shadow_guided_nafnet_crossattn':
        return ShadowGuidedNAFNet_CrossAttn().to(device), 'shadow_guided'
    elif name == 'shadow_guided_nafnet_asf':
        return ShadowGuidedNAFNet_ASF().to(device), 'shadow_guided'
    elif name == 'shadow_guided_nafnet_large':
        return ShadowGuidedNAFNet_Large().to(device), 'shadow_guided'
    elif name == 'restormer':
        return Restormer().to(device), 'simple'
    elif name == 'shadow_guided_restormer':
        return ShadowGuidedRestormer().to(device), 'shadow_guided'
    elif name == 'shadow_guided_restormer_no_sgca':
        return ShadowGuidedRestormer_NoSGCA().to(device), 'shadow_guided'
    elif name == 'shadow_guided_restormer_concat':
        return ShadowGuidedRestormer_Concat().to(device), 'shadow_guided'
    elif name == 'shadow_guided_restormer_crossattn':
        return ShadowGuidedRestormer_CrossAttn().to(device), 'shadow_guided'
    elif name == 'shadow_guided_restormer_film':
        return ShadowGuidedRestormer_FiLM().to(device), 'shadow_guided'
    elif name == 'shadow_guided_restormer_large':
        return ShadowGuidedRestormer_Large().to(device), 'shadow_guided'
    elif name == 'shadow_guided_restormer_asf':
        return ShadowGuidedRestormer_ASF().to(device), 'shadow_guided'
    elif name == 'shadow_guided_restormer_gated':
        return ShadowGuidedRestormer_Gated().to(device), 'shadow_guided'
    elif name == 'shadow_guided_restormer_gated_large':
        return ShadowGuidedRestormer_GatedLarge().to(device), 'shadow_guided'
    elif name == 'shadow_guided_restormer_gated_noshadow':
        return ShadowGuidedRestormer_Gated_NoShadow().to(device), 'shadow_guided'
    elif name == 'shadow_guided_restormer_gated_dec3only':
        return ShadowGuidedRestormer_Gated_Dec3Only().to(device), 'shadow_guided'
    else:
        raise ValueError(f"Unknown model: {name}")


def train_epoch(model, loader, optimizer, criterion, text_detector, device, scaler, epoch, model_type, grad_accum=1):
    model.train()
    total_loss = 0
    has_precomputed_attn = loader.dataset._use_text_detector if hasattr(loader.dataset, '_use_text_detector') else False
    pbar = tqdm(loader, desc=f"E{epoch}", ncols=100)
    optimizer.zero_grad()
    for i, batch in enumerate(pbar):
        text_attn = None
        if has_precomputed_attn and model_type == 'textaware':
            inp, gray, tar, _, text_attn = [b.to(device) if isinstance(b, torch.Tensor) else b for b in batch]
            text_attn = text_attn.to(device)
        else:
            inp, gray, tar, _ = [b.to(device) if isinstance(b, torch.Tensor) else b for b in batch[:4]]
        inp, gray, tar = inp.to(device) if isinstance(inp, torch.Tensor) else inp, \
                         gray.to(device) if isinstance(gray, torch.Tensor) else gray, \
                         tar.to(device) if isinstance(tar, torch.Tensor) else tar
        with torch.cuda.amp.autocast():
            if model_type == 'textaware':
                if not has_precomputed_attn:
                    text_attn = TextDetector.batch_detect(text_detector, tar, device)
                    text_attn_inp = TextDetector.batch_detect(text_detector, inp, device)
                    text_attn = torch.max(text_attn, text_attn_inp)
                res = model(gray, inp, text_attention=text_attn)
                loss, _ = criterion(res, tar, text_attn)
            elif model_type == 'textaware_v2':
                mser_label = text_attn if has_precomputed_attn else None
                res, text_map = model(gray, inp, return_text_map=True)
                loss, _ = criterion(res, tar, text_map.detach(), text_pred=text_map,
                                    text_target_map=mser_label)
            elif model_type == 'baseline':
                res = model(gray, inp)
                loss = nn.functional.mse_loss(res, tar)
            elif model_type == 'shadow_guided':
                res = model(gray, inp)
                loss = F.l1_loss(res, tar)
            else:
                res = model(inp)
                loss = nn.functional.mse_loss(res, tar)
        loss = loss / grad_accum
        scaler.scale(loss).backward()
        if (i + 1) % grad_accum == 0 or (i + 1) == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        total_loss += loss.item() * grad_accum
        pbar.set_postfix({'loss': f'{loss.item() * grad_accum:.4f}'})
    return {'loss': total_loss / len(loader)}


@torch.no_grad()
def evaluate(model, loader, text_detector, device, model_type):
    model.eval()
    psnr_vals, ssim_vals, rmse_vals, text_psnr_vals = [], [], [], []
    has_precomputed_attn = loader.dataset._use_text_detector if hasattr(loader.dataset, '_use_text_detector') else False
    for batch in tqdm(loader, desc="Eval", ncols=100):
        if has_precomputed_attn:
            inp, gray, tar, _, text_attn = [b.to(device) if isinstance(b, torch.Tensor) else b for b in batch]
            text_attn = text_attn.to(device)
        else:
            inp, gray, tar, _ = [b.to(device) if isinstance(b, torch.Tensor) else b for b in batch[:4]]
            text_attn = None
        inp, tar = inp.to(device) if isinstance(inp, torch.Tensor) else inp, \
                   tar.to(device) if isinstance(tar, torch.Tensor) else tar
        with torch.cuda.amp.autocast():
            if model_type == 'textaware':
                res = model(gray, inp, text_attention=text_attn)
            elif model_type == 'textaware_v2':
                res, text_map = model(gray, inp, return_text_map=True)
                text_attn = F.interpolate(text_map, size=res.shape[2:],
                                          mode='bilinear', align_corners=False)
            elif model_type == 'baseline' or model_type == 'shadow_guided':
                res = model(gray, inp)
            else:
                res = model(inp)
        psnr_vals.append(peak_signal_noise_ratio(res, tar, data_range=1).item())
        ssim_vals.append(structural_similarity_index_measure(res, tar, data_range=1).item())
        rmse_vals.append(mean_squared_error(res * 255, tar * 255, squared=False).item())

        if model_type != 'textaware_v2' and not has_precomputed_attn:
            text_attn = TextDetector.batch_detect(text_detector, tar, device)
        tm = text_attn.unsqueeze(1) if text_attn.dim() == 3 else text_attn
        if tm.dim() == 5:
            tm = tm.squeeze(1)
        if tm.shape[2:] != res.shape[2:]:
            tm = F.interpolate(tm, size=res.shape[2:], mode='bilinear', align_corners=False)
        text_mask = (tm > 0.1).float().expand_as(res)
        text_area = text_mask.sum().item()
        if text_area > 100:
            mse_t = torch.mean((res * text_mask - tar * text_mask) ** 2).item()
            t_psnr = 20 * np.log10(1.0 / np.sqrt(max(mse_t, 1e-10)))
            text_psnr_vals.append(min(t_psnr, 100))

    def avg(lst):
        return round(np.mean(lst), 4) if lst else 0.0

    return {'psnr': avg(psnr_vals), 'ssim': avg(ssim_vals), 'rmse': avg(rmse_vals),
            'text_psnr': avg(text_psnr_vals)}


# ---------------------------------------------------------------------------
# DTRM pretraining
# ---------------------------------------------------------------------------
def pretrain_dtrm(model, train_loader, device, num_epochs=10, lr=1e-3):
    """用 MSER pseudo-labels 预训练 LDTD 文字检测器."""
    print(f"\n{'='*60}")
    print(f"DTRM Pretraining: {num_epochs} epochs")
    print(f"{'='*60}")
    optimizer = optim.Adam(model.refine.text_detector.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)
    criterion = nn.BCELoss()

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"DTRM E{epoch}", ncols=100)
        for batch in pbar:
            if train_loader.dataset._use_text_detector:
                inp, gray, tar, _, mser_label = [b.to(device) if isinstance(b, torch.Tensor) else b for b in batch]
            else:
                continue
            mser_label = mser_label.to(device)
            with torch.no_grad():
                mask = model.mask(gray)
                x_res = torch.cat((inp, mask), dim=1)
            text_pred = model.refine._extract_text_attention(x_res)
            mser_resized = F.interpolate(mser_label, size=text_pred.shape[2:],
                                         mode='bilinear', align_corners=False)
            loss = criterion(text_pred, mser_resized)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        scheduler.step()
        print(f"  DTRM epoch {epoch}/{num_epochs}: avg_loss={total_loss/len(train_loader):.4f}")
    return model


# ---------------------------------------------------------------------------
# Single experiment runner
# ---------------------------------------------------------------------------
def run_experiment(model_name, args, dataset_cfg, device, train_loader, val_loader, text_detector, grad_accum=1):
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    latest_path = os.path.join(output_dir, f'{model_name}_latest.pth')
    progress_file = os.path.join(output_dir, '_progress.txt')

    def write_progress(msg):
        with open(progress_file, 'w') as f:
            f.write(msg)

    model_variant = getattr(args, 'model_variant', 'v1')
    print(f"\n{'='*60}")
    print(f"Experiment: {model_name} | dataset={args.dataset} | {args.res}x{args.res} | bs={args.batch_size} | epochs={args.epochs} | variant={model_variant}")
    print(f"{'='*60}")

    seed_everything(3407)
    model, model_type = build_model(model_name, device, model_variant=model_variant)
    params = sum(p.numel() for p in model.parameters())
    print(f"Params: {params:,} | Type: {model_type}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler()
    if model_type == 'textaware':
        # v1: fixed-weight text-region L1 (text=2.0, bg=1.0 — original v1 weights)
        def v1_loss(pred, target, text_mask):
            if text_mask.shape[2:] != pred.shape[2:]:
                text_mask = F.interpolate(text_mask, size=pred.shape[2:],
                                          mode='bilinear', align_corners=False)
            t_area = text_mask.sum().clamp(min=100.0)
            b_mask = (1.0 - text_mask).clamp(min=0.0)
            b_area = b_mask.sum().clamp(min=100.0)
            l_text = (F.l1_loss(pred, target, reduction='none')
                      * text_mask.expand_as(pred)).sum() / t_area * 2.0
            l_bg = (F.l1_loss(pred, target, reduction='none')
                    * b_mask.expand_as(pred)).sum() / b_area
            total = l_text + l_bg
            return total, {'loss_text': l_text.item(), 'loss_bg': l_bg.item(), 'total': total.item()}
        criterion = v1_loss
    elif model_type == 'textaware_v2':
        dtrm_w = getattr(args, 'dtrm_weight', 0.5)
        fixed_w = getattr(args, 'fixed_adaptive_weight', -1.0)
        criterion = TextAwareLoss(ssim_weight=0.3, vgg_weight=0.1, edge_weight=0.5,
                                  dtrm_weight=dtrm_w, fixed_adaptive_weight=fixed_w)
        criterion = criterion.to(device)
    else:
        criterion = None

    start_epoch = 1
    best_psnr = 0
    history = {'epoch': [], 'psnr': [], 'ssim': [], 'rmse': [], 'text_psnr': []}

    # -- Resume logic --
    if args.resume:
        resume_file = args.resume if os.path.isfile(args.resume) else latest_path
        if os.path.isfile(resume_file):
            print(f"Resuming from {resume_file}")
            ckpt = torch.load(resume_file, map_location=device)
            model.load_state_dict(ckpt['model'])
            optimizer.load_state_dict(ckpt['optimizer'])
            start_epoch = ckpt['epoch'] + 1
            scaler.load_state_dict(ckpt['scaler'])
            history = ckpt.get('history', history)
            best_psnr = ckpt.get('best_psnr', 0)
            print(f"  -> Resuming from epoch {start_epoch}")
        else:
            print(f"  [WARN] Resume file not found: {resume_file}, starting fresh")

    write_progress(f'{model_name}:epoch_{start_epoch - 1}')

    t0 = time.time()
    final_metrics = None
    try:
        for epoch in range(start_epoch, args.epochs + 1):
            train_epoch(model, train_loader, optimizer, criterion, text_detector, device, scaler, epoch, model_type, grad_accum)
            scheduler.step()

            # -- Save LATEST every epoch (crash recovery) --
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scaler': scaler.state_dict(),
                'history': history,
                'best_psnr': best_psnr,
            }, latest_path)

            # -- Evaluate & log every 10 epochs --
            if epoch % 10 == 0 or epoch == 1:
                metrics = evaluate(model, val_loader, text_detector, device, model_type)
                elapsed = (time.time() - t0) / 3600
                print(f"  [{epoch}/{args.epochs}] PSNR={metrics['psnr']:.2f} SSIM={metrics['ssim']:.4f} "
                      f"RMSE={metrics['rmse']:.2f} | TextPSNR={metrics['text_psnr']:.2f} | {elapsed:.1f}h")

                # Save permanent checkpoint
                ckpt_path = os.path.join(output_dir, f'{model_name}_epoch{epoch}.pth')
                torch.save({'epoch': epoch, 'model': model.state_dict(),
                            'optimizer': optimizer.state_dict(), 'scaler': scaler.state_dict(),
                            'metrics': metrics, 'history': history, 'best_psnr': best_psnr,
                            'model_type': model_type, 'args': vars(args)}, ckpt_path)
                print(f"  -> Checkpoint: {ckpt_path}")

                # Track history
                history['epoch'].append(epoch)
                for k in ['psnr', 'ssim', 'rmse', 'text_psnr']:
                    history[k].append(metrics[k])

                if metrics['psnr'] > best_psnr:
                    best_psnr = metrics['psnr']
                    torch.save(model.state_dict(), os.path.join(output_dir, f'{model_name}_best.pth'))

            write_progress(f'{model_name}:epoch_{epoch}')

        # Final eval
        final_metrics = evaluate(model, val_loader, text_detector, device, model_type)
        torch.save(model.state_dict(), os.path.join(output_dir, f'{model_name}_final.pth'))
        torch.save({
            'epoch': args.epochs, 'model': model.state_dict(),
            'optimizer': optimizer.state_dict(), 'scaler': scaler.state_dict(),
            'metrics': final_metrics, 'history': history, 'best_psnr': best_psnr,
        }, latest_path)

        print(f"\n{model_name} Final: PSNR={final_metrics['psnr']:.2f} SSIM={final_metrics['ssim']:.4f} "
              f"RMSE={final_metrics['rmse']:.2f} TextPSNR={final_metrics['text_psnr']:.2f}")
    finally:
        del model, optimizer, scaler
        torch.cuda.empty_cache()
        import gc
        gc.collect()

    return final_metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='all',
                        help='Model name(s): all, or comma-separated: baseline,textaware,docdeshadower,bedsr,unet,nafnet,restormer')
    parser.add_argument('--dataset', type=str, default='rdd', choices=['rdd', 'sd7k', 'synthetic', 'lol', 'rain100h'])
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--res', type=int, default=384)
    parser.add_argument('--output', type=str, default='./experiment_results_rdd/')
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--model_variant', type=str, default='v1', choices=['v1', 'v2'],
                        help='TextAware model version (default v1, v2 uses LDTD + SFT + adaptive loss)')
    parser.add_argument('--pretrain_dtrm_only', action='store_true',
                        help='Pretrain LDTD head only (MSER pseudo-labels), then exit')
    parser.add_argument('--pretrain_dtrm_epochs', type=int, default=10,
                        help='Number of DTRM pretraining epochs')
    parser.add_argument('--dtrm_weight', type=float, default=0.5,
                        help='DTRM loss weight (0=ablation no DTRM)')
    parser.add_argument('--grad_accum', type=int, default=1,
                        help='Gradient accumulation steps')
    parser.add_argument('--fixed_adaptive_weight', type=float, default=-1.0,
                        help='Fixed adaptive text weight (-1=adaptive, >0=fixed)')
    parser.add_argument('--train_dir', type=str, default='',
                        help='Override train directory (for subset experiments)')
    parser.add_argument('--val_dir', type=str, default='',
                        help='Override val directory (for subset experiments)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True

    dataset_cfg = DATASET_CONFIGS[args.dataset].copy()
    if args.train_dir:
        dataset_cfg['train_dir'] = args.train_dir
    if args.val_dir:
        dataset_cfg['val_dir'] = args.val_dir
    print(f"Device: {device} | Dataset: {args.dataset} | Res: {args.res} | BS: {args.batch_size} | Epochs: {args.epochs}")
    print(f"Train: {dataset_cfg['train_dir']} | Val: {dataset_cfg['val_dir']}")
    print(f"AMP: ON | Grad Clip: 1.0")

    train_loader = build_dataloader(dataset_cfg, 'train', True, args.batch_size, args.res)
    val_loader = build_dataloader(dataset_cfg, 'test', False, args.batch_size, args.res)
    print(f"Train samples: {len(train_loader.dataset)} | Val samples: {len(val_loader.dataset)}")

    text_detector = TextDetector(method='opencv')

    models_to_run = ['baseline', 'textaware', 'docdeshadower', 'bedsr', 'unet', 'nafnet', 'restormer'] \
        if args.model == 'all' else [m.strip() for m in args.model.split(',')]

    # Per-model settings: resolution override, batch_size override, text_detector
    MODEL_RES = {'docdeshadower': 384, 'shadow_guided_restormer': 256,
                 'shadow_guided_restormer_no_sgca': 256, 'shadow_guided_restormer_concat': 256,
                 'shadow_guided_restormer_crossattn': 192,
                 'shadow_guided_restormer_film': 384,
                 'shadow_guided_restormer_large': 320,
                'shadow_guided_restormer_gated': 256,
                'shadow_guided_restormer_gated_large': 320,
                'shadow_guided_restormer_gated_noshadow': 256,
                'shadow_guided_restormer_gated_dec3only': 256}
    MODEL_BS = {'nafnet': 8, 'unet': 8, 'docdeshadower': 1, 'bedsr': 4, 'restormer': 2,
                'baseline': 4, 'textaware': 2 if args.model_variant == 'v2' else 4,
                'shadow_guided': 4, 'shadow_guided_no_sgca': 4, 'shadow_guided_concat': 8,
                'shadow_guided_restormer': 1, 'shadow_guided_restormer_no_sgca': 1,
                'shadow_guided_restormer_concat': 1,
                'shadow_guided_restormer_crossattn': 1,
                'shadow_guided_restormer_film': 1,
                'shadow_guided_restormer_large': 1,
                'shadow_guided_restormer_gated': 1,
                'shadow_guided_restormer_gated_large': 1,
                'shadow_guided_restormer_gated_noshadow': 1,
                'shadow_guided_restormer_gated_dec3only': 1}
    # v2 uses gradient accumulation (bs=2 × 2 steps = effective bs=4)
    MODEL_TEXT_DET = {'textaware': (args.model_variant == 'v2')}
    MODEL_GRAD_ACCUM = {'textaware': 2} if args.model_variant == 'v2' else {}

    # -- DTRM pretraining (only for v2) --
    if args.pretrain_dtrm_only:
        dtrm_loader = build_dataloader(dataset_cfg, 'train', True, args.batch_size, args.res, text_detector=True)
        model, _ = build_model('textaware', device, model_variant='v2')
        model = pretrain_dtrm(model, dtrm_loader, device,
                              num_epochs=args.pretrain_dtrm_epochs)
        dtrm_path = os.path.join(args.output, 'dtrm_pretrained.pth')
        torch.save(model.refine.text_detector.state_dict(), dtrm_path)
        print(f"DTRM checkpoint saved to {dtrm_path}")
        return

    all_results = {}
    for model_name in models_to_run:
        try:
            _res = MODEL_RES.get(model_name, args.res)
            _bs = MODEL_BS.get(model_name, args.batch_size)
            _td = MODEL_TEXT_DET.get(model_name, False)
            need_rebuild = (_res != args.res or _bs != args.batch_size or _td)
            _train_loader = build_dataloader(dataset_cfg, 'train', True, _bs, _res, text_detector=_td) \
                if need_rebuild else train_loader
            _val_loader = build_dataloader(dataset_cfg, 'test', False, _bs, _res, text_detector=False) \
                if need_rebuild else val_loader
            _ga = max(MODEL_GRAD_ACCUM.get(model_name, 1), args.grad_accum)
            metrics = run_experiment(model_name, args, dataset_cfg, device, _train_loader, _val_loader, text_detector, grad_accum=_ga)
            all_results[model_name] = metrics
        except Exception as e:
            print(f"\n[ERROR] {model_name}: {e}")
            import traceback
            traceback.print_exc()
            print(f"[WARN] Skipping {model_name}, continuing with next model...")
            torch.cuda.empty_cache()

    if len(all_results) > 1:
        print("\n" + "=" * 60)
        print("COMPARISON SUMMARY")
        print("=" * 60)
        header = f"{'Metric':<15}"
        for name in all_results:
            header += f"{name:>15}"
        print(header)
        for metric_name in ['psnr', 'ssim', 'rmse', 'text_psnr']:
            row = f"{metric_name:<15}"
            for name in all_results:
                row += f"{all_results[name][metric_name]:>15.4f}"
            print(row)

    with open(os.path.join(args.output, 'all_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
