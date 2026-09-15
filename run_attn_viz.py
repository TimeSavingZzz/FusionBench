import os, sys, glob
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, '/mnt/ShaDocFormer-main')
from models.comparison_models import (
    Restormer, MDTA,
    ShadowGuidedRestormer_NoSGCA,
    ShadowGuidedRestormer_FiLM,
    ShadowGuidedRestormer_Gated,
)
from data.data_RGB import get_data

_ATTN_STORE = {}

def _mdta_patched_forward(self, x):
    b, c, h, w = x.shape
    qkv = self.qkv_dwconv(self.qkv(x))
    q, k, v = qkv.chunk(3, dim=1)
    head_dim = c // self.num_heads
    q = q.reshape(b, self.num_heads, head_dim, h * w)
    k = k.reshape(b, self.num_heads, head_dim, h * w)
    v = v.reshape(b, self.num_heads, head_dim, h * w)
    q = F.normalize(q, dim=-1)
    k = F.normalize(k, dim=-1)
    attn = (q @ k.transpose(-2, -1)) * self.temperature
    attn = attn.softmax(dim=-1)
    _ATTN_STORE[id(self)] = (attn.detach().cpu(), (h, w))
    out = attn @ v
    out = out.reshape(b, -1, h, w)
    return self.project_out(out)

def create_figure(all_data, save_path):
    n = len(all_data)
    fig, axes = plt.subplots(2, n + 1, figsize=(4.0 * (n + 1), 5.5),
                             gridspec_kw={'height_ratios': [1, 1]})
    Himg = Wimg = 320
    for col_idx, data in enumerate(all_data):
        shadow_np = data['shadow'].permute(1, 2, 0).cpu().numpy()
        shadow_np = np.clip(shadow_np, 0, 1)
        attn = data['attn']
        N = attn.shape[-1]
        attn_received = attn[0].mean(dim=0).mean(dim=0)
        for d in range(int(N**0.5), 0, -1):
            if N % d == 0:
                Hmap, Wmap = d, N // d
                break
        else:
            Hmap = Wmap = int(N**0.5)
        attn_map = attn_received.reshape(Hmap, Wmap).numpy()
        a_min, a_max = attn_map.min(), attn_map.max()
        if a_max > a_min:
            attn_map = (attn_map - a_min) / (a_max - a_min)
        attn_resized = np.array(Image.fromarray(
            (attn_map * 255).astype(np.uint8)).resize((Wimg, Himg), Image.LANCZOS)) / 255.0
        attn_colored = plt.cm.jet(attn_resized)[:, :, :3]
        overlay = shadow_np * 0.45 + attn_colored * 0.55
        output_np = data['output'].permute(1, 2, 0).cpu().numpy()
        output_np = np.clip(output_np, 0, 1)
        axes[0, 1 + col_idx].imshow(overlay)
        axes[0, 1 + col_idx].set_title(data['label'], fontsize=9.5, fontweight='bold')
        axes[0, 1 + col_idx].axis('off')
        axes[1, 1 + col_idx].imshow(output_np)
        axes[1, 1 + col_idx].axis('off')
    ref_shadow = all_data[0]['shadow'].permute(1, 2, 0).cpu().numpy()
    ref_shadow = np.clip(ref_shadow, 0, 1)
    axes[0, 0].imshow(ref_shadow)
    axes[0, 0].set_title('Shadow Input', fontsize=10, fontweight='bold', color='#D62828')
    axes[0, 0].axis('off')
    ref_target = all_data[0]['target'].permute(1, 2, 0).cpu().numpy()
    ref_target = np.clip(ref_target, 0, 1)
    axes[1, 0].imshow(ref_target)
    axes[1, 0].set_title('Ground Truth', fontsize=10, fontweight='bold', color='#2A9D8F')
    axes[1, 0].axis('off')
    fig.text(0.01, 0.73, 'Attn Overlay', fontsize=11, fontweight='bold', ha='left', va='center', rotation=90)
    fig.text(0.01, 0.23, 'Output', fontsize=11, fontweight='bold', ha='left', va='center', rotation=90)
    fig.suptitle('Decoder Self-Attention Visualization (Level 3) -- SD7K', fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout(pad=0.5, rect=[0.04, 0, 1, 0.97])
    fig.savefig(save_path, dpi=250, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'Saved: {save_path}')

print('Starting...')
device = torch.device('cuda')
data_dir = '/mnt/ShaDocFormer-main/dataset/SD7K/test/'
dataset = get_data(data_dir, 'input', 'target', mode='val', img_options={'h': 320, 'w': 320})
loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)
print(f'SD7K test images: {len(dataset)}')

target_idx = 45
for idx, batch in enumerate(loader):
    if idx == target_idx:
        inp_img, gray_img, tar_img, fname = [b.to(device) if isinstance(b, torch.Tensor) else b for b in batch[:4]]
        print(f'Using sample {idx}')
        break

model_configs = [
    ('Restormer\n(no shadow)', lambda: Restormer(),
     '/mnt/ShaDocFormer-main/experiment_results/restormer_sd7k/restormer_best.pth', False),
    ('Concat\n(No SGCA)', lambda: ShadowGuidedRestormer_NoSGCA(),
     '/mnt/ShaDocFormer-main/experiment_results/nosgca_sd7k/shadow_guided_restormer_no_sgca_best.pth', True),
    ('FiLM\n(SGFM)', lambda: ShadowGuidedRestormer_FiLM(),
     '/mnt/ShaDocFormer-main/experiment_results/sgfm_sd7k/shadow_guided_restormer_film_best.pth', True),
    ('Gated\n(SGGF, ours)', lambda: ShadowGuidedRestormer_Gated(),
     '/mnt/ShaDocFormer-main/experiment_results/sggf_sd7k/shadow_guided_restormer_gated_best.pth', True),
]

all_data = []
for label, builder, ckpt_path, use_shadow in model_configs:
    print(f'Model: {label.replace(chr(10), " ")}')
    if not os.path.exists(ckpt_path):
        print(f'  SKIP: no checkpoint')
        continue
    model = builder().to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get('model', ckpt)
    model.load_state_dict(state, strict=False)
    model.eval()
    _ATTN_STORE.clear()
    patched = 0
    for module in model.modules():
        if isinstance(module, MDTA):
            module.forward = _mdta_patched_forward.__get__(module, MDTA)
            patched += 1
    print(f'  Patched {patched} MDTA')
    with torch.no_grad():
        if use_shadow:
            out = model(gray_img, inp_img)
        else:
            out = model(inp_img)
    attn_list = list(_ATTN_STORE.items())
    print(f'  Captured {len(attn_list)} attn maps')
    if len(attn_list) == 0:
        continue
    dec3_idx = len(attn_list) * 2 // 3
    layer_id, (attn_tensor, spatial) = attn_list[dec3_idx]
    all_data.append({
        'label': label,
        'shadow': inp_img[0].cpu(),
        'output': out[0].cpu().clamp(0, 1),
        'target': tar_img[0].cpu(),
        'attn': attn_tensor,
        'spatial': spatial,
    })

if len(all_data) >= 2:
    output_dir = '/mnt/ShaDocFormer-main/attention_figures_sd7k/'
    os.makedirs(output_dir, exist_ok=True)
    create_figure(all_data, os.path.join(output_dir, 'fig_attention_viz.pdf'))
    create_figure(all_data, os.path.join(output_dir, 'fig_attention_viz.png'))
    print('Done!')
else:
    print(f'ERROR: Only {len(all_data)} models loaded')
