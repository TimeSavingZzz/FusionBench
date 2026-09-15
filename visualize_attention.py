"""
Attention Map Visualization for Restormer Decoder (Paper Experiment)
====================================================================
Tests hypothesis: Transformer self-attention already handles cross-modal
interaction — decoder attention maps should show focus on shadow regions
even WITHOUT explicit fusion modules.

Compares: Restormer (no shadow), No SGCA (concat), FiLM, Gated

Usage:
  cd /mnt/ShaDocFormer-main && python visualize_attention.py
"""

import os, sys, argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.comparison_models import (
    Restormer, MDTA,
    ShadowGuidedRestormer_NoSGCA,
    ShadowGuidedRestormer_FiLM,
    ShadowGuidedRestormer_Gated,
)
from data.data_RGB import get_data

# ── Global attention storage ─────────────────────────────────
_ATTN_STORE = {}  # {layer_name: attn_tensor}


def _mdta_patched_forward(self, x):
    """Patched MDTA.forward: identical to original, but stores attention map."""
    b, c, h, w = x.shape
    qkv = self.qkv_dwconv(self.qkv(x))
    q, k, v = qkv.chunk(3, dim=1)

    # Reshape manually (avoids einops dependency issues)
    head_dim = c // self.num_heads
    q = q.reshape(b, self.num_heads, head_dim, h * w)
    k = k.reshape(b, self.num_heads, head_dim, h * w)
    v = v.reshape(b, self.num_heads, head_dim, h * w)

    q = F.normalize(q, dim=-1)
    k = F.normalize(k, dim=-1)
    attn = (q @ k.transpose(-2, -1)) * self.temperature  # [B, H, N, N]
    attn = attn.softmax(dim=-1)

    # Store attention + spatial dims
    layer_id = id(self)
    _ATTN_STORE[layer_id] = (attn.detach().cpu(), (h, w))

    out = attn @ v
    out = out.reshape(b, -1, h, w)
    return self.project_out(out)


def patch_all_mdta(model):
    """Monkey-patch all MDTA modules to capture attention."""
    patched = 0
    for module in model.modules():
        if isinstance(module, MDTA):
            module.forward = _mdta_patched_forward.__get__(module, MDTA)
            patched += 1
    return patched


def get_mdta_names(model):
    """Build {id(module): name} mapping."""
    mapping = {}
    for name, module in model.named_modules():
        if isinstance(module, MDTA):
            mapping[id(module)] = name
    return mapping


# ── Visualization ───────────────────────────────────────────
def render_attention_overlay(img_shadow, img_output, attn_weights, attn_spatial, title, save_path):
    """
    img_shadow: [3, H, W] tensor (0-1 range)
    img_output: [3, H, W] tensor (model output, 0-1 range)
    attn_weights: [B, heads, N, N] attention matrix from one decoder layer
    attn_spatial: (H, W) tuple of the feature map
    """
    b, n_heads, N, _ = attn_weights.shape
    Hmap, Wmap = attn_spatial
    if Hmap * Wmap != N:
        # Fallback: find factor pair closest to square
        for d in range(int(N**0.5), 0, -1):
            if N % d == 0:
                Hmap, Wmap = d, N // d
                break
        else:
            Hmap = Wmap = int(N**0.5)

    # Mean attention received per spatial position (averaged over all heads & queries)
    attn_received = attn_weights[0].mean(dim=0).mean(dim=0)  # [N]
    attn_map = attn_received.reshape(Hmap, Wmap).numpy()

    # Normalize
    a_min, a_max = attn_map.min(), attn_map.max()
    if a_max > a_min:
        attn_map = (attn_map - a_min) / (a_max - a_min)
    else:
        attn_map = np.ones_like(attn_map) * 0.5

    # Resize attention map to match image resolution
    H_img, W_img = img_shadow.shape[-2], img_shadow.shape[-1]
    attn_resized = np.array(Image.fromarray((attn_map * 255).astype(np.uint8)).resize(
        (W_img, H_img), Image.LANCZOS)) / 255.0

    # Convert images to numpy
    shadow_np = img_shadow.permute(1, 2, 0).cpu().numpy()
    shadow_np = np.clip(shadow_np, 0, 1)
    output_np = img_output.permute(1, 2, 0).cpu().numpy()
    output_np = np.clip(output_np, 0, 1)

    # Create overlays
    cmap = plt.cm.jet
    attn_colored = cmap(attn_resized)[:, :, :3]

    # Overlay on shadow image (alpha blend)
    overlay_shadow = shadow_np * 0.55 + attn_colored * 0.45

    # Overlay on output image
    overlay_output = output_np * 0.55 + attn_colored * 0.45

    fig, axes = plt.subplots(1, 5, figsize=(25, 5.5))

    axes[0].imshow(shadow_np)
    axes[0].set_title('(a) Shadow Input', fontsize=11, fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(attn_resized, cmap='jet')
    axes[1].set_title('(b) Attention Map', fontsize=11, fontweight='bold')
    axes[1].axis('off')

    axes[2].imshow(overlay_shadow)
    axes[2].set_title('(c) Attn → Input', fontsize=11, fontweight='bold')
    axes[2].axis('off')

    axes[3].imshow(output_np)
    axes[3].set_title('(d) Restored Output', fontsize=11, fontweight='bold')
    axes[3].axis('off')

    axes[4].imshow(overlay_output)
    axes[4].set_title('(e) Attn → Output', fontsize=11, fontweight='bold')
    axes[4].axis('off')

    fig.suptitle(title, fontsize=13, fontweight='bold', y=0.98)
    plt.tight_layout(pad=1.0)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def render_comparison_figure(all_model_data, save_path, layer_name):
    """
    Create a comprehensive comparison figure across models for one decoder layer.
    all_model_data: list of dicts with keys: name, shadow_img, attn_map, output_img
    """
    n_models = len(all_model_data)
    fig, axes = plt.subplots(n_models, 4, figsize=(20, 5 * n_models))

    if n_models == 1:
        axes = axes.reshape(1, -1)

    for row, data in enumerate(all_model_data):
        shadow_np = data['shadow'].permute(1, 2, 0).cpu().numpy()
        shadow_np = np.clip(shadow_np, 0, 1)
        output_np = data['output'].permute(1, 2, 0).cpu().numpy()
        output_np = np.clip(output_np, 0, 1)

        # Attention map
        attn = data['attn']
        Hmap, Wmap = data.get('spatial', (None, None))
        if Hmap is None or Hmap * Wmap != attn.shape[-1]:
            N = attn.shape[-1]
            for d in range(int(N**0.5), 0, -1):
                if N % d == 0:
                    Hmap, Wmap = d, N // d
                    break
            else:
                Hmap = Wmap = int(N**0.5)
        attn_received = attn[0].mean(dim=0).mean(dim=0).reshape(Hmap, Wmap).numpy()
        a_min, a_max = attn_received.min(), attn_received.max()
        if a_max > a_min:
            attn_received = (attn_received - a_min) / (a_max - a_min)
        else:
            attn_received = np.ones_like(attn_received) * 0.5

        Himg, Wimg = shadow_np.shape[0], shadow_np.shape[1]
        attn_resized = np.array(Image.fromarray((attn_received * 255).astype(np.uint8)).resize(
            (Wimg, Himg), Image.LANCZOS)) / 255.0
        attn_colored = plt.cm.jet(attn_resized)[:, :, :3]
        overlay = shadow_np * 0.5 + attn_colored * 0.5

        axes[row, 0].imshow(shadow_np)
        axes[row, 0].set_ylabel(data['name'], fontsize=11, fontweight='bold')
        if row == 0:
            axes[row, 0].set_title('Shadow Input', fontsize=11, fontweight='bold')
        axes[row, 0].axis('off')

        axes[row, 1].imshow(attn_resized, cmap='jet')
        if row == 0:
            axes[row, 1].set_title(f'Attention ({layer_name})', fontsize=11, fontweight='bold')
        axes[row, 1].axis('off')

        axes[row, 2].imshow(overlay)
        if row == 0:
            axes[row, 2].set_title('Overlay', fontsize=11, fontweight='bold')
        axes[row, 2].axis('off')

        axes[row, 3].imshow(output_np)
        if row == 0:
            axes[row, 3].set_title('Restored Output', fontsize=11, fontweight='bold')
        axes[row, 3].axis('off')

    fig.suptitle(f'Decoder Attention Comparison — {layer_name}',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(pad=1.5)
    fig.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)


# ── Analyze one model ───────────────────────────────────────
def analyze_model(model, model_name, dataloader, device, output_dir, num_samples=5):
    """Extract and visualize attention maps for a model."""
    global _ATTN_STORE

    model.eval()
    id_to_name = get_mdta_names(model)
    n_patched = patch_all_mdta(model)
    print(f"  Patched {n_patched} MDTA modules")

    # Identify decoder-level MDTA modules
    decoder_layers = [n for n in id_to_name.values()
                      if 'decoder' in n and 'attn' in n]
    latent_layers = [n for n in id_to_name.values()
                     if 'latent' in n and 'attn' in n]
    refinement_layers = [n for n in id_to_name.values()
                         if 'refinement' in n and 'attn' in n]

    print(f"  Decoder attn layers: {len(decoder_layers)}")
    print(f"  Latent attn layers: {len(latent_layers)}")
    print(f"  Refinement attn layers: {len(refinement_layers)}")
    print(f"  All MDTA names: {list(id_to_name.values())[:5]}...")

    all_results = []  # [{name, shadow, output, attn_by_layer}]

    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            if idx >= num_samples:
                break

            inp_img, gray_img, tar_img, fname = [b.to(device) if isinstance(b, torch.Tensor) else b
                                                   for b in batch[:4]]

            _ATTN_STORE.clear()

            # Forward — baseline Restormer takes RGB input; shadow-guided takes (gray, shadow)
            if 'no fusion' in model_name.lower() or 'baseline' in model_name.lower():
                out = model(inp_img)
            else:
                out = model(gray_img, inp_img)

            # Map stored attention IDs to names
            sample_attns = {}
            sample_spatials = {}
            for layer_id, (attn_tensor, spatial) in _ATTN_STORE.items():
                name = id_to_name.get(layer_id, f"unknown_{layer_id}")
                sample_attns[name] = attn_tensor
                sample_spatials[name] = spatial

            all_results.append({
                'shadow': inp_img[0].cpu(),
                'output': out[0].cpu().clamp(0, 1),
                'target': tar_img[0].cpu(),
                'attns': sample_attns,
                'spatials': sample_spatials,
                'fname': fname[0] if isinstance(fname, (list, tuple)) else str(fname),
            })

    # Generate single-model visualizations
    model_out_dir = os.path.join(output_dir, model_name)
    os.makedirs(model_out_dir, exist_ok=True)

    # Focus on representative decoder layers
    target_layers = []
    for pattern in ['decoder_level3', 'decoder_level2', 'decoder_level1', 'latent', 'refinement']:
        for name in sorted(id_to_name.values()):
            if pattern in name and 'attn' in name and name not in target_layers:
                target_layers.append(name)
                break

    print(f"  Target layers for viz: {target_layers}")

    for sample_idx, result in enumerate(all_results):
        for layer_name in target_layers:
            if layer_name in result['attns']:
                attn = result['attns'][layer_name]
                spatial = result['spatials'].get(layer_name, (8, 8))
                save_name = f'sample{sample_idx:02d}_{layer_name.replace(".","_")}.png'
                save_path = os.path.join(model_out_dir, save_name)
                title = f'{model_name} | {layer_name} | Sample {sample_idx}'
                render_attention_overlay(result['shadow'], result['output'],
                                         attn, spatial, title, save_path)

    print(f"  Saved {len(all_results)} samples × {len(target_layers)} layers to {model_out_dir}")
    return all_results, target_layers


# ── Main ────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, default='./attention_figures/')
    parser.add_argument('--data_dir', type=str, default='./dataset/SD7K/test/')
    parser.add_argument('--num_samples', type=int, default=3)
    parser.add_argument('--res', type=int, default=320)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Load test data
    dataset = get_data(args.data_dir, 'input', 'target', mode='val',
                       img_options={'h': args.res, 'w': args.res})
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False,
                                          num_workers=4)

    print(f"Test images: {len(dataset)}")

    model_configs = [
        {
            'name': 'Restormer (no fusion)',
            'builder': lambda: Restormer(),
            'ckpt': './experiment_results/restormer_sd7k/restormer_best.pth',
        },
        {
            'name': 'Concat',
            'builder': lambda: ShadowGuidedRestormer_NoSGCA(),
            'ckpt': './experiment_results/nosgca_sd7k/shadow_guided_restormer_no_sgca_best.pth',
        },
        {
            'name': 'FiLM',
            'builder': lambda: ShadowGuidedRestormer_FiLM(),
            'ckpt': './experiment_results/sgfm_sd7k/shadow_guided_restormer_film_best.pth',
        },
        {
            'name': 'Gated',
            'builder': lambda: ShadowGuidedRestormer_Gated(),
            'ckpt': './experiment_results/sggf_sd7k/shadow_guided_restormer_gated_best.pth',
        },
    ]

    all_model_results = {}
    common_layers = None

    for cfg in model_configs:
        ckpt_path = cfg['ckpt']
        if not os.path.exists(ckpt_path):
            print(f"SKIP {cfg['name']}: no checkpoint at {ckpt_path}")
            continue

        print(f"\n{'─'*50}")
        print(f"Model: {cfg['name']}")

        model = cfg['builder']().to(device)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        state = ckpt.get('model', ckpt)
        model.load_state_dict(state, strict=False)

        results, target_layers = analyze_model(model, cfg['name'], loader, device,
                                                args.output, args.num_samples)
        all_model_results[cfg['name']] = results

        if common_layers is None:
            common_layers = target_layers

    # ── Cross-model comparison figure ────────────────────────
    if len(all_model_results) >= 2:
        print(f"\n{'─'*50}")
        print("Generating cross-model comparison figures...")

        comparison_dir = os.path.join(args.output, 'comparison')
        os.makedirs(comparison_dir, exist_ok=True)

        n_samples = min(args.num_samples, min(len(v) for v in all_model_results.values()))

        for sample_idx in range(n_samples):
            for layer_name in (common_layers or []):
                # Gather this layer's data across all models
                comparison_data = []
                for model_name, results in all_model_results.items():
                    if sample_idx < len(results):
                        r = results[sample_idx]
                        if layer_name in r['attns']:
                            comparison_data.append({
                                'name': model_name,
                                'shadow': r['shadow'],
                                'output': r['output'],
                                'attn': r['attns'][layer_name],
                                'spatial': r.get('spatials', {}).get(layer_name),
                            })

                if len(comparison_data) >= 2:
                    save_path = os.path.join(
                        comparison_dir,
                        f'compare_sample{sample_idx:02d}_{layer_name.replace(".","_")}.png'
                    )
                    render_comparison_figure(comparison_data, save_path, layer_name)

    print(f"\nDone! Figures saved to {args.output}")


if __name__ == '__main__':
    main()
