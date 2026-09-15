import os, sys, json
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, '/mnt/ShaDocFormer-main')
from models.fusion_models import ShadowGuidedNAFNet_ASF, ShadowGuidedRestormer_ASF
from data.data_RGB import get_data

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_ckpt(model, path):
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    state = ckpt.get('model', ckpt)
    model.load_state_dict(state, strict=False)
    model.to(DEVICE).eval()
    return model

def collect_alpha(model, loader, alpha_hook, n_samples=60):
    """Collect per-sample alpha values from an ASF module."""
    alphas = []
    captured = {}
    mod = dict(model.named_modules())[alpha_hook]
    handle = mod.register_forward_hook(
        lambda m, a, o: captured.setdefault('alpha', o.detach()))
    with torch.no_grad():
        for idx, batch in enumerate(loader):
            inp_img, gray_img, tar_img = [b.to(DEVICE) if isinstance(b, torch.Tensor) else b
                                           for b in batch[:3]]
            captured.clear()
            model(gray_img, inp_img)
            a = captured['alpha']  # [B,1,1,1]
            alphas.append(a.cpu().flatten())
            if sum(len(x) for x in alphas) >= n_samples:
                break
    handle.remove()
    return torch.cat(alphas).numpy()

def get_loader(data_dir, inp, tar, res=256):
    ds = get_data(data_dir, inp, tar, mode='val',
                  img_options={'h': res, 'w': res})
    return torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=2), len(ds)

CFG = [
    # (name, model_cls, ckpt, data_dir, input, target, alpha_module)
    ('Restormer-ASF SD7K', ShadowGuidedRestormer_ASF,
     'experiment_results/restormer_asf_sd7k/shadow_guided_restormer_asf_best.pth',
     './dataset/SD7K/test/', 'input', 'target', 'asf_dec3.alpha_net'),
    ('Restormer-ASF RDD', ShadowGuidedRestormer_ASF,
     'experiment_results/restormer_asf_rdd/shadow_guided_restormer_asf_best.pth',
     './dataset/RDD/test/', 'img', 'back_gt', 'asf_dec3.alpha_net'),
    ('NAFNet-ASF SD7K', ShadowGuidedNAFNet_ASF,
     'experiment_results/nafnet_asf_sd7k/shadow_guided_nafnet_asf_best.pth',
     './dataset/SD7K/test/', 'input', 'target', 'asf_dec1.alpha_net'),
    ('NAFNet-ASF RDD', ShadowGuidedNAFNet_ASF,
     'experiment_results/nafnet_asf_rdd/shadow_guided_nafnet_asf_best.pth',
     './dataset/RDD/test/', 'img', 'back_gt', 'asf_dec1.alpha_net'),
]

def main():
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5), sharex=True)
    summary = {}
    for i, (name, cls, ckpt, data_dir, inp, tar, alpha_module) in enumerate(CFG):
        try:
            model = cls()
            load_ckpt(model, ckpt)
            loader, n_img = get_loader(data_dir, inp, tar)
            alphas = collect_alpha(model, loader, alpha_module)
            summary[name] = {
                'mean': round(float(alphas.mean()), 4),
                'median': round(float(np.median(alphas)), 4),
                'std': round(float(alphas.std()), 4),
                'pct_lt_0.5': round(float((alphas < 0.5).mean()), 4),
                'n': int(len(alphas)),
            }
            print(name, summary[name])
            ax = axes[i]
            ax.hist(alphas, bins=30, range=(0, 1), color='#4C72B0', edgecolor='white', alpha=0.85)
            ax.axvline(0.5, color='red', ls='--', lw=1)
            ax.set_title('%s\nmean=%.3f  med=%.3f' % (name, alphas.mean(), np.median(alphas)),
                         fontsize=10)
            ax.set_xlim(0, 1)
            if i == 0:
                ax.set_ylabel('count')
            ax.set_xlabel('alpha (FiLM weight)')
        except Exception as e:
            print(name, 'FAILED:', type(e).__name__, str(e)[:200])

    fig.suptitle('ASF mixing weight alpha: synthetic (SD7K) vs real (RDD)', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig('fig_asf_alpha_hist.png', dpi=200)
    with open('asf_alpha_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print('Saved fig_asf_alpha_hist.png and asf_alpha_summary.json')

if __name__ == '__main__':
    main()
