"""LPIPS perceptual metric evaluation for all SD7K models."""
import sys, os, json
sys.path.insert(0, '/mnt/ShaDocFormer-main')
import torch
import lpips
from torch.utils.data import DataLoader
from tqdm import tqdm

from train_compare_models import build_model
from data.dataset_RGB import DataReader

device = torch.device('cuda')
RES = 320
loss_fn = lpips.LPIPS(net='alex').to(device)

MODELS = [
    ('Restormer',   'restormer', 'experiment_results/restormer_sd7k/restormer_best.pth'),
    ('No SGCA',     'shadow_guided_restormer_no_sgca', 'experiment_results/nosgca_sd7k/shadow_guided_restormer_no_sgca_best.pth'),
    ('CrossAttn',   'shadow_guided_restormer_crossattn', 'experiment_results/sgcr_sd7k/shadow_guided_restormer_crossattn_best.pth'),
    ('FiLM',        'shadow_guided_restormer_film', 'experiment_results/sgfm_sd7k/shadow_guided_restormer_film_best.pth'),
    ('Gated',       'shadow_guided_restormer_gated', 'experiment_results/sggf_sd7k/shadow_guided_restormer_gated_best.pth'),
    ('GatedLarge',  'shadow_guided_restormer_gated_large', 'experiment_results/sggf_large_sd7k/shadow_guided_restormer_gated_large_best.pth'),
    ('Large',       'shadow_guided_restormer_large', 'experiment_results/sglarge_sd7k/shadow_guided_restormer_large_best.pth'),
]

dset = DataReader('./dataset/SD7K/test/', 'input', 'target', mode='test', ori=False,
                  img_options={'h': RES, 'w': RES})
loader = DataLoader(dset, batch_size=1, shuffle=False, num_workers=0)
print('SD7K test: %d images' % len(dset))

results = {}
for disp_name, model_name, ck_path in MODELS:
    print('\n=== %s ===' % disp_name)
    try:
        model, model_type = build_model(model_name, device)
        ck = torch.load(ck_path, map_location='cuda')
        model.load_state_dict(ck)
        model.eval()
        lpips_sum, n = 0, 0
        with torch.no_grad():
            for batch in tqdm(loader, desc='LPIPS ' + disp_name):
                inp, gray, tar, _ = batch
                inp, gray, tar = inp.to(device), gray.to(device), tar.to(device)
                if model_type in ('simple', 'baseline',):
                    out = model(inp)
                else:
                    out = model(gray, inp)
                lpips_sum += loss_fn(out, tar).item()
                n += 1
        results[disp_name] = round(lpips_sum/n, 6)
        print('  LPIPS=%.6f' % results[disp_name])
        torch.cuda.empty_cache()
    except Exception as e:
        print('  SKIP: %s' % str(e))
        torch.cuda.empty_cache()

print('\n\n=== LPIPS Summary (lower is better) ===')
for name, val in sorted(results.items(), key=lambda x: x[1]):
    print('  %-14s %.6f' % (name, val))

with open('lpips_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print('\nSaved to lpips_results.json')
