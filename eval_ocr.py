"""OCR evaluation at 512x512 for all models on SD7K."""
import sys, os
sys.path.insert(0, '/mnt/ShaDocFormer-main')
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import easyocr

from train_compare_models import build_model
from data.dataset_RGB import DataReader

device = torch.device('cuda')
RES = 512

MODEL_CH = [
    ('Restormer',   'restormer', 'experiment_results/restormer_sd7k/restormer_best.pth'),
    ('No SGCA',     'shadow_guided_restormer_no_sgca', 'experiment_results/nosgca_sd7k/shadow_guided_restormer_no_sgca_best.pth'),
    ('CrossAttn',   'shadow_guided_restormer_crossattn', 'experiment_results/sgcr_sd7k/shadow_guided_restormer_crossattn_best.pth'),
    ('FiLM',        'shadow_guided_restormer_film', 'experiment_results/sgfm_sd7k/shadow_guided_restormer_film_best.pth'),
    ('Large',       'shadow_guided_restormer_large', 'experiment_results/sglarge_sd7k/shadow_guided_restormer_large_best.pth'),
    ('Gated',       'shadow_guided_restormer_gated', 'experiment_results/sggf_sd7k/shadow_guided_restormer_gated_best.pth'),
    ('GatedLarge',  'shadow_guided_restormer_gated_large', 'experiment_results/sggf_large_sd7k/shadow_guided_restormer_gated_large_best.pth'),
]

ocr = easyocr.Reader(['en'], gpu=True)

def ocr_chars(img_np):
    res = ocr.readtext(img_np, detail=0)
    return sum(len(r) for r in res)

def run_ocr(display_name, model_name, ck_path):
    print(f"\n=== {display_name} (res={RES}) ===")
    model, model_type = build_model(model_name, device)
    ck = torch.load(ck_path, map_location='cuda')
    model.load_state_dict(ck)
    model.eval()

    dset = DataReader('./dataset/SD7K/test/', 'input', 'target', mode='test', ori=False,
                      img_options={'h': RES, 'w': RES})
    loader = DataLoader(dset, batch_size=1, shuffle=False, num_workers=0)

    results = {'ocr_gt': [], 'ocr_inp': [], 'ocr_out': []}
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"OCR {display_name}"):
            inp, gray, tar, _fname = batch
            inp, gray, tar = inp.to(device), gray.to(device), tar.to(device)

            if model_type in ('simple', 'baseline', 'textaware', 'textaware_v2'):
                out = model(inp)
            else:
                out = model(gray, inp)

            inp_np = (inp[0].permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
            out_np = (out[0].permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
            tar_np = (tar[0].permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)

            try:
                results['ocr_gt'].append(ocr_chars(tar_np))
                results['ocr_inp'].append(ocr_chars(inp_np))
                results['ocr_out'].append(ocr_chars(out_np))
            except Exception:
                pass

    ocr_gt = np.mean(results['ocr_gt']) if results['ocr_gt'] else 0
    ocr_inp = np.mean(results['ocr_inp']) if results['ocr_inp'] else 0
    ocr_out = np.mean(results['ocr_out']) if results['ocr_out'] else 0
    recovery = ocr_out / max(ocr_gt, 1) * 100

    print(f"  OCR chars: Inp={ocr_inp:.1f}, Out={ocr_out:.1f}, GT={ocr_gt:.1f}")
    print(f"  Recovery Rate: {recovery:.1f}%")
    return {'ocr_gt': ocr_gt, 'ocr_inp': ocr_inp, 'ocr_out': ocr_out, 'recovery': recovery}

print("=" * 50 + "\nSD7K OCR Evaluation @ 512x512 (GPU OCR)\n" + "=" * 50)
all_r = {}
for disp_name, model_name, ck in MODEL_CH:
    all_r[disp_name] = run_ocr(disp_name, model_name, ck)

print("\n\n=== Summary ===")
print(f"{'Model':<12} {'OCR Inp':>8} {'OCR Out':>8} {'OCR GT':>8} {'Recovery':>10}")
print("-" * 50)
for name, r in all_r.items():
    print(f"{name:<12} {r['ocr_inp']:>8.1f} {r['ocr_out']:>8.1f} {r['ocr_gt']:>8.1f} {r['recovery']:>8.1f}%")
print("Done.")
