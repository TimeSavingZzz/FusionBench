"""Cross-dataset: SD7K-trained models evaluated on RDD test set."""
import sys, os, json
sys.path.insert(0, '/mnt/ShaDocFormer-main')
import torch, numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchmetrics.functional import peak_signal_noise_ratio, structural_similarity_index_measure

from train_compare_models import build_model
from data.dataset_RGB import DataReader

device = torch.device('cuda')
RES = 320

MODELS = [
    ('Restormer',   'restormer', 'experiment_results/restormer_sd7k/restormer_best.pth'),
    ('No SGCA',     'shadow_guided_restormer_no_sgca', 'experiment_results/nosgca_sd7k/shadow_guided_restormer_no_sgca_best.pth'),
    ('FiLM',        'shadow_guided_restormer_film', 'experiment_results/sgfm_sd7k/shadow_guided_restormer_film_best.pth'),
    ('Gated',       'shadow_guided_restormer_gated', 'experiment_results/sggf_sd7k/shadow_guided_restormer_gated_best.pth'),
    ('GatedLarge',  'shadow_guided_restormer_gated_large', 'experiment_results/sggf_large_sd7k/shadow_guided_restormer_gated_large_best.pth'),
    ('Large',       'shadow_guided_restormer_large', 'experiment_results/sglarge_sd7k/shadow_guided_restormer_large_best.pth'),
]

dset = DataReader('./dataset/RDD/test/', 'img', 'gt', mode='test', ori=False,
                  img_options={'h': RES, 'w': RES})
loader = DataLoader(dset, batch_size=1, shuffle=False, num_workers=0)
print('RDD test set: %d images @ %dx%d' % (len(dset), RES, RES))

results = {}
for disp_name, model_name, ck_path in MODELS:
    print('\n=== %s -> RDD (res=%d) ===' % (disp_name, RES))
    try:
        model, model_type = build_model(model_name, device)
        ck = torch.load(ck_path, map_location='cuda')
        model.load_state_dict(ck)
        model.eval()

        psnr_sum, ssim_sum, n = 0, 0, 0
        with torch.no_grad():
            for batch in tqdm(loader, desc='Eval ' + disp_name):
                inp, gray, tar, _ = batch
                inp, gray, tar = inp.to(device), gray.to(device), tar.to(device)
                if model_type in ('simple', 'baseline',):
                    out = model(inp)
                else:
                    out = model(gray, inp)
                psnr_sum += peak_signal_noise_ratio(out, tar, data_range=1).item()
                ssim_sum += structural_similarity_index_measure(out, tar, data_range=1).item()
                n += 1
        results[disp_name] = {'psnr': round(psnr_sum/n, 4), 'ssim': round(ssim_sum/n, 4), 'n': n}
        print('  PSNR=%.2f, SSIM=%.4f' % (results[disp_name]['psnr'], results[disp_name]['ssim']))
        torch.cuda.empty_cache()
    except Exception as e:
        print('  SKIP %s: %s' % (disp_name, str(e)))
        torch.cuda.empty_cache()

with open('/mnt/ShaDocFormer-main/cross_dataset_sd7k_to_rdd.json', 'w') as f:
    json.dump(results, f, indent=2)
print('\nDone. Saved to cross_dataset_sd7k_to_rdd.json')
