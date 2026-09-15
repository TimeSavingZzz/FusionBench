"""Cross-dataset evaluation: RDD-trained models evaluated on SD7K test set."""
import os, sys, json, warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchmetrics.functional import peak_signal_noise_ratio, structural_similarity_index_measure
from torchmetrics.functional.regression import mean_squared_error
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import Model as ShaDocFormer
from models.comparison_models import (
    BEDSRGenerator, UNet, NAFNet, Restormer,
    ShadowGuidedNAFNet, ShadowGuidedNAFNet_NoSGCA, ShadowGuidedNAFNet_Concat,
    ShadowGuidedRestormer, ShadowGuidedRestormer_NoSGCA, ShadowGuidedRestormer_Concat,
)
from data.data_RGB import get_data

CKPT_DIR = './experiment_results_rdd/full_200ep'
SD7K_TEST = './dataset/SD7K/lkljty___ShadowDocument7K/test'
IMG_SIZE = 384
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MODELS = {
    'baseline': {'cls': ShaDocFormer, 'ckpt': 'baseline_best.pth', 'type': 'baseline'},
    'nafnet': {'cls': NAFNet, 'ckpt': 'nafnet_best.pth', 'type': 'simple'},
    'unet': {'cls': UNet, 'ckpt': 'unet_best.pth', 'type': 'simple'},
    'bedsr': {'cls': BEDSRGenerator, 'ckpt': 'bedsr_best.pth', 'type': 'simple'},
    'restormer': {'cls': Restormer, 'ckpt': 'restormer_best.pth', 'type': 'simple'},
    'shadow_guided': {'cls': ShadowGuidedNAFNet, 'ckpt': 'shadow_guided_200ep_best.pth', 'type': 'shadow_guided'},
    'shadow_guided_no_sgca': {'cls': ShadowGuidedNAFNet_NoSGCA, 'ckpt': 'shadow_guided_no_sgca_200ep_best.pth', 'type': 'shadow_guided'},
    'shadow_guided_concat': {'cls': ShadowGuidedNAFNet_Concat, 'ckpt': 'shadow_guided_concat_200ep_best.pth', 'type': 'shadow_guided'},
    'shadow_guided_restormer': {'cls': ShadowGuidedRestormer, 'ckpt': 'shadow_guided_restormer_best.pth', 'type': 'shadow_guided'},
    'shadow_guided_restormer_no_sgca': {'cls': ShadowGuidedRestormer_NoSGCA, 'ckpt': 'shadow_guided_restormer_no_sgca_best.pth', 'type': 'shadow_guided'},
    'shadow_guided_restormer_concat': {'cls': ShadowGuidedRestormer_Concat, 'ckpt': 'shadow_guided_restormer_concat_best.pth', 'type': 'shadow_guided'},
}


@torch.no_grad()
def evaluate_model(name, cfg):
    ckpt_path = os.path.join(CKPT_DIR, cfg['ckpt'])
    if not os.path.exists(ckpt_path):
        print(f'  SKIP: checkpoint not found: {ckpt_path}')
        return None

    model = cfg['cls']().to(DEVICE)
    state = torch.load(ckpt_path, map_location=DEVICE)
    if 'model' in state:
        state = state['model']
    model.load_state_dict(state, strict=False)
    model.eval()

    ds = get_data(SD7K_TEST, 'input', 'target', mode='val',
                  img_options={'h': IMG_SIZE, 'w': IMG_SIZE})
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    psnr_vals, ssim_vals, rmse_vals = [], [], []

    for inp, gray, tar, _ in tqdm(loader, desc=f'  {name}', ncols=80):
        inp, gray, tar = inp.to(DEVICE), gray.to(DEVICE), tar.to(DEVICE)

        if cfg['type'] == 'shadow_guided' or cfg['type'] == 'baseline':
            out = model(gray, inp)
        else:
            out = model(inp)

        out = out.clamp(0, 1)
        psnr_vals.append(peak_signal_noise_ratio(out, tar, data_range=1).item())
        ssim_vals.append(structural_similarity_index_measure(out, tar, data_range=1).item())
        rmse_vals.append(mean_squared_error(out * 255, tar * 255, squared=False).item())

    def avg(lst):
        return round(np.mean(lst), 4)

    result = {'psnr': avg(psnr_vals), 'ssim': avg(ssim_vals), 'rmse': avg(rmse_vals)}
    print(f'  {name}: PSNR={result["psnr"]:.2f} SSIM={result["ssim"]:.4f} RMSE={result["rmse"]:.2f}')
    return result


def main():
    print(f'Device: {DEVICE} | SD7K test: {SD7K_TEST} | img_size: {IMG_SIZE}')
    print(f'Checkpoint dir: {CKPT_DIR}')
    print(f'{"="*60}')

    results = {}
    for name, cfg in MODELS.items():
        print(f'\nEvaluating {name}...')
        r = evaluate_model(name, cfg)
        if r is not None:
            results[name] = r
        torch.cuda.empty_cache()

    print(f'\n{"="*60}')
    print('CROSS-DATASET RESULTS (RDD -> SD7K)')
    print(f'{"="*60}')
    for metric in ['psnr', 'ssim', 'rmse']:
        row = f'{metric:>8}:'
        for name in results:
            row += f'  {name}={results[name][metric]:.4f}'
        print(row)

    out_path = os.path.join(CKPT_DIR, 'cross_dataset_sd7k.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nSaved: {out_path}')


if __name__ == '__main__':
    main()
