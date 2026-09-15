import sys, os, time
sys.path.insert(0, '/mnt/ShaDocFormer-main')
import torch
from fvcore.nn import FlopCountAnalysis
from models.comparison_models import (
    Restormer, ShadowGuidedRestormer_NoSGCA, ShadowGuidedRestormer_Concat,
    ShadowGuidedRestormer_CrossAttn, ShadowGuidedRestormer_FiLM, ShadowGuidedRestormer_Large
)
MODELS = [
    ('Restormer',  Restormer,                         {'dim': 48}, 256, 'single'),
    ('No SGCA',    ShadowGuidedRestormer_NoSGCA,      {'dim': 48}, 256, 'dual'),
    ('Concat',     ShadowGuidedRestormer_Concat,       {'dim': 48}, 256, 'dual'),
    ('CrossAttn',  ShadowGuidedRestormer_CrossAttn,    {'dim': 48}, 192, 'dual'),
    ('FiLM',       ShadowGuidedRestormer_FiLM,         {'dim': 48}, 256, 'dual'),
    ('Large',      ShadowGuidedRestormer_Large,        {'dim': 64}, 320, 'dual'),
]
hdr = f"{'Model':<12} {'Params(M)':>10} {'FLOPs':>10} {'Time(ms)':>10} {'FPS':>8} {'Memory':>10}"
print(hdr)
print('-' * 65)
for name, cls, kwargs, res, mode in MODELS:
    model = cls(**kwargs).cuda().eval()
    params = sum(p.numel() for p in model.parameters()) / 1e6
    model_cpu = cls(**kwargs).cpu().eval()
    if mode == 'single':
        x = torch.randn(1, 3, res, res)
        try:
            flops = FlopCountAnalysis(model_cpu, x)
            gflops = flops.total() / 1e9
            flops_str = f'{gflops:.2f}G' if gflops >= 1 else f'{flops.total()/1e6:.1f}M'
        except Exception as e:
            flops_str = 'N/A'
    else:
        gray = torch.randn(1, 1, res, res)
        inp = torch.randn(1, 3, res, res)
        try:
            flops = FlopCountAnalysis(model_cpu, (gray, inp))
            gflops = flops.total() / 1e9
            flops_str = f'{gflops:.2f}G' if gflops >= 1 else f'{flops.total()/1e6:.1f}M'
        except Exception as e:
            flops_str = 'N/A'
    del model_cpu
    with torch.no_grad():
        if mode == 'single':
            x = torch.randn(1, 3, res, res, device='cuda')
            for _ in range(10): _ = model(x)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(100): _ = model(x)
            torch.cuda.synchronize()
        else:
            gray = torch.randn(1, 1, res, res, device='cuda')
            inp = torch.randn(1, 3, res, res, device='cuda')
            for _ in range(10): _ = model(gray, inp)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(100): _ = model(gray, inp)
            torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / 100 * 1000
        fps = 1000 / ms if ms > 0 else 0
    mem = torch.cuda.max_memory_allocated() / 1024**2
    torch.cuda.reset_peak_memory_stats()
    print(f'{name:<12} {params:>8.2f}M {flops_str:>10} {ms:>8.1f}ms {fps:>6.1f} {mem:>8.0f}MB')
print('Done.')
