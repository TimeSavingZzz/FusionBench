"""Quick test of data loading."""
import sys
sys.path.insert(0, '/mnt/FusionBench')
from data.loader import build_dataloader

for ds in ['sd7k', 'rdd']:
    try:
        train_ldr, val_ldr = build_dataloader(ds, patch_size=256, batch_size=1, num_workers=0)
        print(f"{ds}: train={len(train_ldr.dataset)}, val={len(val_ldr.dataset)}")
        gray, inp, target = next(iter(train_ldr))
        print(f"  shapes: gray={gray.shape}, inp={inp.shape}, target={target.shape}")
    except Exception as e:
        print(f"{ds}: ERROR - {e}")
