"""Data loaders for SD7K, RDD, and Rain100L datasets.

SD7K/RDD: grayscale shadow map + RGB input + RGB target
Rain100L: RGB rainy input + RGB clean target (grayscale rainy as shadow map proxy)
"""

import os
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader


# ==============================================================================
# Base document shadow dataset (SD7K / RDD)
# ==============================================================================
class ShadowDocumentDataset(Dataset):
    """Dataset for SD7K and RDD document shadow removal.

    Directory structure:
        SD7K/
            train/
                input/    # shadowed RGB images (*.png)
                target/   # shadow-free RGB images (*.png)
            test/
                ...
        RDD/
            train/
                input/    # shadowed RGB images
                gt/       # shadow-free RGB images (clean target)
            test/
                ...

    Shadow map is derived as grayscale(input) when no explicit shadow/ directory exists.
    The shadow encoder expects grayscale shadow map as the 'gray' input.
    """
    def __init__(self, root, split='train', patch_size=320, is_train=True):
        self.root = os.path.join(root, split)
        self.patch_size = patch_size
        self.is_train = is_train

        input_dir = os.path.join(self.root, 'input')
        target_dir = os.path.join(self.root, 'target')
        if not os.path.exists(target_dir):
            for alt in ['gt', 'clean', 'label', 'norain']:
                alt_dir = os.path.join(self.root, alt)
                if os.path.exists(alt_dir):
                    target_dir = alt_dir
                    break

        # Shadow map: use explicit directory if available, else derive from input
        shadow_dir = os.path.join(self.root, 'shadow')
        self._derive_shadow = False
        if not os.path.exists(shadow_dir):
            for alt in ['mask', 'shadow_map', 'shadow_mask']:
                alt_dir = os.path.join(self.root, alt)
                if os.path.exists(alt_dir):
                    shadow_dir = alt_dir
                    break
        if not os.path.exists(shadow_dir):
            self._derive_shadow = True

        self.input_files = sorted(os.listdir(input_dir))
        self.target_files = sorted(os.listdir(target_dir))

        self.input_dir = input_dir
        self.target_dir = target_dir
        self.shadow_dir = shadow_dir if not self._derive_shadow else None

        if is_train:
            random.shuffle(self.input_files)

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        inp = self._load_rgb(os.path.join(self.input_dir, self.input_files[idx]))
        target = self._load_rgb(os.path.join(self.target_dir, self.target_files[idx % len(self.target_files)]))

        if self._derive_shadow:
            shadow = 0.2989 * inp[0] + 0.5870 * inp[1] + 0.1140 * inp[2]
            shadow = shadow.unsqueeze(0)
        else:
            shadow = self._load_gray(os.path.join(self.shadow_dir,
                                                   os.listdir(self.shadow_dir)[idx % len(os.listdir(self.shadow_dir))]))

        if self.is_train:
            shadow, inp, target = self._random_crop(shadow, inp, target, self.patch_size)

        return shadow, inp, target

    def _load_rgb(self, path):
        img = Image.open(path).convert('RGB')
        arr = np.array(img).astype(np.float32) / 255.0
        return torch.from_numpy(arr.transpose(2, 0, 1))

    def _load_gray(self, path):
        img = Image.open(path).convert('L')
        arr = np.array(img).astype(np.float32) / 255.0
        return torch.from_numpy(arr).unsqueeze(0)

    def _random_crop(self, shadow, inp, target, size):
        _, h, w = inp.shape
        if h < size or w < size:
            return shadow, inp, target
        top = random.randint(0, h - size)
        left = random.randint(0, w - size)
        return (shadow[:, top:top+size, left:left+size],
                inp[:, top:top+size, left:left+size],
                target[:, top:top+size, left:left+size])


# ==============================================================================
# Deraining dataset (Rain100L)
# ==============================================================================
class DerainingDataset(Dataset):
    """Dataset for Rain100L deraining.

    Directory structure:
        Rain100L/
            train/
                rain/    # rainy images (*.png)
                norain/  # clean images (*.png)
            test/
                ...

    The shadow encoder receives grayscale(rainy) as the guide signal,
    since deraining has no natural shadow map.
    """
    def __init__(self, root, split='train', patch_size=256, is_train=True):
        self.root = os.path.join(root, split)
        self.patch_size = patch_size
        self.is_train = is_train

        rain_dir = os.path.join(self.root, 'rain')
        norain_dir = os.path.join(self.root, 'norain')

        # Try alternative names
        if not os.path.exists(rain_dir):
            for alt in ['rainy', 'input', 'data']:
                alt_dir = os.path.join(self.root, alt)
                if os.path.exists(alt_dir):
                    rain_dir = alt_dir
                    break
        if not os.path.exists(norain_dir):
            for alt in ['clean', 'target', 'gt', 'label']:
                alt_dir = os.path.join(self.root, alt)
                if os.path.exists(alt_dir):
                    norain_dir = alt_dir
                    break

        self.rain_files = sorted(os.listdir(rain_dir))
        self.norain_files = sorted(os.listdir(norain_dir))
        self.rain_dir = rain_dir
        self.norain_dir = norain_dir

        if is_train:
            random.shuffle(self.rain_files)

    def __len__(self):
        return len(self.rain_files)

    def __getitem__(self, idx):
        rainy = self._load_rgb(os.path.join(self.rain_dir, self.rain_files[idx]))
        clean = self._load_rgb(os.path.join(self.norain_dir,
                                             self.norain_files[idx % len(self.norain_files)]))

        # Shadow map proxy: grayscale rainy image
        gray = 0.2989 * rainy[0] + 0.5870 * rainy[1] + 0.1140 * rainy[2]
        gray = gray.unsqueeze(0)  # (1, H, W)

        if self.is_train:
            _, h, w = rainy.shape
            if h >= self.patch_size and w >= self.patch_size:
                top = random.randint(0, h - self.patch_size)
                left = random.randint(0, w - self.patch_size)
                gray = gray[:, top:top+self.patch_size, left:left+self.patch_size]
                rainy = rainy[:, top:top+self.patch_size, left:left+self.patch_size]
                clean = clean[:, top:top+self.patch_size, left:left+self.patch_size]

        return gray, rainy, clean

    def _load_rgb(self, path):
        img = Image.open(path).convert('RGB')
        arr = np.array(img).astype(np.float32) / 255.0
        return torch.from_numpy(arr.transpose(2, 0, 1))


# ==============================================================================
# Dataset paths
# ==============================================================================
DATASET_ROOTS = {
    'sd7k': '/mnt/ShaDocFormer-main/dataset/SD7K',
    'rdd': '/mnt/ShaDocFormer-main/dataset/RDD',
    'rain100l': '/mnt/ShaDocFormer-main/dataset/Rain100L',
}


def build_dataloader(dataset_name, patch_size=320, batch_size=1, num_workers=4):
    """Build train and validation dataloaders for a given dataset."""
    root = DATASET_ROOTS.get(dataset_name)
    if root is None:
        raise ValueError(f"Unknown dataset: {dataset_name}. Known: {list(DATASET_ROOTS.keys())}")

    DatasetClass = DerainingDataset if dataset_name == 'rain100l' else ShadowDocumentDataset

    train_ds = DatasetClass(root, split='train', patch_size=patch_size, is_train=True)
    val_ds = DatasetClass(root, split='test', patch_size=patch_size, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                            num_workers=num_workers, pin_memory=True)

    print(f"[Data] {dataset_name}: train={len(train_ds)}, val={len(val_ds)}")
    return train_loader, val_loader
