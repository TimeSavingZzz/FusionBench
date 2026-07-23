"""Unified training script for cross-paradigm fusion comparison.

Usage:
    python train.py --backbone cnn --fusion concat --dataset sd7k --res 320 --epochs 200
    python train.py --backbone restormer --fusion film --dataset rdd --res 384 --epochs 200
    python train.py --backbone mamba --fusion gated --dataset rain100l --res 256 --epochs 200

Backbones: cnn, restormer, mamba
Fusions: none, concat, cross_attn, film, gated, large
Datasets: sd7k, rdd, rain100l
"""

import os
import sys
import argparse
import time
import json
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

# Add MambaIR basicsr to path for potential MambaIR imports
sys.path.insert(0, '/mnt/MambaIRv2/basicsr')

from models.factory import build_model
from data.loader import build_dataloader


def parse_args():
    p = argparse.ArgumentParser(description='FusionBench Training')
    p.add_argument('--backbone', required=True,
                   choices=['cnn', 'restormer', 'mamba'])
    p.add_argument('--fusion', required=True,
                   choices=['none', 'concat', 'cross_attn', 'film', 'gated', 'large'])
    p.add_argument('--dataset', required=True,
                   choices=['sd7k', 'rdd', 'rain100l'])
    p.add_argument('--dim', type=int, default=48, help='Base channel dimension')
    p.add_argument('--res', type=int, default=320, help='Training patch size')
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--bs', type=int, default=1, help='Batch size per GPU')
    p.add_argument('--grad_accum', type=int, default=2, help='Gradient accumulation steps')
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--seed', type=int, default=3407)
    p.add_argument('--output_dir', type=str, default='/mnt/FusionBench/checkpoints')
    p.add_argument('--log_dir', type=str, default='/mnt/FusionBench/logs')
    p.add_argument('--resume', type=str, default=None)
    return p.parse_args()


def setup_training(args):
    torch.manual_seed(args.seed)
    torch.cuda.set_device(args.gpu)
    device = torch.device(f'cuda:{args.gpu}')

    model = build_model(args.backbone, args.fusion, dim=args.dim)
    model = model.to(device)
    model.train()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[Model] backbone={args.backbone} fusion={args.fusion} dim={args.dim}")
    print(f"[Model] parameters: {n_params:,} ({n_params/1e6:.2f}M)")

    train_loader, val_loader = build_dataloader(
        args.dataset, args.res, args.bs, args.num_workers)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6)

    # Loss: L1
    criterion = nn.L1Loss()

    # Mixed precision
    scaler = GradScaler()

    return model, train_loader, val_loader, optimizer, scheduler, criterion, scaler, device


def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for batch in val_loader:
            if len(batch) == 4:
                gray, inp, target, _ = batch
            else:
                gray, inp, target = batch
            gray, inp, target = gray.to(device), inp.to(device), target.to(device)
            out = model(gray, inp)
            loss = criterion(out, target)
            total_loss += loss.item()
            count += 1
    model.train()
    return total_loss / max(count, 1)


def train(args):
    model, train_loader, val_loader, optimizer, scheduler, criterion, scaler, device = setup_training(args)

    # Output naming
    run_name = f"{args.backbone}_{args.fusion}_{args.dataset}_d{args.dim}_r{args.res}"
    ckpt_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    log_path = os.path.join(args.log_dir, f"{run_name}.jsonl")

    best_val = float('inf')
    optimizer.zero_grad()

    print(f"[Training] {run_name} — {args.epochs} epochs, lr={args.lr}, device={device}")
    print(f"[Training] Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        epoch_loss = 0.0

        for batch_idx, batch in enumerate(train_loader):
            if len(batch) == 4:
                gray, inp, target, _ = batch
            else:
                gray, inp, target = batch
            gray, inp, target = gray.to(device), inp.to(device), target.to(device)

            with autocast():
                out = model(gray, inp)
                loss = criterion(out, target) / args.grad_accum

            scaler.scale(loss).backward()

            if (batch_idx + 1) % args.grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            epoch_loss += loss.item() * args.grad_accum

        scheduler.step()

        # Validation
        val_loss = validate(model, val_loader, criterion, device)

        # Logging
        elapsed = time.time() - epoch_start
        log_entry = {
            'epoch': epoch, 'train_loss': round(epoch_loss / len(train_loader), 6),
            'val_loss': round(val_loss, 6), 'lr': scheduler.get_last_lr()[0],
            'time_s': round(elapsed, 1),
        }
        with open(log_path, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')

        # Checkpoint
        is_best = val_loss < best_val
        if is_best:
            best_val = val_loss
            torch.save({
                'epoch': epoch, 'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss, 'args': vars(args),
            }, os.path.join(ckpt_dir, 'best.pth'))

        if epoch % 50 == 0:
            torch.save({
                'epoch': epoch, 'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(ckpt_dir, f'epoch_{epoch}.pth'))

        if epoch % 10 == 1:
            print(f"  E{epoch:3d}/{args.epochs} | loss={log_entry['train_loss']:.4f} "
                  f"val={val_loss:.4f} | lr={scheduler.get_last_lr()[0]:.2e} | {elapsed:.0f}s")

    print(f"[Done] {run_name} — best_val={best_val:.4f}")
    print(f"[Done] Checkpoint: {os.path.join(ckpt_dir, 'best.pth')}")
    return ckpt_dir


if __name__ == '__main__':
    args = parse_args()
    train(args)
