"""
Dataset Size Scaling Curve Experiment
======================================
3 models (Concat, FiLM, Gated) × N dataset sizes (500, 2000)
+ existing full SD7K results at 6720 pairs
→ Shows fusion strategy effectiveness vs training data volume

Usage:
  # Step 1: Create subsets (run once)
  python run_scaling_curve.py --prepare_only

  # Step 2: Launch training for a specific model+size
  python run_scaling_curve.py --model concat --size 500 --gpu 1
  python run_scaling_curve.py --model film --size 500 --gpu 2

  # Step 3: Or launch all remaining (on available GPUs)
  python run_scaling_curve.py --launch_all
"""

import os, sys, argparse, random, shutil, subprocess
from pathlib import Path


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SD7K_TRAIN = os.path.join(BASE_DIR, 'dataset', 'SD7K', 'train')
SD7K_TRAIN_INPUT = os.path.join(SD7K_TRAIN, 'input')
SD7K_TRAIN_TARGET = os.path.join(SD7K_TRAIN, 'target')
SUBSET_BASE = os.path.join(BASE_DIR, 'dataset', 'SD7K_subsets')

MODEL_MAP = {
    'concat': 'shadow_guided_restormer_no_sgca',
    'film': 'shadow_guided_restormer_film',
    'gated': 'shadow_guided_restormer_gated',
}

SIZES = [500, 1000, 2000, 4000]


def create_subsets(seed=42):
    """Create SD7K subsets by symlinking/copying random samples."""
    random.seed(seed)

    input_files = sorted(os.listdir(SD7K_TRAIN_INPUT))
    target_files = sorted(os.listdir(SD7K_TRAIN_TARGET))
    assert len(input_files) == len(target_files), "Mismatched input/target counts"
    assert len(input_files) >= max(SIZES), f"Need at least {max(SIZES)} pairs, have {len(input_files)}"

    # Shuffle deterministically
    pairs = list(zip(input_files, target_files))
    random.shuffle(pairs)

    for size in SIZES:
        subset_dir = os.path.join(SUBSET_BASE, f'sd7k_{size}')
        subset_input = os.path.join(subset_dir, 'train', 'input')
        subset_target = os.path.join(subset_dir, 'train', 'target')

        # Also copy validation set (full test set for fair comparison)
        val_input = os.path.join(subset_dir, 'test', 'input')
        val_target = os.path.join(subset_dir, 'test', 'target')

        for d in [subset_input, subset_target, val_input, val_target]:
            os.makedirs(d, exist_ok=True)

        # Create symlinks for train subset
        selected = pairs[:size]
        for inp, tar in selected:
            src_inp = os.path.join(SD7K_TRAIN_INPUT, inp)
            dst_inp = os.path.join(subset_input, inp)
            src_tar = os.path.join(SD7K_TRAIN_TARGET, tar)
            dst_tar = os.path.join(subset_target, tar)
            if not os.path.exists(dst_inp):
                os.symlink(os.path.abspath(src_inp), dst_inp)
            if not os.path.exists(dst_tar):
                os.symlink(os.path.abspath(src_tar), dst_tar)

        # Copy validation set (same for all subsets)
        val_src_input = os.path.join(SD7K_TRAIN.replace('train', 'test'), 'input')
        val_src_target = os.path.join(SD7K_TRAIN.replace('train', 'test'), 'target')

        for f in os.listdir(val_src_input):
            dst = os.path.join(val_input, f)
            src = os.path.join(val_src_input, f)
            if not os.path.exists(dst):
                os.symlink(os.path.abspath(src), dst)
        for f in os.listdir(val_src_target):
            dst = os.path.join(val_target, f)
            src = os.path.join(val_src_target, f)
            if not os.path.exists(dst):
                os.symlink(os.path.abspath(src), dst)

        n_train = len(os.listdir(subset_input))
        n_val = len(os.listdir(val_input))
        print(f"  SD7K_{size}: {n_train} train pairs, {n_val} val pairs")


def launch_training(model_key, size, gpu_id, epochs=100):
    """Launch one training job on specified GPU."""
    model_name = MODEL_MAP[model_key]
    subset_dir = os.path.join(SUBSET_BASE, f'sd7k_{size}')
    output_dir = os.path.join(BASE_DIR, 'experiment_results', f'scaling_{model_key}_{size}')

    os.makedirs(output_dir, exist_ok=True)
    log_file = f'/mnt/ShaDocFormer-main/scaling_{model_key}_{size}.log'

    cmd = (
        f"source /opt/miniconda3/etc/profile.d/conda.sh && "
        f"conda activate shadocformer && "
        f"cd /mnt/ShaDocFormer-main && "
        f"CUDA_VISIBLE_DEVICES={gpu_id} nohup python -u train_compare_models.py "
        f"--model {model_name} --dataset sd7k --epochs {epochs} --lr 2e-4 "
        f"--batch_size 1 --res 320 "
        f"--train_dir {subset_dir}/train/ --val_dir {subset_dir}/test/ "
        f"--output {output_dir} "
        f"> {log_file} 2>&1 &"
    )

    print(f"\nLaunching: {model_key} @ {size} pairs on GPU {gpu_id}")
    print(f"  Model: {model_name}")
    print(f"  Epochs: {epochs}")
    print(f"  Log: {log_file}")
    print(f"  Output: {output_dir}")

    return cmd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare_only', action='store_true', help='Create SD7K subsets')
    parser.add_argument('--model', type=str, choices=['concat', 'film', 'gated'])
    parser.add_argument('--size', type=int, choices=[500, 1000, 2000, 4000])
    parser.add_argument('--gpu', type=int, default=1, help='GPU device ID')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--launch_all', action='store_true')
    parser.add_argument('--print_commands', action='store_true',
                        help='Print launch commands for manual execution')
    args = parser.parse_args()

    if args.prepare_only:
        print("Creating SD7K subsets...")
        create_subsets()
        print("Done.")
        return

    if args.print_commands:
        # Print all commands for batch launching
        print("=== Copy-paste these commands in the container ===\n")
        # Round 1: 500 pairs on 3 GPUs
        for i, model in enumerate(['concat', 'film', 'gated']):
            cmd = launch_training(model, 500, i, args.epochs)
            print(cmd)
            print("sleep 3")
        print()
        # Round 2: 2000 pairs on 3 GPUs (after Round 1 finishes)
        for i, model in enumerate(['concat', 'film', 'gated']):
            cmd = launch_training(model, 2000, i, args.epochs)
            print(cmd)
            print("sleep 3")
        return

    if args.model and args.size is not None:
        cmd = launch_training(args.model, args.size, args.gpu, args.epochs)
        print(f"\nRun this in container:\n{cmd}")
        return

    if args.launch_all:
        # Launch what we can now
        print("Launching Round 1 (500 pairs on GPUs 0,1,3):")
        gpu_assignments = [0, 1, 3]  # GPU 2 keeps running GatedLarge finetuning
        for model_key, gpu in zip(['concat', 'film', 'gated'], gpu_assignments):
            cmd = launch_training(model_key, 500, gpu, args.epochs)
            print(cmd)
        return

    parser.print_help()


if __name__ == '__main__':
    main()
