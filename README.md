# FusionBench

Code and experiment scripts for the paper:

> **A Controlled Study of Feature Fusion in Transformer-Based Document Shadow Removal**

This repository contains everything needed to reproduce the controlled comparison
of four fusion mechanisms plus one capacity-scaling control for Transformer-based
document shadow removal, together with the analysis scripts behind every figure
and table in the paper.

---

## 1. What this study is about

Modern image restoration networks commonly adopt a two-stream design: a main
encoder–decoder processes the degraded image, while an auxiliary encoder extracts
prior information (here, a shadow map) that is injected through a *fusion module*.
Many fusion designs have been proposed, but they are almost always evaluated on
different backbones, protocols, and datasets, so it is impossible to tell how much
of the reported gain comes from the fusion design itself.

This repository fixes the backbone, training protocol, and loss, varies only the
fusion operation, and asks a single question: **how large is the difference that
this one factor actually makes?**

---

## 2. Key findings

| # | Finding | Evidence |
|---|---------|----------|
| 1 | **The spread between mechanisms is small and dataset-dependent.** Within the resolution-matched group the mechanisms span 1.03 dB on SD7K and 0.19 dB on RDD, against a shift of roughly 4 dB from changing the evaluation domain. | Table 1; `run_scaling_curve.py` |
| 2 | **Cross-attention does not repay its cost.** It reaches 23.76 dB on SD7K and 36.99 dB on RDD, close to the simpler mechanisms, while requiring about four times the memory. | `eval_efficiency.py` |
| 3 | **The fusion variants do not improve text-region preservation.** On RDD the model with no auxiliary pathway attains the highest text-region PSNR (34.77 dB), above every resolution-matched fusion variant. | Table 1; `eval_ocr.py` |

Across the training-set sizes we tested, the gap between mechanisms changes size
but their ordering does not (`run_scaling_curve.py`). Note that the 200-pair
condition also uses a longer schedule, so the two cannot be fully separated.

---

## 3. Repository structure

```
FusionBench/
├── train_compare_models.py     # Main training entry point (all strategies)
├── eval_lpips.py               # Perceptual metric (LPIPS)
├── eval_ocr.py                 # OCR / text-preservation metric
├── eval_efficiency.py          # Params, FLOPs, inference time, memory
├── eval_cross_dataset.py       # Zero-shot cross-domain transfer
├── cross_dataset_sd7k_to_rdd.py
├── evaluate_checkpoint.py      # Evaluate any saved checkpoint
├── visualize_attention.py      # Decoder self-attention maps (Fig. 3)
├── run_attn_viz.py             # Batch attention visualization
├── run_scaling_curve.py        # Data-volume scaling experiment
├── gen_scaling_fig.py          # Scaling-curve figure
├── alpha_histogram.py          # Gate-value distribution analysis
│
├── models/
│   ├── comparison_models.py    # Restormer + all fusion variants  ← core
│   ├── fusion_models.py        # Additional fusion implementations
│   ├── model.py, refine.py, mask.py, text_aware_model.py
│
├── data/                       # Dataset loaders (SD7K, RDD, ...)
├── config/                     # Configuration
├── utils/                      # Metrics and helpers
│
├── fusion/                     # Standalone, framework-agnostic fusion modules
├── backbones/                  # Backbone implementations
└── train.py                    # Lightweight standalone trainer
```

`models/comparison_models.py` and `train_compare_models.py` are the code used for
all experiments reported in the paper. The `fusion/` and `backbones/` directories
provide the same fusion operators as standalone modules for reuse in other codebases.

---

## 4. Installation

```bash
conda create -n fusionbench python=3.10 -y
conda activate fusionbench

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install torchmetrics tqdm numpy pillow matplotlib opencv-python
pip install lpips easyocr          # for eval_lpips.py / eval_ocr.py
```

Tested with PyTorch 2.x on 4×NVIDIA RTX 3090 (24 GB).

---

## 5. Data preparation

| Dataset | Domain | Size | Source |
|---------|--------|------|--------|
| **SD7K** | synthetic document shadows | 6,720 train / 190 test pairs | <https://github.com/CXH-Research/DocShadow-SD7K> |
| **RDD** | real-world document shadows | 1,080 train pairs | <https://github.com/leiyingtie/DocDeshadower> |

Download each dataset and place it so that the directory layout is:

```
dataset/
├── SD7K/
│   ├── train/
│   │   ├── input/    *.png
│   │   └── target/   *.png
│   └── test/
│       ├── input/
│       └── target/
└── RDD/
    ├── train/  (input/ target/)
    └── test/   (input/ target/)
```

If a shadow-map directory is not provided, the loaders derive the shadow map from
the grayscale input automatically.

---

## 6. Reproducing the experiments

All commands are run **from the repository root**.

### 6.1 Main comparison (Table 1, Fig. 1)

Each strategy is trained under an identical protocol: 200 epochs, AdamW, initial
learning rate 2e-4 halved at epochs 100 and 150, batch size 1 with gradient
accumulation 2.

```bash
# SD7K (synthetic), 320x320
python train_compare_models.py --model restormer                        --dataset sd7k --epochs 200 --res 320
python train_compare_models.py --model shadow_guided_restormer_no_sgca  --dataset sd7k --epochs 200 --res 320
python train_compare_models.py --model shadow_guided_restormer_crossattn --dataset sd7k --epochs 200 --res 192
python train_compare_models.py --model shadow_guided_restormer_film     --dataset sd7k --epochs 200 --res 320
python train_compare_models.py --model shadow_guided_restormer_gated    --dataset sd7k --epochs 200 --res 320
python train_compare_models.py --model shadow_guided_restormer_large    --dataset sd7k --epochs 200 --res 320

# RDD (real-world)
python train_compare_models.py --model restormer                         --dataset rdd --epochs 200 --res 320
python train_compare_models.py --model shadow_guided_restormer_no_sgca   --dataset rdd --epochs 200 --res 320
python train_compare_models.py --model shadow_guided_restormer_crossattn --dataset rdd --epochs 200 --res 192
python train_compare_models.py --model shadow_guided_restormer_film      --dataset rdd --epochs 200 --res 320 --model_variant v2   # gradient clipping
python train_compare_models.py --model shadow_guided_restormer_gated     --dataset rdd --epochs 200 --res 256
python train_compare_models.py --model shadow_guided_restormer_large     --dataset rdd --epochs 200 --res 320
```

> **Note.** `--model_variant v1` (default) trains FiLM without gradient clipping;
> on RDD it diverges to NaN around epoch 125–130 because the `tanh` modulation
> saturates. `--model_variant v2` enables gradient clipping (`max_norm = 1.0`),
> which rescues training and gives 37.01 dB.

> **Note.** Cross-attention is trained at 192×192 because its key/value projections
> exceed 24 GB at 320×320.

### 6.2 Data-volume scaling (Fig. 2)

```bash
# Trains Concat / FiLM / Gated on nested SD7K subsets of 30, 60, 100, 150, 200 pairs
python run_scaling_curve.py
python gen_scaling_fig.py
```

### 6.3 Attention visualization (Fig. 3)

```bash
python visualize_attention.py --checkpoint <path/to/model_best.pth> --sample <image.png>
# or in batch:
python run_attn_viz.py
```

### 6.4 Cross-domain transfer and fine-tuning

```bash
# Zero-shot: train on one domain, evaluate on the other
python eval_cross_dataset.py
python cross_dataset_sd7k_to_rdd.py

# Fine-tuning: SD7K-pretrained model adapted to RDD
python train_compare_models.py --model shadow_guided_restormer_gated --dataset rdd \
       --epochs 200 --res 320 --resume <sd7k_checkpoint.pth>
```

### 6.5 Efficiency, LPIPS, and OCR

```bash
python eval_efficiency.py     # parameters, FLOPs, time, memory
python eval_lpips.py          # perceptual quality
python eval_ocr.py            # text-region PSNR / character recovery
```

---

## 7. Strategy name mapping

The names used in the paper and in the code differ slightly:

| In the paper | `--model` argument |
|--------------|--------------------|
| Restormer (no fusion) | `restormer` |
| Concat | `shadow_guided_restormer_no_sgca` |
| Cross-Attn | `shadow_guided_restormer_crossattn` |
| FiLM | `shadow_guided_restormer_film` |
| Gated | `shadow_guided_restormer_gated` |
| Large | `shadow_guided_restormer_large` |
| GatedLarge | `shadow_guided_restormer_gated_large` |
| Ablation: no shadow encoder | `shadow_guided_restormer_gated_noshadow` |
| Ablation: single decoder level | `shadow_guided_restormer_gated_dec3only` |

---

## 8. Main results (SD7K / RDD, PSNR in dB)

| Strategy | SD7K (synthetic) | RDD (real-world) |
|----------|-----------------|------------------|
| Restormer (no fusion) | 23.43 | 35.72 |
| Concat | 23.74 | **37.20** |
| Cross-Attn | 23.76 | 37.03 |
| FiLM | **24.77** | 37.01 |
| Gated | 24.31 | 35.53 |
| Large | 24.51 | 37.05 |

Best of all on **text preservation** (RDD TextPSNR): the *no-fusion* baseline,
34.77 dB.

---

## 9. Notes on reproducibility

- All experiments use a single seed; the study reports pattern-level evidence from
  over 40 controlled training runs rather than multi-seed variance estimates.
- Training logs, checkpoints, and datasets are excluded from this repository by
  `.gitignore`; only source code and result summaries are tracked.
- The scaling experiment covers 3 of the 7 strategies on synthetic data only.

---

## 10. License

Released for academic use. See `LICENSE` for details.
