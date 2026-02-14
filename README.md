# SSL Vision Transformer for JWST Galaxy Classification

Self-supervised Vision Transformer pipeline for automatic morphological classification of spiral galaxies in the JWST thumbnail catalog (~3.5M galaxies, ~1.5 TB of single-channel FITS images).

## Overview

This project implements **3 SSL pretraining methods** across **3 deep learning frameworks**, yielding 9 experiment configurations that share a common data pipeline, training infrastructure, and evaluation suite.

| | PyTorch + timm | PyTorch + HuggingFace | JAX / Flax |
|---|---|---|---|
| **MAE** | ✅ | ✅ | ✅ |
| **DINO** | ✅ | ✅ | ✅ |
| **MAE → DINO** | ✅ | ✅ | ✅ |

### Key Design Decisions

- **Image size**: 64×64 (balances 20–100px raw thumbnails)
- **Patch size**: 8 → 64 patches per image
- **Input channels**: 1 (single-channel FITS grayscale)
- **ViT sizes**: Configurable — Tiny (192d), Small (384d), Base (768d)
- **Labeling strategy**: Cluster-then-verify (no labeled subset exists)

## Directory Structure

```
vision-transformer/
├── configs/                    # YAML experiment configurations
│   ├── base.yaml               # Shared defaults
│   ├── mae.yaml                # MAE-specific overrides
│   ├── dino.yaml               # DINO-specific overrides
│   └── mae_dino.yaml           # Two-stage MAE→DINO config
├── documentation/
│   └── specification-document.md
├── modules/                    # Git submodules (reserved)
├── output/                     # Training outputs, checkpoints, plots
├── requirements/
│   ├── pytorch.txt             # PyTorch + timm + HuggingFace deps
│   └── jax.txt                 # JAX + Flax + Optax deps
├── scripts/
│   ├── train.py                # Main training entry point
│   ├── extract_embeddings.py   # CLS embedding extraction → HDF5
│   ├── evaluate.py             # Clustering + classification pipeline
│   ├── compare_experiments.py  # Cross-config comparison
│   └── pixel_distribution.py   # Catalog dimension analysis (pre-existing)
└── src/
    ├── data/
    │   ├── fits_dataset.py     # Framework-agnostic FITS loader
    │   ├── augmentations.py    # Astronomy-aware augmentations
    │   ├── pytorch_loader.py   # PyTorch Dataset/DataLoader
    │   └── jax_loader.py       # JAX data iterators
    ├── models/
    │   ├── vit_config.py       # Shared ViT configuration dataclass
    │   ├── pytorch_timm/       # timm ViT + MAE + DINO
    │   ├── pytorch_hf/         # HuggingFace ViT + MAE + DINO
    │   └── jax_flax/           # Flax ViT + MAE + DINO
    ├── trainers/
    │   ├── mae_trainer.py      # MAE training loop (PyTorch + JAX)
    │   ├── dino_trainer.py     # DINO training loop (PyTorch + JAX)
    │   └── mae_dino_trainer.py # Two-stage MAE→DINO trainer
    ├── evaluation/
    │   ├── clustering.py       # k-means + UMAP on embeddings
    │   ├── annotation_tool.py  # Cluster-then-verify labeling UI
    │   └── classifier.py       # Linear probe + fine-tuning
    └── utils/
        ├── distributed.py      # DDP + JAX pmap setup
        ├── checkpointing.py    # Save/load checkpoints
        └── logging_utils.py    # W&B + console logging
```

## Installation

### PyTorch Environment

```bash
conda create -n ssl-vit-pt python=3.11
conda activate ssl-vit-pt
pip install -r requirements/pytorch.txt
```

### JAX Environment

```bash
conda create -n ssl-vit-jax python=3.11
conda activate ssl-vit-jax
pip install -r requirements/jax.txt
```

## Usage

### 1. Training

```bash
# MAE pretraining with timm, ViT-Tiny
python scripts/train.py --config configs/mae.yaml --framework timm --vit_size tiny

# DINO with HuggingFace, ViT-Small
python scripts/train.py --config configs/dino.yaml --framework hf --vit_size small

# MAE→DINO two-stage with JAX
python scripts/train.py --config configs/mae_dino.yaml --framework jax --vit_size tiny

# Multi-GPU (PyTorch DDP)
torchrun --nproc_per_node=4 scripts/train.py \
  --config configs/mae.yaml --framework timm --distributed

# Resume from checkpoint
python scripts/train.py --config configs/mae.yaml --resume output/checkpoints/checkpoint_latest.pt

# Enable Weights & Biases logging
python scripts/train.py --config configs/mae.yaml --wandb
```

### 2. Extract Embeddings

```bash
python scripts/extract_embeddings.py \
  --checkpoint output/checkpoints/checkpoint_best.pt \
  --framework timm --method mae --vit_size tiny \
  --output output/embeddings.h5

# JAX checkpoint
python scripts/extract_embeddings.py \
  --checkpoint output/checkpoints/checkpoint_best \
  --framework jax --method mae \
  --output output/embeddings_jax.h5

# Subset (first 10k images)
python scripts/extract_embeddings.py \
  --checkpoint output/checkpoints/checkpoint_best.pt \
  --framework timm --method mae --max_samples 10000
```

### 3. Evaluate

```bash
# Clustering + UMAP visualization
python scripts/evaluate.py --embeddings output/embeddings.h5 --output_dir output/eval

# Sweep k values for optimal clustering
python scripts/evaluate.py --embeddings output/embeddings.h5 --sweep_k

# Interactive cluster labeling
python scripts/evaluate.py --embeddings output/embeddings.h5 --label

# Linear probe classification (requires labeled CSV)
python scripts/evaluate.py \
  --classify --labels_csv output/eval/labels.csv \
  --checkpoint output/checkpoints/checkpoint_best.pt \
  --framework timm --method mae --linear_probe

# Fine-tuning
python scripts/evaluate.py \
  --classify --labels_csv output/eval/labels.csv \
  --checkpoint output/checkpoints/checkpoint_best.pt \
  --framework timm --method mae --fine_tune
```

### 4. Compare Experiments

```bash
python scripts/compare_experiments.py --results_dir output/experiments
```

## Architecture

### SSL Methods

**MAE (Masked Autoencoder)**
- Masks 75% of image patches randomly
- Encoder processes only visible patches (efficient)
- Lightweight decoder (4 layers, 128d) reconstructs masked patches
- Loss: MSE on normalized pixel values of masked patches

**DINO (Self-Distillation with No Labels)**
- Student-teacher framework with EMA teacher updates
- Multi-crop strategy: 2 global (64×64) + 6 local (32×32) views
- Cross-entropy loss between teacher (global) and student (all) outputs
- Centering + sharpening to prevent mode collapse

**MAE → DINO (Two-Stage)**
- Stage 1: MAE pretraining for general patch-level features
- Stage 2: Initialize DINO student from MAE encoder; run DINO for instance-level alignment
- Combines reconstruction-based and contrastive learning

### ViT Configuration

| Size | Embed Dim | Depth | Heads | Params (approx) |
|------|-----------|-------|-------|------------------|
| Tiny | 192 | 12 | 3 | ~5.5M |
| Small | 384 | 12 | 6 | ~22M |
| Base | 768 | 12 | 12 | ~86M |

All variants use patch_size=8 on 64×64 single-channel images → 64 patches per image.

### Cluster-then-Verify Labeling

Since no labeled subset of JWST galaxy images exists:

1. **Pretrain** an SSL encoder on the full unlabeled catalog
2. **Extract** CLS token embeddings for all images
3. **Cluster** embeddings via k-means (sweep k for optimal clustering)
4. **Visualize** clusters with UMAP to assess quality
5. **Verify** representative samples from each cluster manually
6. **Propagate** cluster-level labels to all images in each cluster
7. **Train** a linear probe or fine-tune a classifier on the labeled data

## Data

- **Source**: JWST galaxy thumbnails at `../JWSP-JWST-to-SpArcFiRe/outputs/JWST-JWST_galaxy_thumbnails`
- **Format**: FITS (Flexible Image Transport System), single-channel grayscale
- **Scale**: ~3.5M images, ~1.5 TB
- **Dimensions**: 20–100px (raw), resized to 64×64
- **Normalization**: Percentile clipping (1st–99th percentile) to handle cosmic ray artifacts

## References

- He, K. et al. "Masked Autoencoders Are Scalable Vision Learners." CVPR 2022.
- Caron, M. et al. "Emerging Properties in Self-Supervised Vision Transformers." ICCV 2021.
- Dosovitskiy, A. et al. "An Image is Worth 16x16 Words." ICLR 2021.