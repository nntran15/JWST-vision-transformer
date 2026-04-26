# SSL Vision Transformer for JWST Galaxy Classification

Self-supervised Vision Transformer pipeline for automatic morphological classification of spiral galaxies from a large local JWST thumbnail corpus (~3.5M random sky-patch thumbnails cropped from MAST and stored outside this repo).

## Overview

This project implements **3 SSL pretraining methods** across **3 ViT sizes**, yielding **9 pilot configurations** that share a common data pipeline, training infrastructure, and evaluation suite. The current pilot keeps the training framework fixed to `timm` so the first comparison isolates method and model size on a small 10k subset of the local thumbnail corpus. The HuggingFace and JAX backends remain available as secondary implementation paths after the method/size winner is clear.

### Why Self-Supervised Learning?

**SSL Pretraining** teaches the model to _see_ without any labels. The model learns representations by solving self-supervised puzzles — MAE reconstructs masked patches, DINO learns that two crops of the same galaxy are related. No human labels are involved. The model discovers that "these pixels form a spiral arm" or "this is a point source" purely from visual statistics. This is where the full unlabeled catalog is used.

**Supervised Fine-tuning** teaches the model to _classify_ using a small labeled dataset. The pretrained model — which already understands galaxy morphology — only needs a few thousand labeled examples from a curated evaluation set to learn the spiral vs. non-spiral task first, and later the CW vs. CCW spin-direction task.

The reason for doing SSL first: labeled galaxy morphology data is scarce and expensive, but unlabeled JWST data is abundant. SSL exploits that abundance to build a strong visual foundation, then the small labeled set does the final job.

### The 9 Pilot Configurations

| #   | SSL Method | ViT Size |
| --- | ---------- | -------- |
| 1   | MAE        | Tiny     |
| 2   | MAE        | Small    |
| 3   | MAE        | Base     |
| 4   | DINO       | Tiny     |
| 5   | DINO       | Small    |
| 6   | DINO       | Base     |
| 7   | MAE→DINO   | Tiny     |
| 8   | MAE→DINO   | Small    |
| 9   | MAE→DINO   | Base     |

The pilot comparison should keep preprocessing fixed and use the same backend for all 9 runs. The examples below assume a 64x64 training input after resize/pad preprocessing, but the new thumbnail corpus must be profiled first because the raw files are not guaranteed to arrive at 64x64.

---

## Directory Structure

```
vision-transformer/
├── configs/                    # YAML experiment configurations
│   ├── base.yaml               # Shared defaults (data paths, training, model)
│   ├── mae.yaml                # MAE-specific overrides
│   ├── dino.yaml               # DINO-specific overrides
│   └── mae_dino.yaml           # Two-stage MAE→DINO config
├── data/                       # Local data (small test subsets)
├── documentation/
│   └── specification-document.md
├── output/                     # All training outputs, checkpoints, plots
│   └── experiments/            # Per-experiment results
├── requirements/
│   ├── pytorch.txt             # PyTorch + timm + HuggingFace deps
│   └── jax.txt                 # JAX + Flax + Optax deps
├── scripts/
│   ├── train.py                # Main training entry point
│   ├── extract_embeddings.py   # CLS embedding extraction → HDF5
│   ├── evaluate.py             # Clustering + UMAP + classification pipeline
│   └── compare_experiments.py  # Cross-config comparison tables & plots
└── src/
    ├── data/
  │   ├── fits_dataset.py     # Framework-agnostic thumbnail loader + indexer
    │   ├── augmentations.py    # Astronomy-aware augmentations
    │   ├── pytorch_loader.py   # PyTorch Dataset/DataLoader wrapper
    │   └── jax_loader.py       # JAX data iterators
    ├── models/
    │   ├── vit_config.py       # Shared ViT configuration dataclass
    │   ├── pytorch_timm/       # timm ViT + MAE + DINO heads
    │   ├── pytorch_hf/         # HuggingFace ViT + MAE + DINO heads
    │   └── jax_flax/           # Flax ViT + MAE + DINO heads
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

---

## Installation

### PyTorch Environment (for timm and HuggingFace frameworks)

```bash
conda create -n ssl-vit-pt python=3.11
conda activate ssl-vit-pt
pip install -r requirements/pytorch.txt
```

### JAX Environment (for JAX/Flax framework)

```bash
conda create -n ssl-vit-jax python=3.11
conda activate ssl-vit-jax
pip install -r requirements/jax.txt
```

---

## End-to-End Workflow

### Phase 0 — Data Preparation

**Significance:** Before any model can learn, the raw local thumbnail corpus must be indexed, filtered, and organized. The file index avoids re-scanning 3.5 million files on every run, and it keeps you from interactively traversing a directory that is too large to inspect safely. Phase 0 is also where you confirm how variable the raw thumbnail sizes are before forcing everything into a common training shape.

#### 0.1 — Build the file index

The first time you run `train.py`, it automatically scans the catalog directory and caches a file index to disk. Because the source directory is massive, treat this as a one-time batch operation and reuse the cached index afterward instead of interactively probing the root. You can trigger the index build manually or point it at a smaller pilot subset:

```bash
# Uses the default catalog path from configs/base.yaml
python scripts/train.py --config configs/mae.yaml --framework timm --epochs 0

# OR point to a local subset for testing
python scripts/train.py --config configs/mae.yaml --framework timm --epochs 0 \
  --catalog_dir "data/10000_highest_SNR_galaxies" \
  --index_path "output/small_index.json"
```

The index is saved to `output/file_index.json` (or whatever `--index_path` you specify). Subsequent runs load the cached index instantly.

#### 0.2 — Visual sanity check

Open 100-200 random thumbnails across width/height bins and background levels and confirm they are usable. For the new corpus, the important questions are whether the files are valid, whether many crops are blank or off-target, and how aggressive the resize/pad policy needs to be. The data pipeline filters images outside the 10-200px dimension range by default (configurable via `data.min_dim` / `data.max_dim` in `configs/base.yaml`).

---

### Phase 1 — Pilot Study: Compare All 9 Method/Size Configurations on a Small Subset

**Significance:** Running all 9 configurations on the full 3.5M-image corpus before knowing which method/size pairing works best is a waste of scarce compute. The pilot uses a small subset (for example 10k thumbnails) to identify the best SSL method/size combination before committing to full-scale training.

#### 1.1 — Train all 9 pilot configurations on the pilot subset

Point `--catalog_dir` at the local thumbnail corpus and use `--max_samples` to cap the pilot set size. Use `--output_dir` to keep each experiment's checkpoints and logs isolated. The current pilot fixes `--framework timm` for all 9 runs so that only SSL method and ViT size vary.

**MAE (3 sizes):**

```bash
python scripts/train.py --config configs/mae.yaml --framework timm --vit_size tiny \
  --catalog_dir "../JWSP-JWST-to-SpArcFiRe/outputs/JWST_galaxy_thumbnails" --max_samples 10000 \
  --output_dir output/experiments/pilot_mae_timm_tiny

python scripts/train.py --config configs/mae.yaml --framework timm --vit_size small \
  --catalog_dir "../JWSP-JWST-to-SpArcFiRe/outputs/JWST_galaxy_thumbnails" --max_samples 10000 \
  --output_dir output/experiments/pilot_mae_timm_small

python scripts/train.py --config configs/mae.yaml --framework timm --vit_size base \
  --catalog_dir "../JWSP-JWST-to-SpArcFiRe/outputs/JWST_galaxy_thumbnails" --max_samples 10000 \
  --output_dir output/experiments/pilot_mae_timm_base
```

**DINO (3 sizes):**

```bash
python scripts/train.py --config configs/dino.yaml --framework timm --vit_size tiny \
  --catalog_dir "../JWSP-JWST-to-SpArcFiRe/outputs/JWST_galaxy_thumbnails" --max_samples 10000 \
  --output_dir output/experiments/pilot_dino_timm_tiny

python scripts/train.py --config configs/dino.yaml --framework timm --vit_size small \
  --catalog_dir "../JWSP-JWST-to-SpArcFiRe/outputs/JWST_galaxy_thumbnails" --max_samples 10000 \
  --output_dir output/experiments/pilot_dino_timm_small

python scripts/train.py --config configs/dino.yaml --framework timm --vit_size base \
  --catalog_dir "../JWSP-JWST-to-SpArcFiRe/outputs/JWST_galaxy_thumbnails" --max_samples 10000 \
  --output_dir output/experiments/pilot_dino_timm_base
```

**MAE→DINO (3 sizes):**

```bash
python scripts/train.py --config configs/mae_dino.yaml --framework timm --vit_size tiny \
  --catalog_dir "../JWSP-JWST-to-SpArcFiRe/outputs/JWST_galaxy_thumbnails" --max_samples 10000 \
  --output_dir output/experiments/pilot_mae_dino_timm_tiny

python scripts/train.py --config configs/mae_dino.yaml --framework timm --vit_size small \
  --catalog_dir "../JWSP-JWST-to-SpArcFiRe/outputs/JWST_galaxy_thumbnails" --max_samples 10000 \
  --output_dir output/experiments/pilot_mae_dino_timm_small

python scripts/train.py --config configs/mae_dino.yaml --framework timm --vit_size base \
  --catalog_dir "../JWSP-JWST-to-SpArcFiRe/outputs/JWST_galaxy_thumbnails" --max_samples 10000 \
  --output_dir output/experiments/pilot_mae_dino_timm_base
```

> **Note:** The codebase still supports `hf` and `jax` backends, but the first pilot should keep the framework fixed. For pilot evaluation, rank configurations primarily by linear-probe performance and embedding structure. Full fine-tuning is most useful on the top 2-3 pilot winners rather than all 9 runs.

#### 1.2 — Extract embeddings from each pilot model

After training, extract CLS token embeddings for downstream evaluation:

```bash
python scripts/extract_embeddings.py \
  --checkpoint output/experiments/pilot_mae_timm_tiny/checkpoints/checkpoint_best.pt \
  --framework timm --method mae --vit_size tiny \
  --catalog_dir "../JWSP-JWST-to-SpArcFiRe/outputs/JWST_galaxy_thumbnails" \
  --max_samples 10000 \
  --output output/experiments/pilot_mae_timm_tiny/embeddings.h5
```

Repeat for all 9 models, substituting the appropriate `--checkpoint`, `--method`, `--vit_size`, and `--output` paths.

#### 1.3 — Evaluate each pilot model (UMAP + linear probe)

**Significance:** The evaluation metric for SSL quality is how well the learned representations separate useful morphology from the mixed sky-patch background _without_ explicit labels. UMAP reveals cluster structure visually; the linear probe measures it quantitatively. The model with the best linear probe accuracy, cleanest UMAP separation, and most consistent cluster representatives wins.

```bash
# Clustering + UMAP for a single model
python scripts/evaluate.py \
  --embeddings output/experiments/pilot_mae_timm_tiny/embeddings.h5 \
  --output_dir output/experiments/pilot_mae_timm_tiny/eval \
  --sweep_k --n_clusters 10

# Interactive cluster labeling (label ~200-500 samples for linear probe)
python scripts/evaluate.py \
  --embeddings output/experiments/pilot_mae_timm_tiny/embeddings.h5 \
  --output_dir output/experiments/pilot_mae_timm_tiny/eval \
  --label --reps_per_cluster 20

# Linear probe on labeled subset
python scripts/evaluate.py \
  --classify --linear_probe \
  --labels_csv output/experiments/pilot_mae_timm_tiny/eval/labels.csv \
  --checkpoint output/experiments/pilot_mae_timm_tiny/checkpoints/checkpoint_best.pt \
  --framework timm --method mae --vit_size tiny
```

#### 1.4 — Compare all 9 pilot models

```bash
python scripts/compare_experiments.py --results_dir output/experiments
```

This generates a comparison table showing final loss, silhouette score, and classification accuracy across all 9 pilot configurations. **Pick the winner** — the method/size pair with the best linear probe accuracy and UMAP quality. If needed, only then widen the comparison to alternative frameworks.

---

### Phase 2 — Full SSL Pretraining

**Significance:** The pilot identified the best configuration. Now that winner is trained on the full local thumbnail corpus to learn the richest possible morphological representations. The full dataset gives the model orders of magnitude more visual diversity, including rare morphologies and edge cases that the 10k pilot cannot capture.

#### 2.1 — Train the winning model on the full corpus

Replace `[METHOD]`, `[SIZE]` with your pilot winner (e.g., `mae`, `small`):

```bash
python scripts/train.py \
  --config configs/[METHOD].yaml \
  --framework timm --vit_size [SIZE] \
  --catalog_dir "../JWSP-JWST-to-SpArcFiRe/outputs/JWST_galaxy_thumbnails" \
  --output_dir output/experiments/full_[METHOD]_timm_[SIZE] \
  --wandb
```

For example, if MAE Small won the pilot:

```bash
python scripts/train.py \
  --config configs/mae.yaml \
  --framework timm --vit_size small \
  --catalog_dir "../JWSP-JWST-to-SpArcFiRe/outputs/JWST_galaxy_thumbnails" \
  --output_dir output/experiments/full_mae_timm_small \
  --wandb
```

For multi-GPU training:

```bash
torchrun --nproc_per_node=4 scripts/train.py \
  --config configs/mae.yaml --framework timm --vit_size small \
  --distributed --wandb \
  --output_dir output/experiments/full_mae_timm_small
```

#### 2.2 — Extract CLS embeddings for ALL sources

```bash
python scripts/extract_embeddings.py \
  --checkpoint output/experiments/full_mae_timm_small/checkpoints/checkpoint_best.pt \
  --framework timm --method mae --vit_size small \
  --output output/experiments/full_mae_timm_small/embeddings.h5
```

---

### Phase 3 — Unsupervised Exploration (Science Result #1)

**Significance:** This phase answers the core research question: _"What did the model learn? Does it group galaxies by morphology without being told to?"_ Discovering morphological structure from JWST imaging without any human labels is a publishable result on its own — it demonstrates that SSL ViTs can autonomously organize galaxy populations by visual structure.

#### 3.1 — UMAP visualization + k-means clustering

```bash
# Run clustering with an elbow sweep to find the optimal k
python scripts/evaluate.py \
  --embeddings output/experiments/full_mae_timm_small/embeddings.h5 \
  --output_dir output/experiments/full_mae_timm_small/eval \
  --sweep_k --k_min 5 --k_max 50 --k_step 5 \
  --umap_max_samples 50000
```

This produces:

- UMAP scatter plot colored by cluster membership
- Elbow curve (inertia vs. k) and silhouette scores
- Cluster assignments saved to HDF5

#### 3.2 — Inspect cluster representatives

```bash
python scripts/evaluate.py \
  --embeddings output/experiments/full_mae_timm_small/embeddings.h5 \
  --output_dir output/experiments/full_mae_timm_small/eval \
  --label --reps_per_cluster 20
```

For each cluster, the tool displays the images closest to the cluster center. You manually identify what each cluster contains: "cluster 7 = edge-on disks", "cluster 12 = spirals", "cluster 3 = compact ellipticals", etc. This is how unsupervised morphological taxonomy emerges.

---

### Phase 4 — Supervised Fine-Tuning for Spiral Classification (Science Result #2)

**Significance:** SSL pretraining gave the model a visual understanding of morphology and background structure. Now a small set of labeled examples teaches it the specific spiral-classification task. Because the pretrained model already _knows_ what spiral structure looks like, it should achieve strong accuracy with far fewer labels than training from scratch.

#### 4.1 — Prepare a labeled evaluation set

Because the new training corpus is a random sky-patch collection rather than a curated COSMOS catalog, treat labels as a separate evaluation asset. Practical options are:

- cluster-then-verify on pilot embeddings to build a clean spiral vs. non-spiral set
- manual labeling of a stratified thumbnail sample drawn from the pilot subset
- external cross-match only where reliable metadata exists for the sampled thumbnails

Save the result as a CSV with columns: `file_path`, `label`.

#### 4.2 — Linear probe (freeze encoder, train classification head only)

```bash
python scripts/evaluate.py \
  --classify --linear_probe \
  --labels_csv output/eval/galaxy_zoo_labels.csv \
  --checkpoint output/experiments/full_mae_timm_small/checkpoints/checkpoint_best.pt \
  --framework timm --method mae --vit_size small \
  --classify_epochs 50 --classify_lr 1e-3 \
  --output_dir output/experiments/full_mae_timm_small/classification
```

The linear probe is the primary SSL evaluation metric — it measures how much morphological information the encoder captured. A strong linear probe result is a key thesis finding.

#### 4.3 — Full fine-tuning (unfreeze encoder, low learning rate)

```bash
python scripts/evaluate.py \
  --classify --fine_tune \
  --labels_csv output/eval/galaxy_zoo_labels.csv \
  --checkpoint output/experiments/full_mae_timm_small/checkpoints/checkpoint_best.pt \
  --framework timm --method mae --vit_size small \
  --classify_epochs 50 --classify_lr 1e-5 \
  --output_dir output/experiments/full_mae_timm_small/classification
```

Compare linear probe vs. fine-tune accuracy to quantify the gap, and compare both against training a ViT from scratch (no pretraining) as a baseline.

---

### Phase 5 — Experiment Comparison & Final Results

**Significance:** Aggregating results across all configurations produces the thesis's main comparison table — which SSL method, which ViT size, and (optionally) which framework yielded the best galaxy morphology representations. This is the quantitative evidence supporting your conclusions.

```bash
python scripts/compare_experiments.py --results_dir output/experiments
```

This generates:

- A comparison table: pretraining loss convergence, silhouette score, linear probe accuracy, fine-tune accuracy
- Bar charts and plots saved to `output/comparison.png`
- Raw metrics as JSON for further analysis

---

## Complete Ordered Checklist

```
PHASE 0 — DATA PREPARATION
  [ ] Build file index (automatic on first train.py run, or run with --epochs 0)
  [ ] Verify output: check index row count, spot-check thumbnails

PHASE 1 — PILOT MODEL SELECTION (10k subset)
  [ ] Train all 9 pilot configurations on pilot subset (method × framework)
  [ ] Extract embeddings from each trained model
  [ ] Generate UMAP for each model — visually compare cluster separation
  [ ] Run linear probe on small manually-labeled subset for each model
  [ ] Optionally fine-tune only the top 2-3 pilot models
  [ ] Compare all 9 pilots — select winning configuration

PHASE 2 — FULL SSL PRETRAINING
  [ ] Train winning model on full thumbnail corpus (1–3 days)
  [ ] Extract CLS embeddings for all sources

PHASE 3 — UNSUPERVISED EXPLORATION (science result #1)
  [ ] UMAP on 50k subsample, colored by cluster membership
  [ ] k-means sweep k=5 to 50, plot elbow curve
  [ ] For best k: manually inspect representatives per cluster
  [ ] Assign morphological labels to clusters
  [ ] → "SSL ViT discovers galaxy morphology from JWST imaging without labels"

PHASE 4 — SUPERVISED FINE-TUNING (science result #2)
  [ ] Build or import a labeled evaluation set for a sampled subset
  [ ] Assign spiral/non-spiral labels first, then add spin-direction labels later
  [ ] Linear probe: freeze encoder, train classification head
  [ ] Full fine-tune: unfreeze encoder, low learning rate
  [ ] Evaluate: accuracy, confusion matrix, per-class F1, ROC-AUC
  [ ] → "JWST-pretrained ViT classifies spiral morphology with X% accuracy"

PHASE 5 — COMPARISON & REPORTING
  [ ] Run compare_experiments.py on all experiment directories
  [ ] Generate final comparison tables and plots
```

---

## SSL Methods

### MAE (Masked Autoencoder)

Masks 75% of image patches randomly. The encoder processes only visible patches (computationally efficient), and a lightweight decoder (4 layers, 128d) reconstructs the masked patches. The loss is MSE on normalized pixel values of the masked regions. This forces the model to learn rich internal representations of galaxy structure to "fill in" what it can't see.

### DINO (Self-Distillation with No Labels)

A student-teacher framework where the teacher is an exponential moving average (EMA) of the student. Multi-crop augmentation produces 2 global (64×64) and several local (32×32) views of the same galaxy. The student must match the teacher's output distribution on all views, while the teacher only sees global views. Centering + sharpening prevent mode collapse. DINO learns instance-level features — it understands that different views of the _same_ galaxy should produce similar representations.

### MAE → DINO (Two-Stage)

Stage 1 runs MAE pretraining to learn general patch-level features (what do galaxy patches look like?). Stage 2 initializes DINO from the MAE encoder and runs self-distillation to refine instance-level alignment (what makes this particular galaxy identifiable?). This combines reconstruction-based and contrastive learning for potentially richer representations.

---

## Configuration

All training is configured through YAML files in `configs/`. The base configuration (`configs/base.yaml`) defines shared defaults; method-specific configs override individual settings.

### Key Configuration Options

| Config Key                 | Default                                                    | Description                           |
| -------------------------- | ---------------------------------------------------------- | ------------------------------------- |
| `data.catalog_dir`         | `../JWSP-JWST-to-SpArcFiRe/outputs/JWST_galaxy_thumbnails` | Root directory of the local thumbnail corpus |
| `data.index_path`          | `output/file_index.json`                                   | Cached file index path                |
| `data.image_size`          | `64`                                                       | Target image size (square)            |
| `data.max_samples`         | `100000`                                                   | Subset size (`null` = all)            |
| `data.min_dim` / `max_dim` | `10` / `200`                                               | Dimension filter range                |
| `model.vit_size`           | `tiny`                                                     | ViT variant: `tiny`, `small`, `base`  |
| `model.patch_size`         | `8`                                                        | Patch size for tokenization           |
| `training.epochs`          | `30`                                                       | Number of training epochs             |
| `training.learning_rate`   | `1.5e-4`                                                   | Base learning rate                    |
| `training.mixed_precision` | `true`                                                     | fp16 (PyTorch) / bf16 (JAX)           |
| `ssl.method`               | `mae`                                                      | SSL method: `mae`, `dino`, `mae_dino` |
| `ssl.framework`            | `timm`                                                     | Framework: `timm`, `hf`, `jax`        |

Any config value can be overridden via CLI flags (e.g., `--vit_size small`, `--epochs 200`, `--batch_size 128`).

---

## Data

- **Source**: local JWST thumbnail corpus cropped from random MAST sky patches
- **Format**: single-channel thumbnail files consumed by the shared loader/indexer
- **Scale**: ~3.5 million thumbnails
- **Dimensions**: variable raw dimensions; profile first, then resize/pad to 64x64 for model input
- **Normalization**: percentile clipping or comparable robust normalization to handle extreme dynamic-range outliers
- **Augmentations**: random rotation (0°/90°/180°/270°), flips, Gaussian noise, brightness/contrast jitter; no color jitter for single-channel data

### Operational Constraints

- Never browse the 3.5M-file root interactively; build and reuse a cached index or manifest.
- For Google Colab, stage pilot subsets or sharded archives instead of pointing Colab at millions of loose files over the network.
- Expect a meaningful fraction of thumbnails to be blank, off-target, or non-galaxy patches; quantify that in Phase 0 before large training runs.

### SNR Filtering Guidance

- **SSL pretraining**: Don't filter aggressively — SSL benefits from volume and diversity
- **Clustering analysis**: Filter moderately (SNR > 10) for interpretable clusters
- **Supervised fine-tuning**: Filter strictly (SNR > 20, unblended) for trustworthy labels

---

## References

- He, K. et al. "Masked Autoencoders Are Scalable Vision Learners." CVPR 2022.
- Caron, M. et al. "Emerging Properties in Self-Supervised Vision Transformers." ICCV 2021.
- Dosovitskiy, A. et al. "An Image is Worth 16x16 Words." ICLR 2021.
