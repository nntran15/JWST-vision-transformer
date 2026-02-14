#!/usr/bin/env python3
"""
End-to-end evaluation pipeline.

Runs clustering, UMAP visualization, and optional classification
on extracted SSL embeddings.

Usage:
  # Full pipeline: clustering + UMAP + elbow plot
  python scripts/evaluate.py \
    --embeddings output/embeddings.h5 \
    --output_dir output/eval

  # Cluster-then-verify labeling
  python scripts/evaluate.py \
    --embeddings output/embeddings.h5 \
    --label --output_dir output/eval

  # Linear probe / fine-tune (requires labeled CSV)
  python scripts/evaluate.py \
    --classify --labels_csv output/eval/labels.csv \
    --checkpoint output/checkpoints/checkpoint_best.pt \
    --framework timm --method mae
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)


def run_clustering_pipeline(args):
    """Run k-means clustering, UMAP, and visualization."""
    from src.evaluation.clustering import (
        load_embeddings, run_kmeans, sweep_k, compute_umap,
        plot_umap_clusters, plot_elbow, save_cluster_results,
    )

    # Load embeddings
    data = load_embeddings(args.embeddings)
    embeddings = data["embeddings"]
    paths = data["paths"]
    metadata = data["metadata"]

    logger.info(
        f"Loaded {embeddings.shape[0]} embeddings (dim={embeddings.shape[1]}) "
        f"from {metadata.get('method', '?')}/{metadata.get('framework', '?')}"
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Elbow analysis
    if args.sweep_k:
        sweep_results = sweep_k(
            embeddings,
            k_range=(args.k_min, args.k_max),
            step=args.k_step,
            seed=args.seed,
        )
        plot_elbow(
            sweep_results["ks"],
            sweep_results["inertias"],
            sweep_results["silhouettes"],
            str(output_dir / "elbow_plot.png"),
        )
        n_clusters = sweep_results["best_k"]
        logger.info(f"Best k from sweep: {n_clusters}")
    else:
        n_clusters = args.n_clusters

    # Run k-means
    km_results = run_kmeans(embeddings, n_clusters=n_clusters, seed=args.seed)

    # UMAP
    projection = compute_umap(
        embeddings,
        n_neighbors=args.umap_neighbors,
        min_dist=args.umap_min_dist,
        max_samples=args.umap_max_samples,
        seed=args.seed,
    )

    # Plot
    method_name = metadata.get("method", "ssl")
    framework_name = metadata.get("framework", "")
    vit_name = metadata.get("vit_size", "")
    title = f"UMAP — {method_name.upper()} ({framework_name}, {vit_name}), k={n_clusters}"

    plot_umap_clusters(
        projection,
        km_results["labels"],
        str(output_dir / "umap_clusters.png"),
        title=title,
    )

    # Save
    save_cluster_results(
        str(output_dir / "cluster_results.h5"),
        paths,
        km_results["labels"],
        embeddings,
        projection,
    )

    return km_results, paths


def run_labeling(args, km_results, paths):
    """Interactive cluster labeling."""
    from src.evaluation.annotation_tool import (
        get_cluster_representatives, interactive_label_clusters,
        propagate_labels,
    )
    from src.evaluation.clustering import load_embeddings

    data = load_embeddings(args.embeddings)

    reps = get_cluster_representatives(
        data["embeddings"], km_results["labels"], paths,
        n_per_cluster=args.reps_per_cluster,
    )

    output_dir = Path(args.output_dir)
    labels_csv = str(output_dir / "cluster_labels.csv")

    interactive_label_clusters(reps, labels_csv)

    # Propagate to full dataset
    full_csv = str(output_dir / "labels.csv")
    propagate_labels(labels_csv, km_results["labels"], paths, full_csv)

    return full_csv


def run_classification(args):
    """Train linear probe or fine-tuner on labeled data."""
    import torch
    from src.evaluation.classifier import (
        LinearProbe, FineTuner, train_classifier, create_train_val_loaders,
    )
    from src.models.vit_config import get_vit_config

    if not args.labels_csv:
        logger.error("--labels_csv required for classification")
        return

    # Build encoder
    vit_config = get_vit_config(size=args.vit_size, image_size=64, patch_size=8, in_channels=1)

    if args.method == "mae":
        if args.framework == "timm":
            from src.models.pytorch_timm.mae import MAE
        else:
            from src.models.pytorch_hf.mae import MAE
        ssl_model = MAE(config=vit_config)
    elif args.method == "dino":
        if args.framework == "timm":
            from src.models.pytorch_timm.dino import DINO
        else:
            from src.models.pytorch_hf.dino import DINO
        ssl_model = DINO(config=vit_config)

    # Load SSL checkpoint
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if "model_state_dict" in state:
        ssl_model.load_state_dict(state["model_state_dict"], strict=False)
    else:
        ssl_model.load_state_dict(state, strict=False)

    encoder = ssl_model.get_encoder()

    # Create data loaders
    train_loader, val_loader, label_to_idx = create_train_val_loaders(
        args.labels_csv, batch_size=args.classify_batch_size, seed=args.seed,
    )
    n_classes = len(label_to_idx)

    output_dir = Path(args.output_dir)

    # Linear probe
    if args.linear_probe or not args.fine_tune:
        logger.info("=== LINEAR PROBE ===")
        probe = LinearProbe(encoder, vit_config.embed_dim, n_classes)
        probe_results = train_classifier(
            probe, train_loader, val_loader,
            epochs=args.classify_epochs,
            lr=args.classify_lr,
            device=args.device,
            output_dir=str(output_dir / "linear_probe"),
        )
        logger.info(f"Linear probe best val acc: {probe_results['best_val_acc']:.4f}")

    # Fine-tuning
    if args.fine_tune:
        logger.info("=== FINE-TUNING ===")
        # Reload encoder (unfrozen)
        encoder_ft = ssl_model.get_encoder()
        tuner = FineTuner(encoder_ft, vit_config.embed_dim, n_classes)
        ft_results = train_classifier(
            tuner, train_loader, val_loader,
            epochs=args.classify_epochs,
            lr=args.classify_lr,
            backbone_lr=args.classify_lr * 0.1,
            device=args.device,
            output_dir=str(output_dir / "fine_tune"),
        )
        logger.info(f"Fine-tune best val acc: {ft_results['best_val_acc']:.4f}")


def main():
    parser = argparse.ArgumentParser(description="SSL Evaluation Pipeline")

    # Input
    parser.add_argument("--embeddings", type=str, default="output/embeddings.h5")
    parser.add_argument("--output_dir", type=str, default="output/eval")

    # Clustering
    parser.add_argument("--n_clusters", type=int, default=10)
    parser.add_argument("--sweep_k", action="store_true", help="Sweep over k values")
    parser.add_argument("--k_min", type=int, default=5)
    parser.add_argument("--k_max", type=int, default=50)
    parser.add_argument("--k_step", type=int, default=5)

    # UMAP
    parser.add_argument("--umap_neighbors", type=int, default=15)
    parser.add_argument("--umap_min_dist", type=float, default=0.1)
    parser.add_argument("--umap_max_samples", type=int, default=50000)

    # Labeling
    parser.add_argument("--label", action="store_true", help="Run interactive labeling")
    parser.add_argument("--reps_per_cluster", type=int, default=20)

    # Classification
    parser.add_argument("--classify", action="store_true", help="Train classifier")
    parser.add_argument("--labels_csv", type=str, help="Path to labeled CSV")
    parser.add_argument("--checkpoint", type=str, help="SSL checkpoint for classification")
    parser.add_argument("--framework", type=str, choices=["timm", "hf"])
    parser.add_argument("--method", type=str, choices=["mae", "dino"])
    parser.add_argument("--vit_size", type=str, default="tiny")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--linear_probe", action="store_true")
    parser.add_argument("--fine_tune", action="store_true")
    parser.add_argument("--classify_epochs", type=int, default=50)
    parser.add_argument("--classify_lr", type=float, default=1e-3)
    parser.add_argument("--classify_batch_size", type=int, default=64)

    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # Stage 1: Clustering + UMAP
    if not args.classify:
        km_results, paths = run_clustering_pipeline(args)

        # Stage 2: Interactive labeling
        if args.label:
            labels_csv = run_labeling(args, km_results, paths)
            logger.info(f"Labels saved to {labels_csv}")

    # Stage 3: Classification
    if args.classify:
        run_classification(args)


if __name__ == "__main__":
    main()
