#!/usr/bin/env python3
"""
End-to-end evaluation pipeline.

Runs clustering and UMAP by default, with optional k-sweep,
interactive cluster labeling, and classification on extracted SSL embeddings.

Usage:
    # Default evaluation: clustering + UMAP only
    python scripts/evaluate.py \
        --embeddings output/experiments/pilot_mae_timm_tiny/embeddings.h5 \
        --output_dir output/eval

    # Add elbow plot by sweeping k
    python scripts/evaluate.py \
        --embeddings output/experiments/pilot_mae_timm_tiny/embeddings.h5 \
        --output_dir output/eval \
        --sweep_k --k_min 2 --k_max 10 --k_step 1 --k_selection elbow

    # Show representative images and label clusters interactively
    python scripts/evaluate.py \
        --embeddings output/experiments/pilot_mae_timm_tiny/embeddings.h5 \
        --output_dir output/eval \
        --sweep_k --label --reps_per_cluster 20

    # Linear probe / fine-tune (requires labeled CSV)
    python scripts/evaluate.py \
        --classify --labels_csv output/eval/labels.csv \
        --checkpoint output/checkpoints/checkpoint_best.pt \
        --framework timm --method mae
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)
LEGACY_DEFAULT_EMBEDDINGS = Path("output/embeddings.h5")


def _discover_experiment_embeddings(root_dir: Path, limit: int | None = None) -> list[Path]:
    """Return experiment-scoped embedding artifacts sorted by newest first."""
    experiments_dir = root_dir / "output" / "experiments"
    if not experiments_dir.exists():
        return []

    candidates = sorted(
        (path for path in experiments_dir.glob("*/embeddings.h5") if path.is_file()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if limit is not None:
        return candidates[:limit]
    return candidates


def _format_embedding_candidates(candidates: list[Path], root_dir: Path) -> str:
    if not candidates:
        return ""

    lines = []
    for candidate in candidates:
        try:
            display_path = candidate.relative_to(root_dir)
        except ValueError:
            display_path = candidate
        lines.append(f"  - {display_path}")
    return "\n".join(lines)


def resolve_embeddings_path(requested_path: str, root_dir: Path | None = None) -> Path:
    """Resolve the requested embeddings file, including the legacy default path."""
    root_dir = root_dir or project_root
    path = Path(requested_path)
    if path.exists():
        return path

    candidates = _discover_experiment_embeddings(root_dir)
    if path == LEGACY_DEFAULT_EMBEDDINGS and candidates:
        selected_path = candidates[0]
        logger.warning(
            "Embeddings not found at %s; using newest experiment artifact %s",
            path,
            selected_path,
        )
        return selected_path

    hint = ""
    if candidates:
        hint = (
            "\nAvailable experiment embeddings:\n"
            f"{_format_embedding_candidates(candidates[:5], root_dir)}"
            "\nPass one of these paths with --embeddings."
        )

    raise FileNotFoundError(f"Embeddings file not found: {path}{hint}")


def resolve_artifact_path(
    requested_path: str,
    *,
    output_dir: Path | None = None,
    root_dir: Path | None = None,
    kind: str = "artifact",
) -> Path:
    """Resolve a checkpoint/CSV path from common local and experiment-relative locations."""
    root_dir = root_dir or project_root
    path = Path(requested_path)

    if path.exists():
        return path

    candidates: list[Path] = []
    if not path.is_absolute():
        candidates.append(root_dir / path)

    if output_dir is not None:
        output_dir = Path(output_dir)
        candidates.extend([
            output_dir / path,
            output_dir.parent / path,
            output_dir.parent / "checkpoints" / path.name,
        ])

        if output_dir.parent.name != "checkpoints":
            checkpoint_root = output_dir.parent / "checkpoints"
            if checkpoint_root.exists():
                for match in sorted(checkpoint_root.rglob(path.name)):
                    if match.is_file():
                        logger.info("Resolved %s path %s -> %s", kind, requested_path, match)
                        return match

    # Common experiment-local layout: output/experiments/<run>/checkpoints/<file>
    candidates.extend([
        root_dir / "output" / "checkpoints" / path.name,
        root_dir / "output" / "experiments" / "checkpoints" / path.name,
    ])

    experiments_dir = root_dir / "output" / "experiments"
    if experiments_dir.exists():
        for match in sorted(experiments_dir.rglob(path.name)):
            if match.is_file() and "checkpoints" in match.parts:
                logger.info("Resolved %s path %s -> %s", kind, requested_path, match)
                return match

    for candidate in candidates:
        if candidate.exists():
            logger.info("Resolved %s path %s -> %s", kind, requested_path, candidate)
            return candidate

    searched = "\n".join(f"  - {candidate}" for candidate in candidates)
    raise FileNotFoundError(
        f"Could not resolve {kind} path: {requested_path}\nSearched:\n{searched}"
    )


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
            selection_strategy=args.k_selection,
        )
        plot_elbow(
            sweep_results["ks"],
            sweep_results["inertias"],
            sweep_results["silhouettes"],
            str(output_dir / "elbow_plot.png"),
            selected_k=sweep_results["best_k"],
            selection_strategy=sweep_results["selection_strategy"],
        )
        n_clusters = sweep_results["best_k"]
        logger.info(
            "Selected k from sweep: %s (%s strategy; max silhouette at k=%s)",
            n_clusters,
            sweep_results["selection_strategy"],
            sweep_results["best_silhouette_k"],
        )
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

    interactive_label_clusters(
        reps,
        labels_csv,
        preview_dir=str(output_dir / "cluster_representatives"),
    )

    # Propagate to full dataset
    full_csv = str(output_dir / "labels.csv")
    propagate_labels(labels_csv, km_results["labels"], paths, full_csv)

    return full_csv


def run_classification(args):
    """Train linear probe or fine-tuner on labeled data."""
    import torch
    from src.evaluation.classifier import (
        LinearProbe, FineTuner, create_train_val_test_loaders, train_classifier,
    )
    from src.models.vit_config import get_vit_config

    if not args.labels_csv:
        logger.error("--labels_csv required for classification")
        return

    labels_csv_path = resolve_artifact_path(
        args.labels_csv,
        output_dir=Path(args.output_dir),
        kind="labels CSV",
    )

    checkpoint_path = resolve_artifact_path(
        args.checkpoint,
        output_dir=Path(args.output_dir),
        kind="checkpoint",
    )

    split_artifact_path = None
    if args.split_artifact:
        split_artifact_path = resolve_artifact_path(
            args.split_artifact,
            output_dir=Path(args.output_dir),
            kind="split artifact",
        )

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
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "model_state_dict" in state:
        ssl_model.load_state_dict(state["model_state_dict"], strict=False)
    else:
        ssl_model.load_state_dict(state, strict=False)

    encoder = ssl_model.get_encoder()

    # Create data loaders
    train_loader, val_loader, test_loader, label_to_idx, class_weights, split_metadata = create_train_val_test_loaders(
        str(labels_csv_path),
        val_fraction=args.classify_val_fraction,
        test_fraction=args.classify_test_fraction,
        batch_size=args.classify_batch_size,
        seed=args.seed,
        balanced_sampling=args.classify_balanced_sampling,
        split_artifact=str(split_artifact_path) if split_artifact_path is not None else None,
    )
    n_classes = len(label_to_idx)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "classification_split_summary.json").write_text(
        json.dumps(split_metadata, indent=2)
    )
    logger.info("Classification split summary: %s", json.dumps(split_metadata, sort_keys=True))

    report_loader = test_loader if test_loader is not None else val_loader
    report_split_name = "test" if test_loader is not None else "val"

    # Linear probe
    if args.linear_probe or not args.fine_tune:
        logger.info("=== LINEAR PROBE ===")
        probe = LinearProbe(encoder, vit_config.embed_dim, n_classes)
        probe_results = train_classifier(
            probe,
            train_loader,
            val_loader,
            report_loader=report_loader,
            epochs=args.classify_epochs,
            lr=args.classify_lr,
            device=args.device,
            output_dir=str(output_dir / "linear_probe"),
            class_weights=class_weights,
            selection_metric=args.classify_selection_metric,
            report_split_name=report_split_name,
        )
        logger.info(
            "Linear probe best checkpoint: epoch=%s, %s=%.4f",
            probe_results["best_epoch"],
            probe_results["best_selection_metric"],
            probe_results["best_selection_score"],
        )

    # Fine-tuning
    if args.fine_tune:
        logger.info("=== FINE-TUNING ===")
        # Reload encoder (unfrozen)
        encoder_ft = ssl_model.get_encoder()
        tuner = FineTuner(encoder_ft, vit_config.embed_dim, n_classes)
        ft_results = train_classifier(
            tuner,
            train_loader,
            val_loader,
            report_loader=report_loader,
            epochs=args.classify_epochs,
            lr=args.classify_lr,
            backbone_lr=args.classify_lr * 0.1,
            device=args.device,
            output_dir=str(output_dir / "fine_tune"),
            class_weights=class_weights,
            selection_metric=args.classify_selection_metric,
            report_split_name=report_split_name,
        )
        logger.info(
            "Fine-tune best checkpoint: epoch=%s, %s=%.4f",
            ft_results["best_epoch"],
            ft_results["best_selection_metric"],
            ft_results["best_selection_score"],
        )


def main():
    parser = argparse.ArgumentParser(description="SSL Evaluation Pipeline")

    # Input
    parser.add_argument(
        "--embeddings",
        type=str,
        default="output/embeddings.h5",
        help=(
            "Embeddings HDF5 path. If the legacy default is missing, "
            "the newest output/experiments/*/embeddings.h5 is used."
        ),
    )
    parser.add_argument("--output_dir", type=str, default="output/eval")

    # Clustering
    parser.add_argument("--n_clusters", type=int, default=4)
    parser.add_argument(
        "--sweep_k",
        action="store_true",
        help="Sweep over k values and save output_dir/elbow_plot.png before choosing best k.",
    )
    parser.add_argument(
        "--k_selection",
        type=str,
        choices=["elbow", "silhouette"],
        default="elbow",
        help="How to choose k after a sweep. 'elbow' is safer for morphology labeling than raw max silhouette.",
    )
    parser.add_argument("--k_min", type=int, default=2)
    parser.add_argument("--k_max", type=int, default=10)
    parser.add_argument("--k_step", type=int, default=1)

    # UMAP
    parser.add_argument("--umap_neighbors", type=int, default=15)
    parser.add_argument("--umap_min_dist", type=float, default=0.1)
    parser.add_argument("--umap_max_samples", type=int, default=50000)

    # Labeling
    parser.add_argument(
        "--label",
        action="store_true",
        help="Prompt for cluster labels and save representative grids under output_dir/cluster_representatives.",
    )
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
    parser.add_argument(
        "--classify_val_fraction",
        type=float,
        default=0.2,
        help="Validation fraction when no frozen split artifact is provided.",
    )
    parser.add_argument(
        "--classify_test_fraction",
        type=float,
        default=0.0,
        help="Test fraction when no frozen split artifact is provided.",
    )
    parser.add_argument(
        "--classify_selection_metric",
        type=str,
        choices=["accuracy", "macro_f1", "macro_recall", "min_class_f1", "spiral_f1", "spiral_recall"],
        default="macro_f1",
        help="Metric used to choose the best classifier checkpoint.",
    )
    parser.add_argument(
        "--classify_balanced_sampling",
        action="store_true",
        help="Use weighted random sampling on the training split for imbalanced labels.",
    )
    parser.add_argument(
        "--split_artifact",
        type=str,
        help="Path to a frozen split artifact directory or split_manifest.json file.",
    )

    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # Stage 1: Clustering + UMAP
    if not args.classify:
        args.embeddings = str(resolve_embeddings_path(args.embeddings))
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
