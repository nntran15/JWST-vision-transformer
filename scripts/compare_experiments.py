#!/usr/bin/env python3
"""
Compare results across all 9 experiment configurations.

Loads embeddings/cluster results from each config and generates
a comparison table and comparative UMAP plots.

Usage:
  python scripts/compare_experiments.py --results_dir output/experiments
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

# All 9 configurations
CONFIGS = [
    {"method": "mae", "framework": fw}
    for fw in ("timm", "hf", "jax")
] + [
    {"method": "dino", "framework": fw}
    for fw in ("timm", "hf", "jax")
] + [
    {"method": "mae_dino", "framework": fw}
    for fw in ("timm", "hf", "jax")
]


def collect_metrics(results_dir: str) -> list:
    """
    Collect metrics from all experiment directories.

    Expects structure:
      results_dir/
        mae_timm_tiny/
          embeddings.h5
          eval/cluster_results.h5
          train.log
        dino_hf_small/
          ...
    """
    results_dir = Path(results_dir)
    all_metrics = []

    for exp_dir in sorted(results_dir.iterdir()):
        if not exp_dir.is_dir():
            continue

        metric = {"name": exp_dir.name}

        # Parse experiment name
        parts = exp_dir.name.split("_")
        if len(parts) >= 2:
            metric["method"] = parts[0]
            metric["framework"] = parts[1]
            metric["vit_size"] = parts[2] if len(parts) > 2 else "tiny"

        # Load embedding metadata
        emb_path = exp_dir / "embeddings.h5"
        if emb_path.exists():
            import h5py
            with h5py.File(emb_path, "r") as f:
                metric["n_samples"] = f["embeddings"].shape[0]
                metric["embed_dim"] = f["embeddings"].shape[1]

        # Load cluster metrics
        cluster_path = exp_dir / "eval" / "cluster_results.h5"
        if cluster_path.exists():
            import h5py
            with h5py.File(cluster_path, "r") as f:
                if "labels" in f:
                    labels = f["labels"][:]
                    metric["n_clusters"] = len(np.unique(labels))

        # Load training log for final loss
        log_path = exp_dir / "train.log"
        if log_path.exists():
            last_loss = _parse_last_loss(log_path)
            if last_loss is not None:
                metric["final_loss"] = last_loss

        # Silhouette score from cluster analysis
        sil_path = exp_dir / "eval" / "silhouette.json"
        if sil_path.exists():
            with open(sil_path) as f:
                sil_data = json.load(f)
                metric["silhouette"] = sil_data.get("silhouette", None)

        # Classification results
        for eval_type in ("linear_probe", "fine_tune"):
            report_path = exp_dir / "eval" / eval_type / "classification_report.txt"
            if report_path.exists():
                acc = _parse_accuracy(report_path)
                if acc is not None:
                    metric[f"{eval_type}_acc"] = acc

        all_metrics.append(metric)

    return all_metrics


def _parse_last_loss(log_path: Path) -> float | None:
    """Parse the final training loss from a log file."""
    last_loss = None
    with open(log_path) as f:
        for line in f:
            if "loss=" in line.lower() or "train_loss" in line.lower():
                # Try to extract numeric value
                for part in line.split():
                    if part.startswith("loss=") or part.startswith("train_loss="):
                        try:
                            last_loss = float(part.split("=")[1].rstrip(","))
                        except ValueError:
                            pass
    return last_loss


def _parse_accuracy(report_path: Path) -> float | None:
    """Parse accuracy from a classification report file."""
    with open(report_path) as f:
        for line in f:
            if "accuracy" in line.lower():
                parts = line.strip().split()
                for p in parts:
                    try:
                        val = float(p)
                        if 0 <= val <= 1:
                            return val
                    except ValueError:
                        pass
    return None


def print_comparison_table(metrics: list):
    """Print a formatted comparison table."""
    if not metrics:
        print("No experiment results found.")
        return

    # Determine columns
    all_keys = set()
    for m in metrics:
        all_keys.update(m.keys())
    all_keys.discard("name")

    columns = ["name", "method", "framework", "vit_size", "embed_dim",
               "final_loss", "silhouette", "linear_probe_acc", "fine_tune_acc"]
    columns = [c for c in columns if c in all_keys or c == "name"]

    # Header
    header = " | ".join(f"{c:>18s}" for c in columns)
    print("\n" + "=" * len(header))
    print("EXPERIMENT COMPARISON")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    # Rows
    for m in metrics:
        row = []
        for c in columns:
            val = m.get(c, "—")
            if isinstance(val, float):
                row.append(f"{val:>18.4f}")
            else:
                row.append(f"{str(val):>18s}")
        print(" | ".join(row))

    print("=" * len(header) + "\n")


def plot_comparison(metrics: list, output_path: str):
    """Generate a bar-chart comparison of key metrics across experiments."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [m.get("name", "?") for m in metrics]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Final training loss
    losses = [m.get("final_loss", 0) for m in metrics]
    if any(l > 0 for l in losses):
        axes[0].barh(names, losses, color="steelblue")
        axes[0].set_title("Final Training Loss")
        axes[0].set_xlabel("Loss")

    # Silhouette
    sils = [m.get("silhouette", 0) for m in metrics]
    if any(s > 0 for s in sils):
        axes[1].barh(names, sils, color="coral")
        axes[1].set_title("Silhouette Score")
        axes[1].set_xlabel("Score")

    # Classification accuracy
    accs = [m.get("linear_probe_acc", m.get("fine_tune_acc", 0)) for m in metrics]
    if any(a > 0 for a in accs):
        axes[2].barh(names, accs, color="seagreen")
        axes[2].set_title("Classification Accuracy")
        axes[2].set_xlabel("Accuracy")

    fig.suptitle("SSL ViT — Experiment Comparison", fontsize=14)
    fig.tight_layout()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Comparison plot saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare SSL Experiments")
    parser.add_argument("--results_dir", type=str, default="output/experiments",
                        help="Directory containing experiment subdirectories")
    parser.add_argument("--output", type=str, default="output/comparison.png",
                        help="Output comparison plot path")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    metrics = collect_metrics(args.results_dir)
    print_comparison_table(metrics)
    if metrics:
        plot_comparison(metrics, args.output)

    # Save raw metrics as JSON
    json_path = Path(args.output).with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    logger.info(f"Raw metrics saved to {json_path}")


if __name__ == "__main__":
    main()
