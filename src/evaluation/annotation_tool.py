"""
Matplotlib-based annotation tool for cluster-then-verify labeling.

Displays representative samples from each cluster for human verification.
Outputs labeled data as a CSV for downstream classification training.
"""

import csv
import logging
from pathlib import Path
from typing import Optional

import numpy as np

from src.data.fits_preprocessing import make_display_image

logger = logging.getLogger(__name__)


def get_cluster_representatives(
    embeddings: np.ndarray,
    labels: np.ndarray,
    paths: list,
    n_per_cluster: int = 20,
    method: str = "nearest_centroid",
) -> dict:
    """
    Select representative images for each cluster.

    Args:
        embeddings: (N, D) embedding matrix.
        labels: (N,) cluster assignment array.
        paths: List of file paths.
        n_per_cluster: Number of representatives per cluster.
        method: 'nearest_centroid' (closest to centroid) or 'random'.

    Returns:
        Dict mapping cluster_id → list of (path, distance_to_centroid).
    """
    unique_labels = np.unique(labels)
    representatives = {}

    for cluster_id in unique_labels:
        mask = labels == cluster_id
        cluster_embeddings = embeddings[mask]
        cluster_paths = [p for p, m in zip(paths, mask) if m]

        if method == "nearest_centroid":
            centroid = cluster_embeddings.mean(axis=0)
            distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
            sorted_idx = np.argsort(distances)
            n = min(n_per_cluster, len(sorted_idx))
            reps = [
                (cluster_paths[i], float(distances[i]))
                for i in sorted_idx[:n]
            ]
        else:
            rng = np.random.RandomState(42)
            n = min(n_per_cluster, len(cluster_paths))
            idx = rng.choice(len(cluster_paths), n, replace=False)
            reps = [(cluster_paths[i], 0.0) for i in idx]

        representatives[int(cluster_id)] = reps

    return representatives


def display_cluster_grid(
    cluster_id: int,
    image_paths: list,
    n_cols: int = 5,
    figsize_per_image: float = 2.0,
) -> None:
    """
    Display a grid of images from a cluster using matplotlib.

    Args:
        cluster_id: The cluster ID being displayed.
        image_paths: List of FITS file paths.
        n_cols: Number of columns in the grid.
        figsize_per_image: Size per image in inches.
    """
    import matplotlib.pyplot as plt
    from astropy.io import fits

    n_images = len(image_paths)
    n_rows = (n_images + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * figsize_per_image, n_rows * figsize_per_image),
    )
    if n_rows == 1:
        axes = [axes]
    axes = np.array(axes).flatten()

    fig.suptitle(f"Cluster {cluster_id} — {n_images} samples", fontsize=14)

    for i, ax in enumerate(axes):
        if i < n_images:
            try:
                with fits.open(image_paths[i], memmap=False) as hdul:
                    data = hdul[0].data.astype(np.float32)
                    header = hdul[0].header
                    display = make_display_image(data, header=header, normalization="header")
                    if display.ndim == 2:
                        ax.imshow(display, cmap="gray", vmin=0.0, vmax=1.0, origin="lower")
                    else:
                        ax.imshow(display, origin="lower")
                    ax.set_title(Path(image_paths[i]).stem, fontsize=6)
            except Exception as e:
                ax.text(0.5, 0.5, "Error", ha="center", va="center")
                logger.warning(f"Failed to load {image_paths[i]}: {e}")
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def interactive_label_clusters(
    representatives: dict,
    output_csv: str,
    label_options: Optional[list] = None,
) -> str:
    """
    Interactive CLI labeling tool for cluster-then-verify strategy.

    Displays each cluster's representatives and asks the user to assign
    a label or mark for further review.

    Args:
        representatives: Dict from get_cluster_representatives().
        output_csv: Path to save labeled CSV.
        label_options: Optional list of predefined labels (e.g., ['spiral', 'elliptical', ...]).

    Returns:
        Path to the output CSV.
    """
    if label_options is None:
        label_options = [
            "spiral",
            "barred_spiral",
            "elliptical",
            "irregular",
            "merger",
            "edge_on",
            "artifact",
            "uncertain",
        ]

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)

    labels_assigned = {}
    cluster_ids = sorted(representatives.keys())

    print("\n" + "=" * 60)
    print("CLUSTER LABELING TOOL")
    print("=" * 60)
    print(f"\nTotal clusters: {len(cluster_ids)}")
    print(f"Predefined labels: {label_options}")
    print("Enter a number to select a label, or type a custom label.")
    print("Enter 'skip' to skip, 'quit' to stop early.\n")

    for cluster_id in cluster_ids:
        reps = representatives[cluster_id]
        image_paths = [r[0] for r in reps]

        print(f"\n--- Cluster {cluster_id} ({len(reps)} representatives) ---")

        try:
            display_cluster_grid(cluster_id, image_paths)
        except Exception as e:
            logger.warning(f"Could not display cluster {cluster_id}: {e}")
            print(f"  (Display failed — showing file paths instead)")
            for i, (path, dist) in enumerate(reps[:5]):
                print(f"    [{i}] {Path(path).name} (dist={dist:.4f})")

        # Show label options
        print("\nLabel options:")
        for i, label in enumerate(label_options):
            print(f"  [{i}] {label}")

        user_input = input(f"\nLabel for cluster {cluster_id}: ").strip()

        if user_input.lower() == "quit":
            print("Stopping early.")
            break
        elif user_input.lower() == "skip":
            labels_assigned[cluster_id] = "unlabeled"
            continue
        elif user_input.isdigit() and int(user_input) < len(label_options):
            labels_assigned[cluster_id] = label_options[int(user_input)]
        else:
            labels_assigned[cluster_id] = user_input if user_input else "unlabeled"

        print(f"  → Assigned: {labels_assigned[cluster_id]}")

    # Write CSV
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file_path", "cluster_id", "label"])
        for cluster_id, reps in representatives.items():
            label = labels_assigned.get(cluster_id, "unlabeled")
            for path, dist in reps:
                writer.writerow([path, cluster_id, label])

    print(f"\nLabeled data saved to {output_csv}")
    return output_csv


def propagate_labels(
    labels_csv: str,
    cluster_labels: np.ndarray,
    all_paths: list,
    output_csv: str,
) -> str:
    """
    Propagate cluster-level labels to all images in those clusters.

    Takes verified labels from the annotation tool and assigns them to
    all images sharing the same cluster assignment.

    Args:
        labels_csv: CSV from interactive_label_clusters() with cluster → label mapping.
        cluster_labels: (N,) cluster assignment array for all images.
        all_paths: List of all image file paths.
        output_csv: Output CSV with full propagated labels.

    Returns:
        Path to full labeled CSV.
    """
    # Read cluster → label mapping from CSV
    cluster_to_label = {}
    with open(labels_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            cid = int(row["cluster_id"])
            if cid not in cluster_to_label:
                cluster_to_label[cid] = row["label"]

    # Propagate
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file_path", "cluster_id", "label"])
        for path, cid in zip(all_paths, cluster_labels):
            label = cluster_to_label.get(int(cid), "unlabeled")
            writer.writerow([path, int(cid), label])
            count += 1

    logger.info(f"Propagated labels to {count} images → {output_csv}")
    return output_csv
