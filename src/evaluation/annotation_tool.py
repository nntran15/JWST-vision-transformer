"""Matplotlib-based annotation tool for cluster-then-verify labeling."""

import csv
import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np

from src.data.fits_preprocessing import make_display_image

logger = logging.getLogger(__name__)


def has_interactive_display() -> bool:
    """Return True when matplotlib can open interactive windows."""
    import matplotlib

    backend = matplotlib.get_backend().lower()
    if backend in {"agg", "pdf", "ps", "svg", "template", "cairo"}:
        return False

    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def get_cluster_representatives(
    embeddings: np.ndarray,
    labels: np.ndarray,
    paths: list,
    n_per_cluster: int = 20,
    method: str = "nearest_centroid",
) -> dict:
    """Select representative images for each cluster."""
    unique_labels = np.unique(labels)
    representatives = {}

    for cluster_id in unique_labels:
        mask = labels == cluster_id
        cluster_embeddings = embeddings[mask]
        cluster_paths = [path for path, keep in zip(paths, mask) if keep]

        if method == "nearest_centroid":
            centroid = cluster_embeddings.mean(axis=0)
            distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
            sorted_idx = np.argsort(distances)
            count = min(n_per_cluster, len(sorted_idx))
            reps = [
                (cluster_paths[idx], float(distances[idx]))
                for idx in sorted_idx[:count]
            ]
        else:
            rng = np.random.RandomState(42)
            count = min(n_per_cluster, len(cluster_paths))
            sampled_idx = rng.choice(len(cluster_paths), count, replace=False)
            reps = [(cluster_paths[idx], 0.0) for idx in sampled_idx]

        representatives[int(cluster_id)] = reps

    return representatives


def display_cluster_grid(
    cluster_id: int,
    image_paths: list,
    n_cols: int = 5,
    figsize_per_image: float = 2.0,
    output_path: Optional[str] = None,
    show: bool = True,
) -> Optional[Path]:
    """Display a grid of images from a cluster using matplotlib."""
    import matplotlib.pyplot as plt
    from astropy.io import fits

    n_images = len(image_paths)
    n_rows = (n_images + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(n_cols * figsize_per_image, n_rows * figsize_per_image),
    )
    if n_rows == 1:
        axes = [axes]
    axes = np.array(axes).flatten()

    fig.suptitle(f"Cluster {cluster_id} - {n_images} samples", fontsize=14)

    for idx, ax in enumerate(axes):
        if idx < n_images:
            try:
                with fits.open(image_paths[idx], memmap=False) as hdul:
                    data = hdul[0].data.astype(np.float32)
                    header = hdul[0].header
                    display = make_display_image(data, header=header, normalization="header")
                    if display.ndim == 2:
                        ax.imshow(display, cmap="gray", vmin=0.0, vmax=1.0, origin="lower")
                    else:
                        ax.imshow(display, origin="lower")
                    ax.set_title(Path(image_paths[idx]).stem, fontsize=6)
            except Exception as exc:
                ax.text(0.5, 0.5, "Error", ha="center", va="center")
                logger.warning(f"Failed to load {image_paths[idx]}: {exc}")
        ax.axis("off")

    plt.tight_layout()
    saved_path = None
    if output_path is not None:
        saved_path = Path(output_path)
        saved_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(saved_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()
    plt.close(fig)

    return saved_path


def interactive_label_clusters(
    representatives: dict,
    output_csv: str,
    label_options: Optional[list] = None,
    preview_dir: Optional[str] = None,
    show_images: Optional[bool] = None,
) -> str:
    """Interactively assign one of the allowed labels to each cluster."""
    if label_options is None:
        label_options = [
            "spiral",
            "not_spiral",
            "artifact",
            "uncertain",
        ]

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    if preview_dir is not None:
        Path(preview_dir).mkdir(parents=True, exist_ok=True)

    if show_images is None:
        show_images = has_interactive_display()

    labels_assigned = {}
    cluster_ids = sorted(representatives.keys())

    print("\n" + "=" * 60)
    print("CLUSTER LABELING TOOL")
    print("=" * 60)
    print(f"\nTotal clusters: {len(cluster_ids)}")
    print(f"Predefined labels: {label_options}")
    print("Enter a number to select a label.")
    print("Enter 'skip' to skip, 'quit' to stop early.\n")
    if not show_images:
        print("Interactive plotting is unavailable in this session; representative grids will be saved to disk.\n")

    for cluster_id in cluster_ids:
        reps = representatives[cluster_id]
        image_paths = [path for path, _ in reps]

        print(f"\n--- Cluster {cluster_id} ({len(reps)} representatives) ---")

        try:
            preview_path = None
            if preview_dir is not None:
                preview_path = str(Path(preview_dir) / f"cluster_{cluster_id:03d}.png")

            saved_path = display_cluster_grid(
                cluster_id,
                image_paths,
                output_path=preview_path,
                show=show_images,
            )
            if saved_path is not None:
                print(f"Saved representative grid: {saved_path}")
        except Exception as exc:
            logger.warning(f"Could not display cluster {cluster_id}: {exc}")
            print("  (Display failed - showing file paths instead)")
            for idx, (path, dist) in enumerate(reps[:5]):
                print(f"    [{idx}] {Path(path).name} (dist={dist:.4f})")

        print("\nLabel options:")
        for idx, label in enumerate(label_options):
            print(f"  [{idx}] {label}")

        while True:
            user_input = input(f"\nLabel for cluster {cluster_id}: ").strip().lower()

            if user_input == "quit":
                print("Stopping early.")
                labels_assigned[cluster_id] = None
                break
            if user_input == "skip":
                labels_assigned[cluster_id] = "unlabeled"
                break
            if user_input.isdigit() and int(user_input) < len(label_options):
                labels_assigned[cluster_id] = label_options[int(user_input)]
                break
            if user_input in label_options:
                labels_assigned[cluster_id] = user_input
                break

            print("Invalid input. Choose a label number, type an allowed label, or use 'skip'/'quit'.")

        if labels_assigned[cluster_id] is None:
            break

        print(f"  -> Assigned: {labels_assigned[cluster_id]}")

    with open(output_csv, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["file_path", "cluster_id", "label"])
        for cluster_id, reps in representatives.items():
            label = labels_assigned.get(cluster_id, "unlabeled")
            for path, _ in reps:
                writer.writerow([path, cluster_id, label])

    print(f"\nLabeled data saved to {output_csv}")
    return output_csv


def propagate_labels(
    labels_csv: str,
    cluster_labels: np.ndarray,
    all_paths: list,
    output_csv: str,
) -> str:
    """Propagate cluster labels to all images assigned to each cluster."""
    cluster_to_label = {}
    with open(labels_csv) as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            cluster_id = int(row["cluster_id"])
            if cluster_id not in cluster_to_label:
                cluster_to_label[cluster_id] = row["label"]

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with open(output_csv, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["file_path", "cluster_id", "label"])
        for path, cluster_id in zip(all_paths, cluster_labels):
            label = cluster_to_label.get(int(cluster_id), "unlabeled")
            writer.writerow([path, int(cluster_id), label])
            count += 1

    logger.info(f"Propagated labels to {count} images -> {output_csv}")
    return output_csv
