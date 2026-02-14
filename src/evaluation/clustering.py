"""
Clustering and visualization of SSL embeddings.

Performs k-means clustering on extracted CLS embeddings and generates
UMAP visualizations for qualitative evaluation of learned representations.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import h5py

logger = logging.getLogger(__name__)


def load_embeddings(h5_path: str) -> dict:
    """Load embeddings and metadata from HDF5 file."""
    with h5py.File(h5_path, "r") as f:
        embeddings = f["embeddings"][:]
        paths = [p.decode("utf-8") if isinstance(p, bytes) else p for p in f["paths"][:]]
        metadata = dict(f.attrs)
    return {"embeddings": embeddings, "paths": paths, "metadata": metadata}


def run_kmeans(
    embeddings: np.ndarray,
    n_clusters: int = 10,
    n_init: int = 10,
    max_iter: int = 300,
    seed: int = 42,
) -> dict:
    """
    Run k-means clustering on embeddings.

    Returns cluster assignments, centroids, inertia, and silhouette score.
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    logger.info(f"Running k-means with k={n_clusters} on {embeddings.shape[0]} embeddings...")

    kmeans = KMeans(
        n_clusters=n_clusters,
        n_init=n_init,
        max_iter=max_iter,
        random_state=seed,
        verbose=0,
    )
    labels = kmeans.fit_predict(embeddings)

    sil_score = silhouette_score(
        embeddings, labels, sample_size=min(10000, len(embeddings)), random_state=seed
    )

    logger.info(f"k-means inertia: {kmeans.inertia_:.2f}, silhouette: {sil_score:.4f}")

    return {
        "labels": labels,
        "centroids": kmeans.cluster_centers_,
        "inertia": kmeans.inertia_,
        "silhouette": sil_score,
        "n_clusters": n_clusters,
    }


def sweep_k(
    embeddings: np.ndarray,
    k_range: tuple = (5, 50),
    step: int = 5,
    seed: int = 42,
) -> dict:
    """
    Sweep over k values to find optimal clustering.

    Returns dict with k values, inertias, silhouette scores, and best k.
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    ks = list(range(k_range[0], k_range[1] + 1, step))
    inertias = []
    silhouettes = []

    logger.info(f"Sweeping k from {k_range[0]} to {k_range[1]} (step={step})...")

    for k in ks:
        kmeans = KMeans(n_clusters=k, n_init=5, random_state=seed)
        labels = kmeans.fit_predict(embeddings)
        inertias.append(kmeans.inertia_)

        sil = silhouette_score(
            embeddings, labels, sample_size=min(10000, len(embeddings)), random_state=seed
        )
        silhouettes.append(sil)
        logger.info(f"  k={k}: inertia={kmeans.inertia_:.2f}, silhouette={sil:.4f}")

    best_idx = int(np.argmax(silhouettes))

    return {
        "ks": ks,
        "inertias": inertias,
        "silhouettes": silhouettes,
        "best_k": ks[best_idx],
        "best_silhouette": silhouettes[best_idx],
    }


def compute_umap(
    embeddings: np.ndarray,
    n_components: int = 2,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "cosine",
    seed: int = 42,
    max_samples: int = 50000,
) -> np.ndarray:
    """
    Compute UMAP projection of embeddings.

    Subsamples to max_samples for efficiency if needed.
    Returns (N, n_components) array of projected coordinates.
    """
    import umap

    if len(embeddings) > max_samples:
        logger.info(f"Subsampling from {len(embeddings)} to {max_samples} for UMAP")
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(embeddings), max_samples, replace=False)
        embeddings = embeddings[idx]

    logger.info(f"Computing UMAP ({n_components}D) for {len(embeddings)} embeddings...")

    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=seed,
    )
    projection = reducer.fit_transform(embeddings)

    return projection


def plot_umap_clusters(
    projection: np.ndarray,
    labels: np.ndarray,
    output_path: str,
    title: str = "UMAP — SSL Embedding Clusters",
    figsize: tuple = (12, 10),
    point_size: float = 0.5,
    alpha: float = 0.4,
):
    """
    Plot UMAP 2D projection colored by cluster assignment.

    Saves figure to output_path.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)

    n_clusters = len(np.unique(labels))
    cmap = plt.cm.get_cmap("tab20", n_clusters)

    scatter = ax.scatter(
        projection[:, 0], projection[:, 1],
        c=labels[:len(projection)],
        cmap=cmap,
        s=point_size,
        alpha=alpha,
        rasterized=True,
    )

    plt.colorbar(scatter, ax=ax, label="Cluster ID")
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"UMAP plot saved to {output_path}")


def plot_elbow(
    ks: list,
    inertias: list,
    silhouettes: list,
    output_path: str,
):
    """Plot elbow diagram (inertia + silhouette) for k sweep."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(ks, inertias, "b-o", markersize=4)
    ax1.set_xlabel("Number of clusters (k)")
    ax1.set_ylabel("Inertia")
    ax1.set_title("Elbow Plot")
    ax1.grid(True, alpha=0.3)

    ax2.plot(ks, silhouettes, "r-o", markersize=4)
    ax2.set_xlabel("Number of clusters (k)")
    ax2.set_ylabel("Silhouette Score")
    ax2.set_title("Silhouette Score vs k")
    ax2.grid(True, alpha=0.3)

    best_idx = int(np.argmax(silhouettes))
    ax2.axvline(ks[best_idx], color="green", linestyle="--", alpha=0.7,
                label=f"Best k={ks[best_idx]}")
    ax2.legend()

    fig.tight_layout()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Elbow plot saved to {output_path}")


def save_cluster_results(
    output_path: str,
    paths: list,
    labels: np.ndarray,
    embeddings: np.ndarray,
    projection: Optional[np.ndarray] = None,
):
    """Save cluster assignments, embeddings, and projections to HDF5."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, "w") as f:
        f.create_dataset("labels", data=labels)
        f.create_dataset("embeddings", data=embeddings, compression="gzip")
        f.create_dataset(
            "paths",
            data=np.array([p.encode("utf-8") for p in paths], dtype=h5py.special_dtype(vlen=bytes)),
        )
        if projection is not None:
            f.create_dataset("umap_projection", data=projection)

    logger.info(f"Cluster results saved to {output_path}")
