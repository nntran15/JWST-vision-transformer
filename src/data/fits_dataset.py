#!/usr/bin/env python3
"""
FITS Dataset: Framework-agnostic data loading for JWST galaxy thumbnails.

Handles scanning, indexing, loading, filtering, and preprocessing of FITS files
from the JWST galaxy thumbnail catalog (3.5M+ files, ~1.5 TB).

Key design decisions:
- File index is cached to disk to avoid re-scanning millions of files
- Images are loaded lazily (one at a time) to avoid OOM
- Returns NumPy arrays for framework-agnostic compatibility
- Supports configurable dimension filtering and normalization
"""

import json
import logging
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
from astropy.io import fits
from tqdm import tqdm

logger = logging.getLogger(__name__)


def build_file_index(
    catalog_dir: str,
    index_path: str = "file_index.json",
    min_dim: int = 10,
    max_dim: int = 200,
    extensions: Tuple[str, ...] = (".fits", ".fits.gz", ".fit"),
) -> List[Dict[str, Any]]:
    """
    Recursively scan a catalog directory for FITS files, extract dimensions,
    filter outliers, and cache the index to disk.

    Args:
        catalog_dir: Root directory containing FITS files.
        index_path: Path to save/load the cached file index.
        min_dim: Minimum pixel dimension (both axes) to include.
        max_dim: Maximum pixel dimension (both axes) to include.
        extensions: File extensions to consider as FITS files.

    Returns:
        List of dicts with keys: 'path', 'x_dim', 'y_dim'.
    """
    index_file = Path(index_path)

    # Return cached index if it exists
    if index_file.exists():
        logger.info(f"Loading cached file index from {index_path}")
        with open(index_file, "r") as f:
            index = json.load(f)
        logger.info(f"Loaded {len(index)} entries from cached index")
        return index

    logger.info(f"Scanning catalog directory: {catalog_dir}")
    catalog_path = Path(catalog_dir)

    if not catalog_path.exists():
        raise FileNotFoundError(f"Catalog directory not found: {catalog_dir}")

    # Collect all FITS file paths
    all_fits_files = []
    for ext in extensions:
        all_fits_files.extend(catalog_path.rglob(f"*{ext}"))

    logger.info(f"Found {len(all_fits_files)} FITS files, extracting dimensions...")

    index = []
    skipped_read = 0
    skipped_dim = 0

    for fits_path in tqdm(all_fits_files, desc="Indexing FITS files"):
        try:
            with fits.open(str(fits_path), memmap=True) as hdul:
                for hdu in hdul:
                    if hdu.data is not None and len(hdu.data.shape) >= 2:
                        y_dim, x_dim = hdu.data.shape[-2], hdu.data.shape[-1]

                        # Filter outliers
                        if (min_dim <= x_dim <= max_dim) and (min_dim <= y_dim <= max_dim):
                            index.append({
                                "path": str(fits_path),
                                "x_dim": int(x_dim),
                                "y_dim": int(y_dim),
                            })
                        else:
                            skipped_dim += 1
                        break  # Only use first HDU with 2D data
        except Exception:
            skipped_read += 1
            continue

    logger.info(
        f"Indexed {len(index)} files | "
        f"Skipped {skipped_dim} (dimension filter) | "
        f"Skipped {skipped_read} (read error)"
    )

    # Cache to disk
    index_file.parent.mkdir(parents=True, exist_ok=True)
    with open(index_file, "w") as f:
        json.dump(index, f)
    logger.info(f"Saved index to {index_path}")

    return index


class FITSDataset:
    """
    Framework-agnostic FITS image dataset.

    Loads FITS galaxy thumbnails lazily, applies normalization and resizing,
    returns NumPy arrays. Framework-specific wrappers (PyTorch Dataset, JAX
    generator) should wrap this class.

    Args:
        index: List of dicts with 'path', 'x_dim', 'y_dim' keys (from build_file_index).
        target_size: Resize all images to (target_size, target_size).
        normalization: Normalization strategy ('minmax', 'percentile', 'zscore').
        percentile_clip: Percentile range for 'percentile' normalization (low, high).
    """

    def __init__(
        self,
        index: List[Dict[str, Any]],
        target_size: int = 64,
        normalization: str = "percentile",
        percentile_clip: Tuple[float, float] = (1.0, 99.0),
    ):
        self.index = index
        self.target_size = target_size
        self.normalization = normalization
        self.percentile_clip = percentile_clip

    def __len__(self) -> int:
        return len(self.index)

    def load_fits(self, path: str) -> Optional[np.ndarray]:
        """
        Load a single FITS file and return the 2D image data.

        Args:
            path: Path to the FITS file.

        Returns:
            2D NumPy array (float32) or None if loading fails.
        """
        try:
            with fits.open(path, memmap=True) as hdul:
                for hdu in hdul:
                    if hdu.data is not None and len(hdu.data.shape) >= 2:
                        data = hdu.data.astype(np.float32)
                        # Take last two dimensions if >2D (e.g., data cubes)
                        if data.ndim > 2:
                            data = data[0]
                        return data
            return None
        except Exception as e:
            logger.debug(f"Failed to load {path}: {e}")
            return None

    def normalize(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize pixel values based on the configured strategy.

        Handles NaN/Inf values and cosmic ray artifacts common in astronomical data.

        Args:
            image: 2D float32 array.

        Returns:
            Normalized 2D float32 array with values in [0, 1].
        """
        # Replace NaN/Inf with 0
        image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)

        if self.normalization == "minmax":
            vmin, vmax = image.min(), image.max()
            if vmax - vmin > 1e-10:
                image = (image - vmin) / (vmax - vmin)
            else:
                image = np.zeros_like(image)

        elif self.normalization == "percentile":
            # Percentile clipping handles cosmic rays and hot pixels
            low, high = self.percentile_clip
            vmin = np.percentile(image, low)
            vmax = np.percentile(image, high)
            if vmax - vmin > 1e-10:
                image = np.clip(image, vmin, vmax)
                image = (image - vmin) / (vmax - vmin)
            else:
                image = np.zeros_like(image)

        elif self.normalization == "zscore":
            mean, std = image.mean(), image.std()
            if std > 1e-10:
                image = (image - mean) / std
                # Clip to [-3, 3] sigmas and rescale to [0, 1]
                image = np.clip(image, -3.0, 3.0)
                image = (image + 3.0) / 6.0
            else:
                image = np.zeros_like(image)

        return image.astype(np.float32)

    def resize(self, image: np.ndarray, size: int) -> np.ndarray:
        """
        Resize image to (size, size) using bilinear interpolation.

        Uses NumPy-only implementation to stay framework-agnostic.

        Args:
            image: 2D float32 array of any shape.
            size: Target dimension (both width and height).

        Returns:
            Resized 2D float32 array of shape (size, size).
        """
        h, w = image.shape
        if h == size and w == size:
            return image

        # Bilinear interpolation via coordinate mapping
        y_coords = np.linspace(0, h - 1, size)
        x_coords = np.linspace(0, w - 1, size)
        xv, yv = np.meshgrid(x_coords, y_coords)

        x0 = np.floor(xv).astype(int)
        y0 = np.floor(yv).astype(int)
        x1 = np.minimum(x0 + 1, w - 1)
        y1 = np.minimum(y0 + 1, h - 1)

        dx = xv - x0
        dy = yv - y0

        resized = (
            image[y0, x0] * (1 - dx) * (1 - dy)
            + image[y0, x1] * dx * (1 - dy)
            + image[y1, x0] * (1 - dx) * dy
            + image[y1, x1] * dx * dy
        )

        return resized.astype(np.float32)

    def __getitem__(self, idx: int) -> Optional[np.ndarray]:
        """
        Load, normalize, and resize a single image.

        Args:
            idx: Index into the file index.

        Returns:
            NumPy array of shape (1, target_size, target_size) — CHW format,
            single channel. Returns None if loading fails.
        """
        entry = self.index[idx]
        image = self.load_fits(entry["path"])

        if image is None:
            return None

        image = self.normalize(image)
        image = self.resize(image, self.target_size)

        # Add channel dimension: (H, W) -> (1, H, W) for CHW format
        image = image[np.newaxis, :, :]

        return image

    def get_metadata(self, idx: int) -> Dict[str, Any]:
        """Return metadata for a given index."""
        return self.index[idx]
