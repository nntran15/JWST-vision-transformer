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

from .fits_preprocessing import collapse_to_single_channel, normalize_fits_data, resize_chw

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
        normalization: Normalization strategy ('header', 'arcsinh_rms',
            'minmax', 'percentile', 'zscore').
        percentile_clip: Percentile range for 'percentile' normalization (low, high).
    """

    def __init__(
        self,
        index: List[Dict[str, Any]],
        target_size: int = 64,
        normalization: str = "header",
        percentile_clip: Tuple[float, float] = (1.0, 99.0),
    ):
        self.index = index
        self.target_size = target_size
        self.normalization = normalization
        self.percentile_clip = percentile_clip
        self.default_channels = 1

    def __len__(self) -> int:
        return len(self.index)

    def load_fits(self, path: str) -> Tuple[Optional[np.ndarray], Optional[fits.Header]]:
        """
        Load a single FITS file and return image data plus header.

        Args:
            path: Path to the FITS file.

        Returns:
            Tuple of (image array, FITS header) or (None, None) if loading fails.
        """
        try:
            with fits.open(path, memmap=True) as hdul:
                for hdu in hdul:
                    if hdu.data is not None and len(hdu.data.shape) >= 2:
                        return hdu.data.astype(np.float32), hdu.header
            return None, None
        except Exception as e:
            logger.debug(f"Failed to load {path}: {e}")
            return None, None

    def normalize(
        self,
        image: np.ndarray,
        header: Optional[fits.Header] = None,
    ) -> np.ndarray:
        """
        Normalize pixel values based on the configured strategy.

        Handles NaN/Inf values and cosmic ray artifacts common in astronomical data.

        Args:
            image: 2D or 3D float32 array.

        Returns:
            Normalized CHW float32 array with values in [0, 1].
        """
        normalized = normalize_fits_data(
            image,
            header=header,
            normalization=self.normalization,
            percentile_clip=self.percentile_clip,
        )
        normalized = collapse_to_single_channel(normalized)
        self.default_channels = normalized.shape[0]
        return normalized.astype(np.float32)

    def resize(self, image: np.ndarray, size: int) -> np.ndarray:
        """
        Resize image to (size, size) using bilinear interpolation.

        Uses NumPy-only implementation to stay framework-agnostic.

        Args:
            image: CHW float32 array of any shape.
            size: Target dimension (both width and height).

        Returns:
            Resized CHW float32 array of shape (C, size, size).
        """
        resized = resize_chw(image, size)
        self.default_channels = resized.shape[0]
        return resized.astype(np.float32)

    def empty_item(self) -> np.ndarray:
        """Return an all-zero CHW tensor matching the current channel count."""
        return np.zeros(
            (self.default_channels, self.target_size, self.target_size),
            dtype=np.float32,
        )

    def __getitem__(self, idx: int) -> Optional[np.ndarray]:
        """
        Load, normalize, and resize a single image.

        Args:
            idx: Index into the file index.

        Returns:
            NumPy array of shape (C, target_size, target_size) in CHW format.
            Returns None if loading fails.
        """
        entry = self.index[idx]
        image, header = self.load_fits(entry["path"])

        if image is None:
            return None

        image = self.normalize(image, header=header)
        image = self.resize(image, self.target_size)

        return image

    def get_metadata(self, idx: int) -> Dict[str, Any]:
        """Return metadata for a given index."""
        return self.index[idx]
