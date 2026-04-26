#!/usr/bin/env python3
"""
JAX/Flax data pipeline for the FITS galaxy dataset.

Uses a Python generator-based approach for lazy loading, with optional
sharding across JAX devices. Compatible with jax.pmap for data parallelism.

Key design:
- Generator yields NumPy arrays (converted to jax.numpy at training time)
- Batching and shuffling done manually for full control
- Device sharding splits batches across local accelerators
"""

from typing import Optional, Callable, Tuple, Iterator, List

import numpy as np

from .fits_dataset import FITSDataset
from .augmentations import AstronomyAugmentations, DINOMultiCropAugmentation


class JAXDataIterator:
    """
    Iterator that yields batches of galaxy images as NumPy arrays,
    ready to be converted to JAX arrays for training.

    Handles shuffling, batching, and optional device sharding.

    Args:
        fits_dataset: FITSDataset instance.
        batch_size: Total batch size (will be split across devices if sharded).
        augmentation: Optional augmentation callable.
        shuffle: Whether to shuffle indices each epoch.
        seed: Random seed.
        num_devices: Number of JAX devices for sharding (batch dim split).
        drop_last: Drop incomplete final batch.
    """

    def __init__(
        self,
        fits_dataset: FITSDataset,
        batch_size: int = 256,
        augmentation: Optional[Callable] = None,
        shuffle: bool = True,
        seed: int = 42,
        num_devices: int = 1,
        drop_last: bool = True,
    ):
        self.dataset = fits_dataset
        self.batch_size = batch_size
        self.augmentation = augmentation
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)
        self.num_devices = num_devices
        self.drop_last = drop_last

        # Ensure batch size is divisible by device count for pmap
        assert batch_size % num_devices == 0, (
            f"batch_size ({batch_size}) must be divisible by num_devices ({num_devices})"
        )
        self.per_device_batch = batch_size // num_devices

    def __len__(self) -> int:
        n = len(self.dataset) // self.batch_size
        if not self.drop_last and len(self.dataset) % self.batch_size > 0:
            n += 1
        return n

    def _load_single(self, idx: int) -> np.ndarray:
        """Load a single image with retries on failure."""
        for _ in range(5):
            image = self.dataset[idx]
            if image is not None:
                if self.augmentation is not None:
                    image = self.augmentation(image, self.rng)
                return image
            idx = self.rng.integers(0, len(self.dataset))

        return self.dataset.empty_item()

    def __iter__(self) -> Iterator[np.ndarray]:
        """
        Yield batches of shape:
        - Without sharding: (batch_size, 1, H, W)
        - With sharding: (num_devices, per_device_batch, 1, H, W)
        """
        indices = np.arange(len(self.dataset))
        if self.shuffle:
            self.rng.shuffle(indices)

        batch = []
        for idx in indices:
            image = self._load_single(idx)
            batch.append(image)

            if len(batch) == self.batch_size:
                batch_array = np.stack(batch, axis=0)

                if self.num_devices > 1:
                    # Reshape for pmap: (devices, per_device_batch, C, H, W)
                    batch_array = batch_array.reshape(
                        self.num_devices, self.per_device_batch, *batch_array.shape[1:]
                    )

                yield batch_array
                batch = []

        # Handle last incomplete batch
        if batch and not self.drop_last:
            # Pad to full batch size
            while len(batch) < self.batch_size:
                batch.append(batch[-1])
            batch_array = np.stack(batch, axis=0)

            if self.num_devices > 1:
                batch_array = batch_array.reshape(
                    self.num_devices, self.per_device_batch, *batch_array.shape[1:]
                )

            yield batch_array

    def epoch_iterator(self) -> Iterator[np.ndarray]:
        """Alias for __iter__ — generates one full epoch of batches."""
        return iter(self)


class JAXDINODataIterator:
    """
    Iterator that yields DINO multi-crop batches as NumPy arrays for JAX.

    Returns:
        Tuple of (global_crops, local_crops):
        - global_crops: list of 2 arrays, each (B, 1, gs, gs)
        - local_crops: list of N arrays, each (B, 1, ls, ls)

    If sharded, shapes become (D, B//D, 1, size, size).
    """

    def __init__(
        self,
        fits_dataset: FITSDataset,
        multi_crop_aug: DINOMultiCropAugmentation,
        batch_size: int = 64,
        shuffle: bool = True,
        seed: int = 42,
        num_devices: int = 1,
        drop_last: bool = True,
    ):
        self.dataset = fits_dataset
        self.multi_crop = multi_crop_aug
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)
        self.num_devices = num_devices
        self.drop_last = drop_last

        assert batch_size % num_devices == 0

    def __len__(self) -> int:
        n = len(self.dataset) // self.batch_size
        if not self.drop_last and len(self.dataset) % self.batch_size > 0:
            n += 1
        return n

    def _load_and_crop(self, idx: int):
        """Load image and generate multi-crop views."""
        for _ in range(5):
            image = self.dataset[idx]
            if image is not None:
                return self.multi_crop(image, self.rng)
            idx = self.rng.integers(0, len(self.dataset))

        # Fallback
        gs = self.multi_crop.global_crop_size
        ls = self.multi_crop.local_crop_size
        c = self.dataset.default_channels
        return (
            [np.zeros((c, gs, gs), dtype=np.float32) for _ in range(2)],
            [np.zeros((c, ls, ls), dtype=np.float32) for _ in range(self.multi_crop.n_local_crops)],
        )

    def _shard(self, arr: np.ndarray) -> np.ndarray:
        """Reshape for pmap if using multiple devices."""
        if self.num_devices > 1:
            per_device = arr.shape[0] // self.num_devices
            return arr.reshape(self.num_devices, per_device, *arr.shape[1:])
        return arr

    def __iter__(self):
        indices = np.arange(len(self.dataset))
        if self.shuffle:
            self.rng.shuffle(indices)

        global_batch = [[] for _ in range(2)]
        n_local = self.multi_crop.n_local_crops
        local_batch = [[] for _ in range(n_local)]
        count = 0

        for idx in indices:
            global_crops, local_crops = self._load_and_crop(idx)
            for i in range(2):
                global_batch[i].append(global_crops[i])
            for i in range(n_local):
                local_batch[i].append(local_crops[i])
            count += 1

            if count == self.batch_size:
                g = [self._shard(np.stack(crops)) for crops in global_batch]
                l = [self._shard(np.stack(crops)) for crops in local_batch]
                yield g, l

                global_batch = [[] for _ in range(2)]
                local_batch = [[] for _ in range(n_local)]
                count = 0

        if count > 0 and not self.drop_last:
            while count < self.batch_size:
                for i in range(2):
                    global_batch[i].append(global_batch[i][-1])
                for i in range(n_local):
                    local_batch[i].append(local_batch[i][-1])
                count += 1
            g = [self._shard(np.stack(crops)) for crops in global_batch]
            l = [self._shard(np.stack(crops)) for crops in local_batch]
            yield g, l
