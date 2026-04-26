#!/usr/bin/env python3
"""
PyTorch Dataset and DataLoader wrappers for the FITS galaxy dataset.

Wraps the framework-agnostic FITSDataset into PyTorch-compatible classes
with support for:
- DistributedSampler for multi-GPU training
- Lazy loading (no full catalog in RAM)
- Configurable augmentations
- Collation with skip-on-error handling
"""

from typing import Optional, Callable, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler

from .fits_dataset import FITSDataset
from .augmentations import AstronomyAugmentations, DINOMultiCropAugmentation


class PyTorchFITSDataset(Dataset):
    """
    PyTorch Dataset wrapping the framework-agnostic FITSDataset.

    Returns torch.Tensor images in CHW format. Optionally applies
    augmentations. Skips failed loads and returns a random replacement.

    Args:
        fits_dataset: FITSDataset instance.
        augmentation: Augmentation callable (operates on NumPy arrays).
        seed: Random seed for augmentation reproducibility.
    """

    def __init__(
        self,
        fits_dataset: FITSDataset,
        augmentation: Optional[Callable] = None,
        seed: int = 42,
    ):
        self.dataset = fits_dataset
        self.augmentation = augmentation
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Load image at idx, apply augmentation, return as torch.Tensor.

        If loading fails, tries a random fallback index (up to 5 retries).
        """
        for attempt in range(5):
            try:
                image = self.dataset[idx]
                if image is None:
                    idx = self.rng.integers(0, len(self.dataset))
                    continue

                if self.augmentation is not None:
                    image = self.augmentation(image, self.rng)

                return torch.from_numpy(image)

            except Exception:
                idx = self.rng.integers(0, len(self.dataset))
                continue

        # Final fallback: return zeros
        return torch.from_numpy(self.dataset.empty_item())


class DINOMultiCropDataset(Dataset):
    """
    PyTorch Dataset that returns multi-crop views for DINO training.

    Each __getitem__ returns a tuple of (global_crops, local_crops) as lists
    of torch.Tensors.

    Args:
        fits_dataset: FITSDataset instance.
        multi_crop_aug: DINOMultiCropAugmentation instance.
        seed: Random seed.
    """

    def __init__(
        self,
        fits_dataset: FITSDataset,
        multi_crop_aug: DINOMultiCropAugmentation,
        seed: int = 42,
    ):
        self.dataset = fits_dataset
        self.multi_crop = multi_crop_aug
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        for attempt in range(5):
            try:
                image = self.dataset[idx]
                if image is None:
                    idx = self.rng.integers(0, len(self.dataset))
                    continue

                global_crops, local_crops = self.multi_crop(image, self.rng)

                global_tensors = [torch.from_numpy(c) for c in global_crops]
                local_tensors = [torch.from_numpy(c) for c in local_crops]

                return global_tensors, local_tensors

            except Exception:
                idx = self.rng.integers(0, len(self.dataset))
                continue

        # Fallback
        gs = self.multi_crop.global_crop_size
        ls = self.multi_crop.local_crop_size
        c = self.dataset.default_channels
        return (
            [torch.zeros(c, gs, gs) for _ in range(2)],
            [torch.zeros(c, ls, ls) for _ in range(self.multi_crop.n_local_crops)],
        )


def dino_collate_fn(batch):
    """
    Custom collate for DINO multi-crop dataset.

    Stacks each crop position across the batch:
    - global_crops: list of 2 tensors, each (B, 1, gs, gs)
    - local_crops: list of N tensors, each (B, 1, ls, ls)
    """
    global_crops_list = [[] for _ in range(2)]
    local_crops_list = [[] for _ in range(len(batch[0][1]))]

    for global_crops, local_crops in batch:
        for i, gc in enumerate(global_crops):
            global_crops_list[i].append(gc)
        for i, lc in enumerate(local_crops):
            local_crops_list[i].append(lc)

    global_crops_batch = [torch.stack(crops) for crops in global_crops_list]
    local_crops_batch = [torch.stack(crops) for crops in local_crops_list]

    return global_crops_batch, local_crops_batch


def create_dataloader(
    fits_dataset: FITSDataset,
    batch_size: int = 256,
    num_workers: int = 8,
    augmentation: Optional[Callable] = None,
    distributed: bool = False,
    world_size: int = 1,
    rank: int = 0,
    seed: int = 42,
    pin_memory: bool = True,
    prefetch_factor: int = 2,
) -> DataLoader:
    """
    Create a PyTorch DataLoader for standard SSL training (MAE).

    Args:
        fits_dataset: FITSDataset instance.
        batch_size: Per-GPU batch size.
        num_workers: Number of data loading worker processes.
        augmentation: Optional augmentation callable.
        distributed: Whether to use DistributedSampler.
        world_size: Total number of GPU processes.
        rank: Current GPU rank.
        seed: Random seed.
        pin_memory: Pin memory for faster GPU transfer.
        prefetch_factor: Number of batches to prefetch per worker.

    Returns:
        PyTorch DataLoader.
    """
    dataset = PyTorchFITSDataset(fits_dataset, augmentation=augmentation, seed=seed)

    sampler = None
    shuffle = True

    if distributed:
        sampler = DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=seed
        )
        shuffle = False

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        drop_last=True,
        persistent_workers=num_workers > 0,
    )


def create_dino_dataloader(
    fits_dataset: FITSDataset,
    multi_crop_aug: DINOMultiCropAugmentation,
    batch_size: int = 64,
    num_workers: int = 8,
    distributed: bool = False,
    world_size: int = 1,
    rank: int = 0,
    seed: int = 42,
    pin_memory: bool = True,
) -> DataLoader:
    """
    Create a PyTorch DataLoader for DINO multi-crop training.

    Args:
        fits_dataset: FITSDataset instance.
        multi_crop_aug: DINOMultiCropAugmentation instance.
        batch_size: Per-GPU batch size (smaller due to multi-crop memory).
        num_workers: Number of data loading workers.
        distributed: Use DistributedSampler.
        world_size: Total GPU count.
        rank: Current GPU rank.
        seed: Random seed.
        pin_memory: Pin memory.

    Returns:
        PyTorch DataLoader with custom collation.
    """
    dataset = DINOMultiCropDataset(fits_dataset, multi_crop_aug, seed=seed)

    sampler = None
    shuffle = True

    if distributed:
        sampler = DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=seed
        )
        shuffle = False

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        collate_fn=dino_collate_fn,
        persistent_workers=num_workers > 0,
    )
