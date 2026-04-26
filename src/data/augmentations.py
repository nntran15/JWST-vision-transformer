#!/usr/bin/env python3
"""
Astronomy-aware data augmentations for galaxy thumbnail images.

Designed for FITS images in CHW format. Key considerations:
- Galaxies have no preferred orientation → free rotation/flip
- No color jitter (single-channel data)
- Noise injection simulates observational conditions
- Multi-crop strategy for DINO (global + local views)
"""

from typing import Tuple, List, Optional
import numpy as np


class AstronomyAugmentations:
    """
    Augmentation pipeline for galaxy thumbnails.

    Applied during SSL pretraining. All operations work on NumPy arrays
    in CHW format (C, H, W) with values in [0, 1].

    Args:
        rotation: Enable random 90-degree rotations.
        flip_h: Enable random horizontal flip.
        flip_v: Enable random vertical flip.
        gaussian_noise_std: Std of additive Gaussian noise (0 to disable).
        brightness_range: (min, max) multiplicative brightness factor.
        contrast_range: (min, max) multiplicative contrast factor.
        random_erasing_prob: Probability of applying random erasing.
        random_erasing_scale: (min, max) fraction of image area to erase.
    """

    def __init__(
        self,
        rotation: bool = True,
        flip_h: bool = True,
        flip_v: bool = True,
        gaussian_noise_std: float = 0.02,
        brightness_range: Tuple[float, float] = (0.9, 1.1),
        contrast_range: Tuple[float, float] = (0.9, 1.1),
        random_erasing_prob: float = 0.0,
        random_erasing_scale: Tuple[float, float] = (0.02, 0.33),
    ):
        self.rotation = rotation
        self.flip_h = flip_h
        self.flip_v = flip_v
        self.gaussian_noise_std = gaussian_noise_std
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.random_erasing_prob = random_erasing_prob
        self.random_erasing_scale = random_erasing_scale

    def __call__(self, image: np.ndarray, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """
        Apply augmentations to a single image.

        Args:
            image: NumPy array of shape (C, H, W), values in [0, 1].
            rng: NumPy random generator for reproducibility.

        Returns:
            Augmented image of same shape and dtype.
        """
        if rng is None:
            rng = np.random.default_rng()

        img = image.copy()

        # Random 90-degree rotation (0, 90, 180, 270)
        if self.rotation:
            k = rng.integers(0, 4)
            img = np.rot90(img, k=k, axes=(1, 2)).copy()

        # Random horizontal flip
        if self.flip_h and rng.random() < 0.5:
            img = np.flip(img, axis=2).copy()

        # Random vertical flip
        if self.flip_v and rng.random() < 0.5:
            img = np.flip(img, axis=1).copy()

        # Random brightness adjustment
        if self.brightness_range != (1.0, 1.0):
            factor = rng.uniform(*self.brightness_range)
            img = img * factor

        # Random contrast adjustment (relative to mean)
        if self.contrast_range != (1.0, 1.0):
            factor = rng.uniform(*self.contrast_range)
            mean = img.mean()
            img = (img - mean) * factor + mean

        # Additive Gaussian noise
        if self.gaussian_noise_std > 0:
            noise = rng.normal(0, self.gaussian_noise_std, img.shape).astype(img.dtype)
            img = img + noise

        # Random erasing / cutout
        if self.random_erasing_prob > 0 and rng.random() < self.random_erasing_prob:
            img = self._random_erasing(img, rng)

        # Clamp to [0, 1]
        img = np.clip(img, 0.0, 1.0)

        return img.astype(np.float32)

    def _random_erasing(self, image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Apply random erasing (cutout) to image."""
        _, h, w = image.shape
        area = h * w
        target_area = rng.uniform(*self.random_erasing_scale) * area
        aspect_ratio = rng.uniform(0.3, 1 / 0.3)

        eh = int(np.sqrt(target_area * aspect_ratio))
        ew = int(np.sqrt(target_area / aspect_ratio))

        if eh < h and ew < w:
            y = rng.integers(0, h - eh)
            x = rng.integers(0, w - ew)
            image[:, y : y + eh, x : x + ew] = rng.uniform(0, 1)

        return image


class DINOMultiCropAugmentation:
    """
    Multi-crop augmentation strategy for DINO self-distillation.

    Produces 2 global crops (full image, heavily augmented) and N local crops
    (smaller random regions). Teacher sees global crops, student sees all.

    Args:
        global_crop_size: Size of global crops (typically the full image size).
        local_crop_size: Size of local crops (smaller, e.g., half the global size).
        n_local_crops: Number of local crops to generate.
        global_augmentation: Augmentation pipeline for global crops.
        local_augmentation: Augmentation pipeline for local crops.
    """

    def __init__(
        self,
        global_crop_size: int = 64,
        local_crop_size: int = 32,
        n_local_crops: int = 6,
        global_augmentation: Optional[AstronomyAugmentations] = None,
        local_augmentation: Optional[AstronomyAugmentations] = None,
    ):
        self.global_crop_size = global_crop_size
        self.local_crop_size = local_crop_size
        self.n_local_crops = n_local_crops

        # Global crops: stronger augmentation
        self.global_aug = global_augmentation or AstronomyAugmentations(
            gaussian_noise_std=0.03,
            brightness_range=(0.85, 1.15),
            contrast_range=(0.85, 1.15),
            random_erasing_prob=0.0,
        )

        # Local crops: lighter augmentation
        self.local_aug = local_augmentation or AstronomyAugmentations(
            gaussian_noise_std=0.02,
            brightness_range=(0.9, 1.1),
            contrast_range=(0.9, 1.1),
            random_erasing_prob=0.0,
        )

    def __call__(
        self, image: np.ndarray, rng: Optional[np.random.Generator] = None
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Generate multi-crop views of a single image.

        Args:
            image: NumPy array of shape (C, H, W), values in [0, 1].
            rng: NumPy random generator.

        Returns:
            Tuple of (global_crops, local_crops) where each is a list of
            augmented NumPy arrays.
        """
        if rng is None:
            rng = np.random.default_rng()

        _, h, w = image.shape

        # 2 global crops (full image, resized to global_crop_size)
        global_crops = []
        for _ in range(2):
            crop = self._resize(image, self.global_crop_size)
            crop = self.global_aug(crop, rng)
            global_crops.append(crop)

        # N local crops (random sub-regions, resized to local_crop_size)
        local_crops = []
        for _ in range(self.n_local_crops):
            crop = self._random_crop(image, self.local_crop_size, rng)
            crop = self.local_aug(crop, rng)
            local_crops.append(crop)

        return global_crops, local_crops

    def _random_crop(
        self, image: np.ndarray, crop_size: int, rng: np.random.Generator
    ) -> np.ndarray:
        """Extract a random crop and resize to crop_size."""
        _, h, w = image.shape

        # Random crop scale: 20-50% of image area
        scale = rng.uniform(0.2, 0.5)
        crop_h = max(4, int(h * np.sqrt(scale)))
        crop_w = max(4, int(w * np.sqrt(scale)))
        crop_h = min(crop_h, h)
        crop_w = min(crop_w, w)

        y = rng.integers(0, max(1, h - crop_h + 1))
        x = rng.integers(0, max(1, w - crop_w + 1))

        crop = image[:, y : y + crop_h, x : x + crop_w]
        return self._resize(crop, crop_size)

    @staticmethod
    def _resize(image: np.ndarray, size: int) -> np.ndarray:
        """Resize a CHW image to (C, size, size) via bilinear interpolation."""
        c, h, w = image.shape
        if h == size and w == size:
            return image.copy()

        y_coords = np.linspace(0, h - 1, size)
        x_coords = np.linspace(0, w - 1, size)
        xv, yv = np.meshgrid(x_coords, y_coords)

        x0 = np.floor(xv).astype(int)
        y0 = np.floor(yv).astype(int)
        x1 = np.minimum(x0 + 1, w - 1)
        y1 = np.minimum(y0 + 1, h - 1)

        dx = xv - x0
        dy = yv - y0

        resized_channels = []
        for channel_idx in range(c):
            resized = (
                image[channel_idx, y0, x0] * (1 - dx) * (1 - dy)
                + image[channel_idx, y0, x1] * dx * (1 - dy)
                + image[channel_idx, y1, x0] * (1 - dx) * dy
                + image[channel_idx, y1, x1] * dx * dy
            )
            resized_channels.append(resized.astype(np.float32))

        return np.stack(resized_channels, axis=0)
