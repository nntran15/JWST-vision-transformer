#!/usr/bin/env python3
"""Shared FITS preprocessing helpers for loading, normalization, and display."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


DEFAULT_N_SIGMA = 8.0


def sanitize_image_data(data: np.ndarray) -> np.ndarray:
    """Replace NaN/Inf and cast to float32."""
    return np.nan_to_num(
        np.asarray(data, dtype=np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )


def ensure_chw(data: np.ndarray) -> np.ndarray:
    """Convert FITS image data to CHW layout."""
    image = sanitize_image_data(data)

    if image.ndim == 2:
        return image[np.newaxis, :, :]

    if image.ndim > 3:
        image = image.reshape((-1, image.shape[-2], image.shape[-1]))

    if image.ndim != 3:
        raise ValueError(f"Expected 2D or 3D FITS data, got shape {image.shape}")

    if image.shape[0] <= 16:
        return image.astype(np.float32)

    if image.shape[-1] <= 16:
        return np.moveaxis(image, -1, 0).astype(np.float32)

    return image[:1].astype(np.float32)


def is_arcsinh_normalized(header: Optional[dict]) -> bool:
    """Return True if the FITS header says the image is already arcsinh-normalized."""
    if header is None:
        return False
    return "ARCSINH" in str(header.get("NORM", "")).upper()


def get_header_n_sigma(header: Optional[dict], default: float = DEFAULT_N_SIGMA) -> float:
    """Read the arcsinh n-sigma value from a FITS header."""
    if header is None:
        return default

    try:
        return float(header.get("NSIGMA", default))
    except Exception:
        return default


def get_header_rms_values(header: Optional[dict], n_channels: int) -> Optional[np.ndarray]:
    """Extract per-channel local RMS values from a FITS header."""
    if header is None:
        return None

    if n_channels == 1:
        for key in ("LRMS", "LRMS1"):
            if key in header:
                return np.array([max(float(header[key]), 1e-12)], dtype=np.float32)

    values = []
    found = False
    for channel_idx in range(n_channels):
        key = f"LRMS{channel_idx + 1}"
        if key in header:
            values.append(max(float(header[key]), 1e-12))
            found = True
        else:
            values.append(np.nan)

    if not found:
        return None

    rms_values = np.asarray(values, dtype=np.float32)
    finite = rms_values[np.isfinite(rms_values) & (rms_values > 0)]
    if finite.size == 0:
        return None

    fill_value = float(np.median(finite))
    rms_values[~np.isfinite(rms_values) | (rms_values <= 0)] = fill_value
    return rms_values


def normalize_arcsinh_rms(
    image: np.ndarray,
    rms_values: np.ndarray,
    n_sigma: float = DEFAULT_N_SIGMA,
) -> np.ndarray:
    """Normalize each channel with an arcsinh stretch anchored to local RMS."""
    chw = ensure_chw(image)
    rms = np.asarray(rms_values, dtype=np.float32)

    if rms.size == 1 and chw.shape[0] > 1:
        rms = np.repeat(rms, chw.shape[0])
    if rms.size != chw.shape[0]:
        raise ValueError(
            f"RMS count ({rms.size}) does not match channel count ({chw.shape[0]})"
        )

    denom = np.arcsinh(1.0)
    normalized = np.empty_like(chw)
    for channel_idx in range(chw.shape[0]):
        scale = max(float(rms[channel_idx]) * float(n_sigma), 1e-12)
        channel = np.arcsinh(chw[channel_idx] / scale) / denom
        normalized[channel_idx] = np.clip(channel, 0.0, 1.0)

    return normalized.astype(np.float32)


def _normalize_minmax(image: np.ndarray) -> np.ndarray:
    normalized = np.empty_like(image)
    for channel_idx, channel in enumerate(image):
        vmin = float(channel.min())
        vmax = float(channel.max())
        if vmax - vmin > 1e-10:
            normalized[channel_idx] = (channel - vmin) / (vmax - vmin)
        else:
            normalized[channel_idx] = np.zeros_like(channel)
    return normalized.astype(np.float32)


def _normalize_percentile(
    image: np.ndarray,
    percentile_clip: Tuple[float, float] = (1.0, 99.0),
) -> np.ndarray:
    low, high = percentile_clip
    normalized = np.empty_like(image)
    for channel_idx, channel in enumerate(image):
        vmin = float(np.percentile(channel, low))
        vmax = float(np.percentile(channel, high))
        if vmax - vmin > 1e-10:
            clipped = np.clip(channel, vmin, vmax)
            normalized[channel_idx] = (clipped - vmin) / (vmax - vmin)
        else:
            normalized[channel_idx] = np.zeros_like(channel)
    return normalized.astype(np.float32)


def _normalize_zscore(image: np.ndarray) -> np.ndarray:
    normalized = np.empty_like(image)
    for channel_idx, channel in enumerate(image):
        mean = float(channel.mean())
        std = float(channel.std())
        if std > 1e-10:
            zscore = np.clip((channel - mean) / std, -3.0, 3.0)
            normalized[channel_idx] = (zscore + 3.0) / 6.0
        else:
            normalized[channel_idx] = np.zeros_like(channel)
    return normalized.astype(np.float32)


def normalize_fits_data(
    image: np.ndarray,
    header: Optional[dict] = None,
    normalization: str = "header",
    percentile_clip: Tuple[float, float] = (1.0, 99.0),
    n_sigma: float = DEFAULT_N_SIGMA,
) -> np.ndarray:
    """Normalize FITS data while honoring metadata when available."""
    chw = ensure_chw(image)
    strategy = normalization.lower()

    if strategy == "header":
        if is_arcsinh_normalized(header):
            return np.clip(chw, 0.0, 1.0).astype(np.float32)

        rms_values = get_header_rms_values(header, chw.shape[0])
        if rms_values is not None:
            return normalize_arcsinh_rms(
                chw,
                rms_values,
                n_sigma=get_header_n_sigma(header, n_sigma),
            )

        return _normalize_percentile(chw, percentile_clip)

    if strategy == "arcsinh_rms":
        rms_values = get_header_rms_values(header, chw.shape[0])
        if rms_values is None:
            return _normalize_percentile(chw, percentile_clip)
        return normalize_arcsinh_rms(
            chw,
            rms_values,
            n_sigma=get_header_n_sigma(header, n_sigma),
        )

    if strategy == "minmax":
        return _normalize_minmax(chw)
    if strategy == "percentile":
        return _normalize_percentile(chw, percentile_clip)
    if strategy == "zscore":
        return _normalize_zscore(chw)

    raise ValueError(f"Unknown normalization strategy: {normalization}")


def resize_2d(image: np.ndarray, size: int) -> np.ndarray:
    """Resize a single 2D channel to the target size."""
    h, w = image.shape
    if h == size and w == size:
        return image.astype(np.float32)

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


def resize_chw(image: np.ndarray, size: int) -> np.ndarray:
    """Resize a CHW image to a square output size."""
    chw = ensure_chw(image)
    if chw.shape[-2:] == (size, size):
        return chw.astype(np.float32)
    return np.stack([resize_2d(channel, size) for channel in chw], axis=0).astype(np.float32)


def collapse_to_single_channel(image: np.ndarray) -> np.ndarray:
    """Project normalized CHW data to one grayscale channel for the current SSL stack."""
    chw = ensure_chw(image)
    if chw.shape[0] == 1:
        return chw.astype(np.float32)
    return np.max(chw, axis=0, keepdims=True).astype(np.float32)


def _range_gate(values: np.ndarray, low: float, high: float) -> np.ndarray:
    """Map values into [0, 1] across a finite interval."""
    scale = max(high - low, 1e-12)
    return np.clip((values - low) / scale, 0.0, 1.0).astype(np.float32)


def _box_blur_2d(image: np.ndarray) -> np.ndarray:
    """Apply a small 3x3 box blur to a 2D float image."""
    padded = np.pad(image, ((1, 1), (1, 1)), mode="edge")
    blurred = (
        padded[:-2, :-2]
        + padded[:-2, 1:-1]
        + padded[:-2, 2:]
        + padded[1:-1, :-2]
        + padded[1:-1, 1:-1]
        + padded[1:-1, 2:]
        + padded[2:, :-2]
        + padded[2:, 1:-1]
        + padded[2:, 2:]
    ) / 9.0
    return blurred.astype(np.float32)


def compose_multiband_rgb(image: np.ndarray) -> np.ndarray:
    """Compose an RGB image from normalized CHW data."""
    chw = np.clip(ensure_chw(image), 0.0, 1.0)

    if chw.shape[0] >= 4:
        red = chw[3]
        green = chw[2]
        blue = 0.5 * (chw[0] + chw[1])
    elif chw.shape[0] == 3:
        red, green, blue = chw[:3]
    else:
        red = green = blue = chw[0]

    rgb = np.stack([red, green, blue], axis=-1)
    base_luminance = np.max(rgb, axis=-1, keepdims=True)
    neutral_rgb = np.repeat(base_luminance, 3, axis=-1)
    shared_signal = np.median(rgb, axis=-1, keepdims=True)

    blurred_rgb = np.stack(
        [_box_blur_2d(rgb[..., channel_idx]) for channel_idx in range(3)],
        axis=-1,
    )
    blurred_luminance = np.max(blurred_rgb, axis=-1, keepdims=True)
    blurred_neutral = np.repeat(blurred_luminance, 3, axis=-1)

    chroma = rgb - neutral_rgb
    blurred_chroma = blurred_rgb - blurred_neutral

    color_scale = _range_gate(shared_signal, 0.08, 0.42)
    color_scale = color_scale * color_scale * color_scale

    # Keep morphology in the neutral luminance path and smooth only chroma.
    luminance = base_luminance
    smoothed_chroma = (
        blurred_chroma * (1.0 - color_scale)
        + chroma * color_scale
    )
    rgb = np.repeat(luminance, 3, axis=-1) + smoothed_chroma * (0.15 + 0.85 * color_scale)
    return np.clip(rgb, 0.0, 1.0).astype(np.float32)


def make_display_image(
    image: np.ndarray,
    header: Optional[dict] = None,
    normalization: str = "header",
    percentile_clip: Tuple[float, float] = (1.0, 99.0),
    n_sigma: float = DEFAULT_N_SIGMA,
    target_size: Optional[int] = None,
) -> np.ndarray:
    """Prepare a 2D grayscale or 3D RGB image for visualization."""
    normalized = normalize_fits_data(
        image,
        header=header,
        normalization=normalization,
        percentile_clip=percentile_clip,
        n_sigma=n_sigma,
    )
    if target_size is not None:
        normalized = resize_chw(normalized, target_size)

    if normalized.shape[0] == 1:
        return normalized[0]
    return compose_multiband_rgb(normalized)