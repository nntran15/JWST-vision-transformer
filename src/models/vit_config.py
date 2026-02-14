#!/usr/bin/env python3
"""
Shared ViT configuration dataclass.

Defines model hyperparameters for Vision Transformer in a framework-agnostic way.
Supports Tiny, Small, and Base variants, configurable for single-channel
astronomical images with small spatial dimensions (20-100px → resized to 64x64).
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ViTConfig:
    """
    Configuration for Vision Transformer architecture.

    Designed for single-channel FITS galaxy thumbnails. Default settings
    create a ViT-Tiny model working on 64×64 single-channel images with
    patch_size=8 (yielding 64 patches = 8×8 grid).

    Attributes:
        image_size: Input image spatial dimension (square).
        patch_size: Patch spatial dimension (image_size must be divisible).
        in_channels: Number of input channels (1 for grayscale FITS).
        embed_dim: Transformer embedding dimension.
        depth: Number of transformer encoder layers.
        num_heads: Number of attention heads (embed_dim must be divisible).
        mlp_ratio: MLP hidden dim = embed_dim * mlp_ratio.
        drop_rate: Dropout rate for embeddings and MLP.
        attn_drop_rate: Dropout rate for attention weights.
        drop_path_rate: Stochastic depth rate.
        use_cls_token: Whether to prepend a [CLS] token.
        layer_norm_eps: Epsilon for LayerNorm.
        model_name: Human-readable model name (auto-set by factory).
    """

    # Image parameters
    image_size: int = 64
    patch_size: int = 8
    in_channels: int = 1

    # Transformer parameters
    embed_dim: int = 192
    depth: int = 12
    num_heads: int = 3
    mlp_ratio: float = 4.0

    # Regularization
    drop_rate: float = 0.0
    attn_drop_rate: float = 0.0
    drop_path_rate: float = 0.1

    # Architecture options
    use_cls_token: bool = True
    layer_norm_eps: float = 1e-6

    # Metadata
    model_name: str = "vit_tiny"

    @property
    def num_patches(self) -> int:
        """Total number of patches in the image."""
        return (self.image_size // self.patch_size) ** 2

    @property
    def grid_size(self) -> int:
        """Number of patches along one axis."""
        return self.image_size // self.patch_size

    @property
    def patch_dim(self) -> int:
        """Flattened patch dimensionality (in_channels * patch_size^2)."""
        return self.in_channels * self.patch_size * self.patch_size

    @property
    def seq_length(self) -> int:
        """Sequence length including optional CLS token."""
        return self.num_patches + (1 if self.use_cls_token else 0)

    @property
    def mlp_hidden_dim(self) -> int:
        """Hidden dimension of MLP blocks."""
        return int(self.embed_dim * self.mlp_ratio)

    def validate(self):
        """Validate configuration consistency."""
        assert self.image_size % self.patch_size == 0, (
            f"image_size ({self.image_size}) must be divisible by "
            f"patch_size ({self.patch_size})"
        )
        assert self.embed_dim % self.num_heads == 0, (
            f"embed_dim ({self.embed_dim}) must be divisible by "
            f"num_heads ({self.num_heads})"
        )
        assert self.in_channels >= 1, "in_channels must be >= 1"


# ---- Preset Configurations ----

_VIT_PRESETS = {
    "tiny": {
        "embed_dim": 192,
        "depth": 12,
        "num_heads": 3,
        "model_name": "vit_tiny",
    },
    "small": {
        "embed_dim": 384,
        "depth": 12,
        "num_heads": 6,
        "model_name": "vit_small",
    },
    "base": {
        "embed_dim": 768,
        "depth": 12,
        "num_heads": 12,
        "model_name": "vit_base",
    },
}


def get_vit_config(
    size: str = "tiny",
    image_size: int = 64,
    patch_size: int = 8,
    in_channels: int = 1,
    drop_rate: float = 0.0,
    attn_drop_rate: float = 0.0,
    drop_path_rate: float = 0.1,
    **kwargs,
) -> ViTConfig:
    """
    Factory function to create a ViTConfig from a size preset.

    Args:
        size: One of 'tiny', 'small', 'base'.
        image_size: Input image size (default: 64 for galaxy thumbnails).
        patch_size: Patch size (default: 8, yielding 64 patches).
        in_channels: Input channels (default: 1 for FITS).
        drop_rate: Dropout rate.
        attn_drop_rate: Attention dropout rate.
        drop_path_rate: Stochastic depth rate.
        **kwargs: Additional overrides for ViTConfig fields.

    Returns:
        ViTConfig instance with validated parameters.
    """
    size = size.lower()
    if size not in _VIT_PRESETS:
        raise ValueError(f"Unknown ViT size '{size}'. Choose from: {list(_VIT_PRESETS.keys())}")

    preset = _VIT_PRESETS[size]

    config = ViTConfig(
        image_size=image_size,
        patch_size=patch_size,
        in_channels=in_channels,
        embed_dim=preset["embed_dim"],
        depth=preset["depth"],
        num_heads=preset["num_heads"],
        mlp_ratio=kwargs.get("mlp_ratio", 4.0),
        drop_rate=drop_rate,
        attn_drop_rate=attn_drop_rate,
        drop_path_rate=drop_path_rate,
        model_name=preset["model_name"],
        **{k: v for k, v in kwargs.items() if k not in ("mlp_ratio",)},
    )

    config.validate()
    return config
