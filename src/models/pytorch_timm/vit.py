#!/usr/bin/env python3
"""
ViT backbone using the timm library.

Wraps timm's Vision Transformer with overrides for:
- Single-channel (1-ch) input for FITS astronomical images
- Custom patch_size and image_size for small galaxy thumbnails (64×64)
- Configurable model size (Tiny/Small/Base)

Provides a unified interface: build_vit(config) → nn.Module
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
    from timm.models.vision_transformer import VisionTransformer as TimmViT
except ImportError:
    raise ImportError("timm is required: pip install timm")

from ..vit_config import ViTConfig


class TimmViTWrapper(nn.Module):
    """
    Wrapper around timm's VisionTransformer adapted for single-channel
    astronomical images.

    The wrapper:
    1. Creates a timm ViT with correct in_chans, img_size, patch_size
    2. Disables the classification head (used for SSL, not classification)
    3. Exposes forward_features() for CLS/patch embeddings

    Args:
        config: ViTConfig instance with model parameters.
    """

    def __init__(self, config: ViTConfig):
        super().__init__()
        self.config = config

        # Map our config to timm model names for initialization
        timm_model_map = {
            "vit_tiny": "vit_tiny_patch16_224",
            "vit_small": "vit_small_patch16_224",
            "vit_base": "vit_base_patch16_224",
        }

        model_name = timm_model_map.get(config.model_name, "vit_tiny_patch16_224")

        # Create timm model with our custom parameters
        self.vit = timm.create_model(
            model_name,
            pretrained=False,
            img_size=config.image_size,
            patch_size=config.patch_size,
            in_chans=config.in_channels,
            embed_dim=config.embed_dim,
            depth=config.depth,
            num_heads=config.num_heads,
            mlp_ratio=config.mlp_ratio,
            drop_rate=config.drop_rate,
            attn_drop_rate=config.attn_drop_rate,
            drop_path_rate=config.drop_path_rate,
            num_classes=0,  # No classification head
        )

        # Allow variable-resolution input (e.g. DINO 32×32 local crops)
        # without changing internal data format (dynamic_img_size changes to NHWC)
        self.vit.patch_embed.strict_img_size = False

        self.embed_dim = config.embed_dim
        self.num_patches = config.num_patches

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning CLS token embedding.

        Args:
            x: Input tensor of shape (B, 1, H, W).

        Returns:
            CLS token embedding of shape (B, embed_dim).
        """
        return self.forward_features(x)[:, 0, :]

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning full sequence (CLS + patch tokens).

        Args:
            x: Input tensor of shape (B, 1, H, W).

        Returns:
            Full sequence tensor of shape (B, 1 + num_patches, embed_dim).
        """
        # timm's forward_features typically returns only CLS token when
        # num_classes=0. We need to access the internals for patch tokens.
        x = self.vit.patch_embed(x)
        N = x.shape[1]  # actual number of patches (varies with input size)

        cls_token = self.vit.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)

        # Interpolate positional embeddings if input resolution differs
        pos_embed = self.vit.pos_embed  # (1, 1 + num_patches_orig, embed_dim)
        if pos_embed.shape[1] != x.shape[1]:
            cls_pos = pos_embed[:, :1, :]
            patch_pos = pos_embed[:, 1:, :]
            orig_grid = int(patch_pos.shape[1] ** 0.5)
            new_grid = int(N ** 0.5)
            patch_pos = patch_pos.reshape(1, orig_grid, orig_grid, -1).permute(0, 3, 1, 2)
            patch_pos = F.interpolate(
                patch_pos, size=(new_grid, new_grid),
                mode="bilinear", align_corners=False,
            )
            patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, new_grid * new_grid, -1)
            pos_embed = torch.cat([cls_pos, patch_pos], dim=1)

        x = x + pos_embed
        x = self.vit.pos_drop(x)

        for blk in self.vit.blocks:
            x = blk(x)

        x = self.vit.norm(x)
        return x

    def get_patch_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning only patch token embeddings (no CLS).

        Args:
            x: Input tensor of shape (B, 1, H, W).

        Returns:
            Patch embeddings of shape (B, num_patches, embed_dim).
        """
        full_seq = self.forward_features(x)
        return full_seq[:, 1:, :]  # Remove CLS token

    def get_cls_token(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning only the CLS token embedding.

        Args:
            x: Input tensor of shape (B, 1, H, W).

        Returns:
            CLS embedding of shape (B, embed_dim).
        """
        full_seq = self.forward_features(x)
        return full_seq[:, 0, :]


def build_vit(config: ViTConfig) -> TimmViTWrapper:
    """
    Factory function to create a timm-based ViT from config.

    Args:
        config: ViTConfig with model specifications.

    Returns:
        TimmViTWrapper nn.Module.
    """
    config.validate()
    return TimmViTWrapper(config)
