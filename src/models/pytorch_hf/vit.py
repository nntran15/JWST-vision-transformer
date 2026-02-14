#!/usr/bin/env python3
"""
ViT backbone using HuggingFace Transformers.

Wraps HuggingFace's ViTModel with overrides for:
- Single-channel (1-ch) input for FITS astronomical images
- Custom patch_size and image_size for small galaxy thumbnails (64×64)
- Configurable model size (Tiny/Small/Base)

Provides a unified interface: build_vit(config) → nn.Module
"""

import torch
import torch.nn as nn

try:
    from transformers import ViTModel, ViTConfig as HFViTConfig
except ImportError:
    raise ImportError("transformers is required: pip install transformers")

from ..vit_config import ViTConfig


class HFViTWrapper(nn.Module):
    """
    Wrapper around HuggingFace's ViTModel adapted for single-channel
    astronomical images.

    Args:
        config: ViTConfig instance with model parameters.
    """

    def __init__(self, config: ViTConfig):
        super().__init__()
        self.config = config

        # Create HuggingFace ViTConfig
        hf_config = HFViTConfig(
            image_size=config.image_size,
            patch_size=config.patch_size,
            num_channels=config.in_channels,
            hidden_size=config.embed_dim,
            num_hidden_layers=config.depth,
            num_attention_heads=config.num_heads,
            intermediate_size=config.mlp_hidden_dim,
            hidden_dropout_prob=config.drop_rate,
            attention_probs_dropout_prob=config.attn_drop_rate,
            layer_norm_eps=config.layer_norm_eps,
        )

        self.vit = ViTModel(hf_config, add_pooling_layer=False)
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
        outputs = self.vit(pixel_values=x, interpolate_pos_encoding=True)
        return outputs.last_hidden_state[:, 0, :]

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning full sequence (CLS + patch tokens).

        Args:
            x: Input tensor of shape (B, 1, H, W).

        Returns:
            Full sequence tensor of shape (B, 1 + num_patches, embed_dim).
        """
        outputs = self.vit(pixel_values=x, interpolate_pos_encoding=True)
        return outputs.last_hidden_state

    def get_patch_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning only patch token embeddings (no CLS).

        Args:
            x: Input tensor of shape (B, 1, H, W).

        Returns:
            Patch embeddings of shape (B, num_patches, embed_dim).
        """
        full_seq = self.forward_features(x)
        return full_seq[:, 1:, :]

    def get_cls_token(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning only the CLS token embedding.

        Args:
            x: Input tensor of shape (B, 1, H, W).

        Returns:
            CLS embedding of shape (B, embed_dim).
        """
        return self.forward(x)


def build_vit(config: ViTConfig) -> HFViTWrapper:
    """
    Factory function to create a HuggingFace-based ViT from config.

    Args:
        config: ViTConfig with model specifications.

    Returns:
        HFViTWrapper nn.Module.
    """
    config.validate()
    return HFViTWrapper(config)
