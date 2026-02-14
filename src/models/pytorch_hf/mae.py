#!/usr/bin/env python3
"""
Masked Autoencoder (MAE) using HuggingFace Transformers ViT backbone.

Adapts the HuggingFace ViTMAE architecture for single-channel FITS
galaxy thumbnails. Implements the same MAE interface as the timm version.

Follows He et al. (2022) "Masked Autoencoders Are Scalable Vision Learners".
"""

from typing import Optional

import torch
import torch.nn as nn
from einops import rearrange

from ..vit_config import ViTConfig

try:
    from transformers import ViTMAEConfig, ViTMAEForPreTraining, ViTMAEModel
except ImportError:
    raise ImportError("transformers is required: pip install transformers")


class MAE(nn.Module):
    """
    MAE using HuggingFace's ViTMAE implementation.

    Adapts ViTMAEForPreTraining for:
    - 1-channel input (FITS grayscale images)
    - Custom image_size / patch_size for small thumbnails
    - Configurable ViT sizes (Tiny/Small/Base)

    Args:
        config: ViTConfig for encoder parameters.
        mask_ratio: Fraction of patches to mask (default: 0.75).
        decoder_embed_dim: Decoder hidden dimension.
        decoder_depth: Number of decoder layers.
        decoder_num_heads: Number of decoder attention heads.
        norm_pix_loss: Normalize reconstruction targets per-patch.
    """

    def __init__(
        self,
        config: ViTConfig,
        mask_ratio: float = 0.75,
        decoder_embed_dim: int = 128,
        decoder_depth: int = 4,
        decoder_num_heads: int = 4,
        norm_pix_loss: bool = True,
    ):
        super().__init__()

        self.config = config
        self.mask_ratio = mask_ratio
        self.norm_pix_loss = norm_pix_loss

        # Create HuggingFace MAE config
        hf_config = ViTMAEConfig(
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
            # Decoder config
            decoder_hidden_size=decoder_embed_dim,
            decoder_num_hidden_layers=decoder_depth,
            decoder_num_attention_heads=decoder_num_heads,
            decoder_intermediate_size=decoder_embed_dim * 4,
            # MAE-specific
            mask_ratio=mask_ratio,
            norm_pix_loss=norm_pix_loss,
        )

        self.mae = ViTMAEForPreTraining(hf_config)
        self.embed_dim = config.embed_dim
        self.patch_size = config.patch_size
        self.in_channels = config.in_channels

    def forward(
        self, images: torch.Tensor, mask_ratio: Optional[float] = None
    ) -> dict:
        """
        Full MAE forward pass using HuggingFace implementation.

        Args:
            images: (B, 1, H, W) input images.
            mask_ratio: Override mask ratio (optional).

        Returns:
            Dict with 'loss', 'pred', 'mask' keys.
        """
        # HuggingFace MAE handles masking internally
        if mask_ratio is not None and mask_ratio != self.mask_ratio:
            # Temporarily override mask ratio
            self.mae.config.mask_ratio = mask_ratio

        outputs = self.mae(pixel_values=images)

        # Restore original mask ratio
        self.mae.config.mask_ratio = self.mask_ratio

        return {
            "loss": outputs.loss,
            "pred": outputs.logits,
            "mask": outputs.mask,
        }

    def get_encoder(self) -> nn.Module:
        """
        Return the encoder backbone for downstream tasks.

        Returns a wrapper that provides the same interface as TimmViTWrapper.
        """
        return _HFMAEEncoderWrapper(self.mae.vit, self.config)

    def patchify(self, images: torch.Tensor) -> torch.Tensor:
        """Convert images to patch sequences."""
        p = self.patch_size
        c = self.in_channels
        h = w = self.config.image_size // p
        return rearrange(
            images, "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
            p1=p, p2=p, h=h, w=w
        )

    def unpatchify(self, patches: torch.Tensor) -> torch.Tensor:
        """Convert patch sequences back to images."""
        p = self.patch_size
        c = self.in_channels
        h = w = self.config.grid_size
        return rearrange(
            patches, "b (h w) (p1 p2 c) -> b c (h p1) (w p2)",
            p1=p, p2=p, h=h, w=w, c=c
        )


class _HFMAEEncoderWrapper(nn.Module):
    """
    Wrapper around HuggingFace ViTMAE encoder to provide the standard
    ViT interface (forward_features, get_cls_token, etc.).
    """

    def __init__(self, vit_mae_model: ViTMAEModel, config: ViTConfig):
        super().__init__()
        self.vit = vit_mae_model
        self.config = config
        self.embed_dim = config.embed_dim
        self.num_patches = config.num_patches

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return CLS token embedding."""
        outputs = self.vit(pixel_values=x, noise=torch.zeros(
            x.shape[0], self.num_patches, device=x.device
        ))
        return outputs.last_hidden_state[:, 0, :]

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return full sequence (CLS + patches)."""
        outputs = self.vit(pixel_values=x, noise=torch.zeros(
            x.shape[0], self.num_patches, device=x.device
        ))
        return outputs.last_hidden_state

    def get_cls_token(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)

    def get_patch_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        full = self.forward_features(x)
        return full[:, 1:, :]
