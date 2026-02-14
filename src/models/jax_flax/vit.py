#!/usr/bin/env python3
"""
ViT backbone using JAX/Flax.

Pure Flax implementation of Vision Transformer for single-channel
astronomical images. No dependency on pretrained weights — designed
for training from scratch on JWST galaxy thumbnails.

Provides the same config-driven interface as the PyTorch variants.
"""

from typing import Optional, Tuple
from functools import partial

import jax
import jax.numpy as jnp
import flax.linen as nn
from einops import rearrange

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from src.models.vit_config import ViTConfig


class PatchEmbed(nn.Module):
    """
    Patch embedding layer: splits image into non-overlapping patches
    and projects each to the embedding dimension.

    Implemented as a single convolution with kernel_size=stride=patch_size.
    """

    config: ViTConfig

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            x: (B, 1, H, W) — CHW format, single channel.

        Returns:
            (B, num_patches, embed_dim)
        """
        # Convert CHW → HWC for Flax Conv (which expects NHWC)
        x = jnp.transpose(x, (0, 2, 3, 1))  # (B, H, W, 1)

        x = nn.Conv(
            features=self.config.embed_dim,
            kernel_size=(self.config.patch_size, self.config.patch_size),
            strides=(self.config.patch_size, self.config.patch_size),
            padding="VALID",
            name="proj",
        )(x)

        # (B, grid_h, grid_w, embed_dim) → (B, num_patches, embed_dim)
        B = x.shape[0]
        x = x.reshape(B, -1, self.config.embed_dim)
        return x


class MLP(nn.Module):
    """Transformer MLP block with GELU activation."""

    hidden_dim: int
    out_dim: int
    drop_rate: float = 0.0

    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.gelu(x)
        x = nn.Dropout(rate=self.drop_rate)(x, deterministic=deterministic)
        x = nn.Dense(self.out_dim)(x)
        x = nn.Dropout(rate=self.drop_rate)(x, deterministic=deterministic)
        return x


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention with optional dropout."""

    num_heads: int
    embed_dim: int
    attn_drop_rate: float = 0.0

    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        B, N, D = x.shape
        head_dim = self.embed_dim // self.num_heads
        scale = head_dim ** -0.5

        qkv = nn.Dense(3 * self.embed_dim, name="qkv")(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, head_dim)
        qkv = jnp.transpose(qkv, (2, 0, 3, 1, 4))  # (3, B, heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ jnp.transpose(k, (0, 1, 3, 2))) * scale
        attn = nn.softmax(attn, axis=-1)
        attn = nn.Dropout(rate=self.attn_drop_rate)(attn, deterministic=deterministic)

        out = (attn @ v).transpose(0, 2, 1, 3).reshape(B, N, self.embed_dim)
        out = nn.Dense(self.embed_dim, name="proj")(out)
        return out


class TransformerBlock(nn.Module):
    """Single transformer encoder block with pre-norm architecture."""

    config: ViTConfig
    drop_path_rate: float = 0.0

    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        # Self-attention with residual
        residual = x
        x = nn.LayerNorm(epsilon=self.config.layer_norm_eps)(x)
        x = MultiHeadAttention(
            num_heads=self.config.num_heads,
            embed_dim=self.config.embed_dim,
            attn_drop_rate=self.config.attn_drop_rate,
        )(x, deterministic=deterministic)
        x = nn.Dropout(rate=self.drop_path_rate)(x, deterministic=deterministic)
        x = residual + x

        # MLP with residual
        residual = x
        x = nn.LayerNorm(epsilon=self.config.layer_norm_eps)(x)
        x = MLP(
            hidden_dim=self.config.mlp_hidden_dim,
            out_dim=self.config.embed_dim,
            drop_rate=self.config.drop_rate,
        )(x, deterministic=deterministic)
        x = nn.Dropout(rate=self.drop_path_rate)(x, deterministic=deterministic)
        x = residual + x

        return x


class VisionTransformer(nn.Module):
    """
    Vision Transformer implemented in JAX/Flax.

    Follows the standard ViT architecture:
    1. Patch embedding (Conv projection)
    2. Positional embedding (learned)
    3. Optional CLS token
    4. N Transformer encoder blocks (pre-norm)
    5. Final LayerNorm

    Args:
        config: ViTConfig with architecture specifications.
    """

    config: ViTConfig

    @nn.compact
    def __call__(
        self, x: jnp.ndarray, deterministic: bool = True
    ) -> jnp.ndarray:
        """
        Forward pass returning full sequence (CLS + patches).

        Args:
            x: (B, 1, H, W) input images.
            deterministic: If True, disable dropout.

        Returns:
            (B, seq_length, embed_dim) — full token sequence.
        """
        B = x.shape[0]
        config = self.config

        # Patch embedding
        x = PatchEmbed(config=config)(x)  # (B, num_patches, embed_dim)

        # CLS token
        if config.use_cls_token:
            cls_token = self.param(
                "cls_token",
                nn.initializers.normal(stddev=0.02),
                (1, 1, config.embed_dim),
            )
            cls_tokens = jnp.broadcast_to(cls_token, (B, 1, config.embed_dim))
            x = jnp.concatenate([cls_tokens, x], axis=1)

        # Positional embedding — always allocate for full config resolution,
        # then interpolate if actual input is smaller (e.g. DINO local crops).
        full_seq_len = config.num_patches + (1 if config.use_cls_token else 0)
        pos_embed = self.param(
            "pos_embed",
            nn.initializers.normal(stddev=0.02),
            (1, full_seq_len, config.embed_dim),
        )

        actual_seq_len = x.shape[1]
        if actual_seq_len != full_seq_len:
            if config.use_cls_token:
                cls_pos = pos_embed[:, :1, :]
                patch_pos = pos_embed[:, 1:, :]
            else:
                cls_pos = None
                patch_pos = pos_embed

            orig_grid = int(patch_pos.shape[1] ** 0.5)
            N_actual = actual_seq_len - (1 if config.use_cls_token else 0)
            new_grid = int(N_actual ** 0.5)

            patch_pos = jnp.reshape(
                patch_pos, (1, orig_grid, orig_grid, config.embed_dim)
            )
            patch_pos = jax.image.resize(
                patch_pos,
                (1, new_grid, new_grid, config.embed_dim),
                method="bilinear",
            )
            patch_pos = jnp.reshape(
                patch_pos, (1, new_grid * new_grid, config.embed_dim)
            )

            if config.use_cls_token:
                pos_embed = jnp.concatenate([cls_pos, patch_pos], axis=1)
            else:
                pos_embed = patch_pos

        x = x + pos_embed

        x = nn.Dropout(rate=config.drop_rate)(x, deterministic=deterministic)

        # Transformer encoder blocks with stochastic depth
        dpr = [
            config.drop_path_rate * i / max(config.depth - 1, 1)
            for i in range(config.depth)
        ]
        for i in range(config.depth):
            x = TransformerBlock(
                config=config,
                drop_path_rate=dpr[i],
            )(x, deterministic=deterministic)

        # Final LayerNorm
        x = nn.LayerNorm(epsilon=config.layer_norm_eps)(x)

        return x

    def get_cls_token(
        self, x: jnp.ndarray, deterministic: bool = True
    ) -> jnp.ndarray:
        """Return only the CLS token embedding: (B, embed_dim)."""
        full_seq = self(x, deterministic=deterministic)
        return full_seq[:, 0, :]

    def get_patch_embeddings(
        self, x: jnp.ndarray, deterministic: bool = True
    ) -> jnp.ndarray:
        """Return only patch embeddings (no CLS): (B, num_patches, embed_dim)."""
        full_seq = self(x, deterministic=deterministic)
        return full_seq[:, 1:, :]


def build_vit(config: ViTConfig) -> VisionTransformer:
    """
    Factory function to create a Flax ViT from config.

    Note: Flax modules are stateless — you must call .init(rng, x) to
    create parameters before using .apply(params, x).

    Args:
        config: ViTConfig with model specifications.

    Returns:
        VisionTransformer Flax Module (not yet initialized).
    """
    config.validate()
    return VisionTransformer(config=config)
