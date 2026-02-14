#!/usr/bin/env python3
"""
Masked Autoencoder (MAE) using JAX/Flax.

Pure Flax implementation of He et al. (2022) MAE architecture for
single-channel FITS galaxy thumbnails. Uses jax.random for mask
sampling and custom transformer blocks for the decoder.

Design:
- Encoder: Full ViT operating on visible patches only
- Decoder: Lightweight transformer that reconstructs masked patches
- Loss: MSE on normalized pixel values of masked patches
"""

from typing import Tuple, Optional
from functools import partial

import jax
import jax.numpy as jnp
import flax.linen as nn
from einops import rearrange

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from src.models.vit_config import ViTConfig
from src.models.jax_flax.vit import VisionTransformer, PatchEmbed, TransformerBlock, MLP, MultiHeadAttention


class MAEDecoder(nn.Module):
    """
    Lightweight transformer decoder for MAE pixel reconstruction.

    Attributes:
        encoder_embed_dim: Dimension from encoder output.
        decoder_embed_dim: Internal decoder dimension.
        decoder_depth: Number of decoder transformer layers.
        decoder_num_heads: Number of attention heads.
        num_patches: Total patches in the image.
        patch_size: Spatial patch size.
        in_channels: Number of input channels.
    """

    encoder_embed_dim: int = 192
    decoder_embed_dim: int = 128
    decoder_depth: int = 4
    decoder_num_heads: int = 4
    num_patches: int = 64
    patch_size: int = 8
    in_channels: int = 1

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        ids_restore: jnp.ndarray,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        """
        Decode visible + mask tokens to pixel predictions.

        Args:
            x: (B, N_visible + 1, encoder_embed_dim) — encoder output with CLS.
            ids_restore: (B, num_patches) — indices to unshuffle.
            deterministic: If True, disable dropout.

        Returns:
            (B, num_patches, patch_size^2 * in_channels) pixel predictions.
        """
        B = x.shape[0]

        # Project to decoder embedding dim
        x = nn.Dense(self.decoder_embed_dim, name="decoder_embed")(x)

        # Mask token
        mask_token = self.param(
            "mask_token",
            nn.initializers.normal(stddev=0.02),
            (1, 1, self.decoder_embed_dim),
        )

        # Number of mask tokens needed
        n_mask = self.num_patches + 1 - x.shape[1]
        mask_tokens = jnp.broadcast_to(mask_token, (B, n_mask, self.decoder_embed_dim))

        # Remove CLS, append mask tokens, unshuffle
        x_no_cls = x[:, 1:, :]
        x_combined = jnp.concatenate([x_no_cls, mask_tokens], axis=1)

        # Unshuffle via gather
        idx = ids_restore[:, :, None]
        idx = jnp.broadcast_to(idx, (B, self.num_patches, self.decoder_embed_dim))
        x_combined = jnp.take_along_axis(x_combined, idx, axis=1)

        # Re-add CLS token
        x = jnp.concatenate([x[:, :1, :], x_combined], axis=1)

        # Positional embedding
        decoder_pos_embed = self.param(
            "decoder_pos_embed",
            nn.initializers.normal(stddev=0.02),
            (1, self.num_patches + 1, self.decoder_embed_dim),
        )
        x = x + decoder_pos_embed

        # Decoder config for TransformerBlocks
        decoder_config = ViTConfig(
            embed_dim=self.decoder_embed_dim,
            depth=self.decoder_depth,
            num_heads=self.decoder_num_heads,
            mlp_ratio=4.0,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
        )

        # Decoder transformer blocks
        for i in range(self.decoder_depth):
            x = TransformerBlock(
                config=decoder_config, drop_path_rate=0.0
            )(x, deterministic=deterministic)

        x = nn.LayerNorm()(x)

        # Project to pixel values (remove CLS)
        patch_dim = self.patch_size * self.patch_size * self.in_channels
        x = nn.Dense(patch_dim, name="decoder_pred")(x[:, 1:, :])

        return x


class MAE(nn.Module):
    """
    Masked Autoencoder (MAE) in JAX/Flax.

    Attributes:
        config: ViTConfig for the encoder.
        mask_ratio: Fraction of patches to mask.
        decoder_embed_dim: Decoder embedding dimension.
        decoder_depth: Decoder transformer depth.
        decoder_num_heads: Decoder attention heads.
        norm_pix_loss: Normalize reconstruction targets per-patch.
    """

    config: ViTConfig
    mask_ratio: float = 0.75
    decoder_embed_dim: int = 128
    decoder_depth: int = 4
    decoder_num_heads: int = 4
    norm_pix_loss: bool = True

    def setup(self):
        self.encoder = VisionTransformer(config=self.config)
        self.decoder = MAEDecoder(
            encoder_embed_dim=self.config.embed_dim,
            decoder_embed_dim=self.decoder_embed_dim,
            decoder_depth=self.decoder_depth,
            decoder_num_heads=self.decoder_num_heads,
            num_patches=self.config.num_patches,
            patch_size=self.config.patch_size,
            in_channels=self.config.in_channels,
        )

    def patchify(self, images: jnp.ndarray) -> jnp.ndarray:
        """(B, C, H, W) → (B, num_patches, patch_dim)"""
        p = self.config.patch_size
        c = self.config.in_channels
        h = w = self.config.grid_size
        return rearrange(
            images, "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
            p1=p, p2=p, h=h, w=w
        )

    def random_masking(
        self, rng: jax.random.PRNGKey, x: jnp.ndarray, mask_ratio: float
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Random masking: shuffle and keep subset.

        Args:
            rng: JAX PRNG key.
            x: (B, N, D) patch embeddings.
            mask_ratio: Fraction to mask.

        Returns:
            x_masked: (B, N_visible, D)
            mask: (B, N) binary mask (1=masked)
            ids_restore: (B, N) unshuffle indices
        """
        B, N, D = x.shape
        num_keep = int(N * (1 - mask_ratio))

        noise = jax.random.uniform(rng, (B, N))
        ids_shuffle = jnp.argsort(noise, axis=1)
        ids_restore = jnp.argsort(ids_shuffle, axis=1)

        ids_keep = ids_shuffle[:, :num_keep]
        x_masked = jnp.take_along_axis(
            x, ids_keep[:, :, None].repeat(D, axis=2), axis=1
        )

        mask = jnp.ones((B, N))
        mask = mask.at[:, :num_keep].set(0)
        mask = jnp.take_along_axis(mask, ids_restore, axis=1)

        return x_masked, mask, ids_restore

    @nn.compact
    def __call__(
        self,
        images: jnp.ndarray,
        rng: jax.random.PRNGKey,
        mask_ratio: Optional[float] = None,
        deterministic: bool = True,
    ) -> dict:
        """
        Full MAE forward pass.

        Args:
            images: (B, 1, H, W) input images.
            rng: JAX PRNG key for masking.
            mask_ratio: Override mask ratio.
            deterministic: Disable dropout.

        Returns:
            Dict with 'loss', 'pred', 'mask'.
        """
        if mask_ratio is None:
            mask_ratio = self.mask_ratio

        config = self.config
        B = images.shape[0]

        # Split RNG for masking and dropout
        rng_mask, rng_drop = jax.random.split(rng)

        # --- ENCODER ---
        # Patch embedding
        images_nhwc = jnp.transpose(images, (0, 2, 3, 1))
        x = nn.Conv(
            features=config.embed_dim,
            kernel_size=(config.patch_size, config.patch_size),
            strides=(config.patch_size, config.patch_size),
            padding="VALID",
            name="encoder_patch_embed",
        )(images_nhwc)
        x = x.reshape(B, -1, config.embed_dim)  # (B, N, D)

        # Positional embedding (patches only)
        pos_embed = self.param(
            "encoder_pos_embed",
            nn.initializers.normal(stddev=0.02),
            (1, config.num_patches + 1, config.embed_dim),
        )
        x = x + pos_embed[:, 1:, :]

        # Random masking
        x, mask, ids_restore = self.random_masking(rng_mask, x, mask_ratio)

        # CLS token
        cls_token = self.param(
            "encoder_cls_token",
            nn.initializers.normal(stddev=0.02),
            (1, 1, config.embed_dim),
        )
        cls_tokens = jnp.broadcast_to(cls_token, (B, 1, config.embed_dim))
        cls_tokens = cls_tokens + pos_embed[:, :1, :]
        x = jnp.concatenate([cls_tokens, x], axis=1)

        # Encoder transformer blocks
        dpr = [
            config.drop_path_rate * i / max(config.depth - 1, 1)
            for i in range(config.depth)
        ]
        for i in range(config.depth):
            x = TransformerBlock(
                config=config, drop_path_rate=dpr[i]
            )(x, deterministic=deterministic)

        x = nn.LayerNorm(epsilon=config.layer_norm_eps, name="encoder_norm")(x)

        # --- DECODER ---
        pred = self.decoder(x, ids_restore, deterministic=deterministic)

        # --- LOSS ---
        target = self.patchify(images)

        if self.norm_pix_loss:
            mean = target.mean(axis=-1, keepdims=True)
            var = target.var(axis=-1, keepdims=True)
            target = (target - mean) / jnp.sqrt(var + 1e-6)

        loss = jnp.square(pred - target)
        loss = loss.mean(axis=-1)  # per-patch loss
        loss = (loss * mask).sum() / mask.sum()

        return {
            "loss": loss,
            "pred": pred,
            "mask": mask,
        }

    @nn.compact
    def encode(self, images: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        """
        Encode-only forward pass (no masking, no decoder).

        Returns full sequence (CLS + all patches) from the encoder.
        Used for downstream embedding extraction.

        Args:
            images: (B, 1, H, W) input images.
            deterministic: Disable dropout.

        Returns:
            (B, 1 + num_patches, embed_dim) encoder output.
        """
        config = self.config
        B = images.shape[0]

        images_nhwc = jnp.transpose(images, (0, 2, 3, 1))
        x = nn.Conv(
            features=config.embed_dim,
            kernel_size=(config.patch_size, config.patch_size),
            strides=(config.patch_size, config.patch_size),
            padding="VALID",
            name="encoder_patch_embed",
        )(images_nhwc)
        x = x.reshape(B, -1, config.embed_dim)

        pos_embed = self.param(
            "encoder_pos_embed",
            nn.initializers.normal(stddev=0.02),
            (1, config.num_patches + 1, config.embed_dim),
        )
        x = x + pos_embed[:, 1:, :]

        cls_token = self.param(
            "encoder_cls_token",
            nn.initializers.normal(stddev=0.02),
            (1, 1, config.embed_dim),
        )
        cls_tokens = jnp.broadcast_to(cls_token, (B, 1, config.embed_dim))
        cls_tokens = cls_tokens + pos_embed[:, :1, :]
        x = jnp.concatenate([cls_tokens, x], axis=1)

        dpr = [
            config.drop_path_rate * i / max(config.depth - 1, 1)
            for i in range(config.depth)
        ]
        for i in range(config.depth):
            x = TransformerBlock(
                config=config, drop_path_rate=dpr[i]
            )(x, deterministic=deterministic)

        x = nn.LayerNorm(epsilon=config.layer_norm_eps, name="encoder_norm")(x)
        return x
