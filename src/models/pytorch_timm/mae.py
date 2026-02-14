#!/usr/bin/env python3
"""
Masked Autoencoder (MAE) using timm ViT backbone.

Implements He et al. (2022) "Masked Autoencoders Are Scalable Vision Learners":
- Encoder operates on visible patches only (75% masking ratio)
- Lightweight decoder reconstructs masked patches from latent + mask tokens
- Loss: MSE on pixel values of masked patches (per-patch normalized)

Adapted for single-channel FITS galaxy thumbnails.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from ..vit_config import ViTConfig
from .vit import build_vit, TimmViTWrapper


class MAEDecoder(nn.Module):
    """
    Lightweight transformer decoder for MAE reconstruction.

    Takes encoder output (visible token embeddings) + mask tokens,
    adds positional embeddings, processes through transformer blocks,
    and projects to pixel space for reconstruction.

    Args:
        encoder_embed_dim: Embedding dim from encoder.
        decoder_embed_dim: Decoder's internal embedding dim.
        decoder_depth: Number of decoder transformer layers.
        decoder_num_heads: Number of attention heads in decoder.
        num_patches: Total number of patches in the image.
        patch_size: Spatial size of each patch.
        in_channels: Number of input channels (1 for FITS).
    """

    def __init__(
        self,
        encoder_embed_dim: int = 192,
        decoder_embed_dim: int = 128,
        decoder_depth: int = 4,
        decoder_num_heads: int = 4,
        num_patches: int = 64,
        patch_size: int = 8,
        in_channels: int = 1,
    ):
        super().__init__()

        self.num_patches = num_patches
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.decoder_embed_dim = decoder_embed_dim

        # Project encoder output to decoder dimension
        self.decoder_embed = nn.Linear(encoder_embed_dim, decoder_embed_dim)

        # Learnable mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        nn.init.normal_(self.mask_token, std=0.02)

        # Positional embedding for decoder (full sequence: CLS + all patches)
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, decoder_embed_dim)
        )
        nn.init.normal_(self.decoder_pos_embed, std=0.02)

        # Decoder transformer blocks
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=decoder_embed_dim,
            nhead=decoder_num_heads,
            dim_feedforward=decoder_embed_dim * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder_blocks = nn.TransformerEncoder(
            decoder_layer, num_layers=decoder_depth
        )

        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)

        # Project to pixel predictions
        self.decoder_pred = nn.Linear(
            decoder_embed_dim, patch_size * patch_size * in_channels
        )

    def forward(
        self, x: torch.Tensor, ids_restore: torch.Tensor
    ) -> torch.Tensor:
        """
        Decode from visible+mask tokens to pixel predictions.

        Args:
            x: Encoder output of shape (B, N_visible + 1, encoder_embed_dim).
               Includes CLS token at position 0.
            ids_restore: Indices to unshuffle tokens back to original order.
                         Shape: (B, num_patches).

        Returns:
            Pixel predictions of shape (B, num_patches, patch_size^2 * in_channels).
        """
        # Embed encoder output
        x = self.decoder_embed(x)

        # Append mask tokens and restore original ordering
        mask_tokens = self.mask_token.expand(
            x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], -1
        )

        # Remove CLS, concat mask tokens, restore ordering
        x_no_cls = x[:, 1:, :]
        x_combined = torch.cat([x_no_cls, mask_tokens], dim=1)

        # Unshuffle: ids_restore maps from shuffled to original order
        x_combined = torch.gather(
            x_combined,
            dim=1,
            index=ids_restore.unsqueeze(-1).expand(-1, -1, x.shape[2]),
        )

        # Re-add CLS token
        x = torch.cat([x[:, :1, :], x_combined], dim=1)

        # Add positional embedding
        x = x + self.decoder_pos_embed

        # Decoder transformer blocks
        x = self.decoder_blocks(x)
        x = self.decoder_norm(x)

        # Project to pixel values (remove CLS)
        x = self.decoder_pred(x[:, 1:, :])

        return x


class MAE(nn.Module):
    """
    Masked Autoencoder (MAE) with timm ViT encoder.

    SSL pretraining: randomly masks patches, encodes visible patches,
    decodes to reconstruct masked patches. Loss = MSE on masked patches.

    Args:
        config: ViTConfig for the encoder backbone.
        mask_ratio: Fraction of patches to mask (default: 0.75).
        decoder_embed_dim: Decoder embedding dimension.
        decoder_depth: Number of decoder transformer layers.
        decoder_num_heads: Number of decoder attention heads.
        norm_pix_loss: If True, normalize target pixels per-patch.
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
        self.patch_size = config.patch_size
        self.in_channels = config.in_channels

        # Encoder (timm ViT)
        self.encoder = build_vit(config)

        # Decoder
        self.decoder = MAEDecoder(
            encoder_embed_dim=config.embed_dim,
            decoder_embed_dim=decoder_embed_dim,
            decoder_depth=decoder_depth,
            decoder_num_heads=decoder_num_heads,
            num_patches=config.num_patches,
            patch_size=config.patch_size,
            in_channels=config.in_channels,
        )

    def patchify(self, images: torch.Tensor) -> torch.Tensor:
        """
        Convert images to patch sequences.

        Args:
            images: (B, C, H, W)

        Returns:
            (B, num_patches, patch_size^2 * C)
        """
        p = self.patch_size
        c = self.in_channels
        h = w = self.config.image_size // p

        x = rearrange(
            images, "b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=p, p2=p, h=h, w=w
        )
        return x

    def unpatchify(self, patches: torch.Tensor) -> torch.Tensor:
        """
        Convert patch sequences back to images.

        Args:
            patches: (B, num_patches, patch_size^2 * C)

        Returns:
            (B, C, H, W)
        """
        p = self.patch_size
        c = self.in_channels
        h = w = self.config.grid_size

        x = rearrange(
            patches, "b (h w) (p1 p2 c) -> b c (h p1) (w p2)", p1=p, p2=p, h=h, w=w, c=c
        )
        return x

    def random_masking(
        self, x: torch.Tensor, mask_ratio: float
    ) -> tuple:
        """
        Perform random masking by shuffling and keeping a subset of patches.

        Args:
            x: (B, N, D) patch embeddings.
            mask_ratio: Fraction of patches to mask.

        Returns:
            x_masked: (B, N_visible, D) visible patch embeddings.
            mask: (B, N) binary mask (1 = masked, 0 = visible).
            ids_restore: (B, N) indices to unshuffle back to original order.
        """
        B, N, D = x.shape
        num_keep = int(N * (1 - mask_ratio))

        # Random noise for shuffling
        noise = torch.rand(B, N, device=x.device)

        # Sort noise: ascending = indices from least to most noisy
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # Keep first num_keep patches (least noisy)
        ids_keep = ids_shuffle[:, :num_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D))

        # Generate binary mask: 1 = masked, 0 = visible
        mask = torch.ones(B, N, device=x.device)
        mask[:, :num_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(
        self, x: torch.Tensor, mask_ratio: float
    ) -> tuple:
        """
        Encode visible patches only.

        Args:
            x: (B, 1, H, W) input images.
            mask_ratio: Fraction to mask.

        Returns:
            latent: (B, N_visible + 1, embed_dim) — visible patches + CLS.
            mask: (B, num_patches) — binary mask.
            ids_restore: (B, num_patches) — unshuffle indices.
        """
        vit = self.encoder.vit

        # Patch embed
        x = vit.patch_embed(x)  # (B, num_patches, embed_dim)

        # Mask
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # Add CLS token + positional embedding for visible patches
        cls_token = vit.cls_token.expand(x.shape[0], -1, -1)

        # We need position embeddings for only the visible patches
        # Extract CLS position embedding and visible patch position embeddings
        cls_pos = vit.pos_embed[:, :1, :]  # CLS position
        x = x + torch.gather(
            vit.pos_embed[:, 1:, :].expand(x.shape[0], -1, -1),
            dim=1,
            index=(ids_restore[:, : x.shape[1]]).unsqueeze(-1).expand(-1, -1, vit.pos_embed.shape[2]),
        )

        # Wait — we need the position embeddings for the *kept* patches, not restored.
        # Let me redo this properly.
        # ids_shuffle was used to select patches. The kept patches are ids_shuffle[:, :num_keep]
        # But we only have ids_restore. We need to get position embeddings for the visible patches.

        # The encoding logic with masking is implemented in forward().
        # Use forward() for training and get_encoder() for inference.
        raise NotImplementedError(
            "Use forward() for masked encoding or get_encoder() for full encoding."
        )

    def forward(
        self, images: torch.Tensor, mask_ratio: Optional[float] = None
    ) -> dict:
        """
        Full MAE forward pass: encode visible → decode → compute loss.

        Args:
            images: (B, 1, H, W) input images.
            mask_ratio: Override mask ratio (uses self.mask_ratio if None).

        Returns:
            Dict with keys: 'loss', 'pred', 'mask'.
        """
        if mask_ratio is None:
            mask_ratio = self.mask_ratio

        vit = self.encoder.vit
        B = images.shape[0]

        # --- ENCODER ---
        # Step 1: Patch embedding
        x = vit.patch_embed(images)  # (B, N, embed_dim)
        N = x.shape[1]

        # Step 2: Add positional embedding (before masking)
        pos_embed_patches = vit.pos_embed[:, 1:, :]  # Skip CLS pos embed
        x = x + pos_embed_patches

        # Step 3: Random masking
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # Step 4: Prepend CLS token (with its positional embedding)
        cls_token = vit.cls_token + vit.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Step 5: Apply dropout and transformer blocks
        x = vit.pos_drop(x)
        for blk in vit.blocks:
            x = blk(x)
        x = vit.norm(x)

        # x: (B, 1 + N_visible, embed_dim)

        # --- DECODER ---
        pred = self.decoder(x, ids_restore)  # (B, N, patch_size^2 * C)

        # --- LOSS ---
        target = self.patchify(images)  # (B, N, patch_size^2 * C)

        if self.norm_pix_loss:
            # Normalize target per-patch (mean=0, var=1)
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1e-6).sqrt()

        # MSE loss only on masked patches
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # (B, N) — per-patch loss
        loss = (loss * mask).sum() / mask.sum()  # average over masked patches

        return {
            "loss": loss,
            "pred": pred,
            "mask": mask,
        }

    def get_encoder(self) -> TimmViTWrapper:
        """Return the encoder backbone for downstream use."""
        return self.encoder
