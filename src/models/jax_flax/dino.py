#!/usr/bin/env python3
"""
DINO (Self-Distillation with No Labels) using JAX/Flax.

Implements Caron et al. (2021) DINO in pure JAX/Flax:
- Student-teacher ViT with EMA teacher updates
- Multi-crop strategy with cross-entropy loss
- Centering to avoid representation collapse
- Uses jax.lax.stop_gradient for teacher

Adapted for single-channel FITS galaxy thumbnails.
"""

from typing import Optional, List, Tuple, Any, Dict
from functools import partial

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from src.models.vit_config import ViTConfig
from src.models.jax_flax.vit import VisionTransformer


class DINOHead(nn.Module):
    """
    DINO projection head in Flax.

    MLP → L2-normalize → prototype linear layer.

    Attributes:
        hidden_dim: MLP hidden dimension.
        bottleneck_dim: Bottleneck before prototypes.
        out_dim: Number of output prototypes.
    """

    hidden_dim: int = 2048
    bottleneck_dim: int = 256
    out_dim: int = 65536

    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        """
        Project CLS token to prototype space.

        Args:
            x: (B, embed_dim) CLS token.
            deterministic: Unused (no dropout in head).

        Returns:
            (B, out_dim) prototype logits.
        """
        # 3-layer MLP
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.LayerNorm()(x)  # Use LayerNorm instead of BatchNorm (JAX-friendly)
        x = nn.gelu(x)

        x = nn.Dense(self.hidden_dim)(x)
        x = nn.LayerNorm()(x)
        x = nn.gelu(x)

        x = nn.Dense(self.bottleneck_dim)(x)

        # L2 normalize
        x = x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + 1e-6)

        # Final prototype projection
        x = nn.Dense(self.out_dim, use_bias=False, name="prototypes")(x)

        return x


class DINOStudent(nn.Module):
    """
    DINO student network: ViT backbone + projection head.

    Attributes:
        config: ViT configuration.
        head_hidden_dim: Projection head hidden dim.
        head_bottleneck_dim: Projection head bottleneck dim.
        head_out_dim: Number of prototypes.
    """

    config: ViTConfig
    head_hidden_dim: int = 2048
    head_bottleneck_dim: int = 256
    head_out_dim: int = 65536

    def setup(self):
        self.backbone = VisionTransformer(config=self.config)
        self.head = DINOHead(
            hidden_dim=self.head_hidden_dim,
            bottleneck_dim=self.head_bottleneck_dim,
            out_dim=self.head_out_dim,
        )

    def __call__(
        self, x: jnp.ndarray, deterministic: bool = True
    ) -> jnp.ndarray:
        """
        Forward pass through backbone + head.

        Args:
            x: (B, 1, H, W) input image.
            deterministic: Disable dropout in backbone.

        Returns:
            (B, out_dim) prototype logits.
        """
        # Get CLS token from backbone
        features = self.backbone(x, deterministic=deterministic)
        cls_token = features[:, 0, :]  # (B, embed_dim)

        # Project to prototype space
        logits = self.head(cls_token, deterministic=deterministic)
        return logits

    def get_cls_features(
        self, x: jnp.ndarray, deterministic: bool = True
    ) -> jnp.ndarray:
        """Get CLS token features before projection head."""
        features = self.backbone(x, deterministic=deterministic)
        return features[:, 0, :]

    def encode(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        """Alias for get_cls_features — returns (B, embed_dim) CLS embeddings."""
        return self.get_cls_features(x, deterministic=deterministic)


def compute_dino_loss(
    student_outputs: List[jnp.ndarray],
    teacher_outputs: List[jnp.ndarray],
    student_temp: float = 0.1,
    teacher_temp: float = 0.04,
    center: jnp.ndarray = None,
) -> jnp.ndarray:
    """
    Compute DINO cross-entropy loss.

    Args:
        student_outputs: List of student logits for all crops.
        teacher_outputs: List of teacher logits for global crops.
        student_temp: Student softmax temperature.
        teacher_temp: Teacher softmax temperature.
        center: Center vector for teacher centering.

    Returns:
        Scalar loss.
    """
    total_loss = jnp.float32(0.0)
    n_terms = 0

    for t_idx, t_out in enumerate(teacher_outputs):
        # Center and sharpen teacher
        if center is not None:
            t_out = t_out - center
        t_probs = jax.nn.softmax(t_out / teacher_temp, axis=-1)
        # Clamp teacher probs to avoid NaN from extreme logits
        t_probs = jnp.clip(t_probs, a_min=1e-7)

        for s_idx, s_out in enumerate(student_outputs):
            if s_idx == t_idx:
                continue

            s_log_probs = jax.nn.log_softmax(s_out / student_temp, axis=-1)
            loss = -jnp.sum(t_probs * s_log_probs, axis=-1).mean()
            total_loss = total_loss + loss
            n_terms += 1

    return total_loss / n_terms


def ema_update_params(
    teacher_params: Any, student_params: Any, momentum: float
) -> Any:
    """
    EMA update of teacher parameters from student.

    teacher = momentum * teacher + (1 - momentum) * student

    Works on pytree structures (Flax parameter dicts).
    """
    return jax.tree.map(
        lambda t, s: momentum * t + (1 - momentum) * s,
        teacher_params,
        student_params,
    )


def update_center(
    center: jnp.ndarray,
    teacher_output: jnp.ndarray,
    center_momentum: float = 0.9,
) -> jnp.ndarray:
    """
    EMA update of center vector.

    center = momentum * center + (1 - momentum) * mean(teacher_output)
    """
    batch_center = teacher_output.mean(axis=0, keepdims=True)
    return center_momentum * center + (1 - center_momentum) * batch_center


class DINO:
    """
    DINO wrapper for JAX/Flax training.

    This is not an nn.Module but a stateful training wrapper that manages
    student/teacher parameters, center, and momentum schedule.

    Usage:
        dino = DINO(config)
        student_params, teacher_params = dino.init(rng, dummy_input)
        # In training loop:
        loss, student_params, teacher_params, center = dino.train_step(...)

    Args:
        config: ViTConfig for backbone.
        out_dim: Prototype dimension.
        hidden_dim: Head hidden dimension.
        bottleneck_dim: Head bottleneck dimension.
        momentum_teacher: Initial EMA momentum.
        teacher_temp: Teacher softmax temperature.
        student_temp: Student softmax temperature.
        center_momentum: Center EMA momentum.
    """

    def __init__(
        self,
        config: ViTConfig,
        out_dim: int = 65536,
        hidden_dim: int = 2048,
        bottleneck_dim: int = 256,
        momentum_teacher: float = 0.996,
        teacher_temp: float = 0.04,
        student_temp: float = 0.1,
        center_momentum: float = 0.9,
    ):
        self.config = config
        self.out_dim = out_dim
        self.momentum_teacher = momentum_teacher
        self.teacher_temp = teacher_temp
        self.student_temp = student_temp
        self.center_momentum = center_momentum

        # Create student/teacher models (same architecture)
        self.student = DINOStudent(
            config=config,
            head_hidden_dim=hidden_dim,
            head_bottleneck_dim=bottleneck_dim,
            head_out_dim=out_dim,
        )
        self.teacher = DINOStudent(
            config=config,
            head_hidden_dim=hidden_dim,
            head_bottleneck_dim=bottleneck_dim,
            head_out_dim=out_dim,
        )

    def init(
        self, rng: jax.random.PRNGKey, dummy_input: jnp.ndarray
    ) -> Tuple[Any, Any, jnp.ndarray]:
        """
        Initialize student and teacher parameters.

        Args:
            rng: PRNG key.
            dummy_input: Sample input for shape inference.

        Returns:
            (student_params, teacher_params, center)
        """
        rng_s, rng_t = jax.random.split(rng)

        student_params = self.student.init(rng_s, dummy_input)
        # Teacher starts as a copy of student
        teacher_params = jax.tree.map(jnp.copy, student_params)

        center = jnp.zeros((1, self.out_dim))

        return student_params, teacher_params, center

    def loss_fn(
        self,
        student_params: Any,
        teacher_params: Any,
        global_crops: List[jnp.ndarray],
        local_crops: List[jnp.ndarray],
        center: jnp.ndarray,
        deterministic: bool = False,
        rng: Optional[jax.random.PRNGKey] = None,
    ) -> Tuple[jnp.ndarray, Dict]:
        """
        Compute DINO loss for a batch.

        Args:
            student_params: Student model parameters.
            teacher_params: Teacher model parameters (stop_gradient applied).
            global_crops: List of 2 global crop tensors.
            local_crops: List of N local crop tensors.
            center: Current center vector.
            deterministic: Disable dropout.

        Returns:
            (loss, aux_dict) where aux_dict contains teacher outputs for center update.
        """
        # Teacher forward (global crops only, stopped gradient)
        teacher_outputs = []
        for gc in global_crops:
            t_out = self.teacher.apply(
                jax.lax.stop_gradient(teacher_params),
                gc,
                deterministic=True,
            )
            teacher_outputs.append(t_out)

        # Student forward (all crops) — pass dropout RNG when not deterministic
        all_crops = global_crops + local_crops
        student_outputs = []
        for i, crop in enumerate(all_crops):
            if deterministic:
                s_out = self.student.apply(student_params, crop, deterministic=True)
            else:
                # Need dropout RNG key for stochastic forward pass
                s_out = self.student.apply(
                    student_params, crop, deterministic=False,
                    rngs={'dropout': jax.random.fold_in(rng, i)} if rng is not None else {},
                )
            student_outputs.append(s_out)

        # Compute loss
        loss = compute_dino_loss(
            student_outputs=student_outputs,
            teacher_outputs=teacher_outputs,
            student_temp=self.student_temp,
            teacher_temp=self.teacher_temp,
            center=center,
        )

        # Concatenate teacher outputs for center update
        all_teacher = jnp.concatenate(teacher_outputs, axis=0)

        return loss, {"teacher_output": all_teacher}
