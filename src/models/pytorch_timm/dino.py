#!/usr/bin/env python3
"""
DINO (Self-Distillation with No Labels) using timm ViT backbone.

Implements Caron et al. (2021) "Emerging Properties in Self-Supervised
Vision Transformers":
- Student-teacher setup with EMA teacher
- Multi-crop strategy (2 global + N local crops)
- Cross-entropy loss between teacher (global) and student (all) outputs
- Centering + sharpening to avoid representation collapse

Adapted for single-channel FITS galaxy thumbnails.
"""

from typing import Optional, List
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..vit_config import ViTConfig
from .vit import build_vit, TimmViTWrapper


class DINOHead(nn.Module):
    """
    DINO projection head: MLP → L2-normalize → prototype layer.

    Maps ViT CLS token to a lower-dimensional space where the
    self-distillation loss is computed.

    Args:
        in_dim: Input dimension (ViT embed_dim).
        hidden_dim: MLP hidden dimension.
        bottleneck_dim: Bottleneck before final projection.
        out_dim: Output dimension (number of prototypes).
        use_bn: Use BatchNorm in MLP layers.
        norm_last_layer: Whether to normalize the last layer weights.
    """

    def __init__(
        self,
        in_dim: int = 192,
        hidden_dim: int = 2048,
        bottleneck_dim: int = 256,
        out_dim: int = 65536,
        use_bn: bool = True,
        norm_last_layer: bool = True,
    ):
        super().__init__()

        # 3-layer MLP
        layers = []
        layers.append(nn.Linear(in_dim, hidden_dim))
        if use_bn:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.GELU())

        layers.append(nn.Linear(hidden_dim, hidden_dim))
        if use_bn:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.GELU())

        layers.append(nn.Linear(hidden_dim, bottleneck_dim))

        self.mlp = nn.Sequential(*layers)

        # L2 normalization applied after MLP
        # Final prototype layer (no bias)
        self.last_layer = nn.utils.weight_norm(
            nn.Linear(bottleneck_dim, out_dim, bias=False)
        )

        if norm_last_layer:
            self.last_layer.weight_g.data.fill_(1)
            self.last_layer.weight_g.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project CLS token to prototype space.

        Args:
            x: (B, in_dim) CLS token embeddings.

        Returns:
            (B, out_dim) prototype logits.
        """
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x


class DINO(nn.Module):
    """
    DINO self-distillation framework with timm ViT backbone.

    Creates student and teacher networks. Teacher is an EMA copy of
    the student. Only the student receives gradient updates.

    Args:
        config: ViTConfig for the backbone.
        out_dim: Projection head output dimension.
        hidden_dim: Projection head hidden dimension.
        bottleneck_dim: Projection head bottleneck dimension.
        momentum_teacher: Initial EMA momentum (increases to 1.0).
        teacher_temp: Teacher softmax temperature.
        student_temp: Student softmax temperature.
        center_momentum: Momentum for center vector EMA.
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
        super().__init__()

        self.config = config
        self.momentum_teacher = momentum_teacher
        self.teacher_temp = teacher_temp
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.out_dim = out_dim

        # Student network: backbone + projection head
        self.student_backbone = build_vit(config)
        self.student_head = DINOHead(
            in_dim=config.embed_dim,
            hidden_dim=hidden_dim,
            bottleneck_dim=bottleneck_dim,
            out_dim=out_dim,
        )

        # Teacher network: deepcopy of student (no gradients)
        self.teacher_backbone = copy.deepcopy(self.student_backbone)
        self.teacher_head = DINOHead(
            in_dim=config.embed_dim,
            hidden_dim=hidden_dim,
            bottleneck_dim=bottleneck_dim,
            out_dim=out_dim,
            norm_last_layer=False,
        )

        # Copy student weights to teacher
        self._init_teacher()

        # Disable teacher gradients
        for p in self.teacher_backbone.parameters():
            p.requires_grad = False
        for p in self.teacher_head.parameters():
            p.requires_grad = False

        # Center vector for teacher outputs (avoids collapse)
        self.register_buffer("center", torch.zeros(1, out_dim))

    def _init_teacher(self):
        """Initialize teacher with student weights."""
        for t_param, s_param in zip(
            self.teacher_backbone.parameters(), self.student_backbone.parameters()
        ):
            t_param.data.copy_(s_param.data)

        for t_param, s_param in zip(
            self.teacher_head.parameters(), self.student_head.parameters()
        ):
            t_param.data.copy_(s_param.data)

    @torch.no_grad()
    def update_teacher(self, momentum: Optional[float] = None):
        """
        EMA update of teacher network.

        teacher_param = momentum * teacher_param + (1 - momentum) * student_param

        Args:
            momentum: Override momentum value (for schedule).
        """
        m = momentum if momentum is not None else self.momentum_teacher

        for t_param, s_param in zip(
            self.teacher_backbone.parameters(), self.student_backbone.parameters()
        ):
            t_param.data.mul_(m).add_(s_param.data, alpha=1 - m)

        for t_param, s_param in zip(
            self.teacher_head.parameters(), self.student_head.parameters()
        ):
            t_param.data.mul_(m).add_(s_param.data, alpha=1 - m)

    @torch.no_grad()
    def update_center(self, teacher_output: torch.Tensor):
        """
        EMA update of center vector from teacher outputs.

        center = momentum * center + (1 - momentum) * mean(teacher_output)
        """
        batch_center = teacher_output.mean(dim=0, keepdim=True)
        self.center = (
            self.center * self.center_momentum
            + batch_center * (1 - self.center_momentum)
        )

    def forward_student(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through student network.

        Args:
            x: (B, 1, H, W) input image.

        Returns:
            (B, out_dim) softmax-tempered logits.
        """
        features = self.student_backbone.get_cls_token(x)
        logits = self.student_head(features)
        return logits / self.student_temp

    @torch.no_grad()
    def forward_teacher(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through teacher network (no gradient).

        Applies centering and sharpening.

        Args:
            x: (B, 1, H, W) input image.

        Returns:
            (B, out_dim) centered, sharpened softmax logits.
        """
        features = self.teacher_backbone.get_cls_token(x)
        logits = self.teacher_head(features)
        # Center and sharpen
        logits = (logits - self.center) / self.teacher_temp
        return logits

    def dino_loss(
        self,
        student_outputs: List[torch.Tensor],
        teacher_outputs: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute DINO cross-entropy loss.

        For each teacher (global) output, compute cross-entropy against
        each student output (excluding same view).

        Args:
            student_outputs: List of student logits for all crops.
            teacher_outputs: List of teacher logits for global crops only.

        Returns:
            Scalar loss.
        """
        total_loss = 0
        n_terms = 0

        for t_idx, t_out in enumerate(teacher_outputs):
            t_probs = F.softmax(t_out, dim=-1)

            for s_idx, s_out in enumerate(student_outputs):
                # Skip same view (first 2 are global, matching teacher indices)
                if s_idx == t_idx:
                    continue

                s_log_probs = F.log_softmax(s_out, dim=-1)
                loss = -torch.sum(t_probs * s_log_probs, dim=-1).mean()
                total_loss += loss
                n_terms += 1

        return total_loss / n_terms

    def forward(
        self,
        global_crops: List[torch.Tensor],
        local_crops: List[torch.Tensor],
    ) -> dict:
        """
        Full DINO forward pass with multi-crop.

        Args:
            global_crops: List of 2 tensors, each (B, 1, gs, gs).
            local_crops: List of N tensors, each (B, 1, ls, ls).

        Returns:
            Dict with 'loss', 'student_outputs', 'teacher_outputs'.
        """
        # Teacher: only global crops (no gradient)
        teacher_outputs = [self.forward_teacher(gc) for gc in global_crops]

        # Update center with teacher outputs
        all_teacher = torch.cat(teacher_outputs, dim=0)
        self.update_center(all_teacher)

        # Student: all crops (global + local)
        all_crops = global_crops + local_crops
        student_outputs = [self.forward_student(crop) for crop in all_crops]

        # Compute loss
        loss = self.dino_loss(student_outputs, teacher_outputs)

        return {
            "loss": loss,
            "student_outputs": student_outputs,
            "teacher_outputs": teacher_outputs,
        }

    def get_encoder(self) -> TimmViTWrapper:
        """Return the student backbone for downstream use."""
        return self.student_backbone
