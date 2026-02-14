#!/usr/bin/env python3
"""
DINO (Self-Distillation with No Labels) using HuggingFace ViT backbone.

Same DINO architecture as the timm version, but using HuggingFace's
ViTModel as the backbone. Provides the same interface.

Implements Caron et al. (2021).
"""

from typing import Optional, List
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..vit_config import ViTConfig
from .vit import build_vit, HFViTWrapper


class DINOHead(nn.Module):
    """DINO projection head (same as timm version)."""

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

        self.last_layer = nn.utils.weight_norm(
            nn.Linear(bottleneck_dim, out_dim, bias=False)
        )

        if norm_last_layer:
            self.last_layer.weight_g.data.fill_(1)
            self.last_layer.weight_g.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x


class DINO(nn.Module):
    """
    DINO self-distillation with HuggingFace ViT backbone.

    Same architecture and interface as the timm DINO, but using
    HuggingFace's ViTModel internally.
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

        # Student
        self.student_backbone = build_vit(config)
        self.student_head = DINOHead(
            in_dim=config.embed_dim,
            hidden_dim=hidden_dim,
            bottleneck_dim=bottleneck_dim,
            out_dim=out_dim,
        )

        # Teacher (deepcopy, frozen)
        self.teacher_backbone = copy.deepcopy(self.student_backbone)
        self.teacher_head = DINOHead(
            in_dim=config.embed_dim,
            hidden_dim=hidden_dim,
            bottleneck_dim=bottleneck_dim,
            out_dim=out_dim,
            norm_last_layer=False,
        )

        self._init_teacher()

        for p in self.teacher_backbone.parameters():
            p.requires_grad = False
        for p in self.teacher_head.parameters():
            p.requires_grad = False

        self.register_buffer("center", torch.zeros(1, out_dim))

    def _init_teacher(self):
        for t_p, s_p in zip(
            self.teacher_backbone.parameters(), self.student_backbone.parameters()
        ):
            t_p.data.copy_(s_p.data)
        for t_p, s_p in zip(
            self.teacher_head.parameters(), self.student_head.parameters()
        ):
            t_p.data.copy_(s_p.data)

    @torch.no_grad()
    def update_teacher(self, momentum: Optional[float] = None):
        m = momentum if momentum is not None else self.momentum_teacher
        for t_p, s_p in zip(
            self.teacher_backbone.parameters(), self.student_backbone.parameters()
        ):
            t_p.data.mul_(m).add_(s_p.data, alpha=1 - m)
        for t_p, s_p in zip(
            self.teacher_head.parameters(), self.student_head.parameters()
        ):
            t_p.data.mul_(m).add_(s_p.data, alpha=1 - m)

    @torch.no_grad()
    def update_center(self, teacher_output: torch.Tensor):
        batch_center = teacher_output.mean(dim=0, keepdim=True)
        self.center = (
            self.center * self.center_momentum
            + batch_center * (1 - self.center_momentum)
        )

    def forward_student(self, x: torch.Tensor) -> torch.Tensor:
        features = self.student_backbone.get_cls_token(x)
        logits = self.student_head(features)
        return logits / self.student_temp

    @torch.no_grad()
    def forward_teacher(self, x: torch.Tensor) -> torch.Tensor:
        features = self.teacher_backbone.get_cls_token(x)
        logits = self.teacher_head(features)
        return (logits - self.center) / self.teacher_temp

    def dino_loss(
        self,
        student_outputs: List[torch.Tensor],
        teacher_outputs: List[torch.Tensor],
    ) -> torch.Tensor:
        total_loss = 0
        n_terms = 0

        for t_idx, t_out in enumerate(teacher_outputs):
            t_probs = F.softmax(t_out, dim=-1)
            for s_idx, s_out in enumerate(student_outputs):
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
        teacher_outputs = [self.forward_teacher(gc) for gc in global_crops]
        all_teacher = torch.cat(teacher_outputs, dim=0)
        self.update_center(all_teacher)

        all_crops = global_crops + local_crops
        student_outputs = [self.forward_student(crop) for crop in all_crops]

        loss = self.dino_loss(student_outputs, teacher_outputs)

        return {
            "loss": loss,
            "student_outputs": student_outputs,
            "teacher_outputs": teacher_outputs,
        }

    def get_encoder(self) -> HFViTWrapper:
        return self.student_backbone
