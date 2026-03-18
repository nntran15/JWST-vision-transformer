#!/usr/bin/env python3
"""
DINO Trainer: Training loop for DINO self-distillation pretraining.

Supports PyTorch (timm, HuggingFace) and JAX/Flax with:
- Multi-crop forward pass (teacher: global, student: all)
- EMA teacher update with momentum schedule
- Teacher temperature warmup
- Centering + sharpening to prevent collapse
- Cosine LR schedule with warmup
"""

import logging
import math
import time
from typing import Dict, Any, Optional, List

import numpy as np

logger = logging.getLogger(__name__)


class DINOTrainer:
    """
    Training loop for DINO SSL pretraining.

    Args:
        config: Full experiment configuration dict.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.framework = config["ssl"]["framework"]
        self.epochs = config["training"]["epochs"]
        self.lr = config["training"]["learning_rate"]
        self.min_lr = config["training"]["min_learning_rate"]
        self.warmup_epochs = config["training"]["warmup_epochs"]
        self.weight_decay = config["training"]["weight_decay"]
        self.mixed_precision = config["training"]["mixed_precision"]
        self.gradient_clip = config["training"]["gradient_clip_norm"]
        self.log_every = config["logging"]["log_every_n_steps"]
        self.save_every = config["logging"]["save_every_n_epochs"]

        # DINO-specific
        dino_cfg = config.get("dino", {})
        self.momentum_start = dino_cfg.get("momentum_teacher_start", 0.996)
        self.momentum_end = dino_cfg.get("momentum_teacher_end", 1.0)
        self.teacher_temp_warmup = dino_cfg.get("teacher_temp_warmup", 0.04)
        self.teacher_temp_final = dino_cfg.get("teacher_temp_final", 0.07)
        self.teacher_temp_warmup_epochs = dino_cfg.get("teacher_temp_warmup_epochs", 30)

    def get_lr(self, epoch: float) -> float:
        """Cosine schedule with linear warmup.

        Args:
            epoch: Fractional epoch (e.g., 0.5 = halfway through epoch 0).
                   Using fractional epochs ensures LR > 0 from the very first step.
        """
        if epoch < self.warmup_epochs:
            return self.lr * epoch / max(self.warmup_epochs, 1)
        progress = (epoch - self.warmup_epochs) / max(
            self.epochs - self.warmup_epochs, 1
        )
        return self.min_lr + 0.5 * (self.lr - self.min_lr) * (
            1 + math.cos(math.pi * progress)
        )

    def get_momentum(self, epoch: int) -> float:
        """Cosine schedule for teacher EMA momentum."""
        progress = epoch / max(self.epochs, 1)
        return self.momentum_end - (self.momentum_end - self.momentum_start) * (
            1 + math.cos(math.pi * progress)
        ) / 2

    def get_teacher_temp(self, epoch: int) -> float:
        """Linear warmup for teacher temperature."""
        if epoch < self.teacher_temp_warmup_epochs:
            return (
                self.teacher_temp_warmup
                + (self.teacher_temp_final - self.teacher_temp_warmup)
                * epoch
                / max(self.teacher_temp_warmup_epochs, 1)
            )
        return self.teacher_temp_final

    # ---- PyTorch Training ----

    def train_pytorch(
        self,
        model,
        dataloader,
        device,
        checkpoint_manager=None,
        wandb_logger=None,
        start_epoch: int = 0,
    ):
        """
        DINO training loop for PyTorch (timm or HuggingFace).

        Args:
            model: DINO nn.Module.
            dataloader: PyTorch DataLoader with DINO multi-crop collation.
            device: torch.device.
            checkpoint_manager: CheckpointManager instance.
            wandb_logger: WandbLogger instance.
            start_epoch: Resume epoch.
        """
        import torch
        from tqdm import tqdm

        model = model.to(device)
        model.train()

        # Only optimize student parameters
        student_params = [
            p for n, p in model.named_parameters()
            if p.requires_grad and "teacher" not in n
        ]

        param_groups = [
            {
                "params": [
                    p for p in student_params
                    if p.dim() >= 2
                ],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [
                    p for p in student_params
                    if p.dim() < 2
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(param_groups, lr=self.lr, betas=(0.9, 0.999))

        scaler = torch.cuda.amp.GradScaler() if self.mixed_precision else None
        global_step = 0

        for epoch in range(start_epoch, self.epochs):
            momentum = self.get_momentum(epoch)
            teacher_temp = self.get_teacher_temp(epoch)
            model.teacher_temp = teacher_temp

            epoch_loss = 0.0
            n_batches = 0
            n_total_batches = len(dataloader) if hasattr(dataloader, '__len__') else None

            pbar = tqdm(
                dataloader,
                desc=f"DINO Epoch {epoch + 1}/{self.epochs}",
                disable=not logger.isEnabledFor(logging.INFO),
            )

            for batch_idx, (global_crops, local_crops) in enumerate(pbar):
                # Fractional epoch for sub-epoch LR warmup (avoids lr=0 at epoch 0)
                frac_epoch = epoch + (batch_idx / n_total_batches if n_total_batches else 0)
                lr = self.get_lr(frac_epoch)
                for pg in optimizer.param_groups:
                    pg["lr"] = lr
                # Move to device
                global_crops = [gc.to(device, non_blocking=True) for gc in global_crops]
                local_crops = [lc.to(device, non_blocking=True) for lc in local_crops]

                optimizer.zero_grad(set_to_none=True)

                if self.mixed_precision:
                    with torch.cuda.amp.autocast():
                        outputs = model(global_crops, local_crops)
                        loss = outputs["loss"]

                    # Check for NaN BEFORE applying gradients to prevent weight corruption
                    if torch.isnan(loss) or torch.isinf(loss):
                        logger.warning(
                            f"NaN/Inf loss at step {global_step}, epoch {epoch + 1}. "
                            f"Skipping optimizer + teacher update."
                        )
                        n_batches += 1
                        global_step += 1
                        continue

                    scaler.scale(loss).backward()

                    if self.gradient_clip > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            student_params, self.gradient_clip
                        )

                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = model(global_crops, local_crops)
                    loss = outputs["loss"]

                    # Check for NaN BEFORE applying gradients to prevent weight corruption
                    if torch.isnan(loss) or torch.isinf(loss):
                        logger.warning(
                            f"NaN/Inf loss at step {global_step}, epoch {epoch + 1}. "
                            f"Skipping optimizer + teacher update."
                        )
                        n_batches += 1
                        global_step += 1
                        continue

                    loss.backward()

                    if self.gradient_clip > 0:
                        torch.nn.utils.clip_grad_norm_(
                            student_params, self.gradient_clip
                        )

                    optimizer.step()

                # EMA update teacher
                model.update_teacher(momentum=momentum)

                epoch_loss += loss.item()
                n_batches += 1
                global_step += 1

                if global_step % self.log_every == 0:
                    avg_loss = epoch_loss / n_batches
                    pbar.set_postfix(
                        loss=f"{loss.item():.4f}",
                        avg=f"{avg_loss:.4f}",
                        lr=f"{lr:.2e}",
                        mom=f"{momentum:.4f}",
                        t_temp=f"{teacher_temp:.3f}",
                    )

                    if wandb_logger:
                        wandb_logger.log_metrics(
                            {
                                "loss": loss.item(),
                                "lr": lr,
                                "momentum": momentum,
                                "teacher_temp": teacher_temp,
                            },
                            step=global_step,
                            prefix="train/",
                        )

            avg_loss = epoch_loss / max(n_batches, 1)
            logger.info(
                f"DINO Epoch {epoch + 1}: avg_loss={avg_loss:.6f}, "
                f"lr={lr:.2e}, momentum={momentum:.4f}, "
                f"teacher_temp={teacher_temp:.3f}"
            )

            if wandb_logger:
                wandb_logger.log_metrics(
                    {"epoch_loss": avg_loss, "epoch": epoch + 1},
                    step=global_step,
                    prefix="train/",
                )

            if self.save_every > 0 and (epoch + 1) % self.save_every == 0:
                if checkpoint_manager:
                    checkpoint_manager.save_pytorch(
                        model, optimizer, None, epoch + 1,
                        metric=avg_loss, tag=f"epoch_{epoch + 1}",
                    )

        if checkpoint_manager:
            checkpoint_manager.save_pytorch(
                model, optimizer, None, self.epochs,
                metric=avg_loss, tag="latest",
            )

        logger.info("DINO training complete")

    # ---- JAX Training ----

    def train_jax(
        self,
        dino_wrapper,
        data_iterator,
        checkpoint_manager=None,
        wandb_logger=None,
        start_epoch: int = 0,
    ):
        """
        DINO training loop with JAX/Flax.

        Args:
            dino_wrapper: DINO wrapper (src.models.jax_flax.dino.DINO).
            data_iterator: JAXDINODataIterator instance.
            checkpoint_manager: CheckpointManager.
            wandb_logger: WandbLogger.
            start_epoch: Resume epoch.
        """
        import jax
        import jax.numpy as jnp
        import optax
        from src.models.jax_flax.dino import ema_update_params, update_center, compute_dino_loss

        rng = jax.random.PRNGKey(self.config["training"]["seed"])

        # Initialize
        img_size = self.config["data"]["image_size"]
        dummy = jnp.ones((1, 1, img_size, img_size))
        student_params, teacher_params, center = dino_wrapper.init(rng, dummy)

        # Optimizer
        total_steps = self.epochs * len(data_iterator)
        warmup_steps = self.warmup_epochs * len(data_iterator)
        # Cap warmup to avoid negative decay_steps when epochs < warmup_epochs
        warmup_steps = min(warmup_steps, max(total_steps - 1, 0))
        total_steps = max(total_steps, warmup_steps + 1)

        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=self.lr,
            warmup_steps=warmup_steps,
            decay_steps=total_steps,
            end_value=self.min_lr,
        )

        optimizer = optax.chain(
            optax.clip_by_global_norm(self.gradient_clip),
            optax.adamw(learning_rate=schedule, weight_decay=self.weight_decay),
        )
        opt_state = optimizer.init(student_params)

        @jax.jit
        def train_step(student_params, teacher_params, opt_state, center,
                       global_crops, local_crops, rng):
            def loss_fn(s_params):
                return dino_wrapper.loss_fn(
                    s_params, teacher_params,
                    global_crops, local_crops, center,
                    deterministic=False, rng=rng,
                )

            (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(student_params)
            updates, opt_state_new = optimizer.update(grads, opt_state, student_params)
            student_params_new = optax.apply_updates(student_params, updates)

            return student_params_new, opt_state_new, loss, aux

        global_step = 0

        for epoch in range(start_epoch, self.epochs):
            momentum = self.get_momentum(epoch)
            epoch_loss = 0.0
            n_batches = 0

            for global_crops, local_crops in data_iterator:
                rng, step_rng = jax.random.split(rng)

                global_crops_jnp = [jnp.array(gc) for gc in global_crops]
                local_crops_jnp = [jnp.array(lc) for lc in local_crops]

                student_params, opt_state, loss, aux = train_step(
                    student_params, teacher_params, opt_state, center,
                    global_crops_jnp, local_crops_jnp, step_rng,
                )

                # EMA update teacher
                teacher_params = ema_update_params(teacher_params, student_params, momentum)

                # Update center
                center = update_center(center, aux["teacher_output"])

                epoch_loss += float(loss)
                n_batches += 1
                global_step += 1

                if global_step % self.log_every == 0 and wandb_logger:
                    wandb_logger.log_metrics(
                        {"loss": float(loss), "momentum": momentum},
                        step=global_step,
                        prefix="train/",
                    )

            avg_loss = epoch_loss / max(n_batches, 1)
            logger.info(f"DINO Epoch {epoch + 1}: avg_loss={avg_loss:.6f}")

            if self.save_every > 0 and (epoch + 1) % self.save_every == 0:
                if checkpoint_manager:
                    checkpoint_manager.save_jax(
                        student_params, opt_state, epoch + 1,
                        metric=avg_loss, tag=f"epoch_{epoch + 1}",
                        extra={"teacher_params": teacher_params, "center": center},
                    )

        if checkpoint_manager:
            checkpoint_manager.save_jax(
                student_params, opt_state, self.epochs,
                metric=avg_loss, tag="latest",
                extra={"teacher_params": teacher_params, "center": center},
            )

        logger.info("DINO JAX training complete")
        return student_params, teacher_params, center
