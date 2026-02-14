#!/usr/bin/env python3
"""
MAE Trainer: Training loop for Masked Autoencoder pretraining.

Supports all three frameworks (timm, HuggingFace, JAX/Flax) with a
unified interface. Handles:
- Forward → mask → reconstruct → MSE loss → backward
- Cosine LR schedule with linear warmup
- Mixed precision (PyTorch AMP / JAX bf16)
- Periodic visualization of reconstructed patches
- Checkpoint saving/loading
"""

import logging
import math
import time
from typing import Dict, Any, Optional

import numpy as np
import yaml

logger = logging.getLogger(__name__)


class MAETrainer:
    """
    Training loop for MAE pretraining across all frameworks.

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
        self.viz_every = config["logging"]["visualize_every_n_epochs"]

    def get_lr(self, epoch: int) -> float:
        """Compute learning rate with warmup + cosine decay."""
        if epoch < self.warmup_epochs:
            return self.lr * epoch / max(self.warmup_epochs, 1)
        # Cosine decay
        progress = (epoch - self.warmup_epochs) / max(
            self.epochs - self.warmup_epochs, 1
        )
        return self.min_lr + 0.5 * (self.lr - self.min_lr) * (
            1 + math.cos(math.pi * progress)
        )

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
        Run MAE training loop with PyTorch (timm or HuggingFace).

        Args:
            model: MAE nn.Module (timm or HF).
            dataloader: PyTorch DataLoader.
            device: torch.device.
            checkpoint_manager: CheckpointManager instance.
            wandb_logger: WandbLogger instance.
            start_epoch: Resume from this epoch.
        """
        import torch
        from tqdm import tqdm

        model = model.to(device)
        model.train()

        # Optimizer: AdamW with weight decay
        param_groups = [
            {
                "params": [
                    p for n, p in model.named_parameters()
                    if p.requires_grad and "bias" not in n and "norm" not in n
                ],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [
                    p for n, p in model.named_parameters()
                    if p.requires_grad and ("bias" in n or "norm" in n)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(param_groups, lr=self.lr, betas=(0.9, 0.95))

        # Mixed precision
        scaler = torch.cuda.amp.GradScaler() if self.mixed_precision else None

        global_step = 0

        for epoch in range(start_epoch, self.epochs):
            # Update learning rate
            lr = self.get_lr(epoch)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            epoch_loss = 0.0
            n_batches = 0

            pbar = tqdm(
                dataloader,
                desc=f"Epoch {epoch + 1}/{self.epochs}",
                disable=not logger.isEnabledFor(logging.INFO),
            )

            for batch_idx, images in enumerate(pbar):
                images = images.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                if self.mixed_precision:
                    with torch.cuda.amp.autocast():
                        outputs = model(images)
                        loss = outputs["loss"]

                    scaler.scale(loss).backward()

                    if self.gradient_clip > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), self.gradient_clip
                        )

                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = model(images)
                    loss = outputs["loss"]
                    loss.backward()

                    if self.gradient_clip > 0:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), self.gradient_clip
                        )

                    optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1
                global_step += 1

                # Logging
                if global_step % self.log_every == 0:
                    avg_loss = epoch_loss / n_batches
                    pbar.set_postfix(loss=f"{loss.item():.4f}", avg=f"{avg_loss:.4f}", lr=f"{lr:.2e}")

                    if wandb_logger:
                        wandb_logger.log_metrics(
                            {"loss": loss.item(), "lr": lr},
                            step=global_step,
                            prefix="train/",
                        )

            # Epoch summary
            avg_loss = epoch_loss / max(n_batches, 1)
            logger.info(f"Epoch {epoch + 1}: avg_loss={avg_loss:.6f}, lr={lr:.2e}")

            if wandb_logger:
                wandb_logger.log_metrics(
                    {"epoch_loss": avg_loss, "epoch": epoch + 1},
                    step=global_step,
                    prefix="train/",
                )

            # Visualization
            if self.viz_every > 0 and (epoch + 1) % self.viz_every == 0:
                self._visualize_reconstruction_pytorch(
                    model, images, outputs, wandb_logger, global_step
                )

            # Save checkpoint
            if self.save_every > 0 and (epoch + 1) % self.save_every == 0:
                if checkpoint_manager:
                    checkpoint_manager.save_pytorch(
                        model, optimizer, None, epoch + 1,
                        metric=avg_loss, tag=f"epoch_{epoch + 1}",
                    )

        # Save final checkpoint
        if checkpoint_manager:
            checkpoint_manager.save_pytorch(
                model, optimizer, None, self.epochs,
                metric=avg_loss, tag="latest",
            )

        logger.info("MAE training complete")

    def _visualize_reconstruction_pytorch(
        self, model, images, outputs, wandb_logger, step
    ):
        """Log MAE reconstruction visualizations."""
        if wandb_logger is None:
            return

        import torch

        model.eval()
        with torch.no_grad():
            pred = outputs["pred"]
            mask = outputs["mask"]

            # Unpatchify predictions
            pred_images = model.unpatchify(pred)  # (B, 1, H, W)

            # Log a few samples
            n_show = min(4, images.shape[0])
            vis_images = []
            captions = []

            for i in range(n_show):
                orig = images[i, 0].cpu().numpy()
                recon = pred_images[i, 0].cpu().numpy()
                recon = np.clip(recon, 0, 1)

                # Side-by-side: original | reconstruction
                combined = np.concatenate([orig, recon], axis=1)
                vis_images.append(combined)
                captions.append(f"Sample {i}: orig | recon")

            wandb_logger.log_image(
                "mae_reconstruction", vis_images, step=step, caption=captions
            )

        model.train()

    # ---- JAX Training ----

    def train_jax(
        self,
        model,
        data_iterator,
        checkpoint_manager=None,
        wandb_logger=None,
        start_epoch: int = 0,
    ):
        """
        Run MAE training loop with JAX/Flax.

        Args:
            model: Flax MAE module.
            data_iterator: JAXDataIterator instance.
            checkpoint_manager: CheckpointManager instance.
            wandb_logger: WandbLogger instance.
            start_epoch: Resume epoch.
        """
        import jax
        import jax.numpy as jnp
        import optax

        # Initialize model
        rng = jax.random.PRNGKey(self.config["training"]["seed"])
        rng, init_rng, mask_rng = jax.random.split(rng, 3)

        dummy_input = jnp.ones((1, 1, self.config["data"]["image_size"],
                                self.config["data"]["image_size"]))
        params = model.init(init_rng, dummy_input, mask_rng)

        # Optimizer: AdamW with cosine schedule
        total_steps = self.epochs * len(data_iterator)
        warmup_steps = self.warmup_epochs * len(data_iterator)

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
        opt_state = optimizer.init(params)

        # JIT-compiled training step
        @jax.jit
        def train_step(params, opt_state, batch, rng):
            def loss_fn(params):
                outputs = model.apply(params, batch, rng)
                return outputs["loss"], outputs

            (loss, outputs), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
            updates, opt_state_new = optimizer.update(grads, opt_state, params)
            params_new = optax.apply_updates(params, updates)
            return params_new, opt_state_new, loss, outputs

        # Training loop
        global_step = 0

        for epoch in range(start_epoch, self.epochs):
            epoch_loss = 0.0
            n_batches = 0

            for batch in data_iterator:
                rng, step_rng = jax.random.split(rng)
                batch = jnp.array(batch)

                params, opt_state, loss, outputs = train_step(
                    params, opt_state, batch, step_rng
                )

                epoch_loss += float(loss)
                n_batches += 1
                global_step += 1

                if global_step % self.log_every == 0 and wandb_logger:
                    wandb_logger.log_metrics(
                        {"loss": float(loss)},
                        step=global_step,
                        prefix="train/",
                    )

            avg_loss = epoch_loss / max(n_batches, 1)
            logger.info(f"Epoch {epoch + 1}: avg_loss={avg_loss:.6f}")

            if wandb_logger:
                wandb_logger.log_metrics(
                    {"epoch_loss": avg_loss, "epoch": epoch + 1},
                    step=global_step,
                    prefix="train/",
                )

            if self.save_every > 0 and (epoch + 1) % self.save_every == 0:
                if checkpoint_manager:
                    checkpoint_manager.save_jax(
                        params, opt_state, epoch + 1,
                        metric=avg_loss, tag=f"epoch_{epoch + 1}",
                    )

        if checkpoint_manager:
            checkpoint_manager.save_jax(
                params, opt_state, self.epochs,
                metric=avg_loss, tag="latest",
            )

        logger.info("MAE JAX training complete")
        return params, opt_state
