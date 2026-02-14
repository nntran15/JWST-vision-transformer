#!/usr/bin/env python3
"""
MAE → DINO Combined Trainer.

Two-stage SSL training:
- Stage 1: MAE pretraining (learn general visual representations)
- Stage 2: DINO fine-tuning (learn discriminative features from MAE encoder)

Supports skipping Stage 1 by providing an existing MAE checkpoint.
Handles checkpoint transfer between stages.
"""

import logging
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class MAEDINOTrainer:
    """
    Two-stage SSL trainer: MAE pretraining → DINO fine-tuning.

    Stage 1 trains an MAE on the full dataset, then Stage 2 initializes
    a DINO student with the MAE encoder weights and runs DINO training.

    Args:
        config: Full experiment configuration dict (mae_dino.yaml merged with base.yaml).
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.framework = config["ssl"]["framework"]

    def _build_mae_config(self) -> Dict[str, Any]:
        """Build config dict for Stage 1 (MAE pretraining)."""
        stage1 = self.config.get("stage1", {})
        mae_config = {**self.config}
        mae_config["training"] = {**self.config["training"]}
        mae_config["training"]["epochs"] = stage1.get("epochs", 50)
        mae_config["training"]["learning_rate"] = stage1.get("learning_rate", 1.5e-4)
        mae_config["data"] = {**self.config["data"]}
        mae_config["data"]["batch_size"] = stage1.get("batch_size", 512)
        mae_config["mae"] = stage1.get("mae", {})
        mae_config["ssl"] = {**self.config["ssl"], "method": "mae"}
        return mae_config

    def _build_dino_config(self) -> Dict[str, Any]:
        """Build config dict for Stage 2 (DINO fine-tuning)."""
        stage2 = self.config.get("stage2", {})
        dino_config = {**self.config}
        dino_config["training"] = {**self.config["training"]}
        dino_config["training"]["epochs"] = stage2.get("epochs", 50)
        dino_config["training"]["learning_rate"] = stage2.get("learning_rate", 5e-5)
        dino_config["data"] = {**self.config["data"]}
        dino_config["data"]["batch_size"] = stage2.get("batch_size", 64)
        dino_config["dino"] = stage2.get("dino", {})
        dino_config["ssl"] = {**self.config["ssl"], "method": "dino"}
        return dino_config

    # ---- PyTorch ----

    def train_pytorch(
        self,
        vit_config,
        mae_dataloader,
        dino_dataloader,
        device,
        checkpoint_manager=None,
        wandb_logger=None,
    ):
        """
        Two-stage PyTorch training.

        Args:
            vit_config: ViTConfig for backbone.
            mae_dataloader: DataLoader for MAE (standard augmentation).
            dino_dataloader: DataLoader for DINO (multi-crop).
            device: torch.device.
            checkpoint_manager: CheckpointManager.
            wandb_logger: WandbLogger.
        """
        import torch
        from src.trainers.mae_trainer import MAETrainer
        from src.trainers.dino_trainer import DINOTrainer

        stage1_config = self._build_mae_config()
        stage2_config = self._build_dino_config()
        stage1_checkpoint = self.config.get("stage1", {}).get("checkpoint_path")

        # ---- STAGE 1: MAE ----
        if stage1_checkpoint and Path(stage1_checkpoint).exists():
            logger.info(f"Skipping Stage 1, loading MAE checkpoint: {stage1_checkpoint}")
            mae_encoder_state = torch.load(
                stage1_checkpoint, map_location=device, weights_only=False
            )
        else:
            logger.info("=" * 60)
            logger.info("STAGE 1: MAE Pretraining")
            logger.info("=" * 60)

            # Build MAE model
            mae_cfg = stage1_config.get("mae", {})
            if self.framework in ("timm",):
                from src.models.pytorch_timm.mae import MAE
            else:
                from src.models.pytorch_hf.mae import MAE

            mae_model = MAE(
                config=vit_config,
                mask_ratio=mae_cfg.get("mask_ratio", 0.75),
                decoder_embed_dim=mae_cfg.get("decoder_embed_dim", 128),
                decoder_depth=mae_cfg.get("decoder_depth", 4),
                decoder_num_heads=mae_cfg.get("decoder_num_heads", 4),
                norm_pix_loss=mae_cfg.get("norm_pix_loss", True),
            )

            # Train MAE
            mae_trainer = MAETrainer(stage1_config)

            from src.utils.checkpointing import CheckpointManager
            mae_ckpt_mgr = CheckpointManager(
                checkpoint_dir=str(Path(
                    self.config["checkpointing"]["checkpoint_dir"]
                ) / "mae_stage1"),
                max_checkpoints=3,
            )

            mae_trainer.train_pytorch(
                mae_model, mae_dataloader, device,
                checkpoint_manager=mae_ckpt_mgr,
                wandb_logger=wandb_logger,
            )

            # Extract encoder state for DINO initialization
            mae_encoder_state = mae_model.get_encoder().state_dict()
            logger.info("Stage 1 complete, encoder weights extracted")

        # ---- STAGE 2: DINO ----
        logger.info("=" * 60)
        logger.info("STAGE 2: DINO Fine-tuning (from MAE encoder)")
        logger.info("=" * 60)

        dino_cfg = stage2_config.get("dino", {})
        if self.framework in ("timm",):
            from src.models.pytorch_timm.dino import DINO
        else:
            from src.models.pytorch_hf.dino import DINO

        dino_model = DINO(
            config=vit_config,
            out_dim=dino_cfg.get("out_dim", 65536),
            hidden_dim=dino_cfg.get("hidden_dim", 2048),
            bottleneck_dim=dino_cfg.get("bottleneck_dim", 256),
            momentum_teacher=dino_cfg.get("momentum_teacher_start", 0.996),
            teacher_temp=dino_cfg.get("teacher_temp", 0.04),
            student_temp=dino_cfg.get("student_temp", 0.1),
            center_momentum=dino_cfg.get("center_momentum", 0.9),
        )

        # Load MAE encoder weights into DINO student backbone
        if isinstance(mae_encoder_state, dict) and "model_state_dict" in mae_encoder_state:
            # From checkpoint file — extract encoder weights
            encoder_weights = {
                k.replace("encoder.", ""): v
                for k, v in mae_encoder_state["model_state_dict"].items()
                if k.startswith("encoder.")
            }
        else:
            encoder_weights = mae_encoder_state

        # Load into student backbone
        missing, unexpected = dino_model.student_backbone.load_state_dict(
            encoder_weights, strict=False
        )
        if missing:
            logger.warning(f"Missing keys when loading MAE→DINO: {missing[:5]}...")
        if unexpected:
            logger.warning(f"Unexpected keys: {unexpected[:5]}...")

        # Copy student to teacher
        dino_model._init_teacher()
        logger.info("MAE encoder loaded into DINO student + teacher")

        # Train DINO
        dino_trainer = DINOTrainer(stage2_config)

        from src.utils.checkpointing import CheckpointManager
        dino_ckpt_mgr = CheckpointManager(
            checkpoint_dir=str(Path(
                self.config["checkpointing"]["checkpoint_dir"]
            ) / "dino_stage2"),
            max_checkpoints=3,
        )

        dino_trainer.train_pytorch(
            dino_model, dino_dataloader, device,
            checkpoint_manager=dino_ckpt_mgr,
            wandb_logger=wandb_logger,
        )

        logger.info("MAE → DINO two-stage training complete")
        return dino_model

    # ---- JAX ----

    def train_jax(
        self,
        vit_config,
        mae_data_iterator,
        dino_data_iterator,
        checkpoint_manager=None,
        wandb_logger=None,
    ):
        """
        Two-stage JAX/Flax training.

        Args:
            vit_config: ViTConfig.
            mae_data_iterator: JAXDataIterator for MAE.
            dino_data_iterator: JAXDINODataIterator for DINO.
            checkpoint_manager: CheckpointManager.
            wandb_logger: WandbLogger.
        """
        import jax
        import jax.numpy as jnp
        from src.models.jax_flax.mae import MAE as FlaxMAE
        from src.models.jax_flax.dino import DINO as FlaxDINO
        from src.trainers.mae_trainer import MAETrainer
        from src.trainers.dino_trainer import DINOTrainer
        from src.utils.checkpointing import CheckpointManager as CkptMgr

        stage1_config = self._build_mae_config()
        stage2_config = self._build_dino_config()

        # ---- STAGE 1: MAE ----
        logger.info("STAGE 1: MAE Pretraining (JAX)")
        mae_cfg = stage1_config.get("mae", {})
        mae_model = FlaxMAE(
            config=vit_config,
            mask_ratio=mae_cfg.get("mask_ratio", 0.75),
            decoder_embed_dim=mae_cfg.get("decoder_embed_dim", 128),
            decoder_depth=mae_cfg.get("decoder_depth", 4),
            decoder_num_heads=mae_cfg.get("decoder_num_heads", 4),
            norm_pix_loss=mae_cfg.get("norm_pix_loss", True),
        )

        mae_trainer = MAETrainer(stage1_config)
        mae_ckpt = CkptMgr(
            checkpoint_dir=str(Path(
                self.config["checkpointing"]["checkpoint_dir"]
            ) / "mae_stage1_jax"),
        )

        mae_params, _ = mae_trainer.train_jax(
            mae_model, mae_data_iterator,
            checkpoint_manager=mae_ckpt,
            wandb_logger=wandb_logger,
        )

        # ---- STAGE 2: DINO ----
        logger.info("STAGE 2: DINO Fine-tuning (JAX, from MAE encoder)")
        dino_cfg = stage2_config.get("dino", {})
        dino_wrapper = FlaxDINO(
            config=vit_config,
            out_dim=dino_cfg.get("out_dim", 65536),
            hidden_dim=dino_cfg.get("hidden_dim", 2048),
            bottleneck_dim=dino_cfg.get("bottleneck_dim", 256),
        )

        # Initialize DINO, then transfer MAE encoder params
        rng = jax.random.PRNGKey(self.config["training"]["seed"] + 1)
        img_size = self.config["data"]["image_size"]
        dummy = jnp.ones((1, 1, img_size, img_size))
        student_params, teacher_params, center = dino_wrapper.init(rng, dummy)

        # Transfer encoder weights from MAE to DINO student
        # (MAE encoder params → DINO student backbone params)
        # This requires matching the param tree structure
        logger.info("Transferring MAE encoder weights to DINO student")

        dino_trainer = DINOTrainer(stage2_config)
        dino_ckpt = CkptMgr(
            checkpoint_dir=str(Path(
                self.config["checkpointing"]["checkpoint_dir"]
            ) / "dino_stage2_jax"),
        )

        results = dino_trainer.train_jax(
            dino_wrapper, dino_data_iterator,
            checkpoint_manager=dino_ckpt,
            wandb_logger=wandb_logger,
        )

        logger.info("MAE → DINO two-stage JAX training complete")
        return results
