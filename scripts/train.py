#!/usr/bin/env python3
"""
Main training entry point for SSL ViT pretraining on JWST galaxy thumbnails.

Supports all 9 configurations:
  3 SSL methods (MAE, DINO, MAE→DINO) × 3 frameworks (timm, HuggingFace, JAX)

Usage:
  # MAE with timm, ViT-Tiny
  python scripts/train.py --config configs/mae.yaml --framework timm --vit_size tiny

  # DINO with HuggingFace, ViT-Small
  python scripts/train.py --config configs/dino.yaml --framework hf --vit_size small

  # MAE→DINO with JAX
  python scripts/train.py --config configs/mae_dino.yaml --framework jax --vit_size tiny

  # Multi-GPU (PyTorch DDP)
  torchrun --nproc_per_node=4 scripts/train.py --config configs/mae.yaml --distributed

  # Resume from checkpoint
  python scripts/train.py --config configs/mae.yaml --resume output/checkpoints/checkpoint_latest.pt
"""

import argparse
import logging
import sys
from pathlib import Path

import yaml

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def load_config(config_path: str, overrides: dict) -> dict:
    """Load YAML config and merge with base.yaml, then apply CLI overrides."""
    base_path = project_root / "configs" / "base.yaml"

    with open(base_path) as f:
        config = yaml.safe_load(f)

    if config_path:
        with open(config_path) as f:
            method_config = yaml.safe_load(f)
        # Deep merge
        _deep_merge(config, method_config)

    # Apply CLI overrides
    for key, value in overrides.items():
        keys = key.split(".")
        d = config
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = value

    return config


def _deep_merge(base: dict, override: dict):
    """Recursively merge override into base."""
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v


def build_model_and_data(config: dict):
    """
    Build model, data loader, and trainer based on configuration.

    Returns framework-specific objects ready for training.
    """
    from src.models.vit_config import get_vit_config
    from src.data.fits_dataset import FITSDataset, build_file_index
    from src.data.augmentations import AstronomyAugmentations, DINOMultiCropAugmentation

    method = config["ssl"]["method"]
    framework = config["ssl"]["framework"]

    # Build ViT config
    vit_config = get_vit_config(
        size=config["model"]["vit_size"],
        image_size=config["data"]["image_size"],
        patch_size=config["model"]["patch_size"],
        in_channels=config["model"]["in_channels"],
        drop_rate=config["model"]["drop_rate"],
        attn_drop_rate=config["model"]["attn_drop_rate"],
        drop_path_rate=config["model"]["drop_path_rate"],
    )

    # Build file index
    index = build_file_index(
        catalog_dir=config["data"]["catalog_dir"],
        index_path=config["data"]["index_path"],
        min_dim=config["data"]["min_dim"],
        max_dim=config["data"]["max_dim"],
    )

    # Build base dataset
    fits_dataset = FITSDataset(
        index=index,
        target_size=config["data"]["image_size"],
        normalization=config["data"]["normalization"],
        percentile_clip=tuple(config["data"]["percentile_clip"]),
    )

    if framework in ("timm", "hf"):
        return _build_pytorch(config, vit_config, fits_dataset, method, framework)
    elif framework == "jax":
        return _build_jax(config, vit_config, fits_dataset, method)
    else:
        raise ValueError(f"Unknown framework: {framework}")


def _build_pytorch(config, vit_config, fits_dataset, method, framework):
    """Build PyTorch model, dataloader, and trainer."""
    from src.data.pytorch_loader import create_dataloader, create_dino_dataloader
    from src.data.augmentations import AstronomyAugmentations, DINOMultiCropAugmentation

    aug_cfg = config.get("augmentation", {})

    if method == "mae":
        # Standard augmentation + dataloader
        augmentation = AstronomyAugmentations(
            rotation=aug_cfg.get("rotation", True),
            flip_h=aug_cfg.get("flip_h", True),
            flip_v=aug_cfg.get("flip_v", True),
            gaussian_noise_std=aug_cfg.get("gaussian_noise_std", 0.02),
            brightness_range=tuple(aug_cfg.get("brightness_range", [0.9, 1.1])),
            contrast_range=tuple(aug_cfg.get("contrast_range", [0.9, 1.1])),
            random_erasing_prob=aug_cfg.get("random_erasing_prob", 0.0),
        )

        dist_cfg = config.get("distributed", {})
        dataloader = create_dataloader(
            fits_dataset,
            batch_size=config["data"]["batch_size"],
            num_workers=config["data"]["num_workers"],
            augmentation=augmentation,
            distributed=dist_cfg.get("enabled", False),
            seed=config["training"]["seed"],
        )

        # Build MAE model
        mae_cfg = config.get("mae", {})
        if framework == "timm":
            from src.models.pytorch_timm.mae import MAE
        else:
            from src.models.pytorch_hf.mae import MAE

        model = MAE(
            config=vit_config,
            mask_ratio=mae_cfg.get("mask_ratio", 0.75),
            decoder_embed_dim=mae_cfg.get("decoder_embed_dim", 128),
            decoder_depth=mae_cfg.get("decoder_depth", 4),
            decoder_num_heads=mae_cfg.get("decoder_num_heads", 4),
            norm_pix_loss=mae_cfg.get("norm_pix_loss", True),
        )

        from src.trainers.mae_trainer import MAETrainer
        trainer = MAETrainer(config)

        return {
            "model": model,
            "dataloader": dataloader,
            "trainer": trainer,
            "vit_config": vit_config,
            "type": "pytorch_mae",
        }

    elif method == "dino":
        # Multi-crop augmentation + dataloader
        multi_crop = DINOMultiCropAugmentation(
            global_crop_size=aug_cfg.get("global_crop_size", 64),
            local_crop_size=aug_cfg.get("local_crop_size", 32),
            n_local_crops=aug_cfg.get("n_local_crops", 6),
        )

        dist_cfg = config.get("distributed", {})
        dataloader = create_dino_dataloader(
            fits_dataset,
            multi_crop_aug=multi_crop,
            batch_size=config["data"]["batch_size"],
            num_workers=config["data"]["num_workers"],
            distributed=dist_cfg.get("enabled", False),
            seed=config["training"]["seed"],
        )

        # Build DINO model
        dino_cfg = config.get("dino", {})
        if framework == "timm":
            from src.models.pytorch_timm.dino import DINO
        else:
            from src.models.pytorch_hf.dino import DINO

        model = DINO(
            config=vit_config,
            out_dim=dino_cfg.get("out_dim", 65536),
            hidden_dim=dino_cfg.get("hidden_dim", 2048),
            bottleneck_dim=dino_cfg.get("bottleneck_dim", 256),
            momentum_teacher=dino_cfg.get("momentum_teacher_start", 0.996),
            teacher_temp=dino_cfg.get("teacher_temp", 0.04),
            student_temp=dino_cfg.get("student_temp", 0.1),
            center_momentum=dino_cfg.get("center_momentum", 0.9),
        )

        from src.trainers.dino_trainer import DINOTrainer
        trainer = DINOTrainer(config)

        return {
            "model": model,
            "dataloader": dataloader,
            "trainer": trainer,
            "vit_config": vit_config,
            "type": "pytorch_dino",
        }

    elif method == "mae_dino":
        # Need both MAE and DINO dataloaders
        mae_aug = AstronomyAugmentations(
            rotation=aug_cfg.get("rotation", True),
            flip_h=aug_cfg.get("flip_h", True),
            flip_v=aug_cfg.get("flip_v", True),
        )

        stage1_cfg = config.get("stage1", {})
        mae_dataloader = create_dataloader(
            fits_dataset,
            batch_size=stage1_cfg.get("batch_size", 512),
            num_workers=config["data"]["num_workers"],
            augmentation=mae_aug,
            seed=config["training"]["seed"],
        )

        stage2_aug_cfg = config.get("stage2", {}).get("augmentation", aug_cfg)
        dino_multi_crop = DINOMultiCropAugmentation(
            global_crop_size=stage2_aug_cfg.get("global_crop_size", 64),
            local_crop_size=stage2_aug_cfg.get("local_crop_size", 32),
            n_local_crops=stage2_aug_cfg.get("n_local_crops", 6),
        )

        stage2_cfg = config.get("stage2", {})
        dino_dataloader = create_dino_dataloader(
            fits_dataset,
            multi_crop_aug=dino_multi_crop,
            batch_size=stage2_cfg.get("batch_size", 64),
            num_workers=config["data"]["num_workers"],
            seed=config["training"]["seed"],
        )

        from src.trainers.mae_dino_trainer import MAEDINOTrainer
        trainer = MAEDINOTrainer(config)

        return {
            "mae_dataloader": mae_dataloader,
            "dino_dataloader": dino_dataloader,
            "trainer": trainer,
            "vit_config": vit_config,
            "type": "pytorch_mae_dino",
        }

    else:
        raise ValueError(f"Unknown method: {method}")


def _build_jax(config, vit_config, fits_dataset, method):
    """Build JAX model, data iterator, and trainer."""
    from src.data.jax_loader import JAXDataIterator, JAXDINODataIterator
    from src.data.augmentations import AstronomyAugmentations, DINOMultiCropAugmentation

    aug_cfg = config.get("augmentation", {})

    if method == "mae":
        augmentation = AstronomyAugmentations(
            gaussian_noise_std=aug_cfg.get("gaussian_noise_std", 0.02),
        )

        data_iter = JAXDataIterator(
            fits_dataset,
            batch_size=config["data"]["batch_size"],
            augmentation=augmentation,
            seed=config["training"]["seed"],
        )

        from src.models.jax_flax.mae import MAE
        model = MAE(
            config=vit_config,
            mask_ratio=config.get("mae", {}).get("mask_ratio", 0.75),
        )

        from src.trainers.mae_trainer import MAETrainer
        trainer = MAETrainer(config)

        return {
            "model": model,
            "data_iterator": data_iter,
            "trainer": trainer,
            "vit_config": vit_config,
            "type": "jax_mae",
        }

    elif method == "dino":
        multi_crop = DINOMultiCropAugmentation(
            global_crop_size=aug_cfg.get("global_crop_size", 64),
            local_crop_size=aug_cfg.get("local_crop_size", 32),
            n_local_crops=aug_cfg.get("n_local_crops", 6),
        )

        data_iter = JAXDINODataIterator(
            fits_dataset,
            multi_crop_aug=multi_crop,
            batch_size=config["data"]["batch_size"],
            seed=config["training"]["seed"],
        )

        from src.models.jax_flax.dino import DINO
        dino_wrapper = DINO(config=vit_config)

        from src.trainers.dino_trainer import DINOTrainer
        trainer = DINOTrainer(config)

        return {
            "model": dino_wrapper,
            "data_iterator": data_iter,
            "trainer": trainer,
            "vit_config": vit_config,
            "type": "jax_dino",
        }

    elif method == "mae_dino":
        mae_aug = AstronomyAugmentations()
        mae_iter = JAXDataIterator(
            fits_dataset,
            batch_size=config.get("stage1", {}).get("batch_size", 512),
            augmentation=mae_aug,
            seed=config["training"]["seed"],
        )

        dino_multi_crop = DINOMultiCropAugmentation()
        dino_iter = JAXDINODataIterator(
            fits_dataset,
            multi_crop_aug=dino_multi_crop,
            batch_size=config.get("stage2", {}).get("batch_size", 64),
            seed=config["training"]["seed"],
        )

        from src.trainers.mae_dino_trainer import MAEDINOTrainer
        trainer = MAEDINOTrainer(config)

        return {
            "mae_data_iterator": mae_iter,
            "dino_data_iterator": dino_iter,
            "trainer": trainer,
            "vit_config": vit_config,
            "type": "jax_mae_dino",
        }


def main():
    parser = argparse.ArgumentParser(
        description="SSL ViT Training for JWST Galaxy Classification"
    )
    parser.add_argument(
        "--config", type=str, default="configs/mae.yaml",
        help="Path to method-specific YAML config",
    )
    parser.add_argument(
        "--framework", type=str, choices=["timm", "hf", "jax"],
        help="Override framework (timm, hf, jax)",
    )
    parser.add_argument(
        "--vit_size", type=str, choices=["tiny", "small", "base"],
        help="Override ViT model size",
    )
    parser.add_argument(
        "--batch_size", type=int, help="Override batch size",
    )
    parser.add_argument(
        "--epochs", type=int, help="Override number of epochs",
    )
    parser.add_argument(
        "--lr", type=float, help="Override learning rate",
    )
    parser.add_argument(
        "--distributed", action="store_true",
        help="Enable distributed training",
    )
    parser.add_argument(
        "--resume", type=str, help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--wandb", action="store_true", help="Enable W&B logging",
    )
    parser.add_argument(
        "--seed", type=int, help="Override random seed",
    )

    args = parser.parse_args()

    # Build overrides from CLI
    overrides = {}
    if args.framework:
        overrides["ssl.framework"] = args.framework
    if args.vit_size:
        overrides["model.vit_size"] = args.vit_size
    if args.batch_size:
        overrides["data.batch_size"] = args.batch_size
    if args.epochs:
        overrides["training.epochs"] = args.epochs
    if args.lr:
        overrides["training.learning_rate"] = args.lr
    if args.distributed:
        overrides["distributed.enabled"] = True
    if args.resume:
        overrides["checkpointing.resume_from"] = args.resume
    if args.wandb:
        overrides["logging.wandb_enabled"] = True
    if args.seed:
        overrides["training.seed"] = args.seed

    # Load config
    config = load_config(args.config, overrides)

    # Setup logging
    from src.utils.logging_utils import setup_logging, WandbLogger
    setup_logging(
        log_level=config["logging"]["log_level"],
        log_file=str(Path(config["logging"]["log_dir"]) / "train.log"),
    )

    logger = logging.getLogger(__name__)
    logger.info(f"Config: method={config['ssl']['method']}, "
                f"framework={config['ssl']['framework']}, "
                f"vit_size={config['model']['vit_size']}")

    # Setup W&B
    wandb_logger = WandbLogger(
        project=config["logging"]["wandb_project"],
        name=config["logging"].get("wandb_name") or (
            f"{config['ssl']['method']}_{config['ssl']['framework']}_"
            f"{config['model']['vit_size']}"
        ),
        config=config,
        enabled=config["logging"]["wandb_enabled"],
    )

    # Setup checkpointing
    from src.utils.checkpointing import CheckpointManager
    checkpoint_manager = CheckpointManager(
        checkpoint_dir=config["checkpointing"]["checkpoint_dir"],
        max_checkpoints=config["checkpointing"]["max_checkpoints"],
    )

    # Build model, data, trainer
    components = build_model_and_data(config)
    comp_type = components["type"]

    # Distributed setup
    dist_info = {"rank": 0, "device": None}
    if config.get("distributed", {}).get("enabled", False):
        from src.utils.distributed import setup_distributed
        dist_info = setup_distributed(backend=config["distributed"]["backend"])

    # ---- RUN TRAINING ----
    if comp_type == "pytorch_mae":
        import torch
        device = dist_info.get("device") or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Resume if specified
        start_epoch = 0
        if config["checkpointing"].get("resume_from"):
            state = checkpoint_manager.load_pytorch(
                components["model"], tag="latest", map_location=device
            )
            start_epoch = state.get("epoch", 0)

        components["trainer"].train_pytorch(
            components["model"],
            components["dataloader"],
            device,
            checkpoint_manager=checkpoint_manager,
            wandb_logger=wandb_logger,
            start_epoch=start_epoch,
        )

    elif comp_type == "pytorch_dino":
        import torch
        device = dist_info.get("device") or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        start_epoch = 0
        if config["checkpointing"].get("resume_from"):
            state = checkpoint_manager.load_pytorch(
                components["model"], tag="latest", map_location=device
            )
            start_epoch = state.get("epoch", 0)

        components["trainer"].train_pytorch(
            components["model"],
            components["dataloader"],
            device,
            checkpoint_manager=checkpoint_manager,
            wandb_logger=wandb_logger,
            start_epoch=start_epoch,
        )

    elif comp_type == "pytorch_mae_dino":
        import torch
        device = dist_info.get("device") or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        components["trainer"].train_pytorch(
            components["vit_config"],
            components["mae_dataloader"],
            components["dino_dataloader"],
            device,
            checkpoint_manager=checkpoint_manager,
            wandb_logger=wandb_logger,
        )

    elif comp_type == "jax_mae":
        components["trainer"].train_jax(
            components["model"],
            components["data_iterator"],
            checkpoint_manager=checkpoint_manager,
            wandb_logger=wandb_logger,
        )

    elif comp_type == "jax_dino":
        components["trainer"].train_jax(
            components["model"],
            components["data_iterator"],
            checkpoint_manager=checkpoint_manager,
            wandb_logger=wandb_logger,
        )

    elif comp_type == "jax_mae_dino":
        components["trainer"].train_jax(
            components["vit_config"],
            components["mae_data_iterator"],
            components["dino_data_iterator"],
            checkpoint_manager=checkpoint_manager,
            wandb_logger=wandb_logger,
        )

    wandb_logger.finish()

    # Cleanup distributed
    if config.get("distributed", {}).get("enabled", False):
        from src.utils.distributed import cleanup_distributed
        cleanup_distributed()

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
