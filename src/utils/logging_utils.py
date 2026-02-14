#!/usr/bin/env python3
"""
Logging utilities for experiment tracking.

Provides:
- W&B (Weights & Biases) integration for metrics, images, configs
- Console logging with structured progress bars
- Periodic image sample logging for visual inspection
"""

import logging
import sys
import time
from pathlib import Path
from typing import Optional, Dict, Any, List

import numpy as np

logger = logging.getLogger(__name__)


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    rank: int = 0,
) -> logging.Logger:
    """
    Configure Python logging for training.

    Only rank 0 logs to console; all ranks log to file (if specified).

    Args:
        log_level: Logging level ('DEBUG', 'INFO', 'WARNING', etc.).
        log_file: Path to log file (optional).
        rank: Process rank (only rank 0 logs to console).

    Returns:
        Root logger.
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))

    # Clear existing handlers
    root_logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler (rank 0 only)
    if rank == 0:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    # File handler (all ranks)
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    return root_logger


class WandbLogger:
    """
    Weights & Biases experiment tracker.

    Handles initialization, metric logging, image logging, and finalization.
    Gracefully falls back to console-only logging if W&B is unavailable.

    Args:
        project: W&B project name.
        name: Run name.
        config: Experiment configuration dict.
        enabled: Whether to actually log to W&B.
        log_dir: Local directory for W&B files.
    """

    def __init__(
        self,
        project: str = "jwst-ssl-vit",
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        enabled: bool = True,
        log_dir: str = "wandb_logs",
    ):
        self.enabled = enabled
        self._step = 0
        self._start_time = time.time()

        if not enabled:
            logger.info("W&B logging disabled")
            return

        try:
            import wandb

            self.wandb = wandb

            wandb.init(
                project=project,
                name=name,
                config=config or {},
                dir=log_dir,
                reinit=True,
            )

            logger.info(f"W&B initialized: project={project}, run={wandb.run.name}")

        except ImportError:
            logger.warning("wandb not installed, falling back to console logging")
            self.enabled = False

        except Exception as e:
            logger.warning(f"W&B init failed ({e}), falling back to console logging")
            self.enabled = False

    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
        prefix: str = "",
    ):
        """
        Log scalar metrics.

        Args:
            metrics: Dict of metric_name → value.
            step: Global step (auto-incremented if None).
            prefix: Prefix for metric names (e.g., 'train/', 'val/').
        """
        if step is None:
            step = self._step
            self._step += 1

        prefixed = {f"{prefix}{k}": v for k, v in metrics.items()}

        if self.enabled:
            self.wandb.log(prefixed, step=step)
        else:
            metrics_str = ", ".join(f"{k}={v:.6f}" for k, v in prefixed.items())
            logger.info(f"[Step {step}] {metrics_str}")

    def log_image(
        self,
        key: str,
        images: List[np.ndarray],
        step: Optional[int] = None,
        caption: Optional[List[str]] = None,
    ):
        """
        Log images to W&B.

        Args:
            key: Image group name.
            images: List of numpy arrays (H, W) or (H, W, C), values in [0, 1].
            step: Global step.
            caption: Optional captions per image.
        """
        if not self.enabled:
            return

        wandb_images = []
        for i, img in enumerate(images):
            # Convert to uint8 for W&B
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            cap = caption[i] if caption and i < len(caption) else None
            wandb_images.append(self.wandb.Image(img, caption=cap))

        self.wandb.log({key: wandb_images}, step=step)

    def log_config(self, config: Dict[str, Any]):
        """Update W&B config with additional parameters."""
        if self.enabled:
            self.wandb.config.update(config)

    def log_summary(self, summary: Dict[str, Any]):
        """Log final summary metrics."""
        if self.enabled:
            for k, v in summary.items():
                self.wandb.run.summary[k] = v
        else:
            logger.info(f"Summary: {summary}")

    def finish(self):
        """Finalize the W&B run."""
        elapsed = time.time() - self._start_time
        logger.info(f"Training completed in {elapsed:.1f}s")

        if self.enabled:
            self.wandb.finish()


class MetricTracker:
    """
    Simple metric accumulator for epoch-level averaging.

    Usage:
        tracker = MetricTracker()
        for batch in loader:
            tracker.update({'loss': loss_val, 'lr': lr})
        epoch_metrics = tracker.average()
        tracker.reset()
    """

    def __init__(self):
        self._sum = {}
        self._count = {}

    def update(self, metrics: Dict[str, float], n: int = 1):
        """
        Add metrics from a batch.

        Args:
            metrics: Dict of metric values.
            n: Batch size (for weighted averaging).
        """
        for k, v in metrics.items():
            self._sum[k] = self._sum.get(k, 0.0) + v * n
            self._count[k] = self._count.get(k, 0) + n

    def average(self) -> Dict[str, float]:
        """Compute average of all tracked metrics."""
        return {
            k: self._sum[k] / max(self._count[k], 1) for k in self._sum
        }

    def reset(self):
        """Reset all accumulators."""
        self._sum.clear()
        self._count.clear()
