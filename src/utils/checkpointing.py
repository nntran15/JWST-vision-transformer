#!/usr/bin/env python3
"""
Checkpoint management for PyTorch and JAX/Flax models.

Handles:
- Save/load model, optimizer, scheduler, and epoch state
- Best-model tracking by validation metric
- Automatic checkpoint directory management
- Framework-agnostic interface with PyTorch and JAX backends
"""

import json
import logging
import shutil
from pathlib import Path
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class CheckpointManager:
    """
    Unified checkpoint manager for training state.

    Saves and loads full training state (model, optimizer, scheduler, epoch).
    Tracks the best model based on a metric (e.g., lowest loss).

    Args:
        checkpoint_dir: Directory to store checkpoints.
        max_checkpoints: Maximum number of checkpoints to keep (0 = unlimited).
        metric_name: Name of the metric to track for best model.
        metric_mode: 'min' or 'max' — whether lower or higher is better.
    """

    def __init__(
        self,
        checkpoint_dir: str = "checkpoints",
        max_checkpoints: int = 5,
        metric_name: str = "loss",
        metric_mode: str = "min",
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self.metric_name = metric_name
        self.metric_mode = metric_mode

        self._best_metric = float("inf") if metric_mode == "min" else float("-inf")
        self._checkpoint_history = []

        # Load checkpoint history if exists
        history_file = self.checkpoint_dir / "checkpoint_history.json"
        if history_file.exists():
            with open(history_file) as f:
                data = json.load(f)
                self._best_metric = data.get("best_metric", self._best_metric)
                self._checkpoint_history = data.get("history", [])

    def _save_history(self):
        """Save checkpoint history metadata."""
        history_file = self.checkpoint_dir / "checkpoint_history.json"
        with open(history_file, "w") as f:
            json.dump(
                {
                    "best_metric": self._best_metric,
                    "history": self._checkpoint_history,
                    "metric_name": self.metric_name,
                    "metric_mode": self.metric_mode,
                },
                f,
                indent=2,
            )

    def _is_better(self, metric: float) -> bool:
        """Check if metric is better than current best."""
        if self.metric_mode == "min":
            return metric < self._best_metric
        return metric > self._best_metric

    def _cleanup_old_checkpoints(self):
        """Remove oldest checkpoints beyond max_checkpoints limit."""
        if self.max_checkpoints <= 0:
            return

        while len(self._checkpoint_history) > self.max_checkpoints:
            oldest = self._checkpoint_history.pop(0)
            path = Path(oldest["path"])
            if path.exists():
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()
                logger.info(f"Removed old checkpoint: {path}")

    # ---- PyTorch ----

    def save_pytorch(
        self,
        model,
        optimizer,
        scheduler,
        epoch: int,
        metric: Optional[float] = None,
        extra: Optional[Dict[str, Any]] = None,
        tag: str = "latest",
    ) -> str:
        """
        Save PyTorch training state.

        Args:
            model: nn.Module (handles DDP unwrapping).
            optimizer: Optimizer state.
            scheduler: LR scheduler state (or None).
            epoch: Current epoch number.
            metric: Current metric value (for best-model tracking).
            extra: Additional items to save.
            tag: Checkpoint tag ('latest', 'best', or epoch number).

        Returns:
            Path to saved checkpoint.
        """
        import torch

        # Handle DDP wrapping
        model_to_save = model.module if hasattr(model, "module") else model

        state = {
            "epoch": epoch,
            "model_state_dict": model_to_save.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }

        if scheduler is not None:
            state["scheduler_state_dict"] = scheduler.state_dict()

        if metric is not None:
            state["metric"] = metric
            state["metric_name"] = self.metric_name

        if extra:
            state.update(extra)

        path = self.checkpoint_dir / f"checkpoint_{tag}.pt"
        torch.save(state, path)
        logger.info(f"Saved checkpoint: {path} (epoch={epoch})")

        # Track in history
        self._checkpoint_history.append({
            "path": str(path),
            "epoch": epoch,
            "metric": metric,
            "tag": tag,
        })

        # Save best model
        if metric is not None and self._is_better(metric):
            self._best_metric = metric
            best_path = self.checkpoint_dir / "checkpoint_best.pt"
            shutil.copy2(path, best_path)
            logger.info(f"New best model: {self.metric_name}={metric:.6f}")

        self._cleanup_old_checkpoints()
        self._save_history()

        return str(path)

    def load_pytorch(
        self,
        model,
        optimizer=None,
        scheduler=None,
        tag: str = "latest",
        map_location=None,
    ) -> Dict[str, Any]:
        """
        Load PyTorch training state.

        Args:
            model: nn.Module to load weights into.
            optimizer: Optimizer to restore state (optional).
            scheduler: Scheduler to restore state (optional).
            tag: Checkpoint tag to load.
            map_location: Device mapping for torch.load.

        Returns:
            Dict with loaded state (epoch, metric, etc.).
        """
        import torch

        path = self.checkpoint_dir / f"checkpoint_{tag}.pt"
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        state = torch.load(path, map_location=map_location, weights_only=False)

        model_to_load = model.module if hasattr(model, "module") else model
        model_to_load.load_state_dict(state["model_state_dict"])

        if optimizer is not None and "optimizer_state_dict" in state:
            optimizer.load_state_dict(state["optimizer_state_dict"])

        if scheduler is not None and "scheduler_state_dict" in state:
            scheduler.load_state_dict(state["scheduler_state_dict"])

        logger.info(f"Loaded checkpoint: {path} (epoch={state.get('epoch', '?')})")
        return state

    # ---- JAX/Flax ----

    def save_jax(
        self,
        params: Any,
        opt_state: Any,
        epoch: int,
        metric: Optional[float] = None,
        extra: Optional[Dict[str, Any]] = None,
        tag: str = "latest",
    ) -> str:
        """
        Save JAX/Flax training state.

        Uses orbax for efficient checkpoint serialization.

        Args:
            params: Flax parameter pytree.
            opt_state: Optax optimizer state.
            epoch: Current epoch.
            metric: Current metric value.
            extra: Additional data.
            tag: Checkpoint tag.

        Returns:
            Path to saved checkpoint directory.
        """
        try:
            import orbax.checkpoint as ocp
            import jax.numpy as jnp

            path = self.checkpoint_dir / f"checkpoint_{tag}"

            checkpointer = ocp.PyTreeCheckpointer()
            state = {
                "params": params,
                "opt_state": opt_state,
                "epoch": epoch,
            }
            if extra:
                state.update(extra)

            if path.exists():
                shutil.rmtree(path)

            checkpointer.save(str(path), state)
            logger.info(f"Saved JAX checkpoint: {path} (epoch={epoch})")

        except ImportError:
            # Fallback: save as numpy arrays
            import numpy as np
            import jax

            path = self.checkpoint_dir / f"checkpoint_{tag}.npz"
            flat_params = jax.tree_util.tree_leaves(params)
            np.savez(str(path), *[np.array(p) for p in flat_params])
            logger.info(f"Saved JAX checkpoint (npz fallback): {path}")

        # Track
        self._checkpoint_history.append({
            "path": str(path),
            "epoch": epoch,
            "metric": metric,
            "tag": tag,
        })

        if metric is not None and self._is_better(metric):
            self._best_metric = metric
            best_path = self.checkpoint_dir / f"checkpoint_best"
            if best_path.exists():
                shutil.rmtree(best_path)
            shutil.copytree(str(path), str(best_path))
            logger.info(f"New best JAX model: {self.metric_name}={metric:.6f}")

        self._cleanup_old_checkpoints()
        self._save_history()

        return str(path)

    def load_jax(self, tag: str = "latest") -> Dict[str, Any]:
        """
        Load JAX/Flax training state.

        Args:
            tag: Checkpoint tag to load.

        Returns:
            Dict with params, opt_state, epoch.
        """
        try:
            import orbax.checkpoint as ocp

            path = self.checkpoint_dir / f"checkpoint_{tag}"
            if not path.exists():
                raise FileNotFoundError(f"JAX checkpoint not found: {path}")

            checkpointer = ocp.PyTreeCheckpointer()
            state = checkpointer.restore(str(path))
            logger.info(f"Loaded JAX checkpoint: {path}")
            return state

        except ImportError:
            raise ImportError("orbax-checkpoint required for JAX checkpoints")
