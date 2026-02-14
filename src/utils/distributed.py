#!/usr/bin/env python3
"""
Distributed training utilities for PyTorch DDP and JAX pmap.

Handles:
- PyTorch DistributedDataParallel (DDP) setup with NCCL backend
- JAX distributed initialization and device sharding
- Auto-detection of GPU count and world size
- Gradient synchronization helpers
"""

import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)


# ---- PyTorch Distributed ----

def setup_distributed(
    backend: str = "nccl",
    rank: Optional[int] = None,
    world_size: Optional[int] = None,
    master_addr: str = "localhost",
    master_port: str = "12355",
) -> dict:
    """
    Initialize PyTorch distributed training.

    Supports launch via:
    - torchrun (env vars RANK, WORLD_SIZE, LOCAL_RANK set automatically)
    - Manual specification of rank/world_size

    Args:
        backend: Communication backend ('nccl' for GPU, 'gloo' for CPU).
        rank: Global rank (auto-detected from env if None).
        world_size: Total number of processes (auto-detected if None).
        master_addr: Master node address.
        master_port: Master node port.

    Returns:
        Dict with 'rank', 'local_rank', 'world_size', 'device'.
    """
    import torch
    import torch.distributed as dist

    # Auto-detect from environment (set by torchrun)
    if rank is None:
        rank = int(os.environ.get("RANK", 0))
    if world_size is None:
        world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if world_size > 1:
        os.environ.setdefault("MASTER_ADDR", master_addr)
        os.environ.setdefault("MASTER_PORT", master_port)

        dist.init_process_group(
            backend=backend,
            rank=rank,
            world_size=world_size,
        )

        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")

        logger.info(
            f"Initialized distributed: rank={rank}, "
            f"local_rank={local_rank}, world_size={world_size}"
        )
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Single-process mode, device={device}")

    return {
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "device": device,
    }


def cleanup_distributed():
    """Tear down PyTorch distributed process group."""
    import torch.distributed as dist

    if dist.is_initialized():
        dist.destroy_process_group()
        logger.info("Destroyed distributed process group")


def wrap_ddp(model, local_rank: int, find_unused_parameters: bool = False):
    """
    Wrap a PyTorch model in DistributedDataParallel.

    Args:
        model: nn.Module to wrap.
        local_rank: Local GPU rank.
        find_unused_parameters: Needed for some architectures (e.g., DINO teacher).

    Returns:
        DDP-wrapped model.
    """
    import torch.nn as nn
    from torch.nn.parallel import DistributedDataParallel as DDP

    model = model.to(local_rank)
    model = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=find_unused_parameters,
    )
    return model


def is_main_process(rank: int = 0) -> bool:
    """Check if current process is the main (rank 0) process."""
    return rank == 0


# ---- JAX Distributed ----

def setup_jax_distributed() -> dict:
    """
    Initialize JAX distributed training.

    Returns:
        Dict with 'num_devices', 'process_index', 'process_count'.
    """
    import jax

    # JAX auto-detects GPUs. For multi-host, call jax.distributed.initialize()
    num_devices = jax.local_device_count()
    process_index = jax.process_index()
    process_count = jax.process_count()

    if process_count > 1:
        jax.distributed.initialize()
        logger.info(
            f"JAX distributed: process {process_index}/{process_count}, "
            f"{num_devices} local devices"
        )
    else:
        logger.info(f"JAX single-process with {num_devices} devices")

    return {
        "num_devices": num_devices,
        "process_index": process_index,
        "process_count": process_count,
    }


def get_device_count(framework: str = "pytorch") -> int:
    """
    Get the number of available accelerator devices.

    Args:
        framework: 'pytorch' or 'jax'.

    Returns:
        Number of GPUs/TPUs available.
    """
    if framework == "pytorch":
        import torch
        return torch.cuda.device_count() if torch.cuda.is_available() else 1
    elif framework == "jax":
        import jax
        return jax.local_device_count()
    else:
        return 1
