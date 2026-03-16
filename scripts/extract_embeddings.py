#!/usr/bin/env python3
"""
Extract CLS token embeddings from a trained SSL encoder.

Runs inference over the full catalog (or a subset) and saves embeddings
to an HDF5 file for downstream use (clustering, classification).

Usage:
  # Extract from a PyTorch MAE checkpoint
  python scripts/extract_embeddings.py \
    --checkpoint output/checkpoints/checkpoint_best.pt \
    --framework timm \
    --method mae \
    --vit_size tiny \
    --output output/embeddings.h5

  # Extract a subset (first 10000 images)
  python scripts/extract_embeddings.py \
    --checkpoint output/checkpoints/checkpoint_best.pt \
    --framework timm \
    --method mae \
    --max_samples 10000 \
    --output output/embeddings_subset.h5

  # JAX checkpoint
  python scripts/extract_embeddings.py \
    --checkpoint output/checkpoints/checkpoint_best \
    --framework jax \
    --method mae \
    --output output/embeddings_jax.h5
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import h5py
from tqdm import tqdm

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)


def extract_pytorch(
    checkpoint_path: str,
    framework: str,
    method: str,
    vit_size: str,
    fits_dataset,
    output_path: str,
    batch_size: int = 256,
    max_samples: int = -1,
    device_str: str = "cuda",
):
    """
    Extract embeddings using a PyTorch checkpoint.

    Loads the encoder from a trained MAE/DINO model, runs inference
    over the dataset, and saves CLS embeddings + metadata to HDF5.
    """
    import torch
    from src.models.vit_config import get_vit_config
    from src.data.pytorch_loader import PyTorchFITSDataset
    from torch.utils.data import DataLoader

    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    # Build ViT config
    vit_config = get_vit_config(size=vit_size, image_size=64, patch_size=8, in_channels=1)

    # Build and load model
    if method == "mae":
        if framework == "timm":
            from src.models.pytorch_timm.mae import MAE
        else:
            from src.models.pytorch_hf.mae import MAE
        model = MAE(config=vit_config)
    elif method == "dino":
        if framework == "timm":
            from src.models.pytorch_timm.dino import DINO
        else:
            from src.models.pytorch_hf.dino import DINO
        model = DINO(config=vit_config)

    # Load checkpoint
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"], strict=False)
    else:
        model.load_state_dict(state, strict=False)

    # Get encoder
    encoder = model.get_encoder()
    encoder = encoder.to(device)
    encoder.eval()

    # Setup dataset
    if max_samples > 0:
        fits_dataset.index = fits_dataset.index[:max_samples]

    dataset = PyTorchFITSDataset(fits_dataset)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True, drop_last=False,
    )

    # Extract embeddings
    all_embeddings = []
    all_paths = []

    logger.info(f"Extracting embeddings from {len(fits_dataset)} images...")

    with torch.no_grad():
        for batch_idx, images in enumerate(tqdm(dataloader, desc="Extracting")):
            images = images.to(device, non_blocking=True)
            cls_embeddings = encoder.get_cls_token(images)  # (B, embed_dim)
            all_embeddings.append(cls_embeddings.cpu().numpy())

            # Track file paths
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(fits_dataset))
            for i in range(start_idx, end_idx):
                all_paths.append(fits_dataset.get_metadata(i)["path"])

    embeddings = np.concatenate(all_embeddings, axis=0)

    # Save to HDF5
    _save_hdf5(output_path, embeddings, all_paths, vit_config, method, framework)

    logger.info(f"Saved {embeddings.shape[0]} embeddings of dim {embeddings.shape[1]} to {output_path}")
    return embeddings


def extract_jax(
    checkpoint_path: str,
    method: str,
    vit_size: str,
    fits_dataset,
    output_path: str,
    batch_size: int = 256,
    max_samples: int = -1,
):
    """Extract embeddings using a JAX checkpoint."""
    import jax
    import jax.numpy as jnp
    from src.models.vit_config import get_vit_config
    from src.data.jax_loader import JAXDataIterator

    vit_config = get_vit_config(size=vit_size, image_size=64, patch_size=8, in_channels=1)

    if method == "mae":
        from src.models.jax_flax.mae import MAE
        model = MAE(config=vit_config)
    elif method == "dino":
        from src.models.jax_flax.dino import DINOStudent
        model = DINOStudent(config=vit_config)

    # Load checkpoint
    from src.utils.checkpointing import CheckpointManager
    ckpt_mgr = CheckpointManager(checkpoint_dir=str(Path(checkpoint_path).parent))
    tag = Path(checkpoint_path).name.replace("checkpoint_", "")
    state = ckpt_mgr.load_jax(tag=tag)
    params = state["params"]

    # Subset
    if max_samples > 0:
        fits_dataset.index = fits_dataset.index[:max_samples]

    # Extract
    data_iter = JAXDataIterator(
        fits_dataset, batch_size=batch_size, shuffle=False, drop_last=False,
    )

    all_embeddings = []
    logger.info(f"Extracting JAX embeddings from {len(fits_dataset)} images...")

    if method == "mae":
        # MAE: use encode-only path (no masking/decoding), extract CLS token
        @jax.jit
        def get_cls(params, x):
            out = model.apply(params, x, method=model.encode, deterministic=True)
            return out[:, 0, :]  # CLS token
    elif method == "dino":
        # DINOStudent: use encode path to get CLS features (not prototype logits)
        @jax.jit
        def get_cls(params, x):
            return model.apply(params, x, method=model.encode, deterministic=True)

    for batch in tqdm(data_iter, desc="Extracting"):
        batch = jnp.array(batch)
        cls = get_cls(params, batch)
        all_embeddings.append(np.array(cls))

    embeddings = np.concatenate(all_embeddings, axis=0)
    all_paths = [fits_dataset.get_metadata(i)["path"] for i in range(len(fits_dataset))]
    _save_hdf5(output_path, embeddings, all_paths, vit_config, method, "jax")

    logger.info(f"Saved {embeddings.shape[0]} JAX embeddings to {output_path}")
    return embeddings


def _save_hdf5(output_path, embeddings, paths, vit_config, method, framework):
    """Save embeddings and metadata to HDF5 file."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, "w") as f:
        f.create_dataset("embeddings", data=embeddings, compression="gzip")
        f.create_dataset(
            "paths",
            data=np.array([p.encode("utf-8") for p in paths], dtype=h5py.special_dtype(vlen=bytes)),
        )

        # Metadata
        f.attrs["embed_dim"] = vit_config.embed_dim
        f.attrs["vit_size"] = vit_config.model_name
        f.attrs["image_size"] = vit_config.image_size
        f.attrs["patch_size"] = vit_config.patch_size
        f.attrs["method"] = method
        f.attrs["framework"] = framework
        f.attrs["n_samples"] = embeddings.shape[0]


def main():
    parser = argparse.ArgumentParser(description="Extract SSL embeddings")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint path")
    parser.add_argument("--framework", type=str, required=True, choices=["timm", "hf", "jax"])
    parser.add_argument("--method", type=str, required=True, choices=["mae", "dino"])
    parser.add_argument("--vit_size", type=str, default="tiny", choices=["tiny", "small", "base"])
    parser.add_argument("--output", type=str, default="output/embeddings.h5")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_samples", type=int, default=-1, help="-1 for all")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--catalog_dir", type=str,
                        default="../JWSP-JWST-to-SpArcFiRe/outputs/JWST_galaxy_thumbnails")
    parser.add_argument("--index_path", type=str, default="output/file_index.json")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    from src.data.fits_dataset import FITSDataset, build_file_index

    index = build_file_index(
        catalog_dir=args.catalog_dir,
        index_path=args.index_path,
    )
    fits_dataset = FITSDataset(index=index, target_size=64)

    if args.framework in ("timm", "hf"):
        extract_pytorch(
            checkpoint_path=args.checkpoint,
            framework=args.framework,
            method=args.method,
            vit_size=args.vit_size,
            fits_dataset=fits_dataset,
            output_path=args.output,
            batch_size=args.batch_size,
            max_samples=args.max_samples,
            device_str=args.device,
        )
    else:
        extract_jax(
            checkpoint_path=args.checkpoint,
            method=args.method,
            vit_size=args.vit_size,
            fits_dataset=fits_dataset,
            output_path=args.output,
            batch_size=args.batch_size,
            max_samples=args.max_samples,
        )


if __name__ == "__main__":
    main()
