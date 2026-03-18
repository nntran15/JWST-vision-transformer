#!/usr/bin/env python3
"""
Smoke test: verify all modules load and produce correct output shapes.

Run this before training to catch import errors, shape mismatches, and
interface bugs. Tests all 9 configurations (3 SSL × 3 frameworks).

Usage:
  python scripts/smoke_test.py                  # All tests
  python scripts/smoke_test.py --pytorch-only   # Skip JAX tests
  python scripts/smoke_test.py --jax-only       # Skip PyTorch tests
"""

import argparse
import sys
import traceback
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"
SKIP = "\033[93m⊘\033[0m"

results = {"passed": 0, "failed": 0, "skipped": 0}


def test(name, fn, skip=False):
    """Run a test function and report pass/fail."""
    if skip:
        print(f"  {SKIP} {name} (skipped)")
        results["skipped"] += 1
        return
    try:
        fn()
        print(f"  {PASS} {name}")
        results["passed"] += 1
    except Exception as e:
        print(f"  {FAIL} {name}")
        traceback.print_exc()
        print()
        results["failed"] += 1


# ===========================================================================
# 1. Config & data pipeline
# ===========================================================================

def test_config_loading():
    """Test YAML config loading and deep merge."""
    from scripts.train import load_config
    config = load_config(str(project_root / "configs" / "mae.yaml"), {})
    assert config["ssl"]["method"] == "mae"
    assert config["data"]["image_size"] == 64
    assert config["model"]["patch_size"] == 8

    # Test override
    config2 = load_config(
        str(project_root / "configs" / "mae.yaml"),
        {"model.vit_size": "base", "data.batch_size": 32},
    )
    assert config2["model"]["vit_size"] == "base"
    assert config2["data"]["batch_size"] == 32


def test_vit_config():
    """Test ViT config dataclass for all sizes."""
    from src.models.vit_config import get_vit_config

    for size, dim, heads in [("tiny", 192, 3), ("small", 384, 6), ("base", 768, 12)]:
        cfg = get_vit_config(size=size, image_size=64, patch_size=8, in_channels=1)
        assert cfg.embed_dim == dim, f"{size}: embed_dim={cfg.embed_dim}, expected {dim}"
        assert cfg.num_heads == heads
        assert cfg.num_patches == 64  # (64/8)^2
        assert cfg.in_channels == 1


def test_fits_dataset():
    """Test FITSDataset with synthetic data (no real FITS files needed)."""
    import numpy as np
    from src.data.fits_dataset import FITSDataset

    # Create a fake index
    fake_index = []
    ds = FITSDataset(index=fake_index, target_size=64)
    assert len(ds) == 0

    # Test resize method directly
    img = np.random.rand(32, 48).astype(np.float32)
    resized = ds.resize(img, 64)
    assert resized.shape == (64, 64)


def test_augmentations():
    """Test augmentations produce correct shapes."""
    import numpy as np
    from src.data.augmentations import AstronomyAugmentations, DINOMultiCropAugmentation

    aug = AstronomyAugmentations()
    img = np.random.rand(1, 64, 64).astype(np.float32)
    out = aug(img)
    assert out.shape == (1, 64, 64), f"Augmented shape: {out.shape}"

    multi = DINOMultiCropAugmentation(
        global_crop_size=64, local_crop_size=32, n_local_crops=4,
    )
    globals_, locals_ = multi(img)
    assert len(globals_) == 2
    assert len(locals_) == 4
    assert globals_[0].shape == (1, 64, 64)
    assert locals_[0].shape == (1, 32, 32)


# ===========================================================================
# 2. PyTorch models
# ===========================================================================

def test_timm_vit():
    """Test timm ViT wrapper forward pass."""
    import torch
    from src.models.vit_config import get_vit_config
    from src.models.pytorch_timm.vit import build_vit

    cfg = get_vit_config(size="tiny", image_size=64, patch_size=8, in_channels=1)
    model = build_vit(cfg)
    x = torch.randn(2, 1, 64, 64)

    features = model.forward_features(x)
    assert features.shape == (2, 65, 192), f"timm features: {features.shape}"

    cls = model.get_cls_token(x)
    assert cls.shape == (2, 192), f"timm CLS: {cls.shape}"


def test_hf_vit():
    """Test HuggingFace ViT wrapper forward pass."""
    import torch
    from src.models.vit_config import get_vit_config
    from src.models.pytorch_hf.vit import build_vit

    cfg = get_vit_config(size="tiny", image_size=64, patch_size=8, in_channels=1)
    model = build_vit(cfg)
    x = torch.randn(2, 1, 64, 64)

    features = model.forward_features(x)
    assert features.shape == (2, 65, 192), f"hf features: {features.shape}"

    cls = model.get_cls_token(x)
    assert cls.shape == (2, 192), f"hf CLS: {cls.shape}"


def test_timm_mae():
    """Test timm MAE forward pass."""
    import torch
    from src.models.vit_config import get_vit_config
    from src.models.pytorch_timm.mae import MAE

    cfg = get_vit_config(size="tiny", image_size=64, patch_size=8, in_channels=1)
    mae = MAE(config=cfg, mask_ratio=0.75)
    x = torch.randn(2, 1, 64, 64)

    out = mae(x)
    assert "loss" in out
    assert "pred" in out
    assert "mask" in out
    assert out["pred"].shape[0] == 2
    assert out["pred"].shape[1] == 64  # num_patches
    assert out["mask"].sum() > 0  # some patches masked

    # Test get_encoder interface
    enc = mae.get_encoder()
    cls = enc.get_cls_token(x)
    assert cls.shape == (2, 192)


def test_hf_mae():
    """Test HuggingFace MAE forward pass."""
    import torch
    from src.models.vit_config import get_vit_config
    from src.models.pytorch_hf.mae import MAE

    cfg = get_vit_config(size="tiny", image_size=64, patch_size=8, in_channels=1)
    mae = MAE(config=cfg)
    x = torch.randn(2, 1, 64, 64)

    out = mae(x)
    assert "loss" in out

    enc = mae.get_encoder()
    cls = enc.get_cls_token(x)
    assert cls.shape == (2, 192)


def test_timm_dino():
    """Test timm DINO forward pass."""
    import torch
    from src.models.vit_config import get_vit_config
    from src.models.pytorch_timm.dino import DINO

    cfg = get_vit_config(size="tiny", image_size=64, patch_size=8, in_channels=1)
    dino = DINO(config=cfg, out_dim=256)  # Small out_dim for speed
    x_global = [torch.randn(2, 1, 64, 64), torch.randn(2, 1, 64, 64)]
    x_local = [torch.randn(2, 1, 32, 32)]

    out = dino(x_global, x_local)
    assert "loss" in out

    # Test EMA update
    dino.update_teacher(momentum=0.996)

    # Test get_encoder
    enc = dino.get_encoder()
    cls = enc.get_cls_token(torch.randn(2, 1, 64, 64))
    assert cls.shape == (2, 192)


def test_hf_dino():
    """Test HuggingFace DINO forward pass."""
    import torch
    from src.models.vit_config import get_vit_config
    from src.models.pytorch_hf.dino import DINO

    cfg = get_vit_config(size="tiny", image_size=64, patch_size=8, in_channels=1)
    dino = DINO(config=cfg, out_dim=256)
    x_global = [torch.randn(2, 1, 64, 64), torch.randn(2, 1, 64, 64)]
    x_local = [torch.randn(2, 1, 32, 32)]

    out = dino(x_global, x_local)
    assert "loss" in out

    enc = dino.get_encoder()
    cls = enc.get_cls_token(torch.randn(2, 1, 64, 64))
    assert cls.shape == (2, 192)


# ===========================================================================
# 3. JAX models
# ===========================================================================

def test_jax_vit():
    """Test Flax ViT forward pass."""
    import jax
    import jax.numpy as jnp
    from src.models.vit_config import get_vit_config
    from src.models.jax_flax.vit import VisionTransformer

    cfg = get_vit_config(size="tiny", image_size=64, patch_size=8, in_channels=1)
    model = VisionTransformer(config=cfg)
    rng = jax.random.PRNGKey(0)
    x = jnp.ones((2, 1, 64, 64))

    params = model.init(rng, x)
    out = model.apply(params, x, deterministic=True)
    assert out.shape == (2, 65, 192), f"JAX ViT out: {out.shape}"


def test_jax_mae():
    """Test Flax MAE forward + encode-only pass."""
    import jax
    import jax.numpy as jnp
    from src.models.vit_config import get_vit_config
    from src.models.jax_flax.mae import MAE

    cfg = get_vit_config(size="tiny", image_size=64, patch_size=8, in_channels=1)
    model = MAE(config=cfg, mask_ratio=0.75)
    rng = jax.random.PRNGKey(0)
    x = jnp.ones((2, 1, 64, 64))

    params = model.init({"params": rng, "dropout": rng}, x, rng)
    out = model.apply(params, x, rng)
    assert "loss" in out
    assert "pred" in out
    assert "mask" in out

    # Test encode-only path
    enc_out = model.apply(params, x, method=model.encode, deterministic=True)
    assert enc_out.shape == (2, 65, 192), f"JAX MAE encode: {enc_out.shape}"
    cls = enc_out[:, 0, :]
    assert cls.shape == (2, 192)


def test_jax_dino():
    """Test Flax DINO forward and training step."""
    import jax
    import jax.numpy as jnp
    from src.models.vit_config import get_vit_config
    from src.models.jax_flax.dino import DINO, DINOStudent

    cfg = get_vit_config(size="tiny", image_size=64, patch_size=8, in_channels=1)

    # Test DINOStudent directly
    student = DINOStudent(config=cfg, head_out_dim=256)
    rng = jax.random.PRNGKey(0)
    x = jnp.ones((2, 1, 64, 64))
    params = student.init(rng, x)

    logits = student.apply(params, x, deterministic=True)
    assert logits.shape == (2, 256), f"DINOStudent logits: {logits.shape}"

    # Test encode path
    cls = student.apply(params, x, method=student.encode, deterministic=True)
    assert cls.shape == (2, 192), f"DINOStudent encode: {cls.shape}"

    # Test DINO wrapper
    dino = DINO(config=cfg, out_dim=256)
    s_params, t_params, center = dino.init(rng, x)

    gc = [jnp.ones((2, 1, 64, 64)), jnp.ones((2, 1, 64, 64))]
    lc = [jnp.ones((2, 1, 32, 32))]
    loss, aux = dino.loss_fn(
        s_params, t_params, gc, lc, center,
        deterministic=False, rng=jax.random.PRNGKey(1),
    )
    assert loss.shape == (), f"DINO loss shape: {loss.shape}"
    assert "teacher_output" in aux


# ===========================================================================
# 4. Training infrastructure
# ===========================================================================

def test_checkpointing():
    """Test checkpoint manager save/load round-trip."""
    import tempfile
    import torch
    from src.utils.checkpointing import CheckpointManager

    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = CheckpointManager(checkpoint_dir=tmpdir, max_checkpoints=3)
        model = torch.nn.Linear(10, 5)
        optimizer = torch.optim.Adam(model.parameters())

        mgr.save_pytorch(model, optimizer, None, epoch=1, metric=0.5, tag="test")
        loaded = mgr.load_pytorch(model, optimizer, tag="test")
        assert loaded["epoch"] == 1


def test_logging():
    """Test logging utilities."""
    from src.utils.logging_utils import MetricTracker, WandbLogger

    tracker = MetricTracker()
    tracker.update({"loss": 1.0})
    tracker.update({"loss": 2.0})
    avg = tracker.average()
    assert abs(avg["loss"] - 1.5) < 1e-6

    # W&B logger with enabled=False should not crash
    wl = WandbLogger(project="test", enabled=False)
    wl.log_metrics({"a": 1}, step=0)
    wl.finish()


# ===========================================================================
# 5. PyTorch data loaders
# ===========================================================================

def test_pytorch_dataloader():
    """Test PyTorch dataloader with synthetic dataset."""
    import numpy as np
    import torch
    from src.data.fits_dataset import FITSDataset
    from src.data.pytorch_loader import PyTorchFITSDataset

    # Mock FITSDataset
    class FakeDataset:
        def __init__(self):
            self.index = [{"path": f"fake_{i}.fits"} for i in range(8)]
        def __len__(self):
            return len(self.index)
        def __getitem__(self, idx):
            return np.random.rand(1, 64, 64).astype(np.float32)
        def get_metadata(self, idx):
            return self.index[idx]

    ds = PyTorchFITSDataset(FakeDataset())
    loader = torch.utils.data.DataLoader(ds, batch_size=4)
    batch = next(iter(loader))
    assert batch.shape == (4, 1, 64, 64)


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="Smoke test")
    parser.add_argument("--pytorch-only", action="store_true")
    parser.add_argument("--jax-only", action="store_true")
    args = parser.parse_args()

    skip_pt = args.jax_only
    skip_jax = args.pytorch_only

    print("\n" + "=" * 60)
    print("SSL ViT — SMOKE TEST")
    print("=" * 60)

    # Check which packages are available
    try:
        import torch
        has_torch = True
    except (ImportError, RuntimeError) as e:
        has_torch = False
        print(f"\n  {SKIP} PyTorch unavailable ({e.__class__.__name__}) — skipping PyTorch tests")

    try:
        import jax
        has_jax = True
    except (ImportError, RuntimeError) as e:
        has_jax = False
        print(f"\n  {SKIP} JAX unavailable ({e.__class__.__name__}) — skipping JAX tests")

    print("\n--- Config & Data Pipeline ---")
    test("YAML config loading", test_config_loading)
    test("ViT config dataclass", test_vit_config)
    test("FITSDataset (synthetic)", test_fits_dataset)
    test("Augmentations", test_augmentations)

    print("\n--- PyTorch Models ---")
    pt_skip = skip_pt or not has_torch
    test("timm ViT forward", test_timm_vit, skip=pt_skip)
    test("HuggingFace ViT forward", test_hf_vit, skip=pt_skip)
    test("timm MAE forward + encoder", test_timm_mae, skip=pt_skip)
    test("HuggingFace MAE forward + encoder", test_hf_mae, skip=pt_skip)
    test("timm DINO forward + encoder", test_timm_dino, skip=pt_skip)
    test("HuggingFace DINO forward + encoder", test_hf_dino, skip=pt_skip)

    print("\n--- JAX/Flax Models ---")
    jax_skip = skip_jax or not has_jax
    test("Flax ViT forward", test_jax_vit, skip=jax_skip)
    test("Flax MAE forward + encode", test_jax_mae, skip=jax_skip)
    test("Flax DINO forward + encode", test_jax_dino, skip=jax_skip)

    print("\n--- Training Infrastructure ---")
    test("Checkpoint save/load", test_checkpointing, skip=pt_skip)
    test("Logging utilities", test_logging)

    print("\n--- Data Loaders ---")
    test("PyTorch DataLoader", test_pytorch_dataloader, skip=pt_skip)

    # Summary
    total = results["passed"] + results["failed"] + results["skipped"]
    print(f"\n{'=' * 60}")
    print(f"Results: {results['passed']}/{total} passed, "
          f"{results['failed']} failed, {results['skipped']} skipped")
    print(f"{'=' * 60}\n")

    return 0 if results["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
