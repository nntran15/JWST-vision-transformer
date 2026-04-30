"""
Linear probe and fine-tuning classifiers for evaluating SSL representations.

Supports:
  1. Linear probe: Frozen encoder + learned linear head
  2. Fine-tuning: Unfrozen encoder + linear head (lower LR for backbone)
"""

import csv
import json
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
from tqdm import tqdm

from src.data.fits_preprocessing import collapse_to_single_channel, normalize_fits_data, resize_chw

logger = logging.getLogger(__name__)


def _unwrap_dataset(dataset: Dataset) -> Dataset:
    """Return the base dataset if wrapped by torch Subset objects."""
    while isinstance(dataset, Subset):
        dataset = dataset.dataset
    return dataset


def _get_idx_to_label(dataset: Dataset) -> Optional[dict[int, str]]:
    """Extract idx_to_label mapping from a dataset or wrapped subset."""
    base_dataset = _unwrap_dataset(dataset)
    if hasattr(base_dataset, "idx_to_label"):
        return dict(base_dataset.idx_to_label)
    return None


def _normalize_label_name(label_name: str) -> str:
    """Normalize label names into stable metric keys."""
    return label_name.strip().lower().replace("-", "_").replace(" ", "_")


def _compute_classification_metrics(
    targets: np.ndarray,
    preds: np.ndarray,
    idx_to_label: Optional[dict[int, str]] = None,
) -> dict:
    """Compute aggregate and per-class metrics for model selection."""
    from sklearn.metrics import precision_recall_fscore_support

    if idx_to_label is not None:
        labels = sorted(idx_to_label)
    else:
        labels = sorted(set(np.concatenate([targets, preds]).tolist()))

    precisions, recalls, f1s, supports = precision_recall_fscore_support(
        targets,
        preds,
        labels=labels,
        zero_division=0,
    )

    metrics = {
        "accuracy": float((preds == targets).mean()),
        "macro_precision": float(np.mean(precisions)),
        "macro_recall": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1s)),
    }

    for index, label_id in enumerate(labels):
        label_name = idx_to_label.get(label_id, str(label_id)) if idx_to_label is not None else str(label_id)
        label_key = _normalize_label_name(label_name)
        metrics[f"{label_key}_precision"] = float(precisions[index])
        metrics[f"{label_key}_recall"] = float(recalls[index])
        metrics[f"{label_key}_f1"] = float(f1s[index])
        metrics[f"{label_key}_support"] = int(supports[index])

    if f1s.size:
        metrics["min_class_f1"] = float(np.min(f1s))

    return metrics


def _select_score(metrics: dict, selection_metric: str) -> float:
    """Return the metric used to choose the best checkpoint."""
    if selection_metric in metrics:
        return float(metrics[selection_metric])

    logger.warning(
        "Selection metric '%s' is unavailable for this run; falling back to macro_f1.",
        selection_metric,
    )
    return float(metrics.get("macro_f1", metrics.get("accuracy", 0.0)))


def _make_sample_key(file_path: str, cluster_id: str, label: str) -> tuple[str, str, str]:
    """Create a stable sample key used by frozen split artifacts."""
    return (str(Path(file_path).expanduser()), cluster_id.strip(), label.strip())


def _resolve_split_artifact_paths(split_artifact: str | Path) -> tuple[Path, Path]:
    """Return the artifact directory and manifest path for a frozen split."""
    split_path = Path(split_artifact)
    if split_path.is_dir():
        artifact_dir = split_path
        manifest_path = artifact_dir / "split_manifest.json"
    else:
        artifact_dir = split_path.parent
        manifest_path = split_path
    return artifact_dir, manifest_path


def _load_frozen_split_indices(
    full_dataset: "LabeledFITSDataset",
    split_artifact: str | Path,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Load frozen train/val/test indices from a split artifact package."""
    artifact_dir, manifest_path = _resolve_split_artifact_paths(split_artifact)
    if not artifact_dir.exists():
        raise FileNotFoundError(f"Split artifact directory does not exist: {artifact_dir}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Split manifest does not exist: {manifest_path}")

    metadata = json.loads(manifest_path.read_text())
    key_to_indices: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for sample_index, sample_key in enumerate(full_dataset.sample_keys):
        key_to_indices[sample_key].append(sample_index)

    split_indices: dict[str, np.ndarray] = {}
    used_keys: set[tuple[str, str, str]] = set()

    for split_name in ("train", "val", "test"):
        split_csv = artifact_dir / f"{split_name}.csv"
        if not split_csv.exists():
            raise FileNotFoundError(f"Missing split CSV for '{split_name}': {split_csv}")

        indices: list[int] = []
        split_counter: Counter[tuple[str, str, str]] = Counter()

        with split_csv.open(newline="") as handle:
            reader = csv.DictReader(handle)
            expected_columns = {"file_path", "cluster_id", "label"}
            if not expected_columns.issubset(set(reader.fieldnames or [])):
                raise ValueError(
                    f"Split CSV {split_csv} is missing required columns {sorted(expected_columns)}"
                )

            for row in reader:
                label = (row.get("label") or "").strip()
                if label in ("", "label", "unlabeled", "uncertain", "artifact"):
                    raise ValueError(f"Frozen split {split_csv} contains an invalid label row: {row}")

                sample_key = _make_sample_key(
                    row.get("file_path") or "",
                    row.get("cluster_id") or "",
                    label,
                )
                available_indices = key_to_indices.get(sample_key)
                if not available_indices:
                    raise KeyError(
                        f"Split row {row} from {split_csv} was not found in label CSV {full_dataset.csv_path}"
                    )

                sample_position = split_counter[sample_key]
                if sample_position >= len(available_indices):
                    raise ValueError(
                        f"Split CSV {split_csv} references the same sample too many times: {row}"
                    )

                resolved_index = available_indices[sample_position]
                split_counter[sample_key] += 1
                indices.append(resolved_index)
                if sample_key in used_keys:
                    raise ValueError(f"Sample {row} appears in multiple split CSV files under {artifact_dir}")
                used_keys.add(sample_key)

        split_indices[split_name] = np.asarray(indices, dtype=np.int64)

    metadata.update(
        {
            "split_source": "artifact",
            "artifact_dir": str(artifact_dir),
            "manifest_path": str(manifest_path),
        }
    )
    return split_indices, metadata


def _random_partition(
    indices: np.ndarray,
    *,
    holdout_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Fallback random split when stratification is not possible."""
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(indices)
    if holdout_fraction <= 0.0:
        return shuffled, np.asarray([], dtype=np.int64)

    holdout_count = int(round(len(indices) * holdout_fraction))
    if len(indices) > 1:
        holdout_count = max(1, min(holdout_count, len(indices) - 1))
    else:
        holdout_count = 0

    holdout_indices = shuffled[:holdout_count]
    keep_indices = shuffled[holdout_count:]
    return keep_indices, holdout_indices


def _create_dynamic_split_indices(
    label_ids: np.ndarray,
    *,
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Create a dynamic train/val/test split when no frozen artifact is provided."""
    from sklearn.model_selection import train_test_split

    if not 0.0 <= test_fraction < 1.0:
        raise ValueError("test_fraction must be between 0 and 1")
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1")
    if val_fraction + test_fraction >= 1.0:
        raise ValueError("val_fraction + test_fraction must be less than 1")

    indices = np.arange(len(label_ids), dtype=np.int64)
    split_metadata: dict[str, Any] = {
        "split_source": "dynamic",
        "seed": seed,
        "val_fraction": val_fraction,
        "test_fraction": test_fraction,
    }

    test_indices = np.asarray([], dtype=np.int64)
    remaining_indices = indices
    if test_fraction > 0.0:
        try:
            remaining_indices, test_indices = train_test_split(
                indices,
                test_size=test_fraction,
                random_state=seed,
                stratify=label_ids,
            )
            split_metadata["test_stratified"] = True
        except ValueError:
            logger.warning(
                "Falling back to an unstratified test split because the dataset is too small for stable stratification."
            )
            remaining_indices, test_indices = _random_partition(
                indices,
                holdout_fraction=test_fraction,
                seed=seed,
            )
            split_metadata["test_stratified"] = False

    val_indices = np.asarray([], dtype=np.int64)
    train_indices = remaining_indices
    remaining_fraction = 1.0 - test_fraction
    if val_fraction > 0.0 and remaining_fraction > 0.0:
        val_share_of_remaining = val_fraction / remaining_fraction
        try:
            train_indices, val_indices = train_test_split(
                remaining_indices,
                test_size=val_share_of_remaining,
                random_state=seed + 1,
                stratify=label_ids[remaining_indices],
            )
            split_metadata["val_stratified"] = True
        except ValueError:
            logger.warning(
                "Falling back to an unstratified validation split because the dataset is too small for stable stratification."
            )
            train_indices, val_indices = _random_partition(
                remaining_indices,
                holdout_fraction=val_share_of_remaining,
                seed=seed + 1,
            )
            split_metadata["val_stratified"] = False

    return {
        "train": np.asarray(train_indices, dtype=np.int64),
        "val": np.asarray(val_indices, dtype=np.int64),
        "test": np.asarray(test_indices, dtype=np.int64),
    }, split_metadata


# ---------------------------------------------------------------------------
# Labeled dataset
# ---------------------------------------------------------------------------

class LabeledFITSDataset(Dataset):
    """
    Dataset that loads FITS images with labels from a CSV file.

    CSV format: file_path, cluster_id, label
    """

    def __init__(
        self,
        csv_path: str,
        target_size: int = 64,
        label_to_idx: Optional[dict] = None,
    ):
        self.csv_path = str(csv_path)
        self.target_size = target_size
        self.samples = []
        self.sample_rows = []
        self.sample_keys = []
        self.label_names = []
        self.default_channels = 1

        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                file_path = (row.get("file_path") or "").strip()
                cluster_id = (row.get("cluster_id") or "").strip()
                label = row["label"].strip()
                if label in ("", "label", "unlabeled", "uncertain", "artifact"):
                    continue
                self.samples.append((file_path, label))
                self.sample_rows.append(
                    {"file_path": file_path, "cluster_id": cluster_id, "label": label}
                )
                self.sample_keys.append(_make_sample_key(file_path, cluster_id, label))

        # Build label mapping
        if label_to_idx is not None:
            self.label_to_idx = label_to_idx
        else:
            unique_labels = sorted(set(s[1] for s in self.samples))
            self.label_to_idx = {l: i for i, l in enumerate(unique_labels)}

        self.idx_to_label = {v: k for k, v in self.label_to_idx.items()}
        self.n_classes = len(self.label_to_idx)

        logger.info(f"Loaded {len(self.samples)} labeled samples, {self.n_classes} classes")
        for label, idx in sorted(self.label_to_idx.items(), key=lambda x: x[1]):
            count = sum(1 for _, l in self.samples if l == label)
            logger.info(f"  [{idx}] {label}: {count} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        try:
            from astropy.io import fits as fits_io
            with fits_io.open(path, memmap=False) as hdul:
                data = hdul[0].data.astype(np.float32)
                header = hdul[0].header
        except Exception:
            data = np.zeros((self.default_channels, self.target_size, self.target_size), dtype=np.float32)
            header = None

        data = normalize_fits_data(data, header=header, normalization="header")
        data = collapse_to_single_channel(data)
        data = resize_chw(data, self.target_size)
        self.default_channels = data.shape[0]

        tensor = torch.from_numpy(data)
        label_idx = self.label_to_idx[label]

        return tensor, label_idx


# ---------------------------------------------------------------------------
# Classifiers
# ---------------------------------------------------------------------------

class LinearProbe(nn.Module):
    """Frozen encoder + learned linear classification head."""

    def __init__(self, encoder: nn.Module, embed_dim: int, n_classes: int):
        super().__init__()
        self.encoder = encoder
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()
        self.head = nn.Linear(embed_dim, n_classes)

    def train(self, mode: bool = True):
        """Keep the frozen encoder deterministic while training the head."""
        super().train(mode)
        self.encoder.eval()
        return self

    def forward(self, x):
        with torch.no_grad():
            features = self.encoder.get_cls_token(x)
        return self.head(features)


class FineTuner(nn.Module):
    """Unfrozen encoder + linear head for end-to-end fine-tuning."""

    def __init__(self, encoder: nn.Module, embed_dim: int, n_classes: int):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(embed_dim, n_classes)

    def forward(self, x):
        features = self.encoder.get_cls_token(x)
        return self.head(features)


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

def train_classifier(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    report_loader: Optional[DataLoader] = None,
    epochs: int = 50,
    lr: float = 1e-3,
    backbone_lr: Optional[float] = None,
    weight_decay: float = 0.01,
    device: str = "cuda",
    output_dir: str = "output/classifier",
    class_weights: Optional[torch.Tensor] = None,
    selection_metric: str = "macro_f1",
    report_split_name: str = "val",
) -> dict:
    """
    Train a classification model (linear probe or fine-tuner).

    Args:
        model: LinearProbe or FineTuner instance.
        train_loader: Training DataLoader with (image, label) pairs.
        val_loader: Validation DataLoader.
        report_loader: Loader used for the final report (defaults to val_loader).
        epochs: Number of training epochs.
        lr: Learning rate for the head.
        backbone_lr: Learning rate for the encoder backbone (fine-tuning only).
        weight_decay: Weight decay.
        device: Device to train on.
        output_dir: Directory to save checkpoints and metrics.
        class_weights: Optional class weights for imbalanced classification.
        selection_metric: Metric used to choose the best checkpoint.
        report_split_name: Human-readable name of the report split.

    Returns:
        Dict with training history and best metrics.
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    report_loader = report_loader or val_loader

    # Separate param groups for fine-tuning
    if backbone_lr is not None and hasattr(model, "encoder"):
        param_groups = [
            {"params": model.encoder.parameters(), "lr": backbone_lr},
            {"params": model.head.parameters(), "lr": lr},
        ]
    else:
        param_groups = [{"params": model.parameters(), "lr": lr}]

    optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)
    if class_weights is not None:
        class_weights = class_weights.to(device)
        logger.info(f"Using class weights: {class_weights.tolist()}")
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_macro_f1": [],
        "val_macro_recall": [],
    }
    best_selection_score = float("-inf")
    best_val_acc = 0.0
    idx_to_label = _get_idx_to_label(val_loader.dataset)

    if epochs == 0:
        initial_loss, initial_acc, initial_preds, initial_targets = evaluate(model, val_loader, criterion, device)
        initial_metrics = _compute_classification_metrics(initial_targets, initial_preds, idx_to_label)
        best_selection_score = _select_score(initial_metrics, selection_metric)
        best_val_acc = initial_acc
        history["val_loss"].append(initial_loss)
        history["val_acc"].append(initial_acc)
        history["val_macro_f1"].append(initial_metrics.get("macro_f1", 0.0))
        history["val_macro_recall"].append(initial_metrics.get("macro_recall", 0.0))
        if "spiral_f1" in initial_metrics:
            history["val_spiral_f1"] = [initial_metrics["spiral_f1"]]
            history["val_spiral_recall"] = [initial_metrics["spiral_recall"]]
        torch.save(
            {
                "epoch": -1,
                "model_state_dict": model.state_dict(),
                "val_acc": initial_acc,
                "selection_metric": selection_metric,
                "selection_score": best_selection_score,
                "metrics": initial_metrics,
            },
            Path(output_dir) / "best_classifier.pt",
        )

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [train]",
                                   leave=False):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(images)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            train_correct += (logits.argmax(dim=1) == labels).sum().item()
            train_total += images.size(0)

        scheduler.step()

        avg_train_loss = train_loss / train_total
        train_acc = train_correct / train_total

        # Validate
        val_loss, val_acc, val_preds, val_targets = evaluate(model, val_loader, criterion, device)
        epoch_metrics = _compute_classification_metrics(val_targets, val_preds, idx_to_label)
        selection_score = _select_score(epoch_metrics, selection_metric)

        history["train_loss"].append(avg_train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_macro_f1"].append(epoch_metrics.get("macro_f1", 0.0))
        history["val_macro_recall"].append(epoch_metrics.get("macro_recall", 0.0))

        if "spiral_f1" in epoch_metrics and "val_spiral_f1" not in history:
            history["val_spiral_f1"] = []
            history["val_spiral_recall"] = []
        if "spiral_f1" in epoch_metrics:
            history["val_spiral_f1"].append(epoch_metrics["spiral_f1"])
            history["val_spiral_recall"].append(epoch_metrics["spiral_recall"])

        logger.info(
            f"Epoch {epoch+1}/{epochs}: "
            f"train_loss={avg_train_loss:.4f}, train_acc={train_acc:.4f}, "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}, "
            f"val_macro_f1={epoch_metrics.get('macro_f1', 0.0):.4f}, "
            f"selection({selection_metric})={selection_score:.4f}"
        )

        # Save best
        if selection_score > best_selection_score:
            best_selection_score = selection_score
            best_val_acc = val_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_acc": val_acc,
                    "selection_metric": selection_metric,
                    "selection_score": selection_score,
                    "metrics": epoch_metrics,
                },
                Path(output_dir) / "best_classifier.pt",
            )

    # Final classification report
    from sklearn.metrics import classification_report, confusion_matrix

    best_checkpoint = torch.load(
        Path(output_dir) / "best_classifier.pt",
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(best_checkpoint["model_state_dict"])

    report_idx_to_label = _get_idx_to_label(report_loader.dataset) or idx_to_label
    _, _, final_preds, final_targets = evaluate(model, report_loader, criterion, device)

    if hasattr(model, "head") and hasattr(model.head, "out_features"):
        # Build label names from dataset
        if report_idx_to_label is not None:
            target_names = [report_idx_to_label[i] for i in range(model.head.out_features)]
        else:
            target_names = [str(i) for i in range(model.head.out_features)]
        report_labels = list(range(len(target_names)))
    else:
        target_names = None
        report_labels = None

    report = classification_report(
        final_targets,
        final_preds,
        labels=report_labels,
        target_names=target_names,
        zero_division=0,
    )
    cm = confusion_matrix(final_targets, final_preds, labels=report_labels)
    final_metrics = _compute_classification_metrics(final_targets, final_preds, report_idx_to_label)

    logger.info(f"\nClassification Report:\n{report}")
    logger.info(f"\nConfusion Matrix:\n{cm}")

    # Save report
    with open(Path(output_dir) / "classification_report.txt", "w") as f:
        f.write(f"Selection metric: {best_checkpoint.get('selection_metric', selection_metric)}\n")
        f.write(f"Selection score: {best_checkpoint.get('selection_score', best_selection_score):.6f}\n")
        f.write(f"Report split: {report_split_name}\n")
        f.write(f"Best epoch: {int(best_checkpoint.get('epoch', -1))}\n\n")
        f.write(report)
        f.write(f"\nConfusion Matrix:\n{cm}\n")

    return {
        "history": history,
        "best_val_acc": best_val_acc,
        "best_selection_metric": best_checkpoint.get("selection_metric", selection_metric),
        "best_selection_score": best_checkpoint.get("selection_score", best_selection_score),
        "best_epoch": int(best_checkpoint.get("epoch", -1)),
        "best_metrics": best_checkpoint.get("metrics", final_metrics),
        "final_metrics": final_metrics,
        "classification_report": report,
        "confusion_matrix": cm,
    }


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple:
    """Evaluate model, return (loss, accuracy, predictions, targets)."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(images)
            loss = criterion(logits, labels)

            total_loss += loss.item() * images.size(0)
            all_preds.append(logits.argmax(dim=1).cpu().numpy())
            all_targets.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    avg_loss = total_loss / len(all_targets)
    accuracy = (all_preds == all_targets).mean()

    return avg_loss, accuracy, all_preds, all_targets


def create_train_val_loaders(
    csv_path: str,
    val_fraction: float = 0.2,
    batch_size: int = 64,
    num_workers: int = 4,
    seed: int = 42,
    balanced_sampling: bool = False,
) -> tuple:
    """Backward-compatible wrapper that returns only train/val loaders."""
    train_loader, val_loader, _, label_to_idx, class_weights, _ = create_train_val_test_loaders(
        csv_path,
        val_fraction=val_fraction,
        test_fraction=0.0,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed,
        balanced_sampling=balanced_sampling,
    )
    return train_loader, val_loader, label_to_idx, class_weights


def create_train_val_test_loaders(
    csv_path: str,
    val_fraction: float = 0.2,
    test_fraction: float = 0.2,
    batch_size: int = 64,
    num_workers: int = 4,
    seed: int = 42,
    balanced_sampling: bool = False,
    split_artifact: Optional[str] = None,
) -> tuple:
    """Create train/val/test DataLoaders from a label CSV or frozen split artifact."""
    full_dataset = LabeledFITSDataset(csv_path)
    label_ids = np.array(
        [full_dataset.label_to_idx[label] for _, label in full_dataset.samples],
        dtype=np.int64,
    )
    n_classes = len(full_dataset.label_to_idx)

    if split_artifact:
        split_indices, split_metadata = _load_frozen_split_indices(full_dataset, split_artifact)
    else:
        split_indices, split_metadata = _create_dynamic_split_indices(
            label_ids,
            val_fraction=val_fraction,
            test_fraction=test_fraction,
            seed=seed,
        )

    train_indices = split_indices["train"]
    val_indices = split_indices["val"]
    test_indices = split_indices["test"]

    if train_indices.size == 0:
        raise ValueError("Training split is empty.")
    if val_indices.size == 0:
        raise ValueError("Validation split is empty.")

    train_dataset = Subset(full_dataset, train_indices.tolist())
    val_dataset = Subset(full_dataset, val_indices.tolist())
    test_dataset = Subset(full_dataset, test_indices.tolist()) if test_indices.size else None

    train_counts = np.bincount(label_ids[train_indices], minlength=n_classes)
    val_counts = np.bincount(label_ids[val_indices], minlength=n_classes)
    test_counts = np.bincount(label_ids[test_indices], minlength=n_classes) if test_indices.size else np.zeros(n_classes, dtype=np.int64)

    logger.info(f"Train split class counts: {train_counts.tolist()}")
    logger.info(f"Val split class counts: {val_counts.tolist()}")
    if test_indices.size:
        logger.info(f"Test split class counts: {test_counts.tolist()}")

    rebalancing_weights = train_counts.sum() / np.maximum(train_counts, 1)
    rebalancing_weights = rebalancing_weights / rebalancing_weights.mean()
    rebalancing_weights = torch.tensor(rebalancing_weights, dtype=torch.float32)

    sampler = None
    class_weights: Optional[torch.Tensor] = rebalancing_weights
    if balanced_sampling:
        sample_weights = rebalancing_weights[label_ids[train_indices]].double()
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_indices),
            replacement=True,
        )
        class_weights = None
        logger.info(
            "Using weighted random sampling for the training split; skipping loss class weights."
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    split_metadata.update(
        {
            "csv_path": str(csv_path),
            "train_count": int(train_indices.size),
            "val_count": int(val_indices.size),
            "test_count": int(test_indices.size),
            "train_class_counts": train_counts.tolist(),
            "val_class_counts": val_counts.tolist(),
            "test_class_counts": test_counts.tolist(),
        }
    )

    return train_loader, val_loader, test_loader, full_dataset.label_to_idx, class_weights, split_metadata
