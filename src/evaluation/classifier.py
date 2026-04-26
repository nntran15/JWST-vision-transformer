"""
Linear probe and fine-tuning classifiers for evaluating SSL representations.

Supports:
  1. Linear probe: Frozen encoder + learned linear head
  2. Fine-tuning: Unfrozen encoder + linear head (lower LR for backbone)
"""

import csv
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from src.data.fits_preprocessing import normalize_fits_data, resize_chw

logger = logging.getLogger(__name__)


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
        self.target_size = target_size
        self.samples = []
        self.label_names = []
        self.default_channels = 1

        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["label"] not in ("unlabeled", "uncertain", "artifact"):
                    self.samples.append((row["file_path"], row["label"]))

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
        self.head = nn.Linear(embed_dim, n_classes)

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
    epochs: int = 50,
    lr: float = 1e-3,
    backbone_lr: Optional[float] = None,
    weight_decay: float = 0.01,
    device: str = "cuda",
    output_dir: str = "output/classifier",
) -> dict:
    """
    Train a classification model (linear probe or fine-tuner).

    Args:
        model: LinearProbe or FineTuner instance.
        train_loader: Training DataLoader with (image, label) pairs.
        val_loader: Validation DataLoader.
        epochs: Number of training epochs.
        lr: Learning rate for the head.
        backbone_lr: Learning rate for the encoder backbone (fine-tuning only).
        weight_decay: Weight decay.
        device: Device to train on.
        output_dir: Directory to save checkpoints and metrics.

    Returns:
        Dict with training history and best metrics.
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Separate param groups for fine-tuning
    if backbone_lr is not None and hasattr(model, "encoder"):
        param_groups = [
            {"params": model.encoder.parameters(), "lr": backbone_lr},
            {"params": model.head.parameters(), "lr": lr},
        ]
    else:
        param_groups = [{"params": model.parameters(), "lr": lr}]

    optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0

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

        history["train_loss"].append(avg_train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        logger.info(
            f"Epoch {epoch+1}/{epochs}: "
            f"train_loss={avg_train_loss:.4f}, train_acc={train_acc:.4f}, "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
        )

        # Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {"epoch": epoch, "model_state_dict": model.state_dict(), "val_acc": val_acc},
                Path(output_dir) / "best_classifier.pt",
            )

    # Final classification report
    from sklearn.metrics import classification_report, confusion_matrix

    _, _, final_preds, final_targets = evaluate(model, val_loader, criterion, device)

    if hasattr(model, "head") and hasattr(model.head, "out_features"):
        # Build label names from dataset
        if hasattr(val_loader.dataset, "idx_to_label"):
            target_names = [val_loader.dataset.idx_to_label[i]
                            for i in range(model.head.out_features)]
        else:
            target_names = [str(i) for i in range(model.head.out_features)]
    else:
        target_names = None

    report = classification_report(final_targets, final_preds, target_names=target_names)
    cm = confusion_matrix(final_targets, final_preds)

    logger.info(f"\nClassification Report:\n{report}")
    logger.info(f"\nConfusion Matrix:\n{cm}")

    # Save report
    with open(Path(output_dir) / "classification_report.txt", "w") as f:
        f.write(report)
        f.write(f"\nConfusion Matrix:\n{cm}\n")

    return {
        "history": history,
        "best_val_acc": best_val_acc,
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
) -> tuple:
    """Create train/val DataLoaders from a labeled CSV."""
    full_dataset = LabeledFITSDataset(csv_path)
    n_val = int(len(full_dataset) * val_fraction)
    n_train = len(full_dataset) - n_val

    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [n_train, n_val], generator=generator
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    return train_loader, val_loader, full_dataset.label_to_idx
