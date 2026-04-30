#!/usr/bin/env python3
"""Create a frozen train/val/test split artifact for manual spiral labels."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LABELS_CSV = PROJECT_ROOT / "output" / "manual_labels" / "spiral_vs_not_spiral.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "JWST" / "splits" / "manual_spiral_v1"
ALLOWED_LABELS = {"spiral", "not_spiral"}
EXCLUDED_LABELS = {"", "label", "unlabeled", "uncertain", "artifact"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a frozen train/val/test split artifact for manual spiral labels."
    )
    parser.add_argument("--labels-csv", type=Path, default=DEFAULT_LABELS_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting an existing split directory.",
    )
    return parser.parse_args()


def prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Output directory already exists and is not empty: {output_dir}. Use --overwrite to replace files."
        )
    output_dir.mkdir(parents=True, exist_ok=True)


def load_and_clean_rows(labels_csv: Path) -> tuple[list[dict[str, str]], dict[str, Any]]:
    with labels_csv.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required_columns = {"file_path", "cluster_id", "label"}
        if not required_columns.issubset(set(reader.fieldnames or [])):
            raise KeyError(f"Labels CSV must contain columns {sorted(required_columns)}")

        cleaned_rows: list[dict[str, str]] = []
        seen_labels: dict[tuple[str, str], str] = {}
        skipped_counts: Counter[str] = Counter()
        duplicate_rows = 0

        for source_row_index, row in enumerate(reader, start=2):
            file_path = (row.get("file_path") or "").strip()
            cluster_id = (row.get("cluster_id") or "").strip()
            label = (row.get("label") or "").strip().lower()

            if not file_path:
                skipped_counts["missing_file_path"] += 1
                continue
            if label in EXCLUDED_LABELS:
                skipped_counts[f"excluded:{label or 'empty'}"] += 1
                continue
            if label not in ALLOWED_LABELS:
                skipped_counts[f"unsupported:{label}"] += 1
                continue

            normalized_path = str(Path(file_path).expanduser())
            sample_key = (normalized_path, cluster_id)
            previous_label = seen_labels.get(sample_key)
            if previous_label is not None:
                if previous_label != label:
                    raise ValueError(
                        "Conflicting duplicate labels found for "
                        f"{normalized_path} (cluster_id={cluster_id!r}): {previous_label!r} vs {label!r}"
                    )
                duplicate_rows += 1
                continue

            seen_labels[sample_key] = label
            cleaned_rows.append(
                {
                    "file_path": normalized_path,
                    "cluster_id": cluster_id,
                    "label": label,
                    "source_row_index": str(source_row_index),
                }
            )

    if not cleaned_rows:
        raise RuntimeError(f"No usable labels found in {labels_csv}")

    label_counts = Counter(row["label"] for row in cleaned_rows)
    if set(label_counts) != ALLOWED_LABELS:
        raise ValueError(
            f"Expected both labels {sorted(ALLOWED_LABELS)} in {labels_csv}, got {dict(label_counts)}"
        )

    metadata = {
        "usable_rows": len(cleaned_rows),
        "duplicate_rows_removed": duplicate_rows,
        "skipped_rows": dict(sorted(skipped_counts.items())),
        "label_counts": dict(sorted(label_counts.items())),
    }
    return cleaned_rows, metadata


def random_partition(
    indices: np.ndarray,
    *,
    holdout_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(indices)
    holdout_count = int(round(len(indices) * holdout_fraction))
    if len(indices) > 1:
        holdout_count = max(1, min(holdout_count, len(indices) - 1))
    else:
        holdout_count = 0

    holdout_indices = shuffled[:holdout_count]
    keep_indices = shuffled[holdout_count:]
    return keep_indices, holdout_indices


def split_rows(
    rows: list[dict[str, str]],
    *,
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> tuple[dict[str, list[dict[str, str]]], dict[str, Any]]:
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("--val-fraction must be between 0 and 1")
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("--test-fraction must be between 0 and 1")
    if val_fraction + test_fraction >= 1.0:
        raise ValueError("--val-fraction + --test-fraction must be less than 1")

    labels = np.array([row["label"] for row in rows])
    label_counts = Counter(labels.tolist())
    if min(label_counts.values()) < 3:
        raise ValueError(
            "Need at least 3 examples per class to create train/val/test splits with stratification."
        )

    indices = np.arange(len(rows), dtype=np.int64)
    test_stratified = True
    try:
        train_val_indices, test_indices = train_test_split(
            indices,
            test_size=test_fraction,
            random_state=seed,
            stratify=labels,
        )
    except ValueError:
        train_val_indices, test_indices = random_partition(
            indices,
            holdout_fraction=test_fraction,
            seed=seed,
        )
        test_stratified = False

    val_share_of_remaining = val_fraction / (1.0 - test_fraction)
    val_stratified = True
    try:
        train_indices, val_indices = train_test_split(
            train_val_indices,
            test_size=val_share_of_remaining,
            random_state=seed + 1,
            stratify=labels[train_val_indices],
        )
    except ValueError:
        train_indices, val_indices = random_partition(
            train_val_indices,
            holdout_fraction=val_share_of_remaining,
            seed=seed + 1,
        )
        val_stratified = False

    split_rows_map = {
        "train": [rows[index] for index in sorted(train_indices.tolist())],
        "val": [rows[index] for index in sorted(val_indices.tolist())],
        "test": [rows[index] for index in sorted(test_indices.tolist())],
    }

    summary = {
        "train_count": len(split_rows_map["train"]),
        "val_count": len(split_rows_map["val"]),
        "test_count": len(split_rows_map["test"]),
        "test_stratified": test_stratified,
        "val_stratified": val_stratified,
        "train_label_counts": dict(sorted(Counter(row["label"] for row in split_rows_map["train"]).items())),
        "val_label_counts": dict(sorted(Counter(row["label"] for row in split_rows_map["val"]).items())),
        "test_label_counts": dict(sorted(Counter(row["label"] for row in split_rows_map["test"]).items())),
    }
    return split_rows_map, summary


def write_csv(rows: list[dict[str, str]], output_path: Path) -> None:
    if not rows:
        raise ValueError(f"No rows to write to {output_path}")

    fieldnames = ["file_path", "cluster_id", "label", "source_row_index"]
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(summary: dict[str, Any], output_path: Path) -> None:
    with output_path.open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)


def print_summary(summary: dict[str, Any]) -> None:
    print("\nFrozen manual spiral split summary")
    print("---------------------------------")
    print(f"Train rows:      {summary['train_count']:,}")
    print(f"Validation rows: {summary['val_count']:,}")
    print(f"Test rows:       {summary['test_count']:,}")
    print(f"Train labels:    {summary['train_label_counts']}")
    print(f"Validation:      {summary['val_label_counts']}")
    print(f"Test labels:     {summary['test_label_counts']}")


def main() -> None:
    args = parse_args()
    args.labels_csv = args.labels_csv.resolve()
    args.output_dir = args.output_dir.resolve()

    if not args.labels_csv.exists():
        raise FileNotFoundError(f"Labels CSV does not exist: {args.labels_csv}")

    prepare_output_dir(args.output_dir, args.overwrite)
    cleaned_rows, cleaning_summary = load_and_clean_rows(args.labels_csv)
    split_rows_map, split_summary = split_rows(
        cleaned_rows,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
    )

    train_csv = args.output_dir / "train.csv"
    val_csv = args.output_dir / "val.csv"
    test_csv = args.output_dir / "test.csv"
    labels_snapshot_csv = args.output_dir / "labels_snapshot.csv"
    manifest_json = args.output_dir / "split_manifest.json"

    write_csv(split_rows_map["train"], train_csv)
    write_csv(split_rows_map["val"], val_csv)
    write_csv(split_rows_map["test"], test_csv)
    write_csv(cleaned_rows, labels_snapshot_csv)

    manifest = {
        "labels_csv": str(args.labels_csv),
        "output_dir": str(args.output_dir),
        "seed": args.seed,
        "train_fraction": 1.0 - args.val_fraction - args.test_fraction,
        "val_fraction": args.val_fraction,
        "test_fraction": args.test_fraction,
        **cleaning_summary,
        **split_summary,
        "files": {
            "train_csv": train_csv.name,
            "val_csv": val_csv.name,
            "test_csv": test_csv.name,
            "labels_snapshot_csv": labels_snapshot_csv.name,
        },
    }
    write_summary(manifest, manifest_json)
    print_summary(manifest)

    print("\nWrote outputs")
    print("-------------")
    print(f"Train CSV:       {train_csv}")
    print(f"Validation CSV:  {val_csv}")
    print(f"Test CSV:        {test_csv}")
    print(f"Label snapshot:  {labels_snapshot_csv}")
    print(f"Manifest JSON:   {manifest_json}")


if __name__ == "__main__":
    main()