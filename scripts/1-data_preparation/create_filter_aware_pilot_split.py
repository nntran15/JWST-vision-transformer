#!/usr/bin/env python3
"""Create a filter-aware train/validation split from a JWST manifest.

The split is deterministic with a seed and keeps common filters represented in
both train and validation sets. Rare filters can be kept in train only so the
validation split is not dominated by singletons.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST_CSV = PROJECT_ROOT / "data" / "JWST" / "resized_10k_files_manifest.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "JWST" / "pilot_split"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a filter-aware train/validation split for the JWST pilot set."
    )
    parser.add_argument("--manifest-csv", type=Path, default=DEFAULT_MANIFEST_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--rare-filter-threshold",
        type=int,
        default=5,
        help="Filters with fewer than this many examples stay in train only.",
    )
    parser.add_argument(
        "--filter-column",
        default="filter",
        help="Manifest column used for filter stratification.",
    )
    parser.add_argument(
        "--status-column",
        default="status",
        help="Manifest column used to keep only successful rows.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting an existing split directory.",
    )
    return parser.parse_args()


def load_rows(manifest_path: Path, status_column: str) -> list[dict[str, str]]:
    with manifest_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    if status_column in reader.fieldnames:
        rows = [row for row in rows if row.get(status_column, "ok") == "ok"]
    return rows


def summarize_filter_counts(rows: list[dict[str, str]], filter_column: str) -> dict[str, int]:
    return dict(sorted(Counter(row.get(filter_column, "unknown") or "unknown" for row in rows).items()))


def compute_val_count(count: int, val_fraction: float, rare_filter_threshold: int) -> int:
    if count < rare_filter_threshold:
        return 0
    proposed = round(count * val_fraction)
    proposed = max(1, proposed)
    if proposed >= count:
        proposed = count - 1
    return max(0, proposed)


def split_rows(
    rows: list[dict[str, str]],
    *,
    filter_column: str,
    val_fraction: float,
    seed: int,
    rare_filter_threshold: int,
) -> tuple[list[dict[str, str]], list[dict[str, str]], dict[str, Any]]:
    by_filter: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        filter_name = row.get(filter_column, "unknown") or "unknown"
        by_filter[filter_name].append(dict(row))

    rng = random.Random(seed)
    train_rows: list[dict[str, str]] = []
    val_rows: list[dict[str, str]] = []
    per_filter_summary: dict[str, Any] = {}

    for filter_name in sorted(by_filter):
        group = by_filter[filter_name]
        rng.shuffle(group)
        val_count = compute_val_count(len(group), val_fraction, rare_filter_threshold)
        val_group = group[:val_count]
        train_group = group[val_count:]

        for row in train_group:
            row["split"] = "train"
            row["split_reason"] = "rare_train_only" if val_count == 0 else "filter_stratified"
        for row in val_group:
            row["split"] = "val"
            row["split_reason"] = "filter_stratified"

        train_rows.extend(train_group)
        val_rows.extend(val_group)
        per_filter_summary[filter_name] = {
            "total": len(group),
            "train": len(train_group),
            "val": len(val_group),
            "rare_train_only": val_count == 0,
        }

    train_rows.sort(key=lambda row: int(row.get("sample_index") or 0))
    val_rows.sort(key=lambda row: int(row.get("sample_index") or 0))

    summary = {
        "train_count": len(train_rows),
        "val_count": len(val_rows),
        "train_filter_counts": summarize_filter_counts(train_rows, filter_column),
        "val_filter_counts": summarize_filter_counts(val_rows, filter_column),
        "per_filter": per_filter_summary,
    }
    return train_rows, val_rows, summary


def write_csv(rows: list[dict[str, str]], output_path: Path) -> None:
    if not rows:
        raise ValueError(f"No rows to write to {output_path}")

    fieldnames = list(rows[0].keys())
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(summary: dict[str, Any], output_path: Path) -> None:
    with output_path.open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)


def prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Output directory already exists and is not empty: {output_dir}. Use --overwrite to replace files."
        )
    output_dir.mkdir(parents=True, exist_ok=True)


def print_summary(summary: dict[str, Any]) -> None:
    print("\nPilot split summary")
    print("-------------------")
    print(f"Train rows:        {summary['train_count']:,}")
    print(f"Validation rows:   {summary['val_count']:,}")
    print(f"Train filters:     {summary['train_filter_counts']}")
    print(f"Validation filters:{summary['val_filter_counts']}")


def main() -> None:
    args = parse_args()
    args.manifest_csv = args.manifest_csv.resolve()
    args.output_dir = args.output_dir.resolve()

    if not args.manifest_csv.exists():
        raise FileNotFoundError(f"Manifest file does not exist: {args.manifest_csv}")
    if not 0.0 < args.val_fraction < 1.0:
        raise ValueError("--val-fraction must be between 0 and 1")
    if args.rare_filter_threshold <= 0:
        raise ValueError("--rare-filter-threshold must be positive")

    prepare_output_dir(args.output_dir, args.overwrite)
    rows = load_rows(args.manifest_csv, args.status_column)
    if not rows:
        raise RuntimeError(f"No usable rows found in manifest: {args.manifest_csv}")
    if args.filter_column not in rows[0]:
        raise KeyError(f"Filter column {args.filter_column!r} not found in manifest")

    train_rows, val_rows, summary = split_rows(
        rows,
        filter_column=args.filter_column,
        val_fraction=args.val_fraction,
        seed=args.seed,
        rare_filter_threshold=args.rare_filter_threshold,
    )

    train_csv = args.output_dir / "train.csv"
    val_csv = args.output_dir / "val.csv"
    summary_json = args.output_dir / "split_summary.json"
    write_csv(train_rows, train_csv)
    write_csv(val_rows, val_csv)

    summary.update(
        {
            "manifest_csv": str(args.manifest_csv),
            "output_dir": str(args.output_dir),
            "val_fraction": args.val_fraction,
            "seed": args.seed,
            "rare_filter_threshold": args.rare_filter_threshold,
        }
    )
    write_summary(summary, summary_json)
    print_summary(summary)

    print("\nWrote outputs")
    print("-------------")
    print(f"Train CSV:         {train_csv}")
    print(f"Validation CSV:    {val_csv}")
    print(f"Summary JSON:      {summary_json}")


if __name__ == "__main__":
    main()