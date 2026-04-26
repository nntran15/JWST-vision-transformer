#!/usr/bin/env python3
"""Render side-by-side QA panels for original vs resized JWST thumbnails.

The script groups thumbnails by original size bins, samples a configurable number
of files per bin, and writes one comparison contact sheet per bin so resizing
artifacts are easy to inspect before training.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ORIGINAL_DIR = PROJECT_ROOT / "data" / "JWST" / "original_10k_files"
DEFAULT_RESIZED_DIR = PROJECT_ROOT / "data" / "JWST" / "resized_10k_files"
DEFAULT_MANIFEST_CSV = PROJECT_ROOT / "data" / "JWST" / "resized_10k_files_manifest.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "JWST" / "resize_qa"
DEFAULT_BINS = "20-24,25-30,31-40,41-50,51-64,65-100"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render side-by-side original vs resized QA panels across size bins."
    )
    parser.add_argument("--original-dir", type=Path, default=DEFAULT_ORIGINAL_DIR)
    parser.add_argument("--resized-dir", type=Path, default=DEFAULT_RESIZED_DIR)
    parser.add_argument("--manifest-csv", type=Path, default=DEFAULT_MANIFEST_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--bins",
        default=DEFAULT_BINS,
        help="Comma-separated inclusive size bins, e.g. 20-24,25-30,31-40.",
    )
    parser.add_argument(
        "--samples-per-bin",
        type=int,
        default=8,
        help="Maximum number of thumbnail pairs to render per size bin.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--size-column",
        default="source_width",
        help="Manifest column used to assign thumbnails to size bins.",
    )
    parser.add_argument(
        "--status-column",
        default="status",
        help="Manifest column used to keep only successful rows.",
    )
    parser.add_argument(
        "--percentile-low",
        type=float,
        default=1.0,
        help="Lower percentile used for display clipping.",
    )
    parser.add_argument(
        "--percentile-high",
        type=float,
        default=99.0,
        help="Upper percentile used for display clipping.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing QA outputs.",
    )
    return parser.parse_args()


def parse_bins(spec: str) -> list[tuple[int, int]]:
    bins: list[tuple[int, int]] = []
    for raw_part in spec.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if "-" not in part:
            raise ValueError(f"Invalid bin specification: {part!r}")
        lower_str, upper_str = part.split("-", maxsplit=1)
        lower = int(lower_str)
        upper = int(upper_str)
        if lower > upper:
            raise ValueError(f"Invalid bin with lower > upper: {part!r}")
        bins.append((lower, upper))

    if not bins:
        raise ValueError("At least one size bin must be provided")
    return bins


def load_manifest_rows(manifest_path: Path, status_column: str) -> list[dict[str, str]]:
    with manifest_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    if status_column in reader.fieldnames:
        rows = [row for row in rows if row.get(status_column, "ok") == "ok"]
    return rows


def read_fits_image(file_path: Path) -> np.ndarray:
    with fits.open(file_path, memmap=False) as hdul:
        for hdu in hdul:
            data = hdu.data
            if data is None:
                continue
            array = np.asarray(data)
            if array.ndim >= 2:
                squeezed = np.squeeze(array)
                if squeezed.ndim != 2:
                    raise ValueError(f"Expected 2D image data after squeeze, found shape {array.shape}")
                image = squeezed.astype(np.float32, copy=False)
                if not np.isfinite(image).all():
                    finite_values = image[np.isfinite(image)]
                    fill_value = float(finite_values.mean()) if finite_values.size else 0.0
                    image = np.where(np.isfinite(image), image, fill_value).astype(np.float32, copy=False)
                return image
    raise ValueError(f"No 2D image HDU found in {file_path}")


def compute_display_limits(
    original_image: np.ndarray,
    resized_image: np.ndarray,
    low: float,
    high: float,
) -> tuple[float, float]:
    combined = np.concatenate([original_image.ravel(), resized_image.ravel()])
    finite = combined[np.isfinite(combined)]
    if finite.size == 0:
        return 0.0, 1.0

    vmin = float(np.percentile(finite, low))
    vmax = float(np.percentile(finite, high))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin = float(np.min(finite))
        vmax = float(np.max(finite))
        if vmin == vmax:
            vmax = vmin + 1.0
    return vmin, vmax


def assign_bins(
    rows: list[dict[str, str]], size_column: str, bins: list[tuple[int, int]]
) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = {format_bin_label(bounds): [] for bounds in bins}
    for row in rows:
        raw_value = row.get(size_column)
        if raw_value in (None, ""):
            continue
        size_value = int(float(raw_value))
        for lower, upper in bins:
            if lower <= size_value <= upper:
                grouped[format_bin_label((lower, upper))].append(row)
                break
    return grouped


def format_bin_label(bounds: tuple[int, int]) -> str:
    return f"{bounds[0]}-{bounds[1]}"


def sanitize_bin_label(bin_label: str) -> str:
    return bin_label.replace("-", "_")


def choose_rows(rows: list[dict[str, str]], samples_per_bin: int, seed: int) -> list[dict[str, str]]:
    if len(rows) <= samples_per_bin:
        return list(rows)
    rng = random.Random(seed)
    return rng.sample(rows, samples_per_bin)


def build_caption(row: dict[str, str]) -> str:
    filename = row.get("filename", "unknown")
    filter_name = row.get("filter", "unknown")
    width = row.get("source_width") or row.get("width") or "?"
    height = row.get("source_height") or row.get("height") or "?"
    return f"{filename}\n{width}x{height} | {filter_name}"


def render_bin_panel(
    *,
    rows: list[dict[str, str]],
    bin_label: str,
    original_dir: Path,
    resized_dir: Path,
    output_path: Path,
    percentile_low: float,
    percentile_high: float,
) -> list[dict[str, Any]]:
    figure, axes = plt.subplots(
        nrows=len(rows),
        ncols=2,
        figsize=(8, max(2.75 * len(rows), 4.0)),
        squeeze=False,
    )
    figure.suptitle(
        f"Resize QA: original size bin {bin_label} px", fontsize=14, y=0.995
    )

    selected_rows: list[dict[str, Any]] = []

    for row_index, row in enumerate(rows):
        original_path = original_dir / row["filename"]
        resized_path = resized_dir / row["filename"]
        if not original_path.exists():
            raise FileNotFoundError(f"Original FITS file not found: {original_path}")
        if not resized_path.exists():
            raise FileNotFoundError(f"Resized FITS file not found: {resized_path}")

        original_image = read_fits_image(original_path)
        resized_image = read_fits_image(resized_path)
        vmin, vmax = compute_display_limits(
            original_image, resized_image, percentile_low, percentile_high
        )

        original_axis = axes[row_index][0]
        resized_axis = axes[row_index][1]
        original_axis.imshow(original_image, origin="lower", cmap="gray", vmin=vmin, vmax=vmax)
        resized_axis.imshow(resized_image, origin="lower", cmap="gray", vmin=vmin, vmax=vmax)

        if row_index == 0:
            original_axis.set_title("Original", fontsize=11)
            resized_axis.set_title("Resized", fontsize=11)

        original_axis.set_xlabel(build_caption(row), fontsize=8)
        resized_axis.set_xlabel(f"{resized_image.shape[1]}x{resized_image.shape[0]}", fontsize=8)

        original_axis.set_xticks([])
        original_axis.set_yticks([])
        resized_axis.set_xticks([])
        resized_axis.set_yticks([])

        selected_rows.append(
            {
                "filename": row["filename"],
                "filter": row.get("filter"),
                "source_width": row.get("source_width") or row.get("width"),
                "source_height": row.get("source_height") or row.get("height"),
                "original_path": str(original_path.resolve()),
                "resized_path": str(resized_path.resolve()),
            }
        )

    figure.tight_layout(rect=(0, 0, 1, 0.98))
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return selected_rows


def prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Output directory already exists and is not empty: {output_dir}. Use --overwrite to replace files."
        )
    output_dir.mkdir(parents=True, exist_ok=True)


def write_index(index_payload: dict[str, Any], output_dir: Path) -> Path:
    output_path = output_dir / "qa_index.json"
    with output_path.open("w") as handle:
        json.dump(index_payload, handle, indent=2, sort_keys=True)
    return output_path


def main() -> None:
    args = parse_args()
    args.original_dir = args.original_dir.resolve()
    args.resized_dir = args.resized_dir.resolve()
    args.manifest_csv = args.manifest_csv.resolve()
    args.output_dir = args.output_dir.resolve()

    if args.samples_per_bin <= 0:
        raise ValueError("--samples-per-bin must be positive")
    if not 0.0 <= args.percentile_low < args.percentile_high <= 100.0:
        raise ValueError("Display percentiles must satisfy 0 <= low < high <= 100")
    if not args.original_dir.exists():
        raise FileNotFoundError(f"Original directory does not exist: {args.original_dir}")
    if not args.resized_dir.exists():
        raise FileNotFoundError(f"Resized directory does not exist: {args.resized_dir}")
    if not args.manifest_csv.exists():
        raise FileNotFoundError(f"Manifest CSV does not exist: {args.manifest_csv}")

    prepare_output_dir(args.output_dir, args.overwrite)
    bins = parse_bins(args.bins)
    rows = load_manifest_rows(args.manifest_csv, args.status_column)
    if not rows:
        raise RuntimeError(f"No usable rows found in manifest: {args.manifest_csv}")
    if args.size_column not in rows[0]:
        raise KeyError(f"Size column {args.size_column!r} not found in manifest")

    grouped_rows = assign_bins(rows, args.size_column, bins)
    index_payload: dict[str, Any] = {
        "manifest_csv": str(args.manifest_csv),
        "original_dir": str(args.original_dir),
        "resized_dir": str(args.resized_dir),
        "output_dir": str(args.output_dir),
        "samples_per_bin": args.samples_per_bin,
        "size_column": args.size_column,
        "bins": {},
    }

    for bin_bounds in bins:
        bin_label = format_bin_label(bin_bounds)
        candidate_rows = grouped_rows[bin_label]
        selected_rows = choose_rows(candidate_rows, args.samples_per_bin, args.seed + bin_bounds[0])
        if not selected_rows:
            index_payload["bins"][bin_label] = {
                "available_rows": 0,
                "selected_rows": 0,
                "panel_path": None,
                "samples": [],
            }
            continue

        panel_path = args.output_dir / f"resize_qa_bin_{sanitize_bin_label(bin_label)}.png"
        if panel_path.exists() and not args.overwrite:
            raise FileExistsError(
                f"Panel already exists and --overwrite was not provided: {panel_path}"
            )

        sampled_payload = render_bin_panel(
            rows=selected_rows,
            bin_label=bin_label,
            original_dir=args.original_dir,
            resized_dir=args.resized_dir,
            output_path=panel_path,
            percentile_low=args.percentile_low,
            percentile_high=args.percentile_high,
        )
        index_payload["bins"][bin_label] = {
            "available_rows": len(candidate_rows),
            "selected_rows": len(selected_rows),
            "panel_path": str(panel_path.resolve()),
            "samples": sampled_payload,
        }
        print(
            f"Rendered {len(selected_rows):,} pairs for bin {bin_label} -> {panel_path}",
            flush=True,
        )

    index_path = write_index(index_payload, args.output_dir)
    print("\nWrote outputs")
    print("-------------")
    print(f"QA directory:       {args.output_dir}")
    print(f"QA index:           {index_path}")


if __name__ == "__main__":
    main()