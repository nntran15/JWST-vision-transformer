#!/usr/bin/env python3
"""Resize the original JWST 10k subset to a fixed square FITS size.

The script reads FITS files from the sampled 10k subset, rescales each image to a
target square resolution, writes resized FITS files to a new directory, and emits
an updated manifest plus summary statistics for the resized dataset.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as torch_f
from astropy.io import fits


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "JWST" / "original_10k_files"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "JWST" / "resized_10k_files"
DEFAULT_MANIFEST_CSV = PROJECT_ROOT / "data" / "JWST" / "original_10k_files_manifest.csv"
DEFAULT_OUTPUT_MANIFEST = PROJECT_ROOT / "data" / "JWST" / "resized_10k_files_manifest.csv"
DEFAULT_OUTPUT_STATS = PROJECT_ROOT / "data" / "JWST" / "resized_10k_files_stats.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resize the original JWST 10k FITS subset to a fixed square size."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--manifest-csv", type=Path, default=DEFAULT_MANIFEST_CSV)
    parser.add_argument("--output-manifest", type=Path, default=DEFAULT_OUTPUT_MANIFEST)
    parser.add_argument("--output-stats", type=Path, default=DEFAULT_OUTPUT_STATS)
    parser.add_argument("--target-size", type=int, default=64)
    parser.add_argument(
        "--interpolation",
        choices=("nearest", "bilinear", "bicubic"),
        default="bilinear",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing resized FITS files and output metadata files.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional cap on the number of input files processed, useful for smoke tests.",
    )
    return parser.parse_args()


def load_manifest_lookup(manifest_path: Path) -> dict[str, dict[str, str]]:
    if not manifest_path.exists():
        return {}

    with manifest_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        return {row["filename"]: row for row in reader if row.get("filename")}


def discover_files(input_dir: Path, max_files: int | None) -> list[Path]:
    files = sorted(path for path in input_dir.glob("*.fits") if path.is_file())
    if max_files is not None:
        files = files[:max_files]
    return files


def find_image_hdu(hdul: fits.HDUList) -> tuple[int, np.ndarray]:
    for hdu_index, hdu in enumerate(hdul):
        data = hdu.data
        if data is None:
            continue
        array = np.asarray(data)
        if array.ndim >= 2:
            squeezed = np.squeeze(array)
            if squeezed.ndim != 2:
                raise ValueError(f"Expected 2D image data after squeeze, found shape {array.shape}")
            return hdu_index, squeezed.astype(np.float32, copy=False)
    raise ValueError("No 2D image HDU found")


def resize_image(image: np.ndarray, target_size: int, interpolation: str) -> np.ndarray:
    if image.ndim != 2:
        raise ValueError(f"Expected a 2D image array, received shape {image.shape}")

    tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)
    kwargs: dict[str, Any] = {"mode": interpolation, "size": (target_size, target_size)}
    if interpolation in {"bilinear", "bicubic"}:
        kwargs["align_corners"] = False

    resized = torch_f.interpolate(tensor, **kwargs)
    return resized.squeeze(0).squeeze(0).cpu().numpy().astype(np.float32, copy=False)


def build_output_header(
    *,
    filename: str,
    original_width: int,
    original_height: int,
    target_size: int,
    interpolation: str,
    metadata_row: dict[str, str] | None,
    hdu_index: int,
) -> fits.Header:
    header = fits.Header()
    header["SRCFILE"] = filename
    header["ORIGW"] = original_width
    header["ORIGH"] = original_height
    header["ORIGHDU"] = hdu_index
    header["RSZSIZE"] = target_size
    header["RSZMETH"] = interpolation

    if metadata_row:
        filter_name = metadata_row.get("filter")
        version = metadata_row.get("version")
        if filter_name:
            header["FILTERID"] = filter_name[:68]
        if version:
            header["SRCVER"] = version[:68]

    header.add_history("Resized from original JWST 10k subset for ViT training input.")
    return header


def summarize_numeric(values: list[float]) -> dict[str, int | float | None]:
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None, "median": None}

    sorted_values = sorted(values)
    midpoint = len(sorted_values) // 2
    if len(sorted_values) % 2 == 0:
        median = (sorted_values[midpoint - 1] + sorted_values[midpoint]) / 2
    else:
        median = sorted_values[midpoint]

    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": round(sum(values) / len(values), 4),
        "median": round(float(median), 4),
    }


def write_manifest(rows: list[dict[str, Any]], output_path: Path) -> None:
    fieldnames = [
        "sample_index",
        "filename",
        "source_path",
        "input_path",
        "input_relpath",
        "resized_path",
        "resized_relpath",
        "filter",
        "version",
        "ra_deg",
        "dec_deg",
        "source_hdu_index",
        "source_width",
        "source_height",
        "source_bitpix",
        "target_size",
        "interpolation",
        "input_min",
        "input_max",
        "output_min",
        "output_max",
        "nonfinite_input_count",
        "output_file_size_bytes",
        "status",
        "error",
    ]

    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def build_stats(rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    error_rows = [row for row in rows if row.get("status") != "ok"]

    input_widths = [int(row["source_width"]) for row in ok_rows]
    input_heights = [int(row["source_height"]) for row in ok_rows]
    output_sizes = [int(row["target_size"]) for row in ok_rows]
    output_file_sizes = [int(row["output_file_size_bytes"]) for row in ok_rows]
    filters = Counter(str(row.get("filter", "unknown")) for row in ok_rows)
    versions = Counter(str(row.get("version", "unknown")) for row in ok_rows)
    errors = Counter(str(row.get("error", "unknown")) for row in error_rows)

    return {
        "input_dir": str(args.input_dir.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "manifest_csv": str(args.output_manifest.resolve()),
        "resizing": {
            "target_size": args.target_size,
            "interpolation": args.interpolation,
        },
        "dataset": {
            "processed_files": len(rows),
            "successful_files": len(ok_rows),
            "failed_files": len(error_rows),
            "source_width": summarize_numeric(input_widths),
            "source_height": summarize_numeric(input_heights),
            "resized_width": summarize_numeric(output_sizes),
            "resized_height": summarize_numeric(output_sizes),
            "output_file_size_bytes": summarize_numeric(output_file_sizes),
            "filter_counts": dict(sorted(filters.items())),
            "version_counts": dict(sorted(versions.items())),
            "top_errors": dict(errors.most_common(10)),
        },
    }


def write_stats(stats: dict[str, Any], output_path: Path) -> None:
    with output_path.open("w") as handle:
        json.dump(stats, handle, indent=2, sort_keys=True)


def ensure_writable_path(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing path without --overwrite: {path}")


def process_file(
    *,
    file_path: Path,
    input_root: Path,
    output_root: Path,
    metadata_row: dict[str, str] | None,
    target_size: int,
    interpolation: str,
    overwrite: bool,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "sample_index": metadata_row.get("sample_index") if metadata_row else None,
        "filename": file_path.name,
        "source_path": metadata_row.get("source_path") if metadata_row else str(file_path.resolve()),
        "input_path": str(file_path.resolve()),
        "input_relpath": str(file_path.relative_to(input_root.parent)),
        "filter": metadata_row.get("filter") if metadata_row else None,
        "version": metadata_row.get("version") if metadata_row else None,
        "ra_deg": metadata_row.get("ra_deg") if metadata_row else None,
        "dec_deg": metadata_row.get("dec_deg") if metadata_row else None,
        "target_size": target_size,
        "interpolation": interpolation,
        "status": "ok",
        "error": "",
    }

    output_path = output_root / file_path.name
    record["resized_path"] = str(output_path.resolve())
    record["resized_relpath"] = str(output_path.relative_to(output_root.parent))

    try:
        if output_path.exists() and not overwrite:
            raise FileExistsError(f"Output file already exists: {output_path}")

        with fits.open(file_path, memmap=False) as hdul:
            hdu_index, image = find_image_hdu(hdul)
            record["source_hdu_index"] = hdu_index
            record["source_width"] = int(image.shape[1])
            record["source_height"] = int(image.shape[0])
            record["source_bitpix"] = int(hdul[hdu_index].header.get("BITPIX", 0))

        nonfinite_mask = ~np.isfinite(image)
        record["nonfinite_input_count"] = int(nonfinite_mask.sum())
        if record["nonfinite_input_count"]:
            finite_values = image[np.isfinite(image)]
            fill_value = float(finite_values.mean()) if finite_values.size else 0.0
            image = np.where(np.isfinite(image), image, fill_value).astype(np.float32, copy=False)

        record["input_min"] = float(np.min(image))
        record["input_max"] = float(np.max(image))

        resized = resize_image(image, target_size, interpolation)
        record["output_min"] = float(np.min(resized))
        record["output_max"] = float(np.max(resized))

        header = build_output_header(
            filename=file_path.name,
            original_width=record["source_width"],
            original_height=record["source_height"],
            target_size=target_size,
            interpolation=interpolation,
            metadata_row=metadata_row,
            hdu_index=hdu_index,
        )
        fits.PrimaryHDU(data=resized, header=header).writeto(output_path, overwrite=overwrite)
        record["output_file_size_bytes"] = output_path.stat().st_size
    except Exception as exc:  # pragma: no cover - depends on data quality
        record["status"] = "error"
        record["error"] = f"{type(exc).__name__}: {exc}"
        record.setdefault("source_hdu_index", None)
        record.setdefault("source_width", None)
        record.setdefault("source_height", None)
        record.setdefault("source_bitpix", None)
        record.setdefault("input_min", None)
        record.setdefault("input_max", None)
        record.setdefault("output_min", None)
        record.setdefault("output_max", None)
        record.setdefault("nonfinite_input_count", None)
        record.setdefault("output_file_size_bytes", None)

    return record


def print_summary(stats: dict[str, Any]) -> None:
    dataset = stats["dataset"]
    print("\nResize summary")
    print("--------------")
    print(f"Processed files:    {dataset['processed_files']:,}")
    print(f"Successful files:   {dataset['successful_files']:,}")
    print(f"Failed files:       {dataset['failed_files']:,}")
    print(f"Source width:       {dataset['source_width']}")
    print(f"Source height:      {dataset['source_height']}")
    print(f"Resized width:      {dataset['resized_width']}")
    print(f"Resized height:     {dataset['resized_height']}")
    print(f"Top filters:        {dataset['filter_counts']}")


def main() -> None:
    args = parse_args()
    args.input_dir = args.input_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.manifest_csv = args.manifest_csv.resolve()
    args.output_manifest = args.output_manifest.resolve()
    args.output_stats = args.output_stats.resolve()

    if args.target_size <= 0:
        raise ValueError("--target-size must be positive")
    if args.max_files is not None and args.max_files <= 0:
        raise ValueError("--max-files must be positive when provided")
    if not args.input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {args.input_dir}")

    ensure_writable_path(args.output_manifest, args.overwrite)
    ensure_writable_path(args.output_stats, args.overwrite)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata_lookup = load_manifest_lookup(args.manifest_csv)
    input_files = discover_files(args.input_dir, args.max_files)
    if not input_files:
        raise RuntimeError(f"No FITS files found in {args.input_dir}")

    rows: list[dict[str, Any]] = []
    total_files = len(input_files)
    for index, file_path in enumerate(input_files, start=1):
        row = process_file(
            file_path=file_path,
            input_root=args.input_dir,
            output_root=args.output_dir,
            metadata_row=metadata_lookup.get(file_path.name),
            target_size=args.target_size,
            interpolation=args.interpolation,
            overwrite=args.overwrite,
        )
        if row.get("sample_index") is None:
            row["sample_index"] = index
        rows.append(row)

        if index % 1000 == 0 or index == total_files:
            ok_count = sum(1 for item in rows if item.get("status") == "ok")
            error_count = len(rows) - ok_count
            print(
                f"Processed {index:,}/{total_files:,} files "
                f"({ok_count:,} ok, {error_count:,} failed)",
                flush=True,
            )

    rows.sort(key=lambda row: int(row.get("sample_index") or 0))
    write_manifest(rows, args.output_manifest)
    stats = build_stats(rows, args)
    write_stats(stats, args.output_stats)
    print_summary(stats)

    print("\nWrote outputs")
    print("-------------")
    print(f"Resized FITS files: {args.output_dir}")
    print(f"Manifest CSV:       {args.output_manifest}")
    print(f"Stats JSON:         {args.output_stats}")


if __name__ == "__main__":
    main()