#!/usr/bin/env python3
"""Sample a pretraining subset from the local JWST small dataset.

This script scans a source directory of FITS thumbnails, records lightweight
metadata for each file, randomly samples a configurable subset, materializes the
sampled files into a dedicated output directory, and writes:

- manifest.csv: per-sample metadata for the extracted subset
- stats.json: source-pool and sampled-subset summary statistics

The script only inspects FITS headers to derive image dimensions, which keeps
the scan lighter than reading full arrays for every file.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import statistics
from collections import Counter
from pathlib import Path

from astropy.io import fits


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "output" / "original_JWST" / "small_dataset"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "original_JWST" / "pretraining_10k"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample a random JWST FITS subset and write manifest/stat outputs."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing the source FITS files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where the sampled subset, manifest, and stats will be written.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=10_000,
        help="Number of files to sample without replacement.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling.",
    )
    parser.add_argument(
        "--mode",
        choices=("copy", "symlink", "hardlink"),
        default="copy",
        help="How to materialize sampled files into the output directory.",
    )
    parser.add_argument(
        "--pattern",
        default="*.fits",
        help="Glob pattern used to discover input files.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search the input directory.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional cap on discovered files, useful for smoke testing.",
    )
    return parser.parse_args()


def discover_files(input_dir: Path, pattern: str, recursive: bool) -> list[Path]:
    if recursive:
        files = sorted(path for path in input_dir.rglob(pattern) if path.is_file())
    else:
        files = sorted(path for path in input_dir.glob(pattern) if path.is_file())
    return files


def parse_filename_metadata(file_path: Path) -> dict[str, object]:
    parts = file_path.stem.split("_")
    metadata: dict[str, object] = {
        "ra_deg": None,
        "dec_deg": None,
        "filter": "unknown",
        "version": "unknown",
    }

    if len(parts) < 4:
        return metadata

    ra_str = parts[0]
    dec_str = parts[1]
    filter_name = "_".join(parts[2:-1]) or "unknown"
    version = parts[-1]

    try:
        metadata["ra_deg"] = float(ra_str)
    except ValueError:
        metadata["ra_deg"] = None

    try:
        metadata["dec_deg"] = float(dec_str)
    except ValueError:
        metadata["dec_deg"] = None

    metadata["filter"] = filter_name
    metadata["version"] = version
    return metadata


def inspect_fits_header(file_path: Path) -> dict[str, object]:
    with fits.open(file_path, memmap=True, lazy_load_hdus=True) as hdul:
        for hdu_index, hdu in enumerate(hdul):
            header = hdu.header
            naxis = int(header.get("NAXIS", 0))
            if naxis < 2:
                continue

            width = header.get("NAXIS1")
            height = header.get("NAXIS2")
            if width is None or height is None:
                continue

            return {
                "hdu_index": hdu_index,
                "naxis": naxis,
                "width": int(width),
                "height": int(height),
                "depth": int(header.get("NAXIS3", 1)) if naxis >= 3 else 1,
                "bitpix": int(header.get("BITPIX", 0)),
            }

    raise ValueError("No image HDU with at least 2 dimensions found")


def scan_source_files(file_paths: list[Path]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    valid_records: list[dict[str, object]] = []
    invalid_records: list[dict[str, object]] = []

    total_files = len(file_paths)
    for index, file_path in enumerate(file_paths, start=1):
        base_record: dict[str, object] = {
            "filename": file_path.name,
            "source_path": str(file_path.resolve()),
            "file_size_bytes": file_path.stat().st_size,
        }
        base_record.update(parse_filename_metadata(file_path))

        try:
            base_record.update(inspect_fits_header(file_path))
            base_record["status"] = "ok"
            valid_records.append(base_record)
        except Exception as exc:  # pragma: no cover - error path is data dependent
            base_record.update(
                {
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "hdu_index": None,
                    "naxis": None,
                    "width": None,
                    "height": None,
                    "depth": None,
                    "bitpix": None,
                }
            )
            invalid_records.append(base_record)

        if index % 5_000 == 0 or index == total_files:
            print(
                f"Scanned {index:,}/{total_files:,} files "
                f"({len(valid_records):,} readable, {len(invalid_records):,} unreadable)",
                flush=True,
            )

    return valid_records, invalid_records


def summarize_numeric(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
        }

    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": round(statistics.fmean(values), 4),
        "median": round(statistics.median(values), 4),
    }


def most_common_shapes(records: list[dict[str, object]], limit: int = 10) -> list[dict[str, object]]:
    shape_counter = Counter(
        (int(record["height"]), int(record["width"]))
        for record in records
        if record.get("height") is not None and record.get("width") is not None
    )
    return [
        {"height": height, "width": width, "count": count}
        for (height, width), count in shape_counter.most_common(limit)
    ]


def summarize_records(records: list[dict[str, object]]) -> dict[str, object]:
    file_sizes = [int(record["file_size_bytes"]) for record in records]
    widths = [int(record["width"]) for record in records if record.get("width") is not None]
    heights = [int(record["height"]) for record in records if record.get("height") is not None]
    depths = [int(record["depth"]) for record in records if record.get("depth") is not None]
    filter_counts = Counter(str(record.get("filter", "unknown")) for record in records)
    version_counts = Counter(str(record.get("version", "unknown")) for record in records)

    return {
        "file_count": len(records),
        "total_bytes": sum(file_sizes),
        "total_gib": round(sum(file_sizes) / (1024**3), 4),
        "file_size_bytes": summarize_numeric(file_sizes),
        "width": summarize_numeric(widths),
        "height": summarize_numeric(heights),
        "depth": summarize_numeric(depths),
        "filter_counts": dict(sorted(filter_counts.items())),
        "version_counts": dict(sorted(version_counts.items())),
        "top_image_shapes": most_common_shapes(records),
    }


def build_stats(
    *,
    input_dir: Path,
    output_dir: Path,
    sample_size: int,
    seed: int,
    mode: str,
    discovered_files: int,
    valid_records: list[dict[str, object]],
    invalid_records: list[dict[str, object]],
    sampled_records: list[dict[str, object]],
) -> dict[str, object]:
    invalid_error_counts = Counter(
        str(record.get("error", "unknown error")) for record in invalid_records
    )
    return {
        "input_dir": str(input_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "sampling": {
            "sample_size": sample_size,
            "seed": seed,
            "materialization_mode": mode,
        },
        "source_pool": {
            "files_discovered": discovered_files,
            "readable_files": len(valid_records),
            "unreadable_files": len(invalid_records),
            "summary": summarize_records(valid_records),
            "top_errors": dict(invalid_error_counts.most_common(10)),
        },
        "sampled_subset": summarize_records(sampled_records),
    }


def ensure_clean_output_dir(output_dir: Path) -> Path:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory already exists and is not empty: {output_dir}"
        )

    files_dir = output_dir / "files"
    files_dir.mkdir(parents=True, exist_ok=True)
    return files_dir


def make_unique_destination(files_dir: Path, filename: str) -> Path:
    destination = files_dir / filename
    if not destination.exists():
        return destination

    stem = Path(filename).stem
    suffix = Path(filename).suffix
    counter = 1
    while True:
        candidate = files_dir / f"{stem}_{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def materialize_sample(
    sampled_records: list[dict[str, object]], files_dir: Path, mode: str
) -> None:
    for index, record in enumerate(sampled_records, start=1):
        source_path = Path(str(record["source_path"]))
        destination = make_unique_destination(files_dir, str(record["filename"]))

        if mode == "copy":
            shutil.copy2(source_path, destination)
        elif mode == "symlink":
            destination.symlink_to(source_path)
        elif mode == "hardlink":
            os.link(source_path, destination)
        else:  # pragma: no cover - argparse enforces choices
            raise ValueError(f"Unsupported mode: {mode}")

        record["sampled_path"] = str(destination.resolve())
        record["sampled_relpath"] = str(destination.relative_to(files_dir.parent))
        record["sample_index"] = index


def write_manifest(sampled_records: list[dict[str, object]], manifest_path: Path) -> None:
    fieldnames = [
        "sample_index",
        "filename",
        "source_path",
        "sampled_path",
        "sampled_relpath",
        "file_size_bytes",
        "ra_deg",
        "dec_deg",
        "filter",
        "version",
        "hdu_index",
        "naxis",
        "width",
        "height",
        "depth",
        "bitpix",
        "status",
    ]

    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in sorted(sampled_records, key=lambda item: int(item["sample_index"])):
            writer.writerow({field: record.get(field) for field in fieldnames})


def write_stats(stats: dict[str, object], stats_path: Path) -> None:
    with stats_path.open("w") as handle:
        json.dump(stats, handle, indent=2, sort_keys=True)


def print_stats_summary(stats: dict[str, object]) -> None:
    source_pool = stats["source_pool"]
    sampled_subset = stats["sampled_subset"]
    print("\nSource pool summary")
    print("-------------------")
    print(f"Files discovered:   {source_pool['files_discovered']:,}")
    print(f"Readable files:     {source_pool['readable_files']:,}")
    print(f"Unreadable files:   {source_pool['unreadable_files']:,}")
    print(f"Total size (GiB):   {source_pool['summary']['total_gib']}")
    print(f"Width stats:        {source_pool['summary']['width']}")
    print(f"Height stats:       {source_pool['summary']['height']}")
    print(f"Top filters:        {source_pool['summary']['filter_counts']}")

    print("\nSampled subset summary")
    print("----------------------")
    print(f"Sample size:        {sampled_subset['file_count']:,}")
    print(f"Total size (GiB):   {sampled_subset['total_gib']}")
    print(f"Width stats:        {sampled_subset['width']}")
    print(f"Height stats:       {sampled_subset['height']}")
    print(f"Top filters:        {sampled_subset['filter_counts']}")


def main() -> None:
    args = parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()

    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    if args.sample_size <= 0:
        raise ValueError("--sample-size must be positive")
    if args.max_files is not None and args.max_files <= 0:
        raise ValueError("--max-files must be positive when provided")

    discovered_files = discover_files(input_dir, args.pattern, args.recursive)
    if args.max_files is not None:
        discovered_files = discovered_files[: args.max_files]

    if not discovered_files:
        raise RuntimeError(
            f"No files matched pattern {args.pattern!r} under {input_dir}"
        )

    print(
        f"Discovered {len(discovered_files):,} files in {input_dir}. "
        f"Scanning FITS headers before sampling...",
        flush=True,
    )
    valid_records, invalid_records = scan_source_files(discovered_files)

    if args.sample_size > len(valid_records):
        raise ValueError(
            f"Requested sample size {args.sample_size:,} exceeds the number of "
            f"readable FITS files ({len(valid_records):,})."
        )

    rng = random.Random(args.seed)
    sampled_records = rng.sample(valid_records, args.sample_size)

    files_dir = ensure_clean_output_dir(output_dir)
    materialize_sample(sampled_records, files_dir, args.mode)

    manifest_path = output_dir / "manifest.csv"
    stats_path = output_dir / "stats.json"

    write_manifest(sampled_records, manifest_path)
    stats = build_stats(
        input_dir=input_dir,
        output_dir=output_dir,
        sample_size=args.sample_size,
        seed=args.seed,
        mode=args.mode,
        discovered_files=len(discovered_files),
        valid_records=valid_records,
        invalid_records=invalid_records,
        sampled_records=sampled_records,
    )
    write_stats(stats, stats_path)
    print_stats_summary(stats)

    print("\nWrote outputs")
    print("-------------")
    print(f"Sampled files:      {files_dir}")
    print(f"Manifest:           {manifest_path}")
    print(f"Stats JSON:         {stats_path}")


if __name__ == "__main__":
    main()