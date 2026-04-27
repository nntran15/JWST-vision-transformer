#!/usr/bin/env python3
"""Manually label random JWST thumbnails as spiral or not_spiral."""

from __future__ import annotations

import argparse
import csv
import logging
import random
import sys
import webbrowser
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.fits_preprocessing import make_display_image

logger = logging.getLogger(__name__)

DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "JWST" / "resized_10k_files"
DEFAULT_LABELS_CSV = PROJECT_ROOT / "output" / "manual_spiral_labels.csv"
DEFAULT_PREVIEW_DIR = PROJECT_ROOT / "output" / "manual_spiral_labeling"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Label random JWST thumbnails as spiral or not_spiral")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--labels-csv", type=Path, default=DEFAULT_LABELS_CSV)
    parser.add_argument("--preview-dir", type=Path, default=DEFAULT_PREVIEW_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument(
        "--open-browser",
        dest="open_browser",
        action="store_true",
        help="Try to open each preview automatically in the default browser.",
    )
    parser.add_argument(
        "--no-open-browser",
        dest="open_browser",
        action="store_false",
        help="Do not auto-open previews; just print the preview path.",
    )
    parser.set_defaults(open_browser=True)
    return parser.parse_args()


def discover_fits_files(data_dir: Path) -> list[Path]:
    files = sorted(path for path in data_dir.glob("*.fits") if path.is_file())
    if not files:
        raise FileNotFoundError(f"No FITS files found under {data_dir}")
    return files


def load_existing_labels(labels_csv: Path) -> dict[str, str]:
    if not labels_csv.exists():
        return {}

    with labels_csv.open(newline="") as handle:
        reader = csv.DictReader(handle)
        return {row["file_path"]: row["label"] for row in reader if row.get("file_path")}


def find_image_data(path: Path) -> tuple[np.ndarray, fits.Header]:
    with fits.open(path, memmap=False) as hdul:
        for hdu in hdul:
            data = getattr(hdu, "data", None)
            if data is not None:
                array = np.asarray(data, dtype=np.float32)
                if array.ndim >= 2:
                    return array, hdu.header
    raise ValueError(f"No image data found in {path}")


def render_preview(fits_path: Path, preview_png: Path) -> Path:
    image, header = find_image_data(fits_path)
    display = make_display_image(image, header=header, normalization="header", target_size=256)

    preview_png.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 6))
    if display.ndim == 2:
        ax.imshow(display, cmap="gray", vmin=0.0, vmax=1.0, origin="lower")
    else:
        ax.imshow(display, origin="lower")
    ax.set_title(fits_path.name, fontsize=10)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(preview_png, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return preview_png


def write_preview_html(preview_png: Path, html_path: Path, index: int, total: int) -> Path:
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(
        """<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <title>Manual Spiral Labeling</title>
  <style>
        body {{ font-family: sans-serif; margin: 24px; background: #111; color: #f3f3f3; }}
        img {{ max-width: min(80vw, 800px); border: 1px solid #444; background: #000; }}
        .meta {{ margin-bottom: 16px; }}
        .hint {{ margin-top: 16px; color: #c7c7c7; }}
  </style>
</head>
<body>
  <div class=\"meta\">
    <h1>Manual Spiral Labeling</h1>
    <p>Sample {index} of {total}</p>
    <p>{filename}</p>
  </div>
  <img src=\"{image_name}\" alt=\"JWST preview\">
  <div class=\"hint\">
    Label in the terminal: <strong>s</strong> spiral, <strong>n</strong> not_spiral, <strong>k</strong> skip, <strong>q</strong> quit.
  </div>
</body>
</html>
""".format(index=index, total=total, filename=preview_png.stem, image_name=preview_png.name),
        encoding="utf-8",
    )
    return html_path


def append_label(labels_csv: Path, file_path: Path, label: str) -> None:
    labels_csv.parent.mkdir(parents=True, exist_ok=True)
    file_exists = labels_csv.exists()
    with labels_csv.open("a", newline="") as handle:
        writer = csv.writer(handle)
        if not file_exists:
            writer.writerow(["file_path", "cluster_id", "label"])
        writer.writerow([str(file_path.resolve()), "manual", label])


def choose_label() -> str | None:
    prompt = "Label [s=spiral, n=not_spiral, k=skip, q=quit]: "
    while True:
        answer = input(prompt).strip().lower()
        if answer in {"s", "spiral"}:
            return "spiral"
        if answer in {"n", "not_spiral", "not-spiral"}:
            return "not_spiral"
        if answer in {"k", "skip"}:
            return "unlabeled"
        if answer in {"q", "quit"}:
            return None
        print("Invalid input. Use s, n, k, or q.")


def maybe_open_browser(html_path: Path, enabled: bool) -> bool:
    if not enabled:
        return False
    try:
        return webbrowser.open(html_path.resolve().as_uri(), new=0)
    except Exception as exc:
        logger.warning(f"Could not open browser automatically: {exc}")
        return False


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    files = discover_fits_files(args.data_dir)
    labeled = load_existing_labels(args.labels_csv)
    candidates = [path for path in files if str(path.resolve()) not in labeled]

    if not candidates:
        print("All available FITS files are already labeled.")
        return 0

    rng = random.Random(args.seed)
    rng.shuffle(candidates)
    if args.max_images is not None:
        candidates = candidates[: args.max_images]

    print(f"Dataset directory: {args.data_dir}")
    print(f"Labels CSV: {args.labels_csv}")
    print(f"Unlabeled candidate count: {len(candidates)}")

    for index, fits_path in enumerate(candidates, start=1):
        preview_png = args.preview_dir / "current_image.png"
        preview_html = args.preview_dir / "current_image.html"
        render_preview(fits_path, preview_png)
        write_preview_html(preview_png, preview_html, index=index, total=len(candidates))

        opened = maybe_open_browser(preview_html, args.open_browser)
        print(f"\nPreview HTML: {preview_html}")
        print(f"Preview image: {preview_png}")
        if not opened:
            print("Automatic browser open was unavailable. Open the preview HTML manually if needed.")

        label = choose_label()
        if label is None:
            print("Stopping without labeling this sample.")
            break

        append_label(args.labels_csv, fits_path, label)
        print(f"Recorded {label} for {fits_path.name}")

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())