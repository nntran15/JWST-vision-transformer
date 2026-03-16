#!/usr/bin/env python3
"""
extract_all.py

Extracts thumbnails for ALL 784,016 sources in the COSMOS-Web catalog.
No filtering — stars, galaxies, blended sources, everything is included.
This is intentional: SSL pretraining benefits from maximum data diversity.

Strategy:
  - Group sources by tile so each sci image is loaded exactly once
  - Crop size is adaptive: RADII_MULTIPLIER × a_image, clamped to [MIN, MAX]
  - Only sources whose crop falls outside the image boundary are skipped
  - Output: one HDF5 file per tile, then a merged master file

Output HDF5 structure:
  thumbnails  : (N, THUMB_SIZE, THUMB_SIZE) float32  — normalized to [0,1]
  id          : (N,)  int64    — COSMOS-Web source ID
  segment_id  : (N,)  int64    — segmentation map ID (useful later)
  ra          : (N,)  float64  — right ascension (degrees)
  dec         : (N,)  float64  — declination (degrees)
  tile        : (N,)  bytes    — which tile, e.g. b'A1'
  snr_f115w   : (N,)  float32  — SNR in F115W (useful for downstream filtering)
  a_image     : (N,)  float32  — semi-major axis in pixels (proxy for galaxy size)
  flag_star   : (N,)  bool     — True = likely star (kept, but labeled)
  flag_blend  : (N,)  bool     — True = blended with neighbor (kept, but labeled)
  crop_half   : (N,)  int16    — half-width of the crop before resizing (for reference)

Usage:
  python extract_all.py

  Estimated runtime at 60mas:
    ~20 min for all 20 tiles on a server with fast disk I/O
    ~45 min on slower NFS-mounted storage

  Estimated output size:
    Per-tile HDF5 files: ~300-500 MB each (gzip compressed)
    Master file:         ~6-8 GB total for 784k × 64×64 thumbnails
"""

import numpy as np
import h5py
from astropy.io import fits
from astropy.table import Table
from pathlib import Path
from collections import defaultdict
from PIL import Image
from tqdm import tqdm
import time
import json

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

CATALOG_PATH = Path(
    "/extra/wayne2/preserve/nntran5/vision-transformer/data/COSMOS/COSMOSWeb_mastercatalog_v1.fits"
)
SCI_DIR = Path(
    "/extra/wayne2/preserve/nntran5/vision-transformer/data/COSMOS/sky_patches"
)
TILE_OUTPUT_DIR = Path(
    "/extra/wayne2/preserve/nntran5/vision-transformer/data/COSMOS/thumbnails_all"
)
MASTER_OUTPUT_PATH = Path(
    "/extra/wayne2/preserve/nntran5/vision-transformer/data/COSMOS/"
    "cosmos_web_all_f115w.h5"
)

FILTER      = "f115w"
PIXEL_SCALE = "30mas"   
THUMB_SIZE  = 64        

RADII_MULTIPLIER = 5
MIN_CROP_HALF    = 16   # smallest crop = 32×32 pixels
MAX_CROP_HALF    = 256  # largest crop  = 512×512 pixels

# ══════════════════════════════════════════════════════════════════════════════


def normalize_arcsinh(img: np.ndarray, softening: float = 0.01) -> np.ndarray:
    """
    Arcsinh stretch → percentile clip → [0, 1] float32.

    Arcsinh is standard for astronomical images because galaxy pixel values
    span several orders of magnitude: the bright nucleus might be 1000×
    the sky background, and spiral arms sit somewhere in between. A linear
    stretch would make arms invisible. Arcsinh gently compresses the bright
    end while lifting faint structure, similar to how astronomers have always
    processed images for visual inspection.

    softening: controls the transition from linear (faint pixels, near 0)
               to log (bright pixels). Smaller = more aggressive compression.
               0.01 works well for JWST NIRCam; tune if cores look blown out.
    """
    img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
    img = np.arcsinh(img / softening)
    p_lo, p_hi = np.percentile(img, [0.5, 99.5])
    if p_hi > p_lo:
        img = np.clip((img - p_lo) / (p_hi - p_lo), 0.0, 1.0)
    else:
        img = np.zeros_like(img)
    return img.astype(np.float32)


def resize_lanczos(img: np.ndarray, size: int) -> np.ndarray:
    """
    Resize a [0,1] float32 2D array to (size × size) using Lanczos resampling.
    Lanczos preserves fine detail better than bilinear for downsampling,
    which matters when compressing a large galaxy crop to 64px.
    """
    pil = Image.fromarray((img * 255).astype(np.uint8), mode="L")
    pil = pil.resize((size, size), Image.LANCZOS)
    return np.array(pil, dtype=np.float32) / 255.0


def sci_filename(filter_: str, pixel_scale: str, tile: str) -> str:
    return f"mosaic_nircam_{filter_}_COSMOS-Web_{pixel_scale}_{tile}_v1.0_sci.fits"


def load_sci_image(sci_dir: Path, filter_: str, pixel_scale: str, tile: str) -> np.ndarray | None:
    """
    Load a science FITS image into a float32 numpy array.
    Returns None if the file doesn't exist.

    Handles both extracted sci files (data in HDU 0) and accidentally
    downloaded i2d files (science data in HDU 1 named 'SCI').
    """
    path = sci_dir / sci_filename(filter_, pixel_scale, tile)
    if not path.exists():
        return None, None

    with fits.open(path, memmap=False) as hdul:
        # Extracted sci files: image is in HDU 0
        if hdul[0].data is not None and hdul[0].data.ndim == 2:
            data = hdul[0].data.astype(np.float32)
            hdr  = hdul[0].header
        # i2d files: science is in the extension named 'SCI' (HDU 1)
        elif len(hdul) > 1 and hdul[1].data is not None:
            data = hdul[1].data.astype(np.float32)
            hdr  = hdul[1].header
        else:
            print(f"  WARNING: could not find image data in {path.name}")
            return None, None

    return data, hdr


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: Load catalog and group by tile
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("COSMOS-Web Full Extraction")
print("=" * 60)

TILE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"\nLoading catalog: {CATALOG_PATH.name}")
t0 = time.time()
cat = Table.read(str(CATALOG_PATH), hdu=1)
print(f"Loaded {len(cat):,} sources in {time.time()-t0:.1f}s")

# Decode byte-string tile column and strip whitespace
# FITS stores fixed-width strings as bytes: b'A1 ' → 'A1'
tile_str = np.array([
    (t.decode("utf-8") if isinstance(t, (bytes, np.bytes_)) else str(t)).strip()
    for t in cat["tile"]
], dtype=str)

# Group row indices by tile name for efficient per-tile processing
tile_groups: dict[str, np.ndarray] = defaultdict(list)
for i, t in enumerate(tile_str):
    tile_groups[t].append(i)
tile_groups = {k: np.array(v) for k, v in tile_groups.items()}

tiles_in_catalog = sorted(tile_groups.keys())
print(f"Tiles found in catalog: {tiles_in_catalog}")
print(f"Sources per tile:")
for tile in tiles_in_catalog:
    print(f"  {tile:4s}: {len(tile_groups[tile]):,}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: Extract thumbnails tile by tile
# ══════════════════════════════════════════════════════════════════════════════

# Track extraction stats across all tiles
global_stats = {
    "total_in_catalog" : len(cat),
    "total_extracted"  : 0,
    "total_skipped"    : 0,
    "tiles_processed"  : 0,
    "tiles_missing"    : [],
}

print(f"\n{'─'*60}")
print(f"Extracting thumbnails ({THUMB_SIZE}×{THUMB_SIZE}px, {FILTER}, {PIXEL_SCALE})")
print(f"{'─'*60}")

for tile in tiles_in_catalog:
    row_indices = tile_groups[tile]
    n           = len(row_indices)
    tile_rows   = cat[row_indices]   # sub-table for this tile

    tile_t0 = time.time()
    print(f"\n[{tile}] {n:,} sources | loading sci image...", end=" ", flush=True)

    sci, hdr = load_sci_image(SCI_DIR, FILTER, PIXEL_SCALE, tile)
    if sci is None:
        print("FILE NOT FOUND — skipping")
        global_stats["tiles_missing"].append(tile)
        continue

    img_h, img_w = sci.shape
    print(f"shape={img_w}×{img_h} | {sci.nbytes/1e6:.0f} MB")

    # Pre-allocate output arrays for this tile
    thumbnails = np.zeros((n, THUMB_SIZE, THUMB_SIZE), dtype=np.float32)
    ids        = np.zeros(n, dtype=np.int64)
    seg_ids    = np.zeros(n, dtype=np.int64)
    ras        = np.zeros(n, dtype=np.float64)
    decs       = np.zeros(n, dtype=np.float64)
    snrs       = np.zeros(n, dtype=np.float32)
    a_images   = np.zeros(n, dtype=np.float32)
    flag_star  = np.zeros(n, dtype=bool)
    flag_blend = np.zeros(n, dtype=bool)
    crop_halfs = np.zeros(n, dtype=np.int16)
    tiles_out  = np.empty(n, dtype="S3")   # byte string, e.g. b'A1'
    succeeded  = np.zeros(n, dtype=bool)

    for i, row in enumerate(tqdm(tile_rows, desc=f"  {tile}", unit="src", leave=False)):

        # ── Galaxy center in 0-indexed pixel coordinates ──────────────────
        # x_image / y_image from Source Extractor are 1-indexed (FITS convention)
        # Subtract 1 to convert to 0-indexed numpy array coordinates
        cx = float(row["x_image"]) - 1.0   # column index (horizontal)
        cy = float(row["y_image"]) - 1.0   # row index    (vertical)

        # ── Adaptive crop size based on source size ───────────────────────
        # a_image is the semi-major axis in pixels from Source Extractor.
        # We multiply by RADII_MULTIPLIER to capture the full extent of the
        # source plus some surrounding sky (important for SSL — the model
        # needs context to understand what it's looking at).
        a    = max(float(row["a_image"]), 1.0)   # guard against 0 or NaN
        half = int(np.clip(
            round(a * RADII_MULTIPLIER),
            MIN_CROP_HALF,
            MAX_CROP_HALF
        ))

        # ── Bounding box in integer pixel coordinates ─────────────────────
        cx_int = int(round(cx))
        cy_int = int(round(cy))
        x0 = cx_int - half
        x1 = cx_int + half
        y0 = cy_int - half
        y1 = cy_int + half

        # ── Skip sources whose crop exceeds the image boundary ────────────
        # These are edge sources whose center is in this tile but whose
        # full extent spills outside. They will appear in the neighboring
        # tile's extraction with a valid crop.
        if x0 < 0 or y0 < 0 or x1 > img_w or y1 > img_h:
            continue

        # ── Cut crop from sci image ───────────────────────────────────────
        # FITS/numpy arrays are indexed [row, col] = [y, x]
        crop = sci[y0:y1, x0:x1]

        if crop.size == 0:
            continue

        # ── Normalize and resize ──────────────────────────────────────────
        thumbnails[i] = resize_lanczos(normalize_arcsinh(crop), THUMB_SIZE)

        # ── Store metadata ────────────────────────────────────────────────
        ids[i]        = int(row["id"])
        seg_ids[i]    = int(row["segment-id"])
        ras[i]        = float(row["ra"])
        decs[i]       = float(row["dec"])
        snrs[i]       = float(row["snr_f115w"])
        a_images[i]   = float(row["a_image"])
        flag_star[i]  = bool(row["flag_star"])
        flag_blend[i] = bool(row["flag_blend"])
        crop_halfs[i] = half
        tiles_out[i]  = tile.encode("utf-8")
        succeeded[i]  = True

    # ── Save tile HDF5 ────────────────────────────────────────────────────────
    v        = succeeded                   # boolean mask of successful extractions
    n_ok     = v.sum()
    n_skip   = n - n_ok
    elapsed  = time.time() - tile_t0

    print(f"  ✓ {n_ok:,} extracted  |  {n_skip:,} skipped (boundary)  |  {elapsed:.0f}s")
    print(f"    Stars: {flag_star[v].sum():,}  |  Blended: {flag_blend[v].sum():,}")

    out_path = TILE_OUTPUT_DIR / f"thumbnails_{tile}.h5"
    with h5py.File(out_path, "w") as f:
        # Chunk size: 256 thumbnails per chunk — good balance for
        # sequential reads (training) and random access (inspection)
        chunk = (min(256, n_ok), THUMB_SIZE, THUMB_SIZE)

        f.create_dataset("thumbnails", data=thumbnails[v],
                         compression="gzip", compression_opts=4, chunks=chunk)
        f.create_dataset("id",         data=ids[v])
        f.create_dataset("segment_id", data=seg_ids[v])
        f.create_dataset("ra",         data=ras[v])
        f.create_dataset("dec",        data=decs[v])
        f.create_dataset("tile",       data=tiles_out[v])
        f.create_dataset("snr_f115w",  data=snrs[v])
        f.create_dataset("a_image",    data=a_images[v])
        f.create_dataset("flag_star",  data=flag_star[v])
        f.create_dataset("flag_blend", data=flag_blend[v])
        f.create_dataset("crop_half",  data=crop_halfs[v])

        f.attrs["tile"]        = tile.encode("utf-8")
        f.attrs["filter"]      = FILTER.encode("utf-8")
        f.attrs["pixel_scale"] = PIXEL_SCALE.encode("utf-8")
        f.attrs["thumb_size"]  = THUMB_SIZE
        f.attrs["n_sources"]   = int(n_ok)
        f.attrs["n_skipped"]   = int(n_skip)

    size_mb = out_path.stat().st_size / 1e6
    print(f"  Saved: {out_path.name}  ({size_mb:.0f} MB)")

    global_stats["total_extracted"] += n_ok
    global_stats["total_skipped"]   += n_skip
    global_stats["tiles_processed"] += 1

    # Free the sci image from RAM before loading the next tile
    del sci

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: Merge all tile HDF5 files into one master file
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n{'─'*60}")
print(f"Merging tile files into master HDF5")
print(f"{'─'*60}")

tile_files = sorted(TILE_OUTPUT_DIR.glob("thumbnails_*.h5"))
print(f"Tile files to merge: {len(tile_files)}")

# First pass: count total sources so we can pre-allocate
n_total = 0
for path in tile_files:
    with h5py.File(path, "r") as f:
        n_total += f.attrs["n_sources"]
print(f"Total sources to merge: {n_total:,}")

# Pre-allocate master arrays
print("Allocating master arrays...")
m_thumbs  = np.zeros((n_total, THUMB_SIZE, THUMB_SIZE), dtype=np.float32)
m_ids     = np.zeros(n_total, dtype=np.int64)
m_segids  = np.zeros(n_total, dtype=np.int64)
m_ras     = np.zeros(n_total, dtype=np.float64)
m_decs    = np.zeros(n_total, dtype=np.float64)
m_tiles   = np.empty(n_total, dtype="S3")
m_snr     = np.zeros(n_total, dtype=np.float32)
m_a       = np.zeros(n_total, dtype=np.float32)
m_fstar   = np.zeros(n_total, dtype=bool)
m_fblend  = np.zeros(n_total, dtype=bool)
m_chalf   = np.zeros(n_total, dtype=np.int16)

# Second pass: fill arrays
cursor = 0
for path in tile_files:
    with h5py.File(path, "r") as f:
        n = f.attrs["n_sources"]
        s = slice(cursor, cursor + n)

        m_thumbs[s] = f["thumbnails"][:]
        m_ids[s]    = f["id"][:]
        m_segids[s] = f["segment_id"][:]
        m_ras[s]    = f["ra"][:]
        m_decs[s]   = f["dec"][:]
        m_tiles[s]  = f["tile"][:]
        m_snr[s]    = f["snr_f115w"][:]
        m_a[s]      = f["a_image"][:]
        m_fstar[s]  = f["flag_star"][:]
        m_fblend[s] = f["flag_blend"][:]
        m_chalf[s]  = f["crop_half"][:]

        cursor += n
        print(f"  Merged {path.stem:20s} ({n:,} sources, cursor={cursor:,})")

# Deduplicate by source ID — sources at tile edges may appear in two tiles.
# np.unique on IDs gives us the first occurrence index of each unique ID.
print(f"\nDeduplicating on source ID...")
_, unique_idx  = np.unique(m_ids, return_index=True)
n_before       = n_total
n_after        = len(unique_idx)
n_dupes        = n_before - n_after
print(f"  Before: {n_before:,}  |  After: {n_after:,}  |  Duplicates removed: {n_dupes:,}")

m_thumbs = m_thumbs[unique_idx]
m_ids    = m_ids[unique_idx]
m_segids = m_segids[unique_idx]
m_ras    = m_ras[unique_idx]
m_decs   = m_decs[unique_idx]
m_tiles  = m_tiles[unique_idx]
m_snr    = m_snr[unique_idx]
m_a      = m_a[unique_idx]
m_fstar  = m_fstar[unique_idx]
m_fblend = m_fblend[unique_idx]
m_chalf  = m_chalf[unique_idx]

# Shuffle so tiles aren't contiguous in the file.
# A fixed seed makes this reproducible — crucial so your train/val
# split in downstream scripts always produces the same partition.
print("Shuffling (seed=42 for reproducibility)...")
rng = np.random.default_rng(seed=42)
order    = rng.permutation(n_after)
m_thumbs = m_thumbs[order]
m_ids    = m_ids[order]
m_segids = m_segids[order]
m_ras    = m_ras[order]
m_decs   = m_decs[order]
m_tiles  = m_tiles[order]
m_snr    = m_snr[order]
m_a      = m_a[order]
m_fstar  = m_fstar[order]
m_fblend = m_fblend[order]
m_chalf  = m_chalf[order]

# Save master file
print(f"\nWriting master file → {MASTER_OUTPUT_PATH}")
chunk = (256, THUMB_SIZE, THUMB_SIZE)
with h5py.File(MASTER_OUTPUT_PATH, "w") as f:
    f.create_dataset("thumbnails", data=m_thumbs,
                     compression="gzip", compression_opts=4, chunks=chunk)
    f.create_dataset("id",         data=m_ids)
    f.create_dataset("segment_id", data=m_segids)
    f.create_dataset("ra",         data=m_ras)
    f.create_dataset("dec",        data=m_decs)
    f.create_dataset("tile",       data=m_tiles)
    f.create_dataset("snr_f115w",  data=m_snr)
    f.create_dataset("a_image",    data=m_a)
    f.create_dataset("flag_star",  data=m_fstar)
    f.create_dataset("flag_blend", data=m_fblend)
    f.create_dataset("crop_half",  data=m_chalf)

    # Metadata
    f.attrs["n_total"]       = n_after
    f.attrs["thumb_size"]    = THUMB_SIZE
    f.attrs["filter"]        = FILTER.encode("utf-8")
    f.attrs["pixel_scale"]   = PIXEL_SCALE.encode("utf-8")
    f.attrs["n_stars"]       = int(m_fstar.sum())
    f.attrs["n_blended"]     = int(m_fblend.sum())
    f.attrs["n_dupes_removed"] = n_dupes
    f.attrs["shuffle_seed"]  = 42

size_gb = MASTER_OUTPUT_PATH.stat().st_size / 1e9
print(f"Done. {size_gb:.2f} GB  |  {n_after:,} sources")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 4: Final summary
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n{'═'*60}")
print(f"EXTRACTION COMPLETE")
print(f"{'═'*60}")
print(f"  Catalog sources         : {global_stats['total_in_catalog']:,}")
print(f"  Successfully extracted  : {global_stats['total_extracted']:,}")
print(f"  Skipped (boundary)      : {global_stats['total_skipped']:,}")
print(f"  Duplicates removed      : {n_dupes:,}")
print(f"  Final master count      : {n_after:,}")
print(f"  Stars included          : {int(m_fstar.sum()):,}")
print(f"  Blended included        : {int(m_fblend.sum()):,}")
if global_stats["tiles_missing"]:
    print(f"  Missing tiles (no file) : {global_stats['tiles_missing']}")
print(f"\n  Output: {MASTER_OUTPUT_PATH}")
print(f"  Size:   {size_gb:.2f} GB")

# Save a JSON summary alongside the HDF5 for quick reference
summary = {
    "catalog_sources"     : global_stats["total_in_catalog"],
    "extracted"           : int(n_after),
    "skipped_boundary"    : int(global_stats["total_skipped"]),
    "dupes_removed"       : int(n_dupes),
    "stars_included"      : int(m_fstar.sum()),
    "blended_included"    : int(m_fblend.sum()),
    "thumb_size"          : THUMB_SIZE,
    "filter"              : FILTER,
    "pixel_scale"         : PIXEL_SCALE,
    "missing_tiles"       : global_stats["tiles_missing"],
}
summary_path = MASTER_OUTPUT_PATH.with_suffix(".json")
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)
print(f"  Summary JSON: {summary_path}")