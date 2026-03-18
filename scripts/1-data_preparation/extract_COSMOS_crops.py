#!/usr/bin/env python3
"""
extract_all_fits.py

Extracts thumbnails for ALL 784,016 sources in the COSMOS-Web catalog.
Applies 2D background subtraction (photutils.Background2D) per tile
before cropping, producing clean thumbnails with flat sky backgrounds.

Pipeline per tile:
    1. Load sci image
    2. Estimate + subtract 2D background (photutils Background2D)
    3. For each source: WCS-project RA/Dec → pixel, crop, resize, normalize
    4. Save as individual FITS file with metadata header

After background subtraction the sky is genuinely flat near zero, so
per-thumbnail arcsinh normalization produces clean images rather than
amplified noise.

Install requirements:
    pip install astropy photutils Pillow tqdm numpy

Runtime estimate (30mas tiles, ~12k×12k pixels each):
    Background subtraction : ~2–5 min per tile (runs before any crops)
    Thumbnail extraction   : ~5–15 min per tile
    Total for 20 tiles     : ~2–4 hours with 7 parallel workers
    (background subtraction is the bottleneck — it cannot be parallelized
     further because each tile already saturates a CPU core)

Disk usage:
    ~20 KB per file × ~700k files ≈ 14 GB
    (some sources are skipped if no signal remains after subtraction)
"""

import warnings
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS, FITSFixedWarning
from astropy.table import Table
from astropy.stats import SigmaClip
from pathlib import Path
from collections import defaultdict
from PIL import Image
from tqdm import tqdm
import time
import csv
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from photutils.background import Background2D, SExtractorBackground, StdBackgroundRMS

warnings.filterwarnings("ignore", category=FITSFixedWarning)
# photutils emits warnings about edge boxes — harmless
warnings.filterwarnings("ignore", message=".*Background2D.*")
warnings.filterwarnings("ignore", message=".*nan.*")

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

CATALOG_PATH = Path(
    "/extra/wayne2/preserve/nntran5/vision-transformer/data/COSMOS/"
    "COSMOSWeb_mastercatalog_v1.fits"
)
SCI_DIR = Path(
    "/extra/wayne2/preserve/nntran5/vision-transformer/data/COSMOS/sky_patches"
)
OUTPUT_DIR = Path(
    "/extra/wayne2/preserve/nntran5/vision-transformer/data/COSMOS/thumbnails"
)
INDEX_PATH = OUTPUT_DIR.parent / "cosmos_web_index.csv"

FILTER           = "f115w"
PIXEL_SCALE      = "30mas"
THUMB_SIZE       = 64

RADII_MULTIPLIER = 5
MIN_CROP_HALF    = 16
MAX_CROP_HALF    = 256
NUM_WORKERS      = 7

# ── Background2D parameters ───────────────────────────────────────────────────
# BOX_SIZE: size of each background estimation cell in pixels.
#   Too small → biased by individual sources
#   Too large → misses real background gradients
#   200 pixels at 30mas = 6 arcsec per box — good for JWST tiles
BKG_BOX_SIZE    = 200

# FILTER_SIZE: median filter applied to the box grid before interpolation.
#   Smooths out boxes dominated by bright stars.
#   (3,3) means each box is replaced by the median of itself + 8 neighbors.
BKG_FILTER_SIZE = 3

# SIGMA_CLIP: pixels more than 3σ from the box median are excluded.
#   This is what removes stars and galaxies from the background estimate.
BKG_SIGMA_CLIP  = 3.0

# ══════════════════════════════════════════════════════════════════════════════


def subtract_background(sci: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Estimate and subtract a 2D background from a science image.

    Uses photutils.Background2D which:
      1. Divides the image into a grid of BOX_SIZE × BOX_SIZE boxes
      2. In each box, sigma-clips to remove sources (BKG_SIGMA_CLIP × σ)
      3. Estimates the background level in each box using SExtractorBackground
         (a mode estimator: 2.5×median - 1.5×mean, robust to faint sources)
      4. Interpolates the box estimates into a smooth 2D background surface
      5. Also estimates the per-box RMS noise → 2D noise map

    Returns:
        bkg_subtracted : sci - background_2d_surface  (float32)
        rms_map        : 2D noise map in same units    (float32)
                         use rms_map[cy, cx] as σ for source at (cx, cy)
    """
    sigma_clip = SigmaClip(sigma=BKG_SIGMA_CLIP, maxiters=10)

    # Replace NaN/Inf at tile edges with 0 before background estimation
    # (NaNs appear where there was no exposure coverage)
    sci_clean = np.nan_to_num(sci, nan=0.0, posinf=0.0, neginf=0.0)

    bkg = Background2D(
        sci_clean,
        box_size        = BKG_BOX_SIZE,
        filter_size     = BKG_FILTER_SIZE,
        sigma_clip      = sigma_clip,
        bkg_estimator   = SExtractorBackground(),
        bkgrms_estimator= StdBackgroundRMS(),
        # edge_method='pad' fills edge boxes by padding rather than
        # interpolating across the boundary — more accurate at tile edges
        edge_method     = "pad",
    )

    bkg_subtracted = (sci_clean - bkg.background).astype(np.float32)
    rms_map        = bkg.background_rms.astype(np.float32)

    return bkg_subtracted, rms_map


def normalize_thumbnail(
    crop    : np.ndarray,
    rms     : float,
    n_sigma : float = 8.0,
) -> np.ndarray:
    """
    Normalize a background-subtracted crop to [0, 1].

    Because the background has been subtracted, sky pixels are genuinely
    near zero. We can now use per-thumbnail arcsinh normalization safely:
      - sky noise (±rms)           → near 0
      - n_sigma detection          → arcsinh(1.0) / arcsinh(1.0) = 1.0
      - anything brighter          → clips to 1.0

    This is different from the earlier broken normalization because:
      OLD: anchored percentiles to noise → noise filled [0,1] range
      NEW: anchored to known noise floor → noise stays near 0, signal rises

    n_sigma=8 means a source 8× the RMS above sky maps to the top of
    the brightness scale. Lower → fainter sources appear brighter.
    Raise to 12–15 if bright stars are saturating and hiding galaxy detail.
    """
    img = np.nan_to_num(crop, nan=0.0, posinf=0.0, neginf=0.0)

    # Scale: n_sigma × local RMS, with a floor to avoid division by zero
    scale = max(n_sigma * float(rms), 1e-12)

    # Arcsinh stretch: compresses dynamic range while preserving faint structure
    img = np.arcsinh(img / scale)

    # Divide by arcsinh(1.0) so the n_sigma level maps exactly to 1.0
    img = img / np.arcsinh(1.0)

    # Clip: below-sky pixels → 0, brighter-than-n_sigma → 1
    img = np.clip(img, 0.0, 1.0)

    return img.astype(np.float32)


def resize_lanczos(img: np.ndarray, size: int) -> np.ndarray:
    """Resize a [0,1] float32 2D array to (size × size) using Lanczos."""
    pil = Image.fromarray((img * 255).astype(np.uint8), mode="L")
    pil = pil.resize((size, size), Image.LANCZOS)
    return np.array(pil, dtype=np.float32) / 255.0


def make_fits_header(
    source_id, segment_id, tile, ra, dec, snr_f115w,
    a_image, b_image, theta_image, kron_rad,
    flag_star, flag_blend, crop_half, thumb_size,
    pixel_scale, filter_, local_rms,
) -> fits.Header:
    hdr = fits.Header()

    hdr["ORIGIN"]   = ("COSMOS-Web DR1",    "Survey origin")
    hdr["FILTER"]   = (filter_.upper(),     "NIRCam filter")
    hdr["PIXSCALE"] = (pixel_scale,         "Pixel scale of source mosaic")
    hdr["THUMBSZ"]  = (thumb_size,          "Thumbnail size in pixels (each side)")
    hdr["NORM"]     = ("BKG_SUB+ARCSINH",   "Background subtracted, arcsinh normalized")

    hdr["SRC_ID"]   = (source_id,           "COSMOS-Web catalog source ID")
    hdr["SEG_ID"]   = (segment_id,          "Segmentation map ID")
    hdr["TILE"]     = (tile,                "Mosaic tile this source was extracted from")
    hdr["RA"]       = (ra,                  "[deg] Right ascension (J2000)")
    hdr["DEC"]      = (dec,                 "[deg] Declination (J2000)")

    hdr["SNR_F115"] = (float(snr_f115w),    "Signal-to-noise ratio in F115W")
    hdr["A_IMAGE"]  = (float(a_image),      "[pix] Semi-major axis (SExtractor)")
    hdr["B_IMAGE"]  = (float(b_image),      "[pix] Semi-minor axis (SExtractor)")
    hdr["THETA"]    = (float(theta_image),  "[deg] Position angle (SExtractor)")
    hdr["KRONRAD"]  = (float(kron_rad),     "[pix] Kron radius")
    hdr["FSTAR"]    = (int(flag_star),      "1 = likely star, 0 = likely galaxy")
    hdr["FBLEND"]   = (int(flag_blend),     "1 = blended with neighbor, 0 = isolated")

    hdr["CROPHALF"] = (crop_half,           "[pix] Crop half-width before resize")
    hdr["CROPFULL"] = (crop_half * 2,       "[pix] Full crop width before resize")
    hdr["LRMS"]     = (float(local_rms),    "[flux] Local RMS at source position")

    orig_arcsec   = 60.0 if pixel_scale == "60mas" else 30.0
    eff_scale_deg = (crop_half * 2 / thumb_size) * orig_arcsec / 3600.0
    hdr["WCSAXES"] = 2
    hdr["CTYPE1"]  = ("RA---TAN",           "Gnomonic projection")
    hdr["CTYPE2"]  = ("DEC--TAN",           "Gnomonic projection")
    hdr["CRPIX1"]  = (thumb_size / 2.0,     "[pix] Reference pixel X (center)")
    hdr["CRPIX2"]  = (thumb_size / 2.0,     "[pix] Reference pixel Y (center)")
    hdr["CRVAL1"]  = (ra,                   "[deg] RA at reference pixel")
    hdr["CRVAL2"]  = (dec,                  "[deg] Dec at reference pixel")
    hdr["CDELT1"]  = (-eff_scale_deg,       "[deg/pix] RA scale (E left)")
    hdr["CDELT2"]  = (eff_scale_deg,        "[deg/pix] Dec scale")
    hdr["CROTA2"]  = (0.0,                  "[deg] Rotation angle")

    return hdr


def load_sci_image(sci_dir: Path, filter_: str, pixel_scale: str, tile: str):
    """
    Load sci FITS, run background subtraction, return
    (bkg_subtracted, rms_map, wcs) or (None, None, None) if missing.
    """
    fname = (
        f"mosaic_nircam_{filter_}_COSMOS-Web_"
        f"{pixel_scale}_{tile}_v1.0_sci.fits"
    )
    path = sci_dir / fname
    if not path.exists():
        return None, None, None

    with fits.open(path, memmap=False) as hdul:
        if hdul[0].data is not None and hdul[0].data.ndim == 2:
            raw = hdul[0].data.astype(np.float32)
            hdr = hdul[0].header
        elif len(hdul) > 1 and hdul[1].data is not None:
            raw = hdul[1].data.astype(np.float32)
            hdr = hdul[1].header
        else:
            return None, None, None

    wcs = WCS(hdr, naxis=2)

    # Background subtraction is the slow step (~2–5 min per tile).
    # tqdm.write is used so progress doesn't corrupt the outer bar.
    tqdm.write(f"  [{tile}] background subtraction started "
               f"(box={BKG_BOX_SIZE}px, σ-clip={BKG_SIGMA_CLIP}) ...")
    t_bkg = time.time()

    bkg_sub, rms_map = subtract_background(raw)

    tqdm.write(f"  [{tile}] background done in {time.time()-t_bkg:.0f}s  "
               f"| rms median={np.median(rms_map[rms_map>0]):.4f}")

    return bkg_sub, rms_map, wcs


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: Load catalog and group by tile
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("COSMOS-Web Full FITS Thumbnail Extraction")
print("with photutils 2D Background Subtraction")
print("=" * 60)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"\nLoading catalog: {CATALOG_PATH.name}")
t0  = time.time()
cat = Table.read(str(CATALOG_PATH), hdu=1)
print(f"Loaded {len(cat):,} sources in {time.time()-t0:.1f}s")

tile_str = np.array([
    t.decode("utf-8").strip() if isinstance(t, bytes) else str(t).strip()
    for t in cat["tile"]
])

tile_groups: dict[str, np.ndarray] = defaultdict(list)
for i, t in enumerate(tile_str):
    tile_groups[t].append(i)
tile_groups = {k: np.array(v) for k, v in tile_groups.items()}

tiles_in_catalog = sorted(tile_groups.keys())
print(f"\nTiles found: {tiles_in_catalog}")
print("Sources per tile:")
for tile in tiles_in_catalog:
    print(f"  {tile:4s}: {len(tile_groups[tile]):,}")

for tile in tiles_in_catalog:
    (OUTPUT_DIR / tile).mkdir(exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: Open master index CSV
# ══════════════════════════════════════════════════════════════════════════════

INDEX_COLUMNS = [
    "filepath", "id", "segment_id", "tile",
    "ra", "dec", "snr_f115w",
    "a_image", "b_image",
    "flag_star", "flag_blend", "crop_half",
    "local_rms",
]

index_file   = open(INDEX_PATH, "w", newline="")
index_writer = csv.DictWriter(index_file, fieldnames=INDEX_COLUMNS)
index_writer.writeheader()
csv_lock = threading.Lock()

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: Per-tile worker
# ══════════════════════════════════════════════════════════════════════════════

global_stats = {
    "total_in_catalog" : len(cat),
    "total_extracted"  : 0,
    "total_skipped"    : 0,
    "tiles_missing"    : [],
}


def process_tile(tile: str) -> tuple:
    try:
        row_indices = tile_groups[tile]
        tile_rows   = cat[row_indices]
        tile_dir    = OUTPUT_DIR / tile
        tile_t0     = time.time()

        # load_sci_image runs background subtraction internally
        bkg_sub, rms_map, tile_wcs = load_sci_image(
            SCI_DIR, FILTER, PIXEL_SCALE, tile
        )
        if bkg_sub is None:
            return (tile, 0, 0, 0.0, True, "sci file not found")

        img_h, img_w = bkg_sub.shape
        n_ok   = 0
        n_skip = 0

        for row in tile_rows:
            ra_src  = float(row["ra"])
            dec_src = float(row["dec"])

            # Project RA/Dec → pixel coordinates via tile WCS
            px, py = tile_wcs.all_world2pix(ra_src, dec_src, 0)
            cx = float(px)
            cy = float(py)

            if not (0 <= cx < img_w and 0 <= cy < img_h):
                n_skip += 1
                continue

            # Adaptive crop size from semi-major axis
            a    = max(float(row["a_image"]), 1.0)
            half = int(np.clip(
                round(a * RADII_MULTIPLIER),
                MIN_CROP_HALF,
                MAX_CROP_HALF
            ))

            cx_int = int(round(cx))
            cy_int = int(round(cy))
            x0, x1 = cx_int - half, cx_int + half
            y0, y1 = cy_int - half, cy_int + half

            if x0 < 0 or y0 < 0 or x1 > img_w or y1 > img_h:
                n_skip += 1
                continue

            # Crop from background-subtracted image
            crop = bkg_sub[y0:y1, x0:x1]
            if crop.size == 0:
                n_skip += 1
                continue

            # Local RMS at source center from the noise map
            # Clamp to image bounds for safety
            ry = int(np.clip(cy_int, 0, img_h - 1))
            rx = int(np.clip(cx_int, 0, img_w - 1))
            local_rms = float(rms_map[ry, rx])
            if local_rms <= 0:
                # Fallback: median of rms_map across the crop region
                rms_crop  = rms_map[y0:y1, x0:x1]
                local_rms = float(np.median(rms_crop[rms_crop > 0])) if np.any(rms_crop > 0) else 1e-6

            # Skip if no pixel in the central region exceeds 3σ
            # (same threshold SExtractor uses for detection)
            center      = crop[
                crop.shape[0]//4 : 3*crop.shape[0]//4,
                crop.shape[1]//4 : 3*crop.shape[1]//4
            ]
            center_peak = float(np.nanmax(center))
            if center_peak < 3.0 * local_rms:
                n_skip += 1
                continue

            # Normalize using local noise floor, then resize
            thumb_norm   = normalize_thumbnail(crop, local_rms, n_sigma=8.0)
            thumbnail    = resize_lanczos(thumb_norm, THUMB_SIZE)

            source_id   = int(row["id"])
            segment_id  = int(row["segment-id"])
            ra          = float(row["ra"])
            dec         = float(row["dec"])
            snr_f115w   = float(row["snr_f115w"])
            a_image     = float(row["a_image"])
            b_image     = float(row["b_image"])
            theta_image = float(row["theta_image"])
            kron_rad    = float(row["kron_rad"])
            flag_star   = bool(row["flag_star"])
            flag_blend  = bool(row["flag_blend"])

            hdr = make_fits_header(
                source_id=source_id, segment_id=segment_id, tile=tile,
                ra=ra, dec=dec, snr_f115w=snr_f115w,
                a_image=a_image, b_image=b_image,
                theta_image=theta_image, kron_rad=kron_rad,
                flag_star=flag_star, flag_blend=flag_blend,
                crop_half=half, thumb_size=THUMB_SIZE,
                pixel_scale=PIXEL_SCALE, filter_=FILTER,
                local_rms=local_rms,
            )

            fname    = f"COSMOS_{source_id:07d}_{tile}.fits"
            out_path = tile_dir / fname
            fits.PrimaryHDU(data=thumbnail, header=hdr).writeto(
                str(out_path), overwrite=True
            )

            with csv_lock:
                index_writer.writerow({
                    "filepath"   : f"{tile}/{fname}",
                    "id"         : source_id,
                    "segment_id" : segment_id,
                    "tile"       : tile,
                    "ra"         : f"{ra:.8f}",
                    "dec"        : f"{dec:.8f}",
                    "snr_f115w"  : f"{snr_f115w:.4f}",
                    "a_image"    : f"{a_image:.4f}",
                    "b_image"    : f"{b_image:.4f}",
                    "flag_star"  : int(flag_star),
                    "flag_blend" : int(flag_blend),
                    "crop_half"  : half,
                    "local_rms"  : f"{local_rms:.6f}",
                })
                if n_ok % 1000 == 0:
                    index_file.flush()

            n_ok += 1

        del bkg_sub, rms_map, tile_wcs
        elapsed = time.time() - tile_t0
        return (tile, n_ok, n_skip, elapsed, False, None)

    except Exception:
        import traceback
        return (tile, 0, 0, 0.0, False, traceback.format_exc())


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4: Run all tiles in parallel
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n{'─'*60}")
print(f"Extracting with background subtraction  ({NUM_WORKERS} workers)")
print(f"Background: box={BKG_BOX_SIZE}px  filter={BKG_FILTER_SIZE}  "
      f"σ-clip={BKG_SIGMA_CLIP}")
print(f"Output: {OUTPUT_DIR}")
print(f"{'─'*60}")
print(f"\nNOTE: Background subtraction takes ~2–5 min per tile.")
print(f"      You will see per-tile 'background done' messages as they finish.")
print(f"      The overall progress bar updates only when a full tile completes.\n")

with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
    futures = {
        executor.submit(process_tile, tile): tile
        for tile in tiles_in_catalog
    }

    with tqdm(total=len(tiles_in_catalog), desc="Tiles complete",
              unit="tile", position=0) as pbar:
        for future in as_completed(futures):
            tile, n_ok, n_skip, elapsed, missing, error = future.result()

            if missing:
                tqdm.write(f"  [{tile}] ✗ sci file not found")
                global_stats["tiles_missing"].append(tile)
            elif error:
                tqdm.write(f"  [{tile}] ✗ CRASHED:\n{error}")
                global_stats["tiles_missing"].append(tile)
            else:
                mins = elapsed / 60
                rate = n_ok / elapsed if elapsed > 0 else 0
                tqdm.write(
                    f"  [{tile}] ✓ {n_ok:,} saved  |  "
                    f"{n_skip:,} skipped  |  "
                    f"{mins:.1f} min  ({rate:.0f} files/sec)"
                )
                global_stats["total_extracted"] += n_ok
                global_stats["total_skipped"]   += n_skip

            pbar.update(1)

index_file.close()

# ══════════════════════════════════════════════════════════════════════════════
# STEP 5: Summary
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n{'═'*60}")
print(f"EXTRACTION COMPLETE")
print(f"{'═'*60}")
print(f"  Catalog sources    : {global_stats['total_in_catalog']:,}")
print(f"  FITS files written : {global_stats['total_extracted']:,}")
print(f"  Skipped            : {global_stats['total_skipped']:,}")
if global_stats["tiles_missing"]:
    print(f"  Problem tiles      : {global_stats['tiles_missing']}")
print(f"\n  Output : {OUTPUT_DIR}")
print(f"  Index  : {INDEX_PATH}")

n_files = global_stats["total_extracted"]
print(f"\n  Estimated disk usage: {n_files * 20_000 / 1e9:.1f} GB")

summary = {
    "total_extracted"    : global_stats["total_extracted"],
    "total_skipped"      : global_stats["total_skipped"],
    "thumb_size"         : THUMB_SIZE,
    "filter"             : FILTER,
    "pixel_scale"        : PIXEL_SCALE,
    "background"         : {
        "tool"           : "photutils.Background2D",
        "box_size"       : BKG_BOX_SIZE,
        "filter_size"    : BKG_FILTER_SIZE,
        "sigma_clip"     : BKG_SIGMA_CLIP,
        "estimator"      : "SExtractorBackground",
    },
    "normalization"      : "arcsinh anchored to local RMS, n_sigma=8",
    "output_dir"         : str(OUTPUT_DIR),
    "index_csv"          : str(INDEX_PATH),
    "missing_tiles"      : global_stats["tiles_missing"],
}
summary_path = OUTPUT_DIR.parent / "extraction_summary.json"
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)
print(f"  Summary: {summary_path}")