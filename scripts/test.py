# test_one_tile.py

import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from astropy.io import fits
from astropy.wcs import WCS, FITSFixedWarning
from astropy.table import Table
from astropy.stats import SigmaClip
from photutils.background import Background2D, SExtractorBackground, StdBackgroundRMS
from PIL import Image
import time

warnings.filterwarnings("ignore", category=FITSFixedWarning)

CATALOG_PATH = "/extra/wayne2/preserve/nntran5/vision-transformer/data/COSMOS/COSMOSWeb_mastercatalog_v1.fits"
SCI_PATH     = "/extra/wayne2/preserve/nntran5/vision-transformer/data/COSMOS/sky_patches/mosaic_nircam_f115w_COSMOS-Web_30mas_A1_v1.0_sci.fits"
THUMB_SIZE   = 64
N_SHOW       = 16    # thumbnails to show in the output grid

print("Loading sci image...")
with fits.open(SCI_PATH, memmap=False) as hdul:
    sci = hdul[0].data.astype(np.float32)
    wcs = WCS(hdul[0].header, naxis=2)
print(f"  Shape: {sci.shape}  |  dtype: {sci.dtype}")
print(f"  Raw pixel stats — median: {np.nanmedian(sci):.4f}  std: {np.nanstd(sci):.4f}")

print("\nRunning Background2D (this is the slow step)...")
t0         = time.time()
sci_clean  = np.nan_to_num(sci, nan=0.0, posinf=0.0, neginf=0.0)
sigma_clip = SigmaClip(sigma=3.0, maxiters=10)
bkg        = Background2D(
    sci_clean,
    box_size         = 200,
    filter_size      = 3,
    sigma_clip       = sigma_clip,
    bkg_estimator    = SExtractorBackground(),
    bkgrms_estimator = StdBackgroundRMS(),
    edge_method      = "pad",
)
bkg_sub  = (sci_clean - bkg.background).astype(np.float32)
rms_map  = bkg.background_rms.astype(np.float32)
elapsed  = time.time() - t0

print(f"  Done in {elapsed:.1f}s")
print(f"  Background median : {np.median(bkg.background):.4f}")
print(f"  RMS map median    : {np.median(rms_map[rms_map>0]):.4f}")
print(f"  Subtracted image  — median: {np.nanmedian(bkg_sub):.4f}  "
      f"std: {np.nanstd(bkg_sub):.4f}")
print(f"\n  Estimated time for all 20 tiles: {elapsed*20/60:.0f}–{elapsed*20*1.5/60:.0f} min "
      f"(single-threaded), less with parallelism")

# Load catalog — get A1 sources sorted by SNR
print("\nLoading catalog and extracting sample thumbnails...")
cat      = Table.read(CATALOG_PATH, hdu=1)
tile_str = np.array([t.decode().strip() if isinstance(t, bytes) else str(t).strip() for t in cat["tile"]])
a1       = cat[tile_str == "A1"]
# Sample across the SNR range — not just the top
snr      = a1["snr_f115w"]
bins     = np.percentile(snr, np.linspace(20, 100, N_SHOW + 1))
sample_rows = []
for lo, hi in zip(bins[:-1], bins[1:]):
    mask = (snr >= lo) & (snr < hi)
    if mask.sum() > 0:
        idx = np.where(mask)[0][0]
        sample_rows.append(a1[idx])

img_h, img_w = bkg_sub.shape

def make_thumb(row):
    px, py = wcs.all_world2pix(float(row["ra"]), float(row["dec"]), 0)
    cx, cy = float(px), float(py)
    if not (0 <= cx < img_w and 0 <= cy < img_h):
        return None, None

    a    = max(float(row["a_image"]), 1.0)
    half = int(np.clip(round(a * 5), 16, 256))
    x0, x1 = int(round(cx))-half, int(round(cx))+half
    y0, y1 = int(round(cy))-half, int(round(cy))+half
    if x0 < 0 or y0 < 0 or x1 > img_w or y1 > img_h:
        return None, None

    crop      = bkg_sub[y0:y1, x0:x1]
    ry        = int(np.clip(int(round(cy)), 0, img_h-1))
    rx        = int(np.clip(int(round(cx)), 0, img_w-1))
    local_rms = float(rms_map[ry, rx])
    if local_rms <= 0:
        return None, None

    scale = max(8.0 * local_rms, 1e-12)
    norm  = np.arcsinh(crop / scale) / np.arcsinh(1.0)
    norm  = np.clip(norm, 0, 1).astype(np.float32)

    pil   = Image.fromarray((norm * 255).astype(np.uint8), mode="L")
    thumb = np.array(pil.resize((THUMB_SIZE, THUMB_SIZE), Image.LANCZOS)) / 255.0
    return thumb, float(row["snr_f115w"])

# Build the grid
fig = plt.figure(figsize=(16, 10))
fig.patch.set_facecolor("black")
gs  = gridspec.GridSpec(2, N_SHOW // 2, figure=fig, hspace=0.05, wspace=0.05)

n_shown = 0
for i, row in enumerate(sample_rows):
    if n_shown >= N_SHOW:
        break
    thumb, snr_val = make_thumb(row)
    if thumb is None:
        continue
    r, c = divmod(n_shown, N_SHOW // 2)
    ax   = fig.add_subplot(gs[r, c])
    ax.imshow(thumb, cmap="gray", origin="lower", vmin=0, vmax=1)
    ax.set_title(f"SNR={snr_val:.0f}", color="white", fontsize=7, pad=2)
    ax.axis("off")
    n_shown += 1

plt.suptitle(
    "Background-subtracted thumbnails — SNR range sampled low→high (left→right)",
    color="white", fontsize=11, y=1.01
)
plt.savefig("test_bkg_sub.png", dpi=120, facecolor="black", bbox_inches="tight")
print(f"\nGrid saved: test_bkg_sub.png")
print("scp it to your laptop:")
print("  scp yournetid@server:/path/to/test_bkg_sub.png .")