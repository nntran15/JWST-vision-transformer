#!/usr/bin/env python3

import subprocess
from pathlib import Path

# ===================== global variables =====================
DOWNLOAD_DIR = Path("/extra/wayne2/preserve/nntran5/vision-transformer/data/COSMOS")
PIXEL_SCALE = "30mas"
FILTER = "f115w"
TILES = ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9", "A10",
         "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9", "B10"]

SEGMENTATION_MAP_URL = "https://cosmos2025:780kgalaxies!@cosmos2025.iap.fr/data/catalog/segmentation_maps/segmentation_maps.tar.gz"
TILE_URL_TEMPLATE = "https://cosmos2025.iap.fr/data/nircam/extensions/mosaic_nircam_{filter}_COSMOS-Web_{pixel_scale}_{tile}_v1.0_sci.fits.gz"  
    # https://cosmos2025.iap.fr/data/nircam/extensions/mosaic_nircam_f115w_COSMOS-Web_30mas_A1_v1.0_sci.fits.gz
    # https://cosmos2025.iap.fr/data/nircam/extensions/mosaic_nircam_f115w_COSMOS-Web_30mas_A2_v1.0_sci.fits.gz
    # https://cosmos2025.iap.fr/data/nircam/extensions/mosaic_nircam_f115w_COSMOS-Web_30mas_B1_v1.0_sci.fits.gz


# ===================== functions =====================
def wget(url, output_path):
    output_path.mkdir(parents=True, exist_ok=True)
    cmd = ["wget", "-c", "--progress=bar:force", "-P", str(output_path), url]
    print(f"\nDownloading {url}...")
    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"Error downloading {url}")
    else:
        print(f"Successfully downloaded {url}")

# ===================== main =====================
def main():
    # download segmentation map
    segmentation_directory = DOWNLOAD_DIR / "segmentation_maps"
    wget(SEGMENTATION_MAP_URL, segmentation_directory)

    tarball = segmentation_directory / "segmentation_maps.tar.gz"
    if tarball.exists():
        print(f"Extracting {tarball}...")
        subprocess.run(["tar", "-xzf", str(tarball), "-C", str(segmentation_directory)])
        tarball.unlink()

    # download tiles
    for tile in TILES:
        tile_url = TILE_URL_TEMPLATE.format(filter=FILTER, pixel_scale=PIXEL_SCALE, tile=tile)
        wget(tile_url, DOWNLOAD_DIR)

    print("\n\nAll downloads completed!")

if __name__ == "__main__":
    main()