#!/usr/bin/env python3

from astropy.io import fits
from astropy.table import Table
import numpy as np

CATALOG_PATH = "/extra/wayne2/preserve/nntran5/vision-transformer/data/COSMOS/COSMOSWeb_mastercatalog_v1.fits"

with fits.open(CATALOG_PATH) as hdul:
    hdul.info()

print("\nReading photometry extension...")
cat = Table.read(CATALOG_PATH, hdu=1)

print(f"\nNumber of sources: {len(cat)}")
print(f"\nColumns: {cat.colnames} total:")
for col in cat.colnames:
    print(f"  - {col:30s} ({cat[col].dtype})")