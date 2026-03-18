'''
copies all .fits files from extract_highest_SNR.py .csv output file to some directory
'''

import pandas as pd
import numpy as np
import shutil
import os

df = pd.read_csv('/extra/wayne2/preserve/nntran5/vision-transformer/output/COSMOS/10000_highest_SNR_galaxies.csv')
os.makedirs('/extra/wayne2/preserve/nntran5/vision-transformer/output/COSMOS/10000_highest_SNR_galaxies', exist_ok=True)

for index, row in df.iterrows():
    source_path = f'/extra/wayne2/preserve/nntran5/vision-transformer/data/COSMOS/thumbnails/{row["filepath"]}'
    filename = os.path.basename(source_path)
    destination_path = f'/extra/wayne2/preserve/nntran5/vision-transformer/output/COSMOS/10000_highest_SNR_galaxies/{filename}'
    shutil.copy(source_path, destination_path)

print('Script complete.')
