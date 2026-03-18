''' 
script to get highest SNR galaxies for model training subset
'''

import pandas as pd
import numpy as np

df = pd.read_csv('/extra/wayne2/preserve/nntran5/vision-transformer/data/COSMOS/cosmos_web_index.csv')      # hard-coded path to .csv produced by extract_COSMOS_crops.py

candidates = df[
    (df['snr_f115w'] > 20) &
    (df['flag_star'] == 0) &
    (df['a_image'] >= 3.0)
].copy()

candidates = candidates.nlargest(10_000, 'snr_f115w')
candidates.to_csv('/extra/wayne2/preserve/nntran5/vision-transformer/output/COSMOS/10000_highest_SNR_galaxies.csv', index=False)
print('Script complete.')