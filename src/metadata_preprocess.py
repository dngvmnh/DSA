# metadata_preprocess.py
import pandas as pd
import numpy as np

meta_df = pd.read_csv("DSA/meta.csv")

meta_df['sequence'] = meta_df['sequence'].fillna(0)  # unknown sequence as 0
meta_df['device_id'] = meta_df['device_id'].fillna(-1)  # unknown device
meta_df['ethnicity '] = meta_df['ethnicity'].fillna('Unknown')

ethnicity_map = {eth: idx for idx, eth in enumerate(meta_df['ethnicity'].unique())}
meta_df['ethnicity_encoded'] = meta_df['ethnicity'].map(ethnicity_map)

metadata_features = ['device_id', 'sequence', 'ethnicity_encoded']
metadata = meta_df[metadata_features].values.astype(np.float32)

metadata = (metadata - metadata.min(axis=0)) / (metadata.ptp(axis=0) + 1e-8)

print("Metadata shape:", metadata.shape)
print(meta_df[metadata_features].head(30))
