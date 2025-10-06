# create_standard_files.py
import pandas as pd
from pathlib import Path
import re

df = pd.read_csv("data_inventory.csv")

# labels.csv
labels_data = []
for _, row in df.iterrows():
    labels_data.append({
        'image_id': Path(row['filename']).stem,  
        'hgb': row['hgb_value']
    })
labels_df = pd.DataFrame(labels_data)
labels_df.to_csv('labels.csv', index=False)
print("Saved labels.csv")

# meta.csv
meta_data = []
device_id_map = {}  
device_counter = 0

for _, row in df.iterrows():
    fname = row['filename']
    stem = Path(fname).stem

    seq_match = re.search(r'_(\d{1,2})$', stem)
    sequence = int(seq_match.group(1)) if seq_match else None

    ethnicity_match = re.search(r'(MiddleEastern|Chinese|EthnicityUnknown|OtherOrigin)', fname, re.IGNORECASE)
    ethnicity = ethnicity_match.group(1) if ethnicity_match else 'Unknown'

    device_key = ethnicity
    if device_key not in device_id_map:
        device_id_map[device_key] = device_counter
        device_counter += 1
    device_id = device_id_map[device_key]

    individual_id = row.get('individual_id')
    if pd.isna(individual_id):
        individual_id = -1  

    if sequence is None:
        sequence = 0  

    meta_data.append({
        'image_id': stem,
        'individual_id': int(individual_id),
        'sequence': int(sequence),
        'device_id': device_id,
        'ethnicity': ethnicity
    })

meta_df = pd.DataFrame(meta_data)

meta_df['ethnicity'] = meta_df['ethnicity'].fillna('Unknown')

meta_df.to_csv('meta.csv', index=False)
print("Saved meta.csv")
print(meta_df.head(100))
# with pd.option_context('display.max_rows', None, 'display.max_columns', None):
#     print(meta_df)