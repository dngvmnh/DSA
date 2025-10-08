# # create_standard_files.py
# import pandas as pd
# from pathlib import Path
# import re

# df = pd.read_csv("DSA/data_inventory.csv")

# # labels.csv
# labels_data = []
# for _, row in df.iterrows():
#     labels_data.append({
#         'image_id': Path(row['filename']).stem,  
#         'hgb': row['hgb_value']
#     })
# labels_df = pd.DataFrame(labels_data)
# labels_df.to_csv('labels.csv', index=False)
# print("Saved labels.csv")

# # meta.csv
# meta_data = []
# device_id_map = {}  
# device_counter = 0

# for _, row in df.iterrows():
#     fname = row['filename']
#     stem = Path(fname).stem

#     seq_match = re.search(r'_(\d{1,2})$', stem)
#     sequence = int(seq_match.group(1)) if seq_match else None

#     ethnicity_match = re.search(r'(MiddleEastern|Chinese|EthnicityUnknown|OtherOrigin)', fname, re.IGNORECASE)
#     ethnicity = ethnicity_match.group(1) if ethnicity_match else 'Unknown'

#     device_key = ethnicity
#     if device_key not in device_id_map:
#         device_id_map[device_key] = device_counter
#         device_counter += 1
#     device_id = device_id_map[device_key]

#     individual_id = row.get('individual_id')
#     if pd.isna(individual_id):
#         individual_id = -1  

#     if sequence is None:
#         sequence = 0  

#     meta_data.append({
#         'image_id': stem,
#         'individual_id': int(individual_id),
#         'sequence': int(sequence),
#         'device_id': device_id,
#         'ethnicity': ethnicity
#     })

# meta_df = pd.DataFrame(meta_data)

# meta_df['ethnicity'] = meta_df['ethnicity'].fillna('Unknown')

# meta_df.to_csv('meta.csv', index=False)
# print("Saved meta.csv")
# print(meta_df.head(100))
# # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
# #     print(meta_df)

import os
import pandas as pd
from pathlib import Path

# --- Paths ---
anemic_dir = 'DSA/training/Anemic'
nonanemic_dir = 'DSA/training/Non-anemic'
meta_csv = 'DSA/training/Anemia_Data_Collection_Sheet.xlsx'

# --- Step 1: Gather image files ---
img_files = []
for label, subdir in [('Anemic', anemic_dir), ('Non-anemic', nonanemic_dir)]:
    for fname in os.listdir(subdir):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_files.append({
                'image_id': Path(fname).stem,         # strip extension
                'filename': os.path.join(subdir, fname),
                'label': label                        # class
            })

img_df = pd.DataFrame(img_files)
print(f"Found {len(img_df)} image files")

# --- Step 2: Load Excel metadata ---
meta_df = pd.read_excel(meta_csv)
print(f"Metadata columns: {meta_df.columns.tolist()}")

# Ensure IMAGE_ID exists in metadata
if 'IMAGE_ID' not in meta_df.columns:
    raise KeyError("The Excel file must contain a column named 'IMAGE_ID'.")

# --- Step 3: Merge images with metadata ---
df = pd.merge(img_df, meta_df, left_on='image_id', right_on='IMAGE_ID', how='inner')
print(f"Merged dataset size: {df.shape}")

# --- Step 4: Save combined inventory ---
df.to_csv('data_inventory.csv', index=False)
print("✅ Saved data_inventory.csv")

# --- Step 5: Export standard files ---
# Adjust these names based on what actually exists in your Excel
label_column = 'HGB_VALUE' if 'HGB_VALUE' in df.columns else 'label'

labels_df = pd.DataFrame({
    'image_id': df['image_id'],
    'hgb': df[label_column] if label_column in df.columns else None
})
labels_df.to_csv('labels.csv', index=False)
print("✅ Saved labels.csv")

meta_columns = [c for c in ['Age(Months)', 'GENDER', 'REMARK', 'Severity']
                if c in df.columns]

meta_out = df[['image_id'] + meta_columns] if meta_columns else df[['image_id']]
meta_out.to_csv('meta.csv', index=False)
print("✅ Saved meta.csv")

print("Preview of data_inventory.csv:")
print(df.head())
