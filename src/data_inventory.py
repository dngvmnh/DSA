# # data_inventory.py
# import os
# import re
# import pandas as pd

# def parse_filename(filename):
#     pattern = r'(?i)(?:HgB[_-]?)?(\d+(?:\.\d+)?)g[dD][lL](?:[_-]?(?:Indi|Invi)vidual(\d+))?'

#     m = re.search(pattern, filename)
#     if m:
#         hgb = float(m.group(1))
#         indiv = int(m.group(2)) if m.group(2) else None
#         return hgb, indiv
#     return None, None


# data_dir = "DSA/data"
# valid_ext = ('.jpg', '.jpeg', '.png', '.heic', '.heif', '.tif', '.bmp')

# data_inventory = []
# skipped = []

# for fname in os.listdir(data_dir):
#     if fname.startswith('.') or os.path.isdir(os.path.join(data_dir, fname)):
#         continue
#     if not fname.lower().endswith(valid_ext):
#         skipped.append(fname)
#         continue

#     hgb, indiv = parse_filename(fname)
#     if hgb is not None:
#         data_inventory.append({
#             'filename': fname,
#             'hgb_value': hgb,
#             'individual_id': indiv
#         })
#     else:
#         skipped.append(fname)

# df = pd.DataFrame(data_inventory)

# print(f"Matched files: {len(df)}")
# print(f"Skipped files: {len(skipped)}")
# if skipped:
#     print("Some skipped examples:", skipped[:5])
# print(f"Unique individuals: {df['individual_id'].nunique(dropna=True)}")
# print(f"HgB range: {df['hgb_value'].min():.1f} – {df['hgb_value'].max():.1f} g/dL")

# df.to_csv("data_inventory.csv", index=False)
# print("Saved inventory to data_inventory.csv")

import os
import pandas as pd
from pathlib import Path

anemic_dir = 'DSA/training/Anemic'
nonanemic_dir = 'DSA/training/Non-anemic'
meta_csv = 'DSA/training/Anemia_Data_Collection_Sheet.xlsx'

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

meta_df = pd.read_excel(meta_csv)
print(f"Metadata columns: {meta_df.columns.tolist()}")

if 'IMAGE_ID' not in meta_df.columns:
    raise KeyError("The Excel file must contain a column named 'IMAGE_ID'.")

df = pd.merge(img_df, meta_df, left_on='image_id', right_on='IMAGE_ID', how='inner')
print(f"Merged dataset size: {df.shape}")

df.to_csv('data_inventory.csv', index=False)
print("Saved data_inventory.csv")

label_column = 'HGB_VALUE' if 'HGB_VALUE' in df.columns else 'label'

labels_df = pd.DataFrame({
    'image_id': df['image_id'],
    'hgb': df[label_column] if label_column in df.columns else None
})
labels_df.to_csv('labels.csv', index=False)
print("Saved labels.csv")

meta_columns = [c for c in ['Age(Months)', 'GENDER', 'REMARK', 'Severity']
                if c in df.columns]

meta_out = df[['image_id'] + meta_columns] if meta_columns else df[['image_id']]
meta_out.to_csv('meta.csv', index=False)
print("Saved meta.csv")

print("Preview of data_inventory.csv:")
print(df.head(100))

