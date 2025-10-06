# data_inventory.py
import os
import re
import pandas as pd

def parse_filename(filename):
    pattern = r'(?i)(?:HgB[_-]?)?(\d+(?:\.\d+)?)g[dD][lL](?:[_-]?(?:Indi|Invi)vidual(\d+))?'

    m = re.search(pattern, filename)
    if m:
        hgb = float(m.group(1))
        indiv = int(m.group(2)) if m.group(2) else None
        return hgb, indiv
    return None, None


data_dir = "SMU/Comps/DSA/data"
valid_ext = ('.jpg', '.jpeg', '.png', '.heic', '.heif', '.tif', '.bmp')

data_inventory = []
skipped = []

for fname in os.listdir(data_dir):
    if fname.startswith('.') or os.path.isdir(os.path.join(data_dir, fname)):
        continue
    if not fname.lower().endswith(valid_ext):
        skipped.append(fname)
        continue

    hgb, indiv = parse_filename(fname)
    if hgb is not None:
        data_inventory.append({
            'filename': fname,
            'hgb_value': hgb,
            'individual_id': indiv
        })
    else:
        skipped.append(fname)

df = pd.DataFrame(data_inventory)

print(f"Matched files: {len(df)}")
print(f"Skipped files: {len(skipped)}")
if skipped:
    print("Some skipped examples:", skipped[:5])
print(f"Unique individuals: {df['individual_id'].nunique(dropna=True)}")
print(f"HgB range: {df['hgb_value'].min():.1f} – {df['hgb_value'].max():.1f} g/dL")

df.to_csv("data_inventory.csv", index=False)
print("Saved inventory to data_inventory.csv")
