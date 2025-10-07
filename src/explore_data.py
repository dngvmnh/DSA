# explore_data.py
import pandas as pd
from PIL import Image
import pillow_heif
import matplotlib.pyplot as plt
import numpy as np
import os

pillow_heif.register_heif_opener()  

df = pd.read_csv("data_inventory.csv")
sample_files = df.sample(30)['filename'].tolist()

os.makedirs("plots", exist_ok=True)

for filename in sample_files:
    img_path = f"DSA/data/{filename}"

    try:
        img = np.array(Image.open(img_path))
    except Exception as e:
        print(f"Warning: Could not read {img_path} — {e}")
        continue

    hgb = df[df['filename'] == filename]['hgb_value'].iloc[0]

    print(f"File: {filename}")
    print(f"HgB: {hgb} g/dL")
    print(f"Image shape: {img.shape}")
    print(f"Image dtype: {img.dtype}")
    print("---")

    plt.imshow(img)
    plt.axis('off')
    plt.title(f"{filename} — HgB: {hgb} g/dL")
    plt.savefig(f"plots/{filename}.png", bbox_inches='tight')
    plt.close()
