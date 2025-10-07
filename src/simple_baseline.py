# simple_baseline.py
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from PIL import Image
import pillow_heif

pillow_heif.register_heif_opener()

df = pd.read_csv("DSA/data_inventory.csv")
data_dir = "DSA/data"

def extract_color_features(image_path):
    img = np.array(Image.open(image_path))
    if img.ndim == 2:  # grayscale
        img = np.stack([img]*3, axis=-1)
    elif img.shape[2] > 3:
        img = img[:, :, :3]

    features = []
    for channel in range(3):
        features.extend([
            np.mean(img[:,:,channel]),
            np.std(img[:,:,channel]),
            np.median(img[:,:,channel])
        ])
    return features

X = [extract_color_features(f"{data_dir}/{fname}") for fname in df['filename']]
y = df['hgb_value'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

baseline_mae = mean_absolute_error(y_test, y_pred)
print(f"Baseline MAE: {baseline_mae:.2f} g/dL (Target: ≤0.8 g/dL)")

baseline_df = pd.DataFrame({
    "filename": [df['filename'].iloc[i] for i in range(len(y_test))],
    "predicted_hgb_baseline": y_pred
})

baseline_df.to_csv("baseline_predictions.csv", index=False)
print("Saved baseline predictions to baseline_predictions.csv")