# enhanced_features.py
import cv2
import numpy as np
from skimage import feature, color
import pandas as pd
from PIL import Image
import pillow_heif
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor

pillow_heif.register_heif_opener()
df = pd.read_csv("data_inventory.csv")
data_dir = "DSA/data"
y = df['hgb_value'].values

def extract_advanced_features(image_path):
    """Extract comprehensive features from lip images"""
    img = np.array(Image.open(image_path))
    
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    
    features = []
    
    for channel in range(3):
        features.extend([
            np.mean(img[:,:,channel]),
            np.std(img[:,:,channel]),
            np.median(img[:,:,channel]),
            np.percentile(img[:,:,channel], 25),
            np.percentile(img[:,:,channel], 75)
        ])
    
    for channel in range(3):
        features.extend([
            np.mean(hsv[:,:,channel]),
            np.std(hsv[:,:,channel])
        ])
    
    for channel in range(3):
        features.extend([
            np.mean(lab[:,:,channel]),
            np.std(lab[:,:,channel])
        ])
    
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    lbp = feature.local_binary_pattern(gray, 24, 3)
    features.extend([
        np.mean(lbp),
        np.std(lbp)
    ])
    
    red_green_ratio = np.mean(img[:,:,0]) / (np.mean(img[:,:,1]) + 1e-8)
    red_blue_ratio = np.mean(img[:,:,0]) / (np.mean(img[:,:,2]) + 1e-8)
    features.extend([red_green_ratio, red_blue_ratio])
    
    return features

df = pd.read_csv("data_inventory.csv")
X_advanced = [extract_advanced_features(f"{data_dir}/{fname}") for fname in df['filename']]

X_train, X_test, y_train, y_test = train_test_split(X_advanced, y, test_size=0.2, random_state=42)

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
advanced_mae = mean_absolute_error(y_test, y_pred_rf)
print(f"Advanced Features MAE: {advanced_mae:.2f} g/dL")
