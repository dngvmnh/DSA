# inference.py
import os
import pandas as pd
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import pillow_heif
import argparse
from pathlib import Path
import numpy as np

class LightweightHemoglobinCNN(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, groups=32, padding=1),
            nn.Conv2d(32, 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, groups=64, padding=1),
            nn.Conv2d(64, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 128, 3, groups=128, padding=1),
            nn.Conv2d(128, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def predict(images_dir, meta_csv, output_file="predictions.csv"):
    """
    Main prediction function required by competition
    Args:
        images_dir: Directory containing test images
        meta_csv: Path to metadata CSV file
        output_file: Where to save predictions
    """
    pillow_heif.register_heif_opener()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LightweightHemoglobinCNN().to(device)
    model.load_state_dict(torch.load("cnn_model.pth", map_location=device))
    model.eval()
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    if os.path.exists(meta_csv):
        meta_df = pd.read_csv(meta_csv)
    else:
        meta_df = pd.DataFrame()
    
    image_files = []
    for ext in ['.jpg', '.jpeg', '.png', '.heic', '.heif']:
        image_files.extend(Path(images_dir).glob(f"*{ext}"))
        image_files.extend(Path(images_dir).glob(f"*{ext.upper()}"))
    
    predictions = []
    
    with torch.no_grad():
        for img_path in image_files:
            try:
                img = Image.open(img_path).convert("RGB")
                img_tensor = transform(img).unsqueeze(0).to(device)
                
                output = model(img_tensor)
                pred_hgb = output.cpu().numpy()[0, 0]
                
                predictions.append({
                    'image_id': img_path.stem,
                    'predicted_hgb': float(pred_hgb)
                })
                
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                predictions.append({
                    'image_id': img_path.stem,
                    'predicted_hgb': 12.0  
                })
    
    pred_df = pd.DataFrame(predictions)
    pred_df.to_csv(output_file, index=False)
    print(f"Saved {len(predictions)} predictions to {output_file}")
    
    return pred_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True, help="Path to images directory")
    parser.add_argument("--meta", required=True, help="Path to metadata CSV")
    parser.add_argument("--out", default="predictions.csv", help="Output file")
    
    args = parser.parse_args()
    
    predict(args.images, args.meta, args.out)
