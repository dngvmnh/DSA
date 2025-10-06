# cnn_model_evaluate.py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import pandas as pd
from PIL import Image
import pillow_heif
import numpy as np
from pathlib import Path

pillow_heif.register_heif_opener()

class HemoglobinDataset(Dataset):
    def __init__(self, df, data_dir, transform=None):
        self.df = df
        self.data_dir = data_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        filename = self.df.iloc[idx]['filename']
        img = Image.open(f"{self.data_dir}/{filename}").convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, filename  

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

data_dir = "SMU/Comps/DSA/data"
df = pd.read_csv("data_inventory.csv")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

dataset = HemoglobinDataset(df, data_dir, transform)
loader = DataLoader(dataset, batch_size=16, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LightweightHemoglobinCNN().to(device)
model.load_state_dict(torch.load("cnn_model.pth", map_location=device))
# model.load_state_dict(torch.load("multimodal_model.pth", map_location=device))
# model.load_state_dict(torch.load("advanced_model.pth", map_location=device))
model.eval()

predictions = []
with torch.no_grad():
    for imgs, filenames in loader:
        imgs = imgs.to(device)
        outputs = model(imgs)
        for fname, pred in zip(filenames, outputs.cpu().numpy()):
            predictions.append({
                "filename": fname,
                "predicted_hgb": float(pred[0])
            })

pred_df = pd.DataFrame(predictions)
pred_df.to_csv("cnn_predictions.csv", index=False)
print("Saved predictions to cnn_predictions.csv")
print(pred_df.head(30))
