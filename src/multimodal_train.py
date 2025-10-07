# multimodal_train.py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pillow_heif
import pandas as pd
import numpy as np
from pathlib import Path

from multimodal_model import MultiModalHemoglobinModel  
from cnn_model import LightweightHemoglobinCNN

pillow_heif.register_heif_opener()

df = pd.read_csv("DSA/data_inventory.csv")
data_dir = "DSA/data"

meta_df = pd.read_csv("DSA/meta.csv")

df['image_id'] = df['filename'].apply(lambda x: Path(x).stem)
meta_df['image_id'] = meta_df['image_id'].apply(lambda x: str(x).strip())

meta_df_aligned = meta_df.set_index('image_id').reindex(df['image_id']).reset_index()

metadata_features = ['device_id'] 
metadata = meta_df_aligned[metadata_features].fillna(0.0).values.astype(np.float32)

metadata = (metadata - metadata.mean(axis=0)) / (metadata.std(axis=0) + 1e-8)

class MultiModalDataset(Dataset):
    def __init__(self, df, metadata, data_dir, transform=None):
        self.df = df
        self.metadata = torch.tensor(metadata, dtype=torch.float32)
        self.data_dir = data_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        fname = self.df.iloc[idx]['filename']
        hgb = self.df.iloc[idx]['hgb_value']
        
        try:
            img = Image.open(f"{self.data_dir}/{fname}").convert("RGB")
        except:
            img = Image.new("RGB", (224, 224), (0, 0, 0))
        
        if self.transform:
            img = self.transform(img)
        
        return img, self.metadata[idx], torch.tensor([hgb], dtype=torch.float32)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

dataset = MultiModalDataset(df, metadata, data_dir, transform)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MultiModalHemoglobinModel(metadata_input_dim=metadata.shape[1]).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

num_epochs = 50
best_val_mae = float('inf')

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, meta, targets in train_loader:
        images, meta, targets = images.to(device), meta.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(images, meta)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    
    epoch_loss = running_loss / len(train_loader.dataset)
    
    model.eval()
    val_preds, val_truths = [], []
    with torch.no_grad():
        for images, meta, targets in test_loader:
            images, meta, targets = images.to(device), meta.to(device), targets.to(device)
            outputs = model(images, meta)
            val_preds.extend(outputs.cpu().numpy())
            val_truths.extend(targets.cpu().numpy())
    
    val_preds = np.array(val_preds).flatten()
    val_truths = np.array(val_truths).flatten()
    val_mae = np.mean(np.abs(val_preds - val_truths))
    
    if val_mae < best_val_mae:
        best_val_mae = val_mae
        torch.save(model.state_dict(), "multimodal_model.pth")
    
    print(f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {epoch_loss:.4f}  Val MAE: {val_mae:.4f}")

print(f"Training finished. Best validation MAE: {best_val_mae:.4f}")
