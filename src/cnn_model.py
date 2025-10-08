# cnn_model.py
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
import pandas as pd
from PIL import Image
import numpy as np

class HemoglobinDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        filename = self.df.iloc[idx]['filename']
        hgb = self.df.iloc[idx]['HB_LEVEL']
        try:
            img = Image.open(filename).convert("RGB")
            if self.transform:
                img = self.transform(img)
            return img, torch.FloatTensor([hgb])
        except Exception as e:
            print(f"ERROR loading image {filename}: {e}")
            img = torch.zeros(3, 224, 224)
            return img, torch.FloatTensor([hgb])

class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch)
        )
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        
    def forward(self, x):
        return self.relu(self.conv(x) + self.shortcut(x))

class ImprovedHemoglobinCNN(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            ResidualBlock(64, 128),
            nn.MaxPool2d(2),
            ResidualBlock(128, 256),
            nn.MaxPool2d(2),
            ResidualBlock(256, 512),
            nn.AdaptiveAvgPool2d(1)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

df = pd.read_csv("data_inventory.csv")
print(f"Total samples in CSV: {len(df)}")
print(f"Columns: {df.columns.tolist()}")

df['exists'] = df['filename'].apply(os.path.exists)
missing_files = df[~df['exists']]
if len(missing_files) > 0:
    print(f"WARNING: {len(missing_files)} files missing. They will be skipped.")
    # print(missing_files['filename'].tolist())

df = df[df['exists']].reset_index(drop=True)
print(f"Dataset initialized with {len(df)} samples.")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

dataset = HemoglobinDataset(df, transform=transform)
val_size = int(0.2 * len(dataset))
train_size = len(dataset) - val_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ImprovedHemoglobinCNN().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

num_epochs = 50
best_val_mae = float('inf')
early_stop_patience = 10
no_improve_epochs = 0

print("Starting training...")

for epoch in range(1, num_epochs+1):
    model.train()
    train_losses = []
    for batch_idx, (imgs, labels) in enumerate(train_loader):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

        if batch_idx % 10 == 0:
            print(f"Epoch {epoch} Batch {batch_idx} Train Loss: {loss.item():.4f}")

    model.eval()
    val_mae_list = []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            val_mae_list.append(torch.mean(torch.abs(outputs - labels)).item())
    val_mae = np.mean(val_mae_list)
    print(f"Epoch [{epoch}/{num_epochs}] Train Loss: {np.mean(train_losses):.4f}  Val MAE: {val_mae:.4f}")

    scheduler.step(val_mae)

    if val_mae < best_val_mae:
        best_val_mae = val_mae
        torch.save(model.state_dict(), "cnn_model_best.pth")
        no_improve_epochs = 0
    else:
        no_improve_epochs += 1
        if no_improve_epochs >= early_stop_patience:
            print("Early stopping triggered.")
            break

print("Training finished. Best validation MAE:", best_val_mae)
