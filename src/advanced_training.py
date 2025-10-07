# advanced_training.py
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import GroupKFold
import torch.optim.lr_scheduler as lr_scheduler

def create_stratified_splits(df):
    df['hgb_bin'] = pd.cut(df['hgb_value'], 
                            bins=[0, 8, 10, 12, 15, 20], 
                            labels=['very_low', 'low', 'normal', 'high', 'very_high'])
    
    gkf = GroupKFold(n_splits=5)
    splits = list(gkf.split(df, df['hgb_bin'], groups=df['individual_id']))
    return splits

class FocalMAELoss(nn.Module):
    def __init__(self, alpha=2.0):
        super().__init__()
        self.alpha = alpha
        
    def forward(self, pred, target):
        mae = torch.abs(pred - target)
        focal_weight = torch.pow(mae / (mae.max() + 1e-8), self.alpha)
        return torch.mean(focal_weight * mae)

def train_model(model, train_loader, val_loader, epochs=50):
    criterion = FocalMAELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_val_mae = float('inf')
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        for images, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        model.eval()
        val_mae = 0
        with torch.no_grad():
            n = 0
            for images, targets in val_loader:
                outputs = model(images)
                val_mae += torch.sum(torch.abs(outputs - targets)).item()
                n += targets.numel()
        val_mae /= n
        
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save(model.state_dict(), 'advanced_model.pth')
        
        scheduler.step()
        print(f"Epoch {epoch+1}: Train Loss: {train_loss/len(train_loader):.4f}, Val MAE: {val_mae:.4f}")

if __name__ == "__main__":
    from torch.utils.data import DataLoader, TensorDataset

    # Dummy data for testing
    X = torch.randn(100, 3, 224, 224)  # 100 RGB images 224x224
    y = torch.randn(100, 1)            # 100 target values

    # Split into train/val
    train_dataset = TensorDataset(X[:80], y[:80])
    val_dataset   = TensorDataset(X[80:], y[80:])

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=8)

    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(3*224*224, 128),
        nn.ReLU(),
        nn.Linear(128, 1)
    )

    train_model(model, train_loader, val_loader, epochs=50)
