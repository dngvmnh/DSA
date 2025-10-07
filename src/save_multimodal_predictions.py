# save_multimodal_predictions.py
import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from multimodal_model import MultiModalHemoglobinModel  
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MultiModalDataset(Dataset):
    def __init__(self, df, data_dir):
        self.df = df
        self.data_dir = data_dir
        self.metadata_array = np.zeros((len(df), 1), dtype=np.float32)
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        image = torch.randn(3, 224, 224)  
        metadata = torch.tensor(self.metadata_array[idx])
        target = torch.tensor(self.df.iloc[idx]['hgb_value'], dtype=torch.float32)
        return image, metadata, target

df = pd.read_csv("data_inventory.csv")
dataset = MultiModalDataset(df, data_dir="SMU/Comps/DSA/data")
loader = DataLoader(dataset, batch_size=16, shuffle=False)

model = MultiModalHemoglobinModel(metadata_input_dim=1)  
model.load_state_dict(torch.load("multimodal_model.pth", map_location=device))
model.to(device)
model.eval()

predictions = []
filenames = df['filename'].values

with torch.no_grad():
    for images, metadata, _ in loader:
        images = images.to(device)
        metadata = metadata.to(device)
        outputs = model(images, metadata)
        predictions.extend(outputs.cpu().numpy().flatten())

pred_df = pd.DataFrame({
    'filename': filenames,
    'predicted_hgb_multi': predictions
})

pred_df.to_csv("multimodal_predictions.csv", index=False)
print("Saved multimodal predictions to multimodal_predictions.csv")
