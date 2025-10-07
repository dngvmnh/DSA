# multimodal_model.py
import torch
import torch.nn as nn
from cnn_model import LightweightHemoglobinCNN  

class MultiModalHemoglobinModel(nn.Module):
    def __init__(self, metadata_input_dim=10):
        super().__init__()
        
        self.image_encoder = LightweightHemoglobinCNN()
        self.image_encoder.classifier = nn.Identity()  
        
        self.metadata_encoder = nn.Sequential(
            nn.Linear(metadata_input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        
        self.fusion = nn.Sequential(
            nn.Linear(256 + 16, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, image, metadata):
        img_features = self.image_encoder(image)
        meta_features = self.metadata_encoder(metadata)
        
        combined = torch.cat([img_features, meta_features], dim=1)
        output = self.fusion(combined)
        return output
