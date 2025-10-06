# final_model_export.py
import torch
import os
import torch.nn as nn
import onnx
from model_compression import quantize_model, export_to_onnx


class LightweightHemoglobinCNN(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        # Copy exact architecture from your cnn_model.py
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

# Load your best model
model = LightweightHemoglobinCNN()
model.load_state_dict(torch.load("cnn_model.pth"))

# Check model size
model_size = os.path.getsize("cnn_model.pth") / (1024 * 1024)  # MB
print(f"Model size: {model_size:.2f} MB")

if model_size > 50:
    print("Model too large! Applying compression...")
    # Apply quantization from your model_compression.py
    model_q = quantize_model(model)
    torch.save(model_q.state_dict(), "best_model_compressed.pth")
    
    # Check size again
    compressed_size = os.path.getsize("best_model_compressed.pth") / (1024 * 1024)
    print(f"Compressed model size: {compressed_size:.2f} MB")

# Export to ONNX (required format)
sample_input = torch.randn(1, 3, 224, 224)
export_to_onnx(model, sample_input)

print("Model exported successfully!")
