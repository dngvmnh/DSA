# model_compression.py
import torch.quantization
import torch.nn as nn

def quantize_model(model, representative_data):
    """Quantize model for mobile deployment"""
    model.eval()
    
    # Post-training quantization
    model_q = torch.quantization.quantize_dynamic(
        model, {nn.Linear, nn.Conv2d}, dtype=torch.qint8
    )
    
    return model_q

def export_to_onnx(model, sample_input):
    """Export to ONNX for cross-platform deployment"""
    torch.onnx.export(
        model,
        sample_input,
        "hemoglobin_model.onnx",
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['image', 'metadata'],
        output_names=['hemoglobin'],
        dynamic_axes={
            'image': {0: 'batch_size'},
            'metadata': {0: 'batch_size'},
            'hemoglobin': {0: 'batch_size'}
        }
    )
