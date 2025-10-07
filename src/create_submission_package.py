# create_submission_package.py
import os
import shutil
from pathlib import Path

def create_submission_structure():
    """Create the exact folder structure required by competition"""
    
    dirs_to_create = [
        "submission/code",
        "submission/weights", 
        "submission/docs"
    ]
    
    for dir_path in dirs_to_create:
        os.makedirs(dir_path, exist_ok=True)
    
    files_to_copy = {
        "inference.py": "submission/code/",
        "cnn_model.py": "submission/code/",  
        "multimodal_model.py": "submission/code/",
        "requirements.txt": "submission/",
        
        "best_model.pth": "submission/weights/final_model.pth",
        "hemoglobin_model.onnx": "submission/weights/",
        
        "model_card.md": "submission/",
    }
    
    for src, dst in files_to_copy.items():
        if os.path.exists(src):
            if os.path.isfile(dst):
                shutil.copy2(src, dst)
            else:
                shutil.copy2(src, os.path.join(dst, os.path.basename(src)))
    
    print("Submission package created!")
    
    print("\nTesting one-command inference...")
    test_cmd = 'python submission/code/inference.py --images test_data --meta test_meta.csv --out test_preds.csv'
    print(f"Test command: {test_cmd}")

create_submission_structure()
