# master_pipeline.py - Python orchestrator with error handling
import os
import sys
import subprocess
import logging
import time
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler()
    ]
)

def run_command(command, stage_name, required=True):
    """Execute a Python command with error handling"""
    logging.info(f"🔄 {stage_name}: {command}")
    start_time = time.time()
    
    try:
        result = subprocess.run(
            ["python"] + command.split(),
            capture_output=True,
            text=True,
            check=True
        )
        
        duration = time.time() - start_time
        logging.info(f"{stage_name} completed in {duration:.1f}s")
        
        if result.stdout:
            logging.info(f"Output: {result.stdout.strip()}")
            
        return True
        
    except subprocess.CalledProcessError as e:
        duration = time.time() - start_time
        logging.error(f"{stage_name} failed after {duration:.1f}s")
        logging.error(f"Error: {e.stderr}")
        
        if required:
            logging.error("Pipeline stopped due to critical error")
            sys.exit(1)
        else:
            logging.warning("Non-critical error, continuing...")
            return False

def check_prerequisites():
    """Check if required files and directories exist"""
    logging.info("Checking prerequisites...")
    
    required_dirs = ["SMU/Comps/DSA/data"]
    required_files = []
    
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            logging.error(f"Required directory not found: {dir_path}")
            return False
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            logging.error(f"Required file not found: {file_path}")
            return False
    
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    logging.info("Prerequisites checked")
    return True

def main():
    logging.info("Starting Hemoglobin Prediction Master Pipeline")
    logging.info("=" * 60)
    
    if not check_prerequisites():
        sys.exit(1)
    
    stages = [
        # Stage 1: Data Preparation
        ("data_inventory.py", "Data Inventory", True),
        ("create_standard_files.py", "Standard Files Creation", True),
        ("metadata_preprocess.py", "Metadata Preprocessing", True),
        ("explore_data.py", "Data Exploration", False),
        
        # Stage 2: Baseline Models
        ("simple_baseline.py", "Simple Baseline Training", True),
        ("enhanced_features.py", "Enhanced Features Training", True),
        
        # Stage 3: Deep Learning Models
        ("cnn_model.py", "CNN Training", True),
        ("cnn_model_evaluate.py", "CNN Evaluation", True),
        
        # Stage 4: Multimodal Training
        ("multimodal_train.py", "Multimodal Training", True),
        ("save_multimodal_predictions.py", "Multimodal Predictions", True),
        
        # Stage 5: Comprehensive Evaluation
        ("comprehensive_evaluation.py", "Comprehensive Evaluation", True),
        ("fairness_evaluation.py", "Fairness Evaluation", True),
        
        # Stage 6: Model Export
        ("final_model_export.py", "Final Model Export", True),
        ("model_compression.py", "Model Compression", False),
        
        # Stage 7: Submission Preparation
        ("create_submission_package.py", "Submission Package", True),
        ("generate_report_data.py", "Report Data Generation", False),
    ]
    
    successful_stages = 0
    total_stages = len(stages)
    
    for script, stage_name, required in stages:
        if os.path.exists(script):
            success = run_command(script, stage_name, required)
            if success:
                successful_stages += 1
        else:
            logging.warning(f"Script not found: {script}")
    
    logging.info("Running final validation...")
    if os.path.exists("inference.py") and os.path.exists("meta.csv"):
        run_command(
            "inference.py --images SMU/Comps/DSA/data --meta meta.csv --out final_predictions.csv",
            "Final Inference Test",
            False
        )
    
    logging.info("")
    logging.info("Pipeline Summary")
    logging.info("=" * 30)
    logging.info(f"Successful stages: {successful_stages}/{total_stages}")
    
    if successful_stages >= total_stages * 0.8:  
        logging.info("Pipeline completed successfully!")
        
        key_outputs = [
            "best_model.pth",
            "final_predictions.csv", 
            "data_inventory.csv",
            "labels.csv",
            "meta.csv"
        ]
        
        logging.info("Key outputs:")
        for output in key_outputs:
            if os.path.exists(output):
                size = os.path.getsize(output)
                logging.info(f" {output} ({size:,} bytes)")
            else:
                logging.info(f" {output} (missing)")
        
        if os.path.exists("submission"):
            logging.info("submission/ directory created")
        
    else:
        logging.error("Pipeline completed with errors")
        logging.error("Check pipeline.log for detailed error messages")
        sys.exit(1)

if __name__ == "__main__":
    main()