# Hemoglobin Prediction Pipeline - One-Click Execution

## Overview
Complete pipeline for training and evaluating hemoglobin prediction models from lip images.

## Prerequisites
```bash
pip install torch torchvision pandas numpy scikit-learn pillow pillow-heif matplotlib seaborn opencv-python
```

## Data Structure Expected
```
project_root/
├── SMU/Comps/DSA/data/          # Raw images with naming: HgB_X.XgdL_IndividualXX_XX.*
├── scripts/                      # All Python files
└── outputs/                      # Generated files
```

## One-Click Execution

### Option 1: Full Training Pipeline
```bash
bash run_full_pipeline.sh
```

### Option 2: Quick Evaluation Only
```bash
bash run_evaluation_only.sh
```

### Option 3: Competition Submission Prep
```bash
bash prepare_submission.sh
```

## Pipeline Stages
1. **Data Preparation** - Parse filenames, create standard CSVs
2. **Baseline Training** - Simple feature-based models
3. **CNN Training** - Deep learning models
4. **Multimodal Training** - Combined image + metadata
5. **Evaluation** - Compare all models
6. **Export** - Create final submission package

## Key Outputs
- `best_model.pth` - Best performing model weights
- `final_predictions.csv` - Model predictions
- `performance_report.csv` - Model comparison metrics
- `submission/` - Competition-ready package

## Reproduction Guarantee
All scripts use deterministic seeds (random_state=42, torch.manual_seed(42)) for exact reproducibility.