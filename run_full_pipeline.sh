#!/bin/bash
# run_full_pipeline.sh - Complete training and evaluation pipeline

set -e  # Exit on any error

echo "Starting Hemoglobin Prediction Pipeline..."
echo "================================================"

# Set up environment
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
mkdir -p outputs plots models

# Stage 1: Data Preparation
echo "Stage 1: Data Preparation"
echo "Creating data inventory..."
python data_inventory.py

echo "Creating standard CSV files..."
python create_standard_files.py

echo "Preprocessing metadata..."
python metadata_preprocess.py

echo "Exploring data (generating plots)..."
python explore_data.py

# Stage 2: Baseline Models
echo "Stage 2: Baseline Training"
echo "Training simple baseline model..."
python simple_baseline.py > outputs/baseline_log.txt

echo "Training enhanced feature model..."
python enhanced_features.py > outputs/enhanced_features_log.txt

# Stage 3: Deep Learning Models
echo "Stage 3: CNN Training"
echo "Training CNN model..."
python cnn_model.py > outputs/cnn_training_log.txt

echo "Evaluating CNN model..."
python cnn_model_evaluate.py

# Stage 4: Multimodal Training
echo "Stage 4: Multimodal Training"
echo "Training multimodal model..."
python multimodal_train.py > outputs/multimodal_training_log.txt

echo "Saving multimodal predictions..."
python save_multimodal_predictions.py

# Stage 5: Comprehensive Evaluation
echo "Stage 5: Model Evaluation"
echo "Running comprehensive evaluation..."
python comprehensive_evaluation.py > outputs/evaluation_report.txt

echo "Evaluating fairness across groups..."
python fairness_evaluation.py > outputs/fairness_report.txt

# Stage 6: Final Model Export
echo "Stage 6: Model Export & Compression"
echo "Exporting and compressing final model..."
python final_model_export.py > outputs/export_log.txt

echo "Compressing model for deployment..."
python model_compression.py > outputs/compression_log.txt

# Stage 7: Competition Submission
echo "Stage 7: Competition Preparation"
echo "Creating submission package..."
python create_submission_package.py

echo "Generating report data..."
python generate_report_data.py > outputs/report_data.txt

# Final validation
echo "Final Validation"
echo "Testing inference pipeline..."
python inference.py --images SMU/Comps/DSA/data --meta meta.csv --out final_predictions.csv

# Summary
echo ""
echo "Pipeline Complete!"
echo "===================="
echo "Key outputs:"
echo "- best_model.pth (best performing model)"
echo "- final_predictions.csv (predictions on all data)"
echo "- submission/ (competition-ready package)"
echo "- outputs/ (all logs and reports)"
echo "- plots/ (visualization outputs)"
echo ""
echo "Next steps:"
echo "1. Review outputs/evaluation_report.txt for model performance"
echo "2. Check outputs/fairness_report.txt for bias analysis"
echo "3. Submit the submission/ folder to competition"
echo ""

# Check if target MAE achieved
if [ -f "outputs/evaluation_report.txt" ]; then
    echo "Performance Summary:"
    grep -E "(MAE|Target)" outputs/evaluation_report.txt || echo "Check evaluation report for performance metrics"
fi

echo "ll done! Pipeline completed successfully."