#!/bin/bash
# run_evaluation_only.sh - Quick evaluation of pre-trained models

set -e

echo "⚡ Quick Evaluation Pipeline"
echo "============================"

# Prerequisites check
if [ ! -f "data_inventory.csv" ]; then
    echo "Creating data inventory..."
    python data_inventory.py
fi

if [ ! -f "labels.csv" ]; then
    echo "Creating standard files..."
    python create_standard_files.py
fi

if [ ! -f "meta.csv" ]; then
    echo "Preprocessing metadata..."
    python metadata_preprocess.py
fi

# Quick model evaluation
echo "Running evaluations..."
mkdir -p outputs

# Evaluate existing models
if [ -f "best_model.pth" ]; then
    echo "Evaluating CNN model..."
    python cnn_model_evaluate.py
fi

if [ -f "multimodal_model.pth" ]; then
    echo "Evaluating multimodal model..."
    python save_multimodal_predictions.py
fi

# Comprehensive comparison
echo "Comprehensive evaluation..."
python comprehensive_evaluation.py > outputs/quick_evaluation.txt

# Fairness check
echo "Fairness evaluation..."
python fairness_evaluation.py > outputs/quick_fairness.txt

# Summary
echo ""
echo "Quick Evaluation Complete!"
echo "============================="
echo "Results saved to:"
echo "- outputs/quick_evaluation.txt"
echo "- outputs/quick_fairness.txt"

if [ -f "outputs/quick_evaluation.txt" ]; then
    echo ""
    echo "Performance Summary:"
    head -20 outputs/quick_evaluation.txt
fi