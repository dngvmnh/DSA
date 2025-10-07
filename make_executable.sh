#!/bin/bash
# make_executable.sh - Make all scripts executable

chmod +x run_full_pipeline.sh
chmod +x run_evaluation_only.sh  
chmod +x prepare_submission.sh

echo "All shell scripts are now executable"
echo ""
echo "Ready to run! Choose your option:"
echo ""
echo "1. Full training pipeline:"
echo "   ./run_full_pipeline.sh"
echo ""
echo "2. Quick evaluation only:"  
echo "   ./run_evaluation_only.sh"
echo ""
echo "3. Competition submission prep:"
echo "   ./prepare_submission.sh"
echo ""
echo "4. Python master pipeline:"
echo "   python master_pipeline.py"