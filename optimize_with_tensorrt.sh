#!/bin/bash
# This script optimizes both the fine-tuned model and the baseline model with TensorRT

set -e  # Exit on any error

echo "Starting TensorRT optimization..."

# First check if ONNX models exist
if [ ! -f "ml_models/models/best.onnx" ] && [ ! -f "ml_models/models/yolo11n.onnx" ]; then
  echo "No ONNX models found. First preparing models..."
  python ml_models/prepare_models.py --all --export
fi

# Then optimize with TensorRT
echo "Optimizing models with TensorRT..."
python ml_models/prepare_models.py --all --tensorrt

echo "TensorRT optimization complete! To test the models, run:"
echo "  python ml_models/verify_model_pipeline.py"
