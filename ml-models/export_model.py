import torch
from ultralytics import YOLO

# Load the trained model
model = YOLO('best.pt')  # Load the best model

# Export to ONNX format
model.export(format='onnx', imgsz=[640, 640], simplify=True)

print("Model exported to ONNX format (best.onnx)")
