from ultralytics import YOLO
import torch
import mlflow
import os

# Load YOLOv11 model
model = YOLO('ml-models/yolo11n.pt')  # Start with nano, upgrade to larger variants if needed

# Training configuration with augmentation
data_yaml = "../datasets/processed/data.yaml"
training_args = {
    "data": data_yaml,
    "epochs": 100,
    "imgsz": 640,
    "batch": 16,
    "device": 0 if torch.cuda.is_available() else "cpu",
    "project": "runs/train",
    "name": "yolov11_kitti_exp",
    "augment": True,  # Enable built-in augmentation
    "mosaic": 1.0,    # Mosaic augmentation
    "mixup": 0.5,     # Mixup augmentation
    "hsv_h": 0.015,   # Hue augmentation for diverse lighting
    "hsv_s": 0.7,     # Saturation
    "hsv_v": 0.4,     # Value
}

# Validate dataset paths
if not os.path.exists(data_yaml):
    print(f"Error: data.yaml file not found at {data_yaml}")
    exit()

# Train with MLflow logging
with mlflow.start_run():
    mlflow.log_params(training_args)
    model.train(**training_args)
    mlflow.pytorch.log_model(model, "model")

# Save the best model
model.save('best.pt')
print("Training completed. Model saved as 'best.pt'.")
