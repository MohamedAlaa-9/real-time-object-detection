from ultralytics import YOLO
import torch
import mlflow
import os
from pathlib import Path

# Define base directory relative to this script
BASE_DIR = Path(__file__).resolve().parent

# Load YOLOv11 model
# Consider making the model path configurable or using different variants (s, m, l, x)
model_path = BASE_DIR / 'yolo11n.pt'
if not model_path.exists():
    print(f"Error: Model file not found at {model_path}")
    # Potentially download a default model if not found
    exit()
model = YOLO(str(model_path))

# Training configuration with augmentation
# Consider using a config file (e.g., YAML) for hyperparameters
# For hyperparameter optimization, consider tools like Ray Tune or Optuna
data_yaml_path = BASE_DIR.parent / "datasets/processed/data.yaml"
training_args = {
    "data": str(data_yaml_path),
    "epochs": 100, # Consider adjusting epochs based on convergence
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
    # Add other relevant hyperparameters like 'patience' for early stopping
    "patience": 20,
}

# Validate dataset path
if not data_yaml_path.exists():
    print(f"Error: data.yaml file not found at {data_yaml_path}")
    print("Please run datasets/preprocess_datasets.py first.")
    exit()

# Train with MLflow logging
print("Starting training...")
with mlflow.start_run() as run:
    print(f"MLflow Run ID: {run.info.run_id}")
    mlflow.log_params(training_args)

    # The train method returns a results object
    results = model.train(**training_args)

    # Log metrics (ultralytics automatically logs to MLflow if installed)
    # You can log additional custom metrics if needed
    # mlflow.log_metric("final_mAP50-95", results.maps['metrics/mAP50-95(B)']) # Example

    # Log the best model artifact (ultralytics saves best.pt automatically)
    best_model_path = Path(results.save_dir) / 'weights/best.pt'
    if best_model_path.exists():
        mlflow.log_artifact(str(best_model_path), artifact_path="best_model")
        print(f"Best model saved at: {best_model_path}")
    else:
        print("Warning: best.pt not found in the expected directory.")

    # Log the entire training run directory for full context
    mlflow.log_artifacts(results.save_dir, artifact_path="training_run")


print(f"Training completed. Results saved in: {results.save_dir}")
print("Check MLflow UI for detailed logs and artifacts.")
