from ultralytics import YOLO
import torch
import mlflow
import yaml # Added for YAML loading
import os
from pathlib import Path

# Define project root directory (assuming script is in ml-models/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- Configuration Loading ---
CONFIG_PATH = PROJECT_ROOT / "config/train_config.yaml"

if not CONFIG_PATH.exists():
    print(f"Error: Configuration file not found at {CONFIG_PATH}")
    exit()

with open(CONFIG_PATH, 'r') as f:
    try:
        config = yaml.safe_load(f)
        print("Loaded training configuration:")
        print(yaml.dump(config, indent=2))
    except yaml.YAMLError as e:
        print(f"Error parsing configuration file: {e}")
        exit()

# Construct absolute paths from config (relative paths are assumed relative to project root)
base_model_path = PROJECT_ROOT / config['base_model']
data_yaml_path = PROJECT_ROOT / config['data_yaml']

# Validate paths
if not base_model_path.exists():
    print(f"Error: Base model file not found at {base_model_path}")
    # Potentially add logic to download a default model if needed
    exit()

if not data_yaml_path.exists():
    print(f"Error: data.yaml file not found at {data_yaml_path}")
    print("Ensure the dataset exists. You might need to run datasets/preprocess_datasets.py first.")
    exit()

# --- Model Initialization ---
model = YOLO(str(base_model_path))

# --- Training Arguments from Config ---
# Select relevant keys from config for model.train()
train_keys = [
    "epochs", "imgsz", "batch", "device", "project", "name",
    "augment", "mosaic", "mixup", "hsv_h", "hsv_s", "hsv_v", "patience"
]
training_args = {k: config[k] for k in train_keys if k in config}

# Add the essential 'data' argument
training_args["data"] = str(data_yaml_path)

# Handle device selection more robustly
if str(training_args.get("device", "cpu")).lower() != "cpu":
    if not torch.cuda.is_available():
        print(f"Warning: CUDA device '{training_args['device']}' requested but CUDA not available. Using CPU.")
        training_args["device"] = "cpu"
    else:
        # Ensure the specified GPU index is valid if needed (YOLO handles basic '0', '0,1' etc.)
        pass # YOLO internal handling is usually sufficient
else:
     training_args["device"] = "cpu"


# --- Training with MLflow ---
print("Starting training with loaded configuration...")
# Ensure MLflow tracking URI is set if not using local default
# os.environ['MLFLOW_TRACKING_URI'] = 'http://your_mlflow_server:5000' # Example

with mlflow.start_run() as run:
    print(f"MLflow Run ID: {run.info.run_id}")
    # Log the loaded configuration parameters
    mlflow.log_params(config) # Log the entire config dict

    # The train method returns a results object
    print(f"Training arguments passed to YOLO: {training_args}")
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
