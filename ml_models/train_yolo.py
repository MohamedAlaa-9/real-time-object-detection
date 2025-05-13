from ultralytics import YOLO
import torch
import yaml # Added for YAML loading
import os
import requests
from pathlib import Path
import logging
import sys
import shutil

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to import MLflow but make it optional
try:
    import mlflow
    MLFLOW_AVAILABLE = True
    logger.info("MLflow is available for experiment tracking")
except ImportError:
    MLFLOW_AVAILABLE = False
    logger.warning("MLflow not installed. Experiment tracking disabled.")

# Define project root directory (assuming script is in ml_models/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- Configuration Loading ---
CONFIG_PATH = PROJECT_ROOT / "config/train_config.yaml"

if not CONFIG_PATH.exists():
    logger.error(f"Error: Configuration file not found at {CONFIG_PATH}")
    sys.exit(1)

with open(CONFIG_PATH, 'r') as f:
    try:
        config = yaml.safe_load(f)
        logger.info("Loaded training configuration:")
        logger.info(yaml.dump(config, indent=2))
    except yaml.YAMLError as e:
        logger.error(f"Error parsing configuration file: {e}")
        sys.exit(1)

# Construct absolute paths from config (relative paths are assumed relative to project root)
base_model_path = PROJECT_ROOT / config['base_model']
data_yaml_path = PROJECT_ROOT / config['data_yaml']

# Function to download pre-trained model
def download_pretrained_model(model_path):
    """Download pre-trained YOLO model if it doesn't exist."""
    model_name = model_path.name
    logger.info(f"Model {model_name} not found. Attempting to download...")
    
    # Create directory if it doesn't exist
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Use YOLO's built-in download functionality
        YOLO(model_name)
        
        # Copy the downloaded model to the desired location
        # The model is typically downloaded to ~/.cache/ultralytics
        cache_dir = Path.home() / ".cache" / "ultralytics" / "models"
        downloaded_model = cache_dir / model_name
        
        if downloaded_model.exists():
            shutil.copy(downloaded_model, model_path)
            logger.info(f"Model {model_name} downloaded and copied to {model_path}")
            return True
        else:
            logger.error(f"Downloaded model not found at {downloaded_model}")
            return False
    except Exception as e:
        logger.error(f"Error downloading model: {e}")
        return False

# Validate paths
if not base_model_path.exists():
    logger.warning(f"Base model file not found at {base_model_path}")
    # Download the model if it doesn't exist
    if not download_pretrained_model(base_model_path):
        logger.error("Failed to download the base model. Exiting.")
        sys.exit(1)

if not data_yaml_path.exists():
    logger.error(f"Error: data.yaml file not found at {data_yaml_path}")
    logger.error("Ensure the dataset exists. You might need to run datasets/preprocess_datasets.py first.")
    sys.exit(1)

# --- Model Initialization ---
model = YOLO(str(base_model_path))

# --- Training Arguments from Config ---
# Select relevant keys for model.train()
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
        logger.warning(f"CUDA device '{training_args['device']}' requested but CUDA not available. Using CPU.")
        training_args["device"] = "cpu"
    else:
        # Ensure the specified GPU index is valid if needed (YOLO handles basic '0', '0,1' etc.)
        pass # YOLO internal handling is usually sufficient
else:
    training_args["device"] = "cpu"

# Cloud-specific MLflow configuration
# Check if we're in a cloud environment and if MLflow tracking URI is specified
MLFLOW_TRACKING_URI = os.environ.get('MLFLOW_TRACKING_URI')
if MLFLOW_AVAILABLE and MLFLOW_TRACKING_URI:
    try:
        logger.info(f"Setting MLflow tracking URI to: {MLFLOW_TRACKING_URI}")
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    except Exception as e:
        logger.warning(f"Failed to set MLflow tracking URI: {e}")
        MLFLOW_AVAILABLE = False

# Disable MLflow auto-logging in YOLO if MLflow isn't properly configured
if not MLFLOW_AVAILABLE:
    # Set the environment variable to disable MLflow integration in ultralytics
    os.environ["ULTRALYTICS_MLFLOW_DISABLED"] = "1"
    logger.info("MLflow integration disabled for this run")

# --- Training with or without MLflow ---
logger.info("Starting training with loaded configuration...")
logger.info(f"Training arguments passed to YOLO: {training_args}")

# Function to handle the training process
def run_training():
    # Train the model
    results = model.train(**training_args)
    
    # Save model path for reference
    best_model_path = Path(results.save_dir) / 'weights/best.pt'
    if best_model_path.exists():
        logger.info(f"Best model saved at: {best_model_path}")
        
        # Update export configuration to use this trained model
        update_export_config(best_model_path)
        
        # Create a symlink to make model available for inference
        create_inference_symlink(best_model_path)
    else:
        logger.warning("Warning: best.pt not found in the expected directory.")
    
    logger.info(f"Training completed. Results saved in: {results.save_dir}")
    return results, best_model_path

def update_export_config(best_model_path):
    """Update export configuration to use the newly trained model"""
    try:
        with open(CONFIG_PATH, 'r') as f:
            current_config = yaml.safe_load(f)
        
        # Update the export section with the new model path
        rel_path = str(best_model_path.relative_to(PROJECT_ROOT))
        if 'export' in current_config:
            current_config['export']['trained_model_path'] = rel_path
            
            with open(CONFIG_PATH, 'w') as f:
                yaml.dump(current_config, f, default_flow_style=False)
            
            logger.info(f"Updated export configuration with new model path: {rel_path}")
        else:
            logger.warning("No 'export' section found in config. Export config not updated.")
    except Exception as e:
        logger.error(f"Failed to update export configuration: {e}")

def create_inference_symlink(source_path):
    """Create a symlink to the best model for the inference system"""
    try:
        # Define the symlink target path (match what the backend expects)
        inference_model_path = PROJECT_ROOT / "ml_models" / "models" / "best.pt"
        
        # Remove existing symlink if it exists
        if inference_model_path.exists() or inference_model_path.is_symlink():
            inference_model_path.unlink()
        
        # Create the symlink
        inference_model_path.symlink_to(source_path.resolve())
        logger.info(f"Created symbolic link from {source_path} to {inference_model_path} for inference")
    except Exception as e:
        logger.error(f"Failed to create symbolic link for inference: {e}")
        # As fallback, try to copy the file
        try:
            shutil.copy(source_path, inference_model_path)
            logger.info(f"Copied {source_path} to {inference_model_path} for inference (symlink failed)")
        except Exception as copy_error:
            logger.error(f"Failed to copy model for inference: {copy_error}")

# Run with or without MLflow based on availability and connectivity
if MLFLOW_AVAILABLE:
    try:
        # Check if we can start an MLflow run (tests connectivity)
        with mlflow.start_run() as run:
            run_id = run.info.run_id
            logger.info(f"MLflow Run ID: {run_id}")
            
            # Log the loaded configuration parameters
            mlflow.log_params(config)
            
            # Run the training
            results, best_model_path = run_training()
            
            # Log the best model artifact if it exists
            if best_model_path.exists():
                try:
                    mlflow.log_artifact(str(best_model_path), artifact_path="best_model")
                    # Try to log the entire training run directory for full context
                    mlflow.log_artifacts(results.save_dir, artifact_path="training_run")
                    logger.info("Artifacts successfully logged to MLflow")
                except Exception as e:
                    logger.warning(f"Failed to log artifacts to MLflow: {e}")
            
        logger.info("MLflow tracking completed successfully")
    except Exception as e:
        logger.warning(f"MLflow tracking failed: {e}")
        logger.warning("Continuing with training without MLflow tracking")
        os.environ["ULTRALYTICS_MLFLOW_DISABLED"] = "1"
        run_training()
else:
    # Run training without MLflow
    run_training()

logger.info("Training process completed.")
