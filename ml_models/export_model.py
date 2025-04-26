import torch
from ultralytics import YOLO
from pathlib import Path
import logging
import sys
import yaml # Added for YAML loading

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define project root directory (assuming script is in ml-models/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- Configuration Loading ---
CONFIG_PATH = PROJECT_ROOT / "config/train_config.yaml"

if not CONFIG_PATH.exists():
    logger.error(f"Configuration file not found at {CONFIG_PATH}")
    sys.exit(1)

logger.info(f"Loading configuration from: {CONFIG_PATH}")
with open(CONFIG_PATH, 'r') as f:
    try:
        config = yaml.safe_load(f)
        if 'export' not in config:
            logger.error("Missing 'export' section in configuration file.")
            sys.exit(1)
        export_config = config['export']
        logger.info("Loaded export configuration:")
        logger.info(yaml.dump(export_config, indent=2))
    except yaml.YAMLError as e:
        logger.error(f"Error parsing configuration file: {e}")
        sys.exit(1)

# --- Extract Configuration Values ---
# Construct absolute paths from config (relative paths are assumed relative to project root)
trained_model_path = PROJECT_ROOT / export_config['trained_model_path']
onnx_output_path = PROJECT_ROOT / export_config['onnx_output_path']

# Ensure the output directory exists
onnx_output_path.parent.mkdir(parents=True, exist_ok=True)

# Export settings from config
input_size = [export_config['input_height'], export_config['input_width']]
opset_version = export_config['opset_version']
enable_dynamic_axes = export_config['enable_dynamic_axes']
simplify_onnx = export_config['simplify_onnx']

# --- Validation ---
if not trained_model_path.exists():
    logger.error(f"Trained model file not found: {trained_model_path}")
    logger.error("Please ensure training was completed successfully and the model exists.")
    sys.exit(1)

# --- Load Model ---
logger.info(f"Loading trained model from: {trained_model_path}")
try:
    model = YOLO(str(trained_model_path))
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    sys.exit(1)

# --- Export to ONNX ---
logger.info(f"Exporting model to ONNX format at: {onnx_output_path}")
logger.info(f"  Input size: {input_size}")
logger.info(f"  Opset version: {opset_version}")
logger.info(f"  Dynamic axes: {enable_dynamic_axes}")
logger.info(f"  Simplify ONNX: {simplify_onnx}")

try:
    export_result = model.export(
        format='onnx',
        imgsz=input_size,
        opset=opset_version,
        dynamic=enable_dynamic_axes,
        simplify=simplify_onnx,
        # Note: ultralytics export saves the file relative to the working directory
        # or the model's path by default, often named like the .pt file but with .onnx
    )

    # The export function might return the path or save to a predictable name
    # Let's find the default exported file name (usually model_name.onnx)
    default_onnx_filename = trained_model_path.with_suffix('.onnx').name
    default_onnx_filepath = Path(default_onnx_filename) # Check in current dir first

    # Check if the export function returned the path directly
    exported_file_path = None
    if isinstance(export_result, str):
        exported_file_path = Path(export_result)
        logger.info(f"Export function returned path: {exported_file_path}")
    elif default_onnx_filepath.exists():
         exported_file_path = default_onnx_filepath
         logger.info(f"Found exported file at default path: {exported_file_path}")
    else:
        # Fallback: Check near the original model if not in cwd
        alt_path = trained_model_path.parent / default_onnx_filename
        if alt_path.exists():
            exported_file_path = alt_path
            logger.info(f"Found exported file near original model: {exported_file_path}")

    if exported_file_path and exported_file_path.exists():
        if exported_file_path.resolve() != onnx_output_path.resolve():
            logger.info(f"Moving exported file from {exported_file_path} to {onnx_output_path}")
            exported_file_path.rename(onnx_output_path)
        else:
            logger.info(f"Exported file is already at the target location: {onnx_output_path}")
    elif not onnx_output_path.exists():
         # If we couldn't find the exported file and the target doesn't exist
         raise FileNotFoundError(f"Exported ONNX file could not be found at expected locations (e.g., {default_onnx_filepath}, near {trained_model_path}) and was not moved to {onnx_output_path}.")
    else:
        # Target path already exists, assume export worked correctly to the target
        logger.info(f"Target ONNX file already exists at: {onnx_output_path}. Assuming export was successful.")


    logger.info(f"Model successfully exported to: {onnx_output_path}")
    sys.exit(0) # Success

except Exception as e:
    logger.error(f"Failed to export model to ONNX: {e}")
    # Clean up partial file if it exists
    if onnx_output_path.exists():
        try:
            onnx_output_path.unlink()
        except OSError:
            pass
    sys.exit(1) # Failure
