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
    model.export(
        format='onnx',
        imgsz=input_size,
        opset=opset_version,
        dynamic=enable_dynamic_axes,
        simplify=simplify_onnx,
        # The export function saves the file relative to the model path by default,
        # or you might need to specify 'file=onnx_output_path' depending on ultralytics version.
        # Let's assume it saves 'best.onnx' in the current dir or model dir.
        # We will explicitly check for the expected output file later.
    )
    
    # Verify output file exists (ultralytics might save it as model_name.onnx)
    expected_onnx_file = Path(str(trained_model_path).replace('.pt', '.onnx')) # Default export name
    if expected_onnx_file.exists() and expected_onnx_file != onnx_output_path:
        logger.info(f"Moving exported file from {expected_onnx_file} to {onnx_output_path}")
        expected_onnx_file.rename(onnx_output_path)
    elif not onnx_output_path.exists():
         # Check common locations if not found immediately
         alt_path1 = BASE_DIR / f"{trained_model_path.stem}.onnx"
         if alt_path1.exists():
             logger.info(f"Moving exported file from {alt_path1} to {onnx_output_path}")
             alt_path1.rename(onnx_output_path)
         else:
             # If still not found, raise error
             raise FileNotFoundError(f"Exported ONNX file not found at expected paths: {onnx_output_path} or similar.")

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
