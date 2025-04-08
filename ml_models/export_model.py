import torch
from ultralytics import YOLO
from pathlib import Path
import logging
import sys

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define base directory relative to this script
BASE_DIR = Path(__file__).resolve().parent

# --- Configuration ---
# Path to the trained PyTorch model (saved by train_yolo.py)
# Assumes the best model from the latest training run is used.
# Consider making this path configurable if multiple models exist.
trained_model_path = BASE_DIR.parent / 'runs/train/yolov11_kitti_exp/weights/best.pt' # More specific path
# Fallback if the specific run path doesn't exist (less ideal)
if not trained_model_path.exists():
    trained_model_path = BASE_DIR / 'best.pt' 

# Output ONNX file path
onnx_output_path = BASE_DIR / 'best.onnx'

# Export settings
input_size = [640, 640] # Height, Width
opset_version = 12 # Recommended opset for compatibility
enable_dynamic_axes = True # Allow dynamic batch size/input size in ONNX/TRT
simplify_onnx = True # Use onnx-simplifier

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
