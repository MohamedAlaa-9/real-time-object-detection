import cv2
import numpy as np
import logging
from pathlib import Path
import atexit
from prometheus_client import Counter, Histogram, start_http_server

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Metrics
INFERENCE_COUNT = Counter("inference_total", "Total number of inferences performed")
INFERENCE_TIME = Histogram("inference_duration_seconds", "Histogram of inference duration")
PREPROCESS_TIME = Histogram("preprocess_duration_seconds", "Histogram of preprocessing duration")
POSTPROCESS_TIME = Histogram("postprocess_duration_seconds", "Histogram of postprocessing duration")

# --- Initialization ---
logger.info("Initializing Inference Engine...")

# Start metrics server
try:
    start_http_server(8000) # Ensure this is 8000
    logger.info("Prometheus metrics server started on port 8000.")
except OSError as e:
    logger.warning(f"Could not start Prometheus server on port 8000: {e}. Metrics may not be available.")

# Define paths relative to the script
BASE_DIR = Path(__file__).resolve().parent
BASE_MODEL_PATH = BASE_DIR / 'yolov8n.pt'

# Flag to determine which model to use
use_pytorch = False
model = None # Initialize model variable
input_h, input_w = 640, 640 # Default YOLO input size

# --- Load PyTorch Model Directly ---
logger.info(f"Attempting to load official PyTorch model: {BASE_MODEL_PATH}")
try:
    from ultralytics import YOLO
    
    # Check if base model exists
    if not BASE_MODEL_PATH.exists():
        logger.error(f"Base model not found at {BASE_MODEL_PATH}")
        exit(1)
        
    # Load the PyTorch YOLO model
    model = YOLO(BASE_MODEL_PATH)
    logger.info(f"Successfully loaded PyTorch YOLO model from {BASE_MODEL_PATH}")
    use_pytorch = True
    
except ImportError:
    logger.error("Failed to import ultralytics. Please install it with 'pip install ultralytics'")
    exit(1)
except Exception as e:
    logger.error(f"Failed to load PyTorch model: {e}")
    exit(1)

# --- Check if any engine initialized ---
if not use_pytorch:
    logger.error("Could not initialize the PyTorch inference engine.")
    exit(1)

def infer(frame: np.ndarray):
    """
    Performs inference on a single frame using PyTorch.

    Args:
        frame: The input image frame (NumPy array BGR).

    Returns:
        Tuple[list, list, list]: Bounding boxes, scores, and class indices.
    """
    boxes, scores, classes = [], [], []

    if frame is None:
        logger.warning("Received None frame, skipping inference.")
        return boxes, scores, classes

    if use_pytorch:
        # --- PyTorch Inference Path ---
        try:
            with INFERENCE_TIME.time():
                # YOLO model processes the image internally
                results = model(frame, conf=0.45, iou=0.5)[0]
                
            with POSTPROCESS_TIME.time():
                # Extract results from YOLO prediction
                detections = results.boxes
                
                if len(detections) > 0:
                    # Convert xyxy boxes to the format expected by the app
                    boxes = detections.xyxy.cpu().numpy().tolist()
                    scores = detections.conf.cpu().numpy().tolist()
                    classes = detections.cls.cpu().numpy().tolist()
                    classes = [int(cls) for cls in classes]  # Ensure integers
                    logger.info(f"PyTorch inference found {len(boxes)} objects.")
                else:
                    boxes, scores, classes = [], [], []
                    logger.info("PyTorch inference: no objects detected.")
                    
            INFERENCE_COUNT.inc()
                
        except Exception as e:
            logger.error(f"Error during PyTorch inference: {e}")
            return [], [], []
    else:
        # This case should not be reached due to the check after initialization
        logger.error("PyTorch engine not initialized, cannot perform inference.")
        return [], [], []

    return boxes, scores, classes

# --- Cleanup function ---
def cleanup():
    logger.info("Releasing resources... (No specific cleanup needed for PyTorch model)")
    pass

atexit.register(cleanup)

logger.info("Inference engine initialization complete. Ready for inference.")
