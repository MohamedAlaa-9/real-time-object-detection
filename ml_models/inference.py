import cv2
import numpy as np
import logging
from pathlib import Path
import atexit
import shutil
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
PROJECT_ROOT = BASE_DIR.parent

# Model paths - check for fine-tuned model first, fallback to base model
FINE_TUNED_MODEL_PATH = BASE_DIR / 'best.pt'  # Our symlink to the latest fine-tuned model
YOLO11_MODEL_PATH = BASE_DIR / 'yolo11n.pt'   # Pre-trained base YOLO11 model
YOLO8_MODEL_PATH = BASE_DIR / 'yolov8n.pt'    # Alternative pre-trained model

# Flag to determine which model to use
use_pytorch = False
model = None # Initialize model variable
input_h, input_w = 640, 640 # Default YOLO input size
model_source = ""  # Track which model we're using

# Function to download and save a model using ultralytics
def download_model(model_name, save_path):
    """
    Download a model from ultralytics and save it to the specified path.
    
    Args:
        model_name: Name of the model to download (e.g., "yolo11n.pt", "yolov8n.pt")
        save_path: Path where to save the downloaded model
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        from ultralytics import YOLO
        
        logger.info(f"Downloading {model_name}...")
        # This will download the model to cache
        temp_model = YOLO(model_name)
        
        # Check if download succeeded (should be in cache)
        cache_dir = Path.home() / ".cache" / "ultralytics" / "models"
        downloaded_model = cache_dir / model_name
        
        if downloaded_model.exists():
            # Create directory if it doesn't exist
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Copy to our desired location
            shutil.copy(downloaded_model, save_path)
            logger.info(f"Model {model_name} downloaded and saved to {save_path}")
            return True
        else:
            logger.error(f"Downloaded model not found at expected location: {downloaded_model}")
            return False
            
    except Exception as e:
        logger.error(f"Error downloading model {model_name}: {e}")
        return False

# --- Load PyTorch Model Directly ---
try:
    from ultralytics import YOLO
    
    # First try to load our fine-tuned model
    if FINE_TUNED_MODEL_PATH.exists():
        try:
            logger.info(f"Loading fine-tuned model from: {FINE_TUNED_MODEL_PATH}")
            model = YOLO(str(FINE_TUNED_MODEL_PATH))
            # Verify model architecture (check if it's YOLOv11)
            model_type = getattr(model, 'type', 'unknown')
            if 'yolo11' in model_type.lower() or 'yolov11' in model_type.lower():
                model_source = "fine-tuned-yolov11"
            else:
                model_source = "fine-tuned"
            use_pytorch = True
            logger.info(f"Successfully loaded fine-tuned YOLO model (type: {model_type})")
        except Exception as e:
            logger.error(f"Failed to load fine-tuned model: {e}")
            model = None
    
    # If fine-tuned model failed, try the base YOLO11 model
    if model is None:
        try:
            # Check if YOLO11 base model exists, download if not
            if not YOLO11_MODEL_PATH.exists():
                logger.warning(f"YOLO11 base model not found at {YOLO11_MODEL_PATH}, attempting to download it")
                if not download_model("yolo11n.pt", YOLO11_MODEL_PATH):
                    logger.warning("Failed to download YOLO11 model, will try with YOLOv8")
            
            # If the model exists now (either it was there or download succeeded)
            if YOLO11_MODEL_PATH.exists():
                logger.info(f"Loading YOLO11 base model from: {YOLO11_MODEL_PATH}")
                model = YOLO(str(YOLO11_MODEL_PATH))
                model_source = "yolo11-base"
                use_pytorch = True
                logger.info("Successfully loaded YOLO11 base model")
        except Exception as e:
            logger.error(f"Failed to load YOLO11 base model: {e}")
            model = None
    
    # Final fallback to YOLO8 model
    if model is None:
        logger.info(f"Attempting to load YOLO8 model: {YOLO8_MODEL_PATH}")
        try:
            # Try to download if it doesn't exist
            if not YOLO8_MODEL_PATH.exists():
                logger.warning(f"YOLO8 model not found, attempting to download it")
                download_model("yolov8n.pt", YOLO8_MODEL_PATH)
            
            # Now load the model (either existing or newly downloaded)
            model = YOLO(str(YOLO8_MODEL_PATH))
            model_source = "yolo8-fallback"
            use_pytorch = True
            logger.info("Successfully loaded YOLO8 fallback model")
        except Exception as e:
            logger.error(f"Failed to load YOLO8 model: {e}")
            exit(1)
    
except ImportError:
    logger.error("Failed to import ultralytics. Please install it with 'pip install ultralytics'")
    exit(1)

# --- Check if any engine initialized ---
if not use_pytorch:
    logger.error("Could not initialize the PyTorch inference engine.")
    exit(1)

logger.info(f"Using {model_source} model for inference")

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
