import cv2
import numpy as np
import logging
from pathlib import Path
import atexit
import shutil
import torch
import os
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
    # Use a different port than the backend server (which uses 8000)
    start_http_server(8001)
    logger.info("Prometheus metrics server started on port 8001.")
except OSError as e:
    logger.warning(f"Could not start Prometheus server on port 8001: {e}. Metrics may not be available.")

# Define paths relative to the script
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

# Model paths - check for all model types
FINE_TUNED_MODEL_PATH = BASE_DIR / 'best.pt'  # Our symlink to the latest fine-tuned model
YOLO11_MODEL_PATH = BASE_DIR / 'yolo11n.pt'   # Pre-trained base YOLO11 model
YOLO11_ONNX_PATH = BASE_DIR / 'yolo11n.onnx'  # ONNX version of the base model

# Flags to determine which model to use
use_pytorch = False
use_onnx = False
model = None  # Initialize PyTorch model variable
ort_session = None  # Initialize ONNX session
input_h, input_w = 640, 640  # Standard YOLO input size
model_source = ""  # Track which model we're using

# Function to download and save a model using ultralytics
def download_model(model_name, save_path):
    """
    Download a model from ultralytics and save it to the specified path.
    
    Args:
        model_name: Name of the model to download (e.g., "yolo11n.pt")
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

# --- Export PyTorch model to ONNX ---
def export_model_to_onnx(pt_model_path, onnx_model_path):
    """Export PyTorch model to ONNX format"""
    try:
        from ultralytics import YOLO
        
        logger.info(f"Exporting {pt_model_path} to ONNX format...")
        model = YOLO(str(pt_model_path))
        success = model.export(format="onnx", imgsz=[input_h, input_w], simplify=True)
        
        # The export function creates the file next to the original with .onnx extension
        expected_path = pt_model_path.with_suffix('.onnx')
        
        if expected_path.exists() and onnx_model_path != expected_path:
            # Move to the desired ONNX path if different
            onnx_model_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(expected_path, onnx_model_path)
            
        if onnx_model_path.exists():
            logger.info(f"Successfully exported to {onnx_model_path}")
            return True
        else:
            logger.error(f"ONNX export failed, file not found at {onnx_model_path}")
            return False
    
    except Exception as e:
        logger.error(f"Error exporting model to ONNX: {e}")
        return False

# --- Try loading ONNX model first ---
try:
    # Only import onnxruntime if we're going to use it
    if YOLO11_ONNX_PATH.exists() or FINE_TUNED_MODEL_PATH.exists():
        try:
            import onnxruntime as ort
            HAVE_ONNX = True
            logger.info("ONNX Runtime imported successfully")
        except ImportError:
            logger.warning("ONNX Runtime not available. Install it with 'pip install onnxruntime' for faster inference.")
            HAVE_ONNX = False
    else:
        HAVE_ONNX = False
    
    # First, check if we have the ONNX version of the fine-tuned model
    fine_tuned_onnx = FINE_TUNED_MODEL_PATH.with_suffix('.onnx')
    if HAVE_ONNX and fine_tuned_onnx.exists():
        logger.info(f"Loading fine-tuned ONNX model from: {fine_tuned_onnx}")
        providers = ['CPUExecutionProvider']
        if 'CUDAExecutionProvider' in ort.get_available_providers():
            providers.insert(0, 'CUDAExecutionProvider')
            logger.info("Using CUDA for ONNX inference")
            
        ort_session = ort.InferenceSession(str(fine_tuned_onnx), providers=providers)
        input_name = ort_session.get_inputs()[0].name
        
        # Get input shape (might be dynamic)
        input_shape = ort_session.get_inputs()[0].shape
        input_h = input_shape[2] if isinstance(input_shape[2], int) else 640
        input_w = input_shape[3] if isinstance(input_shape[3], int) else 640
        
        use_onnx = True
        model_source = "fine-tuned-onnx"
        logger.info(f"Successfully loaded fine-tuned ONNX model. Input shape: {input_h}x{input_w}")
    
    # If fine-tuned ONNX not available, try the base YOLO11 ONNX model
    elif HAVE_ONNX and YOLO11_ONNX_PATH.exists():
        logger.info(f"Loading base YOLO11 ONNX model from: {YOLO11_ONNX_PATH}")
        providers = ['CPUExecutionProvider']
        if 'CUDAExecutionProvider' in ort.get_available_providers():
            providers.insert(0, 'CUDAExecutionProvider')
            logger.info("Using CUDA for ONNX inference")
            
        ort_session = ort.InferenceSession(str(YOLO11_ONNX_PATH), providers=providers)
        input_name = ort_session.get_inputs()[0].name
        
        # Get input shape (might be dynamic)
        input_shape = ort_session.get_inputs()[0].shape
        input_h = input_shape[2] if isinstance(input_shape[2], int) else 640
        input_w = input_shape[3] if isinstance(input_shape[3], int) else 640
        
        use_onnx = True
        model_source = "yolo11n-onnx"
        logger.info(f"Successfully loaded base YOLO11 ONNX model. Input shape: {input_h}x{input_w}")
        
    # If we have PyTorch model but not ONNX, try to create ONNX model
    elif HAVE_ONNX and not YOLO11_ONNX_PATH.exists() and YOLO11_MODEL_PATH.exists():
        logger.info("Trying to create ONNX model from the base PyTorch model...")
        if export_model_to_onnx(YOLO11_MODEL_PATH, YOLO11_ONNX_PATH):
            # If export successful, try loading again
            logger.info(f"Loading newly created ONNX model from: {YOLO11_ONNX_PATH}")
            providers = ['CPUExecutionProvider']
            if 'CUDAExecutionProvider' in ort.get_available_providers():
                providers.insert(0, 'CUDAExecutionProvider')
                
            ort_session = ort.InferenceSession(str(YOLO11_ONNX_PATH), providers=providers)
            input_name = ort_session.get_inputs()[0].name
            input_shape = ort_session.get_inputs()[0].shape
            input_h = input_shape[2] if isinstance(input_shape[2], int) else 640
            input_w = input_shape[3] if isinstance(input_shape[3], int) else 640
            
            use_onnx = True
            model_source = "yolo11n-onnx"
            logger.info(f"Successfully loaded newly created ONNX model. Input shape: {input_h}x{input_w}")
        
except Exception as e:
    logger.warning(f"Failed to load ONNX model: {e}. Will try PyTorch instead.")
    use_onnx = False
    ort_session = None

# --- Load PyTorch Model if ONNX fails ---
if not use_onnx:
    try:
        from ultralytics import YOLO
        
        # Enable memory-efficient inference with torch
        torch.set_grad_enabled(False)  # Disable gradient calculation
        # Set lower precision to reduce memory usage
        if torch.cuda.is_available():
            torch.cuda.set_device(0)  # Use first GPU
            torch.set_default_tensor_type(torch.cuda.FloatTensor)
            logger.info("CUDA is available for PyTorch inference")
        else:
            # Force CPU inference with optimized settings
            torch.set_num_threads(1)  # Reduce number of threads
            logger.info("Using CPU for PyTorch inference")
        
        # First try to load our fine-tuned model
        if FINE_TUNED_MODEL_PATH.exists():
            try:
                logger.info(f"Loading fine-tuned YOLOv11 model from: {FINE_TUNED_MODEL_PATH}")
                model = YOLO(str(FINE_TUNED_MODEL_PATH))
                # Verify model architecture (check if it's YOLOv11)
                model_type = getattr(model, 'type', 'unknown')
                model_source = "fine-tuned-pytorch"
                use_pytorch = True
                logger.info(f"Successfully loaded fine-tuned YOLO model (type: {model_type})")
            except Exception as e:
                logger.error(f"Failed to load fine-tuned model: {e}")
                model = None
        
        # If fine-tuned model failed, use the base YOLO11 model from ultralytics
        if model is None:
            try:
                # Check if YOLO11 base model exists, download if not
                if not YOLO11_MODEL_PATH.exists():
                    logger.info(f"YOLO11 base model not found at {YOLO11_MODEL_PATH}, downloading from ultralytics")
                    if not download_model("yolo11n.pt", YOLO11_MODEL_PATH):
                        logger.error("Failed to download YOLO11 model from ultralytics.")
                        # Don't exit, let the backend handle the failure and use mock inference
                
                # If the model exists now (either it was there or download succeeded)
                if YOLO11_MODEL_PATH.exists():
                    logger.info(f"Loading YOLO11 base model from: {YOLO11_MODEL_PATH}")
                    model = YOLO(str(YOLO11_MODEL_PATH))
                    model_source = "yolo11n-pytorch"
                    use_pytorch = True
                    logger.info("Successfully loaded YOLO11 base model from ultralytics")
                    
                    # Optionally export to ONNX for future use
                    if HAVE_ONNX and not YOLO11_ONNX_PATH.exists():
                        export_model_to_onnx(YOLO11_MODEL_PATH, YOLO11_ONNX_PATH)
            except Exception as e:
                logger.error(f"Failed to load YOLO11 base model: {e}")
                model = None
        
    except ImportError as e:
        logger.error(f"Failed to import ultralytics: {e}. Please install it with 'pip install ultralytics'")
        model = None
        use_pytorch = False

# --- Functions for ONNX Inference ---
def preprocess_onnx(frame: np.ndarray) -> np.ndarray:
    """Prepare a frame for ONNX inference."""
    with PREPROCESS_TIME.time():
        # Resize frame to model input size
        img = cv2.resize(frame, (input_w, input_h))
        
        # Convert BGR to RGB and normalize
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        
        # HWC to CHW format (batch, channels, height, width)
        img = img.transpose(2, 0, 1)
        
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        
    return img

def postprocess_onnx(outputs, orig_img_shape, conf_threshold=0.45, iou_threshold=0.45):
    """Process ONNX model output to get boxes, scores, and classes."""
    with POSTPROCESS_TIME.time():
        # ONNX outputs will be in format [1, 84, 8400] for YOLOv8/11
        # Where 84 = 4 (bbox) + 80 (classes) and 8400 is number of anchors
        output = outputs[0]  # outputs[0] has shape [1, 84, 8400]
        
        # Transpose to [1, 8400, 84]
        output = output.transpose((0, 2, 1))
        
        # Get boxes, scores and classes
        boxes = output[..., :4]  # x, y, w, h
        conf = output[..., 4:]  # class scores
        
        # Get highest score and corresponding class for each box
        scores = np.max(conf, axis=2)
        class_ids = np.argmax(conf, axis=2)
        
        # Filter by confidence
        mask = scores > conf_threshold
        boxes = boxes[mask]
        scores = scores[mask]
        class_ids = class_ids[mask]
        
        if len(boxes) == 0:
            return [], [], []
        
        # Convert from xywh to xyxy (left, top, right, bottom)
        x = boxes[:, 0]
        y = boxes[:, 1]
        w = boxes[:, 2]
        h = boxes[:, 3]
        
        # Convert to xyxy format
        xyxy = np.zeros_like(boxes)
        xyxy[:, 0] = x - w/2  # left
        xyxy[:, 1] = y - h/2  # top
        xyxy[:, 2] = x + w/2  # right
        xyxy[:, 3] = y + h/2  # bottom
        
        # Scale boxes to original image size
        img_h, img_w = orig_img_shape[:2]
        xyxy[:, 0] *= img_w / input_w
        xyxy[:, 2] *= img_w / input_w
        xyxy[:, 1] *= img_h / input_h
        xyxy[:, 3] *= img_h / input_h
        
        # Apply Non-Maximum Suppression (NMS)
        indices = cv2.dnn.NMSBoxes(xyxy.tolist(), scores.tolist(), conf_threshold, iou_threshold)
        
        if len(indices) > 0:
            # If indices is just a flat array (as in OpenCV 4.6+)
            if isinstance(indices[0], (int, np.integer)):  
                final_boxes = xyxy[indices].tolist()
                final_scores = scores[indices].tolist()
                final_class_ids = class_ids[indices].tolist()
            else:  # If indices is a 2D array (older OpenCV)
                final_boxes = xyxy[indices.flatten()].tolist()
                final_scores = scores[indices.flatten()].tolist() 
                final_class_ids = class_ids[indices.flatten()].tolist()
                
            return final_boxes, final_scores, final_class_ids
        else:
            return [], [], []

# --- Universal Inference Function ---
def infer(frame: np.ndarray):
    """
    Performs inference on a single frame using the best available engine.

    Args:
        frame: The input image frame (NumPy array BGR).

    Returns:
        Tuple[list, list, list]: Bounding boxes, scores, and class indices.
    """
    boxes, scores, classes = [], [], []

    if frame is None:
        logger.warning("Received None frame, skipping inference.")
        return boxes, scores, classes

    # Try ONNX inference first if available
    if use_onnx and ort_session is not None:
        try:
            with INFERENCE_TIME.time():
                # Preprocess image
                input_tensor = preprocess_onnx(frame)
                
                # Run inference
                outputs = ort_session.run(None, {ort_session.get_inputs()[0].name: input_tensor})
                
                # Postprocess results
                boxes, scores, classes = postprocess_onnx(outputs, frame.shape)
                
                if boxes:
                    logger.info(f"ONNX inference found {len(boxes)} objects.")
                else:
                    logger.debug("ONNX inference: no objects detected.")
                    
                INFERENCE_COUNT.inc()
                return boxes, scores, classes
                
        except Exception as e:
            logger.error(f"Error during ONNX inference: {e}. Falling back to PyTorch.")
            # Fall through to PyTorch inference
    
    # Try PyTorch inference if ONNX failed or is not available
    if use_pytorch and model is not None:
        try:
            with INFERENCE_TIME.time():
                # YOLO model processes the image internally
                results = model(frame, conf=0.45, iou=0.45, verbose=False)[0]
                
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
                    logger.debug("PyTorch inference: no objects detected.")
                    
            INFERENCE_COUNT.inc()
            
            # Clean up to reduce memory usage
            del results
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            logger.error(f"Error during PyTorch inference: {e}")
            return [], [], []
    else:
        # Neither inference method is available
        logger.error("No inference engine available (ONNX or PyTorch)")

    return boxes, scores, classes

# --- Cleanup function ---
def cleanup():
    global ort_session, model
    logger.info("Releasing resources...")
    if ort_session:
        del ort_session
        logger.info("ONNX session released.")
    if model:
        del model
        logger.info("PyTorch model released.")
    # Clean up GPU memory if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

atexit.register(cleanup)

# --- Final status report ---
if not use_onnx and not use_pytorch:
    logger.warning("No working inference engine found. Backend may fall back to mock inference.")
else:
    logger.info(f"Inference engine initialized using {model_source} model.")
    # Store model type info in a file for monitoring
    with open(BASE_DIR / "model_status.txt", "w") as f:
        f.write(f"model_source={model_source}\n")
        f.write(f"use_onnx={use_onnx}\n")
        f.write(f"use_pytorch={use_pytorch}\n")
