import cv2
import numpy as np
import logging
from pathlib import Path
import atexit
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
    # Use a different port than the backend server (which uses 8080)
    start_http_server(8001)
    logger.info("Prometheus metrics server started on port 8001.")
except OSError as e:
    logger.warning(f"Could not start Prometheus server on port 8001: {e}. Metrics may not be available.")

# Define paths relative to the script
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
MODELS_DIR = BASE_DIR / "models"


# Model paths
FINE_TUNED_MODEL_PATH = MODELS_DIR / 'best.pt'       # Fine-tuned model
FINE_TUNED_ONNX_PATH = MODELS_DIR / 'best.onnx'      # ONNX version of fine-tuned model
YOLO11_MODEL_PATH = MODELS_DIR / 'yolo11n.pt'        # Pre-trained base YOLO11 model
YOLO11_ONNX_PATH = MODELS_DIR / 'yolo11n.onnx'       # ONNX version of the base model

# Flags to determine which model to use
use_pytorch = False
use_onnx = False

# ONNX Runtime session for inference (initialized later)
model = None  # Initialize PyTorch model variable
ort_session = None  # Initialize ONNX session
input_h, input_w = 640, 640  # Standard YOLO input size
model_source = ""  # Track which model we're using

# COCO class names for the YOLOv11 model
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Add KITTI-specific classes
ALL_CLASSES = COCO_CLASSES + ['cyclist', 'van', 'person_sitting', 'tram', 'misc']

# Function to download the YOLOv11 model
def download_model():
    """
    Download the YOLOv11 model from ultralytics.
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        from ultralytics import YOLO
        
        logger.info("Downloading YOLOv11n model...")
        # This will download the model and use it directly
        model = YOLO('yolo11n.pt')
        
        # Save the model to our directory
        model_path = BASE_DIR / 'models/yolo11n.pt'
        model.save(model_path)
        
        logger.info(f"YOLOv11n model saved to {model_path}")
        return True
    except Exception as e:
        logger.error(f"Error downloading YOLOv11n model: {e}")
        return False

# --- Export PyTorch model to ONNX ---
def export_model_to_onnx(model_path, output_path=None):
    """Export YOLOv11 PyTorch model to ONNX format"""
    try:
        from ultralytics import YOLO
        
        if output_path is None:
            output_path = model_path.with_suffix('.onnx')
            
        logger.info(f"Exporting {model_path} to ONNX format...")
        model = YOLO(str(model_path))
        success = model.export(format="onnx", imgsz=[input_h, input_w], simplify=True)
        
        # Check if the file was created
        if output_path.exists():
            logger.info(f"Successfully exported to {output_path}")
            return True
        else:
            logger.error(f"ONNX export failed, file not found at {output_path}")
            return False
    
    except Exception as e:
        logger.error(f"Error exporting model to ONNX: {e}")
        return False

# --- Try loading ONNX model ---
try:
    # Only import onnxruntime if we're going to use it
    try:
        import onnxruntime as ort
        HAVE_ONNX = True
        logger.info("ONNX Runtime imported successfully")
    except ImportError:
        logger.warning("ONNX Runtime not available. Install it with 'pip install onnxruntime' for faster inference than PyTorch.")
        HAVE_ONNX = False
    
    if HAVE_ONNX:
        # First try to use fine-tuned ONNX model
        if FINE_TUNED_ONNX_PATH.exists():
            try:
                logger.info(f"Loading fine-tuned ONNX model from: {FINE_TUNED_ONNX_PATH}")
                ort_session = ort.InferenceSession(
                    str(FINE_TUNED_ONNX_PATH), 
                    providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
                )
                use_onnx = True
                model_source = "best-onnx"
                logger.info("Successfully loaded fine-tuned ONNX model")
            except Exception as e:
                logger.error(f"Error loading fine-tuned ONNX model: {e}")
                use_onnx = False
                ort_session = None
        
        # Try base YOLO11n ONNX model if fine-tuned wasn't loaded
        if not use_onnx and YOLO11_ONNX_PATH.exists():
            try:
                logger.info(f"Loading YOLOv11n ONNX model from: {YOLO11_ONNX_PATH}")
                ort_session = ort.InferenceSession(
                    str(YOLO11_ONNX_PATH), 
                    providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
                )
                use_onnx = True
                model_source = "yolo11n-onnx"
                logger.info("Successfully loaded YOLOv11n ONNX model")
            except Exception as e:
                logger.error(f"Error loading YOLOv11n ONNX model: {e}")
                use_onnx = False
                ort_session = None
                
        # If ONNX model doesn't exist, try to export it from PyTorch
        if not use_onnx and (FINE_TUNED_MODEL_PATH.exists() or YOLO11_MODEL_PATH.exists()):
            try:
                logger.info("Attempting to export PyTorch model to ONNX...")
                
                if FINE_TUNED_MODEL_PATH.exists():
                    export_success = export_model_to_onnx(FINE_TUNED_MODEL_PATH)
                    if export_success and FINE_TUNED_ONNX_PATH.exists():
                        logger.info(f"Loading newly exported fine-tuned ONNX model")
                        ort_session = ort.InferenceSession(
                            str(FINE_TUNED_ONNX_PATH), 
                            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
                        )
                        use_onnx = True
                        model_source = "best-onnx-exported"
                
                if not use_onnx and YOLO11_MODEL_PATH.exists():
                    export_success = export_model_to_onnx(YOLO11_MODEL_PATH)
                    if export_success and YOLO11_ONNX_PATH.exists():
                        logger.info(f"Loading newly exported YOLOv11n ONNX model")
                        ort_session = ort.InferenceSession(
                            str(YOLO11_ONNX_PATH), 
                            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
                        )
                        use_onnx = True
                        model_source = "yolo11n-onnx-exported"
                
            except Exception as e:
                logger.error(f"Error exporting or loading exported ONNX model: {e}")
                use_onnx = False
                ort_session = None
                
except Exception as e:
    logger.warning(f"Failed to load ONNX model: {e}. Will try PyTorch instead.")
    use_onnx = False
    ort_session = None

# --- Load PyTorch Model if ONNX is not available ---
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
        
        # First try to load the fine-tuned model
        try:
            if FINE_TUNED_MODEL_PATH.exists():
                logger.info(f"Loading fine-tuned model from: {FINE_TUNED_MODEL_PATH}")
                model = YOLO(str(FINE_TUNED_MODEL_PATH))
                model_source = "best-pytorch"
                use_pytorch = True
                logger.info("Successfully loaded fine-tuned model")
            else:
                # Try to load the YOLO11 model
                if not YOLO11_MODEL_PATH.exists():
                    logger.info("Downloading YOLOv11n model...")
                    download_model()
                    
                # Load the model
                if YOLO11_MODEL_PATH.exists():
                    logger.info(f"Loading YOLOv11n model from: {YOLO11_MODEL_PATH}")
                    model = YOLO(str(YOLO11_MODEL_PATH))
                    model_source = "yolo11n-pytorch"
                    use_pytorch = True
                    logger.info("Successfully loaded YOLOv11n model")
                else:
                    # If local file doesn't exist, load directly from ultralytics
                    logger.info("Loading YOLOv11n model directly from ultralytics")
                    model = YOLO('yolo11n.pt')
                    model_source = "yolo11n-pytorch-direct"
                    use_pytorch = True
                    logger.info("Successfully loaded YOLOv11n model directly")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
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
    Performs inference on a single frame using ONNX or PyTorch model.

    Args:
        frame: The input image frame (NumPy array BGR).

    Returns:
        Tuple[list, list, list]: Bounding boxes, scores, and class indices.
    """
    boxes, scores, classes = [], [], []

    if frame is None:
        logger.warning("Received None frame, skipping inference.")
        return boxes, scores, classes
        
    # Try ONNX inference
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

# Function to get class names
def get_class_names():
    """Returns the class names for the model"""
    return ALL_CLASSES

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
    engine_type = "ONNX" if use_onnx else "PyTorch"
    logger.info(f"Inference engine initialized using {'fine-tuned' if 'best' in model_source else 'base'} model with {engine_type} ({model_source}).")
    # Store model type info in a file for monitoring
    with open(BASE_DIR / "model_status.txt", "w") as f:
        f.write(f"model_source={model_source}\n")
        f.write(f"use_onnx={use_onnx}\n")
        f.write(f"use_pytorch={use_pytorch}\n")
        f.write(f"using_fine_tuned={'True' if 'best' in model_source else 'False'}\n")
