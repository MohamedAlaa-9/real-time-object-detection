import logging
import os
import yaml
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
BACKEND_ROOT = PROJECT_ROOT / "backend"
UPLOAD_DIR = PROJECT_ROOT / "uploads"
RESULTS_DIR = PROJECT_ROOT / "results"
ML_MODELS_DIR = PROJECT_ROOT / "ml_models"

# Ensure directories exist
UPLOAD_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# --- Model and Class Names Configuration ---
# We're using YOLOv11 model with COCO classes only
MODEL_SOURCE = "yoloV11"

# COCO class names for the YOLOv11 model
CLASS_NAMES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
]

logger.info(f"Using {MODEL_SOURCE} model with {len(CLASS_NAMES)} classes")

# Generate colors for visualizing each class
import random
# Ensure enough colors are generated for all classes
COLORS = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(len(CLASS_NAMES))]

# API settings
API_PORT = int(os.getenv("API_PORT", 8081))  # Ensure this is 8081
API_HOST = os.getenv("API_HOST", "0.0.0.0")