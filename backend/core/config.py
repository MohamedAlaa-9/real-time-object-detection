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
# Check which model we're using to determine class names
FINE_TUNED_MODEL_PATH = ML_MODELS_DIR / 'best.pt'
KITTI_DATA_YAML = PROJECT_ROOT / "datasets" / "processed" / "data.yaml"
MODEL_SOURCE = "unknown"

# Default COCO class names (used as fallback)
COCO_CLASS_NAMES = [
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

# Try to load KITTI class names from data.yaml
KITTI_CLASS_NAMES = []
if KITTI_DATA_YAML.exists():
    try:
        with open(KITTI_DATA_YAML, 'r') as f:
            data_config = yaml.safe_load(f)
            if 'names' in data_config:
                KITTI_CLASS_NAMES = data_config['names']
                logger.info(f"Loaded KITTI class names from {KITTI_DATA_YAML}: {KITTI_CLASS_NAMES}")
    except Exception as e:
        logger.error(f"Failed to load KITTI class names: {e}")

# Determine which class names to use based on model
if FINE_TUNED_MODEL_PATH.exists() and KITTI_CLASS_NAMES:
    CLASS_NAMES = KITTI_CLASS_NAMES
    MODEL_SOURCE = "fine-tuned-kitti"
    logger.info("Using KITTI class names with fine-tuned model")
else:
    CLASS_NAMES = COCO_CLASS_NAMES
    MODEL_SOURCE = "coco-default"
    logger.info("Using COCO class names (default)")

logger.info(f"Using {MODEL_SOURCE} model with {len(CLASS_NAMES)} classes")

# Generate colors for visualizing each class
import random
# Ensure enough colors are generated for all classes
COLORS = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(len(CLASS_NAMES))]

# API settings
API_PORT = int(os.getenv("API_PORT", 8081))  # Ensure this is 8081
API_HOST = os.getenv("API_HOST", "0.0.0.0")