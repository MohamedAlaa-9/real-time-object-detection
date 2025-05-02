import logging
import os
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
BACKEND_ROOT = PROJECT_ROOT / "backend"
UPLOAD_DIR = PROJECT_ROOT / "uploads"
RESULTS_DIR = PROJECT_ROOT / "results"

# Ensure directories exist
UPLOAD_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# Load class names from data.yaml
DATA_YAML_PATH = PROJECT_ROOT / "datasets/processed/data.yaml"

try:
    import yaml
    with open(DATA_YAML_PATH, "r") as f:
        data = yaml.safe_load(f)
        CLASS_NAMES = data["names"]
        logger.info(f"Loaded class names: {CLASS_NAMES}")
except Exception as e:
    logger.error(f"Failed to load class names: {e}")
    CLASS_NAMES = ["pedestrian", "car", "cyclist"]  # Default KITTI classes
    logger.info(f"Using default class names: {CLASS_NAMES}")

# Generate colors for visualizing each class
import random
COLORS = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(len(CLASS_NAMES))]

# API settings
API_PORT = int(os.getenv("API_PORT", 8000))
API_HOST = os.getenv("API_HOST", "0.0.0.0")