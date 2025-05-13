#!/usr/bin/env python3
"""
Model Preparation Pipeline Script

"""

import os
import sys
import shutil
import logging
from pathlib import Path
import argparse
import time

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("ModelPreparation")

# Define base paths
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
MODELS_DIR = BASE_DIR / "models"

# Define model paths
YOLO11_MODEL_PATH = MODELS_DIR / 'yolo11n.pt'
YOLO11_ONNX_PATH = MODELS_DIR / 'yolo11n.onnx'

FINE_TUNED_MODEL_PATH = MODELS_DIR / 'best.pt'  # User is expected to place their model here
FINE_TUNED_ONNX_PATH = MODELS_DIR / 'best.onnx'

# Try importing required libraries
try:
    import torch
    from ultralytics import YOLO
    HAS_ULTRALYTICS = True
except ImportError:
    logger.error("Failed to import required libraries. Please install them with:")
    logger.error("pip install torch ultralytics")
    HAS_ULTRALYTICS = False

# Try importing ONNX runtime
try:
    import onnxruntime as ort
    HAS_ONNX = True
    logger.info(f"ONNX Runtime available. Providers: {ort.get_available_providers()}")
except ImportError:
    logger.warning("ONNX Runtime not available. Install with: pip install onnxruntime")
    HAS_ONNX = False


def download_yolo11n():
    """Download the YOLO11n model if it doesn't exist."""
    if YOLO11_MODEL_PATH.exists():
        logger.info(f"Model {YOLO11_MODEL_PATH} already exists.")
        return
    
    try:
        logger.info(f"Downloading YOLOv11n model to {YOLO11_MODEL_PATH}...")
        if not HAS_ULTRALYTICS:
            logger.error("Ultralytics library not found. Cannot download model.")
            return
        model = YOLO("yolov11n.pt")
        logger.info("YOLOv11n model is available via Ultralytics cache or was pre-existing.")

    except Exception as e:
        logger.error(f"Error downloading YOLOv11n model: {e}")


def export_to_onnx(model_path, output_path=None, imgsz=640):
    """Export PyTorch model to ONNX format."""
    if not HAS_ULTRALYTICS:
        logger.error("Ultralytics library not found. Cannot export model.")
        return
    if not model_path.exists():
        logger.warning(f"Model path {model_path} does not exist. Attempting to load via YOLO.")
    
    if output_path is None:
        output_path = model_path.with_suffix(".onnx")
    
    try:
        logger.info(f"Loading model {model_path} for ONNX export...")
        model = YOLO(str(model_path))
        logger.info(f"Exporting {model_path} to {output_path} with imgsz={imgsz}...")
        model.export(format="onnx", imgsz=imgsz, opset=12, simplify=True, dynamic=True, path=str(output_path))
        logger.info(f"Successfully exported model to {output_path}")
    
    except Exception as e:
        logger.error(f"Error exporting model {model_path} to ONNX: {e}")
        logger.error("Ensure the model path is correct and the environment has necessary packages (torch, onnx, onnx-simplifier).")


def verify_model_loading():
    """Verify that the primary ONNX models can be loaded."""
    logger.info("Verifying model loading...")
    models_to_check = []
    if FINE_TUNED_ONNX_PATH.exists():
        models_to_check.append(FINE_TUNED_ONNX_PATH)
    elif FINE_TUNED_MODEL_PATH.exists():
        logger.warning(f"{FINE_TUNED_MODEL_PATH} exists but corresponding ONNX model {FINE_TUNED_ONNX_PATH} not found. Consider exporting.")
    
    if YOLO11_ONNX_PATH.exists():
        models_to_check.append(YOLO11_ONNX_PATH)
    elif YOLO11_MODEL_PATH.exists():
         logger.warning(f"{YOLO11_MODEL_PATH} exists but corresponding ONNX model {YOLO11_ONNX_PATH} not found. Consider exporting.")

    if not models_to_check:
        logger.error("No primary ONNX models found to verify (e.g., best.onnx or yolo11n.onnx).")
        return

    if HAS_ONNX and ort:
        for model_path in models_to_check:
            try:
                logger.info(f"Attempting to load ONNX model: {model_path}")
                ort_session = ort.InferenceSession(str(model_path), providers=ort.get_available_providers())
                logger.info(f"Successfully loaded {model_path} with providers: {ort_session.get_providers()}")
                for i, input_meta in enumerate(ort_session.get_inputs()):
                    logger.info(f"  Input {i}: Name={input_meta.name}, Shape={input_meta.shape}, Type={input_meta.type}")
                for i, output_meta in enumerate(ort_session.get_outputs()):
                    logger.info(f"  Output {i}: Name={output_meta.name}, Shape={output_meta.shape}, Type={output_meta.type}")

            except Exception as e:
                logger.error(f"Failed to load ONNX model {model_path}: {e}")
    elif not HAS_ONNX:
        logger.warning("ONNX Runtime not available. Skipping ONNX model loading verification.")
    logger.info("Model loading verification finished.")


def display_summary(results):
    """Display a summary of the model preparation steps."""
    logger.info("--- Model Preparation Summary ---")
    for key, value in results.items():
        logger.info(f"{key}: {value}")
    logger.info("-------------------------------")


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    results = {
        "yolo11n_pt_available": False,
        "yolo11n_onnx_exported_or_exists": False,
        "fine_tuned_pt_exists": False,
        "fine_tuned_onnx_exported_or_exists": False,
    }
    logger.info("Starting model preparation...")

    if not YOLO11_MODEL_PATH.exists():
        logger.info(f"{YOLO11_MODEL_PATH} not found. Ultralytics will attempt to download 'yolov11n.pt' upon use.")
    else:
        results["yolo11n_pt_available"] = True
        logger.info(f"Base model {YOLO11_MODEL_PATH} found.")

    pt_model_for_base_onnx = YOLO11_MODEL_PATH if YOLO11_MODEL_PATH.exists() else "yolov11n.pt"
    export_needed = True
    if YOLO11_ONNX_PATH.exists():
        results["yolo11n_onnx_exported_or_exists"] = True
        if YOLO11_MODEL_PATH.exists() and YOLO11_MODEL_PATH.stat().st_mtime <= YOLO11_ONNX_PATH.stat().st_mtime:
            logger.info(f"ONNX model {YOLO11_ONNX_PATH} is up-to-date with {YOLO11_MODEL_PATH}.")
            export_needed = False
        elif not YOLO11_MODEL_PATH.exists():
             logger.info(f"ONNX model {YOLO11_ONNX_PATH} exists. Corresponding .pt not found at {YOLO11_MODEL_PATH}, assuming ONNX is usable.")
             export_needed = False

    if export_needed:
        logger.info(f"Exporting base model ({pt_model_for_base_onnx}) to ONNX at {YOLO11_ONNX_PATH}...")
        export_to_onnx(pt_model_for_base_onnx, YOLO11_ONNX_PATH)
        if YOLO11_ONNX_PATH.exists():
            results["yolo11n_onnx_exported_or_exists"] = True

    if FINE_TUNED_MODEL_PATH.exists():
        logger.info(f"User-provided fine-tuned model {FINE_TUNED_MODEL_PATH} found.")
        results["fine_tuned_pt_exists"] = True
        
        export_ft_needed = True
        if FINE_TUNED_ONNX_PATH.exists():
            results["fine_tuned_onnx_exported_or_exists"] = True
            if FINE_TUNED_MODEL_PATH.stat().st_mtime <= FINE_TUNED_ONNX_PATH.stat().st_mtime:
                logger.info(f"ONNX model {FINE_TUNED_ONNX_PATH} for fine-tuned model is up-to-date.")
                export_ft_needed = False
        
        if export_ft_needed:
            logger.info(f"Exporting {FINE_TUNED_MODEL_PATH} to ONNX at {FINE_TUNED_ONNX_PATH}...")
            export_to_onnx(FINE_TUNED_MODEL_PATH, FINE_TUNED_ONNX_PATH)
            if FINE_TUNED_ONNX_PATH.exists():
                 results["fine_tuned_onnx_exported_or_exists"] = True
    else:
        logger.info(f"User-provided fine-tuned .pt model ({FINE_TUNED_MODEL_PATH}) not found. "
                    f"If you have one, place it there. Otherwise, checking for existing {FINE_TUNED_ONNX_PATH}.")
        if FINE_TUNED_ONNX_PATH.exists():
            logger.info(f"User-provided fine-tuned .onnx model ({FINE_TUNED_ONNX_PATH}) found.")
            results["fine_tuned_onnx_exported_or_exists"] = True

    logger.info("Verifying loadable models...")
    verify_model_loading() 

    display_summary(results)
    logger.info("Model preparation finished.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    main()