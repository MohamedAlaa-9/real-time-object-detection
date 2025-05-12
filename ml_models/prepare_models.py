#!/usr/bin/env python3
"""
Model Preparation Pipeline Script

This script prepares both the fine-tuned model and the official pre-trained model,
ensuring that the backend service can use either one with proper fallback logic.

Steps:
1. Check if the base YOLO11n model exists, download if not
2. Export the base model to ONNX format for faster inference
3. Check if a fine-tuned model exists, try to symlink it if available
4. Export the fine-tuned model to ONNX if available
5. Verify that at least one model is ready for inference
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

# Define model paths
YOLO11_MODEL_PATH = BASE_DIR / 'yolo11n.pt'
YOLO11_ONNX_PATH = BASE_DIR / 'yolo11n.onnx'
FINE_TUNED_MODEL_PATH = BASE_DIR / 'best.pt'
FINE_TUNED_ONNX_PATH = BASE_DIR / 'best.onnx'

# Try importing required libraries
try:
    import torch
    from ultralytics import YOLO
    HAS_ULTRALYTICS = True
except ImportError:
    logger.error("Failed to import required libraries. Please install them with:")
    logger.error("pip install torch ultralytics")
    HAS_ULTRALYTICS = False

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
        logger.info(f"Base YOLO11n model already exists at {YOLO11_MODEL_PATH}")
        return True
    
    try:
        logger.info("Downloading YOLO11n model...")
        model = YOLO("yolo11n.pt")
        
        # Check if download succeeded (model should now be accessible)
        if not isinstance(model.model, torch.nn.Module):
            logger.error("Failed to load YOLOv11n model")
            return False
        
        # Save model to our target location
        model.save(str(YOLO11_MODEL_PATH))
        
        if YOLO11_MODEL_PATH.exists():
            logger.info(f"Successfully downloaded and saved YOLO11n model to {YOLO11_MODEL_PATH}")
            return True
        else:
            logger.error(f"Model download appeared to succeed, but {YOLO11_MODEL_PATH} doesn't exist")
            return False
    
    except Exception as e:
        logger.error(f"Error downloading YOLO11n model: {e}")
        return False


def export_to_onnx(model_path, output_path=None, imgsz=640):
    """Export PyTorch model to ONNX format."""
    if not model_path.exists():
        logger.error(f"Source model {model_path} doesn't exist, cannot export to ONNX")
        return False
    
    # If no output path provided, use the same name with .onnx extension
    if output_path is None:
        output_path = model_path.with_suffix('.onnx')
    
    try:
        logger.info(f"Exporting {model_path} to ONNX format...")
        model = YOLO(str(model_path))
        
        success = model.export(format="onnx", imgsz=imgsz, simplify=True)
        
        # The export function creates the file with the same name but .onnx extension
        expected_path = model_path.with_suffix('.onnx')
        
        if expected_path.exists() and expected_path != output_path:
            logger.info(f"Moving {expected_path} to {output_path}")
            shutil.move(expected_path, output_path)
        
        if output_path.exists():
            logger.info(f"Successfully exported to {output_path}")
            return True
        else:
            logger.error(f"ONNX export failed: {output_path} not created")
            return False
    
    except Exception as e:
        logger.error(f"Error exporting model to ONNX: {e}")
        return False


def find_latest_trained_model():
    """Find the latest fine-tuned model from the runs directory."""
    runs_dir = PROJECT_ROOT / "runs" / "train"
    if not runs_dir.exists():
        logger.warning(f"Training directory {runs_dir} doesn't exist")
        return None
    
    # Look for weight files in the runs directory
    weight_files = []
    for exp_dir in runs_dir.iterdir():
        if exp_dir.is_dir():
            weights_dir = exp_dir / "weights"
            if weights_dir.exists():
                for weight_file in weights_dir.glob("*.pt"):
                    if weight_file.name in ["best.pt", "last.pt"]:
                        weight_files.append((weight_file, weight_file.stat().st_mtime))
    
    if not weight_files:
        logger.warning("No trained model weights found")
        return None
    
    # Sort by modification time, newest first
    weight_files.sort(key=lambda x: x[1], reverse=True)
    latest_model = weight_files[0][0]
    logger.info(f"Found latest trained model: {latest_model} (modified {time.ctime(weight_files[0][1])})")
    
    return latest_model


def link_fine_tuned_model():
    """Find and link the best fine-tuned model."""
    latest_model = find_latest_trained_model()
    
    if not latest_model:
        logger.warning("No fine-tuned model found")
        return False
    
    try:
        # If the target is already a symlink, remove it
        if FINE_TUNED_MODEL_PATH.is_symlink() or FINE_TUNED_MODEL_PATH.exists():
            FINE_TUNED_MODEL_PATH.unlink()
        
        # Create a symlink to the latest model
        FINE_TUNED_MODEL_PATH.symlink_to(latest_model)
        
        if FINE_TUNED_MODEL_PATH.exists():
            logger.info(f"Successfully linked {FINE_TUNED_MODEL_PATH} → {latest_model}")
            return True
        else:
            logger.error(f"Failed to create symlink to {latest_model}")
            return False
    
    except Exception as e:
        logger.error(f"Error linking fine-tuned model: {e}")
        return False


def verify_model_loading():
    """Verify that models can be loaded."""
    results = {
        "yolo11n_pt": False,
        "yolo11n_onnx": False,
        "fine_tuned_pt": False,
        "fine_tuned_onnx": False
    }
    
    # Verify PyTorch models
    if HAS_ULTRALYTICS:
        # Test base YOLO11n model
        if YOLO11_MODEL_PATH.exists():
            try:
                model = YOLO(str(YOLO11_MODEL_PATH))
                logger.info(f"Successfully loaded base YOLO11n model: {model.type}")
                results["yolo11n_pt"] = True
            except Exception as e:
                logger.error(f"Failed to load base YOLO11n model: {e}")
        
        # Test fine-tuned model
        if FINE_TUNED_MODEL_PATH.exists():
            try:
                model = YOLO(str(FINE_TUNED_MODEL_PATH))
                logger.info(f"Successfully loaded fine-tuned model: {model.type}")
                results["fine_tuned_pt"] = True
            except Exception as e:
                logger.error(f"Failed to load fine-tuned model: {e}")
    
    # Verify ONNX models
    if HAS_ONNX:
        # Test base YOLO11n ONNX model
        if YOLO11_ONNX_PATH.exists():
            try:
                session = ort.InferenceSession(str(YOLO11_ONNX_PATH), providers=['CPUExecutionProvider'])
                logger.info(f"Successfully loaded base YOLO11n ONNX model")
                results["yolo11n_onnx"] = True
            except Exception as e:
                logger.error(f"Failed to load base YOLO11n ONNX model: {e}")
        
        # Test fine-tuned ONNX model
        if FINE_TUNED_ONNX_PATH.exists():
            try:
                session = ort.InferenceSession(str(FINE_TUNED_ONNX_PATH), providers=['CPUExecutionProvider'])
                logger.info(f"Successfully loaded fine-tuned ONNX model")
                results["fine_tuned_onnx"] = True
            except Exception as e:
                logger.error(f"Failed to load fine-tuned ONNX model: {e}")
    
    return results


def display_summary(results):
    """Display a summary of model preparation results."""
    print("\n" + "=" * 60)
    print(" MODEL PREPARATION SUMMARY ".center(60, "="))
    print("=" * 60)
    
    # PyTorch models
    print("\nPyTorch Models:")
    print(f"  Base YOLO11n model: {'✅ Ready' if results['yolo11n_pt'] else '❌ Not available'}")
    print(f"  Fine-tuned model:   {'✅ Ready' if results['fine_tuned_pt'] else '❌ Not available'}")
    
    # ONNX models
    print("\nONNX Models:")
    print(f"  Base YOLO11n model: {'✅ Ready' if results['yolo11n_onnx'] else '❌ Not available'}")
    print(f"  Fine-tuned model:   {'✅ Ready' if results['fine_tuned_onnx'] else '❌ Not available'}")
    
    # Overall status
    any_ready = any(results.values())
    print("\nOverall Status:")
    if any_ready:
        print("  ✅ At least one model is ready for inference")
        
        # Determine fallback order
        if results["fine_tuned_onnx"]:
            print("  ➡️ System will use: Fine-tuned ONNX model (fastest)")
        elif results["fine_tuned_pt"]:
            print("  ➡️ System will use: Fine-tuned PyTorch model")
        elif results["yolo11n_onnx"]:
            print("  ➡️ System will use: Base YOLO11n ONNX model")
        elif results["yolo11n_pt"]:
            print("  ➡️ System will use: Base YOLO11n PyTorch model")
    else:
        print("  ❌ No models are ready for inference")
        print("  ⚠️ System will fall back to mock inference")
    
    print("\nNext Steps:")
    if not any_ready:
        print("  - Run this script with --download to get the base YOLO11n model")
    elif not (results["fine_tuned_pt"] or results["fine_tuned_onnx"]):
        print("  - Train a model with 'python ml_models/train_yolo.py'")
        print("  - Or manually place a trained model at ml_models/best.pt")
    elif not (results["fine_tuned_onnx"] or results["yolo11n_onnx"]):
        print("  - Export to ONNX with --export for better performance")
    else:
        print("  - All models are prepared! Start the backend with:")
        print("    python backend/main.py")
    
    print("=" * 60 + "\n")


def main():
    """Main entry point for the model preparation script."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--all", action="store_true", help="Prepare all models (both fine-tuned and base)")
    parser.add_argument("--fine-tuned", action="store_true", help="Prepare only the fine-tuned model")
    parser.add_argument("--base", action="store_true", help="Prepare only the base YOLOv11 model")
    parser.add_argument("--verify", action="store_true", help="Only verify model loading without preparation")
    args = parser.parse_args()
    
    # Handle the case where no arguments are provided - default to all
    if not (args.all or args.fine_tuned or args.base or args.verify):
        args.all = True
    
    results = {
        "yolo11n_pt": False, 
        "yolo11n_onnx": False,
        "fine_tuned_pt": False, 
        "fine_tuned_onnx": False
    }
    
    # If we're only verifying, skip preparation
    if args.verify:
        logger.info("Verifying model loading without preparation...")
        verify_model_loading()
        return
    
    # Prioritize fine-tuned model if requested or 'all' is specified
    if args.all or args.fine_tuned:
        logger.info("Preparing fine-tuned model...")
        
        # 1. Link the fine-tuned model or copy from the latest training run
        if not FINE_TUNED_MODEL_PATH.exists():
            results["fine_tuned_pt"] = link_fine_tuned_model()
        else:
            logger.info(f"Fine-tuned model already exists at {FINE_TUNED_MODEL_PATH}")
            results["fine_tuned_pt"] = True
            
        # 2. Export the fine-tuned model to ONNX for faster inference
        if results["fine_tuned_pt"]:
            results["fine_tuned_onnx"] = export_to_onnx(FINE_TUNED_MODEL_PATH, FINE_TUNED_ONNX_PATH)
    
    # Prepare base model if requested or if fine-tuned model preparation failed
    if args.all or args.base or (args.fine_tuned and not results["fine_tuned_pt"]):
        logger.info("Preparing base YOLOv11 model...")
        
        # 1. Download the base YOLO11n model if needed
        if not YOLO11_MODEL_PATH.exists():
            results["yolo11n_pt"] = download_yolo11n()
        else:
            logger.info(f"Base YOLO11n model already exists at {YOLO11_MODEL_PATH}")
            results["yolo11n_pt"] = True
        
        # 2. Export base model to ONNX if needed
        if results["yolo11n_pt"] and not YOLO11_ONNX_PATH.exists():
            results["yolo11n_onnx"] = export_to_onnx(YOLO11_MODEL_PATH, YOLO11_ONNX_PATH)
        elif YOLO11_ONNX_PATH.exists():
            logger.info(f"YOLO11n ONNX model already exists at {YOLO11_ONNX_PATH}")
            results["yolo11n_onnx"] = True
    
    # Verify the models can be loaded
    if verify_model_loading():
        logger.info("All models loaded successfully!")
    else:
        logger.warning("Some models could not be loaded. Check the logs for details.")
    
    # Display summary of the preparation process
    display_summary(results)
    
    # Create a symlink for best.pt in the project root if it doesn't exist
    project_best_path = PROJECT_ROOT / "best.pt"
    if FINE_TUNED_MODEL_PATH.exists() and not project_best_path.exists():
        try:
            project_best_path.symlink_to(FINE_TUNED_MODEL_PATH)
            logger.info(f"Created project root symlink to fine-tuned model: {project_best_path}")
        except Exception as e:
            logger.warning(f"Failed to create project root symlink: {e}")
    
    return results


if __name__ == "__main__":
    main()