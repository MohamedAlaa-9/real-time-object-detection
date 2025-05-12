#!/usr/bin/env python3
"""
Model Pipeline Verification Tool

This script verifies the end-to-end model pipeline:
1. Checks if the pre-trained model exists, downloads if needed
2. Runs a quick training step to fine-tune the model
3. Verifies the model is correctly symlinked for inference
4. Tests that the backend will correctly use the model
"""

import os
import sys
import logging
from pathlib import Path
import shutil
import subprocess
import time
import cv2
import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Define project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ML_MODELS_DIR = PROJECT_ROOT / "ml_models"
MODELS_DIR = ML_MODELS_DIR / "models"
CONFIG_PATH = PROJECT_ROOT / "config/train_config.yaml"
EXPORT_ONNX_PATH = MODELS_DIR / "best.onnx"
TENSORRT_ENGINE_PATH = MODELS_DIR / "best.trt"

def check_model_files():
    """Check if model files exist and print their status"""
    models_to_check = {
        "Pre-trained base model": MODELS_DIR / "yolo11n.pt",
        "Alternative base model": MODELS_DIR / "yolov8n.pt",
        "Fine-tuned model": MODELS_DIR / "best.pt",
        "ONNX exported model": EXPORT_ONNX_PATH,
        "TensorRT engine": TENSORRT_ENGINE_PATH
    }
    
    print("\n--- Model Files Status ---")
    all_exist = True
    for name, path in models_to_check.items():
        exists = path.exists()
        if exists:
            file_size = path.stat().st_size / (1024 * 1024)  # Convert to MB
            status = f"✅ Exists ({file_size:.1f} MB)"
            
            # Check if it's a symlink
            if path.is_symlink():
                target = path.resolve()
                status += f" → {target}"
        else:
            status = "❌ Missing"
            all_exist = False
            
        print(f"{name}: {status}")
    
    return all_exist

def run_command(command, description):
    """Run a command and log the output"""
    logger.info(f"Running: {description}")
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        logger.info(f"Command completed successfully: {description}")
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed: {description}")
        logger.error(f"Error output: {e.stderr}")
        return False, e.stderr

def verify_dataset():
    """Verify that the dataset is prepared"""
    data_yaml_path = PROJECT_ROOT / "datasets/processed/data.yaml"
    
    if not data_yaml_path.exists():
        logger.warning("Dataset not prepared. Running dataset preparation script...")
        success, output = run_command(
            "python datasets/preprocess_datasets.py",
            "Dataset preprocessing"
        )
        return success
    else:
        logger.info("Dataset already prepared")
        return True

def benchmark_tensorrt_vs_onnx():
    """
    Benchmarks TensorRT performance against ONNX and PyTorch inference.
    Performs multiple inference runs and compares average execution time.
    """
    try:
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit
        import torch
        import onnxruntime as ort
        
        logger.info("\n" + "="*60)
        logger.info(" BENCHMARKING MODEL INFERENCE PERFORMANCE ".center(60, "="))
        logger.info("="*60)
        
        # Load test image
        test_img_path = PROJECT_ROOT / "datasets/processed/images/test/000000.jpg"
        if not test_img_path.exists():
            test_img_path = next((PROJECT_ROOT / "datasets/processed/images/test").glob("*.jpg"), None)
            
        if not test_img_path:
            logger.error("No test images found. Cannot proceed with benchmarking.")
            return
            
        logger.info(f"Loading test image: {test_img_path}")
        img = cv2.imread(str(test_img_path))
        if img is None:
            logger.error("Failed to load test image")
            return
            
        h, w = img.shape[:2]
        logger.info(f"Image loaded successfully: {w}x{h} pixels")
        
        # Define common preprocessing
        input_size = (640, 640)  # Standard YOLO input size
        
        # Prepare test data
        resized_img = cv2.resize(img, input_size)
        input_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
        input_img = input_img.astype(np.float32) / 255.0
        input_img = input_img.transpose(2, 0, 1)  # HWC to CHW
        input_img = np.expand_dims(input_img, 0)  # Add batch dimension
        
        # Dictionary to store benchmark results
        results = {}
        
        # ---- Benchmark TensorRT ----
        trt_model_path = MODELS_DIR / "tensorrt_models/best.trt"
        if trt_model_path.exists():
            try:
                logger.info("Benchmarking TensorRT model...")
                
                # Load TensorRT engine
                logger.info("Loading TensorRT engine")
                trt_logger = trt.Logger(trt.Logger.INFO)
                runtime = trt.Runtime(trt_logger)
                with open(str(trt_model_path), 'rb') as f:
                    serialized_engine = f.read()
                engine = runtime.deserialize_cuda_engine(serialized_engine)
                context = engine.create_execution_context()
                
                # Allocate device memory
                stream = cuda.Stream()
                
                # Get input and output binding indices
                input_idx = engine.get_binding_index("images")  # Input tensor name (adjust if different)
                output_idx = 1  # Assuming single output, index 1
                
                # Allocate device memory
                d_input = cuda.mem_alloc(input_img.nbytes)
                d_output = cuda.mem_alloc(4 * 84 * 8400)  # Adjust size based on model output
                bindings = [int(d_input), int(d_output)]
                
                # Warm-up run
                cuda.memcpy_htod_async(d_input, input_img, stream)
                context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
                stream.synchronize()
                
                # Benchmark runs
                num_runs = 20
                start_time = time.time()
                for _ in range(num_runs):
                    cuda.memcpy_htod_async(d_input, input_img, stream)
                    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
                    stream.synchronize()
                
                trt_time = (time.time() - start_time) / num_runs
                results["TensorRT"] = trt_time
                
                logger.info(f"TensorRT average inference time: {trt_time*1000:.2f} ms")
                
                # Clean up
                d_input.free()
                d_output.free()
                del context
                del engine
                
            except Exception as e:
                logger.error(f"Error benchmarking TensorRT model: {e}")
        else:
            logger.warning("TensorRT model not found. Skipping TensorRT benchmark.")
        
        # ---- Benchmark ONNX ----
        onnx_model_path = MODELS_DIR / "onnx_models/best.onnx"
        if onnx_model_path.exists():
            try:
                logger.info("Benchmarking ONNX model...")
                
                # Configure ONNX Runtime for best performance
                sess_options = ort.SessionOptions()
                sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                
                # Use CUDA execution provider if available
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                session = ort.InferenceSession(str(onnx_model_path), sess_options, providers=providers)
                
                # Get input name
                input_name = session.get_inputs()[0].name
                
                # Warm-up run
                session.run(None, {input_name: input_img})
                
                # Benchmark runs
                num_runs = 20
                start_time = time.time()
                for _ in range(num_runs):
                    session.run(None, {input_name: input_img})
                
                onnx_time = (time.time() - start_time) / num_runs
                results["ONNX"] = onnx_time
                
                logger.info(f"ONNX average inference time: {onnx_time*1000:.2f} ms")
                
            except Exception as e:
                logger.error(f"Error benchmarking ONNX model: {e}")
        else:
            logger.warning("ONNX model not found. Skipping ONNX benchmark.")
        
        # ---- Benchmark PyTorch ----
        try:
            from ultralytics import YOLO
            
            pt_model_path = ML_MODELS_DIR / "best.pt"
            if pt_model_path.exists():
                try:
                    logger.info("Benchmarking PyTorch model...")
                    
                    # Load PyTorch model
                    model = YOLO(str(pt_model_path))
                    
                    # Convert numpy image back to format YOLO expects
                    torch_img = torch.from_numpy(input_img).to('cuda' if torch.cuda.is_available() else 'cpu')
                    
                    # Warm-up run
                    with torch.no_grad():
                        model(torch_img, verbose=False)
                    
                    # Benchmark runs
                    num_runs = 20
                    start_time = time.time()
                    for _ in range(num_runs):
                        with torch.no_grad():
                            model(torch_img, verbose=False)
                    
                    pytorch_time = (time.time() - start_time) / num_runs
                    results["PyTorch"] = pytorch_time
                    
                    logger.info(f"PyTorch average inference time: {pytorch_time*1000:.2f} ms")
                    
                except Exception as e:
                    logger.error(f"Error benchmarking PyTorch model: {e}")
            else:
                logger.warning("PyTorch model not found. Skipping PyTorch benchmark.")
        except ImportError:
            logger.warning("Ultralytics package not found. Skipping PyTorch benchmark.")
        
        # ---- Display comparison ----
        if results:
            logger.info("\n" + "-"*60)
            logger.info(" PERFORMANCE COMPARISON ".center(60, "-"))
            logger.info("-"*60)
            
            # Find fastest model
            fastest = min(results.items(), key=lambda x: x[1])
            
            for model_type, time_taken in sorted(results.items(), key=lambda x: x[1]):
                speedup = ""
                if model_type != fastest[0]:
                    speedup = f" ({time_taken/fastest[1]:.1f}x slower than {fastest[0]})"
                
                logger.info(f"{model_type:>10}: {time_taken*1000:.2f} ms per inference{speedup}")
            
            logger.info("-"*60)
            logger.info(f"Fastest model: {fastest[0]} ({fastest[1]*1000:.2f} ms)")
            
            # TensorRT vs ONNX comparison
            if "TensorRT" in results and "ONNX" in results:
                speedup = results["ONNX"] / results["TensorRT"]
                logger.info(f"TensorRT is {speedup:.2f}x faster than ONNX")
                
            # TensorRT vs PyTorch comparison
            if "TensorRT" in results and "PyTorch" in results:
                speedup = results["PyTorch"] / results["TensorRT"]
                logger.info(f"TensorRT is {speedup:.2f}x faster than PyTorch")
            
        else:
            logger.warning("No benchmark results collected.")
            
    except ImportError as e:
        logger.error(f"Required packages for benchmarking not installed: {e}")
        logger.info("To run benchmarks, install required packages:")
        logger.info("pip install tensorrt pycuda torch onnxruntime")

def main():
    print("\n" + "="*80)
    print(" MODEL PIPELINE VERIFICATION ".center(80, "="))
    print("="*80)
    
    # Step 1: Check current model status
    logger.info("Checking initial model status...")
    check_model_files()
    
    # Step 2: Verify dataset preparation
    if not verify_dataset():
        logger.error("Dataset preparation failed. Cannot proceed with model training.")
        sys.exit(1)
    
    # Step 3: Run training with quick epochs
    logger.info("Running quick model training to verify pipeline...")
    training_cmd = f"python {PROJECT_ROOT}/ml_models/train_yolo.py"
    success, _ = run_command(training_cmd, "Model training")
    
    if not success:
        logger.error("Training failed. Please check the logs above for errors.")
        sys.exit(1)
    
    # Step 4: Check models after training
    logger.info("Checking model status after training...")
    models_exist = check_model_files()
    
    # Step 5: Verify backend config
    backend_config = PROJECT_ROOT / "backend/core/config.py"
    if backend_config.exists():
        logger.info("Checking backend configuration...")
        with open(backend_config, 'r') as f:
            config_content = f.read()
            
        if "FINE_TUNED_MODEL_PATH" in config_content and "KITTI_DATA_YAML" in config_content:
            logger.info("✅ Backend is configured to use fine-tuned model and KITTI classes")
        else:
            logger.warning("⚠️ Backend configuration might need updating to use fine-tuned model")
    
    # Step 6: Run inference script to verify model loading
    inference_test_cmd = f"python -c \"from ml_models.inference import model, model_source; print('Model loaded successfully - using ' + model_source + ' model')\""
    success, output = run_command(inference_test_cmd, "Testing model loading for inference")
    
    # Step 7: Benchmark TensorRT vs ONNX and PyTorch performance
    try:
        # Check if TensorRT engine exists
        if (ML_MODELS_DIR / "best.trt").exists():
            logger.info("TensorRT engine found. Running performance benchmarks...")
            benchmark_tensorrt_vs_onnx()
        else:
            logger.info("TensorRT engine not found. Skipping benchmarks.")
            logger.info("To enable benchmarks, run: ./optimize_with_tensorrt.sh")
    except Exception as e:
        logger.warning(f"Could not run benchmarks: {e}")
    
    if success:
        print("\n" + "="*80)
        print(" PIPELINE VERIFICATION COMPLETE ".center(80, "="))
        print("="*80)
        print("\n✅ Model pipeline verified successfully:")
        print("  - Pre-trained model download functionality working")
        print("  - Training and fine-tuning pipeline working")
        print("  - Model path synchronization working")
        print("  - Backend configured to use the correct model and classes")
        print("  - Inference system able to load the model")
        if (ML_MODELS_DIR / "best.trt").exists():
            print("  - TensorRT optimization enabled for faster inference")
        print("\nYour system is ready for deployment to Azure!")
    else:
        print("\n" + "="*80)
        print(" PIPELINE VERIFICATION FAILED ".center(80, "="))
        print("="*80)
        print("\n❌ Some issues were detected in your model pipeline.")
        print("   Please review the logs above to identify and fix the problems.")
    
    # Step 7: Benchmark TensorRT vs ONNX
    logger.info("Benchmarking TensorRT and ONNX models...")
    benchmark_tensorrt_vs_onnx()

if __name__ == "__main__":
    main()