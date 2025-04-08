import tensorrt as trt
import pycuda.driver as cuda # Required for TensorRT, even if not directly used here sometimes
import pycuda.autoinit # Initializes CUDA context
from pathlib import Path
import logging
import sys

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define base directory relative to this script
BASE_DIR = Path(__file__).resolve().parent

def build_engine(onnx_file_path: Path, engine_file_path: Path, trt_logger: trt.Logger, min_shape: tuple, opt_shape: tuple, max_shape: tuple):
    """Builds the TensorRT engine from an ONNX model."""
    
    if not onnx_file_path.exists():
        logger.error(f"ONNX file not found: {onnx_file_path}")
        return None

    builder = trt.Builder(trt_logger)
    
    # Create Network - EXPLICIT_BATCH allows dynamic batch size via optimization profiles
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    
    # Create ONNX Parser
    parser = trt.OnnxParser(network, trt_logger)

    # Parse ONNX model file
    logger.info(f"Parsing ONNX file: {onnx_file_path}")
    with open(onnx_file_path, 'rb') as model:
        if not parser.parse(model.read()):
            logger.error("Failed to parse the ONNX file.")
            for error in range(parser.num_errors):
                logger.error(parser.get_error(error))
            return None
    logger.info("ONNX file parsed successfully.")

    # --- Builder Configuration ---
    config = builder.create_builder_config()
    
    # Workspace Size: Adjust based on GPU memory (1 GiB example)
    # More workspace can allow TRT to find better tactics.
    config.max_workspace_size = 1 << 30 
    logger.info(f"Setting max workspace size to {config.max_workspace_size / (1024**2):.0f} MiB")

    # Precision Modes: FP16 or INT8 can improve performance
    if builder.platform_has_fast_fp16:
        logger.info("Enabling FP16 mode.")
        config.flags |= 1 << int(trt.BuilderFlag.FP16)
    # else: # INT8 requires calibration dataset and setup
    #     logger.info("FP16 not supported or enabled. Consider INT8 calibration for further optimization.")
    
    # --- Optimization Profile for Dynamic Shapes ---
    # Get input tensor name (assuming single input)
    input_tensor = network.get_input(0)
    input_name = input_tensor.name
    logger.info(f"Network input name: {input_name}, shape: {input_tensor.shape}")

    profile = builder.create_optimization_profile()
    # Set shape constraints for the input tensor.
    # Format: (min_shape, optimal_shape, max_shape)
    # Ensure these shapes are compatible with your model and use case.
    profile.set_shape(input_name, min_shape, opt_shape, max_shape)
    logger.info(f"Adding optimization profile for input '{input_name}':")
    logger.info(f"  MIN shape: {min_shape}")
    logger.info(f"  OPT shape: {opt_shape}")
    logger.info(f"  MAX shape: {max_shape}")
    config.add_optimization_profile(profile)

    # --- Build Engine ---
    logger.info("Building TensorRT engine... (This may take a while)")
    # builder.max_batch_size = max_shape[0] # Set max batch size for implicit batch networks if needed

    try:
        # Use build_serialized_network for newer TRT versions if build_engine is deprecated
        serialized_engine = builder.build_serialized_network(network, config)
        if serialized_engine is None:
            logger.error("Failed to build serialized engine.")
            return None
        
        logger.info("TensorRT engine built successfully.")

        # Save the engine to file
        logger.info(f"Saving engine to: {engine_file_path}")
        with open(engine_file_path, "wb") as f:
            f.write(serialized_engine)
        
        # Deserialize to return an ICudaEngine object (optional, if needed immediately)
        runtime = trt.Runtime(trt_logger)
        engine = runtime.deserialize_cuda_engine(serialized_engine)
        return engine

    except Exception as e:
        logger.error(f"Error during engine build: {e}")
        # Clean up partial file if it exists
        if engine_file_path.exists():
            try:
                engine_file_path.unlink()
            except OSError:
                pass
        return None


if __name__ == '__main__':
    # --- Configuration ---
    # Assumes the ONNX model is exported by export_model.py
    onnx_file = BASE_DIR / 'best.onnx' 
    engine_file = BASE_DIR / 'best.trt'
    
    # Define optimization profile shapes (Batch, Channel, Height, Width)
    # These should match the expected input dimensions during inference
    # Example: Allow batch size 1 only, fixed 640x640 resolution
    min_input_shape = (1, 3, 640, 640)
    opt_input_shape = (1, 3, 640, 640)
    max_input_shape = (1, 3, 640, 640)
    # If dynamic batch size is needed: e.g., min=(1,3,640,640), opt=(4,3,640,640), max=(8,3,640,640)

    # TensorRT Logger Severity
    trt_verbosity = trt.Logger.WARNING # Or INFO, VERBOSE for more details

    # --- Execution ---
    trt_logger = trt.Logger(trt_verbosity)
    engine = build_engine(onnx_file, engine_file, trt_logger, min_input_shape, opt_input_shape, max_input_shape)

    if engine:
        logger.info(f"Engine build complete. Saved to {engine_file}")
        # You can optionally inspect the engine bindings here
        # logger.info(f"Number of bindings: {engine.num_bindings}")
        # for i in range(engine.num_bindings):
        #     logger.info(f"Binding {i}: Name={engine.get_binding_name(i)}, Shape={engine.get_binding_shape(i)}, Dtype={engine.get_binding_dtype(i)}")
        sys.exit(0) # Success
    else:
        logger.error("Engine build failed.")
        sys.exit(1) # Failure
