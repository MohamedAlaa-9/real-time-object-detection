import tensorrt as trt
import pycuda.driver as cuda # Required for TensorRT, even if not directly used here sometimes
import pycuda.autoinit # Initializes CUDA context
from pathlib import Path
import logging
import sys
import yaml # Added for YAML loading

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define project root directory (assuming script is in ml-models/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

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
    # --- Configuration Loading ---
    CONFIG_PATH = PROJECT_ROOT / "config/train_config.yaml"

    if not CONFIG_PATH.exists():
        logger.error(f"Configuration file not found at {CONFIG_PATH}")
        sys.exit(1)

    logger.info(f"Loading configuration from: {CONFIG_PATH}")
    with open(CONFIG_PATH, 'r') as f:
        try:
            config = yaml.safe_load(f)
            if 'export' not in config:
                logger.error("Missing 'export' section in configuration file.")
                sys.exit(1)
            export_config = config['export']
            logger.info("Loaded export configuration relevant for optimization:")
            logger.info(f"  onnx_output_path: {export_config['onnx_output_path']}")
            logger.info(f"  input_height: {export_config['input_height']}")
            logger.info(f"  input_width: {export_config['input_width']}")
        except yaml.YAMLError as e:
            logger.error(f"Error parsing configuration file: {e}")
            sys.exit(1)

    # --- Paths and Settings from Config ---
    onnx_file = PROJECT_ROOT / export_config['onnx_output_path']
    # Keep engine output path relative to this script's dir for simplicity
    engine_file = Path(__file__).resolve().parent / 'best.trt' 
    
    input_h = export_config['input_height']
    input_w = export_config['input_width']
    
    # Define optimization profile shapes (Batch, Channel, Height, Width)
    # Using fixed batch size of 1 based on previous hardcoded value.
    # Make batch size configurable if needed.
    min_input_shape = (1, 3, input_h, input_w)
    opt_input_shape = (1, 3, input_h, input_w)
    max_input_shape = (1, 3, input_h, input_w)
    
    # TensorRT Logger Severity
    trt_verbosity = trt.Logger.WARNING # Or INFO, VERBOSE for more details

    # --- Execution ---
    trt_logger = trt.Logger(trt_verbosity)
    
    # Ensure engine output directory exists
    engine_file.parent.mkdir(parents=True, exist_ok=True)
    
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
