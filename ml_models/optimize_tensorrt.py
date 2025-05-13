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

# Define project root directory (assuming script is in ml_models/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

def build_engine(onnx_file_path: Path, engine_file_path: Path, trt_logger: trt.Logger, enable_dynamic: bool, min_shape: tuple, opt_shape: tuple, max_shape: tuple):
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
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30) 
    logger.info(f"Setting max workspace size to {config.get_memory_pool_limit(trt.MemoryPoolType.WORKSPACE) / (1024**2):.0f} MiB")

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
    # logger.info(f"Network input name: {input_name}, shape: {input_tensor.shape}") # Original logging
    actual_onnx_input_shape = input_tensor.shape
    logger.info(f"Network input name: {input_name}, ONNX shape: {actual_onnx_input_shape}")


    # These are the shapes derived from config in the main block and passed as arguments:
    # min_shape, opt_shape, max_shape (arguments to build_engine)

    profile_min_shape = list(min_shape)
    profile_opt_shape = list(opt_shape)
    profile_max_shape = list(max_shape)

    # Check and adjust profile shapes if ONNX dimensions are static
    for i in range(input_tensor.ndim):
        if actual_onnx_input_shape[i] > 0:  # Dimension i is static in ONNX model
            # If the profile for this dimension is not already fixed to the ONNX static size
            if not (profile_min_shape[i] == actual_onnx_input_shape[i] and
                    profile_opt_shape[i] == actual_onnx_input_shape[i] and
                    profile_max_shape[i] == actual_onnx_input_shape[i]):
                # Log a warning if an adjustment is made
                # The 'enable_dynamic' variable here is an argument to build_engine, 
                # reflecting the 'enable_dynamic_axes' from config.
                if enable_dynamic: 
                    logger.warning(
                        f"ONNX input '{input_name}' dimension {i} is static ({actual_onnx_input_shape[i]}), "
                        f"but 'enable_dynamic_axes' is True in config. Profile for dim {i} was ({profile_min_shape[i]}, {profile_opt_shape[i]}, {profile_max_shape[i]}). "
                        f"Adjusting to ({actual_onnx_input_shape[i]},{actual_onnx_input_shape[i]},{actual_onnx_input_shape[i]})."
                    )
                # Also adjust if enable_dynamic is false but profile somehow mismatches static ONNX
                elif profile_opt_shape[i] != actual_onnx_input_shape[i]: 
                     logger.warning(
                        f"ONNX input '{input_name}' dimension {i} is static ({actual_onnx_input_shape[i]}), "
                        f"and 'enable_dynamic_axes' is False. Profile for dim {i} was ({profile_min_shape[i]}, {profile_opt_shape[i]}, {profile_max_shape[i]}). "
                        f"Adjusting to ({actual_onnx_input_shape[i]},{actual_onnx_input_shape[i]},{actual_onnx_input_shape[i]})."
                    )
                
                profile_min_shape[i] = actual_onnx_input_shape[i]
                profile_opt_shape[i] = actual_onnx_input_shape[i]
                profile_max_shape[i] = actual_onnx_input_shape[i]
    
    final_min_shape = tuple(profile_min_shape)
    final_opt_shape = tuple(profile_opt_shape)
    final_max_shape = tuple(profile_max_shape)

    profile = builder.create_optimization_profile()
    logger.info(f"Setting optimization profile for input '{input_name}':")
    logger.info(f"  MIN shape: {final_min_shape}")
    logger.info(f"  OPT shape: {final_opt_shape}")
    logger.info(f"  MAX shape: {final_max_shape}")
    profile.set_shape(input_name, final_min_shape, final_opt_shape, final_max_shape)
    
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
            logger.info(f"  enable_dynamic_axes: {export_config.get('enable_dynamic_axes', False)}") # Get dynamic flag
        except yaml.YAMLError as e:
            logger.error(f"Error parsing configuration file: {e}")
            sys.exit(1)

    # --- Paths and Settings from Config ---
    onnx_file = PROJECT_ROOT / export_config['onnx_output_path']
    # Derive engine path from ONNX path
    engine_file = onnx_file.with_suffix('.trt')
    
    input_h = export_config['input_height']
    input_w = export_config['input_width']
    enable_dynamic = export_config.get('enable_dynamic_axes', False)
    
    # Define optimization profile shapes (Batch, Channel, Height, Width)
    if enable_dynamic:
        # Example: Allow batch size from 1 to 16, optimize for 1
        min_batch_size = 1
        opt_batch_size = 1
        max_batch_size = 16 # Adjust as needed
        min_input_shape = (min_batch_size, 3, input_h, input_w)
        opt_input_shape = (opt_batch_size, 3, input_h, input_w)
        max_input_shape = (max_batch_size, 3, input_h, input_w)
    else:
        # Fixed batch size of 1
        min_input_shape = (1, 3, input_h, input_w)
        opt_input_shape = (1, 3, input_h, input_w)
        max_input_shape = (1, 3, input_h, input_w)
    
    # TensorRT Logger Severity
    trt_verbosity = trt.Logger.WARNING # Or INFO, VERBOSE for more details

    # --- Execution ---
    trt_logger = trt.Logger(trt_verbosity)
    
    # Ensure engine output directory exists
    engine_file.parent.mkdir(parents=True, exist_ok=True)
    
    engine = build_engine(onnx_file, engine_file, trt_logger, enable_dynamic, min_input_shape, opt_input_shape, max_input_shape)

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
