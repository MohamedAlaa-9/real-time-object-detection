import cv2
import numpy as np
import tensorrt as trt
from datasets.data_ingestion import ingest_video
import pycuda.driver as cuda
import pycuda.autoinit  # Note: autoinit initializes CUDA context, might be better to manage explicitly
from prometheus_client import Counter, Histogram, start_http_server
from utils import post_process_yolo  # Assuming this handles NMS correctly
from pathlib import Path
import logging

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
    start_http_server(8000)
    logger.info("Prometheus metrics server started on port 8000.")
except OSError as e:
    logger.warning(f"Could not start Prometheus server on port 8000: {e}. Metrics may not be available.")

# Define paths relative to the script
BASE_DIR = Path(__file__).resolve().parent
ENGINE_PATH = BASE_DIR / 'best.trt'  # Make path relative

# Validate engine path
if not ENGINE_PATH.exists():
    logger.error(f"TensorRT engine file not found at {ENGINE_PATH}")
    logger.error("Please run export_model.py and optimize_tensorrt.py first.")
    exit()

# Load TensorRT engine
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
try:
    with open(ENGINE_PATH, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    logger.info(f"TensorRT engine loaded successfully from {ENGINE_PATH}")
except Exception as e:
    logger.error(f"Failed to load TensorRT engine: {e}")
    exit()

# Create execution context
try:
    context = engine.create_execution_context()
    if not context:
        raise RuntimeError("Failed to create TensorRT execution context.")
    logger.info("TensorRT execution context created.")
except Exception as e:
    logger.error(f"Failed to create TensorRT execution context: {e}")
    exit()

# Allocate buffers for input and output (do this once)
inputs, outputs, bindings, stream = [], [], [], cuda.Stream()
input_shape = None
output_shape = None  # Shape needs adjustment based on model
num_classes = 3  # KITTI classes: pedestrian, car, cyclist
output_components = num_classes + 5  # x, y, w, h, conf + num_classes

logger.info("Allocating CUDA memory buffers...")
try:
    for binding in engine:
        binding_idx = engine.get_binding_index(binding)
        shape = engine.get_binding_shape(binding)
        # Handle dynamic shapes if necessary, assuming fixed shape here
        if -1 in shape:
            # Example: Set a max batch size if dynamic
            # shape = (engine.max_batch_size, *shape[1:]) # Adjust as needed
            logger.warning(f"Binding {binding} has dynamic shape {shape}. Using context shape.")
            shape = context.get_binding_shape(binding_idx)  # Get shape from context for dynamic

        size = trt.volume(shape) * engine.max_batch_size  # Use max_batch_size or 1 if batching not used
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append({'host': host_mem, 'device': device_mem})
            input_shape = shape  # Store input shape
            logger.info(f"Allocated input buffer '{{binding}}': shape={{shape}}, size={{size}}, dtype={{dtype}}")
        else:
            outputs.append({'host': host_mem, 'device': device_mem})
            output_shape = shape  # Store output shape
            logger.info(f"Allocated output buffer '{{binding}}': shape={{shape}}, size={{size}}, dtype={{dtype}}")

    if not inputs or not outputs:
        raise RuntimeError("Failed to allocate input/output buffers.")
    logger.info("CUDA memory buffers allocated successfully.")

    # Assuming first input and first output for simplicity
    input_h, input_w = input_shape[-2:]  # Assuming HWC or CHW format, get H, W
    # Example output shape: (batch_size, num_boxes, num_classes + 5) -> e.g., (1, 8400, 8) for 3 classes
    # Need to confirm the exact output shape from the model export/optimization step
    # Let's assume the output shape is (1, num_predictions, output_components)
    # We will reshape later based on actual output
    logger.info(f"Model expected input shape (H, W): ({{input_h}}, {{input_w}})")
    logger.info(f"Model raw output shape: {{output_shape}}")


except Exception as e:
    logger.error(f"Error during buffer allocation: {e}")
    exit()


def infer(frame: np.ndarray):
    """
    Performs inference on a single frame using the initialized TensorRT engine.

    Args:
        frame: The input image frame (NumPy array BGR).

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Bounding boxes, scores, and class indices.
                                                  Returns empty arrays if no detections.
    """
    boxes, scores, classes = np.array([]), np.array([]), np.array([]) # Default empty return

    if frame is None:
        logger.warning("Received None frame, skipping inference.")
        return boxes, scores, classes

    # --- Preprocessing ---
    try:
        with PREPROCESS_TIME.time():
            # Resize and normalize (ensure this matches training)
            input_image = cv2.resize(frame, (input_w, input_h))
            # Transpose HWC -> CHW and normalize (0-1)
            input_image = input_image.transpose((2, 0, 1)).astype(np.float32) / 255.0
            # Add batch dimension (if model expects batch) -> NCHW
            input_image = np.expand_dims(input_image, axis=0)
            # Flatten and copy to host buffer
            np.copyto(inputs[0]['host'], input_image.ravel())
    except Exception as e:
        logger.error(f"Error during preprocessing: {e}")
        return boxes, scores, classes # Return empty on error

    # --- Inference ---
    try:
        with INFERENCE_TIME.time():
            # Transfer input data to the GPU.
            cuda.memcpy_htod_async(inputs[0]['device'], inputs[0]['host'], stream)
            # Run inference.
            context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
            # Transfer predictions back from the GPU.
            cuda.memcpy_dtoh_async(outputs[0]['host'], outputs[0]['device'], stream)
            # Synchronize the stream
            stream.synchronize()
    except Exception as e:
        logger.error(f"Error during TensorRT execution: {e}")
        return boxes, scores, classes # Return empty on error

    # --- Post-processing ---
    try:
        with POSTPROCESS_TIME.time():
            # Reshape the raw output based on the known output shape
            # Example: output shape might be (1, num_predictions, num_classes + 5)
            # Adjust the reshape based on the actual model output structure
            raw_output = outputs[0]['host'].reshape(engine.max_batch_size, -1, output_components)  # Adjust shape as needed

            # Assuming post_process_yolo handles NMS and returns filtered results
            # Pass the first batch's output, original frame dimensions for scaling
            original_h, original_w = frame.shape[:2]
            processed_boxes, processed_scores, processed_classes = post_process_yolo(raw_output[0], input_w, input_h, original_w, original_h) # Pass original dims

            # Ensure results are not None before assigning
            if processed_boxes is not None and processed_scores is not None and processed_classes is not None:
                boxes, scores, classes = processed_boxes, processed_scores, processed_classes
                logger.info(f"Inference successful. Found {len(boxes)} objects.")
            else:
                 logger.info("Inference successful. No objects detected.")
                 # Keep boxes, scores, classes as empty arrays initialized earlier

        INFERENCE_COUNT.inc()

    except Exception as e:
        logger.error(f"Error during post-processing: {e}")
        # Return empty arrays on error

    return boxes, scores, classes

# --- Cleanup (Optional but good practice) ---
# Consider adding a cleanup function to release CUDA resources if the script runs long
# or is part of a larger application.
# def cleanup():
#     logger.info("Releasing CUDA resources...")
#     # Free device memory (example for one input/output)
#     if inputs:
#         cuda.mem_free(inputs[0]['device'])
#     if outputs:
#         cuda.mem_free(outputs[0]['device'])
#     # Destroy stream, context, engine if needed
#     # stream.detach() # Or similar cleanup
#     # context.destroy() # Careful with context lifecycle
#     # engine.destroy() # Careful with engine lifecycle
# import atexit
# atexit.register(cleanup)

logger.info("Inference engine initialization complete. Ready for inference.")

# Example Usage (Optional - for testing the modified function)
# if __name__ == "__main__":
#     # Create a dummy frame
#     dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8) # Example size
#     cv2.imwrite("dummy_frame_input.png", dummy_frame) # Save for inspection
#     logger.info("Running test inference on dummy frame...")
#     test_boxes, test_scores, test_classes = infer(dummy_frame)
#     if test_boxes.size > 0: # Check if the array is not empty
#         logger.info(f"Test inference successful. Found {len(test_boxes)} objects.")
#         # print("Boxes:", test_boxes)
#         # print("Scores:", test_scores)
#         # print("Classes:", test_classes)
#     elif test_boxes is not None: # Check if it's an empty array (no detections)
#         logger.info("Test inference successful. No objects detected.")
#     else: # Should not happen with current logic, but good to check
#         logger.error("Test inference failed or returned None unexpectedly.")
