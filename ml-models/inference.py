import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from prometheus_client import Counter, Histogram, start_http_server
from utils import post_process_yolo  # Helper function defined below

# Metrics
INFERENCE_COUNT = Counter("inference_total", "Total number of inferences")
INFERENCE_TIME = Histogram("inference_duration_seconds", "Inference duration")

# Start metrics server
start_http_server(8000)

# Load TensorRT engine
with open('best.trt', 'rb') as f:
    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    engine = runtime.deserialize_cuda_engine(f.read())

context = engine.create_execution_context()

# Inference function
def infer(frame):
    with INFERENCE_TIME.time():
        inputs, outputs, bindings, stream = [], [], [], cuda.Stream()
        for binding in engine:
            size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings.append(int(device_mem))
            if engine.binding_is_input(binding):
                inputs.append({'host': host_mem, 'device': device_mem})
            else:
                outputs.append({'host': host_mem, 'device': device_mem})

        # Preprocess frame
        input_frame = cv2.resize(frame, (640, 640))
        input_frame = input_frame.transpose((2, 0, 1)).astype(np.float32) / 255.0
        np.copyto(inputs[0]['host'], input_frame.ravel())

        # Run inference
        [cuda.memcpy_htod_async(inp['device'], inp['host'], stream) for inp in inputs]
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        [cuda.memcpy_dtoh_async(out['host'], out['device'], stream) for out in outputs]
        stream.synchronize()

        # Post-process
        detections = outputs[0]['host'].reshape(-1, 85)  # Adjust based on YOLOv11 output
        boxes, scores, classes = post_process_yolo(detections, 640, 640)
        INFERENCE_COUNT.inc()
        return boxes, scores, classes

# Helper function for post-processing
def post_process_yolo(detections, img_w, img_h, conf_thres=0.5, iou_thres=0.6):
    boxes, scores, classes = [], [], []
    for det in detections:
        conf = det[4] * max(det[5:])  # Objectness * max class score
        if conf > conf_thres:
            x_center, y_center, w, h = det[0:4]
            x_min = int((x_center - w / 2) * img_w)
            y_min = int((y_center - h / 2) * img_h)
            x_max = int((x_center + w / 2) * img_w)
            y_max = int((y_center + h / 2) * img_h)
            boxes.append([x_min, y_min, x_max, y_max])
            scores.append(conf)
            classes.append(np.argmax(det[5:]))
    
    # Apply NMS
    indices = cv2.dnn.NMSBoxes(boxes, scores, conf_thres, iou_thres)
    return [boxes[i] for i in indices], [scores[i] for i in indices], [classes[i] for i in indices]
