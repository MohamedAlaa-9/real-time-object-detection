import asyncio
import base64
import cv2
import numpy as np
import time
import sys
import os
from pathlib import Path

# Add the project root to the Python path to ensure imports work correctly
project_root = str(Path(__file__).resolve().parents[2])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import logger first
from backend.core.config import logger, CLASS_NAMES, COLORS, RESULTS_DIR

# Try loading real inference with improved error handling
try:
    from ml_models.inference import infer, model_source, use_pytorch, use_onnx, get_class_names
    
    # Update class names to use those from the inference module
    model_class_names = get_class_names()
    
    inference_type = "Unknown"
    if use_onnx:
        inference_type = "ONNX"
    elif use_pytorch:
        inference_type = "PyTorch"
    
    using_real_inference = True
    logger.info(f"Successfully loaded real inference engine using {model_source} model ({inference_type})")
except Exception as e:
    logger.error(f"Error importing real inference engine: {e}")
    logger.warning("Using mock inference function - objects will not be detected correctly")
    logger.warning("To fix this, run: python ml_models/prepare_models.py --all")

# Dictionary to track processing status
video_processing_status = {}


async def process_frame(frame_data: str):
    """Process a single frame through the object detection pipeline"""
    try:
        # Decode base64 image
        decoded_data = base64.b64decode(frame_data.split(',')[1])
        np_arr = np.frombuffer(decoded_data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        # Perform inference
        boxes, scores, classes = infer(frame)
        
        # Draw bounding boxes on the frame
        for box, score, cls in zip(boxes, scores, classes):
            # Ensure coordinates are integers
            x_min, y_min, x_max, y_max = map(int, box)
            class_idx = int(cls)
            # Use model class names if available, otherwise fall back to config
            if class_idx < len(model_class_names):
                label = f"{model_class_names[class_idx]}: {score:.2f}"
            else:
                label = f"Class {class_idx}: {score:.2f}"
                
            color = COLORS[class_idx % len(COLORS)]
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
            cv2.putText(frame, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Encode the processed frame to base64
        _, buffer = cv2.imencode('.jpg', frame)
        encoded_frame = base64.b64encode(buffer).decode('utf-8')
        
        return {
            "processed_frame": f"data:image/jpeg;base64,{encoded_frame}",
            "detections": [
                {"box": box.tolist() if isinstance(box, np.ndarray) else box, 
                 "score": float(score), 
                 "class_id": int(cls), 
                 "label": model_class_names[int(cls)] if int(cls) < len(model_class_names) else f"Class {int(cls)}"} 
                for box, score, cls in zip(boxes, scores, classes)
            ],
            "using_real_inference": using_real_inference,
            "model_source": model_source if using_real_inference else "mock"
        }
    except Exception as e:
        logger.error(f"Error processing frame: {e}")
        return {"error": str(e)}


async def process_video(video_id: str, video_path: Path):
    """Process a video file and save the results"""
    try:
        # Update status
        video_processing_status[video_id] = {"status": "processing", "progress": 0}
        
        # Open the video file
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            video_processing_status[video_id] = {"status": "failed", "error": "Could not open video file"}
            return
            
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Create output video writer
        output_path = RESULTS_DIR / f"{video_id}_processed.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        # Process each frame
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Process frame with inference model
            boxes, scores, classes = infer(frame)
            
            # Draw bounding boxes
            for box, score, cls in zip(boxes, scores, classes):
                # Ensure coordinates are integers
                x_min, y_min, x_max, y_max = map(int, box)
                class_idx = int(cls)
                # Use model class names if available
                if class_idx < len(model_class_names):
                    label = f"{model_class_names[class_idx]}: {score:.2f}"
                else:
                    label = f"Class {class_idx}: {score:.2f}"
                    
                color = COLORS[class_idx % len(COLORS)]
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
                cv2.putText(frame, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Write frame to output video
            out.write(frame)
            
            # Update progress
            frame_count += 1
            progress = min(99, int((frame_count / total_frames) * 100)) if total_frames > 0 else 0
            video_processing_status[video_id] = {"status": "processing", "progress": progress}
            
            # Allow other tasks to run
            await asyncio.sleep(0.001)
            
        # Release resources
        cap.release()
        out.release()
        
        # Create a thumbnail
        thumbnail_path = RESULTS_DIR / f"{video_id}_thumbnail.jpg"
        cap = cv2.VideoCapture(str(output_path))
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(str(thumbnail_path), frame)
        cap.release()
        
        # Update status to completed
        video_processing_status[video_id] = {
            "status": "completed", 
            "progress": 100,
            "output_path": str(output_path),
            "thumbnail_path": str(thumbnail_path) if thumbnail_path.exists() else None,
            "using_real_inference": using_real_inference,
            "model_source": model_source if using_real_inference else "mock"
        }
        
    except Exception as e:
        logger.error(f"Error processing video: {e}")
        video_processing_status[video_id] = {"status": "failed", "error": str(e)}