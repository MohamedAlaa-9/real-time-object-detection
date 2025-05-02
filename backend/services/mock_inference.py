import cv2
import numpy as np
import time
import random
from backend.core.config import logger, CLASS_NAMES

def infer(frame: np.ndarray):
    """
    Mock inference function that returns synthetic detection results.
    This is used when the real TensorRT-based inference is not available.
    
    Args:
        frame: The input image frame (NumPy array BGR).
        
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Bounding boxes, scores, and class indices.
    """
    logger.info("Using mock inference function (TensorRT not available)")
    
    # Get image dimensions
    height, width = frame.shape[:2]
    
    # Simulate processing time
    time.sleep(0.05)
    
    # Generate a random number of detections (0 to 5)
    num_detections = random.randint(0, 5)
    
    if num_detections == 0:
        return np.array([]), np.array([]), np.array([])
    
    # Generate random boxes, scores and classes
    boxes = []
    scores = []
    classes = []
    
    for _ in range(num_detections):
        # Random box dimensions (ensure they fit in the frame)
        box_width = random.randint(50, width // 3)
        box_height = random.randint(50, height // 3)
        
        # Random box position
        x_min = random.randint(0, width - box_width)
        y_min = random.randint(0, height - box_height)
        x_max = x_min + box_width
        y_max = y_min + box_height
        
        # Random score (confidence)
        score = random.uniform(0.5, 0.95)
        
        # Random class
        class_id = random.randint(0, len(CLASS_NAMES) - 1)
        
        boxes.append([x_min, y_min, x_max, y_max])
        scores.append(score)
        classes.append(class_id)
    
    return np.array(boxes), np.array(scores), np.array(classes)