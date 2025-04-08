from typing import List, Tuple
import cv2
import numpy as np

def post_process_yolo(
    detections: np.ndarray,
    input_w: int,
    input_h: int,
    original_w: int,
    original_h: int,
    conf_thres: float = 0.45, # Adjusted default threshold
    iou_thres: float = 0.5   # Adjusted default threshold
) -> Tuple[List[List[int]], List[float], List[int]]:
    """
    Post-processes raw YOLO detections to produce filtered bounding boxes, scores, and class IDs,
    scaled to the original image dimensions.

    Args:
        detections: Raw output from the YOLO model (e.g., shape [num_predictions, 5 + num_classes]).
                    Assumes format [x_center, y_center, width, height, confidence, class_prob1, ...].
        input_w: Width of the model's input image.
        input_h: Height of the model's input image.
        original_w: Width of the original image frame.
        original_h: Height of the original image frame.
        conf_thres: Confidence threshold for filtering detections.
        iou_thres: IoU threshold for Non-Maximum Suppression (NMS).

    Returns:
        A tuple containing:
        - boxes: List of filtered bounding boxes [[x_min, y_min, x_max, y_max], ...].
        - scores: List of confidence scores corresponding to the boxes.
        - classes: List of class IDs corresponding to the boxes.
    """
    boxes, scores, class_ids = [], [], []
    
    # Calculate scaling factors
    scale_x = original_w / input_w
    scale_y = original_h / input_h

    for det in detections:
        # Extract confidence and class probabilities
        confidence = det[4]
        class_scores = det[5:]
        
        # Calculate combined score (objectness * class probability)
        max_class_score = np.max(class_scores)
        combined_score = confidence * max_class_score
        
        if combined_score > conf_thres:
            # Get class ID with the highest score
            class_id = np.argmax(class_scores)
            
            # Extract box coordinates (normalized to input dimensions)
            cx, cy, w, h = det[0:4]
            
            # Convert to absolute coordinates relative to input dimensions
            x_min_input = (cx - w / 2)
            y_min_input = (cy - h / 2)
            x_max_input = (cx + w / 2)
            y_max_input = (cy + h / 2)

            # Scale coordinates to original image dimensions
            x_min = int(x_min_input * scale_x)
            y_min = int(y_min_input * scale_y)
            x_max = int(x_max_input * scale_x)
            y_max = int(y_max_input * scale_y)

            # Clip coordinates to image bounds
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(original_w, x_max)
            y_max = min(original_h, y_max)

            boxes.append([x_min, y_min, x_max, y_max])
            scores.append(float(combined_score)) # Use combined score
            class_ids.append(int(class_id))

    if not boxes:
        return [], [], []

    # Apply Non-Maximum Suppression (NMS)
    # Ensure boxes are integers for cv2.dnn.NMSBoxes if needed, though it often handles floats
    # NMSBoxes expects boxes in [x_min, y_min, width, height] format if using that specific function,
    # but the standard cv2.dnn.NMSBoxes takes [x_min, y_min, x_max, y_max] - let's stick to that.
    # However, the function signature requires width and height for the boxes parameter.
    # Let's convert boxes to [x, y, w, h] format for NMSBoxes.
    nms_boxes = [[b[0], b[1], b[2] - b[0], b[3] - b[1]] for b in boxes] # Convert to [x, y, w, h]

    indices = cv2.dnn.NMSBoxes(nms_boxes, scores, conf_thres, iou_thres)

    if len(indices) > 0:
        # Flatten indices if necessary (it might return a column vector)
        if isinstance(indices, np.ndarray):
            indices = indices.flatten()
        
        # Filter results based on NMS indices
        final_boxes = [boxes[i] for i in indices]
        final_scores = [scores[i] for i in indices]
        final_classes = [class_ids[i] for i in indices]
        return final_boxes, final_scores, final_classes
    else:
        # Return empty lists if NMS removed all boxes
        return [], [], []
