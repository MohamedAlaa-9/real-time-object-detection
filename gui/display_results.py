import cv2
from ml_models.inference import infer

def display_detections(frame):
    boxes, scores, classes = infer(frame)
    class_names = ["pedestrian", "vehicle"]
    colors = [(0, 255, 0), (0, 0, 255)]
    
    for box, score, cls in zip(boxes, scores, classes):
        x_min, y_min, x_max, y_max = box
        label = f"{class_names[cls]}: {score:.2f}"
        color = colors[cls]
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
        cv2.putText(frame, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    
    return frame
