import cv2
import yaml
from ml_models.inference import infer
from pathlib import Path

def display_detections(frame):
    boxes, scores, classes = infer(frame)
    
    # Load class names from data.yaml
    with open(Path("../datasets/processed/data.yaml"), "r") as f:
        data = yaml.safe_load(f)
        class_names = data["names"]
    
    # Define colors for each class (you can customize these)
    colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 255, 0), (0, 255, 255), (255, 0, 255), (192, 192, 192), (128, 128, 128), (128, 0, 0), (128, 128, 0), (0, 128, 0)]
    
    for box, score, cls in zip(boxes, scores, classes):
        x_min, y_min, x_max, y_max = box
        label = f"{class_names[cls]}: {score:.2f}"
        color = colors[cls % len(colors)]  # Ensure color is within range
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
        cv2.putText(frame, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    
    return frame
