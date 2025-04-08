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
    import random
    colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(len(class_names))]
    
    for box, score, cls in zip(boxes, scores, classes):
        x_min, y_min, x_max, y_max = box
        label = f"{class_names[cls]}: {score:.2f}"
        color = colors[cls]  # Use color directly
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
        cv2.putText(frame, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    
    return frame
