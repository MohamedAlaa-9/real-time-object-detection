import os
import cv2
import numpy as np
from pathlib import Path

def convert_kitti_to_yolo(img_file, kitti_labels, PROCESSED_DIR):
    """
    Converts KITTI labels to YOLO format.
    Args:
        img_file: Path to the image file.
        kitti_labels: Path to the KITTI label directory.
        PROCESSED_DIR: Path to the processed data directory.
    Returns:
        img: The image as a NumPy array.
        yolo_labels: A list of YOLO labels.
    """
    img = cv2.imread(str(img_file))
    if img is None:
        print(f"Warning: Could not read image {img_file}. Skipping.")
        return None, []

    # Check if the image is grayscale
    if len(img.shape) == 2 or img.shape[2] == 1:
        print(f"Warning: Image {img_file} is grayscale. Converting to color.")
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    h, w = img.shape[:2]
    label_file = kitti_labels / f"{img_file.stem}.txt"
    
    if not label_file.exists():
        print(f"Warning: Label file {label_file} does not exist. Skipping.")
        return img, []
    
    # Define mapping from KITTI class names to YOLO class indices based on COCO and KITTI-specific classes
    # Using same mapping as in main_pipeline.py
    class_map = {
        "Pedestrian": 0,        # Map to 'person' in COCO
        "Car": 2,               # Map to 'car' in COCO
        "Cyclist": 80,          # New class (not in COCO)
        "Van": 81,              # New class (not in COCO)
        "Truck": 7,             # Map to 'truck' in COCO
        "Person_sitting": 82,   # New class (not in COCO)
        "Tram": 83,             # New class (not in COCO)
        "Misc": 84              # New class (not in COCO)
    }
    
    yolo_labels = set()
    try:
        with open(label_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                
                # Skip if the line is too short
                if len(parts) < 8:
                    print(f"Warning: Invalid label format in {label_file}, line: {line}. Skipping.")
                    continue
                
                # Get the class name
                cls = parts[0]
                
                # Skip if class not in our mapping
                if cls not in class_map:
                    print(f"Warning: Unknown class '{cls}' in {label_file}. Skipping.")
                    continue
                
                try:
                    # Try the standard format first (indices 4,5,6,7)
                    x1, y1, x2, y2 = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
                    
                    # Convert from top-left/bottom-right to YOLO center format
                    width = x2 - x1
                    height = y2 - y1
                    x_center = x1 + width / 2
                    y_center = y1 + height / 2
                    
                except (ValueError, IndexError):
                    try:
                        # Try alternative format (indices 3,4,5,6)
                        x_center, y_center, width, height = float(parts[3]), float(parts[4]), float(parts[5]), float(parts[6])
                    except (ValueError, IndexError):
                        print(f"Warning: Could not parse coordinates in {label_file}, line: {line}. Skipping.")
                        continue
                
                # Map class name to index
                cls_id = class_map[cls]
                
                # Convert to YOLO format (normalized center coordinates)
                x_center_norm = x_center / w
                y_center_norm = y_center / h
                w_norm = width / w
                h_norm = height / h
                
                # Basic sanity check - ensure values are within [0,1]
                if 0 <= x_center_norm <= 1 and 0 <= y_center_norm <= 1 and 0 < w_norm <= 1 and 0 < h_norm <= 1:
                    yolo_labels.add(f"{cls_id} {x_center_norm} {y_center_norm} {w_norm} {h_norm}")
                else:
                    print(f"Warning: Invalid normalized coordinates in {label_file}, values outside [0,1] range. Skipping.")
    except Exception as e:
        print(f"Error processing {label_file}: {e}")
        return img, []

    return img, list(yolo_labels)

if __name__ == "__main__":
    # Example usage (replace with your actual paths)
    img_file = Path("datasets/raw/kitti/image/000000.png") 
    kitti_labels = Path("datasets/raw/kitti/lable")  # Note: using "lable" to match folder name
    PROCESSED_DIR = Path("datasets/processed")
    img, yolo_labels = convert_kitti_to_yolo(img_file, kitti_labels, PROCESSED_DIR)
    print(f"YOLO labels: {yolo_labels}")
