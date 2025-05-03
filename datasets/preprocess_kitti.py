import os
import yaml
import numpy as np
import cv2  # Added missing import
from pathlib import Path
import random  # Import random for seed setting if needed, numpy is already imported
from datasets.convert_kitti_to_yolo import convert_kitti_to_yolo

# Function to calculate Intersection over Union (IoU)
def calculate_iou(box1, box2):
    """
    Calculates the Intersection over Union (IoU) between two bounding boxes.
    Args:
        box1: (x_center, y_center, width, height) - YOLO format
        box2: (x_center, y_center, width, height) - YOLO format
    Returns:
        iou: Intersection over Union
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # Calculate coordinates of the top-left and bottom-right corners of the boxes
    x1_min = x1 - w1 / 2
    y1_min = y1 - h1 / 2
    x1_max = x1 + w1 / 2
    y1_max = y1 + h1 / 2

    x2_min = x2 - w2 / 2
    y2_min = y2 - h2 / 2
    x2_max = x2 + w2 / 2
    y2_max = y2 + h2 / 2

    # Calculate the area of intersection
    x_intersect_min = max(x1_min, x2_min)
    y_intersect_min = max(y1_min, y2_min)
    x_intersect_max = min(x1_max, x2_max)
    y_intersect_max = min(y1_max, y2_max)

    intersect_width = max(0, x_intersect_max - x_intersect_min)
    intersect_height = max(0, y_intersect_max - y_intersect_min)
    intersect_area = intersect_width * intersect_height

    # Calculate the area of each box
    box1_area = w1 * h1
    box2_area = w2 * h2

    # Calculate the union area
    union_area = box1_area + box2_area - intersect_area

    # Calculate the IoU
    iou = intersect_area / union_area if union_area > 0 else 0
    return iou

# Paths
BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "raw"
PROCESSED_DIR = BASE_DIR / "processed"
for dir in [RAW_DIR, PROCESSED_DIR]:
    dir.mkdir(parents=True, exist_ok=True)

def preprocess_kitti():
    # Convert KITTI to YOLO format - updated paths to match your folder structure
    kitti_images = RAW_DIR / "kitti/image"
    kitti_labels = RAW_DIR / "kitti/lable"  # Note: using "lable" to match your folder name

    # Set a seed for reproducible train/val splits
    np.random.seed(42)
    print("Using random seed 42 for train/val split.")

    image_files = list(kitti_images.glob("*.png"))
    print(f"Found {len(image_files)} images to process.")

    for img_file in image_files:
        img, yolo_labels = convert_kitti_to_yolo(img_file, kitti_labels, PROCESSED_DIR)
        if img is None:  # Check for None instead of truthiness
            continue
        
        # Check if we got any labels
        if not yolo_labels:
            print(f"No valid labels found for {img_file.name}, skipping.")
            continue

        # Remove overlapping bounding boxes
        iou_threshold = 0.5
        
        # Convert yolo_labels to a list of lists
        boxes = []
        for label in yolo_labels:
            cls_id, x_center, y_center, w_norm, h_norm = map(float, label.split())
            boxes.append([cls_id, x_center, y_center, w_norm, h_norm])

        # Remove overlapping boxes
        filtered_boxes = []
        while boxes:
            box1 = boxes.pop(0)
            
            # Filter out boxes that overlap significantly with box1
            boxes = [box2 for box2 in boxes if calculate_iou(box1[1:], box2[1:]) <= iou_threshold]
            filtered_boxes.append(box1)

        # Convert the filtered boxes back to YOLO format
        yolo_labels = []
        for box in filtered_boxes:
            cls_id, x_center, y_center, w_norm, h_norm = box
            yolo_labels.append(f"{int(cls_id)} {x_center} {y_center} {w_norm} {h_norm}")
        
        # Save processed data
        split = "train" if np.random.rand() < 0.8 else "val"
        (PROCESSED_DIR / split / "images").mkdir(parents=True, exist_ok=True)
        (PROCESSED_DIR / split / "labels").mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(PROCESSED_DIR / split / "images" / img_file.name), img)
        with open(PROCESSED_DIR / split / "labels" / f"{img_file.stem}.txt", "w") as f:
            f.write("\n".join(yolo_labels))

# Data config for YOLO
def create_data_yaml():
    # Use absolute path for robustness
    absolute_processed_path = str(PROCESSED_DIR.resolve())
    print(f"Using absolute path in data.yaml: {absolute_processed_path}")

    data_yaml = {
        "path": absolute_processed_path, # Use absolute path
        "train": "train/images",
        "val": "val/images",
        "names": ["pedestrian", "car", "cyclist", "van", "truck", "person_sitting", "tram", "misc"]
    }
    with open(PROCESSED_DIR / "data.yaml", "w") as f:
        yaml.dump(data_yaml, f)

if __name__ == "__main__":
    preprocess_kitti()
    create_data_yaml()
    print("Datasets preprocessed and saved in", PROCESSED_DIR)
