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
        yolo_labels: A list of YOLO labels.
    """
    img = cv2.imread(str(img_file))
    if img is None:
        print(f"Warning: Could not read image {img_file}. Skipping.")
        return []

    # Check if the image is grayscale
    if len(img.shape) == 2 or img.shape[2] == 1:
        print(f"Warning: Image {img_file} is grayscale. Converting to color.")
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    h, w = img.shape[:2]
    label_file = kitti_labels / f"{img_file.stem}.txt"
    yolo_labels = set()
    with open(label_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            cls, x, y, width, height = parts[0], float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
            if cls in ["Pedestrian", "Car", "Cyclist"]:  # Map to YOLO classes
                cls_id = {"Pedestrian": 0, "Car": 1, "Cyclist": 2}[cls]
                x_center, y_center = x / w, y / h
                w_norm, h_norm = width / w, height / h
                yolo_labels.add(f"{cls_id} {x_center} {y_center} {w_norm} {h_norm}")

    return img, list(yolo_labels)

if __name__ == "__main__":
    # Example usage (replace with your actual paths)
    img_file = Path("datasets/raw/kitti/training/image_2/000000.png")
    kitti_labels = Path("datasets/raw/kitti/training/label_2")
    PROCESSED_DIR = Path("datasets/processed")
    img, yolo_labels = convert_kitti_to_yolo(img_file, kitti_labels, PROCESSED_DIR)
    print(f"YOLO labels: {yolo_labels}")
