import os
import cv2
import yaml
import zipfile
import numpy as np
from pathlib import Path
from urllib.request import urlretrieve

# Paths
BASE_DIR = Path("../datasets")
RAW_DIR = BASE_DIR / "raw"
PROCESSED_DIR = BASE_DIR / "processed"
for dir in [RAW_DIR, PROCESSED_DIR]:
    dir.mkdir(parents=True, exist_ok=True)

# Download and preprocess KITTI
def preprocess_kitti():
    kitti_url = "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2/KITTI_object.zip"  # Actual URL
    kitti_path = RAW_DIR / "kitti.zip"
    if not kitti_path.exists():
        print("Downloading KITTI dataset...")
        urlretrieve(kitti_url, kitti_path)
        with zipfile.ZipFile(kitti_path, 'r') as zip_ref:
            zip_ref.extractall(RAW_DIR / 'kitti')
    
    # Convert KITTI to YOLO format (simplified example)
    kitti_images = RAW_DIR / "kitti/training/image_2"
    kitti_labels = RAW_DIR / "kitti/training/label_2"
    for img_file in kitti_images.glob("*.png"):
        img = cv2.imread(str(img_file))
        h, w = img.shape[:2]
        label_file = kitti_labels / f"{img_file.stem}.txt"
        yolo_labels = []
        with open(label_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                cls, x, y, width, height = parts[0], float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
                if cls in ["Pedestrian", "Car", "Cyclist"]:  # Map to YOLO classes
                    cls_id = {"Pedestrian": 0, "Car": 1, "Cyclist": 2}[cls]
                    x_center, y_center = x / w, y / h
                    w_norm, h_norm = width / w, height / h
                    yolo_labels.append(f"{cls_id} {x_center} {y_center} {w_norm} {h_norm}")
        
        # Save processed data
        split = "train" if np.random.rand() < 0.8 else "val"
        (PROCESSED_DIR / split / "images").mkdir(parents=True, exist_ok=True)
        (PROCESSED_DIR / split / "labels").mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(PROCESSED_DIR / split / "images" / img_file.name), img)
        with open(PROCESSED_DIR / split / "labels" / f"{img_file.stem}.txt", "w") as f:
            f.write("\n".join(yolo_labels))


# Data config for YOLO
def create_data_yaml():
    data_yaml = {
        "path": str(PROCESSED_DIR),
        "train": "train/images",
        "val": "val/images",
        "names": ["pedestrian", "car", "cyclist"]
    }
    with open(PROCESSED_DIR / "data.yaml", "w") as f:
        yaml.dump(data_yaml, f)

if __name__ == "__main__":
    preprocess_kitti()
    create_data_yaml()
    print("Datasets preprocessed and saved in", PROCESSED_DIR)
