import os
import cv2
import json
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
    kitti_url = "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip"  # Example URL
    kitti_path = RAW_DIR / "kitti.zip"
    if not kitti_path.exists():
        print("Downloading KITTI dataset...")
        urlretrieve(kitti_url, kitti_path)
        os.system(f"unzip {kitti_path} -d {RAW_DIR / 'kitti'}")
    
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

# Download and preprocess nuScenes
def preprocess_nuscenes():
    nuscenes_path = RAW_DIR / "nuscenes"
    if not nuscenes_path.exists():
        print("Downloading nuScenes mini dataset...")
        os.system(f"wget -P {RAW_DIR} https://www.nuscenes.org/data/v1.0-mini.tgz")
        os.system(f"tar -xzf {RAW_DIR / 'v1.0-mini.tgz'} -C {RAW_DIR}")
    
    from nuscenes.nuscenes import NuScenes
    nusc = NuScenes(version="v1.0-mini", dataroot=str(nuscenes_path), verbose=True)
    for sample in nusc.sample:
        cam_data = nusc.get("sample_data", sample["data"]["CAM_FRONT"])
        img_path = nuscenes_path / cam_data["filename"]
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]
        
        anns = [nusc.get("sample_annotation", ann) for ann in sample["anns"]]
        yolo_labels = []
        for ann in anns:
            if ann["category_name"] in ["human.pedestrian.adult", "vehicle.car"]:
                cls_id = 0 if "pedestrian" in ann["category_name"] else 1
                box = ann["bbox"]  # [x_min, y_min, x_max, y_max]
                x_center = (box[0] + box[2]) / 2 / w
                y_center = (box[1] + box[3]) / 2 / h
                w_norm = (box[2] - box[0]) / w
                h_norm = (box[3] - box[1]) / h
                yolo_labels.append(f"{cls_id} {x_center} {y_center} {w_norm} {h_norm}")
        
        split = "train" if np.random.rand() < 0.8 else "val"
        cv2.imwrite(str(PROCESSED_DIR / split / "images" / f"{sample['token']}.jpg"), img)
        with open(PROCESSED_DIR / split / "labels" / f"{sample['token']}.txt", "w") as f:
            f.write("\n".join(yolo_labels))

# Download and preprocess Open Images
def preprocess_open_images():
    oi_path = RAW_DIR / "open_images"
    if not oi_path.exists():
        print("Downloading Open Images subset...")
        os.system(f"wget -P {RAW_DIR} https://storage.googleapis.com/openimages/v6/oidv6-train-images-boxable.csv")
        os.system(f"wget -P {RAW_DIR} https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-bbox.csv")
        # Download a subset of images (e.g., 1000) using oid_downloader
        os.system(f"python -m oid_downloader --classes Person Car --type_csv train --limit 1000 --dest_dir {oi_path}")

    import pandas as pd
    images_df = pd.read_csv(RAW_DIR / "oidv6-train-images-boxable.csv")
    anns_df = pd.read_csv(RAW_DIR / "oidv6-train-annotations-bbox.csv")
    class_map = {"Person": 0, "Car": 1}
    
    for _, row in images_df.iterrows():
        img_id = row["ImageID"]
        img_path = oi_path / f"{img_id}.jpg"
        if not img_path.exists():
            continue
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]
        
        anns = anns_df[anns_df["ImageID"] == img_id]
        yolo_labels = []
        for _, ann in anns.iterrows():
            if ann["LabelName"] in ["/m/01g317", "/m/0k4j"]:  # Person, Car
                cls_id = class_map[{"/m/01g317": "Person", "/m/0k4j": "Car"}[ann["LabelName"]]]
                x_center = (ann["XMin"] + ann["XMax"]) / 2
                y_center = (ann["YMin"] + ann["YMax"]) / 2
                w_norm = ann["XMax"] - ann["XMin"]
                h_norm = ann["YMax"] - ann["YMin"]
                yolo_labels.append(f"{cls_id} {x_center} {y_center} {w_norm} {h_norm}")
        
        split = "train" if np.random.rand() < 0.8 else "val"
        cv2.imwrite(str(PROCESSED_DIR / split / "images" / f"{img_id}.jpg"), img)
        with open(PROCESSED_DIR / split / "labels" / f"{img_id}.txt", "w") as f:
            f.write("\n".join(yolo_labels))

# Data config for YOLO
def create_data_yaml():
    data_yaml = {
        "path": str(PROCESSED_DIR),
        "train": "train/images",
        "val": "val/images",
        "names": ["pedestrian", "vehicle", "traffic_sign", "obstacle"]
    }
    with open(PROCESSED_DIR / "data.yaml", "w") as f:
        json.dump(data_yaml, f)

if __name__ == "__main__":
    preprocess_kitti()
    preprocess_nuscenes()
    preprocess_open_images()
    create_data_yaml()
    print("Datasets preprocessed and saved in", PROCESSED_DIR)
