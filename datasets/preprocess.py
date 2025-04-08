import os
import shutil
import json
import argparse
import zipfile
import requests
import tarfile
import yaml
from pathlib import Path

# Base dataset directory
DATASET_DIR = "datasets/"
PROCESSED_DIR = os.path.join(DATASET_DIR, "processed/")
RAW_DIR = os.path.join(DATASET_DIR, "raw/")

# Ensure directories exist
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(RAW_DIR, exist_ok=True)

# Dataset sources
DATASETS = {
    "nuscenes": "https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/v1.0-mini.tgz",
    "kitti": "http://www.cvlibs.net/download.php?file=data_depth_annotated.zip",
    "coco": "http://images.cocodataset.org/zips/train2017.zip",
    "bdd100k": "https://bdd-data.berkeley.edu/bdd100k_images.zip"
}

def download_dataset(name, url):
    """Downloads and extracts a dataset."""
    file_path = os.path.join(RAW_DIR, f"{name}.zip" if ".zip" in url else f"{name}.tgz")
    extract_path = os.path.join(RAW_DIR, name)
    
    if not os.path.exists(extract_path):
        print(f"Downloading {name}...")
        response = requests.get(url, stream=True)
        with open(file_path, "wb") as file:
            shutil.copyfileobj(response.raw, file)
        print(f"Extracting {name}...")
        
        if file_path.endswith(".zip"):
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(extract_path)
        elif file_path.endswith(".tgz"):
            with tarfile.open(file_path, "r:gz") as tar:
                tar.extractall(extract_path)
        os.remove(file_path)
    else:
        print(f"{name} dataset already exists.")

def convert_to_yolo_format(dataset_name):
    """Converts datasets to a unified YOLO format."""
    yolo_path = os.path.join(PROCESSED_DIR, dataset_name)
    os.makedirs(yolo_path, exist_ok=True)
    
    if dataset_name == "openimages":
        openimages_to_yolo()
    elif dataset_name == "kitti":
        kitti_to_yolo()
    elif dataset_name == "bdd100k":
        bdd100k_to_yolo()
    elif dataset_name == "nuscenes":
        nuscenes_to_yolo()
    print(f"Converted {dataset_name} to YOLO format.")

def merge_datasets():
    """Merges all datasets into a single dataset versioned by timestamp."""
    version = "merged_" + str(int(os.path.getmtime(PROCESSED_DIR)))
    merged_path = os.path.join(PROCESSED_DIR, version)
    os.makedirs(merged_path, exist_ok=True)
    
    for dataset in DATASETS.keys():
        dataset_path = os.path.join(PROCESSED_DIR, dataset)
        if os.path.exists(dataset_path):
            shutil.copytree(dataset_path, merged_path, dirs_exist_ok=True)
    print(f"Merged datasets into {merged_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", nargs="*", help="Datasets to download (e.g., nuscenes kitti coco bdd100k)")
    parser.add_argument("--convert", nargs="*", help="Datasets to convert to YOLO format")
    parser.add_argument("--merge", action="store_true", help="Merge processed datasets into a single dataset")
    args = parser.parse_args()
    
    if args.download:
        for dataset in args.download:
            if dataset in DATASETS:
                download_dataset(dataset, DATASETS[dataset])
            else:
                print(f"Dataset {dataset} not found!")
    
    if args.convert:
        for dataset in args.convert:
            convert_to_yolo_format(dataset)
    
    if args.merge:
        merge_datasets()

if __name__ == "__main__":
    main()
