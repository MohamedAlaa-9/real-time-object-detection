import os
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

# KITTI Dataset URLs - Uncommented and verified URLs
KITTI_IMAGE_URL = "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip"
KITTI_LABEL_URL = "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip"
KITTI_CALIB_URL = "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_calib.zip"

# Standard naming conventions for KITTI dataset files
KITTI_IMAGE_FILENAME = "data_object_image_2.zip"
KITTI_LABEL_FILENAME = "data_object_label_2.zip"
KITTI_CALIB_FILENAME = "data_object_calib.zip"

# Paths
BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

def download_kitti():
    """Download KITTI dataset or use existing files in raw directory."""
    # Define file paths for standard KITTI filenames
    kitti_standard_image_path = RAW_DIR / KITTI_IMAGE_FILENAME
    kitti_standard_label_path = RAW_DIR / KITTI_LABEL_FILENAME
    kitti_standard_calib_path = RAW_DIR / KITTI_CALIB_FILENAME
    
    # Define file paths for custom filenames that might be used
    kitti_custom_image_path = RAW_DIR / "kitti_image.zip"
    kitti_custom_label_path = RAW_DIR / "kitti_label.zip"
    kitti_custom_calib_path = RAW_DIR / "kitti_calib.zip"
    
    # Create KITTI extraction directory
    kitti_dir = RAW_DIR / 'kitti'
    kitti_dir.mkdir(exist_ok=True, parents=True)
    
    # Image data - check both standard and custom filenames
    image_path = None
    if kitti_standard_image_path.exists():
        image_path = kitti_standard_image_path
        print(f"Using existing standard KITTI image data at {image_path}")
    elif kitti_custom_image_path.exists():
        image_path = kitti_custom_image_path
        print(f"Using existing custom KITTI image data at {image_path}")
    else:
        print("KITTI image data not found. Downloading...")
        try:
            urlretrieve(KITTI_IMAGE_URL, kitti_standard_image_path)
            image_path = kitti_standard_image_path
            print(f"Downloaded KITTI image data to {image_path}")
        except Exception as e:
            print(f"Failed to download KITTI image data: {e}")
    
    # Extract image data if found
    if image_path and image_path.exists():
        print(f"Extracting KITTI image data from {image_path}...")
        try:
            with zipfile.ZipFile(image_path, 'r') as zip_ref:
                zip_ref.extractall(kitti_dir)
            print("KITTI image data extracted successfully")
        except Exception as e:
            print(f"Failed to extract KITTI image data: {e}")
    
    # Label data - check both standard and custom filenames
    label_path = None
    if kitti_standard_label_path.exists():
        label_path = kitti_standard_label_path
        print(f"Using existing standard KITTI label data at {label_path}")
    elif kitti_custom_label_path.exists():
        label_path = kitti_custom_label_path
        print(f"Using existing custom KITTI label data at {label_path}")
    else:
        print("KITTI label data not found. Downloading...")
        try:
            urlretrieve(KITTI_LABEL_URL, kitti_standard_label_path)
            label_path = kitti_standard_label_path
            print(f"Downloaded KITTI label data to {label_path}")
        except Exception as e:
            print(f"Failed to download KITTI label data: {e}")
    
    # Extract label data if found
    if label_path and label_path.exists():
        print(f"Extracting KITTI label data from {label_path}...")
        try:
            with zipfile.ZipFile(label_path, 'r') as zip_ref:
                zip_ref.extractall(kitti_dir)
            print("KITTI label data extracted successfully")
        except Exception as e:
            print(f"Failed to extract KITTI label data: {e}")
    
    # Calibration data - check both standard and custom filenames
    calib_path = None
    if kitti_standard_calib_path.exists():
        calib_path = kitti_standard_calib_path
        print(f"Using existing standard KITTI calibration data at {calib_path}")
    elif kitti_custom_calib_path.exists():
        calib_path = kitti_custom_calib_path
        print(f"Using existing custom KITTI calibration data at {calib_path}")
    else:
        print("KITTI calibration data not found. Downloading...")
        try:
            urlretrieve(KITTI_CALIB_URL, kitti_standard_calib_path)
            calib_path = kitti_standard_calib_path
            print(f"Downloaded KITTI calibration data to {calib_path}")
        except Exception as e:
            print(f"Failed to download KITTI calibration data: {e}")
    
    # Extract calibration data if found
    if calib_path and calib_path.exists():
        print(f"Extracting KITTI calibration data from {calib_path}...")
        try:
            with zipfile.ZipFile(calib_path, 'r') as zip_ref:
                zip_ref.extractall(kitti_dir)
            print("KITTI calibration data extracted successfully")
        except Exception as e:
            print(f"Failed to extract KITTI calibration data: {e}")
    
    # Check if extraction was successful - at minimum we need images and labels
    images_dir = kitti_dir / "training" / "image_2"
    labels_dir = kitti_dir / "training" / "label_2"
    
    if not images_dir.exists() or not labels_dir.exists():
        print("WARNING: KITTI dataset extraction may not be complete.")
        print(f"Images directory exists: {images_dir.exists()}")
        print(f"Labels directory exists: {labels_dir.exists()}")
    else:
        print(f"KITTI dataset ready with {len(list(images_dir.glob('*.png')))} images")

if __name__ == "__main__":
    download_kitti()
    print("KITTI dataset preparation completed. Data path:", RAW_DIR / 'kitti')
