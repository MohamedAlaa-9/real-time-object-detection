import os
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

# KITTI Dataset URLs
KITTI_IMAGE_URL = "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip"
KITTI_LABEL_URL = "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip"
KITTI_CALIB_URL = "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_calib.zip"

# Paths
BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

def download_kitti():
    kitti_image_path = RAW_DIR / "kitti_image.zip"
    kitti_label_path = RAW_DIR / "kitti_label.zip"
    kitti_calib_path = RAW_DIR / "kitti_calib.zip"

    if not kitti_image_path.exists():
        print("Downloading KITTI image data...")
        try:
            urlretrieve(KITTI_IMAGE_URL, kitti_image_path)
        except Exception as e:
            print(f"Failed to download KITTI image data: {e}")
            return
        with zipfile.ZipFile(kitti_image_path, 'r') as zip_ref:
            zip_ref.extractall(RAW_DIR / 'kitti')

    if not kitti_label_path.exists():
        print("Downloading KITTI label data...")
        try:
            urlretrieve(KITTI_LABEL_URL, kitti_label_path)
        except Exception as e:
            print(f"Failed to download KITTI label data: {e}")
            return
        with zipfile.ZipFile(kitti_label_path, 'r') as zip_ref:
            zip_ref.extractall(RAW_DIR / 'kitti')

    if not kitti_calib_path.exists():
        print("Downloading KITTI calibration data...")
        try:
            urlretrieve(KITTI_CALIB_URL, kitti_calib_path)
        except Exception as e:
            print(f"Failed to download KITTI calibration data: {e}")
            return
        with zipfile.ZipFile(kitti_calib_path, 'r') as zip_ref:
            zip_ref.extractall(RAW_DIR / 'kitti')

if __name__ == "__main__":
    download_kitti()
    print("KITTI dataset downloaded to", RAW_DIR)
