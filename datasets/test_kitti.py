import os
import sys
from pathlib import Path
import shutil

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).resolve().parent.parent))
from datasets.download_kitti import download_kitti
from datasets.preprocess_kitti import preprocess_kitti, create_data_yaml

# Paths
BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "raw"
PROCESSED_DIR = BASE_DIR / "processed"

def test_kitti():
    print("Starting KITTI dataset test...")

    # Clean up previous runs if they exist
    if RAW_DIR.exists():
        print(f"Removing existing raw directory: {RAW_DIR}")
        shutil.rmtree(RAW_DIR)
    if PROCESSED_DIR.exists():
        print(f"Removing existing processed directory: {PROCESSED_DIR}")
        shutil.rmtree(PROCESSED_DIR)

    # 1. Download the KITTI dataset
    print("\n--- Testing Download ---")
    try:
        download_kitti()
        # Check if the downloaded data exists
        if not (RAW_DIR / "kitti").exists():
            print("Error: KITTI dataset download failed or extracted incorrectly.")
            return False
        print("Download test passed.")
    except Exception as e:
        print(f"Error during download test: {e}")
        return False

    # 2. Preprocess the KITTI dataset (includes conversion and splitting)
    print("\n--- Testing Preprocessing ---")
    try:
        preprocess_kitti()
        # Check if the preprocessed data exists
        train_img_dir = PROCESSED_DIR / "train/images"
        val_img_dir = PROCESSED_DIR / "val/images"
        train_lbl_dir = PROCESSED_DIR / "train/labels"
        val_lbl_dir = PROCESSED_DIR / "val/labels"

        if not train_img_dir.exists() or not val_img_dir.exists() or \
           not train_lbl_dir.exists() or not val_lbl_dir.exists():
            print("Error: KITTI dataset preprocessing failed (missing output directories).")
            return False

        # Check if directories contain files
        if not any(train_img_dir.iterdir()) or not any(val_img_dir.iterdir()) or \
           not any(train_lbl_dir.iterdir()) or not any(val_lbl_dir.iterdir()):
             print("Error: KITTI dataset preprocessing failed (output directories are empty).")
             return False
        print("Preprocessing test passed.")
    except Exception as e:
        print(f"Error during preprocessing test: {e}")
        return False

    # 3. Create data.yaml
    print("\n--- Testing data.yaml Creation ---")
    try:
        create_data_yaml()
        data_yaml_path = PROCESSED_DIR / "data.yaml"
        if not data_yaml_path.exists():
            print("Error: data.yaml file was not created.")
            return False
        print("data.yaml creation test passed.")
    except Exception as e:
        print(f"Error during data.yaml creation test: {e}")
        return False

    print("\n-----------------------------------------")
    print("All KITTI dataset tests passed successfully!")
    print("-----------------------------------------")
    return True

if __name__ == "__main__":
    if test_kitti():
        print("Test execution finished successfully.")
    else:
        print("Test execution failed.")
