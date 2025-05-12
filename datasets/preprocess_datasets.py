from datasets.download_kitti import download_kitti
from datasets.preprocess_kitti import preprocess_kitti, create_data_yaml
import os
from pathlib import Path

if __name__ == "__main__":
    # First, make sure data directories exist
    BASE_DIR = Path(__file__).resolve().parent
    PROCESSED_DIR = BASE_DIR / "processed"
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    # Uncommented to ensure dataset is downloaded and processed
    download_result = download_kitti()
    preprocess_result = preprocess_kitti()
    
    # Always create data.yaml file, even if preprocessing was skipped
    # This ensures we always have a proper data.yaml for training
    if not (PROCESSED_DIR / "data.yaml").exists():
        print("Creating data.yaml file...")
        create_data_yaml()
    
    print("KITTI dataset preparation completed!")
    print(f"Data YAML file available at: {PROCESSED_DIR / 'data.yaml'} and {BASE_DIR / 'data.yaml'}")
