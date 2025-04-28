from datasets.download_kitti import download_kitti
from datasets.preprocess_kitti import preprocess_kitti

if __name__ == "__main__":
    download_kitti()
    preprocess_kitti()
    print("KITTI dataset downloaded and preprocessed successfully!")
