# Datasets

This directory contains scripts and resources for downloading, preprocessing, and converting datasets for use in real-time object detection models.

## Contents

**Note:** The `processed/` and `raw/` directories might be missing in your repository if they have been added to the `.gitignore` file. These directories are intended for storing large dataset files, which are often excluded from version control.

* `processed/`: Directory to store processed datasets.
* `raw/`: Directory to store raw downloaded datasets.
* `download_kitti.py`: Script to download the KITTI dataset.
* `download_nuscenes.py`: Script to download the nuScenes dataset.
* `preprocess.py`: Script to preprocess the downloaded datasets.
* `convert_to_yolo.py`: Script to convert datasets to YOLO format.
