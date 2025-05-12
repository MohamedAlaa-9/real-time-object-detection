# Architecture

## Overview

The real-time object detection system consists of the following main components:

1. **Data Acquisition and Preprocessing:** Scripts in the `datasets/` folder handle downloading (e.g., `download_kitti.py`) and preparing data (e.g., `preprocess_kitti.py`, `convert_kitti_to_yolo.py`) from sources like the KITTI dataset. The main script `preprocess_datasets.py` orchestrates this.
2. **Object Detection Model:** The core model is based on YOLO (e.g., `yolo11n.pt`). Training is handled by `ml_models/train_yolo.py`, utilizing configurations from `config/train_config.yaml`.
3. **Model Export and Optimization:** After training, the model is exported to ONNX format using `ml_models/export_model.py`. It's then optimized for NVIDIA GPUs using TensorRT via `ml_models/optimize_tensorrt.py` and potentially `optimize_with_tensorrt.sh`.
4. **Inference Engine:** Real-time inference is performed by `ml_models/inference.py` using the optimized TensorRT engine.
5. **Backend Server:** A FastAPI application (`backend/main.py`) serves the model and handles API requests.
    * `backend/api/video.py`: Endpoints for video processing.
    * `backend/api/websocket.py`: WebSocket for real-time communication.
    * `backend/services/video_processor.py`: Handles the video processing logic, likely integrating with the inference engine.
    * `backend/core/config.py`: Backend specific configurations.
    * `backend/schemas/video.py`: Pydantic models for data validation.
6. **Frontend Application:** A Svelte application (`frontend/`) provides the user interface.
    * `frontend/src/main.js`: Entry point for the Svelte app.
    * The frontend is built using Vite (`frontend/vite.config.js`).
7. **Deployment Infrastructure:**
    * `Dockerfile`: For containerizing the application.
    * `infra/azure-deploy.yaml`: GitHub Actions workflow for Azure deployment.
    * `infra/azure-deployment-guide.md`: Manual for Azure deployment.
8. **Monitoring Infrastructure:**
    * `infra/monitoring_setup.sh` and `infra/monitoring_init.sh`: Scripts for setting up Prometheus and Grafana.
    * `infra/prometheus.yml`: Prometheus configuration.
    * `infra/grafana-dashboards/model-performance.json`: Grafana dashboard for model performance.
9. **MLOps and Utilities:**
    * `ml_models/mlflow_utils.py` and `ml_models/mlops_utils.py`: Utilities for MLOps practices, potentially including experiment tracking with MLflow (evidenced by `mlruns/` directory).
    * `ml_models/auto_retrain.py`: Script for automated model retraining.
    * `ml_models/model_monitoring.py`: Script for monitoring model performance in production.

## Data Flow

1. Raw data (e.g., KITTI dataset) is downloaded and stored in `datasets/raw/`.
2. Preprocessing scripts in `datasets/` (e.g., `preprocess_kitti.py`, `convert_kitti_to_yolo.py`) transform the raw data into a YOLO-compatible format, typically stored in `datasets/processed/`. The `datasets/data.yaml` file configures these datasets for YOLO.
3. The YOLO model (e.g., `ml_models/train_yolo.py`) is trained using the processed data, with configurations from `config/train_config.yaml`. Training artifacts and MLflow runs are stored in `runs/` and `mlruns/` respectively.
4. The trained PyTorch model (e.g., `yolo11n.pt`) is exported to ONNX format (`yolo11n.onnx`) using `ml_models/export_model.py`.
5. The ONNX model is optimized with TensorRT using `ml_models/optimize_tensorrt.py` to create a TensorRT engine.
6. The FastAPI backend (`backend/main.py`) loads the optimized TensorRT model via `ml_models/inference.py`.
7. The Svelte frontend (`frontend/`) allows users to upload videos (stored temporarily in `uploads/`).
8. The backend processes the video using `backend/services/video_processor.py`, which calls the inference engine.
9. Object detection results are sent back to the frontend, potentially via WebSockets (`backend/api/websocket.py`) for real-time updates. Processed videos/results might be stored in `results/`.
10. Performance metrics are collected (e.g., by `ml_models/model_monitoring.py`) and can be visualized using Grafana, fed by Prometheus.

## Key Scripts and Files

* `run_system.sh`: Likely a master script to start various components of the system.
* `main_pipeline.ipynb`: A Jupyter notebook that might orchestrate or demonstrate the main ML pipeline steps.
* `KITTI_EDA.ipynb`: Exploratory Data Analysis for the KITTI dataset.
* `TENSORRT.md`: Contains specific information and notes about TensorRT usage.
* `requirements.txt`: Lists Python dependencies.
* `config/mlops_config.yaml`: Configuration for MLOps related tasks.
* `ml_models/model_status.txt`: May contain information about the currently active or best model.
* `ml_models/verify_model_pipeline.py`: Script to test and verify the integrity of the model pipeline.
* `ml_models/prepare_models.py`: Script to prepare models, possibly downloading them or converting formats.
