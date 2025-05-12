# ML Pipeline

## Overview

The ML pipeline for the real-time object detection system consists of the following stages:

1. **Data Ingestion and Preparation:** Acquiring and preparing the dataset for training. This involves downloading raw data, converting formats, and splitting into training/validation sets.
    * Scripts: `datasets/download_kitti.py`, `datasets/preprocess_kitti.py`, `datasets/convert_kitti_to_yolo.py`, `datasets/data_ingestion.py`, `datasets/preprocess_datasets.py`.
    * Configuration: `datasets/data.yaml`.
    * Output: Processed data in `datasets/processed/`.
2. **Model Training:** Training the YOLO object detection model.
    * Script: `ml_models/train_yolo.py`.
    * Configuration: `config/train_config.yaml`.
    * Tracking: MLflow (`ml_models/mlflow_utils.py`), artifacts in `mlruns/` and `runs/train/`.
    * Base Model: e.g., `ml_models/models/yolo11n.pt`.
3. **Model Export:** Converting the trained PyTorch model to ONNX format for broader compatibility and optimized inference.
    * Script: `ml_models/export_model.py`.
    * Output: ONNX model (e.g., `ml_models/models/yolo11n.onnx`).
4. **Model Optimization:** Optimizing the ONNX model using TensorRT for high-performance inference on NVIDIA GPUs.
    * Script: `ml_models/optimize_tensorrt.py` (and potentially `optimize_with_tensorrt.sh`).
    * Output: TensorRT engine file.
    * Documentation: `TENSORRT.md`.
5. **Model Inference:** Using the optimized model for real-time object detection.
    * Script: `ml_models/inference.py`.
    * Integrated into: `backend/services/video_processor.py`.
6. **Model Verification & Preparation:** Ensuring the pipeline and models work as expected.
    * Scripts: `ml_models/verify_model_pipeline.py`, `ml_models/prepare_models.py`.
7. **Deployment:** Packaging and deploying the application (backend, frontend, and ML models).
    * Containerization: `Dockerfile`.
    * Cloud Deployment: `infra/azure-deploy.yaml` (Azure), `infra/azure-deployment-guide.md`.
8. **Monitoring and MLOps:** Continuously monitoring model performance and managing the ML lifecycle.
    * Scripts: `ml_models/model_monitoring.py`, `ml_models/auto_retrain.py`, `ml_models/mlops_utils.py`.
    * Configuration: `config/mlops_config.yaml`.
    * Infrastructure: Prometheus (`infra/prometheus.yml`), Grafana (`infra/grafana-dashboards/model-performance.json`), setup scripts (`infra/monitoring_setup.sh`, `infra/monitoring_init.sh`).
    * Status: `ml_models/model_status.txt`.

## Stages in Detail

### 1. Data Ingestion and Preparation

* **Download Data:** The `datasets/download_kitti.py` script (or similar for other datasets) fetches raw data (images, labels, calibration files) and stores it in `datasets/raw/`.
* **Preprocessing:** `datasets/preprocess_kitti.py` handles KITTI-specific tasks like unzipping, organizing files. `datasets/preprocess_datasets.py` might be a higher-level script.
* **Format Conversion:** `datasets/convert_kitti_to_yolo.py` converts the labels and directory structure to the format required by YOLO.
* **Dataset Configuration:** `datasets/data.yaml` (and `datasets/processed/data.yaml`) defines paths to training/validation data and class names for YOLO.
* **Exploratory Data Analysis (EDA):** `KITTI_EDA.ipynb` can be used to understand the dataset characteristics.

### 2. Model Training

* **Configuration:** `config/train_config.yaml` specifies parameters like learning rate, batch size, number of epochs, model architecture variant, etc.
* **Training Script:** `ml_models/train_yolo.py` loads the preprocessed data based on `datasets/data.yaml`, initializes the YOLO model (potentially from a base like `yolo11n.pt`), and starts the training process.
* **MLflow Integration:** `ml_models/mlflow_utils.py` is used to log parameters, metrics, and model artifacts to MLflow. Experiment data is stored in `mlruns/`. Output models and training artifacts might also be saved in `runs/train/`.

### 3. Model Export

* **Script:** `ml_models/export_model.py` takes a trained PyTorch model (`.pt` file) and converts it into ONNX format (`.onnx`). This often involves specifying input/output names and dynamic axes for flexibility.

### 4. Model Optimization (TensorRT)

* **Script:** `ml_models/optimize_tensorrt.py` (or `optimize_with_tensorrt.sh`) takes the `.onnx` model and uses NVIDIA TensorRT to build an optimized inference engine. This engine is tailored to the specific GPU hardware, potentially with specified precision (FP16, INT8).
* **Dependencies:** `install_tensorrt_dependencies.sh` helps set up the TensorRT environment.

### 5. Model Inference

* **Inference Script:** `ml_models/inference.py` contains the logic to load the TensorRT engine and perform predictions on input data (e.g., video frames).
* **Backend Integration:** The `backend/services/video_processor.py` uses this inference script to process uploaded videos.

### 6. Model Verification & Preparation

* `ml_models/verify_model_pipeline.py`: This script likely runs a series of checks to ensure that each step of the pipeline (data loading, preprocessing, inference) works correctly with the current models and configurations.
* `ml_models/prepare_models.py`: This could involve downloading pre-trained base models, or ensuring models are in the correct location or format before training or inference.

### 7. Deployment

* **Containerization:** The `Dockerfile` defines how to build a Docker image containing the backend, frontend (built static assets), and necessary ML model files and dependencies.
* **Azure Deployment:** `infra/azure-deploy.yaml` defines a GitHub Actions workflow to build the Docker image, push it to Azure Container Registry, and deploy it to an Azure service (e.g., Azure Kubernetes Service or Azure Container Instances). `infra/azure-deployment-guide.md` provides manual steps or further details.

### 8. Monitoring and MLOps

* **Performance Monitoring:** `ml_models/model_monitoring.py` likely includes logic to track inference speed, accuracy drift, or other relevant metrics of the deployed model. These metrics can be exposed for Prometheus to scrape.
* **Automated Retraining:** `ml_models/auto_retrain.py` could be triggered based on monitoring alerts (e.g., performance degradation) or a schedule to retrain the model with new data.
* **MLOps Utilities:** `ml_models/mlops_utils.py` provides helper functions for various MLOps tasks. `config/mlops_config.yaml` stores configurations for these processes.
* **Visualization:** Prometheus collects metrics, and Grafana (`infra/grafana-dashboards/model-performance.json`) visualizes them, providing insights into the system's health and model performance.
* **Notebooks:** `main_pipeline.ipynb` might serve as an overarching notebook to run or test parts of this pipeline interactively.
