# End-to-End Tutorial: Real-Time Object Detection

This tutorial guides you through the complete process of setting up the project, preparing the KITTI dataset using the provided scripts, training a YOLO model, optimizing it with TensorRT, and running real-time inference with the GUI.

## 1. Prerequisites

Before starting, ensure your system meets the following requirements:

* **Python:** Version 3.8+
* **NVIDIA GPU:** Required for training and TensorRT inference.
* **NVIDIA Drivers:** Latest compatible drivers.
* **CUDA Toolkit:** Compatible version for PyTorch and TensorRT.
* **cuDNN:** Compatible version for your CUDA Toolkit.
* **TensorRT:** Download and install from NVIDIA. Ensure Python bindings (`python3-libnvinfer`) are installed.
* **PyCUDA:** Install via `pip install pycuda` (may require `CUDA_ROOT` environment variable).

**Note:** TensorRT and PyCUDA installation can be complex. Refer to official NVIDIA documentation. These are **not** installed via `requirements.txt`.

## 2. Project Setup

1. **Clone the Repository:**

    ```bash
    git clone <repository_url>
    cd real-time-object-detection
    ```

2. **Install Python Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

    (Remember to install prerequisites separately).

## 3. Dataset Preparation (KITTI)

This project uses specific scripts to handle the KITTI dataset.

1. **Download KITTI Data:**
    Use the `download_kitti.py` script to fetch the required KITTI image and label files.

    ```bash
    python datasets/download_kitti.py
    ```

    Data will be downloaded to `datasets/raw/kitti`.

2. **Convert KITTI to YOLO Format:**
    Run the `convert_kitti_to_yolo.py` script to process the raw KITTI data and convert it into the YOLO format suitable for training. This script creates `train`, `val`, and `test` splits along with the necessary label files (`.txt`) in `datasets/processed/kitti_yolo`. It also generates the `datasets/data.yaml` configuration file.

    ```bash
    python datasets/convert_kitti_to_yolo.py
    ```

    *(Ensure paths within the script are correct if you modified the default locations).*

3. **Verify YOLO Dataset (Optional but Recommended):**
    Use the Jupyter Notebook `verify_kitti_yolo_dataset.ipynb` to visually inspect the converted dataset and ensure bounding boxes align correctly with objects in the images.

    ```bash
    # Install Jupyter if needed: pip install jupyterlab
    jupyter lab datasets/verify_kitti_yolo_dataset.ipynb
    ```

    Run the cells within the notebook.

4. **Test Dataset Loading:**
    Run `test_kitti.py` to confirm that the dataset configuration (`data.yaml`) is correct and the processed data can be loaded.

    ```bash
    python datasets/test_kitti.py
    ```

## 4. Model Training

Train the YOLO model using the prepared KITTI dataset.

```bash
python ml_models/train_yolo.py
```

* This script uses the configuration from `datasets/data.yaml`.
* Training artifacts (checkpoints, logs) are saved under `runs/train/`.
* MLflow is used for tracking if configured.
* The best performing model checkpoint is saved as `best.pt` (e.g., `runs/train/yolov11_kitti_exp/weights/best.pt`).

## 5. Model Export to ONNX

Export the best trained PyTorch model (`best.pt`) to the ONNX format for broader compatibility and optimization.

```bash
python ml_models/export_model.py
```

* This script looks for `best.pt` in the latest training run directory.
* The exported model is saved as `ml_models/best.onnx`.
* Dynamic axes are enabled for flexible input sizes during inference.

## 6. Optimize Model with TensorRT

Optimize the ONNX model for high-performance inference on NVIDIA GPUs using TensorRT.

```bash
python ml_models/optimize_tensorrt.py
```

* This script takes `ml_models/best.onnx` as input.
* It builds a TensorRT engine, typically using FP16 precision if supported.
* The optimized engine is saved as `ml_models/best.trt`.
* Requires TensorRT to be correctly installed (see Prerequisites).

## 7. Run Real-Time Inference GUI

Launch the graphical user interface to perform real-time object detection using the optimized TensorRT engine.

```bash
python gui/app.py
```

* The GUI uses `ml_models/inference.py` which loads the `ml_models/best.trt` engine.
* It can process video files or live camera streams (configure in `gui/app.py` if needed).
* Detected objects with bounding boxes will be visualized on the video frames.

## 8. Deployment and Monitoring (Overview)

The `infra/` directory contains example configurations for deployment and monitoring:

* **Deployment:** `infra/azure-deploy.yaml` provides an example for deploying the application to an Azure VM, likely using Docker.
* **Monitoring:** `infra/monitoring_setup.sh` and `infra/prometheus.yml` provide examples for setting up monitoring with Prometheus and Grafana.

Refer to the `README.md` and the files within `infra/` for more details on potential deployment strategies.

---

This tutorial covers the main workflow from data preparation to inference. Consult individual scripts and documentation files for more specific configuration options and details.
