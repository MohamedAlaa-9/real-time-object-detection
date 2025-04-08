# Real-Time Object Detection for Autonomous Vehicles

This project focuses on building and optimizing a machine learning pipeline for real-time object detection, specifically tailored for autonomous vehicle scenarios using the KITTI dataset and YOLO models. The pipeline includes data preprocessing, model training (YOLOv11), export to ONNX, optimization with TensorRT, and real-time inference capabilities.

**Recent Improvements (April 2025):**

* Refactored data preprocessing for robustness and reproducibility.
* Enhanced training script with better path handling and MLflow logging.
* Optimized TensorRT inference script for performance (memory allocation) and clarity.
* Improved ONNX export process with dynamic axes support.
* Refined post-processing logic for correct bounding box scaling.
* Updated dependency management and documentation.

## Project Structure

```plantext
real-time-object-detection/
├── ml-models/                # Model training and optimization
│   ├── train_yolo.py         # YOLOv11 training script (with MLflow)
│   ├── export_model.py       # Export trained model to ONNX format
│   ├── optimize_tensorrt.py  # Optimize ONNX model using TensorRT
│   ├── inference.py          # Real-time inference using TensorRT engine
│   ├── yolo11n.pt            # Base pre-trained model (example)
│   └── best.onnx             # Exported ONNX model (generated)
│   └── best.trt              # Optimized TensorRT engine (generated)
├── gui/                      # GUI for real-time testing and visualization
│   ├── app.py                # Main GUI application script
│   ├── video_stream.py       # Handles video input (file or camera)
│   └── display_results.py    # Visualizes detection results on frames
├── datasets/                 # Dataset handling scripts and configuration
│   ├── preprocess_datasets.py # Downloads and preprocesses KITTI dataset
│   ├── data.yaml             # YOLO dataset configuration file (generated)
│   ├── README.md             # Dataset specific instructions
│   ├── raw/                  # Raw downloaded data (e.g., KITTI zip files)
│   └── processed/            # Preprocessed data in YOLO format (train/val splits)
├── infra/                    # Infrastructure and deployment scripts
│   ├── azure-deploy.yaml     # Example Azure VM deployment config
│   ├── monitoring_setup.sh   # Example monitoring setup (Prometheus/Grafana)
│   └── prometheus.yml        # Example Prometheus configuration
├── docs/                     # Project documentation
│   ├── architecture.md       # System architecture overview
│   └── ml_pipeline.md        # Detailed ML pipeline guide
├── utils.py                  # Utility functions (e.g., post-processing)
├── requirements.txt          # Python package dependencies (pip)
├── Dockerfile                # Docker configuration for containerization
├── .gitignore                # Git ignore rules
└── README.md                 # This file
```

## Datasets

* **KITTI:** The primary dataset used. The `datasets/preprocess_datasets.py` script handles downloading and converting KITTI data (images and labels) into the YOLO format required for training. It automatically creates the `datasets/processed/data.yaml` file.

### Using the KITTI Dataset

The preprocessing script (`datasets/preprocess_datasets.py`) will automatically download the required KITTI files if they are not found in the `datasets/raw/` directory. The necessary components are:

* Left color images (`data_object_image_2.zip`)
* Training labels (`data_object_label_2.zip`)
* Camera calibration data (`data_object_calib.zip`) - *Note: Calibration data is downloaded but not currently used in the simplified YOLO conversion.*

Simply run `python datasets/preprocess_datasets.py` to initiate the download and preprocessing.

### Adding New Datasets

While the project is currently focused on KITTI, adding other datasets would involve:

1. Placing raw data in a suitable location (e.g., `datasets/raw/new_dataset_name`).
2. Modifying or extending `datasets/preprocess_datasets.py` to handle the new dataset's format and convert it to YOLO format (creating train/val splits in `datasets/processed`).
3. Ensuring the `datasets/processed/data.yaml` reflects the structure and class names of the combined or new dataset.

## Key Focus Areas

1. Real-Time Detection: Ensuring that the system can accurately detect and classify objects in real time, necessary for autonomous vehicle operation.
2. Transfer Learning: Leveraging pre-trained models (e.g., COCO) for fast adaptation and better performance.
3. Environmental Adaptation: Developing a robust system capable of performing well across different driving environments (urban, highways, night, and adverse weather).
4. Continuous Monitoring: Using MLOps tools to track model performance, detect drifts, and retrain the system as new data becomes available.
5. Safety and Reliability: Ensuring the system operates reliably with robust object detection crucial for passenger and pedestrian safety.

## Prerequisites

Before running the project, ensure you have the following installed:

1. **Python:** Version 3.8 or higher recommended.
2. **NVIDIA GPU:** Required for accelerated training and TensorRT inference.
3. **NVIDIA Drivers:** Install the appropriate drivers for your GPU.
4. **CUDA Toolkit:** Install a version compatible with your drivers and the required libraries (PyTorch, TensorRT).
5. **cuDNN:** Install the cuDNN library compatible with your CUDA version.
6. **TensorRT:** Download and install NVIDIA TensorRT. Ensure the Python bindings (`python3-libnvinfer`) are installed correctly for your environment. This is often done via Debian packages or Tar archives provided by NVIDIA, not pip.
7. **PyCUDA:** Install PyCUDA. This usually requires compilation and needs to match your CUDA toolkit version. Often installed via `pip install pycuda`, but might require environment variables (`CUDA_ROOT`) to be set.

**Note:** TensorRT and PyCUDA installation can be complex and system-dependent. Refer to the official NVIDIA documentation for detailed instructions specific to your OS and CUDA version. These packages are **not** installed by the `requirements.txt` file.

## Project Running Instructions

Follow these steps to set up and run the ML pipeline:

1. **Install Python Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

    This installs the core Python packages listed in `requirements.txt`. Remember to install the prerequisites (CUDA, TensorRT, PyCUDA) separately as described above.
2. **Download and Preprocess Dataset:**

    ```bash
    python datasets/preprocess_datasets.py
    ```

    This script downloads the KITTI dataset (if not present) and converts it into the YOLO format under `datasets/processed/`. It also creates `datasets/processed/data.yaml`.
3. **Train the Model:**

    ```bash
    python ml-models/train_yolo.py
    ```

    This trains the YOLOv11 model using the processed dataset. Training logs and checkpoints are saved under `runs/train/`. MLflow is used for experiment tracking if available. The best model checkpoint (`best.pt`) will be saved in the experiment directory (e.g., `runs/train/yolov11_kitti_exp/weights/best.pt`).
4. **Export Trained Model to ONNX:**

    ```bash
    python ml-models/export_model.py
    ```

    This script loads the `best.pt` checkpoint from the training run and exports it to `ml-models/best.onnx`. It enables dynamic axes for flexibility.
5. **Optimize ONNX Model with TensorRT:**

    ```bash
    python ml-models/optimize_tensorrt.py
    ```

    This takes the `best.onnx` model and builds an optimized TensorRT engine (`ml-models/best.trt`). This step requires TensorRT to be correctly installed (see Prerequisites). The engine is built with FP16 precision if supported by the GPU.
6. **Run Real-Time Inference GUI:**

    ```bash
    python gui/app.py
    ```

    This starts the GUI application, which uses the optimized TensorRT engine (`best.trt`) via `ml-models/inference.py` to perform real-time object detection on a video file or camera stream.

## Detailed Step-by-Step Guide (Removed)

*The previous detailed step-by-step guide has been integrated into the concise "Project Running Instructions" above.*

## Deployment

The `README.md` file mentions the `infra` directory, which contains deployment infrastructure configurations:

* `azure-deploy.yaml`: This file is likely used to set up an Azure Virtual Machine (VM) for deploying the project. It might contain configurations for the VM size, operating system, and other settings.
* `monitoring_setup.sh`: This script is likely used to set up monitoring for the deployed project using Grafana and Prometheus. It might contain commands to install and configure these tools, as well as to define metrics to track.

To deploy the project, you would typically follow these steps:

1. Set up an Azure VM using the `azure-deploy.yaml` file. This might involve using the Azure CLI or the Azure portal.
2. Configure monitoring for the VM using the `monitoring_setup.sh` script. This might involve installing Grafana and Prometheus on the VM and configuring them to collect and display metrics.
3. Copy the project files to the VM.
4. Install the required dependencies on the VM using `pip install -r requirements.txt`.
5. Download and preprocess the KITTI dataset on the VM using `python datasets/preprocess_datasets.py`.
6. Train the YOLOv11 model on the VM using `python ml-models/train_yolo.py`.
7. Export the model to ONNX format on the VM using `python ml-models/export_model.py`.
8. Optimize the ONNX model for TensorRT on the VM using `python ml-models/optimize_tensorrt.py`.
9. Run the GUI on the VM using `python gui/app.py`.

## Configurations

The project's behavior can be configured by modifying various files:

* `ml-models/train_yolo.py`: This file contains the training parameters for the YOLOv11 model, such as the number of epochs, the batch size, and the image size. You can modify these parameters to improve the model's performance or to adapt it to different datasets.
* `gui/app.py`: This file contains the configuration for the GUI, such as the video source and the display settings. You can modify these settings to customize the GUI to your needs.
* `datasets/preprocess_datasets.py`: This file contains the code for downloading and preprocessing the datasets. You can modify this file to add support for new datasets or to change the preprocessing steps.
* `infra/azure-deploy.yaml`: This file contains the configuration for the Azure VM. You can modify this file to change the VM size, operating system, or other settings.
* `infra/monitoring_setup.sh`: This file contains the configuration for the monitoring system. You can modify this file to add new metrics to track or to change the way the metrics are displayed.

By modifying these files, you can customize the project to your specific needs and requirements.
