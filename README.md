# Real-Time Object Detection for Autonomous Vehicles

This project focuses on building a machine learning model that can detect and classify objects in the environment, such as pedestrians, vehicles, traffic signs, and obstacles. The model will be deployed in autonomous vehicle systems to enhance safety and decision making in real-time driving scenarios.

## Project Structure

```
real-time-object-detection/
├── ml-models/                # Model training and optimization
│   ├── train_yolo.py         # YOLOv11 training script
│   ├── export_model.py       # Export model to ONNX format
│   ├── optimize_tensorrt.py  # Optimize ONNX model for TensorRT
│   └── inference.py          # Real-time inference
├── gui/                      # GUI for real-time testing
│   ├── app.py                # Main GUI script
│   ├── video_stream.py       # Video capture
│   └── display_results.py    # Detection visualization
├── datasets/                 # Training datasets (placeholders)
│   ├── raw/                  # Raw KITTI data (not implemented)
│   └── processed/            # Preprocessed data (not implemented)
├── infra/                    # Deployment infrastructure
│   ├── azure-deploy.yaml     # Azure VM setup
│   └── monitoring_setup.sh   # Grafana/Prometheus config
├── docs/                     # Documentation
│   ├── architecture.md       # System overview
│   └── ml_pipeline.md        # Training/deployment guide
└── README.md                 # Project overview
```

## Datasets

*   KITTI

To add new datasets, you need to:

1.  Download the dataset.
2.  Create a new function in `datasets/preprocess_datasets.py` to preprocess the dataset.
3.  Add the new function to the `if __name__ == "__main__":` block in `datasets/preprocess_datasets.py`.
4.  Update the `create_data_yaml` function in `datasets/preprocess_datasets.py` to include the new dataset.
5.  Update the `ml-models/train_yolo.py` file to include the new dataset (if necessary).
6.  Update the `gui/display_results.py` file to include the new dataset.

To use the KITTI dataset, you need to download the following files:

*   Left color images of object data set (12 GB): https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip
*   Training labels of object data set (5 MB): https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip
*   Camera calibration matrices of object data set (16 MB): https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_calib.zip

## Key Focus Areas

1.  Real-Time Detection: Ensuring that the system can accurately detect and classify objects in real time, necessary for autonomous vehicle operation.
2.  Transfer Learning: Leveraging pre-trained models (e.g., COCO) for fast adaptation and better performance.
3.  Environmental Adaptation: Developing a robust system capable of performing well across different driving environments (urban, highways, night, and adverse weather).
4.  Continuous Monitoring: Using MLOps tools to track model performance, detect drifts, and retrain the system as new data becomes available.
5.  Safety and Reliability: Ensuring the system operates in a fail-safe manner with robust object detection to ensure the safety of passengers and pedestrians.

## Project Running Instructions

1.  Install the required dependencies: `pip install -r requirements.txt`
2.  Download and preprocess the KITTI dataset: `python datasets/preprocess_datasets.py`
3.  Train the YOLOv11 model: `python ml-models/train_yolo.py`
4.  Export the model to ONNX format: `python ml-models/export_model.py`
5.  Optimize the ONNX model for TensorRT: `python ml-models/optimize_tensorrt.py`
6.  Run the GUI: `python gui/app.py`

## Running the Project: A Step-by-Step Guide

The following instructions provide a detailed guide on how to run the project:

1.  **Install the required dependencies:** `pip install -r requirements.txt`

    *   This command uses `pip`, the Python package installer, to install all the libraries and dependencies listed in the `requirements.txt` file. This file contains a list of all the necessary Python packages required to run the project, such as `ultralytics`, `torch`, `opencv-python`, etc.
    *   **Step-by-step explanation:**
        *   Open a terminal or command prompt.
        *   Navigate to the project's root directory (where the `requirements.txt` file is located) using the `cd` command.
        *   Run the command `pip install -r requirements.txt`.
        *   Wait for the installation to complete. `pip` will download and install all the required packages.
2.  **Download and preprocess the KITTI dataset:** `python datasets/preprocess_datasets.py`

    *   This command executes the `preprocess_datasets.py` script located in the `datasets` directory. This script downloads the KITTI dataset (if it's not already downloaded) and preprocesses it into a format suitable for training the YOLO model.
    *   **Step-by-step explanation:**
        *   Ensure that you have enough disk space to download and store the KITTI dataset (approximately 12 GB for the images and additional space for the labels and calibration files).
        *   Open a terminal or command prompt.
        *   Navigate to the project's root directory.
        *   Run the command `python datasets/preprocess_datasets.py`.
        *   Wait for the script to complete. It will download the dataset (if necessary), extract the relevant files, and preprocess them.
3.  **Train the YOLOv11 model:** `python ml-models/train_yolo.py`

    *   This command executes the `train_yolo.py` script located in the `ml-models` directory. This script trains the YOLOv11 model using the preprocessed KITTI dataset.
    *   **Step-by-step explanation:**
        *   Ensure that you have a compatible GPU and the necessary drivers installed if you want to train the model on a GPU. Otherwise, the training will be performed on the CPU, which can be significantly slower.
        *   Open a terminal or command prompt.
        *   Navigate to the project's root directory.
        *   Run the command `python ml-models/train_yolo.py`.
        *   Wait for the training to complete. The training time can vary depending on the hardware and the training parameters.
4.  **Export the model to ONNX format:** `python ml-models/export_model.py`

    *   This command executes the `export_model.py` script located in the `ml-models` directory. This script exports the trained YOLOv11 model to the ONNX (Open Neural Network Exchange) format, which is a standard format for representing machine learning models.
    *   **Step-by-step explanation:**
        *   Open a terminal or command prompt.
        *   Navigate to the project's root directory.
        *   Run the command `python ml-models/export_model.py`.
        *   Wait for the script to complete.
5.  **Optimize the ONNX model for TensorRT:** `python ml-models/optimize_tensorrt.py`

    *   This command executes the `optimize_tensorrt.py` script located in the `ml-models` directory. This script optimizes the ONNX model for TensorRT, which is a high-performance inference engine developed by NVIDIA.
    *   **Step-by-step explanation:**
        *   Ensure that you have TensorRT installed and configured correctly.
        *   Open a terminal or command prompt.
        *   Navigate to the project's root directory.
        *   Run the command `python ml-models/optimize_tensorrt.py`.
        *   Wait for the script to complete.
6.  **Run the GUI:** `python gui/app.py`

    *   This command executes the `app.py` script located in the `gui` directory. This script starts the GUI, which allows you to test the real-time object detection model using a video stream.
    *   **Step-by-step explanation:**
        *   Open a terminal or command prompt.
        *   Navigate to the project's root directory.
        *   Run the command `python gui/app.py`.
        *   The GUI should appear, and you can start testing the model.

## Deployment

The `README.md` file mentions the `infra` directory, which contains deployment infrastructure configurations:

*   `azure-deploy.yaml`: This file is likely used to set up an Azure Virtual Machine (VM) for deploying the project. It might contain configurations for the VM size, operating system, and other settings.
*   `monitoring_setup.sh`: This script is likely used to set up monitoring for the deployed project using Grafana and Prometheus. It might contain commands to install and configure these tools, as well as to define metrics to track.

To deploy the project, you would typically follow these steps:

1.  Set up an Azure VM using the `azure-deploy.yaml` file. This might involve using the Azure CLI or the Azure portal.
2.  Configure monitoring for the VM using the `monitoring_setup.sh` script. This might involve installing Grafana and Prometheus on the VM and configuring them to collect and display metrics.
3.  Copy the project files to the VM.
4.  Install the required dependencies on the VM using `pip install -r requirements.txt`.
5.  Download and preprocess the KITTI dataset on the VM using `python datasets/preprocess_datasets.py`.
6.  Train the YOLOv11 model on the VM using `python ml-models/train_yolo.py`.
7.  Export the model to ONNX format on the VM using `python ml-models/export_model.py`.
8.  Optimize the ONNX model for TensorRT on the VM using `python ml-models/optimize_tensorrt.py`.
9.  Run the GUI on the VM using `python gui/app.py`.

## Configurations

The project's behavior can be configured by modifying various files:

*   `ml-models/train_yolo.py`: This file contains the training parameters for the YOLOv11 model, such as the number of epochs, the batch size, and the image size. You can modify these parameters to improve the model's performance or to adapt it to different datasets.
*   `gui/app.py`: This file contains the configuration for the GUI, such as the video source and the display settings. You can modify these settings to customize the GUI to your needs.
*   `datasets/preprocess_datasets.py`: This file contains the code for downloading and preprocessing the datasets. You can modify this file to add support for new datasets or to change the preprocessing steps.
*   `infra/azure-deploy.yaml`: This file contains the configuration for the Azure VM. You can modify this file to change the VM size, operating system, or other settings.
*   `infra/monitoring_setup.sh`: This file contains the configuration for the monitoring system. You can modify this file to add new metrics to track or to change the way the metrics are displayed.

By modifying these files, you can customize the project to your specific needs and requirements.
