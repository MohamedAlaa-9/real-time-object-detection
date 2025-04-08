# Architecture

## Overview

The real-time object detection system for autonomous vehicles consists of the following main components:

1.  **Data Acquisition and Preprocessing:** This component is responsible for acquiring data from various sources, such as the KITTI, nuScenes, and Open Images datasets. It also preprocesses the data to prepare it for training the object detection model.
2.  **Object Detection Model:** This component is responsible for detecting and classifying objects in the environment. It uses the YOLOv11 architecture, which is known for its speed and accuracy.
3.  **Inference Engine:** This component is responsible for running the object detection model in real time. It uses the TensorRT inference engine, which is optimized for NVIDIA GPUs.
4.  **Graphical User Interface (GUI):** This component provides a user interface for visualizing the object detection results. It uses the PyQt5 framework.
5.  **Deployment Infrastructure:** This component is responsible for deploying the object detection system to Azure. It uses Docker and Azure Container Registry.
6.  **Monitoring Infrastructure:** This component is responsible for monitoring the performance of the object detection system. It uses Prometheus and Grafana.

## Data Flow

The data flows through the system as follows:

1.  The data acquisition and preprocessing component acquires data from the KITTI, nuScenes, and Open Images datasets.
2.  The data preprocessing component preprocesses the data to prepare it for training the object detection model.
3.  The object detection model is trained on the preprocessed data.
4.  The trained object detection model is exported to ONNX format.
5.  The ONNX model is optimized with TensorRT.
6.  The inference engine loads the optimized TensorRT model.
7.  The inference engine receives video frames from the camera.
8.  The inference engine runs the object detection model on the video frames.
9.  The inference engine outputs the object detection results.
10. The GUI displays the object detection results.
11. The monitoring infrastructure monitors the performance of the object detection system.

## Components

### Data Acquisition and Preprocessing

The data acquisition and preprocessing component consists of the following subcomponents:

*   **Data Acquisition:** This subcomponent is responsible for acquiring data from the KITTI, nuScenes, and Open Images datasets.
*   **Data Preprocessing:** This subcomponent is responsible for preprocessing the data to prepare it for training the object detection model.

### Object Detection Model

The object detection model consists of the following subcomponents:

*   **YOLOv11 Architecture:** This subcomponent implements the YOLOv11 architecture.
*   **Training Algorithm:** This subcomponent implements the training algorithm.

### Inference Engine

The inference engine consists of the following subcomponents:

*   **TensorRT Inference Engine:** This subcomponent implements the TensorRT inference engine.
*   **Model Loader:** This subcomponent loads the optimized TensorRT model.

### Graphical User Interface (GUI)

The GUI consists of the following subcomponents:

*   **PyQt5 Framework:** This subcomponent implements the GUI using the PyQt5 framework.
*   **Video Display:** This subcomponent displays the video frames.
*   **Object Detection Display:** This subcomponent displays the object detection results.

### Deployment Infrastructure

The deployment infrastructure consists of the following subcomponents:

*   **Docker:** This subcomponent containerizes the application.
*   **Azure Container Registry:** This subcomponent stores the Docker image.
*   **Azure Virtual Machine:** This subcomponent runs the Docker container.

### Monitoring Infrastructure

The monitoring infrastructure consists of the following subcomponents:

*   **Prometheus:** This subcomponent collects metrics from the application.
*   **Grafana:** This subcomponent visualizes the metrics.
