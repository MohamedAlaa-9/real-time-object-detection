# ML Pipeline

## Overview

The ML pipeline for the real-time object detection system consists of the following stages:

1. **Data Collection:** The pipeline collects data from the KITTI, nuScenes, and Open Images datasets.
2. **Data Preprocessing:** The pipeline preprocesses the data to prepare it for training the object detection model.
3. **Model Training:** The pipeline trains the object detection model using the preprocessed data.
4. **Model Evaluation:** The pipeline evaluates the performance of the trained model.
5. **Model Deployment:** The pipeline deploys the trained model to Azure.
6. **Model Monitoring:** The pipeline monitors the performance of the deployed model.

## Stages

### Data Collection

The data collection stage consists of the following steps:

1. Download the KITTI dataset.
2. Download the nuScenes dataset.
3. Download the Open Images dataset.

### Data Preprocessing

The data preprocessing stage consists of the following steps:

1. Resize the images to a consistent size.
2. Normalize the pixel values.
3. Perform data augmentation.

### Model Training

The model training stage consists of the following steps:

1. Load the preprocessed data.
2. Train the YOLOv11 model.
3. Evaluate the performance of the trained model.

### Model Evaluation

The model evaluation stage consists of the following steps:

1. Load the trained model.
2. Evaluate the performance of the model on a test dataset.
3. Calculate the mAP, IoU, and FPS metrics.

### Model Deployment

The model deployment stage consists of the following steps:

1. Create a Docker image of the application.
2. Push the Docker image to Azure Container Registry.
3. Deploy the Docker image to an Azure Virtual Machine.

### Model Monitoring

The model monitoring stage consists of the following steps:

1. Collect metrics from the application using Prometheus.
2. Visualize the metrics using Grafana.
3. Monitor the performance of the deployed model.
