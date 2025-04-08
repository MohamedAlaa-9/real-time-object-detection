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

### Datasets

*   KITTI
*   nuScenes (mini version)

### Using nuScenes dataset

To use the nuScenes dataset, you need to:

1.  Download the nuScenes dataset (mini version).
2.  Extract the dataset to the `datasets/raw` directory.
3.  Run the `datasets/preprocess_datasets.py` script to preprocess the dataset.

To use the KITTI dataset, you need to download the following files from the KITTI dataset website:

*   Left color images of object data set (12 GB)
*   Training labels of object data set (5 MB)
*   Camera calibration matrices of object data set (16 MB)

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
