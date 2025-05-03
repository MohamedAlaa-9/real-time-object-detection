# Real-Time Object Detection for Autonomous Vehicles

This project focuses on building and optimizing a machine learning pipeline for real-time object detection, specifically tailored for autonomous vehicle scenarios using the KITTI dataset and YOLO models. The pipeline includes data preprocessing, model training (YOLOv11), export to ONNX, optimization with TensorRT, and real-time inference capabilities delivered through a modern web application.

**Recent Improvements (May 2025):**

* Refactored the application to use a modern web architecture (FastAPI backend + Svelte frontend)
* Enhanced data preprocessing for robustness and reproducibility
* Optimized TensorRT inference script for performance (memory allocation) and clarity
* Improved ONNX export process with dynamic axes support
* Refined post-processing logic for correct bounding box scaling
* Updated dependency management and documentation

## Project Structure

```plaintext
real-time-object-detection/
├── backend/                 # FastAPI backend server
│   ├── api/                 # API endpoints
│   │   ├── video.py         # Video processing endpoints
│   │   └── websocket.py     # WebSocket for real-time communication
│   ├── core/                # Core backend configuration
│   │   └── config.py        # Backend configuration settings
│   ├── schemas/             # Pydantic data models
│   │   └── video.py         # Video data schemas
│   ├── services/            # Service layer
│   │   ├── mock_inference.py # Mock inference for testing
│   │   └── video_processor.py # Video processing service
│   ├── utils/               # Utility functions
│   └── main.py              # Main FastAPI application entry point
├── frontend/                # Svelte frontend application
│   ├── src/                 # Frontend source code
│   │   ├── components/      # Reusable UI components
│   │   ├── routes/          # Application routes/pages
│   │   └── main.js          # Frontend entry point
│   ├── index.html           # Root HTML template
│   ├── package.json         # Frontend dependencies
│   ├── svelte.config.js     # Svelte configuration
│   ├── tsconfig.json        # TypeScript configuration
│   └── vite.config.js       # Vite bundler configuration
├── ml_models/               # Model training and optimization
│   ├── train_yolo.py        # YOLOv11 training script (with MLflow)
│   ├── export_model.py      # Export trained model to ONNX format
│   ├── optimize_tensorrt.py # Optimize ONNX model using TensorRT
│   ├── inference.py         # Real-time inference using TensorRT engine
│   ├── yolo11n.pt           # Base pre-trained model
│   └── yolo11x.pt           # Larger pre-trained model variant
├── datasets/                # Dataset handling scripts and configuration
│   ├── preprocess_datasets.py # Downloads and preprocesses KITTI dataset
│   ├── data.yaml            # YOLO dataset configuration file
│   ├── README.md            # Dataset specific instructions
│   ├── raw/                 # Raw downloaded data (e.g., KITTI zip files)
│   └── processed/           # Preprocessed data in YOLO format (train/val splits)
├── infra/                   # Infrastructure and deployment scripts
│   ├── azure-deploy.yaml    # Azure VM deployment config via GitHub Actions
│   ├── monitoring_setup.sh  # Monitoring setup (Prometheus/Grafana)
│   └── prometheus.yml       # Prometheus configuration
├── docs/                    # Project documentation
│   ├── architecture.md      # System architecture overview
│   └── ml_pipeline.md       # Detailed ML pipeline guide
├── config/                  # Configuration files
│   └── train_config.yaml    # Training configuration
├── uploads/                 # Temporary storage for uploaded videos
├── results/                 # Processed video results
├── runs/                    # Training run artifacts
├── mlruns/                  # MLflow experiment tracking
├── utils.py                 # Utility functions
├── requirements.txt         # Python package dependencies
├── Dockerfile               # Docker configuration for containerization
├── Tutorial.md              # End-to-end tutorial guide
└── README.md                # This file
```

## Datasets

* **KITTI:** The primary dataset used. The `datasets/preprocess_datasets.py` script handles downloading and converting KITTI data (images and labels) into the YOLO format required for training.

### Using the KITTI Dataset

The preprocessing script (`datasets/preprocess_datasets.py`) will automatically download the required KITTI files if they are not found in the `datasets/raw/` directory:

* Left color images (`kitti_image.zip`)
* Training labels (`kitti_label.zip`)
* Camera calibration data (`kitti_calib.zip`)

Simply run `python datasets/preprocess_datasets.py` to initiate the download and preprocessing.

## Key Focus Areas

1. Real-Time Detection: Ensuring that the system can accurately detect and classify objects in real time, necessary for autonomous vehicle operation.
2. Transfer Learning: Leveraging pre-trained models for fast adaptation and better performance.
3. Environmental Adaptation: Developing a robust system capable of performing well across different driving environments.
4. Continuous Monitoring: Using MLOps tools to track model performance, detect drifts, and retrain the system.
5. Safety and Reliability: Ensuring the system operates reliably with robust object detection crucial for passenger and pedestrian safety.

## Prerequisites

Before running the project, ensure you have the following installed:

1. **Python:** Version 3.8 or higher recommended.
2. **NVIDIA GPU:** Required for accelerated training and TensorRT inference.
3. **NVIDIA Drivers:** Install the appropriate drivers for your GPU.
4. **CUDA Toolkit:** Install a version compatible with your drivers and the required libraries.
5. **cuDNN:** Install the cuDNN library compatible with your CUDA version.
6. **TensorRT:** Download and install NVIDIA TensorRT. Ensure the Python bindings are installed correctly.
7. **Node.js and npm:** Required for building and running the frontend.
8. **Docker:** Required for containerization and deployment.

## Project Running Instructions

### Local Development

Follow these steps to set up and run the ML pipeline locally:

1. **Install Python Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

2. **Install Frontend Dependencies and Build:**

    ```bash
    cd frontend
    npm install
    npm run build
    cd ..
    ```

3. **Download and Preprocess Dataset:**

    ```bash
    python datasets/preprocess_datasets.py
    ```

4. **Train the Model:**

    ```bash
    python ml_models/train_yolo.py
    ```

5. **Export Trained Model to ONNX:**

    ```bash
    python ml_models/export_model.py
    ```

6. **Optimize ONNX Model with TensorRT:**

    ```bash
    python ml_models/optimize_tensorrt.py
    ```

7. **Start the Backend Server:**

    ```bash
    python backend/main.py
    ```

8. **Start the Frontend Development Server (for development):**

    ```bash
    cd frontend
    npm run dev
    ```

9. **Access the Web Application:**
   Open your browser and navigate to `http://localhost:5173` (or whichever port the frontend is running on)

### Docker Deployment

For containerized deployment:

```bash
# Build the Docker image
docker build -t object-detection:latest .

# Run the container
docker run -p 8000:8000 --gpus all object-detection:latest
```

## Azure Deployment

The project includes infrastructure files for deployment to Azure:

1. **Set up GitHub Action Secrets:**
   Configure the following secrets in your GitHub repository:
   - `AZURE_REGISTRY`: URL of your Azure Container Registry
   - `AZURE_USERNAME`: Username for ACR
   - `AZURE_PASSWORD`: Password for ACR
   - `AZURE_RESOURCE_GROUP`: Azure resource group name
   - `AZURE_VM_NAME`: Name of the Azure VM with GPU support

2. **Deploy using GitHub Actions:**
   Push to the main branch to trigger the GitHub Actions workflow defined in `infra/azure-deploy.yaml`.

3. **Configure Monitoring:**
   After deployment, SSH into your Azure VM and run:
   ```bash
   bash infra/monitoring_setup.sh
   ```

4. **Access the Application:**
   Navigate to the public IP address of your Azure VM on port 8000 (default).

## Configuration

The project's behavior can be configured by modifying various files:

* `config/train_config.yaml`: Training parameters for the YOLOv11 model
* `backend/core/config.py`: Backend server configuration
* `ml_models/inference.py`: Inference engine settings
* `datasets/preprocess_datasets.py`: Dataset preprocessing options

For detailed information about the architecture and ML pipeline, refer to the documentation in the `docs/` directory.
