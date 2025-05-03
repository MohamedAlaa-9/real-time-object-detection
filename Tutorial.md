# End-to-End Tutorial: Real-Time Object Detection with Web Interface

This tutorial guides you through the complete process of setting up the project, preparing the KITTI dataset, training a YOLO model, optimizing it with TensorRT, and running the real-time inference web application.

## 1. Prerequisites

Before starting, ensure you have the following installed:

### System Requirements
- Python 3.8+ with pip
- NVIDIA GPU with CUDA support
- CUDA Toolkit and cuDNN
- Node.js and npm (for the frontend)
- Docker (for deployment)

### Installing NVIDIA Components
1. **NVIDIA Drivers**: Install the appropriate drivers for your GPU from [NVIDIA's website](https://www.nvidia.com/Download/index.aspx).
2. **CUDA Toolkit**: Download and install from [NVIDIA's CUDA website](https://developer.nvidia.com/cuda-downloads).
3. **cuDNN**: Download from [NVIDIA's cuDNN website](https://developer.nvidia.com/cudnn) (requires NVIDIA account).
4. **TensorRT**: Download from [NVIDIA's TensorRT website](https://developer.nvidia.com/tensorrt).

## 2. Project Setup

1. **Clone the Repository (if applicable)**:
   ```bash
   git clone https://github.com/your-username/real-time-object-detection.git
   cd real-time-object-detection
   ```

2. **Install Python Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set Up the Frontend**:
   ```bash
   cd frontend
   npm install
   cd ..
   ```

## 3. Dataset Preparation (KITTI)

The project uses the KITTI dataset for training the object detection model:

1. **Run the Dataset Preprocessing Script**:
   ```bash
   python datasets/preprocess_datasets.py
   ```

   This script will:
   - Download the KITTI dataset files if not present
   - Extract them to the appropriate directories
   - Convert the KITTI format to YOLO format
   - Create train/val splits
   - Generate a data.yaml configuration file

2. **Verify the Dataset**:
   After preprocessing, check that the dataset structure looks like this:
   ```
   datasets/
   ├── processed/
   │   ├── data.yaml
   │   ├── train/
   │   │   ├── images/
   │   │   └── labels/
   │   └── val/
   │       ├── images/
   │       └── labels/
   └── raw/
       ├── kitti_calib.zip
       ├── kitti_image.zip
       ├── kitti_label.zip
       └── kitti/
           └── ... (extracted files)
   ```

## 4. Model Training

1. **Review Training Configuration**:
   Before training, review and adjust the training parameters in `config/train_config.yaml` if needed.

2. **Start Training**:
   ```bash
   python ml_models/train_yolo.py
   ```

   This will:
   - Initialize a YOLOv11 model (either from scratch or using pre-trained weights)
   - Train the model on the KITTI dataset
   - Log metrics using MLflow
   - Save the best model checkpoint to `runs/train/<experiment_name>/weights/best.pt`

3. **Monitor Training Progress**:
   - Training metrics are printed to the console
   - Review training curves in the MLflow UI: `mlflow ui`

## 5. Model Export to ONNX

1. **Export the Trained Model to ONNX Format**:
   ```bash
   python ml_models/export_model.py
   ```

   This script:
   - Loads the best model checkpoint from training
   - Exports it to ONNX format with dynamic axes
   - Saves the ONNX model in the `ml_models` directory

## 6. Optimize Model with TensorRT

1. **Build a TensorRT Engine**:
   ```bash
   python ml_models/optimize_tensorrt.py
   ```

   This script:
   - Takes the ONNX model
   - Builds an optimized TensorRT engine
   - Saves the engine file in the `ml_models` directory
   
   > Note: This requires TensorRT to be properly installed on your system.

## 7. Run the Web Application

### Build the Frontend

1. **Build the Svelte Frontend**:
   ```bash
   cd frontend
   npm run build
   cd ..
   ```

   This compiles the Svelte application into static files that can be served by the backend.

### Start the Backend Server

2. **Start the FastAPI Backend**:
   ```bash
   python backend/main.py
   ```

   This will:
   - Start the FastAPI server (default: http://localhost:8000)
   - Load the optimized TensorRT model for inference
   - Serve the frontend static files
   - Provide API endpoints for video processing

3. **Access the Web Interface**:
   Open your browser and navigate to http://localhost:8000 to access the web interface.

### Using the Web Application

4. **Upload a Video**:
   - Click the "Upload Video" button
   - Select a video file (.mp4, .avi, etc.)
   - The video will be uploaded to the server

5. **Process the Video**:
   - Click "Process" to run object detection on the uploaded video
   - The processing status will be displayed

6. **View Results**:
   - Once processing is complete, the video with detected objects will be displayed
   - Download the processed video using the "Download" button

## 8. Docker Deployment

To deploy the application using Docker:

1. **Build the Docker Image**:
   ```bash
   docker build -t object-detection:latest .
   ```

2. **Run the Container**:
   ```bash
   docker run -p 8000:8000 --gpus all object-detection:latest
   ```

3. **Access the Application**:
   Open your browser and navigate to http://localhost:8000

## 9. Azure Deployment

The project includes configuration for deploying to Azure using GitHub Actions.

### Prerequisites

- Azure account with subscription
- Azure Container Registry (ACR)
- Azure VM with GPU support

### Deployment Steps

1. **Configure GitHub Secrets**:
   In your GitHub repository, add the following secrets:
   - `AZURE_REGISTRY`: Your ACR login server URL
   - `AZURE_USERNAME`: ACR username
   - `AZURE_PASSWORD`: ACR password
   - `AZURE_RESOURCE_GROUP`: Resource group name
   - `AZURE_VM_NAME`: VM name

2. **Set Up Azure Resources**:
   ```bash
   # Login to Azure
   az login

   # Create resource group (if needed)
   az group create --name myResourceGroup --location eastus

   # Create Container Registry (if needed)
   az acr create --resource-group myResourceGroup --name myacr --sku Basic

   # Create VM with GPU support
   az vm create \
     --resource-group myResourceGroup \
     --name myVM \
     --image UbuntuLTS \
     --size Standard_NC6 \
     --admin-username azureuser \
     --generate-ssh-keys
   ```

3. **Configure VM**:
   SSH into your VM and install Docker and NVIDIA drivers:
   ```bash
   # Install Docker
   curl -fsSL https://get.docker.com -o get-docker.sh
   sudo sh get-docker.sh

   # Install NVIDIA drivers and Docker GPU support
   # (Instructions vary based on VM image and GPU)
   ```

4. **Trigger Deployment**:
   Push to the `main` branch to trigger the GitHub Actions workflow:
   ```bash
   git push origin main
   ```

5. **Access the Deployed Application**:
   Once deployment is complete, access your application at the VM's public IP address on port 8000.

## 10. Monitoring Setup

After deployment, set up monitoring:

1. **SSH into your VM**:
   ```bash
   ssh azureuser@your-vm-ip
   ```

2. **Run the Monitoring Setup Script**:
   ```bash
   bash infra/monitoring_setup.sh
   ```

3. **Access Monitoring Dashboards**:
   - Prometheus: http://your-vm-ip:9090
   - Grafana: http://your-vm-ip:3000 (default login: admin/admin)

---

This tutorial covers the main workflow from data preparation to deployment. Consult the documentation files in the `docs/` directory for more specific details about the architecture and ML pipeline.
