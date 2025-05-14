# Azure Deployment Guide for Real-Time Object Detection

This guide provides step-by-step instructions for deploying the real-time object detection application on Azure, including setting up a GPU virtual machine, container registry, and monitoring infrastructure.

## Prerequisites

- Azure account with an active subscription
- Azure CLI installed locally
- Docker installed locally
- Git repository with your application code

## 1. Setting Up Azure Resources

### 1.1. Create a Resource Group

```bash
# Create a resource group
az group create --name real-time-object-detection --location eastus
```

### 1.2. Create an Azure Container Registry (ACR)

```bash
# Create a container registry
az acr create --resource-group real-time-object-detection --name objectdetectionacr --sku Standard --admin-enabled true

# Get the ACR credentials
az acr credential show --name objectdetectionacr
```

Save the username and password for later use.

### 1.3. Create a GPU Virtual Machine

```bash
# Create a GPU VM (NC-series with NVIDIA GPU)
az vm create \
  --resource-group real-time-object-detection \
  --name kazzaz \
  --image Canonical:UbuntuServer:18.04-LTS:latest \
  --admin-username azureuser \
  --generate-ssh-keys \
  --size Standard_NC6s_v3 \
  --public-ip-sku Standard

# Open ports for the application
az vm open-port --resource-group real-time-object-detection --name kazzaz --port 8080 --priority 1001
az vm open-port --resource-group real-time-object-detection --name kazzaz --port 3000 --priority 1002
az vm open-port --resource-group real-time-object-detection --name kazzaz --port 5000 --priority 1003
az vm open-port --resource-group real-time-object-detection --name kazzaz --port 9090 --priority 1004
```

## 2. Setting Up the Virtual Machine

### 2.1. Connect to the VM

```bash
# Get the public IP address
az vm show -d -g real-time-object-detection -n kazzaz --query publicIps -o tsv

# SSH into the VM (IP: 40.76.126.51)
ssh azureuser@40.76.126.51
```

### 2.2. Install NVIDIA Drivers and Docker

```bash
# Update package list
sudo apt-get update

# Install required packages
sudo apt-get install -y linux-headers-$(uname -r) build-essential

# Install NVIDIA drivers
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin
sudo mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-ubuntu1804-11-8-local_11.8.0-520.61.05-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1804-11-8-local_11.8.0-520.61.05-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu1804-11-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda-11-8

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Add user to the docker group
sudo usermod -aG docker $USER

# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker

# Test NVIDIA Docker installation
sudo docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu20.04 nvidia-smi
```

Log out and log back in to apply the docker group changes.

### 2.3. Install Docker Compose

```bash
# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/download/v2.17.2/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
```

## 3. Building and Deploying the Application

### 3.1. Clone the Repository

```bash
# Clone your repository
git clone https://github.com/yourusername/real-time-object-detection.git
cd real-time-object-detection
```

### 3.2. Log in to Azure Container Registry

```bash
# Log in to ACR
docker login objectdetectionacr.azurecr.io -u <username> -p <password>
```

### 3.3. Build and Push the Docker Image

```bash
# Build the image
docker build -t objectdetectionacr.azurecr.io/object-detection:latest .

# Push the image to ACR
docker push objectdetectionacr.azurecr.io/object-detection:latest
```

### 3.4. Create a Docker Compose File

Create a `docker-compose.yml` file with the following content:

```yaml
version: '3'
services:
  # Main application
  app:
    image: objectdetectionacr.azurecr.io/object-detection:latest
    ports:
      - "8080:8080"
    volumes:
      - ./data:/app/data
      - ./ml_models:/app/ml_models
    deploy:
      resources:
        reservations:
          devices:
            - driver: 
              count: 1
              capabilities: []
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - PROMETHEUS_MULTIPROC_DIR=/tmp

  # Prometheus
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./infra/prometheus.yml:/etc/prometheus/prometheus.yml
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/usr/share/prometheus/console_libraries'
      - '--web.console.templates=/usr/share/prometheus/consoles'

  # Grafana
  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    volumes:
      - ./infra/grafana-dashboards:/etc/grafana/provisioning/dashboards
      - ./infra/grafana-datasources:/etc/grafana/provisioning/datasources
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
      - GF_USERS_ALLOW_SIGN_UP=false
    depends_on:
      - prometheus

  # MLflow
  mlflow:
    image: ghcr.io/mlflow/mlflow:latest
    ports:
      - "5000:5000"
    volumes:
      - ./mlruns:/mlruns
    command: mlflow server --host 0.0.0.0 --serve-artifacts
    environment:
      - MLFLOW_TRACKING_URI=http://localhost:5000
      - MLFLOW_DEFAULT_ARTIFACT_ROOT=/mlruns
```

### 3.5. Create Grafana Configuration Directories

```bash
# Create Grafana configuration directories
mkdir -p infra/grafana-datasources
mkdir -p infra/grafana-dashboards
```

Create `infra/grafana-datasources/prometheus.yml`:

```yaml
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
```

Create `infra/grafana-dashboards/dashboard.yml`:

```yaml
apiVersion: 1

providers:
  - name: 'default'
    folder: ''
    type: file
    options:
      path: /etc/grafana/provisioning/dashboards
```

Copy your Grafana dashboard JSON to `infra/grafana-dashboards/model-performance.json`.

### 3.6. Start the Application

```bash
# Start the application with Docker Compose
docker-compose up -d
```

## 4. Verifying the Deployment

### 4.1. Check Container Status

```bash
docker-compose ps
```

### 4.2. Access the Application

- Backend API: http://40.76.126.51:8080
- Frontend: http://40.76.126.51:8080 (served by the backend)
- Grafana: http://40.76.126.51:3000 (login: admin/admin)
- Prometheus: http://40.76.126.51:9090
- MLflow: http://40.76.126.51:5000

## 5. Setting Up CI/CD with GitHub Actions

### 5.1. Create Azure Service Principal

```bash
# Create a service principal with contributor role
az ad sp create-for-rbac --name "object-detection-ci-cd" --role contributor --scopes /subscriptions/your-subscription-id/resourceGroups/real-time-object-detection --sdk-auth
```

Save the output JSON for use in GitHub secrets.

### 5.2. Configure GitHub Secrets

In your GitHub repository, add these secrets:

- `AZURE_CREDENTIALS`: The entire JSON output from the service principal creation
- `AZURE_REGISTRY`: The ACR login server (e.g., objectdetectionacr.azurecr.io)
- `AZURE_USERNAME`: The ACR username
- `AZURE_PASSWORD`: The ACR password
- `AZURE_RESOURCE_GROUP`: real-time-object-detection
- `AZURE_VM_NAME`: kazzaz

### 5.3. Create GitHub Workflow

Create a file `.github/workflows/deploy.yml` with the following content:

```yaml
name: Deploy to Azure
on:
  push:
    branches:
      - main

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v2
      
      - name: Login to Azure Container Registry
        uses: docker/login-action@v2
        with:
          registry: ${{ secrets.AZURE_REGISTRY }}
          username: ${{ secrets.AZURE_USERNAME }}
          password: ${{ secrets.AZURE_PASSWORD }}
      
      - name: Build and push Docker image
        uses: docker/build-push-action@v3
        with:
          context: .
          push: true
          tags: ${{ secrets.AZURE_REGISTRY }}/object-detection:latest
      
      - name: Deploy to Azure VM
        uses: azure/CLI@v1
        with:
          azcliversion: latest
          inlineScript: |
            az vm run-command invoke \
              -g ${{ secrets.AZURE_RESOURCE_GROUP }} \
              -n ${{ secrets.AZURE_VM_NAME }} \
              --command-id RunShellScript \
              --scripts "cd /home/azureuser/real-time-object-detection && docker pull ${{ secrets.AZURE_REGISTRY }}/object-detection:latest && docker-compose down && docker-compose up -d"
```

## 6. Monitoring and Maintenance

### 6.1. View Logs

```bash
# View logs from all containers
docker-compose logs

# View logs from a specific service
docker-compose logs app
```

### 6.2. Update the Application

When you push changes to your main branch, the GitHub Actions workflow will automatically:
1. Build a new Docker image
2. Push it to Azure Container Registry
3. Pull the latest image on the VM
4. Restart the containers with the new image

### 6.3. Manual Updates

```bash
# SSH into the VM with IP 40.76.126.51
ssh azureuser@40.76.126.51

# Navigate to the project directory
cd /home/azureuser/real-time-object-detection

# Pull the latest code
git pull

# Pull the latest image
docker pull objectdetectionacr.azurecr.io/object-detection:latest

# Restart containers
docker-compose down
docker-compose up -d
```

### 6.4. Updating Monitoring Services

If you need to update the monitoring setup:

```bash
# SSH into the VM
ssh azureuser@40.76.126.51

# Navigate to the project directory
cd /home/azureuser/real-time-object-detection

# Update monitoring configuration
cp infra/monitoring_init.sh infra/monitoring_init.sh.bak  # Backup
nano infra/monitoring_init.sh  # Edit configuration as needed

# Run the updated monitoring script
bash infra/monitoring_init.sh

# Restart monitoring services
docker-compose -f infra/docker-compose.monitoring.yml down
docker-compose -f infra/docker-compose.monitoring.yml up -d
```

### 6.5. Troubleshooting Monitoring Issues

If you encounter issues with the monitoring services on the VM (IP: 40.76.126.51):

```bash
# Check if monitoring containers are running
docker ps | grep -E 'prometheus|grafana|mlflow'

# Check monitoring container logs
docker logs $(docker ps -q -f name=prometheus)
docker logs $(docker ps -q -f name=grafana)
docker logs $(docker ps -q -f name=mlflow)

# Verify Prometheus targets and metrics
curl http://localhost:9090/api/v1/targets
curl http://localhost:9090/api/v1/query?query=up

# Verify Prometheus configuration
docker exec -it $(docker ps -q -f name=prometheus) cat /etc/prometheus/prometheus.yml

# Verify MLflow is functioning
curl http://localhost:5000/api/2.0/mlflow/experiments/list

# Check for data volume issues
df -h
du -sh mlruns/

# Restart specific service if needed (example for Grafana)
docker restart $(docker ps -q -f name=grafana)

# Check if there are issues with model monitoring
docker logs $(docker ps -q -f name=app)

# Rebuild monitoring configuration if needed
bash infra/monitoring_init.sh
docker-compose -f infra/docker-compose.monitoring.yml down
docker-compose -f infra/docker-compose.monitoring.yml up -d
```

### 6.6. Backup MLflow Data

```bash
# Backup MLflow data
zip -r mlflow_backup_$(date +%Y%m%d).zip mlruns/
```

## 7. Troubleshooting

### 7.1. Check GPU Access

```bash
# Check if GPU is visible to Docker
docker exec -it object-detection-real-time-object-detection-app-1 nvidia-smi
```

### 7.2. Test Model Inference

```bash
# Run a quick test with the inference script
docker exec -it object-detection-real-time-object-detection-app-1 python -c "from ml_models.inference import infer; import cv2; img = cv2.imread('/app/data/sample.jpg'); results = infer(img); print(f'Found {len(results[0])} objects')"
```

### 7.3. Check Monitoring Services

```bash
# Check Prometheus metrics
curl http://localhost:9090/api/v1/query?query=up

# Check MLflow status
curl http://localhost:5000/api/2.0/mlflow/experiments/list
```

## 8. Scaling and Optimization

### 8.1. Optimize for Performance

- Adjust inference batch size in `ml_models/inference.py`
- Use TensorRT optimization for ONNX models
- Monitor model drift and latency using Grafana dashboards

### 8.2. Scaling Up

For higher workloads, consider:

```bash
# Stop the current VM
az vm stop --resource-group real-time-object-detection --name kazzaz

# Resize to a larger GPU VM
az vm resize --resource-group real-time-object-detection --name kazzaz --size Standard_NC12s_v3

# Start the VM again
az vm start --resource-group real-time-object-detection --name kazzaz
```

## 9. Cleanup

When you no longer need the resources:

```bash
# Delete the entire resource group
az group delete --name real-time-object-detection --yes
```

This will remove all resources including the VM, ACR, and associated networking components.

## 10. Application Updates and Common Issues

### 10.1. Updating the Application on VM (IP: 40.76.126.51)

For a complete update of the application including code, models, and configuration:

```bash
# SSH into the VM
ssh azureuser@40.76.126.51

# Navigate to project directory
cd /home/azureuser/real-time-object-detection

# Pull latest code changes
git pull

# Stop the existing services
docker-compose down

# Rebuild the Docker image with latest code
docker build -t objectdetectionacr.azurecr.io/object-detection:latest .

# Push the updated image to ACR
docker push objectdetectionacr.azurecr.io/object-detection:latest

# Restart the application with the new image
docker-compose up -d
```

### 10.2. Common Issues and Fixes

#### GPU Not Available for Inference

If NVIDIA GPU is not being detected:

```bash
# Verify NVIDIA drivers are properly installed
nvidia-smi

# Check if nvidia-docker is working correctly
sudo docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu20.04 nvidia-smi

# If there are issues, reinstall the NVIDIA Container Toolkit
sudo apt-get purge -y nvidia-docker2
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

#### Monitoring Pipeline Failures

If the monitoring tools are not collecting metrics correctly:

```bash
# Check if MLflow, Prometheus, and Grafana are running
docker ps | grep -E 'mlflow|prometheus|grafana'

# Check Python package dependencies
docker exec -it $(docker ps -q -f name=app) pip list | grep -E 'mlflow|psutil|pynvml'

# Install missing packages if needed
docker exec -it $(docker ps -q -f name=app) pip install mlflow psutil pynvml

# Verify monitoring directories exist and have proper permissions
docker exec -it $(docker ps -q -f name=app) ls -la /app/metrics
docker exec -it $(docker ps -q -f name=app) mkdir -p /app/metrics

# Restart monitoring services
bash infra/monitoring_init.sh
docker-compose -f infra/docker-compose.monitoring.yml down
docker-compose -f infra/docker-compose.monitoring.yml up -d
```

#### Model Drift Detection Issues

If model drift detection is not working:

```bash
# Check if metrics files are being created
docker exec -it $(docker ps -q -f name=app) ls -la /app/metrics

# Manually trigger drift detection
docker exec -it $(docker ps -q -f name=app) python -c "from ml_models.mlops_utils import DriftDetector; detector = DriftDetector(); result = detector.detect_drift(); print(f'Drift detected: {result[0]}, Details: {result[1]}')"

# Update MLOps configuration if needed
docker cp config/mlops_config.yaml $(docker ps -q -f name=app):/app/config/mlops_config.yaml
```

### 10.3. Performance Tuning

For optimizing performance on the VM:

```bash
# Check current GPU utilization during inference
docker exec -it $(docker ps -q -f name=app) nvidia-smi dmon

# Monitor real-time resource usage
docker exec -it $(docker ps -q -f name=app) python -c "from ml_models.mlops_utils import HardwareMonitor; monitor = HardwareMonitor(); print(monitor.log_hardware_metrics())"

# Adjust batch size for better GPU utilization
docker exec -it $(docker ps -q -f name=app) sed -i 's/batch_size = [0-9]*/batch_size = 8/' /app/ml_models/inference.py
```