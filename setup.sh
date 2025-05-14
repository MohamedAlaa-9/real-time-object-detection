#!/bin/bash
set -e

# Real-Time Object Detection System Setup Script
echo "=========================================="
echo "Real-Time Object Detection System Setup"
echo "=========================================="
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Docker not found. Installing Docker..."
    sudo apt-get update
    sudo apt-get install -y apt-transport-https ca-certificates curl software-properties-common
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
    sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
    sudo apt-get update
    sudo apt-get install -y docker-ce docker-ce-cli containerd.io
    sudo systemctl enable docker
    sudo systemctl start docker
    sudo usermod -aG docker $USER
    echo "Docker installed successfully."
else
    echo "Docker is already installed."
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "Docker Compose not found. Installing Docker Compose..."
    sudo curl -L "https://github.com/docker/compose/releases/download/v2.12.2/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose
    echo "Docker Compose installed successfully."
else
    echo "Docker Compose is already installed."
fi

# Create required directories
echo "Creating required directories..."
mkdir -p infra/grafana-dashboards/dashboards
mkdir -p infra/grafana-datasources
mkdir -p uploads
mkdir -p results
mkdir -p mlruns

# Set permissions for created directories
chmod -R 777 uploads results mlruns

# Check for GPU support
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU detected. Installing NVIDIA Container Toolkit..."
    
    # Install NVIDIA Container Toolkit
    curl -s -L https://nvidia.github.io/nvidia-container-runtime/gpgkey | sudo apt-key add -
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
    curl -s -L https://nvidia.github.io/nvidia-container-runtime/$distribution/nvidia-container-runtime.list | sudo tee /etc/apt/sources.list.d/nvidia-container-runtime.list
    sudo apt-get update
    sudo apt-get install -y nvidia-container-runtime
    
    # Restart Docker
    sudo systemctl restart docker
    
    echo "NVIDIA Container Toolkit installed successfully."
else
    echo "No NVIDIA GPU detected. Continuing with CPU-only configuration."
    
    # Update docker-compose.yml to remove GPU settings
    sed -i '/driver: nvidia/d' docker-compose.yml
    sed -i '/count: 1/d' docker-compose.yml
    sed -i '/capabilities: \[gpu\]/d' docker-compose.yml
    
    # Update ml_models requirements to use CPU versions
    sed -i 's/onnxruntime-gpu/onnxruntime/g' ml_models/requirements.txt
fi

echo "Starting the application stack..."
docker-compose up -d

echo "=========================================="
echo "Setup completed successfully!"
echo ""
echo "You can access the applications at:"
echo "Frontend: http://localhost:3000"
echo "Backend API: http://localhost:8080"
echo "MLflow UI: http://localhost:5000"
echo "Grafana: http://localhost:3001 (admin/admin)"
echo "Prometheus: http://localhost:9090"
echo "=========================================="
