# Deployment Guide for Real-Time Object Detection to Azure VM (40.76.126.51)

This guide provides step-by-step instructions for deploying your local real-time object detection application to the Azure VM with IP `40.76.126.51`.

## 1. Preparing for Deployment

### 1.1. Prerequisites
- SSH access to the Azure VM at 40.76.126.51
- Docker installed on your local machine
- Access to your Azure Container Registry (ACR)

### 1.2. Network Port Configuration

The application requires the following ports to be open on your Azure VM:

| Port | Service | Direction | Purpose |
|------|---------|-----------|---------|
| 22 | SSH | Inbound | SSH access to VM (enabled by default) |
| 8081 | Application | Inbound | Main application API and frontend access |
| 3000 | Grafana | Inbound | Monitoring dashboard access |
| 5000 | MLflow | Inbound | MLflow tracking server access |
| 9090 | Prometheus | Inbound | Prometheus metrics access |
| * | Various | Outbound | Allow VM to access Docker Hub, GitHub, etc. |

To open these ports in Azure, run the following commands:

```bash
# Open required ports for the application
az vm open-port --resource-group real-time-object-detection --name kazzaz --port 8081 --priority 1001
az vm open-port --resource-group real-time-object-detection --name kazzaz --port 3000 --priority 1002
az vm open-port --resource-group real-time-object-detection --name kazzaz --port 5000 --priority 1003
az vm open-port --resource-group real-time-object-detection --name kazzaz --port 9090 --priority 1004
```

To verify port connectivity, use the `verify_azure_ports.sh` script:

```bash
./infra/verify_azure_ports.sh 40.76.126.51
``` 

### 1.2. Building the Docker Image Locally

```bash
# Navigate to your project root
cd /home/sci/WSL_Space/real-time-object-detection

# Build the Docker image
docker build -t real-time-object-detection:latest .

# Test the image locally (optional)
docker run --gpus all -p 8081:8081 --rm real-time-object-detection:latest
```

### 1.3. Pushing to Azure Container Registry

```bash
# Tag the image for your ACR
docker tag real-time-object-detection:latest objectdetectionacr.azurecr.io/object-detection:latest

# Log in to Azure Container Registry
az acr login --name objectdetectionacr

# Push the image to ACR
docker push objectdetectionacr.azurecr.io/object-detection:latest
```

## 2. Deploying to Azure VM

### 2.1. Connect to the VM

```bash
# SSH into the VM
ssh azureuser@40.76.126.51
```

### 2.2. Setup Project Directory on VM

```bash
# Create project directory if it doesn't exist
mkdir -p ~/real-time-object-detection
cd ~/real-time-object-detection

# Clone your repository (if not already cloned)
git clone https://github.com/yourusername/real-time-object-detection.git .
# If repository already exists, update it:
# git pull
```

### 2.3. Create Docker Compose File

Create a `docker-compose.yml` file with the following content:

```yaml
version: '3'
services:
  # Main application
  app:
    image: objectdetectionacr.azurecr.io/object-detection:latest
    ports:
      - "8081:8081"
    volumes:
      - ./data:/app/data
      - ./ml_models:/app/ml_models
      - ./uploads:/app/uploads
      - ./results:/app/results
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - PROMETHEUS_MULTIPROC_DIR=/tmp
    restart: unless-stopped
```

### 2.4. Log in to Azure Container Registry on the VM

```bash
# Log in to ACR (use credentials from Azure portal)
docker login objectdetectionacr.azurecr.io -u <username> -p <password>
```

### 2.5. Start the Application

```bash
# Pull the latest image
docker pull objectdetectionacr.azurecr.io/object-detection:latest

# Start the application
docker-compose up -d
```

## 3. Setting Up Monitoring Infrastructure

### 3.1. Create Monitoring Directory Structure

```bash
# Create necessary directories
mkdir -p ~/real-time-object-detection/infra/grafana-dashboards
mkdir -p ~/real-time-object-detection/infra/grafana-datasources
mkdir -p ~/real-time-object-detection/mlruns
```

### 3.2. Copy Configuration Files

```bash
# Copy prometheus.yml
cat > ~/real-time-object-detection/infra/prometheus.yml << 'EOL'
global:
  scrape_interval: 15s
  evaluation_interval: 15s
  scrape_timeout: 10s

# Alerting rules
rule_files:
  # - "alert_rules.yml"

# Alertmanager configuration
alerting:
  alertmanagers:
    - static_configs:
        - targets:
          # - alertmanager:9093

# Scrape configuration
scrape_configs:
  # Object detection model metrics endpoint
  - job_name: "object_detection"
    metrics_path: /metrics
    static_configs:
      - targets: ["app:8001"]
        labels:
          service: "object_detection"
          component: "model"

  # Backend application metrics
  - job_name: "backend"
    metrics_path: /metrics
    static_configs:
      - targets: ["app:8081"]
        labels:
          service: "object_detection"
          component: "backend"

  # Prometheus self-monitoring
  - job_name: "prometheus"
    static_configs:
      - targets: ["localhost:9090"]

  # Node exporter for host metrics
  - job_name: "node"
    static_configs:
      - targets: ["node-exporter:9100"]
        labels:
          service: "object_detection"
          component: "host"

  # MLflow metrics
  - job_name: "mlflow"
    metrics_path: /metrics
    static_configs:
      - targets: ["mlflow:5000"]
        labels:
          service: "object_detection" 
          component: "mlops"
EOL

# Copy Grafana datasource config
cat > ~/real-time-object-detection/infra/grafana-datasources/prometheus.yml << 'EOL'
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
EOL

# Copy Grafana dashboard config
cat > ~/real-time-object-detection/infra/grafana-dashboards/dashboard.yml << 'EOL'
apiVersion: 1

providers:
  - name: 'default'
    folder: ''
    type: file
    options:
      path: /etc/grafana/provisioning/dashboards
EOL

# Copy model performance dashboard
scp /home/sci/WSL_Space/real-time-object-detection/infra/grafana-dashboards/model-performance.json azureuser@40.76.126.51:~/real-time-object-detection/infra/grafana-dashboards/
```

### 3.3. Create Monitoring Docker Compose File

```bash
# Create docker-compose.monitoring.yml
cat > ~/real-time-object-detection/infra/docker-compose.monitoring.yml << 'EOL'
version: '3'
services:
  # Prometheus for metrics collection
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
    restart: unless-stopped
    networks:
      - monitoring-net
      - default

  # Grafana for visualization
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
    restart: unless-stopped
    networks:
      - monitoring-net
      - default

  # MLflow for experiment tracking
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
    restart: unless-stopped
    networks:
      - monitoring-net
      - default

  # Node exporter for host metrics
  node-exporter:
    image: quay.io/prometheus/node-exporter:latest
    container_name: node-exporter
    command:
      - '--path.rootfs=/host'
    pid: host
    restart: unless-stopped
    volumes:
      - '/:/host:ro,rslave'
    ports:
      - "9100:9100"
    networks:
      - monitoring-net
      - default

networks:
  monitoring-net:
    driver: bridge
EOL
```

### 3.4. Start Monitoring Services

```bash
# Create Docker network for monitoring
docker network create monitoring-net

# Start monitoring services
cd ~/real-time-object-detection
docker-compose -f infra/docker-compose.monitoring.yml up -d
```

### 3.5. Verify Monitoring Services

```bash
# Check if all containers are running
docker ps

# Check monitoring service endpoints
echo "Prometheus: http://40.76.126.51:9090"
echo "Grafana: http://40.76.126.51:3000 (login: admin/admin)"
echo "MLflow: http://40.76.126.51:5000"

# Test Prometheus targets
curl http://localhost:9090/api/v1/targets | grep "object_detection"

# Test Grafana API
curl -u admin:admin http://localhost:3000/api/health

# Test MLflow API
curl http://localhost:5000/api/2.0/mlflow/experiments/list
```

## 4. Validating the Deployment

### 4.1. Testing the Application

```bash
# Check if app is running
docker ps | grep app

# Test the application health endpoint
curl http://localhost:8081/api/health

# Check application logs
docker-compose logs app
```

### 4.2. Testing GPU Availability

```bash
# Check if GPU is accessible to the container
docker exec -it $(docker ps -q -f name=app) nvidia-smi

# Test inference with a sample image
docker exec -it $(docker ps -q -f name=app) python -c "from ml_models.inference import infer; import cv2; import numpy as np; img = np.zeros((640, 640, 3), dtype=np.uint8); results = infer(img); print(f'Inference successful with shape {img.shape}')"
```

### 4.3. Testing Monitoring Integration

```bash
# Check if metrics are collected in Prometheus
curl -s "http://localhost:9090/api/v1/query?query=up" | grep -o '"value":\[.*\]'

# Verify metrics collection for the application
curl -s "http://localhost:9090/api/v1/query?query=object_detection_inference_count" || echo "No metrics collected yet"
```

## 5. Updating the Application and Monitoring

### 5.1. Updating the Application

```bash
# Pull the latest code
cd ~/real-time-object-detection
git pull

# Build and push a new image locally (from your development machine)
# docker build -t objectdetectionacr.azurecr.io/object-detection:latest .
# docker push objectdetectionacr.azurecr.io/object-detection:latest

# On the VM, pull the latest image and restart
docker pull objectdetectionacr.azurecr.io/object-detection:latest
docker-compose down
docker-compose up -d

# Check application logs after update
docker-compose logs --tail 50 app
```

### 5.2. Updating Monitoring Configuration

```bash
# Create backups of current configuration
cp infra/prometheus.yml infra/prometheus.yml.bak
cp infra/docker-compose.monitoring.yml infra/docker-compose.monitoring.yml.bak

# Edit configuration files as needed
nano infra/prometheus.yml

# Restart monitoring services with updated configuration
docker-compose -f infra/docker-compose.monitoring.yml down
docker-compose -f infra/docker-compose.monitoring.yml up -d

# Verify services are running with new configurations
docker ps | grep -E 'prometheus|grafana|mlflow'
```

## 6. Troubleshooting Monitoring Issues

### 6.1. Common Prometheus Issues

```bash
# Check Prometheus container status
docker ps -a | grep prometheus

# View Prometheus logs
docker logs $(docker ps -q -f name=prometheus)

# Verify Prometheus configuration
docker exec -it $(docker ps -q -f name=prometheus) cat /etc/prometheus/prometheus.yml

# Check if targets are being scraped
curl http://localhost:9090/api/v1/targets | grep "object_detection"

# Restart Prometheus if needed
docker restart $(docker ps -q -f name=prometheus)
```

### 6.2. Common Grafana Issues

```bash
# Check Grafana status
docker ps -a | grep grafana

# View Grafana logs
docker logs $(docker ps -q -f name=grafana)

# Reset admin password if needed
docker exec -it $(docker ps -q -f name=grafana) grafana-cli admin reset-admin-password admin

# Check dashboard provisioning
docker exec -it $(docker ps -q -f name=grafana) ls -la /etc/grafana/provisioning/dashboards
```

### 6.3. Common MLflow Issues

```bash
# Check MLflow status
docker ps -a | grep mlflow

# View MLflow logs
docker logs $(docker ps -q -f name=mlflow)

# Check if MLflow API is responsive
curl http://localhost:5000/api/2.0/mlflow/experiments/list

# Inspect data directory permissions
ls -la mlruns/
```

### 6.4. Model Monitoring Issues

```bash
# Check if model monitoring is working
docker exec -it $(docker ps -q -f name=app) python -c "from ml_models.model_monitoring import ModelMonitor; monitor = ModelMonitor(); print('Model monitor initialized successfully')"

# Check for metrics files
docker exec -it $(docker ps -q -f name=app) ls -la /app/metrics

# Check Python package dependencies
docker exec -it $(docker ps -q -f name=app) pip list | grep -E 'mlflow|prometheus|psutil|pynvml'
```

## 7. Backup and Restoration Procedures

### 7.1. Backup Key Data

```bash
# Backup MLflow data
cd ~/real-time-object-detection
zip -r mlflow_backup_$(date +%Y%m%d).zip mlruns/

# Backup configuration files
mkdir -p backups/$(date +%Y%m%d)
cp -r infra/*.yml infra/*.sh backups/$(date +%Y%m%d)/
cp docker-compose.yml backups/$(date +%Y%m%d)/
tar -czvf config_backup_$(date +%Y%m%d).tar.gz backups/$(date +%Y%m%d)/
```

### 7.2. Restoring from Backup

```bash
# Restore MLflow data
cd ~/real-time-object-detection
docker-compose -f infra/docker-compose.monitoring.yml stop mlflow
rm -rf mlruns/
unzip mlflow_backup_YYYYMMDD.zip  # Replace YYYYMMDD with the actual date
docker-compose -f infra/docker-compose.monitoring.yml start mlflow

# Restore configuration files
cd ~/real-time-object-detection
mkdir -p restore
tar -xzvf config_backup_YYYYMMDD.tar.gz -C restore/  # Replace YYYYMMDD with the actual date
cp -r restore/backups/*/infra/*.yml infra/
cp -r restore/backups/*/infra/*.sh infra/
cp restore/backups/*/docker-compose.yml .
```

## 8. Post-Deployment Tasks

### 8.1. Setting Up Automatic Updates

```bash
# Set up a cron job to pull the latest image daily
(crontab -l 2>/dev/null; echo "0 2 * * * cd ~/real-time-object-detection && docker pull objectdetectionacr.azurecr.io/object-detection:latest && docker-compose down && docker-compose up -d") | crontab -
```

### 8.2. Regular Monitoring Check

```bash
# Create a simple monitoring check script
cat > ~/monitoring_check.sh << 'EOL'
#!/bin/bash
echo "Checking monitoring services status at $(date)"
echo "=============================================="
echo "Container status:"
docker ps | grep -E 'prometheus|grafana|mlflow|node-exporter'
echo "=============================================="
echo "Application container status:"
docker ps | grep app
echo "=============================================="
echo "Testing Prometheus API:"
curl -s "http://localhost:9090/api/v1/query?query=up" | grep -o '"value":\[.*\]'
echo "=============================================="
echo "Testing MLflow API:"
curl -s "http://localhost:5000/api/2.0/mlflow/experiments/list" | grep -o '"experiments"'
echo "=============================================="
echo "Checking disk space:"
df -h | grep -E 'Filesystem|/$'
echo "=============================================="
echo "Check complete"
EOL

chmod +x ~/monitoring_check.sh

# Schedule regular checks
(crontab -l 2>/dev/null; echo "0 * * * * ~/monitoring_check.sh >> ~/monitoring_checks.log 2>&1") | crontab -
```

### 8.3. Data Retention Policy

```bash
# Create a cleanup script for old processed videos
cat > ~/cleanup_old_data.sh << 'EOL'
#!/bin/bash
# Cleanup files older than 30 days
find ~/real-time-object-detection/results -type f -name "*.mp4" -mtime +30 -delete
find ~/real-time-object-detection/results -type f -name "*.jpg" -mtime +30 -delete
find ~/real-time-object-detection/uploads -type f -mtime +30 -delete

# Clean up docker system periodically
docker system prune -f
EOL

chmod +x ~/cleanup_old_data.sh

# Schedule weekly cleanup
(crontab -l 2>/dev/null; echo "0 3 * * 0 ~/cleanup_old_data.sh >> ~/cleanup_logs.log 2>&1") | crontab -
```
