# Application and Monitoring Update Guide

This guide provides instructions for updating the real-time object detection application and its monitoring infrastructure on the Azure VM with IP `40.76.126.51`.

## 1. Connecting to the VM

```bash
# SSH into the VM
ssh azureuser@40.76.126.51
```

## 2. Updating the Application

### 2.1. Manual Application Update

```bash
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

### 2.2. Verifying the Update

```bash
# Check if containers are running
docker ps

# Check application logs
docker-compose logs app

# Test the application endpoint
curl http://localhost:8081/api/health
```

## 3. Updating the Monitoring Infrastructure

### 3.1. Update Monitoring Configuration

```bash
# Navigate to the project directory
cd /home/azureuser/real-time-object-detection

# Backup current monitoring configuration
cp infra/prometheus.yml infra/prometheus.yml.bak
cp infra/monitoring_init.sh infra/monitoring_init.sh.bak

# Edit the monitoring configuration
nano infra/prometheus.yml
nano infra/monitoring_init.sh
```

### 3.2. Update Monitoring Services

```bash
# Run the updated monitoring initialization script
bash infra/monitoring_init.sh

# Restart monitoring services
docker-compose -f infra/docker-compose.monitoring.yml down
docker-compose -f infra/docker-compose.monitoring.yml up -d
```

### 3.3. Verify Monitoring Services

```bash
# Check if monitoring containers are running
docker ps | grep -E 'prometheus|grafana|mlflow'

# Access monitoring dashboards
echo "Prometheus: http://40.76.126.51:9090"
echo "Grafana: http://40.76.126.51:3000 (login: admin/admin)"
echo "MLflow: http://40.76.126.51:5000"
```

## 4. Troubleshooting Common Issues

### 4.1. Container Issues

```bash
# View container status
docker ps -a

# Check container logs
docker logs $(docker ps -q -f name=prometheus)
docker logs $(docker ps -q -f name=grafana)
docker logs $(docker ps -q -f name=mlflow)

# Restart specific containers
docker restart $(docker ps -q -f name=prometheus)
```

### 4.2. Prometheus Issues

```bash
# Check Prometheus targets
curl http://localhost:9090/api/v1/targets | jq

# Verify Prometheus configuration
docker exec -it $(docker ps -q -f name=prometheus) cat /etc/prometheus/prometheus.yml

# Check for syntax errors in the config
docker exec -it $(docker ps -q -f name=prometheus) promtool check config /etc/prometheus/prometheus.yml
```

### 4.3. Grafana Issues

```bash
# Reset Grafana admin password if needed
docker exec -it $(docker ps -q -f name=grafana) grafana-cli admin reset-admin-password admin

# Verify Grafana datasources
docker exec -it $(docker ps -q -f name=grafana) ls -la /etc/grafana/provisioning/datasources
docker exec -it $(docker ps -q -f name=grafana) cat /etc/grafana/provisioning/datasources/prometheus.yml

# Check Grafana logs
docker logs $(docker ps -q -f name=grafana)
```

### 4.4. MLflow Issues

```bash
# Check if MLflow API is responsive
curl http://localhost:5000/api/2.0/mlflow/experiments/list

# Check MLflow logs
docker logs $(docker ps -q -f name=mlflow)

# Check MLflow data storage
ls -la mlruns/
du -sh mlruns/
```

### 4.5. Disk Space Issues

```bash
# Check disk space
df -h

# Find large files and directories
du -sh /* | sort -hr | head -10
du -sh /var/* | sort -hr | head -10

# Clean up Docker resources if needed
docker system prune -f
```

## 5. Backing Up Data

### 5.1. Back Up MLflow Data

```bash
# Create a backup of MLflow data
cd /home/azureuser/real-time-object-detection
zip -r mlflow_backup_$(date +%Y%m%d).zip mlruns/

# Copy backup to local machine (run this from your local machine)
scp azureuser@40.76.126.51:/home/azureuser/real-time-object-detection/mlflow_backup_*.zip .
```

### 5.2. Back Up Prometheus Data

```bash
# Create a backup of Prometheus data
cd /home/azureuser/real-time-object-detection
docker-compose -f infra/docker-compose.monitoring.yml stop prometheus
tar -czvf prometheus_backup_$(date +%Y%m%d).tar.gz /prometheus
docker-compose -f infra/docker-compose.monitoring.yml start prometheus

# Copy backup to local machine (run this from your local machine)
scp azureuser@40.76.126.51:/home/azureuser/real-time-object-detection/prometheus_backup_*.tar.gz .
```

### 5.3. Back Up Configuration Files

```bash
# Create a backup of configuration files
cd /home/azureuser/real-time-object-detection
mkdir -p backups/$(date +%Y%m%d)
cp -r infra/*.yml infra/*.sh backups/$(date +%Y%m%d)/
cp docker-compose.yml backups/$(date +%Y%m%d)/

# Archive the backup
tar -czvf config_backup_$(date +%Y%m%d).tar.gz backups/$(date +%Y%m%d)/

# Copy backup to local machine (run this from your local machine)
scp azureuser@40.76.126.51:/home/azureuser/real-time-object-detection/config_backup_*.tar.gz .
```

## 6. Restore Procedures

### 6.1. Restore MLflow Data

```bash
# Upload backup to VM (run this from your local machine)
scp mlflow_backup_20250510.zip azureuser@40.76.126.51:/home/azureuser/real-time-object-detection/

# On the VM, restore the data
cd /home/azureuser/real-time-object-detection
docker-compose -f infra/docker-compose.monitoring.yml stop mlflow
rm -rf mlruns/
unzip mlflow_backup_20250510.zip
docker-compose -f infra/docker-compose.monitoring.yml start mlflow
```

### 6.2. Restore Prometheus Data

```bash
# Upload backup to VM (run this from your local machine)
scp prometheus_backup_20250510.tar.gz azureuser@40.76.126.51:/home/azureuser/real-time-object-detection/

# On the VM, restore the data
cd /home/azureuser/real-time-object-detection
docker-compose -f infra/docker-compose.monitoring.yml stop prometheus
rm -rf /prometheus/*
tar -xzvf prometheus_backup_20250510.tar.gz -C /
docker-compose -f infra/docker-compose.monitoring.yml start prometheus
```

### 6.3. Restore Configuration Files

```bash
# Upload backup to VM (run this from your local machine)
scp config_backup_20250510.tar.gz azureuser@40.76.126.51:/home/azureuser/real-time-object-detection/

# On the VM, restore the configuration
cd /home/azureuser/real-time-object-detection
mkdir -p restore
tar -xzvf config_backup_20250510.tar.gz -C restore/
cp -r restore/backups/*/infra/*.yml infra/
cp -r restore/backups/*/infra/*.sh infra/
cp restore/backups/*/docker-compose.yml .

# Restart services with restored configuration
docker-compose down
docker-compose up -d
docker-compose -f infra/docker-compose.monitoring.yml down
docker-compose -f infra/docker-compose.monitoring.yml up -d
```
