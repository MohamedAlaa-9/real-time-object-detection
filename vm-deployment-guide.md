# Real-Time Object Detection System Deployment Guide

This guide provides step-by-step instructions for deploying the Real-Time Object Detection System on a Virtual Machine (VM).

## System Requirements

- Ubuntu 20.04 LTS or later
- Minimum 4 CPU cores
- Minimum 8GB RAM
- At least 20GB free disk space
- NVIDIA GPU (optional but recommended for better performance)

## Deployment Steps

### 1. Clone the Repository

```bash
git clone https://github.com/your-organization/real-time-object-detection.git
cd real-time-object-detection
```

### 2. Run the Setup Script

The setup script will automatically install all dependencies, configure the environment, and start all the services:

```bash
./setup.sh
```

The script performs the following tasks:
- Installs Docker and Docker Compose (if not already installed)
- Creates necessary directories
- Sets up GPU support (if a compatible NVIDIA GPU is detected)
- Starts all services defined in the docker-compose.yml file

### 3. Access the Services

After successful deployment, you can access the various components of the system:

- **Frontend**: http://your-vm-ip:3000
- **Backend API**: http://your-vm-ip:8080
- **MLflow Tracking Server**: http://your-vm-ip:5000
- **Grafana Dashboards**: http://your-vm-ip:3001 (default credentials: admin/admin)
- **Prometheus Metrics**: http://your-vm-ip:9090

### 4. Monitoring the System

The system includes comprehensive monitoring solutions:

- **Prometheus**: Collects metrics from all services
- **Grafana**: Provides visualization dashboards for:
  - Model performance metrics (inference time, accuracy)
  - System resources (CPU, memory, GPU utilization)
  - MLflow experiment tracking

### 5. Managing the Services

To manage the services after deployment, use the following Docker Compose commands:

```bash
# View service status
docker-compose ps

# Stop all services
docker-compose stop

# Start all services
docker-compose start

# Restart all services
docker-compose restart

# Stop and remove all containers
docker-compose down

# View service logs
docker-compose logs -f [service_name]
```

## Troubleshooting

### Common Issues

1. **Services fail to start**:
   - Check Docker service status: `sudo systemctl status docker`
   - Check for container errors: `docker-compose logs [service_name]`

2. **GPU not being utilized**:
   - Verify NVIDIA drivers are installed: `nvidia-smi`
   - Check NVIDIA container toolkit: `docker run --gpus all nvidia/cuda:11.0-base nvidia-smi`

3. **Memory issues**:
   - Check system memory usage: `free -h`
   - Adjust container memory limits in docker-compose.yml if needed

4. **Network connectivity issues**:
   - Check if ports are open: `sudo netstat -tulpn | grep [port]`
   - Verify firewall settings: `sudo ufw status`

### Support

For additional support, please contact the system administrator or raise an issue in the repository.
