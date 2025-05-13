#!/bin/bash
# Complete System Startup Script
# This script runs all components of the real-time object detection system

set -e  # Exit on any error

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"  # Ensure we're in the project root

# Define colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Real-Time Object Detection System Startup ===${NC}"

# 1. Check if required packages are installed
echo -e "${YELLOW}Checking required packages...${NC}"
pip install -r requirements.txt

# 1.1. Check and build frontend if needed
if [ -d "frontend" ] && [ -f "frontend/package.json" ]; then
    echo -e "${YELLOW}Checking frontend dependencies and building if necessary...${NC}"
    (cd frontend && npm install && npm run build)
else
    echo -e "${YELLOW}Frontend directory or package.json not found, skipping frontend build.${NC}"
fi

# 2. Dataset processing step skipped as training pipeline is removed
echo -e "${GREEN}Dataset processing step skipped as training pipeline is removed.${NC}"

# 3. Prepare models
echo -e "${YELLOW}Preparing models (ensuring base model and any user-provided model are ready)...${NC}"
python ml_models/prepare_models.py

# 4. Check if MLflow server is running
MLFLOW_RUNNING=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:5000/api/2.0/mlflow/experiments/list || echo "404")
if [ "$MLFLOW_RUNNING" = "200" ]; then
    echo -e "${GREEN}MLflow is already running.${NC}"
else
    echo -e "${YELLOW}Starting monitoring services...${NC}"
    # Make monitoring_init.sh executable
    chmod +x infra/monitoring_init.sh
    # Run the monitoring initialization script
    infra/monitoring_init.sh
    # Start monitoring services
    docker-compose -f infra/docker-compose.monitoring.yml up -d

    # Wait for services to start
    echo -e "${YELLOW}Waiting for monitoring services to start...${NC}"
    sleep 5
fi

# 5. Set environment variables for MLflow
export MLFLOW_TRACKING_URI=http://localhost:5000
export PROMETHEUS_MULTIPROC_DIR=/tmp

# 6. Check if backend is already running
BACKEND_RUNNING=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8081/api/health || echo "404")
if [ "$BACKEND_RUNNING" = "200" ]; then
    echo -e "${GREEN}Backend is already running.${NC}"
else
    # Start the backend as a background process
    echo -e "${YELLOW}Starting backend service...${NC}"
    python backend/main.py &
    BACKEND_PID=$!
    echo $BACKEND_PID > backend.pid
    echo -e "${GREEN}Backend started with PID ${BACKEND_PID}${NC}"
    
    # Wait for backend to start
    echo -e "${YELLOW}Waiting for backend to start...${NC}"
    while [ "$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/api/health 2>/dev/null)" != "200" ]; do
        echo -n "."
        sleep 1
    done
    echo -e "${GREEN}Backend is now running!${NC}"
fi

# 7. Start model monitoring if not already running
MODEL_MONITORING_RUNNING=$(ps aux | grep -v grep | grep "model_monitoring.py" | wc -l)
if [ "$MODEL_MONITORING_RUNNING" -gt 0 ]; then
    echo -e "${GREEN}Model monitoring is already running.${NC}"
else
    echo -e "${YELLOW}Starting model monitoring service...${NC}"
    python ml_models/model_monitoring.py --run-service &
    MONITORING_PID=$!
    echo $MONITORING_PID > monitoring.pid
    echo -e "${GREEN}Model monitoring started with PID ${MONITORING_PID}${NC}"
fi

# 8. Display access information
MY_IP=$(hostname -I | awk '{print $1}')
echo -e "${BLUE}===================================================${NC}"
echo -e "${GREEN}All services are now running!${NC}"
echo -e "${BLUE}===================================================${NC}"
echo -e "Access points:"
echo -e "- Frontend/Backend: ${YELLOW}http://localhost:8081${NC}"
echo -e "- MLflow: ${YELLOW}http://localhost:5000${NC}"
echo -e "- Grafana: ${YELLOW}http://localhost:3000${NC} (login: admin/admin)"
echo -e "- Prometheus: ${YELLOW}http://localhost:9090${NC}"
echo -e "${BLUE}===================================================${NC}"
echo -e "To stop all services:"
echo -e "- Backend: ${YELLOW}kill \$(cat backend.pid)${NC}"
echo -e "- Monitoring: ${YELLOW}kill \$(cat monitoring.pid)${NC}"
echo -e "- Docker services: ${YELLOW}docker-compose -f infra/docker-compose.monitoring.yml down${NC}"
echo -e "${BLUE}===================================================${NC}"

echo -e "${GREEN}Press Ctrl+C to stop this script (services will continue running in background).${NC}"
wait