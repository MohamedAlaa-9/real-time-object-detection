#!/bin/bash
# Install Docker
sudo apt update
sudo apt install -y docker.io

# Create Docker network for monitoring
docker network create monitoring-net

# Run Prometheus with config using path relative to script location
SCRIPT_DIR=$(dirname "$0")
docker run -d --name prometheus \
  --network monitoring-net \
  -p 9090:9090 \
  -v "${SCRIPT_DIR}/prometheus.yml":/etc/prometheus/prometheus.yml \
  prom/prometheus

# Run Grafana
docker run -d --name grafana \
  --network monitoring-net \
  -p 3000:3000 \
  grafana/grafana

# Run MLflow server
docker run -d --name mlflow \
  --network monitoring-net \
  -p 5000:5000 \
  -v "${SCRIPT_DIR}/../mlruns":/mlruns \
  -e MLFLOW_TRACKING_URI=http://localhost:5000 \
  -e MLFLOW_DEFAULT_ARTIFACT_ROOT=/mlruns \
  ghcr.io/mlflow/mlflow:latest \
  mlflow server \
  --host 0.0.0.0 \
  --serve-artifacts

# Run Node Exporter to collect host metrics (CPU, memory, disk, etc.)
docker run -d --name node-exporter \
  --network monitoring-net \
  --pid="host" \
  -p 9100:9100 \
  -v "/:/host:ro,rslave" \
  quay.io/prometheus/node-exporter:latest \
  --path.rootfs=/host

# Wait for services to start
sleep 2

# Import Grafana dashboards
# Note: This would typically be done with provisioning in production
GRAFANA_URL="http://localhost:3000"
GRAFANA_USER="admin"
GRAFANA_PASSWORD="admin"

# Add Prometheus as a datasource
curl -s -X POST -H "Content-Type: application/json" -d '{
    "name": "Prometheus",
    "type": "prometheus",
    "url": "http://prometheus:9090",
    "access": "proxy",
    "isDefault": true
}' -u "${GRAFANA_USER}:${GRAFANA_PASSWORD}" "${GRAFANA_URL}/api/datasources"

echo "========================================================="
echo "Monitoring Infrastructure Setup"
echo "========================================================="
echo "Prometheus running on port 9090"
echo "Grafana running on port 3000 (login: admin/admin)"
echo "MLflow running on port 5000"
echo "Node Exporter running on port 9100"
echo 
echo "Next steps:"
echo "1. Access Grafana at http://localhost:3000"
echo "2. Access MLflow at http://localhost:5000"
echo "3. Start the model monitoring service:"
echo "   python ml_models/model_monitoring.py --run-service"
echo "========================================================="
