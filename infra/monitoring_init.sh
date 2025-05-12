#!/bin/bash
# Monitoring Infrastructure Initialization Script
# This script sets up MLflow, Prometheus, and Grafana for model monitoring

set -e  # Exit on any error

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Create necessary directories
mkdir -p "$PROJECT_ROOT/mlruns"
mkdir -p "$SCRIPT_DIR/grafana-dashboards"
mkdir -p "$SCRIPT_DIR/grafana-datasources"

echo "=== Creating Prometheus configuration ==="
cat > "$SCRIPT_DIR/prometheus.yml" << EOL
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
      # Target the metrics endpoint exposed by inference.py
      - targets: ["${METRICS_ENDPOINT:-localhost:8001}"]
        labels:
          service: "object_detection"
          component: "model"

  # Backend application metrics
  - job_name: "backend"
    metrics_path: /metrics
    static_configs:
      - targets: ["${BACKEND_ENDPOINT:-localhost:8081}"]
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
      - targets: ["${NODE_EXPORTER:-localhost:9100}"]
        labels:
          service: "object_detection"
          component: "host"

  # MLflow metrics
  - job_name: "mlflow"
    metrics_path: /metrics
    static_configs:
      - targets: ["${MLFLOW_ENDPOINT:-localhost:5000}"]
        labels:
          service: "object_detection" 
          component: "mlops"
EOL

echo "=== Creating Grafana datasource configuration ==="
mkdir -p "$SCRIPT_DIR/grafana-datasources"
cat > "$SCRIPT_DIR/grafana-datasources/prometheus.yml" << EOL
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
EOL

echo "=== Creating Grafana dashboard configuration ==="
mkdir -p "$SCRIPT_DIR/grafana-dashboards"
cat > "$SCRIPT_DIR/grafana-dashboards/dashboard.yml" << EOL
apiVersion: 1

providers:
  - name: 'default'
    folder: ''
    type: file
    options:
      path: /etc/grafana/provisioning/dashboards
EOL

# Copy the dashboard file into the grafana dashboards directory
cp "$SCRIPT_DIR/grafana-dashboards/model-performance.json" "$SCRIPT_DIR/grafana-dashboards/" 2>/dev/null || echo "Dashboard file not found, this will be created later."

echo "=== Creating Docker Compose file for monitoring services ==="
cat > "$SCRIPT_DIR/docker-compose.monitoring.yml" << EOL
version: '3'
services:
  # Prometheus for metrics collection
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ${SCRIPT_DIR}/prometheus.yml:/etc/prometheus/prometheus.yml
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/usr/share/prometheus/console_libraries'
      - '--web.console.templates=/usr/share/prometheus/consoles'
    restart: unless-stopped

  # Grafana for visualization
  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    volumes:
      - ${SCRIPT_DIR}/grafana-dashboards:/etc/grafana/provisioning/dashboards
      - ${SCRIPT_DIR}/grafana-datasources:/etc/grafana/provisioning/datasources
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
      - GF_USERS_ALLOW_SIGN_UP=false
    depends_on:
      - prometheus
    restart: unless-stopped

  # MLflow for experiment tracking
  mlflow:
    image: ghcr.io/mlflow/mlflow:latest
    ports:
      - "5000:5000"
    volumes:
      - ${PROJECT_ROOT}/mlruns:/mlruns
    command: mlflow server --host 0.0.0.0 --serve-artifacts
    environment:
      - MLFLOW_TRACKING_URI=http://localhost:5000
      - MLFLOW_DEFAULT_ARTIFACT_ROOT=/mlruns
    restart: unless-stopped

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
EOL

echo "=== Setup complete ==="
echo "You can now start the monitoring services with:"
echo "docker-compose -f $SCRIPT_DIR/docker-compose.monitoring.yml up -d"
echo
echo "Access points:"
echo "- Prometheus: http://localhost:9090"
echo "- Grafana: http://localhost:3000 (login: admin/admin)"
echo "- MLflow: http://localhost:5000"
echo
echo "To connect your object detection application, set the environment variables:"
echo "export MLFLOW_TRACKING_URI=http://localhost:5000"
echo "export PROMETHEUS_MULTIPROC_DIR=/tmp" 