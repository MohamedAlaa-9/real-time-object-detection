#!/bin/bash
# Install Docker
sudo apt update
sudo apt install -y docker.io

# Run Prometheus with config using path relative to script location
SCRIPT_DIR=$(dirname "$0")
docker run -d -p 9090:9090 -v "${SCRIPT_DIR}/prometheus.yml":/etc/prometheus/prometheus.yml prom/prometheus

# Run Grafana
docker run -d -p 3000:3000 grafana/grafana

echo "Prometheus running on port 9090"
echo "Grafana running on port 3000 (login: admin/admin)"
