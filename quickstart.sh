#!/bin/bash
set -e


chmod +x quickstart.sh && ./quickstart.sh --start


# Quick Start Script for Real-Time Object Detection System
echo "=========================================="
echo "Quick Start - Real-Time Object Detection"
echo "=========================================="
echo ""

# Function to check prerequisites
check_prerequisites() {
  echo "Checking prerequisites..."
  
  # Check Docker
  if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    echo "   Visit https://docs.docker.com/get-docker/"
    exit 1
  else
    echo "✅ Docker is installed."
  fi
  
  # Check Docker Compose
  if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    echo "   Visit https://docs.docker.com/compose/install/"
    exit 1
  else
    echo "✅ Docker Compose is installed."
  fi
  
  # Check Git
  if ! command -v git &> /dev/null; then
    echo "❌ Git is not installed. Please install Git first."
    echo "   Visit https://git-scm.com/downloads"
    exit 1
  else
    echo "✅ Git is installed."
  fi
}

# Function to start the system
start_system() {
  echo "Starting the system..."
  
  # Build and start the containers
  docker-compose up -d
  
  # Wait for services to be ready
  echo "Waiting for services to start..."
  sleep 10
  
  echo "System is running! Access the components at:"
  echo "- Frontend: http://localhost:3000"
  echo "- Backend API: http://localhost:8080"
  echo "- MLflow UI: http://localhost:5000"
  echo "- Grafana: http://localhost:3001 (admin/admin)"
  echo "- Prometheus: http://localhost:9090"
}

# Main execution
check_prerequisites

if [ "$1" == "--help" ] || [ "$1" == "-h" ]; then
  echo "Usage: ./quickstart.sh [OPTION]"
  echo ""
  echo "Options:"
  echo "  --help, -h     Show this help message"
  echo "  --start, -s    Start the system"
  echo "  --stop         Stop the system"
  echo "  --status       Check system status"
  echo ""
  exit 0
elif [ "$1" == "--start" ] || [ "$1" == "-s" ] || [ -z "$1" ]; then
  start_system
elif [ "$1" == "--stop" ]; then
  echo "Stopping the system..."
  docker-compose down
elif [ "$1" == "--status" ]; then
  echo "Checking system status..."
  docker-compose ps
else
  echo "Unknown option: $1"
  echo "Use --help to see available options."
  exit 1
fi
