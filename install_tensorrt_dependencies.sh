#!/bin/bash
# Script to install TensorRT and its dependencies for the real-time object detection system

set -e  # Exit on any error

echo "======================================================="
echo "         TensorRT Dependencies Installation            "
echo "======================================================="
echo
echo "This script will guide you through installing TensorRT and its dependencies"
echo "for the real-time object detection system."
echo

# Check if CUDA is installed
if command -v nvcc &> /dev/null; then
    cuda_version=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)
    echo "✓ CUDA detected: version $cuda_version"
else
    echo "❌ CUDA not detected. TensorRT requires CUDA."
    echo "Please install CUDA from https://developer.nvidia.com/cuda-downloads first."
    exit 1
fi

# Check for Python
if command -v python3 &> /dev/null; then
    python_cmd="python3"
elif command -v python &> /dev/null; then
    python_cmd="python"
else
    echo "Python not found. Please install Python 3.6 or newer."
    exit 1
fi

echo "Python: $($python_cmd --version)"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    $python_cmd -m venv venv
    echo "Virtual environment created"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install PyTorch with CUDA support first
echo "Installing PyTorch with CUDA support..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install base dependencies
echo "Installing base dependencies..."
pip install numpy opencv-python onnx onnxruntime-gpu

# Install pycuda
echo "Installing PyCUDA..."
pip install pycuda

# TensorRT installation instructions based on OS
echo
echo "======================================================="
echo "                TensorRT Installation                  "
echo "======================================================="
echo
echo "TensorRT must be installed based on your CUDA version and OS."
echo "Here are the recommended installation methods:"
echo

os_name=$(uname -s)
if [ "$os_name" = "Linux" ]; then
    # Check for different Linux distributions
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        if [[ "$ID" == "ubuntu" ]]; then
            echo "Ubuntu detected. TensorRT can be installed using apt:"
            echo "1. Add NVIDIA package repositories:"
            echo "   sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub"
            echo "   sudo add-apt-repository \"deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /\""
            echo "   sudo apt-get update"
            echo
            echo "2. Install TensorRT:"
            echo "   sudo apt-get install libnvinfer8 libnvinfer-plugin8 python3-libnvinfer tensorrt"
            echo
            echo "3. Install Python package:"
            echo "   pip install tensorrt"
        elif [[ "$ID" == "centos" || "$ID" == "rhel" ]]; then
            echo "CentOS/RHEL detected. Install TensorRT using yum:"
            echo "Follow instructions at: https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html#installing-rpm"
        else
            echo "Your Linux distribution is $ID."
            echo "Please download TensorRT from NVIDIA Developer website:"
            echo "https://developer.nvidia.com/tensorrt-download"
        fi
    fi
elif [ "$os_name" = "Darwin" ]; then
    echo "macOS detected. TensorRT support on macOS is limited."
    echo "Consider using Docker with NVIDIA container runtime for TensorRT on macOS."
else
    echo "Windows detected (or unknown OS)."
    echo "For Windows, download TensorRT from NVIDIA Developer website:"
    echo "https://developer.nvidia.com/tensorrt-download"
fi

echo
echo "After installing TensorRT, you can install the Python package with:"
echo "pip install tensorrt"
echo
echo "For WSL users, you may need to install CUDA for WSL first:"
echo "https://docs.nvidia.com/cuda/wsl-user-guide/index.html"
echo

echo "======================================================="
echo "                    NEXT STEPS                        "
echo "======================================================="
echo "1. Install TensorRT using the instructions above"
echo "2. Run ./optimize_with_tensorrt.sh to optimize models"
echo "3. Run python ml_models/verify_model_pipeline.py to verify the pipeline with benchmarks"
echo

echo "Would you like to attempt TensorRT installation via pip? (y/n)"
read -r pip_install
if [[ "$pip_install" == "y" || "$pip_install" == "Y" ]]; then
    echo "Installing TensorRT via pip (this might not work on all systems)..."
    pip install tensorrt
    echo "Installation attempted. You may still need to install the system packages."
fi

echo "Dependencies installation completed!"
