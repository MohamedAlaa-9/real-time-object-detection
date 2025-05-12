FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    libopencv-dev \
    wget \
    unzip \
    curl \
    gnupg \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js and npm for frontend build
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
    && apt-get update && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Install TensorRT
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub \
    && echo "deb https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu2004/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list \
    && apt-get update && apt-get install -y libnvinfer8 libnvinfer-plugin8 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app
COPY . /app

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Build frontend
WORKDIR /app/frontend
RUN npm install
RUN npm run build

# Expose port for backend API
EXPOSE 8081

# Return to app root directory
WORKDIR /app

# Copy frontend build to a directory that can be served by the backend
RUN mkdir -p backend/static
RUN cp -r frontend/dist/* backend/static/

# Command to run the backend service
CMD ["python3", "backend/main.py"]
