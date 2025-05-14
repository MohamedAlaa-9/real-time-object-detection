FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev \
    wget curl gnupg && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install PyTorch (CPU version for simplicity - use CUDA version for GPU support)
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install ONNX runtime (CPU version for simplicity - use GPU version for acceleration)
RUN pip install --no-cache-dir onnxruntime

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model files and scripts
COPY . .

# Create directory for models if not exists
RUN mkdir -p models

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV MODEL_TYPE=onnx
ENV METRICS_ENABLED=true

# Expose port for inference API
EXPOSE 8001

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8001/health || exit 1

# Run the inference script
CMD ["python", "inference.py"]
