FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsm6 libxext6 wget curl gnupg && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code
COPY . .

# Create necessary directories
RUN mkdir -p uploads results

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV MODEL_HOST=inference
ENV MODEL_PORT=8001
ENV METRICS_ENABLED=true

# Expose port for API
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8080/health || exit 1

# Run the API
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
