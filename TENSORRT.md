# TensorRT Optimization for Real-time Object Detection

This guide explains how to use TensorRT optimization to significantly improve inference performance for the real-time object detection system.

## What is TensorRT?

TensorRT is NVIDIA's high-performance deep learning inference optimizer and runtime that delivers low latency and high throughput for deep learning applications. It includes a deep learning inference optimizer and runtime that delivers low latency and high-throughput for deep learning inference applications.

## Benefits of TensorRT

- **2-5x Faster Inference**: Compared to PyTorch or standard ONNX runtime
- **Reduced Memory Usage**: Optimized model uses less GPU memory
- **Lower Latency**: Critical for real-time object detection applications
- **Better Battery Life**: More efficient processing on edge devices

## Installation Requirements

TensorRT requires:
1. NVIDIA GPU with compute capability 3.0 or higher
2. CUDA Toolkit (version compatible with your drivers)
3. cuDNN library
4. Python dependencies: pycuda, tensorrt

To install the dependencies, run:
```bash
./install_tensorrt_dependencies.sh
```

## TensorRT Pipeline

The system includes a complete TensorRT optimization pipeline:

1. **Model Preparation**: Convert PyTorch models to ONNX format
2. **TensorRT Optimization**: Transform ONNX models to TensorRT engines
3. **Inference**: Use optimized TensorRT engines for real-time detection

## How to Optimize Models

To optimize your models with TensorRT:

```bash
./optimize_with_tensorrt.sh
```

This script:
- Checks for ONNX models and exports them if needed
- Runs TensorRT optimization on both fine-tuned and baseline models
- Creates `.trt` engine files in the `ml_models` directory

## Performance Benchmarks

To compare performance between PyTorch, ONNX, and TensorRT models:

```bash
python ml_models/verify_model_pipeline.py
```

This runs inference benchmark tests with all available model formats, allowing you to see the performance improvement from TensorRT.

## Troubleshooting

### Common Issues:
1. **Missing Dependencies**: Run `./install_tensorrt_dependencies.sh` to install required packages
2. **CUDA Version Mismatch**: Ensure TensorRT, CUDA, and cuDNN versions are compatible
3. **Import Errors**: Make sure the environment has the correct Python path

### Checking Status:
To verify which inference engine is being used:
```bash
cat ml_models/model_status.txt
```

## Advanced Configuration

The TensorRT optimization can be configured in `config/train_config.yaml`:

- **Precision**: FP16 or FP32 (INT8 requires calibration)
- **Batch Size**: Fixed or dynamic batch sizing
- **Workspace Size**: Memory allocation for optimization
