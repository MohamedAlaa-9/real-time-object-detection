import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

def build_engine(onnx_file_path, engine_file_path, logger):
    """Builds the TensorRT engine from the ONNX model."""
    builder = trt.Builder(logger)
    config = builder.create_builder_config()
    config.max_workspace_size = 1 << 30  # 1GiB
    # Enable mixed precision, more helpful for newer GPUs
    if builder.platform_has_fast_fp16:
        config.flags |= 1 << int(trt.BuilderFlag.FP16)
    
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    with open(onnx_file_path, 'rb') as model:
        if not parser.parse(model.read()):
            print ('ERROR: Failed to parse the ONNX file.')
            for error in range(parser.num_errors):
                print (parser.get_error(error))
            return None
    
    profile = builder.create_optimization_profile()
    profile.set_shape("images", (1, 3, 640, 640), (1, 3, 640, 640), (1, 3, 640, 640))
    config.add_optimization_profile(profile)

    engine = builder.build_engine(network, config)
    if engine is None:
        print ("Failed to create engine")
        return None
    
    print("Completed creating Engine")
    with open(engine_file_path, "wb") as f:
        f.write(engine.serialize())
    return engine

if __name__ == '__main__':
    onnx_file_path = 'best.onnx'
    engine_file_path = 'best.trt'
    logger = trt.Logger(trt.Logger.WARNING)
    engine = build_engine(onnx_file_path, engine_file_path, logger)
    if engine:
        print("Engine build complete.")
    else:
        print("Engine build failed.")
