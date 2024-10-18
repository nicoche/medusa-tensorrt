from flask import Flask, request, jsonify
import tensorrt as trt
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit

app = Flask(__name__)

# Load TensorRT engine
def load_engine(engine_file_path):
    TRT_LOGGER = trt.Logger(trt.Logger.INFO)
    with open(engine_file_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

engine = load_engine("rank0.engine")
context = engine.create_execution_context()

# Define input/output buffers
input_shape = (1, 512)  # Modify based on your model input
output_shape = (1, 2048)  # Modify based on your model output

input_memory = cuda.mem_alloc(np.prod(input_shape) * np.float32(0).nbytes)
output_memory = cuda.mem_alloc(np.prod(output_shape) * np.float32(0).nbytes)

@app.route('/infer', methods=['POST'])
def infer():
    data = request.get_json()
    input_data = np.array(data['input'], dtype=np.float32)

    # Transfer input data to the GPU
    cuda.memcpy_htod(input_memory, input_data)

    # Run inference
    context.execute_v2(bindings=[int(input_memory), int(output_memory)])

    # Transfer output data back from the GPU
    output_data = np.empty(output_shape, dtype=np.float32)
    cuda.memcpy_dtoh(output_data, output_memory)

    return jsonify({'output': output_data.tolist()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
