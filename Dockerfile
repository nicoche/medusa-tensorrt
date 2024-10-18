# Use NVIDIA CUDA base image
FROM nvidia/cuda:12.5.1-devel-ubuntu22.04

# Set working directory
WORKDIR /app

# Install dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3.10 \
    git \
    libopenmpi-dev

# Clone the latest version of TensorRT-LLM main branch
RUN git clone --branch main https://github.com/NVIDIA/TensorRT-LLM.git

# Navigate to the medusa example folder to install requirements
WORKDIR /app/TensorRT-LLM/examples/medusa
RUN pip3 install -r requirements.txt

# Create directories to copy the engine and config
RUN mkdir -p /app/tmp/medusa/7B/trt_engines/fp16/1-gpu

# Copy your engine and config files
COPY rank0.engine /app/tmp/medusa/7B/trt_engines/fp16/1-gpu/
COPY config.json /app/tmp/medusa/7B/trt_engines/fp16/1-gpu/

# Copy your script to run inference
COPY run.sh /app/
RUN chmod +x /app/run.sh

# Expose port 8080 (if needed)
EXPOSE 8080

# Command to run the inference
CMD ["./run.sh"]
