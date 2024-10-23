# Use NVIDIA CUDA base image
FROM nvidia/cuda:12.5.1-devel-ubuntu22.04

# Set working directory
WORKDIR /app

# Install dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3.10 \
    git \
    wget \
    libopenmpi-dev

# Install Hugging Face client library for downloading models
RUN pip3 install huggingface_hub fastapi uvicorn

# Clone the latest version of TensorRT-LLM main branch
RUN git clone --branch main https://github.com/NVIDIA/TensorRT-LLM.git

# Navigate to the medusa example folder to install requirements
WORKDIR /app/TensorRT-LLM/examples/medusa
RUN pip3 install -r requirements.txt

# Create directories to store downloaded engine and config
RUN mkdir -p /app/tmp/medusa/7B/trt_engines/fp16/1-gpu

# Set working directory back to /app where the main.py file is
WORKDIR /app

# Copy FastAPI script
COPY main.py /app/

# Set Hugging Face token environment variable
ENV HF_AUTH_TOKEN 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

# Command to run FastAPI server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
