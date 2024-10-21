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
RUN pip3 install huggingface_hub

# Clone the latest version of TensorRT-LLM main branch
RUN git clone --branch main https://github.com/NVIDIA/TensorRT-LLM.git

# Navigate to the medusa example folder to install requirements
WORKDIR /app/TensorRT-LLM/examples/medusa
RUN pip3 install -r requirements.txt

# Create directories to store downloaded engine and config
RUN mkdir -p /app/tmp/medusa/7B/trt_engines/fp16/1-gpu

# Set Hugging Face token environment variable
ENV HF_AUTH_TOKEN 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

# Download rank0.engine and config.json from Hugging Face model repo with authentication
RUN python3 -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='aayushmittalaayush/vicuna-7b-medusa-engine', filename='rank0.engine', local_dir='/app/tmp/medusa/7B/trt_engines/fp16/1-gpu', use_auth_token=True); hf_hub_download(repo_id='aayushmittalaayush/vicuna-7b-medusa-engine', filename='config.json', local_dir='/app/tmp/medusa/7B/trt_engines/fp16/1-gpu', use_auth_token=True)"

# Copy your script to run inference
COPY run.sh /app/
RUN chmod +x /app/run.sh

# Expose port 8080 (if needed)
EXPOSE 8080

# Command to run the inference
CMD ["./run.sh"]
