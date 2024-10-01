# Base image: Use NVIDIA's CUDA image with development tools
FROM nvidia/cuda:12.5.1-devel-ubuntu22.04

# Set up the environment
ENV DEBIAN_FRONTEND=noninteractive

# Install Python and dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    openmpi-bin \
    libopenmpi-dev \
    git \
    git-lfs \
    && rm -rf /var/lib/apt/lists/*

# Install TensorRT-LLM and dependencies
RUN pip3 install tensorrt_llm -U --pre --extra-index-url https://pypi.nvidia.com

# Clone the TensorRT-LLM repo for examples
RUN git clone https://github.com/NVIDIA/TensorRT-LLM.git && \
    cd TensorRT-LLM && \
    git lfs install && \
    pip3 install -r examples/medusa/requirements.txt

# Download the required models (ensure Git LFS is used to fetch large weights)
RUN git lfs install && \
    git clone https://huggingface.co/lmsys/vicuna-7b-v1.3 && \
    git clone https://huggingface.co/FasterDecoding/medusa-vicuna-7b-v1.3

# Define the working directory
WORKDIR /TensorRT-LLM

# Convert and build the Medusa TensorRT engine
RUN python3 convert_checkpoint.py --model_dir ./vicuna-7b-v1.3 \
                                  --medusa_model_dir medusa-vicuna-7b-v1.3 \
                                  --output_dir ./tllm_checkpoint_1gpu_medusa \
                                  --dtype float16 \
                                  --num_medusa_heads 4

# Build TensorRT engine
RUN trtllm-build --checkpoint_dir ./tllm_checkpoint_1gpu_medusa \
                 --output_dir ./tmp/medusa/7B/trt_engines/fp16/1-gpu/ \
                 --gemm_plugin float16 \
                 --speculative_decoding_mode medusa \
                 --max_batch_size 4

EXPOSE 8000


# Set the entry point to ensure the engine is ready
CMD ["echo", "TensorRT Engine build completed."]
