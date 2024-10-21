#!/bin/bash

# Default input text if not provided
DEFAULT_INPUT_TEXT="Write me a 400 word blog on use of Traditional Machine Learning Alogithm in the new world of deep learning and LLMs."

# Use environment variable INPUT_TEXT, or fall back to default input text
INPUT_TEXT=${INPUT_TEXT:-$DEFAULT_INPUT_TEXT}

# Execute the inference with the provided or default input text
python3 /app/TensorRT-LLM/examples/run.py \
    --engine_dir /app/tmp/medusa/7B/trt_engines/fp16/1-gpu/ \
    --tokenizer_dir /app/vicuna-7b-v1.3/ \
    --max_output_len=100 \
    --medusa_choices="[[0], [1], [2], [3], [4]]" \
    --temperature 1.0 \
    --input_text "$INPUT_TEXT"
