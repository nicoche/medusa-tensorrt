#!/bin/bash
# Log start of inference
echo "Starting TensorRT Medusa inference..." >> /app/inference.log

# Run the TensorRT LLM inference
python3 /app/TensorRT-LLM/examples/run.py \
    --engine_dir /app/tmp/medusa/7B/trt_engines/fp16/1-gpu/ \
    --tokenizer_dir /app/vicuna-7b-v1.3/ \
    --max_output_len=100 \
    --medusa_choices="[[0], [1], [2], [3], [4]]" \
    --temperature 1.0 \
    --input_text "Once upon a time" >> /app/inference.log 2>&1

# Keep the container running to avoid deployment shutdown
tail -f /dev/null
