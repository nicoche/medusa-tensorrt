#!/bin/bash
python3 /app/TensorRT-LLM/examples/run.py \
    --engine_dir /app/tmp/medusa/7B/trt_engines/fp16/1-gpu/ \
    --tokenizer_dir /app/vicuna-7b-v1.3/ \
    --max_output_len=100 \
    --medusa_choices="[[0], [1], [2], [3], [4]]" \
    --temperature 1.0 \
    --input_text "In a small village, there lived a"
