from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from huggingface_hub import login, hf_hub_download

# FastAPI app setup
app = FastAPI()

# LLM Setup: Perform Hugging Face login and download engine/config
# Ensure this runs only once to avoid repeated downloads
login(token=os.getenv("HF_AUTH_TOKEN"))

engine_path = "/app/tmp/medusa/7B/trt_engines/fp16/1-gpu/rank0.engine"
config_path = "/app/tmp/medusa/7B/trt_engines/fp16/1-gpu/config.json"

hf_hub_download(repo_id='aayushmittalaayush/vicuna-7b-medusa-engine', filename='rank0.engine', local_dir='/app/tmp/medusa/7B/trt_engines/fp16/1-gpu')
hf_hub_download(repo_id='aayushmittalaayush/vicuna-7b-medusa-engine', filename='config.json', local_dir='/app/tmp/medusa/7B/trt_engines/fp16/1-gpu')

# Request schema for inference
class InferenceRequest(BaseModel):
    input_text: str
    max_output_len: int = 100
    temperature: float = 1.0

# Define an endpoint for Medusa decoding inference
@app.post("/infer")
def infer(request: InferenceRequest):
    try:
        # Command to execute inference
        cmd = f"""
        python3 /app/TensorRT-LLM/examples/run.py \
            --engine_dir /app/tmp/medusa/7B/trt_engines/fp16/1-gpu/ \
            --tokenizer_dir /app/vicuna-7b-v1.3/ \
            --max_output_len={request.max_output_len} \
            --medusa_choices="[[0], [1], [2], [3], [4]]" \
            --temperature {request.temperature} \
            --input_text "{request.input_text}"
        """
        # Execute the command
        os.system(cmd)
        return {"message": "Inference executed. Check the logs for results."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
