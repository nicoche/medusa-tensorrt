from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from huggingface_hub import login, hf_hub_download
import logging

# FastAPI app setup
app = FastAPI()

# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# LLM Setup: Perform Hugging Face login and download engine/config
try:
    logger.info("Logging into Hugging Face...")
    login(token=os.getenv("HF_AUTH_TOKEN"))
    logger.info("Logged into Hugging Face successfully.")
except Exception as e:
    logger.error("Failed to log in to Hugging Face: %s", str(e))
    raise

# Paths for the engine and config files
engine_path = "/app/tmp/medusa/7B/trt_engines/fp16/1-gpu/rank0.engine"
config_path = "/app/tmp/medusa/7B/trt_engines/fp16/1-gpu/config.json"

try:
    logger.info("Downloading engine and config files...")
    hf_hub_download(repo_id='aayushmittalaayush/vicuna-7b-medusa-engine', filename='rank0.engine', local_dir='/app/tmp/medusa/7B/trt_engines/fp16/1-gpu')
    hf_hub_download(repo_id='aayushmittalaayush/vicuna-7b-medusa-engine', filename='config.json', local_dir='/app/tmp/medusa/7B/trt_engines/fp16/1-gpu')
    logger.info("Engine and config files downloaded successfully.")
except Exception as e:
    logger.error("Failed to download files from Hugging Face: %s", str(e))
    raise

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
        logger.info(f"Executing command: {cmd}")
        os.system(cmd)
        return {"message": "Inference executed. Check the logs for results."}
    except Exception as e:
        logger.error("Inference execution failed: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))
