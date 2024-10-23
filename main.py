from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from huggingface_hub import login, hf_hub_download
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app setup
app = FastAPI()

@app.on_event("startup")
async def startup_event():
    logger.info("Logging into Hugging Face...")
    login(token=os.getenv("HF_AUTH_TOKEN"))
    logger.info("Logged into Hugging Face successfully.")

    logger.info("Downloading engine and config files...")
    hf_hub_download(repo_id='aayushmittalaayush/vicuna-7b-medusa-engine', filename='rank0.engine', local_dir='/app/tmp/medusa/7B/trt_engines/fp16/1-gpu')
    hf_hub_download(repo_id='aayushmittalaayush/vicuna-7b-medusa-engine', filename='config.json', local_dir='/app/tmp/medusa/7B/trt_engines/fp16/1-gpu')
    logger.info("Downloaded engine and config files successfully.")

class InferenceRequest(BaseModel):
    input_text: str
    max_output_len: int = 100
    temperature: float = 1.0

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/infer")
def infer(request: InferenceRequest):
    try:
        # Command to execute inference (pseudo command for the example)
        cmd = f"""
        python3 /app/TensorRT-LLM/examples/run.py \
            --engine_dir /app/tmp/medusa/7B/trt_engines/fp16/1-gpu/ \
            --tokenizer_dir /app/vicuna-7b-v1.3/ \
            --max_output_len={request.max_output_len} \
            --medusa_choices="[[0], [1], [2], [3], [4]]" \
            --temperature {request.temperature} \
            --input_text "{request.input_text}"
        """
        os.system(cmd)
        return {"message": "Inference executed. Check the logs for results."}
    except Exception as e:
        logger.error(f"Inference error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
