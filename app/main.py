from fastapi import FastAPI, UploadFile, File
from PIL import Image
from app.model import predict_image
from app.utils import log_latency
import io
import time
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
)

app = FastAPI()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    load_start = time.perf_counter()
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    load_latency = (time.perf_counter() - load_start) * 1000
    predictions, preprocess_latency, inference_latency = predict_image(image)
    log_latency("Load", load_latency)
    log_latency("Preprocess", preprocess_latency)
    log_latency("Inference", inference_latency)
    server_latency = (time.perf_counter() - load_start) * 1000
    return {"predictions": predictions, "load_latency": round(load_latency, 2), "preprocess_latency": round(preprocess_latency, 2), "inference_latency": round(inference_latency, 2), "server_latency": round(server_latency, 2)}
