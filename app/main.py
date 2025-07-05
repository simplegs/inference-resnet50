from fastapi import FastAPI, UploadFile, File
from PIL import Image
from app.model import predict_image
from app.utils import log_latency
import io

app = FastAPI()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    predictions, latency = predict_image(image)
    log_latency(latency)
    return {"predictions": predictions, "latency_ms": round(latency, 2)}
