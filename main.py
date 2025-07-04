from fastapi import FastAPI, UploadFile, File
from PIL import Image
from model import predict_image
import io

app = FastAPI()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    predictions = predict_image(image)
    return {"predictions": predictions}
