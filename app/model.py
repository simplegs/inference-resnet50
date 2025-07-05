import time
import torch
from torchvision import models, transforms
from PIL import Image
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pretrained model
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
model.eval().to(device)

# Preprocessing
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Categories
labels = models.ResNet50_Weights.DEFAULT.meta["categories"]

def predict_image(image: Image.Image):
    preprocess_start = time.perf_counter()
    input_tensor = preprocess(image).unsqueeze(0).to(device)
    preprocess_latency = (time.perf_counter() - preprocess_start) * 1000
    

    inference_start = time.perf_counter()
    with torch.no_grad():
        output = model(input_tensor)
    inference_latency = (time.perf_counter() - inference_start) * 1000

    # Get top 5 predictions
    probs = torch.nn.functional.softmax(output[0], dim=0)
    top5 = torch.topk(probs, 5)
    prediction = [(labels[idx], round(probs[idx].item(), 4)) for idx in top5.indices]

    return prediction, preprocess_latency, inference_latency
