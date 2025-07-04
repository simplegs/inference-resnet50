import torch
from torchvision import models, transforms
from PIL import Image

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
    input_tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.nn.functional.softmax(output[0], dim=0)
        top5 = torch.topk(probs, 5)
    return [(labels[idx], round(probs[idx].item(), 4)) for idx in top5.indices]
