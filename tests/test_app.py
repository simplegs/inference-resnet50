import requests
import time
import os

# === Config ===
API_URL = "http://172.19.75.30:8000/predict"
IMAGE_PATH = "images/puppy_real.jpg"  # Update this to your test image

def test_inference():
    if not os.path.exists(IMAGE_PATH):
        print(f"❌ File not found: {IMAGE_PATH}")
        return
    
    print(f"📸 Testing inference with image: {IMAGE_PATH}")

    with open(IMAGE_PATH, "rb") as f:
        files = {"file": (os.path.basename(IMAGE_PATH), f, "image/jpeg")}
        
        print(f"📤 Sending request to {API_URL} with file {IMAGE_PATH}")
        start_time = time.time()
        response = requests.post(API_URL, files=files)
        duration = (time.time() - start_time) * 1000  # ms

    if response.status_code == 200:
        predictions = response.json()
        print(f"✅ Success! Response in {duration:.2f} ms")
        print(f"🔎 Top predictions:  {predictions}")
        for label, prob in predictions["predictions"]:
            print(f" - {label}: {round(prob * 100, 2)}%")
    else:
        print(f"❌ Error {response.status_code}: {response.text}")

if __name__ == "__main__":
    test_inference()
