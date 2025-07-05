import requests, time
import statistics
from utils import print_latency_summary

# === Config ===
API_URL = "http://172.19.75.30:8000/predict"
IMAGE_PATH = "images/puppy_real.jpg"  # Update this to your test image
NUM_REQUESTS = 50
client_latencies = []
load_latencies = []
preprocess_latencies = []
inference_latencies = []
server_latencies = []

with open(IMAGE_PATH, "rb") as f:
    image_bytes = f.read()

for i in range(NUM_REQUESTS):
    # Use a monotonic clock
    start = time.perf_counter()
    response = requests.post(
        API_URL,
        files={"file": ("puppy_real.jpg", image_bytes, "image/jpeg")}
    )
    end = time.perf_counter()
    duration = (end - start) * 1000
    client_latencies.append(duration)

    if response.status_code == 200: 
        predictions = response.json()
        print(f"✅ Success! Response in {duration:.2f} ms")
        print(f"🔎 Top predictions:  {predictions}")
        for label, prob in predictions["predictions"]:
            print(f" - {label}: {round(prob * 100, 2)}%")
        load_latencies.append(predictions["load_latency"])
        preprocess_latencies.append(predictions["preprocess_latency"])
        inference_latencies.append(predictions["inference_latency"])
        server_latencies.append(predictions["server_latency"])
    else:
        print(f"❌ Error {response.status_code}: {response.text}")

# Report summary
print_latency_summary("Client", client_latencies)
print_latency_summary("Load", load_latencies)
print_latency_summary("Preprocess", preprocess_latencies)
print_latency_summary("Inference", inference_latencies)
print_latency_summary("Server", server_latencies)