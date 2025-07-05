from concurrent.futures import ThreadPoolExecutor
import requests, time
from utils import print_latency_summary

# === Config ===
NUM_THREADS = 200
NUM_REQUESTS = 2000
API_URL = "http://172.19.75.30:8000/predict"
IMAGE_PATH = "images/puppy_real.jpg"  # Update this to your test image
client_latencies = []
gpu_latencies = []

def send_request():
    with open(IMAGE_PATH, "rb") as f:
        files = {"file": ("cat.jpg", f, "image/jpeg")}
        start = time.perf_counter()
        try:
            response = requests.post(API_URL, files=files)
        except requests.exceptions.RequestException as e:
            print(f"❌ Network error: {e}")
            return None, None
        duration = (time.perf_counter() - start) * 1000  # ms

        try:
            json = response.json()
        except Exception as e:
            print(f"❌ JSON parse error: {e}")
            return duration, None

        if response.status_code != 200:
            print(f"❌ HTTP {response.status_code}: {json}")
            return duration, None

        return duration, json.get("latency_ms")

with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
    futures = [executor.submit(send_request) for _ in range(NUM_REQUESTS)]
    results = [f.result() for f in futures]

for client_latency, gpu_latency in results:
    if client_latency is not None:
        client_latencies.append(client_latency)
    if gpu_latency is not None:
        gpu_latencies.append(gpu_latency)

# Report summary
print("\n📦 Load Test Summary")
print(f"🔢 Total Requests: {NUM_REQUESTS}")
print(f"🧵 Threads Used:   {NUM_THREADS}")
print_latency_summary("Client", client_latencies)
print_latency_summary("GPU Inference", gpu_latencies)