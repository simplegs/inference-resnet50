import asyncio
import httpx
import io
import time
from concurrent.futures import ThreadPoolExecutor
import requests, time
from utils import print_latency_summary
import threading

# === Config ===
CONCURRENCY = 10
DURATION_SECONDS = 10
API_URL = "http://172.19.75.30:8000/predict"
IMAGE_PATH = "images/puppy_real.jpg"  # Update this to your test image
# Global variable to store cached image bytes
CACHED_IMAGE_BYTES = None

client_latencies = []
request_latencies = []
load_latencies = []
preprocess_latencies = []
inference_latencies = []
server_latencies = []

def load_image_bytes_once():
    """Loads image bytes only once."""
    global CACHED_IMAGE_BYTES
    if CACHED_IMAGE_BYTES is None:
        try:
            with open(IMAGE_PATH, "rb") as f:
                CACHED_IMAGE_BYTES = f.read()
            print(f"Image '{IMAGE_PATH}' loaded into memory.")
        except FileNotFoundError:
            print(f"Error: Image file not found at {IMAGE_PATH}")
            exit(1)
        except Exception as e:
            print(f"Error loading image: {e}")
            exit(1)
    return CACHED_IMAGE_BYTES

async def send_request_async(client):
    image_bytes = load_image_bytes_once()
    if image_bytes is None:
        return None, None, None

    file_like_object = io.BytesIO(image_bytes)
    files = {"file": ("cat.jpg", file_like_object, "image/jpeg")}

    start = time.perf_counter()
    try:
        response = await client.post(API_URL, files=files)
        request_duration = (time.perf_counter() - start) * 1000
    except httpx.RequestError as e:
        print(f"❌ Network error: {e}")
        return None, None, None

    try:
        prediction = response.json()
    except Exception as e:
        print(f"❌ JSON parse error: {e}")
        return None, None, None

    if response.status_code != 200:
        print(f"❌ HTTP {response.status_code}: {prediction}")
        return None, None, None

    total_client_duration = (time.perf_counter() - start) * 1000  # ms
    return request_duration, total_client_duration, prediction

async def send_request_loop(client, end_time):
    results = []
    while time.time() < end_time:
        request_duration, total_client_duration, prediction = await send_request_async(client)
        if request_duration is not None and total_client_duration is not None and prediction is not None:
            results.append((request_duration, total_client_duration , prediction))
    return results

async def async_main():
    start_time = time.time()

    global CACHED_IMAGE_BYTES
    with open(IMAGE_PATH, "rb") as f:
        CACHED_IMAGE_BYTES = f.read()

    end_time = start_time + DURATION_SECONDS

    async with httpx.AsyncClient(timeout=10.0) as client:
        tasks = [send_request_loop(client, end_time) for _ in range(CONCURRENCY)]
        all_results = await asyncio.gather(*tasks)

    
    for task_results in all_results:  # Each task returns a list of (latency, prediction)
        for request_latency, client_latency, predictions in task_results:
            request_latencies.append(request_latency)
            client_latencies.append(client_latency)
            load_latencies.append(predictions["load_latency"])
            preprocess_latencies.append(predictions["preprocess_latency"])
            inference_latencies.append(predictions["inference_latency"])
            server_latencies.append(predictions["server_latency"])

    # Report summary
    print("\n📦 Load Test Summary")
    print(f"🕓 Duration:        {DURATION_SECONDS} seconds")
    print(f"⚙️  Concurrency:     {CONCURRENCY}")
    rps = len(client_latencies) / DURATION_SECONDS
    print(f"🚀 Throughput: {rps:.2f} req/s")
    print_latency_summary("Client request only", request_latencies)
    print_latency_summary("Client", client_latencies)
    print_latency_summary("Load", load_latencies)
    print_latency_summary("Preprocess", preprocess_latencies)
    print_latency_summary("Inference", inference_latencies)
    print_latency_summary("Server", server_latencies)

if __name__ == "__main__":
    asyncio.run(async_main())