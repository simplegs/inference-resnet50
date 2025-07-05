import logging

def log_latency(latency_ms):
    logging.info(f"Inference latency: {round(latency_ms, 2)} ms")
