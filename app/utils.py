import logging

def log_latency(name, latency_ms):
    logging.info(f"{name} latency: {round(latency_ms, 2)} ms")
