import psutil
import statistics

def print_latency_summary(name, latencies):
    print(f"\n📊 {name} Latency Summary:")
    if not latencies:
        print("❌ No data collected.")
        return
    print(f"Count: {len(latencies)}")
    print(f"Min:   {min(latencies):.2f} ms")
    print(f"Max:   {max(latencies):.2f} ms")
    print(f"Mean:  {statistics.mean(latencies):.2f} ms")
    if len(latencies) > 1:
        print(f"Stdev: {statistics.stdev(latencies):.2f} ms")
    else:
        print("Stdev: N/A (only one data point)")

def log_resources():
    cpu = psutil.cpu_percent()
    mem = psutil.virtual_memory().percent
    print(f"🧠 CPU: {cpu:.1f}% | Memory: {mem:.1f}%")