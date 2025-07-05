# 🧠 GPU-Powered ResNet50 Inference Server 🚀

This project demonstrates how to build a **local inference server** using:
- **FastAPI** for a lightweight HTTP API
- **PyTorch** and **ResNet50** (pretrained on ImageNet)
- **GPU acceleration** via your RTX 5060 GPU

Run deep learning inference on your local machine with real-time performance!

---

## 🖥️ Requirements

- Windows 11 or WSL2 (Linux works too)
- Python 3.8+
- NVIDIA GPU (e.g., RTX 5060)
- CUDA-enabled PyTorch

---

From the project root, run below with replace your ip adress

```bash
uvicorn app.main:app --host 172.19.75.30 --port 8000 --reload

---
