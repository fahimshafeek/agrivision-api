# ============================================================
# Dockerfile for MediScan Vision AI Microservice
# Track C: Chest X-Ray Pneumonia Detection
# ============================================================
FROM python:3.10-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Python dependencies (CPU-only PyTorch for deployment)
RUN pip install --no-cache-dir \
    torch==2.1.0+cpu torchvision==0.16.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir \
    fastapi==0.104.1 uvicorn==0.24.0 \
    python-multipart==0.0.6 Pillow==10.1.0

# Copy application files
COPY app.py vision_model.pth ./

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
