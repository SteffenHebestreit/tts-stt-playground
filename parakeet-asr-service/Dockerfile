FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    curl \
    git \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Create symlink
RUN ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.11 /usr/bin/python

# Upgrade pip
RUN pip3 install --upgrade pip setuptools wheel

# Install PyTorch with CUDA 12.8 support (required for Blackwell / RTX 50 series, sm_120+)
RUN pip3 install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cu128

# Install application dependencies (NeMo ASR is large; this layer is slow on first build)
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .
COPY transcription.py .

# Pre-download model during build (cached in Docker layer)
# Uncomment to bake the model into the image for faster startup:
# RUN python3 -c "import nemo.collections.asr as a; a.models.ASRModel.from_pretrained('nvidia/parakeet-tdt-0.6b-v3')"

EXPOSE 5005

HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
    CMD curl -f http://localhost:5005/health || exit 1

CMD ["python3", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5005"]
