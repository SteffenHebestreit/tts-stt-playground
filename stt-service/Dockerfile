# cuDNN 9 base (required by ctranslate2 >=4.5); CUDA 12.8 covers Blackwell
# (RTX 50 series, sm_120) like the Qwen3 service images.
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

WORKDIR /app

# System dependencies. ffmpeg is the decoder librosa falls back to; PyAV, which
# faster-whisper decodes with, ships its own libav* in the wheel, so the
# libav*-dev, git and wget packages that used to be here served nothing (every
# requirement resolves to a binary wheel on this base, so nothing compiles).
RUN apt-get update && apt-get install -y \
    python3 \
    python3-dev \
    python3-pip \
    curl \
    ffmpeg \
    libffi-dev \
    libssl-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Create symlink for python command
RUN ln -sf /usr/bin/python3 /usr/bin/python

# pip's cache directory is a runtime volume (see docker-compose); keep the mount point writable
RUN mkdir -p /root/.cache/pip && chmod 777 /root/.cache/pip

# Install PyTorch with CUDA 12.8 support first so requirements.txt cannot
# pull in a CPU-only torch wheel as a transitive dependency. It comes BEFORE
# `COPY requirements.txt` so that editing a requirement does not invalidate this
# layer and download several GB of torch again. --no-cache-dir keeps the wheels
# out of the image; they would otherwise sit in the layer next to the install.
RUN pip3 install --no-cache-dir --upgrade pip wheel setuptools
RUN pip3 install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cu128
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy startup script and application
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh && sed -i 's/\r$//' /app/start.sh
COPY app.py .
COPY json_utils.py .
COPY residency.py .
COPY local_agreement.py .

# Create models directory
RUN mkdir -p /app/models && chmod 755 /app/models

EXPOSE 8000

# Set to 1 once the model is cached: the Hub is then not asked for a revision on
# every load, and the container needs no network. A model that is not cached fails
# to load with a message saying so (see WHISPER_MODEL_SIZE for local paths).
ENV HF_HUB_OFFLINE=0

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["/app/start.sh"]
