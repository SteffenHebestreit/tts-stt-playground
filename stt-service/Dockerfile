# cuDNN 9 base (required by ctranslate2 >=4.5); CUDA 12.8 covers Blackwell
# (RTX 50 series, sm_120) like the Qwen3 service images.
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

WORKDIR /app

# System dependencies. ffmpeg is the decoder librosa falls back to; PyAV, which
# faster-whisper decodes with, ships its own libav* in the wheel, so the
# libav*-dev, git and wget packages that used to be here served nothing (every
# requirement resolves to a binary wheel on this base, so nothing compiles).
#
# Python 3.11 like every other image. The base's own python3 is 3.10, which reaches
# end of life this month and on which the resolver held back numpy (2.2 instead of
# 2.4), scipy, scikit-learn and onnxruntime (1.23 instead of 1.30). 22.04 has
# python3.11 in universe.
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    curl \
    ffmpeg \
    libffi-dev \
    libssl-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# python3 and python must be the 3.11 interpreter: pip3 and start.sh run through them.
RUN ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.11 /usr/bin/python

# pip's cache directory is a runtime volume (see docker-compose); keep the mount point writable
RUN mkdir -p /root/.cache/pip && chmod 777 /root/.cache/pip

# Install PyTorch with CUDA 12.8 support first so requirements.txt cannot
# pull in a CPU-only torch wheel as a transitive dependency. It comes BEFORE
# `COPY requirements.txt` so that editing a requirement does not invalidate this
# layer and download several GB of torch again. --no-cache-dir keeps the wheels
# out of the image; they would otherwise sit in the layer next to the install.
#
# Pinned: unpinned, a rebuild took whatever the cu128 index served last. PyTorch
# publishes cu128 wheels for 2.7 through 2.11 only (2.12 and later ship CUDA
# 12.6 / 13.x), so a bump past 2.11 needs a different index and a driver >= 580.
# torch is only used for device detection (app.py); ctranslate2 does the decoding,
# and needs just CUDA 12 + cuDNN 9, which the base image provides. 2.11.0 is what
# the NeMo images in this repo run on the same index.
ARG TORCH_VERSION=2.11.0
ARG TORCHAUDIO_VERSION=2.11.0
RUN pip3 install --no-cache-dir --upgrade pip wheel setuptools
# `torch.cuda.get_arch_list()` is empty without a GPU, which a build host does not
# have; `_cuda_getArchFlags` reads the compiled-in list without a device.
RUN pip3 install --no-cache-dir "torch==${TORCH_VERSION}" "torchaudio==${TORCHAUDIO_VERSION}" \
        --index-url https://download.pytorch.org/whl/cu128 && \
    python3 -c "import sys, torch; cuda = torch.version.cuda; archs = torch._C._cuda_getArchFlags().split(); print('torch', torch.__version__, 'built for CUDA', cuda, 'kernels for', ' '.join(archs)); sys.exit(0 if cuda == '12.8' and 'sm_120' in archs else 'FATAL: this torch build is not CUDA 12.8 with sm_120 (Blackwell, RTX 50 series) kernels; set TORCH_VERSION/TORCHAUDIO_VERSION to a release that has a cu128 wheel')"
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy startup script and application
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh && sed -i 's/\r$//' /app/start.sh
COPY app.py .
COPY json_utils.py .
COPY residency.py .
COPY local_agreement.py .
# Request-body limit and cross-origin guard, imported by app.py at startup.
COPY body_limit.py .
COPY origin_guard.py .

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
