FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

# torch/torchaudio are pinned so a rebuild cannot silently move to a release
# whose cu128 wheels drop Blackwell. Override with --build-arg to move on; the
# sm_120 check below fails the build if the result cannot run on an RTX 50 card.
ARG TORCH_VERSION=2.8.0
ARG TORCHAUDIO_VERSION=2.8.0

WORKDIR /app

# Install system dependencies
# (ffmpeg decodes every upload to 16 kHz mono; libsndfile1 is soundfile's backend.)
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
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel

# Install PyTorch with CUDA 12.8 support (required for Blackwell / RTX 50 series, sm_120+).
# Before `COPY requirements.txt` so editing a requirement does not invalidate this
# multi-GB layer.
RUN pip3 install --no-cache-dir "torch==${TORCH_VERSION}" "torchaudio==${TORCHAUDIO_VERSION}" \
    --index-url https://download.pytorch.org/whl/cu128

# flash-attn is deliberately not installed. It compiles CUDA kernels, and this
# -runtime base has no nvcc, so the install could only ever succeed through an
# exact-match prebuilt wheel and otherwise failed silently (the old layer ended in
# `|| echo`); the service never asked for it either. It runs on PyTorch SDPA. To
# experiment, build your own image with flash-attn and set
# QWEN3_ASR_ATTN_IMPLEMENTATION=flash_attention_2.

# Install application dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Fail the build, not the first request on the card, if the torch that ended up
# installed (requirements may re-resolve it) has no Blackwell kernels. Works
# without a GPU: torch.cuda.get_arch_list() is empty there, so read the
# compiled-in list with _cuda_getArchFlags instead.
RUN python3 -c "import torch; arches = torch._C._cuda_getArchFlags().split(); assert 'sm_120' in arches, 'torch %s has no sm_120 kernels (Blackwell): %s' % (torch.__version__, arches)"

# Copy application code
COPY app.py .
COPY audio_pipeline.py .
COPY model_lifecycle.py .

# Pre-download model during build (cached in Docker layer)
# Uncomment to bake model into image for faster startup:
# RUN python3 -c "from qwen_asr import Qwen3ASRModel; Qwen3ASRModel.from_pretrained('Qwen/Qwen3-ASR-1.7B', dtype='bfloat16', device_map='cpu')"

EXPOSE 5002

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:5002/health || exit 1

# 120 s keep-alive: the gateway pools connections, and uvicorn's 5 s default
# closes them first, so pooled sockets came back dead.
CMD ["python3", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5002", "--timeout-keep-alive", "120"]
