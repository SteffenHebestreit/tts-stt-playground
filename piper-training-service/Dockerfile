# CUDA 12.8 + cu128 wheels, matching every other GPU service in this repo.
# This was the last service still on 11.8/cu118, which caps kernel support at
# ~sm_90 and therefore fails on Blackwell (RTX 50-series, sm_120) with
# cudaErrorNoKernelImageForDevice.
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

# Python 3.11 like the other services (the pinned onnxruntime has no 3.10 wheel).
# System dependencies, including AMD ROCm support detection tools
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    git \
    wget \
    curl \
    build-essential \
    libsndfile1 \
    sox \
    ffmpeg \
    espeak-ng \
    pciutils \
    lshw \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.11 /usr/bin/python

# Optional: Add AMD ROCm repository (commented out by default)
# RUN wget -q -O - https://repo.radeon.com/rocm/rocm.gpg.key | apt-key add - \
#     && echo 'deb [arch=amd64] https://repo.radeon.com/rocm/apt/debian/ ubuntu main' > /etc/apt/sources.list.d/rocm.list \
#     && apt-get update && apt-get install -y rocm-dev rocm-libs \
#     && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Upgrade pip and install basic build tools
RUN pip3 install --upgrade pip setuptools wheel

# Install Python dependencies in stages to avoid conflicts
COPY requirements.txt .

# Install core dependencies first with CUDA support.
#
# torch is pinned, not "whatever is newest". model_exporter.py exports with the
# TorchScript tracer (torch.onnx.export(dynamo=False)), which PyTorch has
# deprecated: the default exporter since 2.9 is the torch.export-based one, which
# needs onnxscript. Unpinned, the image takes whatever the index serves newest,
# so a rebuild can change the exporter under the service.
#
# 2.11.0 is the release the export test (tests/test_piper_training_core_export.py)
# was run on; the NeMo services in this repo already resolve it from this index
# (see the PYTORCH_JIT note in their Dockerfiles). Bump it together with a re-run
# of that test.
#
# torchaudio is not installed: nothing in this service imports it.
RUN pip3 install --no-cache-dir torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128

# Install the rest of the requirements
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Make the startup script executable
RUN chmod +x /app/start.sh

EXPOSE 8080

# Use the device detection startup script
CMD ["/app/start.sh"]
