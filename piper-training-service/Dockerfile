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
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel

# Install PyTorch first, with CUDA 12.8 support (Blackwell / RTX 50 series, sm_120).
# It comes BEFORE `COPY requirements.txt`: a COPY whose file changed invalidates
# every layer after it, so with the COPY first, editing any requirement downloaded
# this multi-GB wheel again.
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
#
# PyTorch publishes cu128 wheels for 2.7 through 2.11 only (2.12 and later ship
# CUDA 12.6 / 13.x), so a bump past 2.11 needs a different index and driver.
ARG TORCH_VERSION=2.11.0
RUN pip3 install --no-cache-dir "torch==${TORCH_VERSION}" --index-url https://download.pytorch.org/whl/cu128

# Fail here, not after the rest of the build, if the wheel just installed cannot
# run on an RTX 50 card. torch.cuda.get_arch_list() is empty on a build host
# without a GPU, so read the compiled-in list with _cuda_getArchFlags instead.
RUN python3 -c "import torch; arches = torch._C._cuda_getArchFlags().split(); assert str(torch.version.cuda).startswith('12.8'), 'torch %s is built for CUDA %s, not 12.8' % (torch.__version__, torch.version.cuda); assert 'sm_120' in arches, 'torch %s has no sm_120 kernels (Blackwell): %s' % (torch.__version__, arches)"

# Install the rest of the requirements. torch is held by a constraint: if a
# requirement ever asks for another release, the build fails here instead of
# replacing the cu128 wheel with a PyPI one.
COPY requirements.txt .
RUN printf 'torch==%s\n' "${TORCH_VERSION}" > /tmp/torch-constraints.txt \
    && pip3 install --no-cache-dir -c /tmp/torch-constraints.txt -r requirements.txt

# Copy application code
COPY . .

# Make the startup script executable
RUN chmod +x /app/start.sh

EXPOSE 8080

# Use the device detection startup script
CMD ["/app/start.sh"]
