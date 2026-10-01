FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

WORKDIR /app

# Install system dependencies.
#
# python3.11 comes from the deadsnakes PPA, NOT from Ubuntu 22.04: jammy's own
# python3.11 package is 3.11.0~rc1, a release candidate from before 3.11 shipped.
# NeMo 3.0 unpacks every .nemo checkpoint with tarfile's extraction `filter=`
# argument (3.11.4 and later), so on the stock package no model can be loaded:
#   TypeError: TarFile.extract() got an unexpected keyword argument 'filter'
# The tarfile check below makes the build fail instead of the first model load.
#
# DEBIAN_FRONTEND is set per command, not as ENV, so it does not leak into the
# running container. Without it, software-properties-common pulls in tzdata, which
# stops to ask "Geographic area:" and an unattended build hangs until it times out.
RUN DEBIAN_FRONTEND=noninteractive apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        software-properties-common gpg-agent ca-certificates && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
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

# NeMo 3 cannot unpack a .nemo file on a Python without tarfile.data_filter
# (3.11.4+); fail here, before the multi-GB torch and NeMo layers, not at runtime.
RUN python3 -c "import sys, tarfile; sys.exit(0 if hasattr(tarfile, 'data_filter') else 'FATAL: python %s has no tarfile.data_filter (needs 3.11.4 or later), so NeMo 3 cannot unpack a .nemo file' % sys.version.split()[0])"

# Install PyTorch with CUDA 12.8 support (required for Blackwell / RTX 50 series, sm_120+).
# Pinned: unpinned, a rebuild took whatever the cu128 index served last. PyTorch
# publishes cu128 wheels for 2.7 through 2.11 only (2.12 and later ship CUDA
# 12.6 / 13.x), so a bump past 2.11 needs a different index and a driver >= 580.
# NeMo 3.0 asks for torch>=2.6 (its README says >=2.7); 2.11 is what this image ran.
ARG TORCH_VERSION=2.11.0
ARG TORCHAUDIO_VERSION=2.11.0
# `torch.cuda.get_arch_list()` is empty without a GPU, which a build host does not
# have; `_cuda_getArchFlags` reads the compiled-in list without a device.
RUN pip3 install --no-cache-dir "torch==${TORCH_VERSION}" "torchaudio==${TORCHAUDIO_VERSION}" \
        --index-url https://download.pytorch.org/whl/cu128 && \
    python3 -c "import sys, torch; cuda = torch.version.cuda; archs = torch._C._cuda_getArchFlags().split(); print('torch', torch.__version__, 'built for CUDA', cuda, 'kernels for', ' '.join(archs)); sys.exit(0 if cuda == '12.8' and 'sm_120' in archs else 'FATAL: this torch build is not CUDA 12.8 with sm_120 (Blackwell, RTX 50 series) kernels; set TORCH_VERSION/TORCHAUDIO_VERSION to a release that has a cu128 wheel')"

# Install application dependencies (NeMo ASR is large; this layer is slow on first build).
# NEMO_TOOLKIT_SPEC replaces the nemo_toolkit line of requirements.txt, so one image
# recipe builds either release line without editing files:
#   --build-arg NEMO_TOOLKIT_SPEC='>=2.7.3,<3'   roll back to the 2.x line
#   --build-arg NEMO_TOOLKIT_SPEC='>=3.0.0,<4'   accept 3.1 and later 3.x releases
# The torch check runs again because installing NeMo must not swap the wheel.
#
# torch and torchaudio are also HELD during the install by a constraint file (the
# way chatterbox-tts-service does it): a NeMo release whose requirements want another
# torch (the `>=3.0.0,<4` line above accepts future ones) then fails the resolver
# right here, instead of downloading several GB of a different, non-cu128 torch and
# failing only at the check below.
COPY requirements.txt .
ARG NEMO_TOOLKIT_SPEC=">=3.0.0,<3.1"
RUN grep -viE '^[[:space:]]*nemo[-_]toolkit' requirements.txt > requirements.app.txt && \
    printf 'torch==%s\ntorchaudio==%s\n' "${TORCH_VERSION}" "${TORCHAUDIO_VERSION}" > /tmp/torch-constraints.txt && \
    pip3 install --no-cache-dir -c /tmp/torch-constraints.txt "nemo_toolkit[asr]${NEMO_TOOLKIT_SPEC}" -r requirements.app.txt && \
    python3 -c "import sys, torch; cuda = torch.version.cuda; archs = torch._C._cuda_getArchFlags().split(); print('torch', torch.__version__, 'built for CUDA', cuda, 'kernels for', ' '.join(archs)); sys.exit(0 if cuda == '12.8' and 'sm_120' in archs else 'FATAL: this torch build is not CUDA 12.8 with sm_120 (Blackwell, RTX 50 series) kernels; set TORCH_VERSION/TORCHAUDIO_VERSION to a release that has a cu128 wheel')"

# Copy application code
COPY app.py .
COPY transcription.py .
COPY model_lifecycle.py .
COPY nemo_common.py .
# Request-body limit and cross-origin guard, imported by app.py at startup.
COPY body_limit.py .
COPY origin_guard.py .

# Pre-download model during build (cached in Docker layer)
# Uncomment to bake the model into the image for faster startup:
# RUN python3 -c "import nemo.collections.asr as a; a.models.ASRModel.from_pretrained('nvidia/canary-180m-flash')"

# torch 2.11's TorchScript compiler segfaults scripting NeMo's
# online_clustering.py at import time; disable JIT scripting (functions
# run eagerly — NeMo works fine without it).
ENV PYTORCH_JIT=0

EXPOSE 5006

HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
    CMD curl -f http://localhost:5006/health || exit 1

# 120 s keep-alive: the gateway pools connections, and uvicorn's 5 s default
# closes them first, so pooled sockets came back dead.
CMD ["python3", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5006", "--timeout-keep-alive", "120"]
