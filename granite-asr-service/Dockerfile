FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

WORKDIR /app

# Install system dependencies.
#
# python3.11 comes from the deadsnakes PPA, NOT from Ubuntu 22.04: jammy's own
# python3.11 package is 3.11.0~rc1, a release candidate from before 3.11 shipped
# (the NeMo images found that out the hard way; this image follows their recipe so
# every GPU image here runs the same Python).
#
# DEBIAN_FRONTEND is set per command, not as ENV, so it does not leak into the
# running container. Without it, the python3.11 install pulls in tzdata, which stops
# to ask "Geographic area:" and an unattended build hangs until it times out.
#
# software-properties-common is needed for add-apt-repository only, and it brings
# Ubuntu's Python 3.10 packages into /usr/lib/python3/dist-packages, which is on
# python3.11's sys.path too, so it is purged as soon as the PPA is added.
# ffmpeg decodes every upload to 16 kHz mono; libsndfile1 is soundfile's backend.
RUN DEBIAN_FRONTEND=noninteractive apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        software-properties-common gpg-agent ca-certificates && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    DEBIAN_FRONTEND=noninteractive apt-get purge -y --auto-remove software-properties-common && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
        python3.11 \
        python3.11-dev \
        python3-pip \
        curl \
        ffmpeg \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Create symlink
RUN ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.11 /usr/bin/python

# Upgrade pip
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel

# Install PyTorch with CUDA 12.8 support (required for Blackwell / RTX 50 series, sm_120+).
# Pinned like the NeMo images: PyTorch publishes cu128 wheels for 2.7 through 2.11 only,
# so a bump past 2.11 needs a different index and a driver >= 580. torchaudio is not used
# for decoding (ffmpeg does that), but GraniteSpeechFeatureExtractor refuses to run
# without it, and both images share this layer.
# `torch.cuda.get_arch_list()` is empty without a GPU, which a build host does not have;
# `_cuda_getArchFlags` reads the compiled-in list without a device.
ARG TORCH_VERSION=2.11.0
ARG TORCHAUDIO_VERSION=2.11.0
RUN pip3 install --no-cache-dir "torch==${TORCH_VERSION}" "torchaudio==${TORCHAUDIO_VERSION}" \
        --index-url https://download.pytorch.org/whl/cu128 && \
    python3 -c "import sys, torch; cuda = torch.version.cuda; archs = torch._C._cuda_getArchFlags().split(); print('torch', torch.__version__, 'built for CUDA', cuda, 'kernels for', ' '.join(archs)); sys.exit(0 if cuda == '12.8' and 'sm_120' in archs else 'FATAL: this torch build is not CUDA 12.8 with sm_120 (Blackwell, RTX 50 series) kernels; set TORCH_VERSION/TORCHAUDIO_VERSION to a release that has a cu128 wheel')"

# Application dependencies. torch and torchaudio are HELD by a constraint file: a
# transformers or silero-vad release that wants another torch fails the resolver right
# here, instead of downloading a different, non-cu128 torch and failing only at the
# check below.
# PIP_NO_CACHE_DIR is an ARG (not in the running container) declared after the torch
# layer, so that layer's cache stays valid; it also covers the subprocess pip builds an
# sdist in, which ignores --no-cache-dir and would leave wheels in /root/.cache, the
# model-cache volume.
COPY requirements.txt .
ARG PIP_NO_CACHE_DIR=1
RUN printf 'torch==%s\ntorchaudio==%s\n' "${TORCH_VERSION}" "${TORCHAUDIO_VERSION}" > /tmp/torch-constraints.txt && \
    pip3 install --no-cache-dir -c /tmp/torch-constraints.txt -r requirements.txt && \
    python3 -c "import sys, torch; cuda = torch.version.cuda; archs = torch._C._cuda_getArchFlags().split(); print('torch', torch.__version__, 'built for CUDA', cuda, 'kernels for', ' '.join(archs)); sys.exit(0 if cuda == '12.8' and 'sm_120' in archs else 'FATAL: this torch build is not CUDA 12.8 with sm_120 (Blackwell, RTX 50 series) kernels; set TORCH_VERSION/TORCHAUDIO_VERSION to a release that has a cu128 wheel')"

# Fail the build, not the first model load, when the installed transformers lacks the
# native Granite Speech classes (GraniteSpeech5ForCTC, the 470m-turboctc, needs 5.16 or
# later) or the VAD cannot load its bundled model. No GPU and no network needed.
RUN python3 -c "from transformers import GraniteSpeechForConditionalGeneration, GraniteSpeechProcessor, GraniteSpeech5ForCTC, GraniteSpeech5Processor; from silero_vad import load_silero_vad; load_silero_vad()" &&     rm -rf /root/.cache

# Copy application code
COPY app.py .
COPY audio_pipeline.py .
COPY transformers_asr.py .
COPY granite_support.py .
COPY model_lifecycle.py .
# Request-body limit and cross-origin guard, imported by app.py at startup.
COPY body_limit.py .
COPY origin_guard.py .

# The weights land in /root/.cache/huggingface (the granite-asr-cache volume).
EXPOSE 5010

# start-period covers a first start that downloads the checkpoint (~4.6 GB for the
# 2b, ~0.95 GB for the 470m-turboctc).
HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3     CMD curl -f http://localhost:5010/health || exit 1

# 120 s keep-alive: the gateway pools connections, and uvicorn's 5 s default
# closes them first, so pooled sockets came back dead.
CMD ["python3", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5010", "--timeout-keep-alive", "120"]
