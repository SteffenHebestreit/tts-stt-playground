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
# running container. Without it, the python3.11 install pulls in tzdata (deadsnakes'
# libpython3.11-stdlib depends on it), which stops to ask "Geographic area:" and an
# unattended build hangs until it times out.
#
# software-properties-common is needed for add-apt-repository only, and it brings
# about 25 of Ubuntu's Python 3.10 packages into /usr/lib/python3/dist-packages,
# which is on python3.11's sys.path too: pip cannot uninstall the distutils-era
# blinker 1.4 from there, takes six and more-itertools from there instead of
# installing its own, and `pip check` fails on pygobject. So it is purged as soon as
# the PPA is added. The PPA's list and key stay, and so do gpg-agent (the CUDA
# base's gnupg2 depends on it) and ca-certificates.
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
        git \
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
# Pinned like the other NeMo images: PyTorch publishes cu128 wheels for 2.7 through 2.11
# only, so a bump past 2.11 needs a different index and a driver >= 580. Magpie-TTS ran
# on 2.11 with NeMo 3.0.0.
ARG TORCH_VERSION=2.11.0
ARG TORCHAUDIO_VERSION=2.11.0
# `torch.cuda.get_arch_list()` is empty without a GPU, which a build host does not
# have; `_cuda_getArchFlags` reads the compiled-in list without a device.
RUN pip3 install --no-cache-dir "torch==${TORCH_VERSION}" "torchaudio==${TORCHAUDIO_VERSION}" \
        --index-url https://download.pytorch.org/whl/cu128 && \
    python3 -c "import sys, torch; cuda = torch.version.cuda; archs = torch._C._cuda_getArchFlags().split(); print('torch', torch.__version__, 'built for CUDA', cuda, 'kernels for', ' '.join(archs)); sys.exit(0 if cuda == '12.8' and 'sm_120' in archs else 'FATAL: this torch build is not CUDA 12.8 with sm_120 (Blackwell, RTX 50 series) kernels; set TORCH_VERSION/TORCHAUDIO_VERSION to a release that has a cu128 wheel')"

# Install application dependencies (NeMo is large; this layer is slow on first build).
# NEMO_TOOLKIT_SPEC replaces the nemo_toolkit line of requirements.txt, so the release
# line is a build argument, not an edit. docker-compose.yml fills it from
# MAGPIE_NEMO_TOOLKIT_SPEC, not from the NEMO_TOOLKIT_SPEC that rolls the ASR images
# back to NeMo 2.x, so that variable does not reach this image (an explicit
# `docker compose build --build-arg NEMO_TOOLKIT_SPEC=...` still does). The `tts`
# extra brings nemo_text_processing, which the text normalization (digits, dates,
# times) depends on.
# The torch check runs again because installing NeMo must not swap the wheel, and torch
# and torchaudio are HELD during the install by a constraint file: a NeMo release that
# wants another torch then fails the resolver right here, instead of downloading several
# GB of a different, non-cu128 torch and failing only at the check below.
COPY requirements.txt .
ARG NEMO_TOOLKIT_SPEC=">=3.0.0,<3.1"
# pip builds a package that ships only as source in a subprocess that ignores
# --no-cache-dir but reads this variable; without it this layer kept 75 MB of wheels
# in /root/.cache/pip, which Docker copies into every new magpie-tts-cache volume.
# An ARG, so the running container does not have it, declared after the torch layer,
# so that layer's cache stays valid.
ARG PIP_NO_CACHE_DIR=1
RUN grep -viE '^[[:space:]]*nemo[-_]toolkit' requirements.txt > requirements.app.txt && \
    printf 'torch==%s\ntorchaudio==%s\n' "${TORCH_VERSION}" "${TORCHAUDIO_VERSION}" > /tmp/torch-constraints.txt && \
    pip3 install --no-cache-dir -c /tmp/torch-constraints.txt "nemo_toolkit[tts]${NEMO_TOOLKIT_SPEC}" -r requirements.app.txt && \
    python3 -c "import sys, torch; cuda = torch.version.cuda; archs = torch._C._cuda_getArchFlags().split(); print('torch', torch.__version__, 'built for CUDA', cuda, 'kernels for', ' '.join(archs)); sys.exit(0 if cuda == '12.8' and 'sm_120' in archs else 'FATAL: this torch build is not CUDA 12.8 with sm_120 (Blackwell, RTX 50 series) kernels; set TORCH_VERSION/TORCHAUDIO_VERSION to a release that has a cu128 wheel')"

# Fail the build, not the first model load, when the NeMo the service needs is not
# there. NeMo 2.x has MagpieTTSModel but not LANGUAGE_TOKENIZER_MAP, which app.py
# imports; and NeMo swallows a missing or broken nemo_text_processing (pynini) and
# then drops every digit from the text. About 10 s, with no GPU or network.
# PYTORCH_JIT=0 for the reason given at the ENV below (which is not set yet here);
# MPLCONFIGDIR keeps the font cache matplotlib builds on import out of the image.
RUN MPLCONFIGDIR=/tmp/mpl PYTORCH_JIT=0 python3 -c "from nemo.collections.tts.models import MagpieTTSModel; from nemo.collections.tts.parts.utils.tts_dataset_utils import LANGUAGE_TOKENIZER_MAP; from nemo_text_processing.text_normalization.normalize import Normalizer" && \
    rm -rf /tmp/mpl /root/.cache

# Bake the OpenJTalk dictionary (about 107 MB) into the image. pyopenjtalk 0.4.1,
# which nemo_toolkit[tts] pulls in for Japanese, otherwise downloads it (22.6 MB,
# from GitHub) on the first Japanese request: under the single-flight model lock,
# through an urlopen with no timeout, into the container layer, which every
# recreation loses, and never on a host without egress. It lands in pyopenjtalk's
# package directory; do not point OPEN_JTALK_DICT_DIR into /root/.cache, which the
# TrueNAS bind mount hides. The \u escapes spell "tesuto" in katakana and keep this
# file ASCII.
RUN python3 -c "import os, pyopenjtalk; assert pyopenjtalk.g2p('\u30c6\u30b9\u30c8'), 'pyopenjtalk g2p returned nothing'; assert os.path.isdir(pyopenjtalk.OPEN_JTALK_DICT_DIR), 'no OpenJTalk dictionary at %s' % pyopenjtalk.OPEN_JTALK_DICT_DIR" && \
    rm -rf /root/.cache

# Copy application code
COPY app.py .
COPY magpie_support.py .
COPY text_splitting.py .
COPY model_lifecycle.py .
# Request-body limit and cross-origin guard, imported by app.py at startup.
COPY body_limit.py .
COPY origin_guard.py .

# torch 2.11's TorchScript compiler segfaults scripting NeMo's
# online_clustering.py at import time; disable JIT scripting (functions
# run eagerly; NeMo works fine without it).
ENV PYTORCH_JIT=0

EXPOSE 5008

# start-period covers a first start that downloads the model and its audio codec.
HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
    CMD curl -f http://localhost:5008/health || exit 1

# 120 s keep-alive: the gateway pools connections, and uvicorn's 5 s default
# closes them first, so pooled sockets came back dead.
CMD ["python3", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5008", "--timeout-keep-alive", "120"]
