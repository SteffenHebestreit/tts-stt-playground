FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

# torch/torchaudio are pinned so a rebuild cannot silently move to a release
# whose cu128 wheels drop Blackwell. Override with --build-arg to move on; the
# sm_120 check below fails the build if the result cannot run on an RTX 50 card.
#
# ASSUMPTION: torch/torchaudio 2.8.0 exist as +cu128 wheels on
# download.pytorch.org (the qwen3-tts image pins the same pair). The environment
# this file was written in could not reach that index, so only the PyPI release
# list was checked; the build-time check below is what proves it.
ARG TORCH_VERSION=2.8.0
ARG TORCHAUDIO_VERSION=2.8.0

# chatterbox-tts comes from GitHub, not PyPI. The PyPI release (0.1.7, 2026-03-26)
# predates the v3 multilingual checkpoint (GitHub commits 3f35dfc, 2026-05-01, and
# 65b1843, 2026-06-10): its from_pretrained() has no t3_model argument, and it
# pins torch==2.6.0 / torchaudio==2.6.0 (PyPI builds: cu124, no sm_120 kernels) and
# gradio, which would downgrade the cu128 torch installed below.
#
# CHATTERBOX_REF is a commit SHA, resolved with `git ls-remote <repo> HEAD` on
# 2026-09-29 (master at that time: "Update ChatterboxNano (#542)", 2026-07-21; its
# multilingual code is the v3 release's). A SHA, not "master": rebuilds must not
# move by themselves. To update, run
#     git ls-remote https://github.com/resemble-ai/chatterbox.git HEAD
# and pass the result with --build-arg CHATTERBOX_REF=<sha>. The package version
# string stays "0.1.7" on master, so /status reports this ref as well.
ARG CHATTERBOX_REPO=https://github.com/resemble-ai/chatterbox.git
ARG CHATTERBOX_REF=5de7a54aa4e5e2baadb0182dde554908b48b85c2

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

# setuptools<82: resemble-perth (the watermarker chatterbox loads with the model)
# imports pkg_resources, which setuptools 82 removed. See requirements.txt.
RUN pip3 install --no-cache-dir --upgrade pip wheel "setuptools<82"

# Install PyTorch with CUDA 12.8 support first (Blackwell / RTX 50 series, sm_120+).
# Before `COPY requirements.txt` so editing a requirement does not invalidate this
# multi-GB layer.
RUN pip3 install --no-cache-dir "torch==${TORCH_VERSION}" "torchaudio==${TORCHAUDIO_VERSION}" \
    --index-url https://download.pytorch.org/whl/cu128

# Fail here, not after several GB of further downloads, if the wheel just installed
# has no Blackwell kernels. torch.cuda.get_arch_list() is empty on a build host
# without a GPU, so read the compiled-in list with _cuda_getArchFlags instead.
RUN python3 -c "import torch; arches = torch._C._cuda_getArchFlags().split(); assert str(torch.version.cuda).startswith('12.8'), 'torch %s is built for CUDA %s, not 12.8' % (torch.__version__, torch.version.cuda); assert 'sm_120' in arches, 'torch %s has no sm_120 kernels (Blackwell): %s' % (torch.__version__, arches)"

# Application dependencies. torch/torchaudio are held by a constraint: if anything
# in requirements.txt ever asks for a different pair, the build fails here instead
# of replacing the cu128 wheels with PyPI ones.
COPY requirements.txt .
RUN printf 'torch==%s\ntorchaudio==%s\n' "${TORCH_VERSION}" "${TORCHAUDIO_VERSION}" > /tmp/torch-constraints.txt \
    && pip3 install --no-cache-dir -c /tmp/torch-constraints.txt -r requirements.txt

# chatterbox itself, WITHOUT its pinned dependency set (requirements.txt lists what
# it needs at runtime).
RUN pip3 install --no-cache-dir --no-deps "chatterbox-tts @ git+${CHATTERBOX_REPO}@${CHATTERBOX_REF}"

# Fail the build, not the first request on the card, if the torch that ended up
# installed (a requirement may have re-resolved it) has no Blackwell kernels ...
RUN python3 -c "import torch; arches = torch._C._cuda_getArchFlags().split(); assert str(torch.version.cuda).startswith('12.8'), 'torch %s is built for CUDA %s, not 12.8' % (torch.__version__, torch.version.cuda); assert 'sm_120' in arches, 'torch %s has no sm_120 kernels (Blackwell): %s' % (torch.__version__, arches)"

# ... or if the chatterbox stack cannot be imported or lacks v3 support. resemble-perth
# hides an ImportError by setting PerthImplicitWatermarker = None, which would only
# surface as "'NoneType' object is not callable" when the model first loads.
RUN python3 -c "import inspect, perth; assert perth.PerthImplicitWatermarker is not None, 'resemble-perth could not import its watermarker (pkg_resources? torchaudio?)'; from chatterbox.mtl_tts import ChatterboxMultilingualTTS as C; assert 't3_model' in inspect.signature(C.from_pretrained).parameters, 'this CHATTERBOX_REF has no t3_model (v3) support'"

# Copy application code
COPY app.py .
COPY model_lifecycle.py .
# Request-body limit and cross-origin guard, imported by app.py at startup.
COPY body_limit.py .
COPY origin_guard.py .

# Reported by /status, since the package version does not change between refs.
ENV CHATTERBOX_REF=${CHATTERBOX_REF}

EXPOSE 5007

# /health is liveness only: it stays 200 while the model downloads or sits unloaded.
# The first start fetches ~3 GB from Hugging Face, hence the long start period; use
# /ready for "the weights have loaded at least once".
HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
    CMD curl -f http://localhost:5007/health || exit 1

# keep-alive above uvicorn's 5 s default so a gateway reusing connections between
# requests does not race the server closing them.
CMD ["python3", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5007", "--timeout-keep-alive", "120"]
