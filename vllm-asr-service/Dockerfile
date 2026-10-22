# vllm-asr-service: the playground's stt-form-v1 adapter in front of `vllm serve`.
#
# The base is the exact vLLM release Steffen's luna-stt app runs on the TrueNAS card
# (RTX 5060 Ti, Blackwell, driver 580.173), so its CUDA 13 build is proven on
# Blackwell; it needs a driver >= 580. vLLM, torch, fastapi, uvicorn, httpx,
# huggingface_hub, numpy, python-multipart and ffmpeg all come with it: this image
# adds the Silero VAD and the adapter's own code, nothing else.
#
# The base is about 9 GB to download and 32 GB unpacked, more than a GitHub runner has
# free, so CI does not build this image and the publish workflow does not push it
# (scripts/plan_publish.py, LOCAL_BUILD_ONLY). It is built on the host that runs it,
# where the base is usually there already: `docker compose --profile vllm-asr build`,
# or on the NAS `docker build -t ghcr.io/steffenhebestreit/tts-stt-vllm-asr:local
# <checkout>/vllm-asr-service`. tests/test_docker_image_contents.py checks statically
# that every module app.py imports is copied below.
FROM vllm/vllm-openai:v0.31.0

WORKDIR /app

# Silero VAD for the pause-based chunking and the non-speech gate (transformers_asr.py).
# 6.x ships its model inside the wheel and needs only torch and packaging, both in the
# base: --no-deps, so pip can never swap the engine's torch for another build. The
# import check fails the build, not the first request, if that ever stops holding.
# PIP_NO_CACHE_DIR is an ARG, not in the running container: /root/.cache is the
# model-cache volume.
COPY requirements.txt .
ARG PIP_NO_CACHE_DIR=1
RUN pip install --no-cache-dir --no-deps -r requirements.txt && \
    python3 -c "from silero_vad import load_silero_vad; load_silero_vad(); import torch; assert not torch.cuda.is_initialized()" && \
    rm -rf /root/.cache

# Copy application code
COPY app.py .
COPY vllm_engine.py .
COPY vllm_support.py .
COPY vad_pool.py .
COPY audio_pipeline.py .
COPY transformers_asr.py .
COPY cohere_support.py .
COPY granite_support.py .
COPY model_lifecycle.py .
# Request-body limit and cross-origin guard, imported by app.py at startup.
COPY body_limit.py .
COPY origin_guard.py .

# The checkpoint lands in /root/.cache/huggingface and vLLM's compile cache in
# /root/.cache/vllm (both on the vllm-asr-cache volume, so a restart reuses them).
EXPOSE 5011

# start-period covers a first start; /health answers at once anyway, the engine
# starts in the background (/ready says when it is up).
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:5011/health || exit 1

# The base's entrypoint is `vllm serve`; this container runs the adapter, which
# starts the engine itself. 120 s keep-alive: the gateway pools connections.
ENTRYPOINT []
CMD ["python3", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5011", "--timeout-keep-alive", "120"]
