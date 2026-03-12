#!/bin/bash
set -e

MODEL="${WHISPER_MODEL:-large-v3}"
MODEL_FILE="/models/ggml-${MODEL}.bin"
HF_BASE="https://huggingface.co/ggerganov/whisper.cpp/resolve/main"

# Download model if not present
if [ ! -f "$MODEL_FILE" ]; then
    echo "[whisper-cpp] Model '$MODEL' not found at $MODEL_FILE — downloading..."
    mkdir -p /models
    curl -L --progress-bar \
        "${HF_BASE}/ggml-${MODEL}.bin" \
        -o "$MODEL_FILE"
    echo "[whisper-cpp] Download complete."
fi

echo "[whisper-cpp] Starting server — model: $MODEL, port: 8080"

VULKAN_ARGS=""
if [ -n "${GGML_VULKAN_DEVICE+x}" ]; then
    VULKAN_ARGS="--device ${GGML_VULKAN_DEVICE}"
fi

exec whisper-server \
    --model "$MODEL_FILE" \
    --host 0.0.0.0 \
    --port 8080 \
    ${VULKAN_ARGS} \
    ${EXTRA_ARGS:-}
