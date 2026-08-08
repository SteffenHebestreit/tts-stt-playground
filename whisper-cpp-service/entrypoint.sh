#!/bin/bash
set -e

MODEL="${WHISPER_MODEL:-large-v3}"
MODEL_FILE="/models/ggml-${MODEL}.bin"
HF_BASE="https://huggingface.co/ggerganov/whisper.cpp/resolve/main"

# Download model if not present.
#
# `--fail` is load-bearing. Without it, a bad WHISPER_MODEL name gets a 404,
# curl still exits 0, and the HTML error page is written to
# /models/ggml-<name>.bin on a NAMED VOLUME. The `[ ! -f ]` guard above then
# skips the download forever, so a single typo bricks the service until someone
# manually deletes the volume. Delete the partial file on any failure.
if [ ! -f "$MODEL_FILE" ]; then
    echo "[whisper-cpp] Model '$MODEL' not found at $MODEL_FILE — downloading..."
    mkdir -p /models
    if ! curl -fL --progress-bar "${HF_BASE}/ggml-${MODEL}.bin" -o "$MODEL_FILE"; then
        rm -f "$MODEL_FILE"
        echo "[whisper-cpp] ERROR: could not download model '$MODEL'." >&2
        echo "[whisper-cpp] Check the name. Note that quantised builds are NOT" >&2
        echo "[whisper-cpp] published for every size: 'small' and 'base' have" >&2
        echo "[whisper-cpp] q5_1 and q8_0 but no q5_0; q5_0 exists for" >&2
        echo "[whisper-cpp] large-v3-turbo. See https://huggingface.co/ggerganov/whisper.cpp" >&2
        exit 1
    fi
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
