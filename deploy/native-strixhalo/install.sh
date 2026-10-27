#!/usr/bin/env bash
# Install the native speech stack on a Strix Halo host (no Docker, no sudo):
# whisper.cpp with Vulkan, the Whisper and VAD models, Piper in a venv, two
# voices, and the OpenAI Speech shim. Idempotent: re-running skips what exists.
#
#   bash deploy/native-strixhalo/install.sh
#
# Needs: git, cmake, a C++ compiler, the Vulkan headers and glslc (all present
# where llama.cpp was built with GGML_VULKAN), python3 with venv, curl.
# Afterwards add llama-swap.speech.yaml to the llama-swap config (README.md).
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
WHISPER_DIR="${WHISPER_DIR:-$HOME/whisper.cpp}"
MODELS="${MODELS:-$HOME/models}"
VENV="${VENV:-$HOME/.venv-piper}"
VOICES="${VOICES:-de_DE-thorsten-high en_US-lessac-high}"
HF=https://huggingface.co

if [ ! -x "$WHISPER_DIR/build/bin/whisper-server" ]; then
  [ -d "$WHISPER_DIR" ] || git clone --depth 1 https://github.com/ggml-org/whisper.cpp.git "$WHISPER_DIR"
  # nice: the build must not slow down the LLMs on the same host
  nice -n 10 cmake -S "$WHISPER_DIR" -B "$WHISPER_DIR/build" -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release -DWHISPER_BUILD_TESTS=OFF
  nice -n 10 cmake --build "$WHISPER_DIR/build" -j "$(nproc)" --config Release --target whisper-server whisper-cli
fi

mkdir -p "$MODELS/whisper" "$MODELS/piper"
fetch() {  # url dest: download once, atomically
  [ -s "$2" ] || { curl -fsSL -o "$2.part" "$1" && mv "$2.part" "$2"; }
}
fetch "$HF/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo-q8_0.bin" "$MODELS/whisper/ggml-large-v3-turbo-q8_0.bin"
fetch "$HF/ggml-org/whisper-vad/resolve/main/ggml-silero-v5.1.2.bin" "$MODELS/whisper/ggml-silero-v5.1.2.bin"

[ -x "$VENV/bin/python" ] || python3 -m venv "$VENV"
"$VENV/bin/pip" install -q --upgrade pip
# lameenc: mp3 (the OpenAI default format) without ffmpeg; soxr: the 24 kHz pcm format
"$VENV/bin/pip" install -q piper-tts lameenc soxr
# shellcheck disable=SC2086
"$VENV/bin/python" -m piper.download_voices --data-dir "$MODELS/piper" $VOICES

mkdir -p "$HOME/bin"
install -m 0755 "$HERE/piper_openai_shim.py" "$HOME/bin/piper_openai_shim.py"
echo "done. Next: add $HERE/llama-swap.speech.yaml to ~/.config/llama-swap/config.yaml"
