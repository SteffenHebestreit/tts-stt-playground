# TTS-STT Studio

Self-hosted neural **Text-to-Speech** and **Speech-to-Text** in one web UI and one
OpenAI-compatible API, German first. Everything runs locally; no data leaves your server.

- **TTS:** Piper (CPU, 40+ voices), Qwen3-TTS (voice cloning, GPU), optional Chatterbox
  (streaming German TTS with cloning)
- **STT:** Whisper (live microphone transcription), Qwen3-ASR, optional Canary and Parakeet
  (25 European languages), optional whisper.cpp (CPU only)
- **Voice training:** upload recordings, segment, train and export a Piper voice (optional)

## After install

Open the Web UI on the configured port (default **3000**). The first start downloads the
models in the background (several GB): the app is usable while that runs, and the status row
in the UI shows each backend's state.

- All data lives in the dataset you chose under **Storage**.
- Only the Web UI port is published. Backends stay on the internal network.
- Open the UI by IP address, `*.local` or a plain name such as `truenas`. A real domain or a
  reverse proxy needs its name under **Access** (`403 host_not_allowed` otherwise).
- **Image release** defaults to a pinned version: nothing updates behind your back. To update,
  snapshot the dataset, change the release and save. See docs/truenas-installation-guide.md.
- GPU images need an NVIDIA driver that supports CUDA 12.8 (R570 or newer; GeForce cards need
  exactly that). Run `scripts/truenas/preflight.sh` in the TrueNAS shell for a readiness check.
