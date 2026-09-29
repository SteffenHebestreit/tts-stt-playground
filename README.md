# TTS-STT Platform

A self-hosted, Docker-based platform for text-to-speech synthesis, speech-to-text transcription, custom voice training, and voice cloning. German first: every default that has a language defaults to German.

## Services

| Service | Port | Description |
|---------|------|-------------|
| **Frontend** | 3000 | Web UI, OpenAI-compatible API (`/v1`) and API documentation hub |
| **PiperTTS** | 5000 | Text-to-speech with 40+ voices and custom model support (CPU) |
| **STT (faster-whisper)** | 5001 | Speech-to-text — Python, CUDA/ROCm/CPU |
| **whisper-cpp** | 5003 | Optional speech-to-text — C++, Vulkan/CPU, OpenAI-compatible API |
| **Piper Training** | 8080 | VITS neural network voice training pipeline (opt-in profile `training`) |
| **Qwen3-TTS** | 5004 | Voice cloning and multilingual TTS |
| **Qwen3-ASR** | 5002 | Fast multilingual speech recognition |
| **Parakeet-ASR** | 5005 | Optional realtime STT — 25 EU languages (NeMo) |
| **Canary-ASR** | 5006 | Optional fastest STT — en/de/es/fr with punctuation (NeMo) |
| **Chatterbox-TTS** | 5007 | Optional multilingual TTS + voice cloning (MIT, watermarked) |

Only the **Frontend** is meant to be reached over the network: it proxies every backend call, the live
microphone WebSocket included. The backend ports are published on `127.0.0.1` only (for `curl`, benchmarks
and scripts on the host itself), because the backends have no authentication. Set `BACKEND_BIND_ADDR=0.0.0.0`
to reach them from other machines, or `FRONTEND_BIND_ADDR=127.0.0.1` to keep the UI local behind a reverse proxy.

## Quick Start

The default stack expects an NVIDIA GPU (the GPU services reserve one device); without one, use the
CPU-capable services shown below.

```bash
cp .env.example .env                        # every setting is documented there; the defaults work
docker compose --profile all up -d          # default full stack: UI, Piper, Whisper, Qwen3-ASR, Qwen3-TTS (NVIDIA GPU)

# No NVIDIA GPU: CPU-capable services only
docker compose --profile piper-tts --profile frontend up -d            # TTS and the UI
ENABLE_WHISPER_CPP=true docker compose --profile whisper-cpp --profile frontend up -d  # whisper.cpp surfaced in UI

# Optional extras (heavy images, off by default; the UI also needs the matching ENABLE_* flag)
ENABLE_WHISPER_CPP=true docker compose --profile all --profile whisper-cpp up -d       # plus whisper.cpp
ENABLE_CHATTERBOX_TTS=true docker compose --profile all --profile chatterbox-tts up -d # plus Chatterbox

# Check status
docker compose ps
docker compose logs -f
curl -fsS http://127.0.0.1:5001/ready       # is the Whisper model loaded? 503 with a reason until it is

# Stop everything
docker compose down
```

Open **http://localhost:3000** for the web interface. The first start downloads the models in the
background (several GB), so containers report healthy long before they can transcribe or speak; the UI
shows each backend's state, and `GET /ready` on a backend says whether it can serve a request yet.

`whisper-cpp`, Parakeet, Canary and Chatterbox are opt-in. Starting a profile alone is not enough for the
service to appear in the frontend; set the matching `ENABLE_WHISPER_CPP`, `ENABLE_PARAKEET_ASR`,
`ENABLE_CANARY_ASR` or `ENABLE_CHATTERBOX_TTS=true` for the frontend service as well.

Images are pulled from GHCR as `ghcr.io/steffenhebestreit/tts-stt-*:${IMAGE_TAG}` (`latest` by default, which
follows the newest release or master build). Pin `IMAGE_TAG=X.Y.Z` where a deployment has to be reproducible,
and `PULL_POLICY=always` together with `latest` to fetch a moved tag on `up`.

## Developer Docs

- [docs/api.md](docs/api.md) - the API: `/v1` (OpenAI-compatible), access control, limits, readiness, errors
- [docs/provider-contracts.md](docs/provider-contracts.md) - provider registry schema and API contracts used by the frontend
- [docs/developer-roadmap.md](docs/developer-roadmap.md) - planned architecture and migration steps for provider-generalization, with the current status
- [docs/device-deployment-matrix.md](docs/device-deployment-matrix.md) - which services and models to run on which device, with ready-made presets in [`deploy/profiles/`](deploy/profiles/)
- [docs/model-and-optimisation-research-2026-08.md](docs/model-and-optimisation-research-2026-08.md) - model and optimisation decisions, with an implementation-status addendum
- [benchmarks/README.md](benchmarks/README.md) - measure German WER and latency of a backend on your own audio, and compare Whisper, Parakeet, Qwen3-ASR and a German fine-tune

---

## GPU Setup

Different services support different GPU backends. Choose based on your hardware and host OS.

### Which backend does each service use?

| Service | CPU | NVIDIA CUDA | AMD ROCm | Vulkan |
|---------|:---:|:-----------:|:--------:|:------:|
| STT (faster-whisper) | ✓ | ✓ | ✓ | — |
| whisper-cpp | ✓ | — | — | ✓ |
| Piper Training | ✓ (slow) | ✓ | ✓ | — |
| Qwen3-ASR | ✓ (slow) | ✓ | ✓ | — |
| Qwen3-TTS | ✓ (slow) | ✓ | ✓ | — |
| Parakeet-ASR | ✓ (slow) | ✓ | ✓ (NeMo 2.x image) | — |
| Canary-ASR | ✓ (slow) | ✓ | — | — |
| Chatterbox-TTS | ✓ (slow) | ✓ | — | — |
| PiperTTS | CPU only | — | — | — |

The CUDA images are built for Blackwell (RTX 50 series, sm_120) as well as older cards: CUDA 12.8 and
PyTorch cu128 wheels, so the NVIDIA driver must report CUDA 12.8 or newer (R570+).

### Which GPU works in which environment?

| Environment | NVIDIA CUDA | AMD ROCm | Vulkan |
|-------------|:-----------:|:--------:|:------:|
| **Native Linux** | ✓ | ✓ | ✓ |
| **WSL2 + Docker Engine** (no Desktop) | ✓ | — | ✓ |
| **WSL2 + Docker Desktop** (Windows) | ✓ | — | — |
| **Docker Desktop** (Windows, no WSL2) | ✓ | — | — |

> ROCm requires `/dev/kfd` which is only available on native Linux (bare metal or passthrough VM).
> Vulkan requires `/dev/dri` which Docker Desktop's internal VM does not expose. It works with Docker Engine running directly inside WSL2.

---

### NVIDIA CUDA

**Works on:** native Linux · WSL2 + Docker Engine · WSL2 + Docker Desktop · Docker Desktop (Windows)

**Prerequisites:**
- NVIDIA drivers installed on the host
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) (`nvidia-docker2` / `nvidia-container-toolkit`)

**Verify:**
```bash
nvidia-smi                             # host check
docker run --rm --gpus all nvidia/cuda:12-base nvidia-smi  # container check
```

**Run (default — no overlay needed):**
```bash
docker compose --profile all up -d
# or specific services:
docker compose --profile stt up -d
docker compose --profile qwen3-asr up -d
docker compose --profile training up -d
ENABLE_WHISPER_CPP=true docker compose --profile all --profile whisper-cpp up -d
```

The base `docker-compose.yml` already includes NVIDIA device reservations (`deploy.resources.reservations.devices`).

**WSL2 + Docker Desktop:** Enable GPU support in Docker Desktop → Settings → Resources → GPU. NVIDIA's WSL2 driver handles CUDA passthrough automatically.

---

### AMD ROCm

**Works on:** native Linux only

**Prerequisites (native Linux):**
- [ROCm 6.2+](https://rocm.docs.amd.com/en/latest/deploy/linux/index.html) installed
- User added to `video` and `render` groups: `sudo usermod -aG video,render $USER`
- Verify: `rocminfo | grep -i "gfx\|gpu"`

**Run:**
```bash
docker compose -f docker-compose.yml -f docker-compose.rocm.yml --profile all up -d

# Single service:
docker compose -f docker-compose.yml -f docker-compose.rocm.yml --profile qwen3-asr up -d
```

The ROCm overlay (`docker-compose.rocm.yml`) replaces CUDA-based images with ROCm variants, removes NVIDIA device reservations, and mounts `/dev/kfd` + `/dev/dri`.

**Strix Halo (gfx1151 / RDNA 3.5 / Radeon 8060S):** gfx1151 is supported by current ROCm. The overlay defaults `HSA_OVERRIDE_GFX_VERSION=11.0.0` only because the `Dockerfile.rocm` images pin the rocm6.2 PyTorch wheel index, whose wheels carry no gfx1151 kernels; rocm7.0+ wheels do, so clear the override when those base images are bumped. `PYTORCH_ALLOC_CONF=expandable_segments:True` is set automatically for the unified LPDDR5X pool. See `docker-compose.rocm.yml` for the full gfx target table.

> **WSL2 limitation:** `/dev/kfd` is not exposed by Docker Desktop's VM, nor by the WSL2 GPU-PV driver. ROCm requires bare-metal Linux or a Linux VM with direct PCIe GPU passthrough.

---

### Vulkan (whisper-cpp only)

**Works on:** native Linux · WSL2 + Docker Engine (not Docker Desktop)

Vulkan is supported only by `whisper-cpp` (GGML Vulkan backend). It works on any Vulkan 1.2+ GPU — AMD RDNA, Intel Arc, NVIDIA — without needing CUDA or ROCm drivers.

**Why not for Python services?** PyTorch and CTranslate2 (used by `stt-service`, `qwen3-asr`, etc.) have no Vulkan backend.

**Prerequisites:**
- GPU with Vulkan 1.2+ support
- `/dev/dri` accessible from the container (see environment notes below)
- Verify: `vulkaninfo --summary` (install `vulkan-tools` if needed)

**Run:**
```bash
docker compose -f docker-compose.yml -f docker-compose.vulkan.yml --profile whisper-cpp up -d
```

The Vulkan overlay (`docker-compose.vulkan.yml`) replaces the CPU `Dockerfile` with `Dockerfile.vulkan` (compiled with `-DGGML_VULKAN=1`), mounts `/dev/dri`, and sets `GGML_VULKAN_DEVICE=0`.

**Strix Halo (gfx1151 / Radeon 8060S):** No GFX version override needed — Vulkan requires no per-target kernel compilation, which is what makes it the portable AMD path. Set `GGML_VULKAN_DEVICE=1` if you have a discrete GPU and want device index 1 instead.

#### Vulkan on WSL2 with Docker Engine (no Docker Desktop)

WSL2's GPU-PV driver exposes the GPU as `/dev/dri/renderD128` inside the WSL2 instance. If Docker Engine is installed directly in WSL2 (not Docker Desktop), containers can access it:

```bash
# Inside your WSL2 Ubuntu distro:
sudo apt-get install docker.io
sudo usermod -aG docker $USER
newgrp docker

# Verify GPU is visible:
ls /dev/dri/          # should show card0 and renderD128
vulkaninfo --summary  # should list your GPU

# Then run the Vulkan compose from WSL2:
docker compose -f docker-compose.yml -f docker-compose.vulkan.yml --profile whisper-cpp up -d
```

> **Docker Desktop on Windows:** Docker Desktop runs containers inside its own LinuxKit VM, which does **not** expose `/dev/dri`. The Vulkan overlay will fail with "no such file or directory" in this environment. Use the CPU build instead (default), or switch to Docker Engine in WSL2.

---

### CPU (no GPU)

**Works on:** all environments — native Linux, WSL2, Docker Desktop, Windows

No overlay needed. All services fall back to CPU automatically when no GPU is detected. Performance varies:

| Service | CPU performance |
|---------|----------------|
| PiperTTS | Fast (real-time) |
| STT (faster-whisper) | Acceptable for small/medium models |
| whisper-cpp | Acceptable — optimized C++ inference |
| Qwen3-ASR | Very slow — not recommended for production |
| Qwen3-TTS | Very slow — not recommended for production |
| Piper Training | Slow — hours per epoch |

Use smaller models (`WHISPER_MODEL_SIZE=small`, `WHISPER_MODEL=small`) to improve CPU throughput.

---

## Features

### Text-to-Speech (PiperTTS)
- 40+ pre-trained voices across English, German, French, Spanish, Italian, Dutch
- Intelligent voice selection by language, quality, and gender
- Custom trained model support (ONNX format)
- Adjustable speech speed

### Speech-to-Text — faster-whisper (`stt-service`)
- Multiple model sizes: tiny, base, small, medium, large-v3 (all multilingual)
- `distil-large-v3` available for English-only use cases (fastest) — **do not use for non-English**
- Configurable VAD filter (`vad_filter`, `vad_threshold`) — disable if short clips are being rejected
- Streaming transcription via SSE (`/transcribe-stream`)
- Language detection (`/detect_language`)
- `/health` returns `multilingual` flag to avoid accidentally using an English-only model
- CUDA/ROCm/CPU; set `FORCE_ACCELERATION=cuda|rocm|cpu`

### Speech-to-Text — whisper-cpp
- C++ implementation — lower memory footprint than Python/PyTorch stack
- GGUF quantized models (smaller downloads, faster inference)
- OpenAI-compatible `/v1/audio/transcriptions` endpoint
- CPU + Vulkan GPU backends; no CUDA or ROCm required
- Model downloaded automatically from HuggingFace on first start

### Voice Training (Piper Training)
- VITS neural network architecture
- Automatic audio preprocessing and transcription via STT service
- FP32 training with automatic batch size adjustment on OOM
- Checkpoint saving and recovery after each epoch
- ONNX model export for PiperTTS deployment

### Voice Cloning (Qwen3-TTS)
- Upload a short voice sample (3–10 seconds) and generate speech in that voice
- Cross-lingual cloning across 13 languages

### Speech Recognition (Qwen3-ASR)
- Fast multilingual speech recognition via Qwen3-ASR-1.7B
- Long recordings are cut into pieces at the quietest point near `QWEN3_ASR_CHUNK_S` seconds (30 by default), so nothing is silently truncated
- Requires GPU for practical use; CPU inference is very slow (~60s per minute of audio vs ~2s on GPU)

### Speech Recognition — Parakeet and Canary (NeMo, optional)
- Parakeet (`parakeet-tdt-0.6b-v3`, 25 European languages incl. German) and Canary (`canary-180m-flash`, en/de/es/fr with punctuation)
- Images default to NeMo 3.0.x on torch 2.11 (cu128), with a one-argument rollback to NeMo 2.x (`NEMO_TOOLKIT_SPEC`, or the prebuilt `-nemo2` images via `NEMO_IMAGE_SUFFIX=-nemo2`)
- A local `.nemo` checkpoint (for example a German fine-tune) can be used through `PARAKEET_ASR_MODEL` / `CANARY_ASR_MODEL`
- Opt-in: profile `parakeet-asr` / `canary-asr` plus `ENABLE_PARAKEET_ASR` / `ENABLE_CANARY_ASR` on the frontend

### Text-to-Speech — Chatterbox (optional)
- Multilingual TTS and voice cloning (MIT licence; every file carries a PerTh watermark)
- Long texts are generated in chunks (up to `CHATTERBOX_MAX_TEXT_CHARS`, 5000, then HTTP 413) and can be streamed sentence by sentence (`/tts-stream`)
- The image installs the multilingual v3 checkpoint's code from a pinned GitHub commit; the PyPI release has no v3
- Opt-in: profile `chatterbox-tts` plus `ENABLE_CHATTERBOX_TTS` on the frontend

### Model memory
Every GPU model is released after `MODEL_TTL` seconds without use (300 by default; per-service `STT_MODEL_TTL`,
`ASR_MODEL_TTL`, `TTS_MODEL_TTL` override it, `-1` keeps a model loaded) and reloads on the next request, so several
multi-GB models can share one card. `POST /api/providers/<id>/unload` frees one immediately.

---

## API Endpoints

The endpoints below are the backends' own (ports as in the service table, published on `127.0.0.1` by
default). Use the gateway on port 3000 for anything beyond a local check: its OpenAI-compatible `/v1`
surface and the `/api/*` routes are described in [docs/api.md](docs/api.md), and each spec is
served at `/static/openapi/{gateway,stt,pipertts,training}.json`.

Every model service has `GET /health` (liveness: 200 while a model loads or is idle-unloaded) and
`GET /ready` (readiness: 503 while the first load runs or after it failed, with the reason; it never loads a model).
Uploads and recordings beyond a service's limit answer 413.

### PiperTTS (port 5000)
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/tts` | Generate speech (JSON: `text`, `language`, `quality`, `gender`, `speed`); no language means German |
| POST | `/synthesize` | Generate speech with a specific voice |
| POST | `/upload_model` | Upload a custom ONNX model |
| GET | `/voices` | List the installed voices |
| POST | `/refresh_voices` | Rescan the models directory (a voice file copied in is picked up without a restart) |
| GET | `/ready` | 503 while no voice is installed |
| GET | `/health` | Health check |

Adding a Piper voice: copy its `.onnx` and `.onnx.json` files into `<models dir>/default`. A fresh install is
seeded with the voices baked into the image by the one-shot `piper-voices-seed` service; `PIPER_VOICES` (a
build argument) chooses which voices a custom image ships.

### STT — faster-whisper (port 5001)
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/transcribe` | Transcribe audio (`audio`, `task`, `language`, `beam_size`, `vad_filter`, `vad_threshold`, `no_speech_threshold`) |
| POST | `/transcribe-stream` | SSE streaming transcription (same params) |
| WS | `/ws/transcribe` | Live transcription: stream PCM16 mono 16 kHz frames, receive partial (confirmed/pending) and final transcripts; `{"language": "de"}` config frame, `{"event": "stop"}` to finish |
| POST | `/detect_language` | Detect spoken language |
| POST | `/unload` | Free the model now (409 while a request is using it) |
| GET | `/ready` | Readiness: 503 until the model is loaded (200 again after an idle unload) |
| GET | `/health` | Health — includes `multilingual`, `model_size`, and the device it actually runs on |
| GET | `/models` | List models with size and multilingual flag |
| GET | `/tasks` | Describe `transcribe` vs `translate` |

> Use a multilingual model (`medium`, `large-v3`) for non-English languages. `distil-large-v3` is English-only and silently outputs English regardless of the `language` parameter.

### STT — whisper-cpp (port 5003)
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/inference` | Native whisper-server endpoint (`file`, `language`, `response_format`) — used by the frontend gateway |
| POST | `/v1/audio/transcriptions` | OpenAI-compatible route — only on whisper.cpp builds that include it (current master does not) |

This backend is optional and is only shown in the frontend when `ENABLE_WHISPER_CPP=true` is set for `frontend-service`.

### Piper Training (port 8080)
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/ready` | 503 with the reasons a job would fail (unwritable data directories, a PyTorch build without kernels for the GPU, low disk) |
| POST | `/train` | Start training (`model_name`, `language`, `audio_files`); one job runs at a time (`TRAINING_MAX_CONCURRENT`), a second is 409 |
| POST | `/resume-training` | Resume interrupted training from checkpoint |
| POST | `/train-from-dataset` | Train from an already-prepared dataset |
| POST | `/retrain-from-segments` | Re-transcribe existing clips and retrain |
| POST | `/prepare-dataset` | Create dataset from STT segments |
| POST | `/generate-missing-mels` | Regenerate missing mel spectrograms |
| POST | `/export/{job_id}` | Export model to ONNX |
| GET | `/status/{job_id}` | Training progress |
| GET | `/jobs` | List all training jobs |
| GET | `/download/{job_id}` | Download trained model |
| DELETE | `/job/{job_id}` | Cancel training job |
| DELETE | `/model/{job_id}` | Delete trained model and data |

### Qwen3-TTS (port 5004)
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/tts` | Generate speech with built-in speaker |
| POST | `/clone` | Clone voice from audio sample |
| POST | `/clone-with-ref-text` | Clone voice with manual reference transcript |
| POST | `/voice_design` | Design voice via text description |
| POST | `/voices/save` | Save a voice profile from reference audio |
| POST | `/voices/{voice_id}/tts` | Synthesise with a saved voice |
| DELETE | `/voices/{voice_id}` | Delete a saved voice |
| GET | `/voices` | List saved voice profiles |
| GET | `/speakers` | List built-in speakers |
| GET | `/models` | List available model variants |
| POST | `/load_model` | Switch to a different model variant |
| GET | `/status` | Model and GPU status |
| GET | `/ready` | Readiness (503 while the model loads or after a failed load) |
| GET | `/health` | Health check |

`/tts` (built-in speakers) needs a CustomVoice model variant and answers 409 on a Base model; the Base models
clone voices (`/clone`, `/voices/save`). No language means German (`QWEN3_DEFAULT_LANGUAGE`); an unsupported one is a 400.

### Qwen3-ASR (port 5002)
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/transcribe` | Transcribe audio (recordings up to `ASR_MAX_AUDIO_SECONDS`, 7200; long ones are chunked) |
| POST | `/transcribe-batch` | Batch-transcribe multiple files |
| POST | `/detect_language` | Detect language (from the first piece only) |
| GET | `/status` | Detailed GPU and model info |
| GET | `/ready` | Readiness (503 while the model loads or after a failed load) |
| GET | `/health` | Health check |

### Parakeet-ASR (port 5005) and Canary-ASR (port 5006)
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/transcribe` | Transcribe audio (`audio`, `language`; Canary answers 422 for a language its model cannot decode) |
| POST | `/transcribe-batch` | Batch-transcribe multiple files, with a per-file `status` for failures |
| GET | `/status` | Model info; `runtime` names the NeMo, torch and CUDA versions the image was built with |
| GET | `/ready` | Readiness (503 while the model loads or after a failed load) |
| GET | `/health` | Health check |

### Chatterbox-TTS (port 5007)
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/tts` | Generate speech (long text is chunked; `X-Chunk-Count` reports how many pieces) |
| POST | `/tts-stream` | The same, streamed sentence by sentence |
| POST | `/clone`, `/clone-with-ref-text` | Clone a voice from a reference clip (only the first `CHATTERBOX_REF_MAX_SECONDS` are used) |
| GET | `/status` | Loaded checkpoint (`t3_model`), Hugging Face revision, image build ref |
| GET | `/ready` | Readiness (503 while the model loads or after a failed load) |
| GET | `/health` | Health check |

---

## Configuration

Copy `.env.example` to `.env` and adjust as needed. The file documents every variable (over 130), including
the per-service limits; the ones most people touch:

| Variable | Default | Description |
|----------|---------|-------------|
| `IMAGE_TAG` | `latest` | Image release to run; pin `X.Y.Z` for a reproducible deployment |
| `PULL_POLICY` | `missing` | `always` pulls on every `up` (needed to follow a moved tag such as `latest`) |
| `APP_DATA_DIR` | (empty: `./`) | One directory (for example a TrueNAS dataset) that holds models, caches, voices and output |
| `BACKEND_BIND_ADDR` | `127.0.0.1` | Interface the backend ports are published on; `0.0.0.0` exposes unauthenticated services |
| `FRONTEND_BIND_ADDR` | `0.0.0.0` | Interface the web UI / API gateway is published on |
| `WHISPER_MODEL_SIZE` | `large-v3-turbo` | faster-whisper model: `tiny` `base` `small` `medium` `large-v3` `large-v3-turbo` `distil-large-v3`, a Hugging Face repo id, or the path of a CTranslate2 export. Turbo is the best latency/accuracy trade for live use but **cannot translate**; unknown names fall back to `large-v3` |
| `WHISPER_MODEL` | `large-v3-turbo-q5_0` | whisper-cpp GGML model: `tiny` `base` `small` `medium` `large-v3` `large-v3-turbo`, quantised builds only for some sizes (`large-v3-turbo-q5_0`, `small-q5_1`, ...) |
| `STT_DEFAULT_LANGUAGE` | (empty: auto-detect) | Language `stt-service` uses when a request names none; the web UI sends `auto` itself |
| `FORCE_ACCELERATION` | (empty: auto-detect) | Only `rocm` changes behaviour (faster-whisper on AMD); use `USE_CUDA=false` to force the CPU |
| `WHISPER_CPP_PORT` | `5003` | Host port for whisper-cpp |
| `ENABLE_WHISPER_CPP`, `ENABLE_PARAKEET_ASR`, `ENABLE_CANARY_ASR`, `ENABLE_CHATTERBOX_TTS` | `false` | Expose the optional backend in the frontend provider registry and status checks |
| `DEFAULT_STT_PROVIDER`, `DEFAULT_TTS_PROVIDER` | `whisper`, `piper` | Engine the UI preselects; the ARM64 and Strix Halo presets use `whisper-cpp` because `stt-service` does not run there |
| `GGML_VULKAN_DEVICE` | `0` | Vulkan GPU device index |
| `HIP_VISIBLE_DEVICES` | `0` | ROCm GPU device index |
| `HSA_OVERRIDE_GFX_VERSION` | `11.0.0` | ROCm GFX override (Strix Halo: keep at `11.0.0`) |
| `CUDA_VISIBLE_DEVICES` | `0` | NVIDIA GPU device index |
| `MODEL_TTL` | `300` | Seconds of idleness before a GPU model is unloaded (`-1` never; `STT_MODEL_TTL`, `ASR_MODEL_TTL`, `TTS_MODEL_TTL` override it per group) |
| `QWEN3_TTS_MODEL` | `Qwen/Qwen3-TTS-12Hz-0.6B-Base` | Qwen3 TTS model ID (the 1.7B needs headroom on a small card) |
| `QWEN3_DEFAULT_LANGUAGE`, `PIPER_DEFAULT_LANGUAGE` | `German`, `de` | Language TTS uses when a request names none |
| `QWEN3_ASR_MODEL` | `Qwen/Qwen3-ASR-1.7B` | Qwen3 ASR model ID |
| `ALLOWED_ORIGINS` | (empty: same origin only) | Extra origins that may call the gateway from a browser; `*` opens it to any page. Behind a reverse proxy that rewrites `Host`, set `TRUSTED_ORIGINS` |
| `API_KEY` | (empty: open) | Optional shared secret required as `Authorization: Bearer` on `/v1/*` and on state-changing `/api/*` calls |
| `MAX_UPLOAD_MB`, `MAX_TTS_CHARS` | `512`, `20000` | Gateway limits; each backend has its own (`ASR_MAX_UPLOAD_MB`, `PIPER_MAX_UPLOAD_MB`, `CHATTERBOX_MAX_TEXT_CHARS`, ... see `.env.example`) |
| `BACKEND_ALLOWED_ORIGINS` | `*` | CORS for the backends' own ports (only relevant with `BACKEND_BIND_ADDR=0.0.0.0` and browser callers) |
| `ALLOW_CREDENTIALS` | `false` | Enables CORS credentials only when origins are explicit |
| `STT_SERVICE_URL` | `http://stt-service:8000` | STT endpoint used by Piper Training for audio labelling |

---

## Project Structure

```
tts-stt-playground/
├── docker-compose.yml           # Base stack — NVIDIA CUDA or CPU
├── docker-compose.rocm.yml      # AMD ROCm overlay (native Linux only)
├── docker-compose.vulkan.yml    # Vulkan overlay for whisper-cpp (Linux / WSL2+Docker Engine)
├── docker-compose.arm64.yml     # ARM64 / no-GPU overlay (RK3588S): frontend, Piper, whisper-cpp
├── docker-compose.dev.yml       # Bind-mounts service source for local development
├── docker-compose.truenas*.yml  # TrueNAS SCALE: Custom App YAML and the build-from-source overlay
├── .env.example                 # Every setting, documented
├── deploy/profiles/             # Ready-made .env presets per device (RK3588S, 5060 Ti, 4080, Strix Halo)
├── frontend-service/            # Web UI + API gateway (FastAPI + Jinja2)
├── piper-tts-service/           # PiperTTS synthesis
├── piper-training-service/      # VITS training pipeline
├── stt-service/                 # faster-whisper STT (Python, CUDA/ROCm/CPU)
│   ├── app.py
│   ├── Dockerfile               # CUDA / CPU
│   └── Dockerfile.rocm          # AMD ROCm
├── whisper-cpp-service/         # whisper.cpp STT (C++, Vulkan/CPU)
│   ├── entrypoint.sh            # Downloads GGUF model from HuggingFace on first start
│   ├── Dockerfile               # CPU build
│   └── Dockerfile.vulkan        # Vulkan GPU build
├── qwen3-tts-service/           # Qwen3 TTS + voice cloning
├── qwen3-asr-service/           # Qwen3 ASR
├── parakeet-asr-service/        # Parakeet STT (NeMo), optional
├── canary-asr-service/          # Canary STT (NeMo), optional
├── chatterbox-tts-service/      # Chatterbox TTS + cloning, optional
│                                #   each service dir: app.py, Dockerfile (CUDA), Dockerfile.rocm where one exists
├── benchmarks/                  # German ASR evaluation (WER, latency, significance test)
├── scripts/                     # sync_openapi.py, plan_publish.py, check_no_skips.py, scripts/truenas/*
├── tests/                       # pytest suite (unit, config-drift and, opt-in, live-stack tests)
├── docs/                        # API reference, provider contracts, device matrix, TrueNAS guide, research
├── truenas/                     # TrueNAS custom catalog scaffold
└── models/                      # Shared model storage (gitignored)
```

---

## Troubleshooting

### Health checks
`/health` says the process is up; `/ready` says it can serve a request (503 with a reason while the first model
is downloading or loading, or when the load failed). Run these on the Docker host: the backend ports are bound
to `127.0.0.1` unless `BACKEND_BIND_ADDR` says otherwise.
```bash
curl http://localhost:5000/ready    # PiperTTS (503 if no voice is installed)
curl http://localhost:5001/ready    # STT faster-whisper
curl http://localhost:5002/ready    # Qwen3-ASR
curl http://localhost:5004/ready    # Qwen3-TTS
curl http://localhost:5005/ready    # Parakeet-ASR (optional)
curl http://localhost:5006/ready    # Canary-ASR (optional)
curl http://localhost:5007/ready    # Chatterbox-TTS (optional)
curl http://localhost:8080/ready    # Piper Training (names unwritable directories, a wrong PyTorch build, low disk)
curl http://localhost:5003/         # whisper-cpp (any HTTP response = up)
curl http://localhost:3000/health   # Frontend
curl http://localhost:3000/api/health   # every backend's state, as the UI shows it
```
Replace `/ready` with `/health` for the liveness view (what the Docker healthchecks use).

### Common issues

**CUDA not available in container**
```bash
# Verify NVIDIA Container Toolkit is installed
docker run --rm --gpus all nvidia/cuda:12-base nvidia-smi
# If this fails: install nvidia-container-toolkit and restart Docker
```

**ROCm: "No such file or directory: /dev/kfd"**
ROCm requires native Linux. Docker Desktop (WSL2 or Windows) does not expose `/dev/kfd`. Run on bare-metal Linux.

**Vulkan: "no such file or directory: /dev/dri"**
Docker Desktop's VM does not expose `/dev/dri`. Options:
- Use the CPU build (default, no overlay)
- Install Docker Engine directly inside WSL2 (not Docker Desktop)
- Run on native Linux

**Vulkan: "no suitable Vulkan device found"**
```bash
# Check inside the container:
docker exec <container> vulkaninfo --summary
# If empty: the GPU driver is not visible. Verify /dev/dri is mounted and
# the container user has access (render/video groups).
```

**STT translating instead of transcribing**
The `distil-large-v3` model is English-only — it outputs English regardless of the `language` parameter. Switch to `large-v3` or `medium` for multilingual transcription:
```bash
WHISPER_MODEL_SIZE=large-v3   # for stt-service
WHISPER_MODEL=large-v3        # for whisper-cpp
```

**No speech detected / audio rejected by VAD**
The Silero VAD filter can be too aggressive on short clips or compressed audio (webm/opus). Disable or tune it:
```
POST /transcribe
  vad_filter=false          # disable completely
  vad_threshold=0.3         # lower threshold (default: 0.5)
  no_speech_threshold=0.95  # raise silence tolerance (default: 0.6)
```

VAD parameter defaults (applied to all `/transcribe` and `/transcribe-stream` requests):

| Parameter | Default | Effect |
|-----------|---------|--------|
| `vad_filter` | `true` | Enable/disable VAD pre-filtering |
| `vad_threshold` | `0.5` | Speech probability threshold; lower = more permissive |
| `no_speech_threshold` | `0.6` | Segment-level silence threshold; raise to `0.95` for compressed/noisy audio |

For browser recordings (webm/opus), use `no_speech_threshold=0.95` to avoid false silence detection.

**Slow first start**
Models are downloaded on first launch, in the background: containers are healthy before they can serve, and
`/ready` (or the UI's status indicators) shows when a backend is done. Sizes:
- Whisper large-v3-turbo: ~1.6 GB (large-v3: ~3 GB)
- Qwen3-ASR-1.7B: ~3.5 GB
- Qwen3-TTS-0.6B: ~2.5 GB (the 1.7B variant: ~4.5 GB)

**"Backend unreachable" from another machine**
Backend ports are published on `127.0.0.1` only. Use the UI or the API on port 3000, which proxies every
backend; set `BACKEND_BIND_ADDR=0.0.0.0` only on a network you trust.

**Every button in the UI answers 403 behind a reverse proxy**
The proxy rewrites the `Host` header, so the UI's requests look cross-origin. Add the public URL to
`TRUSTED_ORIGINS` (and set `TRUST_PROXY_HEADERS=true` if the proxy overwrites `X-Forwarded-Host`).

**Out of memory**
- Use a smaller model (`small`, `medium`)
- Lower the idle TTL (`MODEL_TTL`) so idle models give their memory back, or free one now with `POST /api/providers/<id>/unload`
- For Qwen3 on ROCm/APU: `PYTORCH_HIP_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512` (set automatically in ROCm overlay)
- For training: reduce batch size

**Port conflicts**
Default ports: 3000, 5000-5007, 8080. Override with env vars: `FRONTEND_PORT`, `PIPER_TTS_PORT`, `STT_PORT`, `QWEN3_ASR_PORT`, `WHISPER_CPP_PORT`, `QWEN3_TTS_PORT`, `PARAKEET_ASR_PORT`, `CANARY_ASR_PORT`, `CHATTERBOX_TTS_PORT`, `TRAINING_PORT`.

---

## Local Development

The base `docker-compose.yml` runs from the built images (deployment-ready). For
development, add the `docker-compose.dev.yml` overlay, which bind-mounts service
source files so code changes apply on container restart — no rebuild needed:

```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml --profile all up -d
docker compose restart stt-service   # pick up an edited file
```

All processing is local. No data is sent to external services.

### Tests, lint and generated files

```bash
pip install -r tests/requirements.txt
pytest tests/                       # ~2 minutes; tests that need a running stack skip themselves
ruff check .                        # syntax errors and undefined names only (ruff.toml)
python scripts/sync_openapi.py      # regenerate the OpenAPI specs after changing a route or model
```

The suite is offline: the heavy dependencies (torch, NeMo, faster-whisper, librosa) are stubbed, so it needs
only the packages in `tests/requirements.txt`. Besides the unit tests it holds config-drift suites that fail
when compose, `.env.example`, the device presets, the Dockerfiles and the OpenAPI specs disagree with each
other (`REQUIRE_DRIFT_SUITES=1`, set in CI, turns a missing dependency into a failure instead of a skip). CI
(`.github/workflows/ci.yml`) runs them on Python 3.11 and 3.12 together with lint, `docker compose config` for
every overlay, actionlint and a JavaScript syntax check; shellcheck, hadolint, a dependency audit and the
training and browser suites report without failing the run. Images are published by
`.github/workflows/publish-images.yml` only after those checks pass: a `vX.Y.Z` tag publishes `X.Y.Z`, `X.Y` and
`latest`, a push to master publishes `latest`, and a manual run never touches `latest` or a version tag.

### Storage location

Persistent data (models, output, caches, training data) defaults to `./` under
the repo. Set `APP_DATA_DIR=/path/to/data` (e.g. a TrueNAS dataset) to relocate
everything with one variable; see `docs/truenas-deployment.md`.

---

## TrueNAS Deployment

Runs on TrueNAS SCALE 24.10 (Electric Eel) or newer as a **Custom App from prebuilt images**: no
clone and no build. All data lives in one dataset and only the web UI port (3000) is published;
the frontend proxies every backend call, live microphone transcription included.
**[`docs/truenas-installation-guide.md`](docs/truenas-installation-guide.md)** is the complete
guide (install, update, roll back, VRAM planning, troubleshooting).

### Install in 5 minutes

1. Create a dataset, for example `/mnt/tank/apps/tts-stt`.
2. Optional, in the TrueNAS shell: `scripts/truenas/preflight.sh --data-dir /mnt/tank/apps/tts-stt`
   checks Docker, the NVIDIA driver, VRAM, disk space and the port (the guide shows how to fetch
   the scripts without cloning the repository).
3. Apps → Discover Apps → Custom App → **Install via YAML**: paste
   [`docker-compose.truenas-app.yml`](docker-compose.truenas-app.yml), replace
   `${APP_DATA_DIR:?set dataset path}` (every occurrence) with your dataset path, install.
4. Open `http://<truenas-ip>:3000`. Models download in the background on the first start.

The images (`ghcr.io/steffenhebestreit/tts-stt-*`) are about 25 GB and TrueNAS aborts an Apps job
after 20 minutes: run `scripts/truenas/pull-images.sh` first on a slow link.

### Updates

The compose file **pins a release** (`IMAGE_TAG`), so nothing changes until you change it.
`scripts/truenas/update-check.sh` is a read-only check that prints `UPDATE AVAILABLE` when a newer
release (or a newer image for a floating tag) is published. To update: snapshot the dataset,
change the release in the app's YAML, Save; to roll back, put the previous release back (data
folders are only added to). `IMAGE_TAG:-latest` with `PULL_POLICY:-always` follows the newest build
instead, and an optional `diun` service sends notifications only. Details are in the guide.

### Other routes

- **Settings file:** the same compose file with a `settings.env` on the dataset, so an update is
  one changed line ([`truenas/custom-app-include.yml`](truenas/custom-app-include.yml)).
- **Custom catalog** (a form instead of YAML): [`truenas/tts-stt/`](truenas/tts-stt/) is a
  scaffold that TrueNAS cannot load as it stands, see [`truenas/README.md`](truenas/README.md).
- **Build from source** on the NAS with `docker-compose.truenas.yml`:
  [`docs/truenas-deployment.md`](docs/truenas-deployment.md).

### Requirements and sizing

- An NVIDIA driver that supports CUDA 12.8 for the GPU services (R570 or newer; RTX 50 series
  cards require it), installed from Apps → Settings → Install NVIDIA Drivers.
- The default set (Whisper `large-v3-turbo`, Qwen3-ASR, Qwen3-TTS 0.6B) is about 8 GB of VRAM
  resident; optional services add more (Canary, Parakeet, Chatterbox, training). Every GPU model is
  freed after `MODEL_TTL` seconds idle (300), so the sum need not fit at once. Budget against what
  `nvidia-smi` reports for your card; the per-service table is in the guide.
- Remote browser access needs nothing beyond port 3000 (`ALLOWED_ORIGINS` stays empty; behind a
  proxy that rewrites `Host` set `TRUSTED_ORIGINS`). The microphone from another machine needs HTTPS.

### Cleanup (optional)

Remove leftover XTTS cache from an earlier version, in the repository checkout (requires root):

```bash
sudo rm -rf cache/tts cache/xtts-cache   # ~3.6 GB
sudo rm -f models/luna.json models/luna.onnx  # old root-level export
```

## API

The stack is usable as a drop-in **OpenAI-compatible speech API** — point the official SDK at
the gateway and it works unchanged across every device:

```python
from openai import OpenAI
client = OpenAI(base_url="http://your-host:3000/v1", api_key="unused")  # your API_KEY if the gateway has one
client.audio.transcriptions.create(model="whisper-1", file=open("audio.wav", "rb")).text
```

`POST /v1/audio/transcriptions` (json, text, verbose_json, srt, vtt) · `POST /v1/audio/speech` (mp3, wav, pcm) · `GET /v1/models`

The same request returns the same response shape whether the backend is whisper-cpp on an ARM
SBC or faster-whisper on a workstation GPU. Full reference, including the language-handling
rules that matter for German: **[`docs/api.md`](docs/api.md)**.

The richer project-native `/api/*` surface (segment timestamps, streaming TTS, voice training)
is unchanged and documented in [`docs/provider-contracts.md`](docs/provider-contracts.md).

