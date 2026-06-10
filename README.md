# TTS-STT Platform

A self-hosted, Docker-based platform for text-to-speech synthesis, speech-to-text transcription, custom voice training, and voice cloning.

## Services

| Service | Port | Description |
|---------|------|-------------|
| **Frontend** | 3000 | Web UI and API documentation hub |
| **PiperTTS** | 5000 | Text-to-speech with 40+ voices and custom model support |
| **STT (faster-whisper)** | 5001 | Speech-to-text — Python, CUDA/ROCm/CPU |
| **whisper-cpp** | 5003 | Optional speech-to-text — C++, Vulkan/CPU, OpenAI-compatible API |
| **Piper Training** | 8080 | VITS neural network voice training pipeline |
| **Qwen3-TTS** | 5004 | Voice cloning and multilingual TTS |
| **Qwen3-ASR** | 5002 | Fast multilingual speech recognition |

## Quick Start

```bash
# CPU-only — works everywhere, no GPU setup required
docker compose --profile stt up -d          # faster-whisper STT
ENABLE_WHISPER_CPP=true docker compose --profile whisper-cpp --profile frontend up -d  # whisper.cpp surfaced in UI
docker compose --profile piper-tts up -d    # TTS
docker compose --profile all up -d          # default full stack (without optional whisper-cpp)
ENABLE_WHISPER_CPP=true docker compose --profile all --profile whisper-cpp up -d  # full stack plus whisper.cpp

# Check status
docker compose ps
docker compose logs -f

# Stop everything
docker compose down
```

Open **http://localhost:3000** for the web interface.

`whisper-cpp` is opt-in. Starting the `whisper-cpp` profile alone is not enough for it to appear in the frontend; set `ENABLE_WHISPER_CPP=true` for the frontend service as well.

## Developer Docs

- [docs/developer-roadmap.md](docs/developer-roadmap.md) - planned architecture and migration steps for provider-generalization
- [docs/provider-contracts.md](docs/provider-contracts.md) - provider registry schema and API contracts used by the frontend

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
| PiperTTS | CPU only | — | — | — |

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

**Strix Halo (gfx1201 / RDNA 4):** gfx1201 is not in ROCm 6.2's official support list. The overlay defaults `HSA_OVERRIDE_GFX_VERSION=11.0.0` to make ROCm treat it as gfx1100 (RDNA 3), which runs correctly on RDNA 4. `PYTORCH_HIP_ALLOC_CONF=expandable_segments:True` is also set automatically for the unified LPDDR5x memory pool.

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

**Strix Halo (gfx1201 / Radeon 890M):** No GFX version override needed — the Radeon 890M is natively Vulkan 1.3 capable. Set `GGML_VULKAN_DEVICE=1` if you have a discrete GPU and want to use device index 1 instead.

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
- Requires GPU for practical use; CPU inference is very slow (~60s per minute of audio vs ~2s on GPU)

---

## API Endpoints

### PiperTTS (port 5000)
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/tts` | Generate speech (JSON: `text`, `language`, `quality`, `gender`, `speed`) |
| POST | `/synthesize` | Generate speech with a specific voice |
| POST | `/upload_model` | Upload a custom ONNX model |
| GET | `/voices` | List available voices |
| GET | `/health` | Health check |

### STT — faster-whisper (port 5001)
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/transcribe` | Transcribe audio (`audio`, `task`, `language`, `beam_size`, `vad_filter`, `vad_threshold`, `no_speech_threshold`) |
| POST | `/transcribe-stream` | SSE streaming transcription (same params) |
| POST | `/detect_language` | Detect spoken language |
| GET | `/health` | Health — includes `multilingual` and `model_size` |
| GET | `/models` | List models with size and multilingual flag |
| GET | `/tasks` | Describe `transcribe` vs `translate` |

> Use a multilingual model (`medium`, `large-v3`) for non-English languages. `distil-large-v3` is English-only and silently outputs English regardless of the `language` parameter.

### STT — whisper-cpp (port 5003)
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/v1/audio/transcriptions` | Transcribe audio — OpenAI-compatible (`file`, `language`, `response_format`) |
| POST | `/inference` | Native whisper.cpp inference endpoint |

This backend is optional and is only shown in the frontend when `ENABLE_WHISPER_CPP=true` is set for `frontend-service`.

### Piper Training (port 8080)
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/train` | Start training (`model_name`, `language`, `audio_files`) |
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
| GET | `/health` | Health check |

### Qwen3-ASR (port 5002)
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/transcribe` | Transcribe audio |
| POST | `/transcribe-batch` | Batch-transcribe multiple files |
| POST | `/detect_language` | Detect language |
| GET | `/status` | Detailed GPU and model info |
| GET | `/health` | Health check |

---

## Configuration

Copy `.env.example` to `.env` and adjust as needed.

| Variable | Default | Description |
|----------|---------|-------------|
| `WHISPER_MODEL_SIZE` | `small` | faster-whisper model: `tiny` `base` `small` `medium` `large-v3` `distil-large-v3` |
| `WHISPER_MODEL` | `large-v3` | whisper-cpp GGUF model: `tiny` `base` `small` `medium` `large-v3` `large-v3-turbo` |
| `FORCE_ACCELERATION` | `cuda` | faster-whisper backend: `cuda` `rocm` `cpu` |
| `WHISPER_CPP_PORT` | `5003` | Host port for whisper-cpp |
| `ENABLE_WHISPER_CPP` | `false` | Exposes whisper-cpp in the frontend provider registry and status checks |
| `GGML_VULKAN_DEVICE` | `0` | Vulkan GPU device index |
| `HIP_VISIBLE_DEVICES` | `0` | ROCm GPU device index |
| `HSA_OVERRIDE_GFX_VERSION` | `11.0.0` | ROCm GFX override (Strix Halo: keep at `11.0.0`) |
| `CUDA_VISIBLE_DEVICES` | `0` | NVIDIA GPU device index |
| `QWEN3_TTS_MODEL` | `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | Qwen3 TTS model ID |
| `QWEN3_ASR_MODEL` | `Qwen/Qwen3-ASR-1.7B` | Qwen3 ASR model ID |
| `ALLOWED_ORIGINS` | `*` | CORS allowed origins (set explicitly in production) |
| `ALLOW_CREDENTIALS` | `false` | Enables CORS credentials only when origins are explicit |
| `STT_SERVICE_URL` | `http://stt-service:8000` | STT endpoint used by Piper Training for audio labelling |

---

## Project Structure

```
tts-stt-playground/
├── docker-compose.yml           # Base stack — NVIDIA CUDA or CPU
├── docker-compose.rocm.yml      # AMD ROCm overlay (native Linux only)
├── docker-compose.vulkan.yml    # Vulkan overlay for whisper-cpp (Linux / WSL2+Docker Engine)
├── .env.example
├── frontend-service/            # Web UI (FastAPI + Jinja2)
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
│   ├── app.py
│   ├── Dockerfile               # CUDA / CPU
│   └── Dockerfile.rocm          # AMD ROCm
├── qwen3-asr-service/           # Qwen3 ASR
│   ├── app.py
│   ├── Dockerfile               # CUDA / CPU
│   └── Dockerfile.rocm          # AMD ROCm
└── models/                      # Shared model storage (gitignored)
```

---

## Troubleshooting

### Health checks
```bash
curl http://localhost:5000/health   # PiperTTS
curl http://localhost:5001/health   # STT faster-whisper
curl http://localhost:5003/         # whisper-cpp (any HTTP response = up)
curl http://localhost:8080/health   # Piper Training
curl http://localhost:5004/health   # Qwen3-TTS
curl http://localhost:5002/health   # Qwen3-ASR
curl http://localhost:3000/health   # Frontend
```

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
Models are downloaded on first launch. Sizes:
- Whisper large-v3: ~3 GB
- Qwen3-ASR-1.7B: ~3.5 GB
- Qwen3-TTS-1.7B: ~4 GB

**Out of memory**
- Use a smaller model (`small`, `medium`)
- For Qwen3 on ROCm/APU: `PYTORCH_HIP_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512` (set automatically in ROCm overlay)
- For training: reduce batch size

**Port conflicts**
Default ports: 3000, 5000, 5001, 5002, 5003, 5004, 8080. Override with env vars: `FRONTEND_PORT`, `PIPER_TTS_PORT`, `STT_PORT`, `QWEN3_ASR_PORT`, `WHISPER_CPP_PORT`, `QWEN3_TTS_PORT`, `TRAINING_PORT`.

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

### Storage location

Persistent data (models, output, caches, training data) defaults to `./` under
the repo. Set `APP_DATA_DIR=/path/to/data` (e.g. a TrueNAS dataset) to relocate
everything with one variable; see `docs/truenas-deployment.md`.

---

## TrueNAS Deployment (RTX 5060 Ti 16 GB)

This project supports deployment on a dedicated TrueNAS SCALE server with an
NVIDIA RTX 5060 Ti (16 GB VRAM). Use the `docker-compose.truenas.yml` overlay
for optimised GPU memory allocation and single-GPU pinning.

Start from `.env.truenas.example` (copy it to `.env`) and set `APP_DATA_DIR` to a
dataset path. If the frontend is opened from another machine, change
`ALLOWED_ORIGINS` and the `BROWSER_*_URL` values before deployment.

### Install without building (prebuilt images)

CI publishes every service image to GHCR (`ghcr.io/steffenhebestreit/tts-stt-*`),
so you can deploy on TrueNAS **without cloning or building**:

- **Install via YAML** — paste [`docker-compose.truenas-app.yml`](docker-compose.truenas-app.yml)
  into Apps → Custom App, set the data dataset, allocate the GPU, deploy.
- **Custom catalog (point-and-click form)** — add this repo as a community
  catalog; see [`truenas/README.md`](truenas/README.md) and the
  [`truenas/tts-stt/`](truenas/tts-stt/) scaffold (`questions.yaml`).

See [`truenas/README.md`](truenas/README.md) for both paths and their caveats.

For the build-from-source workflow, dataset layout, and profile strategy, see
`docs/truenas-deployment.md`.
For the first rollout, use `docs/truenas-custom-app-checklist.md`.
For always-on versus training-window service recommendations, use
`docs/truenas-service-profiles.md`.

### Prerequisites

1. **TrueNAS SCALE** with Docker/Compose support (Apps → Custom App or shell access)
2. **NVIDIA drivers** installed (TrueNAS SCALE ships them for supported GPUs)
3. **NVIDIA Container Toolkit** configured (`nvidia-ctk runtime configure`)
4. Clone this repo to a dataset: `/mnt/pool/apps/tts-stt`

### VRAM Budget (16 GB)

| Service | Estimated VRAM | Notes |
|---------|---------------:|-------|
| STT (faster-whisper large-v3) | ~3 GB | Fits easily |
| Qwen3-ASR-1.7B | ~4 GB | bfloat16 |
| Qwen3-TTS-0.6B | ~3 GB | Default in `docker-compose.truenas.yml` |
| Piper Training (VITS) | ~2–4 GB | Depends on batch size |

> Running STT + Qwen3-ASR + Qwen3-TTS simultaneously uses roughly 10 to 11 GB,
> leaving
> headroom for training. Avoid running training concurrently with all three
> inference services if you hit OOM.

### Launch

```bash
cd /mnt/pool/apps/tts-stt

# Start from the checked-in defaults and then edit for your host/network
cp .env.example .env

# Full stack with NVIDIA GPU
docker compose --env-file .env -f docker-compose.yml -f docker-compose.truenas.yml --profile all up -d

# Just STT + TTS (low VRAM)
docker compose --env-file .env -f docker-compose.yml -f docker-compose.truenas.yml \
  --profile stt --profile piper-tts up -d
```

### TrueNAS Custom App YAML

If you paste a merged Compose file into Apps -> Discover -> Install via YAML and
want `qwen3-asr-service` plus `qwen3-tts-service` to share the same NVIDIA GPU,
pin both services to GPU `0` with `device_ids: ["0"]` and set
`NVIDIA_VISIBLE_DEVICES=0` plus `CUDA_VISIBLE_DEVICES=0` in each service.
Do not use `count` and `device_ids` together.

### Persistent Storage

Bind-mount a TrueNAS dataset so model caches survive container rebuilds:

```
/mnt/pool/apps/tts-stt/models  → /app/models          (PiperTTS, Training)
/mnt/pool/apps/tts-stt/output  → /app/output           (PiperTTS)
```

Docker named volumes (`stt-cache`, `qwen3-tts-cache`, `qwen3-asr-cache`) are
already defined in the base compose file — they persist automatically on TrueNAS.

### Remote Browser Access

If the frontend is opened from another machine on the network, do not leave the
browser-facing URLs at `localhost`. Set `ALLOWED_ORIGINS` to the frontend origin
and update the `BROWSER_TTS_URL`, `BROWSER_STT_URL`, `BROWSER_TRAINING_URL`,
`BROWSER_QWEN3_ASR_URL`, and `BROWSER_QWEN3_TTS_URL` variables in `.env`.

### Cleanup (optional)

Remove leftover XTTS cache from an earlier version (requires root):

```bash
sudo rm -rf cache/tts cache/xtts-cache   # ~3.6 GB
sudo rm -f models/luna.json models/luna.onnx  # old root-level export
```
