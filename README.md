# TTS-STT Platform

A self-hosted, Docker-based platform for text-to-speech synthesis, speech-to-text transcription, custom voice training, and voice cloning.

## Services

| Service | Port | Description |
|---------|------|-------------|
| **Frontend** | 3000 | Web UI and API documentation hub |
| **PiperTTS** | 5000 | Text-to-speech with 40+ voices and custom model support |
| **STT (Whisper)** | 5001 | Speech-to-text transcription and audio segmentation |
| **whisper-cpp** | 5003 | OpenAI-compatible STT — CPU or Vulkan GPU via overlay |
| **Piper Training** | 8080 | VITS neural network voice training pipeline |
| **Qwen3-TTS** | 5004 | Voice cloning and multilingual TTS (replaces XTTS) |
| **Qwen3-ASR** | 5002 | Fast multilingual speech recognition |

## Quick Start

Each service has its own profile. Start only what you need:

```bash
# Full stack
docker-compose --profile all up -d

# Single service — dependencies are pulled in automatically
docker-compose --profile qwen3-asr up -d
docker-compose --profile qwen3-tts up -d   # also starts qwen3-asr
docker-compose --profile piper-tts up -d
docker-compose --profile stt up -d
docker-compose --profile training up -d    # also starts stt
docker-compose --profile frontend up -d    # also starts piper-tts + stt

# Check service health
docker-compose ps

# View logs
docker-compose logs -f

# Stop everything
docker-compose down
```

Open **http://localhost:3000** for the web interface or **http://localhost:3000/api-docs** for API documentation.

## Prerequisites

- Docker and Docker Compose
- 16 GB+ RAM recommended
- GPU for training and Qwen3 services (CPU fallback available but slow):
  - **NVIDIA**: CUDA drivers + NVIDIA Container Toolkit (`nvidia-docker2`)
  - **AMD / any Vulkan GPU** (whisper-cpp only): use `docker-compose.vulkan.yml` (see below) — requires native Linux or Docker Engine in WSL2 (not Docker Desktop)
  - **AMD** (all Python services): ROCm 6.2+ — use `docker-compose.rocm.yml` (see below)

## Vulkan GPU (AMD / Intel / NVIDIA)

`whisper-cpp` uses the GGML Vulkan backend — no ROCm, no CUDA required. This is the recommended GPU path on **Strix Halo APUs** and any system running Docker Desktop on WSL2, because Vulkan uses `/dev/dri` which is exposed through WSL2's GPU virtualization layer.

```bash
# Start whisper-cpp with Vulkan acceleration
docker compose -f docker-compose.yml -f docker-compose.vulkan.yml --profile whisper-cpp up -d
```

The overlay replaces the CPU `Dockerfile` with `Dockerfile.vulkan` (built with `-DGGML_VULKAN=1`) and mounts `/dev/dri`. Set `GGML_VULKAN_DEVICE=0` (default) to select the GPU by index.

**Strix Halo (gfx1201 / Radeon 890M):** No GFX version override needed — the Radeon 890M is natively Vulkan 1.3 capable. On APU systems with a discrete GPU as device 1, set `GGML_VULKAN_DEVICE=0` to stay on the iGPU or `=1` for the dGPU.

> **WSL2 + Docker Desktop note:** Docker Desktop's VM does not expose `/dev/dri` to containers, so the Vulkan overlay will fail to start on this setup. Vulkan GPU acceleration requires **native Linux** or **Docker Engine running directly inside WSL2** (not Docker Desktop). On Windows, the CPU build works fine.

> **Why not use Vulkan for the Python services?** PyTorch and CTranslate2 (used by `stt-service`, `qwen3-asr-service`, etc.) do not have Vulkan backends. For those, use ROCm (AMD) or CUDA (NVIDIA).

## AMD GPU (ROCm)

All GPU services have a `Dockerfile.rocm` variant. Use the `docker-compose.rocm.yml` override alongside the main compose file:

```bash
# Full stack on AMD GPU
docker-compose -f docker-compose.yml -f docker-compose.rocm.yml --profile all up -d

# Single service
docker-compose -f docker-compose.yml -f docker-compose.rocm.yml --profile qwen3-asr up -d
```

The overlay switches base images to `rocm/dev-ubuntu-22.04:6.2-complete`, installs ROCm PyTorch wheels, and replaces NVIDIA device reservations with `/dev/kfd` + `/dev/dri` mounts.

**Strix Halo (gfx1201 / RDNA 4 iGPU):** gfx1201 is not in ROCm 6.2's official support list. The compose overlay defaults `HSA_OVERRIDE_GFX_VERSION=11.0.0` which makes ROCm treat it as gfx1100 (RDNA 3) — this works correctly on RDNA 4. The `PYTORCH_HIP_ALLOC_CONF=expandable_segments:True` setting is also applied automatically to prevent fragmentation of the unified LPDDR5x memory pool.

> **WSL2 note:** AMD ROCm GPU passthrough is not available in Docker Desktop's WSL2 backend (`/dev/kfd` is not exposed). The ROCm compose file requires native Linux or a Linux VM with direct PCIe GPU access.

## Features

### Text-to-Speech (PiperTTS)
- 40+ pre-trained voices across English, German, French, Spanish, Italian, Dutch
- Intelligent voice selection by language, quality, and gender
- Custom trained model support (ONNX format)
- Adjustable speech speed

### Speech-to-Text (Whisper)
- Multiple model sizes: tiny, base, small, medium, large-v3
- Automatic language detection
- Audio segmentation for training data preparation

### Voice Training (Piper Training)
- VITS neural network architecture
- Automatic audio preprocessing and transcription via STT service
- FP32 training with automatic batch size adjustment on OOM
- Checkpoint saving and recovery after each epoch
- ONNX model export for PiperTTS deployment
- GPU acceleration with CPU fallback

### Voice Cloning (Qwen3-TTS)
- Upload a short voice sample (3-10 seconds) and generate speech in that voice
- Cross-lingual cloning across 13 languages
- Replaces the previous XTTS service

### Speech Recognition (Whisper STT)
- Multiple model sizes: tiny, base, small, medium, large-v3 (all multilingual)
- `distil-large-v3` available for English-only use cases (fastest)
- Configurable VAD filter (`vad_filter`, `vad_threshold`) — disable if short clips are being filtered out
- Streaming transcription via SSE (`/transcribe-stream`)
- Language detection (`/detect_language`)
- `/health` returns `multilingual` flag to avoid accidentally using an English-only model

### Speech Recognition (Qwen3-ASR)
- Fast multilingual speech recognition via Qwen3-ASR-1.7B
- Language detection
- Requires GPU (CUDA/ROCm) for practical use; CPU inference is very slow

## API Endpoints

### PiperTTS (port 5000)
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/tts` | Generate speech (JSON body: text, language, quality, gender, speed) |
| POST | `/synthesize` | Generate speech with a specific voice |
| POST | `/upload_model` | Upload a custom ONNX model |
| GET | `/voices` | List available voices |
| GET | `/health` | Health check |

### STT (port 5001 -> 8000 internal)
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/transcribe` | Transcribe audio (form: `audio`, `task`, `language`, `beam_size`, `vad_filter`, `vad_threshold`, `no_speech_threshold`) |
| POST | `/transcribe-stream` | SSE streaming transcription (same params as `/transcribe`) |
| POST | `/detect_language` | Detect spoken language |
| GET | `/health` | Health check — includes `multilingual` and `model_size` fields |
| GET | `/models` | List available models with size and multilingual flag |
| GET | `/tasks` | Describe `transcribe` vs `translate` tasks |

**Model selection:** Use a multilingual model (`medium`, `large-v3`) for non-English languages. `distil-large-v3` is English-only and will silently output English regardless of the `language` parameter.

### Piper Training (port 8080)
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/train` | Start training (multipart: model_name, language, audio_files) |
| GET | `/status/{job_id}` | Training progress |
| GET | `/jobs` | List all training jobs |
| POST | `/export/{job_id}` | Export model to ONNX |
| GET | `/download/{job_id}` | Download trained model |
| DELETE | `/job/{job_id}` | Cancel training job |
| DELETE | `/model/{job_id}` | Delete trained model |

### Qwen3-TTS (port 5004)
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/tts` | Generate speech with built-in speaker |
| POST | `/clone` | Clone voice from audio sample |
| GET | `/status` | Model and GPU status |
| GET | `/health` | Health check |

### whisper-cpp (port 5003)

OpenAI-compatible endpoint — drop-in replacement for any client that targets `/v1/audio/transcriptions`.

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/v1/audio/transcriptions` | Transcribe audio (OpenAI-compatible; supports `language`, `response_format`) |
| POST | `/inference` | Native whisper.cpp inference endpoint |

### Qwen3-ASR (port 5002)
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/transcribe` | Transcribe audio |
| POST | `/detect_language` | Detect language |
| GET | `/health` | Health check |

## Configuration

Copy `.env.example` to `.env` and adjust as needed. Key settings:

| Variable | Default | Description |
|----------|---------|-------------|
| `WHISPER_MODEL_SIZE` | small | STT model (stt-service): tiny, base, small, medium, large-v3 |
| `WHISPER_MODEL` | large-v3 | GGUF model for whisper-cpp: tiny, base, small, medium, large-v3, large-v3-turbo |
| `WHISPER_CPP_PORT` | 5003 | Host port for whisper-cpp service |
| `GGML_VULKAN_DEVICE` | 0 | Vulkan GPU device index (0 = first GPU) |
| `CUDA_VISIBLE_DEVICES` | 0 | NVIDIA GPU device index |
| `QWEN3_TTS_MODEL` | Qwen/Qwen3-TTS-12Hz-1.7B-Base | Qwen3 TTS model ID |
| `QWEN3_ASR_MODEL` | Qwen/Qwen3-ASR-1.7B | Qwen3 ASR model ID |
| `ALLOWED_ORIGINS` | * | CORS allowed origins |

## Project Structure

```
tts-stt/
├── docker-compose.yml
├── docker-compose.rocm.yml    # AMD ROCm overlay (all Python services)
├── docker-compose.vulkan.yml  # Vulkan overlay (whisper-cpp only; works in WSL2)
├── .env.example
├── frontend-service/          # Web UI (FastAPI + Jinja2)
│   ├── app.py
│   ├── templates/index.html
│   └── static/
├── piper-tts-service/         # PiperTTS synthesis
│   └── app.py
├── piper-training-service/    # VITS training pipeline
│   ├── app.py
│   ├── training_pipeline.py
│   ├── vits_model.py
│   ├── data_processor.py
│   ├── model_exporter.py
│   ├── dataset.py
│   └── training_utils.py
├── stt-service/               # faster-whisper STT (Python)
│   └── app.py
├── whisper-cpp-service/       # whisper.cpp STT (C++, OpenAI-compat API)
│   ├── Dockerfile             # CPU build
│   ├── Dockerfile.vulkan      # Vulkan GPU build
│   └── entrypoint.sh          # Downloads GGUF model on first start
├── qwen3-tts-service/         # Qwen3 TTS + voice cloning
│   └── app.py
├── qwen3-asr-service/         # Qwen3 ASR
│   └── app.py
└── models/                    # Shared model storage (gitignored)
```

## Troubleshooting

```bash
# Check GPU availability inside a container
docker exec tts-stt-piper-training-service-1 nvidia-smi

# Monitor training progress
curl http://localhost:8080/jobs

# Check individual service health
curl http://localhost:5000/health   # PiperTTS
curl http://localhost:5001/health   # STT (faster-whisper, mapped from internal 8000)
curl http://localhost:5003/         # whisper-cpp (any response = healthy)
curl http://localhost:8080/health   # Training
curl http://localhost:5004/health   # Qwen3-TTS
curl http://localhost:5002/health   # Qwen3-ASR
curl http://localhost:3000/health   # Frontend

# Rebuild a single service
docker-compose build piper-tts-service
docker-compose up -d piper-tts-service
```

Common issues:
- **CUDA not available**: Install NVIDIA Container Toolkit and restart Docker
- **Out of memory**: Reduce `training-batch-size` or use a smaller Whisper model
- **Slow first start**: Models are downloaded on first launch (several GB for Qwen3)
- **Port conflicts**: Check that ports 3000, 5000-5004, 8080 are free

## Local Development

Service source files are bind-mounted in `docker-compose.yml`, so code changes are reflected on container restart without rebuilding:

```bash
docker-compose restart piper-tts-service
```

All processing happens locally. No data is sent to external services.
