# TTS-STT Platform

A self-hosted, Docker-based platform for text-to-speech synthesis, speech-to-text transcription, custom voice training, and voice cloning.

## Services

| Service | Port | Description |
|---------|------|-------------|
| **Frontend** | 3000 | Web UI and API documentation hub |
| **PiperTTS** | 5000 | Text-to-speech with 40+ voices and custom model support |
| **STT (Whisper)** | 5001 | Speech-to-text transcription and audio segmentation |
| **Piper Training** | 8080 | VITS neural network voice training pipeline |
| **Qwen3-TTS** | 5004 | Voice cloning and multilingual TTS (replaces XTTS) |
| **Qwen3-ASR** | 5002 | Fast multilingual speech recognition |

## Quick Start

```bash
# Clone and start all services
docker-compose up -d

# Check service health
docker-compose ps

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

Open **http://localhost:3000** for the web interface or **http://localhost:3000/api-docs** for API documentation.

## Prerequisites

- Docker and Docker Compose
- NVIDIA GPU with CUDA drivers (recommended for training and Qwen3 services)
- NVIDIA Container Toolkit (`nvidia-docker2`)
- 16 GB+ RAM recommended

CPU-only mode works for PiperTTS and STT but training and Qwen3 services require a GPU.

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

### Speech Recognition (Qwen3-ASR)
- Fast multilingual speech recognition
- Language detection

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
| POST | `/transcribe` | Transcribe audio file |
| POST | `/detect_language` | Detect spoken language |
| GET | `/health` | Health check |

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
| `WHISPER_MODEL_SIZE` | small | STT model: tiny, base, small, medium, large-v3 |
| `CUDA_VISIBLE_DEVICES` | 0 | GPU device index |
| `QWEN3_TTS_MODEL` | Qwen/Qwen3-TTS-12Hz-1.7B-Base | Qwen3 TTS model ID |
| `QWEN3_ASR_MODEL` | Qwen/Qwen3-ASR-1.7B | Qwen3 ASR model ID |
| `ALLOWED_ORIGINS` | * | CORS allowed origins |

## Project Structure

```
tts-stt/
├── docker-compose.yml
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
├── stt-service/               # Whisper STT
│   └── app.py
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
curl http://localhost:5001/health   # STT (mapped from internal 8000)
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
