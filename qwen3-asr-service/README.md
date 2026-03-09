# Qwen3-ASR Service

Fast multilingual speech recognition using [Qwen3-ASR](https://huggingface.co/Qwen/Qwen3-ASR-1.7B).

## Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/transcribe` | Transcribe audio file (multipart: audio) |
| POST | `/detect_language` | Detect spoken language |
| GET | `/health` | Health check |

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `QWEN3_ASR_MODEL` | `Qwen/Qwen3-ASR-1.7B` | HuggingFace model ID |

## Requirements

- NVIDIA GPU with CUDA support
- ~4 GB VRAM for the 1.7B model
- Model is downloaded automatically on first start
