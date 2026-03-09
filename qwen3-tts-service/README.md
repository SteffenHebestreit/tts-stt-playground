# Qwen3-TTS Service

Voice cloning and text-to-speech using [Qwen3-TTS](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base). Replaces the previous XTTS service.

## Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/tts` | Generate speech with built-in speaker |
| POST | `/clone` | Clone voice from an audio sample (multipart: text, lang, file) |
| GET | `/status` | Model load state and GPU info |
| GET | `/health` | Health check |

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `QWEN3_TTS_MODEL` | `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | HuggingFace model ID |

## Requirements

- NVIDIA GPU with CUDA support
- ~4 GB VRAM for the 1.7B model
- Model is downloaded automatically on first start
