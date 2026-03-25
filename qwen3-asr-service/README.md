# Qwen3-ASR Service

Fast multilingual speech recognition using [Qwen3-ASR](https://huggingface.co/Qwen/Qwen3-ASR-1.7B). This service complements the `faster-whisper` backend with stronger Qwen-family ASR quality and a simpler response schema for the Qwen3 voice workflows.

## Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/transcribe` | Transcribe one audio file (`audio`) |
| POST | `/transcribe-batch` | Transcribe multiple files in one request |
| POST | `/detect_language` | Detect spoken language and return sample text |
| GET | `/health` | Health and model/device status |
| GET | `/status` | Detailed GPU and model information |

## Usage

```bash
curl -X POST "http://localhost:5002/transcribe" \
	-F "audio=@sample.wav" \
	-F "language=auto"
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `QWEN3_ASR_MODEL` | `Qwen/Qwen3-ASR-1.7B` | Hugging Face model ID |
| `ALLOWED_ORIGINS` | `*` | Comma-separated CORS origins |
| `ALLOW_CREDENTIALS` | `false` | Enables CORS credentials when origins are explicit |

## Requirements

- CUDA or ROCm GPU strongly recommended
- CPU mode works, but is too slow for practical long-form usage
- The 1.7B model typically needs about 4 GB of VRAM
- Model weights are downloaded automatically on first start and cached in the container volume
