# STT Service

Speech-to-text transcription using [OpenAI Whisper](https://github.com/openai/whisper). Used both for general transcription and as the audio segmentation backend for the Piper Training Service.

## Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/transcribe` | Transcribe an audio file |
| POST | `/transcribe-stream` | Transcribe with streaming response |
| POST | `/detect_language` | Detect spoken language |
| GET | `/models` | List available Whisper model sizes |
| GET | `/tasks` | List running transcription tasks |
| GET | `/info` | Service and model info |
| GET | `/health` | Health check |

## Usage

### Transcribe audio

```bash
curl -X POST "http://localhost:5001/transcribe" \
  -F "audio_file=@recording.wav" \
  -F "language=de"
```

Response includes transcribed text and word-level timestamp segments.

### Detect language

```bash
curl -X POST "http://localhost:5001/detect_language" \
  -F "audio_file=@recording.wav"
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `WHISPER_MODEL_SIZE` | `small` | Model size: tiny, base, small, medium, large-v3 |
| `FORCE_ACCELERATION` | `cuda` | `cuda` or `cpu` |

## Model Sizes

| Size | VRAM | Speed | Accuracy |
|------|------|-------|----------|
| tiny | ~1 GB | fastest | lowest |
| base | ~1 GB | fast | low |
| small | ~2 GB | balanced | good |
| medium | ~5 GB | slower | high |
| large-v3 | ~10 GB | slowest | best |

Models are downloaded automatically on first use and cached in `~/.cache/whisper`.

## Requirements

- NVIDIA GPU recommended (CPU fallback available but slow for large files)
- Internal port: 8000 (mapped to host port 5001 via docker-compose)
