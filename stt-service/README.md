# STT Service

Speech-to-text transcription with `faster-whisper`. This service is the main Python STT backend in the stack and is also used by the Piper Training service for dataset preparation and segmentation.

## Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/transcribe` | Transcribe one file (`audio`) or many files (`audios`) |
| POST | `/transcribe-stream` | Stream partial transcription results over SSE |
| POST | `/detect_language` | Detect spoken language and return a sample transcript |
| GET | `/health` | Health, model, device, and multilingual status |
| GET | `/info` | Detailed service and GPU information |
| GET | `/models` | List supported Whisper model variants |
| GET | `/tasks` | Describe `transcribe` and `translate` modes |

## Usage

### Transcribe a single file

```bash
curl -X POST "http://localhost:5001/transcribe" \
  -F "audio=@recording.wav" \
  -F "language=de" \
  -F "vad_filter=true"
```

### Batch transcription

```bash
curl -X POST "http://localhost:5001/transcribe" \
  -F "audios=@clip1.wav" \
  -F "audios=@clip2.wav" \
  -F "language=auto"
```

### Detect language

```bash
curl -X POST "http://localhost:5001/detect_language" \
  -F "file=@recording.wav"
```

## Important Request Parameters

| Field | Default | Notes |
|-------|---------|-------|
| `task` | `transcribe` | `translate` is supported, but only to English |
| `language` | `auto` | Use a language code like `en`, `de`, `fr` |
| `beam_size` | `5` | Higher is slower but can improve quality |
| `vad_filter` | `true` | Disable for short/noisy clips if speech is dropped |
| `vad_threshold` | `0.5` | Lower for more permissive speech detection |
| `no_speech_threshold` | `0.6` | Raise for compressed browser audio |

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `WHISPER_MODEL_SIZE` | `small` | Whisper model: `tiny`, `base`, `small`, `medium`, `large-v3`, `distil-large-v3` |
| `FORCE_ACCELERATION` | unset | Force backend: `cuda`, `rocm`, or `cpu` |
| `USE_CUDA` | `true` | Set to `false` to force CPU mode |
| `ALLOWED_ORIGINS` | `*` | Comma-separated CORS origins |
| `ALLOW_CREDENTIALS` | `false` | Enables CORS credentials when origins are explicit |

## Model Notes

| Model | Multilingual | Typical Use |
|-------|--------------|-------------|
| `small` | yes | Balanced default for modest GPUs / CPU |
| `medium` | yes | Better multilingual accuracy |
| `large-v3` | yes | Highest quality, more VRAM |
| `distil-large-v3` | no | Fast English-only inference |

Models download automatically on first use and are cached in the container volume mounted at `.cache/`.

## Requirements

- NVIDIA CUDA, AMD ROCm, and CPU are supported
- Internal service port is `8000` (mapped to host port `5001` by Compose)
- For multilingual transcription, avoid English-only distilled models
