# Parakeet-ASR Service

Fast multilingual speech recognition using NVIDIA [parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) (FastConformer-TDT). Covers **25 European languages including German** with automatic language detection, and is among the fastest open ASR models (real-time factor in the thousands on GPU, fast even on CPU) — a strong realtime/streaming-friendly complement to the `faster-whisper` and `Qwen3-ASR` backends.

It exposes the project's native `stt-form-v1` contract (`/transcribe` with segment timestamps, so it works in the UI segmentation flow **and** as a training-pipeline STT backend) plus an OpenAI-compatible `/v1/audio/transcriptions` endpoint.

## Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/transcribe` | Transcribe one audio file (`audio`); returns text + segments |
| POST | `/v1/audio/transcriptions` | OpenAI-compatible transcription (`file`) |
| POST | `/transcribe-batch` | Transcribe multiple files in one request |
| POST | `/detect_language` | Return a transcript sample (Parakeet auto-detects language internally) |
| GET | `/health` | Health and model/device status |
| GET | `/status` | Detailed GPU and model information |

## Usage

```bash
curl -X POST "http://localhost:5005/transcribe" \
	-F "audio=@sample.wav"
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `PARAKEET_ASR_MODEL` | `nvidia/parakeet-tdt-0.6b-v3` | Hugging Face model ID |
| `ALLOWED_ORIGINS` | `*` | Comma-separated CORS origins |
| `ALLOW_CREDENTIALS` | `false` | Enables CORS credentials when origins are explicit |

## Notes

- Input audio is auto-converted to 16 kHz mono via `ffmpeg` (handles WAV/MP3/M4A/FLAC).
- Practical clip length is up to ~24 minutes with default (full) attention. For multi-hour files, chunk upstream or switch the model to local attention.
- License: NVIDIA Open Model License (see the model card). Heavier dependency footprint than Whisper (pulls in `nemo_toolkit`); model weights download on first start and are cached in the `parakeet-asr-cache` volume.
- CUDA or ROCm GPU recommended for realtime; CPU works but is slower (still competitive vs. Whisper on CPU).
