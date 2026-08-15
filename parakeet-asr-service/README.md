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
| POST | `/unload` | Release the model and its VRAM now (`409` while a request is in flight) |

## Usage

```bash
curl -X POST "http://localhost:5005/transcribe" \
	-F "audio=@sample.wav"
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `PARAKEET_ASR_MODEL` | `nvidia/parakeet-tdt-0.6b-v3` | Hugging Face model ID |
| `ASR_MODEL_TTL` / `MODEL_TTL` | `300` | Seconds idle before the ~3 GB model is released. `0` releases as soon as it falls idle, `-1` pins it resident. |
| `ASR_MAX_CONCURRENCY` | `1` | Concurrent inferences on the shared model |
| `ASR_MAX_BATCH` | `8` | Batch size for `/transcribe-batch`; peak VRAM scales with it |
| `ALLOWED_ORIGINS` | `*` | Comma-separated CORS origins |
| `ALLOW_CREDENTIALS` | `false` | Enables CORS credentials when origins are explicit |

## Notes

- Input audio is auto-converted to 16 kHz mono via `ffmpeg` (handles WAV/MP3/M4A/FLAC).
- Practical clip length is up to ~24 minutes with default (full) attention. For multi-hour files, chunk upstream or switch the model to local attention.
- License: NVIDIA Open Model License (see the model card). Heavier dependency footprint than Whisper (pulls in `nemo_toolkit`); model weights download on first start and are cached in the `parakeet-asr-cache` volume.
- CUDA or ROCm GPU recommended for realtime; CPU works but is slower (still competitive vs. Whisper on CPU).

### Residency

The model is loaded on demand and released after `ASR_MODEL_TTL` seconds idle,
then reloaded on the next request. This service is opt-in and bursty — selected
for a batch of files, then idle — so holding the card between bursts is the
worst trade available on a VRAM-bound host.

Unloading is reference counted: `POST /unload` answers `409` with the
outstanding `active_requests` while a forward pass is running, and `/health`
returns **200 with `model_resident: false`** for an idle-unloaded model. That is
idle, not down.
