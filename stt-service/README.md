# STT Service

Speech-to-text transcription with `faster-whisper`. This service is the main Python STT backend in the stack and is also used by the Piper Training service for dataset preparation and segmentation.

## Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/transcribe` | Transcribe one file (`audio`) or many files (`audios`) |
| POST | `/transcribe-stream` | Stream a finished file's segments over SSE as they decode |
| WS | `/ws/transcribe` | Live microphone transcription — partials while you speak |
| POST | `/detect_language` | Detect spoken language and return a sample transcript |
| POST | `/unload` | Release the model and its VRAM now (`409` while a decode is in flight) |
| GET | `/health` | Health, residency, device, and multilingual status |
| GET | `/info` | Detailed service and GPU information |
| GET | `/models` | List supported Whisper model variants |
| GET | `/tasks` | Describe `transcribe` and `translate` modes |

`/transcribe-stream` and `/ws/transcribe` are different things and easy to
confuse: the first pushes segments of a **finished upload** as the decoder
produces them, the second takes **live PCM16 frames** and emits interim
hypotheses. Only the second is what the browser's live-mic panel uses.

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
| `WHISPER_MODEL_SIZE` | `large-v3-turbo` | Whisper model: `tiny`, `base`, `small`, `medium`, `large-v3`, `large-v3-turbo` |
| `WHISPER_COMPUTE_TYPE` | unset | Pin the CT2 compute type. Empty probes the GPU and picks the most memory-efficient supported option. |
| `WHISPER_OOM_FALLBACK` | `small` | Smaller multilingual model tried before abandoning the GPU |
| `WHISPER_NUM_WORKERS` | `2` | CT2 workers on the shared model; gives live and batch traffic independent slots |
| `STT_MODEL_TTL` / `MODEL_TTL` | `300` | Seconds idle before the weights are released. `0` releases the moment the service falls idle, `-1` pins them resident. |
| `FORCE_ACCELERATION` | unset | Force backend: `cuda`, `rocm`, or `cpu` |
| `USE_CUDA` | `true` | Set to `false` to force CPU mode |
| `ALLOWED_ORIGINS` | `*` | Comma-separated CORS origins |
| `ALLOW_CREDENTIALS` | `false` | Enables CORS credentials when origins are explicit |

Live-transcription tuning (`WS_WINDOW_S`, `WS_MIN_NEW_AUDIO_S`, `WS_MAX_SESSIONS`,
`WS_MAX_BUFFER_S` and the two hallucination filters) is documented with its
reasoning in [`.env.example`](../.env.example).

### Residency

The model is reference counted and released after `STT_MODEL_TTL` seconds idle,
then reloaded on demand — which is what lets Whisper share a card with the ASR
and TTS services instead of holding 1.6–3.1 GB for the container's lifetime. Two
consequences worth knowing:

- `/health` returns **200 with `model_resident: false`** for an unloaded model.
  That is idle, not down; 503 is reserved for "every load attempt failed".
- An unload never interrupts work in progress. `POST /unload` answers `409` with
  the outstanding `model_refs` while a decode is running.

## Model Notes

| Model | Multilingual | Typical Use |
|-------|--------------|-------------|
| `small` | yes | Balanced for modest GPUs / CPU; also the OOM fallback |
| `medium` | yes | Better multilingual accuracy |
| `large-v3-turbo` | yes | Default. 4 decoder layers against large-v3's 32 — fastest multilingual option, but **cannot translate** |
| `large-v3` | yes | Highest quality, more VRAM; the only choice that supports `task=translate` |
| `distil-large-v3` | no | Fast English-only inference. Never auto-selected: it has no `.en` suffix but would silently drop German. |

Models download automatically on first use and are cached in the container volume mounted at `.cache/`.

## Requirements

- NVIDIA CUDA, AMD ROCm, and CPU are supported
- Internal service port is `8000` (mapped to host port `5001` by Compose)
- For multilingual transcription, avoid English-only distilled models
