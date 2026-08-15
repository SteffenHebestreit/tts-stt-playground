# Canary-ASR Service

Speech-to-text built on NVIDIA `canary-180m-flash` — a 182M-parameter multilingual ASR model with punctuation/capitalisation that runs at RTFx >1000 on GPU, making it the lowest-latency transcription backend in this stack. Supports **English, German, Spanish, and French**.

Canary has no language identification: the request `language` selects the decoder language. `auto`/unsupported values fall back to `CANARY_DEFAULT_LANGUAGE` (default `de`).

## Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/transcribe` | Transcribe one file (`audio`, optional `language`) — `stt-form-v1` shape |
| POST | `/transcribe-batch` | Batch transcription (`audios[]`, optional `language`) |
| POST | `/v1/audio/transcriptions` | OpenAI-compatible endpoint (`file`, `language`, `response_format`) |
| POST | `/detect_language` | Returns a transcript sample (Canary has no LID — no language/confidence) |
| GET | `/status` | Device, GPU memory, model, supported languages |
| GET | `/health` | Health check |
| POST | `/unload` | Release the model and its VRAM now (`409` while a request is in flight) |

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `CANARY_ASR_MODEL` | `nvidia/canary-180m-flash` | NeMo model name (`nvidia/canary-1b-v2` for the 25-language, higher-accuracy variant) |
| `CANARY_DEFAULT_LANGUAGE` | `de` | Language used when the request says `auto` or something unsupported |
| `ASR_MODEL_TTL` / `MODEL_TTL` | `300` | Seconds idle before the ~2 GB model is released. `0` releases as soon as it falls idle, `-1` pins it resident. |
| `ASR_MAX_CONCURRENCY` | `1` | Concurrent inferences on the shared model |
| `ALLOWED_ORIGINS` | `*` | Comma-separated CORS origins |
| `ALLOW_CREDENTIALS` | `false` | Enables CORS credentials when origins are explicit |

## Running

Opt-in profile (heavy NeMo image, like Parakeet):

```bash
ENABLE_CANARY_ASR=true docker compose --profile canary-asr --profile frontend up -d
```

Set `ENABLE_CANARY_ASR=true` on the frontend service so the provider appears in the browser UI.

### Residency

The model is loaded on demand and released after `ASR_MODEL_TTL` seconds idle,
then reloaded on the next request. This service is opt-in and bursty — selected
for a batch of files, then idle — so holding the card between bursts is the
worst trade available on a VRAM-bound host.

Unloading is reference counted: `POST /unload` answers `409` with the
outstanding `active_requests` while a forward pass is running, and `/health`
returns **200 with `model_resident: false`** for an idle-unloaded model. That is
idle, not down.
