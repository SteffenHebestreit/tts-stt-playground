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

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `CANARY_ASR_MODEL` | `nvidia/canary-180m-flash` | NeMo model name (`nvidia/canary-1b-v2` for the 25-language, higher-accuracy variant) |
| `CANARY_DEFAULT_LANGUAGE` | `de` | Language used when the request says `auto` or something unsupported |
| `ALLOWED_ORIGINS` | `*` | Comma-separated CORS origins |
| `ALLOW_CREDENTIALS` | `false` | Enables CORS credentials when origins are explicit |

## Running

Opt-in profile (heavy NeMo image, like Parakeet):

```bash
ENABLE_CANARY_ASR=true docker compose --profile canary-asr --profile frontend up -d
```

Set `ENABLE_CANARY_ASR=true` on the frontend service so the provider appears in the browser UI.
