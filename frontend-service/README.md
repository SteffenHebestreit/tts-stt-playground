# Frontend Service

Browser-facing UI and API hub for the TTS-STT platform. Built with FastAPI, Jinja2, and static assets served from the container.

## Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Main web application |
| GET | `/api-docs` | Static API documentation page |
| GET | `/health` | Service health plus configured backend URLs |

## Features

- Frontend entrypoint for STT, TTS, Qwen3 voice cloning, and training flows
- Injects browser-facing service URLs into the HTML template
- Serves static assets with restart-based cache busting
- Exposes health data for all backend services used by the UI

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `TTS_SERVICE_URL` | `http://piper-tts-service:5000` | Internal PiperTTS URL |
| `STT_SERVICE_URL` | `http://stt-service:8000` | Internal STT URL |
| `VOICE_TRAINING_URL` | `http://piper-training-service:8080` | Internal training URL |
| `QWEN3_TTS_SERVICE_URL` | `http://qwen3-tts-service:5004` | Internal Qwen3-TTS URL |
| `QWEN3_ASR_SERVICE_URL` | `http://qwen3-asr-service:5002` | Internal Qwen3-ASR URL |
| `BROWSER_TTS_URL` | `http://localhost:5000` | Browser-visible PiperTTS URL |
| `BROWSER_STT_URL` | `http://localhost:5001` | Browser-visible STT URL |
| `BROWSER_TRAINING_URL` | `http://localhost:8080` | Browser-visible training URL |
| `BROWSER_QWEN3_TTS_URL` | `http://localhost:5004` | Browser-visible Qwen3-TTS URL |
| `BROWSER_QWEN3_ASR_URL` | `http://localhost:5002` | Browser-visible Qwen3-ASR URL |
| `ALLOWED_ORIGINS` | `*` | Comma-separated CORS origins |
| `ALLOW_CREDENTIALS` | `false` | Enables CORS credentials when origins are explicit |

Access the UI at `http://localhost:3000`.
