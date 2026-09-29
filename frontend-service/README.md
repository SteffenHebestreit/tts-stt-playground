# Frontend Service

Browser-facing UI and API hub for the TTS-STT platform. Built with FastAPI, Jinja2, and static assets served from the container.

## Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Main web application |
| GET | `/api-docs` | Static API documentation page |
| GET | `/health` | Service health plus configured backend URLs |
| GET | `/providers` | Provider registry with capabilities and contract metadata |
| GET | `/api/health` | Health of every backend, probed concurrently and cached for `HEALTH_CACHE_TTL` seconds (default 2) |
| GET | `/api/providers/{provider_id}/voices` | Normalized voice or speaker catalog for a TTS provider |
| GET | `/api/providers/{provider_id}/custom-voices` | Managed custom voice catalog for providers that support custom model deletion |
| GET | `/api/providers/{provider_id}/models` | Normalized model catalog for providers that support model switching |
| POST | `/api/providers/{provider_id}/models/select` | Switch the active provider model variant |
| GET | `/api/providers/{provider_id}/status` | Provider runtime status for advanced TTS backends |
| POST | `/api/providers/{provider_id}/unload` | Ask a backend to release its model now (upstream status and body are passed through) |
| GET | `/api/providers/{provider_id}/saved-voices` | List saved provider voice profiles |
| POST | `/api/providers/{provider_id}/saved-voices` | Save a reusable provider voice profile |
| DELETE | `/api/providers/{provider_id}/custom-voices/{voice_id}` | Delete a managed custom provider voice |
| DELETE | `/api/providers/{provider_id}/saved-voices/{voice_id}` | Delete a saved provider voice profile |
| POST | `/api/providers/{provider_id}/saved-voices/{voice_id}/tts` | Generate speech with a saved provider voice |
| POST | `/api/providers/{provider_id}/voice-clone` | Run provider-scoped voice cloning via the frontend adapter |
| POST | `/api/providers/{provider_id}/voice-design` | Run provider-scoped voice design via the frontend adapter |
| POST | `/api/tts` | Normalized frontend TTS adapter for basic synthesis |
| POST | `/api/stt` | Normalized frontend STT adapter for basic transcription |
| GET | `/api/training/deployment-targets` | Training deployment target metadata through the frontend adapter |
| POST | `/api/training/train` | Start a training job through the frontend adapter |
| POST | `/api/training/train-from-dataset` | Start dataset-backed training through the frontend adapter |
| POST | `/api/training/resume` | Resume training through the frontend adapter |
| GET | `/api/training/jobs` | List training jobs through the frontend adapter |
| GET | `/api/training/status/{job_id}` | Get training status through the frontend adapter |
| POST | `/api/training/export/{job_id}` | Export and deploy through the frontend adapter |
| GET | `/api/training/download/{job_id}` | Download exported model through the frontend adapter |
| DELETE | `/api/training/model/{job_id}` | Delete a trained model through the frontend adapter |
| DELETE | `/api/training/job/{job_id}` | Cancel a job through the frontend adapter |
| WS | `/ws/stt?provider=whisper` | Live-transcription relay to the STT service (same-origin or `ALLOWED_ORIGINS` only) |
| POST | `/v1/audio/transcriptions` | OpenAI-compatible transcription (`json`, `text`, `verbose_json`, `srt`, `vtt`); max 25 MB |
| POST | `/v1/audio/speech` | OpenAI-compatible speech (`mp3` default, `wav`, `pcm`) |
| GET | `/v1/models`, `/v1/models/{id}` | OpenAI-compatible model list |

`job_id` and `voice_id` path parameters must match `[A-Za-z0-9_-]{1,128}` (uuid4 job ids and the voice names the backends generate); anything else is a 422 and never reaches a backend.

### The `/v1` surface

Every `/v1` failure - including the ones FastAPI raises itself (missing form field, unknown route, wrong method, oversize body, a crash) - uses the OpenAI error envelope `{"error": {"message", "type", "param", "code"}}`. `/api/*` keeps FastAPI's `{"detail": ...}`.

- **`response_format` for transcriptions**: `json`, `text`, `verbose_json`, `srt`, `vtt`. `diarized_json` is refused (`400 unsupported_value`). Backends that report no segments get one segment spanning the whole file. `verbose_json` fills the fields a backend cannot measure with neutral values (`tokens: []`, `compression_ratio: 0`) so that openai-python's validation passes; `language` is whatever the backend reports (an ISO code from faster-whisper, a full name from whisper.cpp).
- **`response_format` for speech**: `mp3` (default; needs `ffmpeg`, otherwise `501`), `wav`, `pcm`. `pcm` is raw 16-bit little-endian mono at the backend's **native** sample rate (Piper 22050 Hz, Qwen3 and Chatterbox 24000 Hz) - nothing is resampled, and the rate is returned in the `X-Sample-Rate` header. OpenAI documents 24 kHz, so a client that hard-codes it must read the header for Piper voices.
- **Default STT fallback**: `/v1/audio/transcriptions` always uses `DEFAULT_STT_PROVIDER`. If that provider cannot be *reached* (connection refused, DNS, timeout - not an error answer) and exactly one other STT provider is healthy, the request is served by that one and the response carries `X-Provider: <used>` and `X-Provider-Fallback: <default>-><used>`; a warning is logged. With no healthy alternative, or more than one, the request fails as before. Explicit selections (`/api/stt` with a `provider` field) never fall back. This is what keeps a whisper.cpp-only device (`rk3588`, `strixhalo`) working when `DEFAULT_STT_PROVIDER` was left at `whisper`; set it to `whisper-cpp` there anyway.

## Features

- Frontend entrypoint for STT, TTS, Qwen3 voice cloning, and training flows
- Injects a provider registry plus browser-facing service URLs into the HTML template
- Serves static assets with restart-based cache busting
- Exposes health data for all backend services used by the UI
- Exposes optional providers such as `whisper-cpp` only when explicitly enabled in the frontend environment
- Supports multiple STT request contracts, including OpenAI-compatible `/v1/audio/transcriptions`
- Exposes a normalized basic TTS adapter so multiple providers can share one browser-facing synthesis contract
- Exposes a normalized basic STT adapter so multiple providers can share one browser-facing transcription contract
- Exposes provider-scoped advanced TTS adapters for model switching, saved voices, voice cloning, and voice design
- Exposes provider-scoped custom voice management adapters so the browser does not call Piper directly for trained-voice operations
- Normalizes saved-voice library responses into stable preview and display-timestamp fields for the browser
- Uses provider `settings` metadata to drive language, quality, speaker, and training defaults in the UI
- Uses registry `ui.copy` and provider `ui.sections` metadata to drive tab labels, section copy, and provider-specific placeholders in the UI
- Uses provider `ui.forms` schemas for advanced feature controls such as training and voice cloning labels, hints, and action text
- Uses provider `ui.messages` schemas for runtime workflow messaging such as validation, progress, status, notifications, list empty states, management actions, and the basic TTS/STT submit flows
- Loads training deployment targets from the training service so export/deploy actions are target-aware rather than Piper-only
- Proxies training actions through `/api/training/*` so the browser no longer needs to call the training service directly
- Normalizes training job, training detail, and export deployment payloads so the browser consumes stable labels and summaries instead of raw training-service fields

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `TTS_SERVICE_URL` | `http://piper-tts-service:5000` | Internal PiperTTS URL |
| `STT_SERVICE_URL` | `http://stt-service:8000` | Internal STT URL |
| `VOICE_TRAINING_URL` | `http://piper-training-service:8080` | Internal training URL |
| `QWEN3_TTS_SERVICE_URL` | `http://qwen3-tts-service:5004` | Internal Qwen3-TTS URL |
| `QWEN3_ASR_SERVICE_URL` | `http://qwen3-asr-service:5002` | Internal Qwen3-ASR URL |
| `WHISPER_CPP_SERVICE_URL` | `http://whisper-cpp:8080` | Internal whisper.cpp URL |
| `ENABLE_WHISPER_CPP` | `false` | Include whisper.cpp in the provider registry, STT selector, and health polling |
| `BROWSER_TTS_URL` | `http://localhost:5000` | Browser-visible PiperTTS URL |
| `BROWSER_STT_URL` | `http://localhost:5001` | Browser-visible STT URL |
| `BROWSER_TRAINING_URL` | `http://localhost:8080` | Browser-visible training URL |
| `BROWSER_QWEN3_TTS_URL` | `http://localhost:5004` | Browser-visible Qwen3-TTS URL |
| `BROWSER_QWEN3_ASR_URL` | `http://localhost:5002` | Browser-visible Qwen3-ASR URL |
| `BROWSER_WHISPER_CPP_URL` | `http://localhost:5003` | Browser-visible whisper.cpp URL |
| `DEFAULT_TTS_PROVIDER` | `piper` | Default TTS provider ID: preselected by the UI and used by `/v1/audio/speech` |
| `DEFAULT_STT_PROVIDER` | `whisper` | Default STT provider ID: preselected by the UI and used by `/v1/audio/transcriptions` (falls back to the one healthy alternative if unreachable, see above) |
| `TRAINING_PROVIDER` | `piper-training` | Provider ID used for training workflows |
| `PROVIDER_REGISTRY_JSON` | unset | Optional JSON override for provider metadata, per-service settings, and defaults |
| `ALLOWED_ORIGINS` | *(empty)* | Comma-separated origins allowed to call the API from a browser. **Empty or unset: same-origin only, no CORS headers.** `*` allows every origin and must be set explicitly |
| `ALLOW_CREDENTIALS` | `false` | Enables CORS credentials when origins are explicit |
| `TRUSTED_ORIGINS` | *(empty)* | Origins that are *this UI under another name* (a reverse proxy that does not forward `Host`). They pass the state-changing origin check but get no CORS headers |
| `TRUST_PROXY_HEADERS` | `false` | Believe `X-Forwarded-Host` when comparing an `Origin` with the host. Only enable behind a proxy that sets it |
| `API_KEY` | *(empty)* | Optional shared secret, see Security below |
| `MAX_UPLOAD_MB` | `512` | Largest request body for uploads (training audio, voice samples, STT files) - about 48 minutes of 44.1 kHz mono 16-bit WAV. Larger requests get `413`. Everything is held in memory while being forwarded, so this is also the worst-case memory cost of one request |
| `MAX_TTS_CHARS` | `20000` | Longest `text` accepted by `/api/tts` and voice design (`422` beyond it) |
| `HEALTH_CACHE_TTL` | `2` | Seconds `/api/health` results are reused; `0` disables the cache |
| `PROVIDER_HEALTH_TIMEOUT` | `6` | Per-provider timeout for a health probe, in seconds |
| `FRONTEND_WORKERS` | `2` | uvicorn worker processes (the container's `entrypoint.sh` reads it; a value that is not a positive integer falls back to 2). Use `1` on memory-constrained boards |

JSON request bodies are capped at 1 MiB regardless of `MAX_UPLOAD_MB`, and `/v1/audio/transcriptions` at 25 MB (plus multipart framing), as OpenAI does. Other free-text fields (`voice` 256, `language` 64, `instructions` and `voice_description` 4000 characters) are bounded too.

## Security

The gateway is the only service a browser talks to, and it can delete trained models, unload backends and start training. Its defaults therefore assume a browser on the LAN is hostile:

- **Same-origin by default.** No CORS headers are sent unless `ALLOWED_ORIGINS` lists an origin (or is `*`). Independently of CORS, a request that changes state (anything except `GET`/`HEAD`/`OPTIONS`) carrying an `Origin` header is refused with `403` unless the origin matches the `Host` the request was addressed to, is listed in `ALLOWED_ORIGINS`/`TRUSTED_ORIGINS`, or - with `TRUST_PROXY_HEADERS=true` - matches `X-Forwarded-Host`. Requests with no `Origin` (curl, OpenAI SDKs, other services) are not affected. The same rule guards `/ws/stt`, which browsers do not subject to the same-origin policy.
- **Behind a reverse proxy** that does not pass `Host` through, add the public origin to `TRUSTED_ORIGINS` (for example `https://tts.example.com`), or forward `X-Forwarded-Host` and set `TRUST_PROXY_HEADERS=true`. The symptom of getting it wrong is `403 Cross-origin request ... refused` on every button in the UI.
- **`API_KEY`** (unset = open, as before). When set, every `/v1/*` request must send `Authorization: Bearer <key>` (`401` in the OpenAI envelope otherwise), and so must every state-changing `/api/*` request unless it is a same-origin browser request (`Sec-Fetch-Site: same-origin`/`none`, or a matching `Origin`) - the bundled UI has no key to send. Reads such as `/health` and `/api/health` stay open. This keeps other web pages and non-browser scripts out; it is not user authentication, since a script can forge browser headers. If the port is reachable by people you do not trust, put real authentication in front of it.
- Responses carry `X-Content-Type-Options: nosniff`, `Referrer-Policy: same-origin` and `X-Frame-Options: SAMEORIGIN`. There is deliberately no `Content-Security-Policy`: the page still uses inline event handlers.
- The provider registry is embedded in the page with the `tojson` filter, so a `</script>` inside `PROVIDER_REGISTRY_JSON` cannot break out of it.

## Container

The image is `python:3.12-slim` and starts through `entrypoint.sh`, which `exec`s uvicorn so `docker stop` reaches it. Worker count comes from `FRONTEND_WORKERS`.

`whisper-cpp` remains optional even when its backend container is running. To surface it in the browser UI, set `ENABLE_WHISPER_CPP=true` for `frontend-service` and start the `whisper-cpp` compose profile.

Access the UI at `http://localhost:3000`.
