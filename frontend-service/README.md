# Frontend Service

Browser-facing UI and API hub for the TTS-STT platform. Built with FastAPI, Jinja2, and static assets served from the container.

## Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Main web application |
| GET | `/api-docs` | Static API documentation page |
| GET | `/health` | Service health plus configured backend URLs |
| GET | `/providers` | Provider registry with capabilities and contract metadata |
| GET | `/api/health` | Health of every backend, probed concurrently and cached for `HEALTH_CACHE_TTL` seconds (default 2). A backend that reports a `default_language` has it forwarded |
| POST | `/api/auth/check` | `204` when the caller may use the state-changing API, `401` when `API_KEY` is set and was not sent. The UI calls it before opening the live-transcription socket |
| GET | `/api/providers/{provider_id}/voices` | Normalized voice or speaker catalog for a TTS provider, plus the backend's `default_language` (what "auto" resolves to) when it reports one |
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
| WS | `/ws/stt?provider=whisper` | Live-transcription relay to the STT service (same-origin or `ALLOWED_ORIGINS` only; needs the `API_KEY` when one is set, see Security) |
| POST | `/v1/audio/transcriptions` | OpenAI-compatible transcription (`json`, `text`, `verbose_json`, `srt`, `vtt`); max 25 MB |
| POST | `/v1/audio/speech` | OpenAI-compatible speech (`mp3` default, `wav`, `pcm`) |
| GET | `/v1/models`, `/v1/models/{id}` | OpenAI-compatible model list |

`job_id` and `voice_id` path parameters must match `[A-Za-z0-9_-]{1,128}` (uuid4 job ids and the voice names the backends generate); anything else is a 422 and never reaches a backend.

### The `/v1` surface

Every `/v1` failure - including the ones FastAPI raises itself (missing form field, unknown route, wrong method, oversize body, a crash) - uses the OpenAI error envelope `{"error": {"message", "type", "param", "code"}}`. `/api/*` keeps FastAPI's `{"detail": ...}`.

- **`response_format` for transcriptions**: `json`, `text`, `verbose_json`, `srt`, `vtt`. `diarized_json` is refused (`400 unsupported_value`). Backends that report no segments get one segment spanning the whole file. `verbose_json` fills the fields a backend cannot measure with neutral values (`tokens: []`, `compression_ratio: 0`) so that openai-python's validation passes; `language` is whatever the backend reports (an ISO code from faster-whisper, a full name from whisper.cpp).
- **`response_format` for speech**: `mp3` (default; needs `ffmpeg`, otherwise `501`), `wav`, `pcm`. `pcm` is raw 16-bit little-endian mono at **24 kHz**, as OpenAI documents and stock clients assume: Qwen3 and Chatterbox already produce 24 kHz and only lose the WAV header, Piper voices (22.05 or 16 kHz) are resampled with `ffmpeg` (`501` without it, as for `mp3`). `X-Sample-Rate` is always `24000`. `wav` keeps the backend's own rate, which its header states.
- **Busy responses**: at most `MAX_CONCURRENT_FFMPEG` conversions (`mp3`, resampled `pcm`) run per worker; the next request gets `503` with `Retry-After` and `error.code: "server_busy"`, which the OpenAI SDKs retry on their own.
- **Errors**: a backend failure is reported as a sentence and a request id (`Speech backend failed: The backend service failed to handle the request (HTTP 500). Request id: 3f9a1c2b7d4e.`); the backend's own text and the internal URL are only in the gateway log under that id.
- **Default STT fallback**: `/v1/audio/transcriptions` always uses `DEFAULT_STT_PROVIDER`. If that provider cannot be *reached* (connection refused, DNS, timeout - not an error answer) and exactly one other STT provider is healthy, the request is served by that one and the response carries `X-Provider: <used>` and `X-Provider-Fallback: <default>-><used>`; a warning is logged. With no healthy alternative, or more than one, the request fails as before. Explicit selections (`/api/stt` with a `provider` field) never fall back. This is what keeps a whisper.cpp-only device (`rk3588`, `strixhalo`) working when `DEFAULT_STT_PROVIDER` was left at `whisper`; set it to `whisper-cpp` there anyway.

## Features

- Frontend entrypoint for STT, TTS, Qwen3 voice cloning, and training flows
- Injects the provider registry into the HTML template (no backend URLs: every call goes through this gateway)
- Serves static assets with restart-based cache busting
- Exposes health data for all backend services used by the UI
- Exposes optional providers such as `whisper-cpp` only when explicitly enabled in the frontend environment
- Takes Canary's language list (and the model it runs) from the canary service's own `/status`, re-read at most every five minutes and only while the page or `/providers` is requested; the built-in `de/en/fr/es` list is the fallback while the service cannot be asked. A `canary` entry in `PROVIDER_REGISTRY_JSON` is never overwritten
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
| `PARAKEET_ASR_SERVICE_URL` | `http://parakeet-asr-service:5005` | Internal Parakeet URL |
| `CANARY_ASR_SERVICE_URL` | `http://canary-asr-service:5006` | Internal Canary URL |
| `CHATTERBOX_TTS_SERVICE_URL` | `http://chatterbox-tts-service:5007` | Internal Chatterbox URL |
| `ENABLE_WHISPER_CPP` | `false` | Include whisper.cpp in the provider registry, STT selector, and health polling |
| `ENABLE_PARAKEET_ASR`, `ENABLE_CANARY_ASR`, `ENABLE_CHATTERBOX_TTS` | `false` | The same for Parakeet, Canary and Chatterbox |
| `DEFAULT_TTS_PROVIDER` | `piper` | Default TTS provider ID: preselected by the UI and used by `/v1/audio/speech` |
| `DEFAULT_STT_PROVIDER` | `whisper` | Default STT provider ID: preselected by the UI and used by `/v1/audio/transcriptions` (falls back to the one healthy alternative if unreachable, see above) |
| `TRAINING_PROVIDER` | `piper-training` | Provider ID used for training workflows |
| `PROVIDER_REGISTRY_JSON` | unset | Optional JSON override for provider metadata, per-service settings, and defaults |
| `ALLOWED_ORIGINS` | *(empty)* | Comma-separated origins allowed to call the API from a browser. **Empty or unset: same-origin only, no CORS headers.** `*` allows every origin and must be set explicitly |
| `ALLOW_CREDENTIALS` | `false` | Enables CORS credentials when origins are explicit |
| `TRUSTED_ORIGINS` | *(empty)* | Origins that are *this UI under another name* (a reverse proxy that does not forward `Host`). They pass the state-changing origin check but get no CORS headers |
| `TRUSTED_HOSTS` | *(empty)* | Hostnames (besides the ones accepted automatically, see Security) that may address this service: `tts.example.com`, `*.tail1234.ts.net`. Ports, schemes and case are ignored. The hostname of every `TRUSTED_ORIGINS` entry counts too |
| `ALLOWED_HOSTS` | *(empty)* | `*` switches the Host check off. Nothing else is read from it: hostnames go in `TRUSTED_HOSTS` |
| `TRUST_PROXY_HEADERS` | `false` | Believe `X-Forwarded-Host` (instead of `Host`) as the name the browser used, for the Host and Origin checks. Only enable behind a proxy that overwrites it |
| `API_KEY` | *(empty)* | Optional shared secret, see Security below |
| `MAX_UPLOAD_MB` | `512` | Largest request body for uploads (training audio, voice samples, STT files) - about 48 minutes of 44.1 kHz mono 16-bit WAV. Larger requests get `413`. Everything is held in memory while being forwarded, so this is also the worst-case memory cost of one request |
| `MAX_CONCURRENT_UPLOADS` | `4` | Uploads one worker process forwards at the same time; the next one gets `503` + `Retry-After: 5` before its body is read. Worst-case upload memory is `FRONTEND_WORKERS x MAX_CONCURRENT_UPLOADS x MAX_UPLOAD_MB` (2 x 4 x 512 MB = 4 GB by default), so lower `MAX_UPLOAD_MB` first on a small machine. JSON calls (capped at 1 MiB) do not count |
| `MAX_CONCURRENT_FFMPEG` | `4` | `ffmpeg` processes one worker runs at the same time for `/v1/audio/speech` (`mp3`, resampled `pcm`); the next request gets `503` + `Retry-After: 2` |
| `MAX_TTS_CHARS` | `5000` | Longest `text` accepted by `/api/tts` and voice design (`422` beyond it). Keep it `<=` the smallest `MAX_TEXT_CHARS` of the TTS backends in use: chatterbox and qwen3-tts default to 5000, so a larger value only turns the early, clear `422` into a `413` from the backend. A Piper-only deployment (backend default 20000) can raise it |
| `HEALTH_CACHE_TTL` | `2` | Seconds `/api/health` results are reused; `0` disables the cache |
| `PROVIDER_HEALTH_TIMEOUT` | `6` | Per-provider timeout for a health probe, in seconds |
| `FRONTEND_WORKERS` | `2` | uvicorn worker processes (the container's `entrypoint.sh` reads it; a value that is not a positive integer falls back to 2). Use `1` on memory-constrained boards |

There are no browser-visible backend URLs to configure: the browser only talks to this service (`/api/*`, `/v1/*`, and the `/ws/stt` relay), and the gateway proxies every backend call over the internal `*_SERVICE_URL` addresses above. The former `BROWSER_*_URL` variables are no longer read anywhere.

JSON request bodies are capped at 1 MiB regardless of `MAX_UPLOAD_MB`, and `/v1/audio/transcriptions` at 25 MB (plus multipart framing), as OpenAI does. Other free-text fields (`voice` 256, `language` 64, `instructions` and `voice_description` 4000 characters) are bounded too.

## Security

The gateway is the only service a browser talks to, and it can delete trained models, unload backends and start training. Its defaults therefore assume a browser on the LAN is hostile:

- **Same-origin by default.** No CORS headers are sent unless `ALLOWED_ORIGINS` lists an origin (or is `*`). Independently of CORS, a request that changes state (anything except `GET`/`HEAD`/`OPTIONS`) carrying an `Origin` header is refused with `403` unless the origin matches the `Host` the request was addressed to, is listed in `ALLOWED_ORIGINS`/`TRUSTED_ORIGINS`, or - with `TRUST_PROXY_HEADERS=true` - matches `X-Forwarded-Host`. Requests with no `Origin` (curl, OpenAI SDKs, other services) are not affected. The same rule guards `/ws/stt`, which browsers do not subject to the same-origin policy; the relay then dials the STT service without an `Origin` header (the backends run their own origin guard against their own `Host`, so a forwarded browser origin would be refused).
- **The `Host` header is validated (DNS rebinding).** The origin check above compares `Origin` with `Host`, and a web page that re-points its own DNS name at this service (DNS rebinding) controls both - it sends `Host: evil.example:3000` and `Origin: http://evil.example:3000`, which match, and the browser even labels the request `Sec-Fetch-Site: same-origin`. So every request (and `/ws/stt`) must address a name that belongs to this service, or it is refused with `403 host_not_allowed` before any handler runs. Accepted without configuration, because none of them can be rebound by a third party: any IP literal (`192.168.1.20`, `[::1]`, a Tailscale `100.x` address - the browser does no DNS lookup), `localhost` and `*.localhost`, `*.local`, `*.localdomain`, `*.lan`, `*.internal`, `*.home.arpa`, and single-label names (`truenas`, `frontend-service`, which is what a Docker network or a LAN search domain gives you). Anything else - a real domain in front of a reverse proxy, a Tailscale MagicDNS name - goes into **`TRUSTED_HOSTS`** (`TRUSTED_HOSTS=tts.example.com,*.tail1234.ts.net`); the hostname of each `TRUSTED_ORIGINS` entry counts as listed. `ALLOWED_HOSTS=*` switches the check off. With `TRUST_PROXY_HEADERS=true` the name in `X-Forwarded-Host` is the one that is validated; otherwise that header is ignored.
- **Behind a reverse proxy** that passes `Host` through, list its name in `TRUSTED_HOSTS` (for example `tts.example.com`). If it does not pass `Host`, the service sees its own container name (fine) and the browser's `Origin` no longer matches it: add the public origin to `TRUSTED_ORIGINS` (`https://tts.example.com`, which also trusts the host), or forward `X-Forwarded-Host`, set `TRUST_PROXY_HEADERS=true` and list the name in `TRUSTED_HOSTS`. The symptoms of getting it wrong are `403 Host ... is not allowed` on every request, or `403 Cross-origin request ... refused` on every button in the UI.
- **`API_KEY`** (unset = open, as before). When set, every `/v1/*` request and every state-changing `/api/*` request must send `Authorization: Bearer <key>` (`401` with `WWW-Authenticate: Bearer`; in the OpenAI envelope on `/v1`), and so must the `/ws/stt` upgrade. There is deliberately no exemption for requests that look like the bundled UI (matching `Origin`, `Sec-Fetch-Site: same-origin`): those are exactly what a rebinding page sends. Reads such as `/health`, `/providers` and `/api/health` stay open.
  - **The UI** learns it needs a key from the first `401` carrying that challenge, asks for it once with a browser prompt, keeps it in `sessionStorage` (this tab only, never `localStorage`) and sends it on every later gateway request. A key that is refused is dropped and asked for again. With no `API_KEY` nothing is prompted and nothing extra is sent.
  - **WebSocket**: browsers cannot set headers on a WebSocket handshake, so the key travels as a subprotocol next to the real one: `new WebSocket(url, ["tts-stt.v1", "bearer." + base64url(key)])` (base64url without padding, UTF-8). The gateway echoes back only `tts-stt.v1`, never the credential. Scripts may send `Authorization: Bearer <key>` on the upgrade request instead. A socket without a valid key is accepted and immediately closed with code `1008` and reason `A valid API key is required`, so a page can tell that from a dead server.
  - This keeps other web pages and scripts that lack the key out. It is a shared secret, not user authentication: everyone who has the key can do everything, the key is typed into a `prompt()` (not masked), and over plain `http://` it crosses the network unencrypted. If the port is reachable by people you do not trust, put real authentication and TLS in front of it.
- **Backend errors do not leak.** A backend's error body and connection errors contain file paths, tracebacks and internal URLs. Clients get a status-appropriate sentence plus a request id (`... Request id: 3f9a1c2b7d4e.`) and the detail is logged under that id. A `4xx` that is one short line of prose (a validation message such as `text is 5001 characters; the limit is 5000`) still passes through, because the UI shows it; one that contains a traceback, a filesystem path or a URL is treated like a `5xx`. `/health` and `/providers` show the provider URLs on purpose.
- **Memory and process bounds.** Uploads are held in RAM while forwarded, so at most `MAX_CONCURRENT_UPLOADS` per worker are in flight (`503` + `Retry-After` beyond that, before the body is read), on top of the `MAX_UPLOAD_MB` size cap; `MAX_CONCURRENT_FFMPEG` bounds the encoder processes.
- Responses carry `X-Content-Type-Options: nosniff`, `Referrer-Policy: same-origin` and `X-Frame-Options: SAMEORIGIN`. There is deliberately no `Content-Security-Policy`: the page still uses inline event handlers.
- The provider registry is embedded in the page with the `tojson` filter, so a `</script>` inside `PROVIDER_REGISTRY_JSON` cannot break out of it.

## Container

The image is `python:3.12-slim` and starts through `entrypoint.sh`, which `exec`s uvicorn so `docker stop` reaches it. Worker count comes from `FRONTEND_WORKERS`.

`whisper-cpp` remains optional even when its backend container is running. To surface it in the browser UI, set `ENABLE_WHISPER_CPP=true` for `frontend-service` and start the `whisper-cpp` compose profile.

Access the UI at `http://localhost:3000`.
