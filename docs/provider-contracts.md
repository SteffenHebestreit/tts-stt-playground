# Provider Contracts

This repository now exposes a frontend provider registry that describes which services are available, which capabilities they implement, and which request contracts the UI should use.

## Registry Endpoint

- Frontend endpoint: `/providers`
- Response shape:

```json
{
  "providers": {
    "provider-id": {
      "kind": "tts|stt|training",
      "display_name": "Human readable name",
      "short_name": "Compact UI label",
      "internal_url": "http://service:port",
      "health_endpoint": "/health",
      "capabilities": ["capability-name"],
      "contracts": {
        "feature": "contract-name"
      },
      "settings": {
        "defaults": {},
        "languages": [],
        "qualities": []
      },
      "ui": {
        "family": "optional-ui-family",
        "selectable_as_engine": true,
        "selectable_as_stt": false,
        "show_status": true,
        "tab_label": "Optional tab label override",
        "sections": {},
        "forms": {},
        "messages": {}
      }
    }
  },
  "ui": {
    "default_tts_provider": "piper",
    "default_stt_provider": "whisper",
    "training_provider": "piper-training",
    "enable_whisper_cpp": false,
    "copy": {}
  }
}
```

### Providers shipped with the gateway

| Provider id | Kind | Service | Registered when | Contracts (feature: contract) |
|---|---|---|---|---|
| `piper` | tts | `piper-tts-service` | always | `tts: simple-json-tts-v1`, `voice_catalog: voice-catalog-v1`, `managed_voices: custom-voice-library-v1` |
| `qwen3` | tts | `qwen3-tts-service` | always | `tts: simple-json-tts-v1`, `voice_catalog: speaker-catalog-v1`, `model_catalog`, `model_selection`, `runtime_status`, `saved_voices: saved-voice-library-v1`, `voice_clone`, `voice_design` |
| `chatterbox` | tts | `chatterbox-tts-service` | `ENABLE_CHATTERBOX_TTS=true` | `tts: simple-json-tts-v1`, `voice_clone: voice-clone-tts-v1`, `tts_stream: chunked-wav-stream-v1` |
| `whisper` | stt | `stt-service` | always | `transcribe: stt-form-v1`, `detect_language: stt-detect-language-v1` (also live transcription over `/ws/stt`) |
| `qwen3-asr` | stt | `qwen3-asr-service` | always | `transcribe: stt-form-v1`, `detect_language: stt-detect-language-v1` |
| `parakeet` | stt | `parakeet-asr-service` | `ENABLE_PARAKEET_ASR=true` | `transcribe: stt-form-v1` |
| `canary` | stt | `canary-asr-service` | `ENABLE_CANARY_ASR=true` | `transcribe: stt-form-v1` |
| `whisper-cpp` | stt | `whisper-cpp` | `ENABLE_WHISPER_CPP=true` | `transcribe: openai-audio-transcriptions-v1` |
| `piper-training` | training | `piper-training-service` | always | `training: voice-training-job-v1` |

Optional providers are omitted from the registry entirely until their `ENABLE_*` flag is set on
`frontend-service`; the flag and the compose profile that starts the service are separate switches
and both are needed (a flag without its service is a permanently red status indicator, a service
without its flag is invisible).

Five gateway variables shape the registry: `DEFAULT_TTS_PROVIDER` and `DEFAULT_STT_PROVIDER` (which
engine the UI preselects; must be a provider id from this endpoint and, for STT, one that is
running: `whisper` is `stt-service`, which the ARM64 and Strix Halo presets do not start, so they
set `DEFAULT_STT_PROVIDER=whisper-cpp`), `TRAINING_PROVIDER` (the provider id that owns training),
`PROVIDER_HEALTH_TIMEOUT`, and `PROVIDER_REGISTRY_JSON`. The latter is JSON with `providers` and/or
`ui` objects: an entry with the same id **replaces** the built-in one as a whole (there is no deep
merge, so restate every field you keep), and malformed JSON stops the gateway from starting. All
five are forwarded by `docker-compose.yml`.

The `settings` object is intentionally provider-specific metadata for configurable UI defaults and allowed values. The frontend now uses it to drive service configuration controls such as language lists, quality levels, training defaults, and built-in speaker defaults.

The top-level `ui.copy` object and provider-level `ui.sections` metadata are used for labels and descriptive copy that would otherwise be hardcoded in the frontend template, such as tab labels, panel titles, placeholder text, and provider-mode-specific help text.

The provider-level `ui.forms` object is a capability-oriented schema for advanced feature UIs. The frontend now uses it for field labels, placeholders, hints, and action labels in workflows such as training and voice cloning.

The provider-level `ui.messages` object is a runtime copy schema for workflow status text, progress updates, notifications, validation messages, list empty states, and management actions. The frontend now uses it for the basic TTS/STT submit flows as well as advanced Qwen3 and training lifecycle messaging instead of hardcoding those strings in JavaScript.

## Frontend Adapter Endpoints

The frontend service also exposes normalized browser-facing adapter endpoints.

### `GET /api/providers/{provider_id}/voices`

Returns a normalized voice list for TTS providers that expose a catalog.

Normalized response:

```json
{
  "provider": "piper",
  "contract": "voice-catalog-v1",
  "voices": [
    {
      "id": "en_US-lessac-medium",
      "name": "lessac",
      "language": "en_US",
      "description": "medium",
      "kind": "default",
      "raw": {}
    }
  ]
}
```

### `GET /api/providers/{provider_id}/custom-voices`

Returns a normalized list of managed custom voices for providers that support deletion or lifecycle management of user-trained voices.

Normalized response:

```json
{
  "provider": "piper",
  "contract": "custom-voice-library-v1",
  "voices": [
    {
      "id": "demo_custom_voice",
      "name": "demo_custom_voice",
      "language": "en_US",
      "description": "medium",
      "kind": "custom",
      "raw": {}
    }
  ]
}
```

### `GET /api/providers/{provider_id}/models`

Returns a normalized model catalog for providers that support runtime model selection.

Normalized response:

```json
{
  "provider": "qwen3",
  "current_model": {
    "id": "Qwen3-0.6B",
    "name": "Qwen3 0.6B",
    "description": "Fast model",
    "capabilities": ["voice_clone"],
    "capabilities_text": "voice_clone",
    "is_current": true,
    "raw": {}
  },
  "models": [
    {
      "id": "Qwen3-0.6B",
      "name": "Qwen3 0.6B",
      "description": "Fast model",
      "capabilities": ["voice_clone"],
      "capabilities_text": "voice_clone",
      "is_current": true,
      "raw": {}
    }
  ]
}
```

### `POST /api/providers/{provider_id}/models/select`

Switches the active model for providers that expose model variants.

Request:

```json
{
  "model": "Qwen3-0.6B"
}
```

Normalized response:

```json
{
  "provider": "qwen3",
  "message": "switched",
  "model": {
    "id": "Qwen3-0.6B",
    "name": "Qwen3 0.6B"
  }
}
```

### `GET /api/providers/{provider_id}/status`

Returns normalized runtime metadata for advanced TTS backends.

Normalized response:

```json
{
  "provider": "qwen3",
  "device_name": "CUDA",
  "device_type": "gpu",
  "model_loaded": true,
  "model_name": "Qwen3 0.6B",
  "gpu_memory_gb": 2.0,
  "speakers": ["Vivian", "Ryan"]
}
```

### `GET /api/providers/{provider_id}/saved-voices`

Lists saved voice profiles for providers that expose a reusable voice library.

Response:

```json
{
  "provider": "qwen3",
  "voices": [
    {
      "id": "voice-1",
      "name": "Demo Voice",
      "language": "English",
      "reference_text": "sample reference text",
      "reference_preview": "sample reference text",
      "created_at": "2025-01-01T00:00:00Z",
      "created_at_display": "2025-01-01 00:00 UTC"
    }
  ]
}
```

### `POST /api/providers/{provider_id}/saved-voices`

Creates a saved voice profile using multipart form data.

Expected form fields depend on the provider contract. For the current `saved-voice-library-v1` implementation, the adapter forwards the provider-native fields such as `name`, `lang`, and `file`.

### `DELETE /api/providers/{provider_id}/saved-voices/{voice_id}`

Deletes a saved voice profile.

### `DELETE /api/providers/{provider_id}/custom-voices/{voice_id}`

Deletes a managed custom voice for providers that expose user-trained model lifecycle operations.

### `POST /api/providers/{provider_id}/saved-voices/{voice_id}/tts`

Runs speech synthesis with a saved voice profile. The adapter returns an audio payload and forwards `X-*` response headers.

### `POST /api/providers/{provider_id}/voice-clone`

Runs provider-scoped voice cloning through the frontend adapter. For the current `voice-clone-tts-v1` implementation, this route accepts the provider-native multipart fields and chooses the appropriate backend clone flow based on whether `ref_text` is present.

### `POST /api/providers/{provider_id}/voice-design`

Runs provider-scoped voice design through the frontend adapter.

Request:

```json
{
  "text": "Hello world",
  "voice_description": "Warm narrator",
  "lang": "English"
}
```

### `POST /api/tts`

This is the normalized browser-facing basic TTS contract used by the frontend adapter.

Request:

```json
{
  "provider": "piper",
  "text": "Hello world",
  "voice": "en_US-lessac-medium",
  "language": "en",
  "quality": "medium",
  "gender": "female",
  "speed": 1.0,
  "instructions": "",
  "output_format": "wav"
}
```

Response:

- audio binary payload
- passthrough `X-*` headers from backend provider when present (`X-Language`, `X-Language-Requested`, `X-Language-Fallback`, `X-Chunk-Count`, …)
- `X-Provider` header added by the adapter

`language` is compared case-insensitively and trimmed: `auto`, `AUTO`, ` Auto ` and an empty value all
mean "no language given", so `/api/tts` and `/v1/audio/speech` send a backend the same body. What
a backend does with it is the backend's contract: Piper resolves it to `PIPER_DEFAULT_LANGUAGE`
(German) after a text-based guess, Qwen3-TTS to `QWEN3_DEFAULT_LANGUAGE` (German). `text` is capped
at `MAX_TTS_CHARS` (422); a backend's own limit answers 413 with its own message, and a provider
declaring `tts_stream` (Chatterbox) is called on `/tts-stream`.

### `POST /api/stt`

This is the normalized browser-facing basic STT contract used by the frontend adapter.

Request:

- multipart field `provider`
- multipart field `audio`
- optional multipart field `language`

Response:

- JSON object with:
  - `text`: string
  - `segments`: array of `{start, end, text}`
  - `language`: string or `null`
  - `duration`: number or `null`
- passthrough `X-*` headers from backend provider when present
- `X-Provider` header added by the adapter

The provider is exactly the one named in `provider`: this route never falls back to another
backend (only the implicit default of `POST /v1/audio/transcriptions` does, and it says so in
`X-Provider-Fallback`, see [`api.md`](./api.md)).

### Live transcription: `WS /ws/stt`

The gateway relays the browser's microphone stream to a provider that declares `live_transcribe`
(today `whisper`, selected with `?provider=`); any other provider is refused with close code 1008 and a
reason. The same Host and origin rules as the REST API apply (`403` before the upgrade, see
[`api.md`](./api.md#network-exposure-and-access-control)), and so does `API_KEY`. Browsers cannot send
an `Authorization` header on a WebSocket, so the key travels as a subprotocol next to the real one:
`new WebSocket(url, ["tts-stt.v1", "bearer." + base64url(key)])`, and the gateway echoes back only
`tts-stt.v1`. A socket without a valid key is accepted and closed at once with code 1008 and the reason
`A valid API key is required`. The backend's frames pass through unchanged:
`{"type": "partial", "confirmed": …, "pending": …}` where `confirmed` is the whole session's
committed text so far (it only grows), `{"type": "final", …}`, and `{"type": "error", "code": …,
"message": …, "error": …}`. A live socket that sends nothing for `WS_IDLE_TIMEOUT_S` (60) is closed
with code 4408; a language code that is not valid is answered with an error frame and the previous
language is kept.

### `/api/training/*`

The frontend service also exposes a training adapter namespace. These routes proxy the training provider selected in the registry and keep browser-side code independent from direct training-service URLs.

Available routes:

- `GET /api/training/deployment-targets`
- `POST /api/training/train`
- `POST /api/training/train-from-dataset`
- `POST /api/training/resume`
- `GET /api/training/jobs`
- `GET /api/training/status/{job_id}`
- `POST /api/training/export/{job_id}`
- `GET /api/training/download/{job_id}`
- `DELETE /api/training/model/{job_id}`
- `DELETE /api/training/job/{job_id}`

Normalized training summary fields exposed by `GET /api/training/jobs`:

- `voice_name`: stable display name for the trained voice or job
- `deployment_target_label`: browser-facing label for the selected deployment target
- `created_at_display`: stable timestamp label or `N/A`

Normalized training detail fields exposed by `GET /api/training/status/{job_id}`:

- `voice_name`
- `deployment_target_label`
- `created_at_display`
- `config_summary`: `{epochs, batch_size, learning_rate}`
- `best_loss`: numeric value when available
- `best_loss_display`: preformatted string when available
- `recent_logs`: array of `{timestamp, timestamp_display, message, display}`

Normalized deployment fields exposed by `POST /api/training/export/{job_id}`:

- `deployment.target_label`: browser-facing label for the resolved deployment target

## Supported Contracts

### `stt-form-v1`

Used by:

- `whisper`
- `qwen3-asr`
- `parakeet`
- `canary`

Request:

- `POST /transcribe`
- multipart field `audio`
- optional form field `language`

Response:

- `text`: string
- `segments`: array of `{start, end, text, ...}` when available
- `language`: optional string
- `duration`: optional number (it is what lets `benchmarks/run_german_eval.py` report a real-time factor)

Status codes every implementation shares: **400** for an empty file or an unknown `language`/`task`,
**413** when the upload exceeds the service's `MAX_UPLOAD_MB` or the recording its length limit,
**422** for audio that cannot be decoded (and, on Canary, for a language its model cannot decode),
**503** while the model cannot be loaded or the GPU is out of memory, and, on `qwen3-asr`, `parakeet`
and `canary`, when the service is busy: `ASR_MAX_CONCURRENCY + ASR_MAX_QUEUE` requests are already
admitted, or this one waited `ASR_QUEUE_TIMEOUT_S` for its turn (always with `Retry-After`).
`GET /ready` (see [`api.md`](./api.md#health-versus-ready-on-the-backends)) never blocks on a load.

Per-service differences worth knowing:

- `qwen3-asr` cuts a recording into pieces of `QWEN3_ASR_CHUNK_S` seconds and adds `chunks`,
  `truncated` and `warnings` at the top level and `language` and `truncated` on each segment.
  Segments carry no `confidence`, because the model has none. `/detect_language` analyses only the
  first piece.
- `parakeet` and `canary` add a per-file `status` to the error entries of `POST /transcribe-batch`.
  `GET /status` reports the NeMo, torch and CUDA versions the image was built with under `runtime`.
- `whisper` (`stt-service`) takes `language=null` to mean `STT_DEFAULT_LANGUAGE` (empty: detect) and
  reports `preferred_device`, `preferred_compute_type` and `degraded` in `/health`.

### `stt-detect-language-v1`

Used by `whisper` and `qwen3-asr`: `POST /detect_language` with the same multipart `audio`, returning
the detected `language` and its probability. Parakeet has the route but always returns `null`, and
Canary has no language identification at all: the registry's `language_detect` field says which
providers can actually do it.

### `openai-audio-transcriptions-v1`

Used by:

- `whisper-cpp`

This contract family is optional in the default stack because the frontend only exposes `whisper-cpp` when `ENABLE_WHISPER_CPP=true`.

Request:

- `POST /v1/audio/transcriptions`
- multipart field `file`
- optional form field `language`
- optional form field `response_format=json`

Normalized UI response:

- `text`: string
- `segments`: optional normalized array when provider returns segment data
- `language`: optional string
- `duration`: optional number

### `simple-json-tts-v1`

Used by:

- `piper`
- `qwen3` for basic built-in speaker synthesis via frontend adapter mapping
- `chatterbox`

Request:

- `POST /tts`
- JSON body with `text`
- optional `voice`, `language`, `quality`, `gender`, `speed`, `output_format`

Notes:

- Every TTS backend exposes `GET /health` (liveness) and `GET /ready` (Piper: 503 without an installed voice; the model backends: 503 while the first load runs or after it failed).
- Busy is **503 + `Retry-After`**, never a hang: `qwen3` and `chatterbox` admit `TTS_MAX_CONCURRENCY + TTS_MAX_QUEUE` requests and refuse the next at once, and refuse one that waited `TTS_QUEUE_TIMEOUT_S`; Piper refuses a request when no synthesis slot frees up within `PIPER_TIMEOUT_S`.
- An unexpected failure is a `500` with a generic message and a request id (also in an `X-Request-ID` header on Piper, Qwen3-TTS and Chatterbox); the detail is in the service log under that id.
- Qwen3 does not natively expose this exact payload. The frontend adapter translates the shared fields into the provider's native `lang`, `speaker`, and `instruct` request schema for basic TTS.

Response:

- audio binary payload

### `voice-catalog-v1`, `speaker-catalog-v1`

`voice-catalog-v1` (Piper): `GET /voices` lists the **installed** voices grouped by language, plus
`default_language`, `default_voice` and `catalog_only` (true when the image ships no voices);
`POST /refresh_voices` rescans the models directory and also returns `default_voices`. The gateway's
`GET /api/providers/piper/voices` normalises it. `speaker-catalog-v1` (Qwen3-TTS): `GET /speakers`,
which is empty on the Base models (they clone voices; built-in speakers need a CustomVoice variant)
and reports `speakers_source`.

### `saved-voice-library-v1`

Used by `qwen3`. A saved voice is created from reference audio (`name`, `lang`, `file`) with an
optional `mode` (`xvector`, the default, stores the speaker embedding only; `icl` also keeps the
reference text, given as `ref_text` or transcribed) and answers with `mode` and `ref_text_source`. `POST /voices/{voice_id}/tts` synthesises with it; a `/tts` request against a
model that has no built-in speakers is **409**, not 400.

### `chunked-wav-stream-v1`

Used by `chatterbox` (`POST /tts-stream`): a WAV header followed by audio generated sentence by
sentence, with `CHATTERBOX_CHUNK_GAP_MS` of silence between sentences. Text longer than
`MAX_TEXT_CHARS` is 413 before any audio is sent; a model that fails to load is an HTTP 500 with a
detail, not a truncated stream. `POST /tts` and `/clone` use the same chunking and report
`X-Chunk-Count`.

### `model-catalog-v1`

Used by:

- `qwen3`

Notes:

- exposes a normalized model list for advanced TTS backends that support runtime model variants

### `model-selection-v1`

Used by:

- `qwen3`

Notes:

- switches the currently active model variant for an advanced TTS provider

### `runtime-status-v1`

Used by:

- `qwen3`

Notes:

- exposes provider runtime metadata such as loaded model, device, memory usage, and built-in speaker availability

### `voice-clone-tts-v1`

Used by:

- `qwen3`

Notes:

- accepts multipart voice-cloning inputs and returns synthesized audio
- the frontend selects the provider-native backend path based on whether reference text was supplied
- a reference clip that decodes to more than `QWEN3_TTS_REF_MAX_SECONDS` (60) is **413**, whatever its file size; one that cannot be decoded, or is empty, is **400**

### `voice-design-tts-v1`

Used by:

- `qwen3`

Notes:

- accepts text plus a target voice description and returns synthesized audio

### `custom-voice-library-v1`

Used by:

- `piper`

Notes:

- exposes provider-managed custom voice lifecycle operations separate from the generic voice catalog
- the frontend currently uses this for listing and deleting trained Piper voices without direct browser calls to the Piper service
- `POST /upload_model` validates a voice before publishing it: the model loads and answers a test request in a throw-away child process (`PIPER_ONNX_VALIDATE_TIMEOUT_S`, 30 s), otherwise **400**. It is refused with **409** for the id of a built-in voice or when `PIPER_MAX_CUSTOM_VOICES` (20; `0` = no new upload) voices are installed, **413** when the custom voices would exceed `PIPER_MAX_CUSTOM_MB` (2048) in total, and **503** with `Retry-After` while another upload is running
- `POST /analyze_audio` refuses a recording that decodes to more than `PIPER_ANALYZE_MAX_SECONDS` (600) with **413**, whatever its file size

### `voice-training-job-v1`

Used by:

- `piper-training`

Notes:

- training now separates export from deployment target selection
- deployment is governed by explicit target contracts such as `manual-artifact-v1`, `piper-shared-volume-v1`, and `piper-upload-api-v1`
- the training runtime is still Piper-oriented in model bundle format, but no longer assumes Piper as the only active deployment path
- one job runs at a time by default (`TRAINING_MAX_CONCURRENT`); a second `POST /train`, `/train-from-dataset`, `/retrain-from-segments` or `/resume-training` is **409** naming the running job
- job status carries `trainer_kind` and `trainer_caveat`, and `GET /ready` reports `accepting_jobs`, `active_jobs`, storage and device checks
- `/prepare-dataset` reads audio only from `data/` plus `TRAINING_ALLOWED_AUDIO_DIRS` and downloads only from `TRAINING_ALLOWED_URL_HOSTS`; without an allowed host a URL is refused

## Behaviour every backend shares

These hold for all eight backend services (`stt-service`, `qwen3-asr`, `parakeet`, `canary`,
`piper-tts`, `qwen3-tts`, `chatterbox`, `piper-training`), independent of the contract they
implement. `whisper-cpp` is the unmodified upstream server and has none of them.

- **Closed to browsers by default.** `ALLOWED_ORIGINS` (compose: `BACKEND_ALLOWED_ORIGINS`) is empty
  = no CORS headers, and a state-changing request (anything but `GET`, `HEAD`, `OPTIONS`, and a
  WebSocket handshake) that carries an `Origin` header naming a different host than its `Host` is
  **403**. Requests without an `Origin` (the gateway, `curl`, benchmarks) are unaffected. `*` must be
  written down and is logged as a warning. It is a check against browser-borne requests, not
  authentication.
- **Request bodies are bounded while they arrive.** A declared `Content-Length` over the limit is
  **413** before a byte is read, and the bytes actually received are counted too, so a chunked upload
  or a lying header cannot get past it. Nothing beyond the limit is spooled or buffered.
- **Busy is 503 + `Retry-After`,** conflicts (a built-in voice, a limit of stored items, an in-flight
  request on `/unload`) are **409**, and over-limit input is **413**. The variables that set them
  are in [`api.md`](./api.md#limits-per-service).
- **Errors do not leak.** An unexpected failure is a generic message with a request id; the detail is
  logged under that id, and the TTS backends also send it in `X-Request-ID`.

## Capability Guidelines

Capabilities should describe what a provider can do, not which brand it belongs to.

Good capability names:

- `transcribe`
- `segments`
- `detect_language`
- `tts`
- `voice_clone`
- `saved_voices`
- `model_switching`
- `voice_training`
- `model_export`

Avoid using provider names as capabilities.

## Integration Rule

When adding a new backend:

1. Prefer matching an existing contract.
2. If that is not possible, add a small adapter before adding new UI branching.
3. Only create a new contract when the provider exposes genuinely different semantics.
4. Add or extend contract tests alongside the integration.