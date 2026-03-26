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
      "browser_url": "http://localhost:port",
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

Optional providers may be omitted from the registry entirely. In the current frontend implementation, `whisper-cpp` is only registered when `ENABLE_WHISPER_CPP=true` is set for `frontend-service`.

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
- passthrough `X-*` headers from backend provider when present
- `X-Provider` header added by the adapter

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

Request:

- `POST /transcribe`
- multipart field `audio`
- optional form field `language`

Response:

- `text`: string
- `segments`: array of `{start, end, text, ...}` when available
- `language`: optional string
- `duration`: optional number

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

Request:

- `POST /tts`
- JSON body with `text`
- optional `voice`, `language`, `quality`, `gender`, `speed`, `output_format`

Notes:

- Qwen3 does not natively expose this exact payload. The frontend adapter translates the shared fields into the provider's native `lang`, `speaker`, and `instruct` request schema for basic TTS.

Response:

- audio binary payload

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

### `voice-training-job-v1`

Used by:

- `piper-training`

Notes:

- training now separates export from deployment target selection
- deployment is governed by explicit target contracts such as `manual-artifact-v1`, `piper-shared-volume-v1`, and `piper-upload-api-v1`
- the training runtime is still Piper-oriented in model bundle format, but no longer assumes Piper as the only active deployment path

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