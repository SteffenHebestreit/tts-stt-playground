# API reference

This project is used primarily as a speech API. There are two surfaces:

| Surface | For | Stability |
|---|---|---|
| **`/v1/*`** | OpenAI-compatible. Use this for anything new. | Follows the OpenAI spec |
| `/api/*` | The web UI's own contract. Richer, but project-specific. | Stable, additive |

Everything is served by the **frontend gateway on port 3000**. The backends are
reachable from the machine itself for `curl` and benchmarks, but not from the network — see
[Network exposure](#network-exposure-and-access-control).

The machine-readable specs are served by the gateway too: `/static/openapi/gateway.json` (the
gateway's own `/v1` and `/api` routes, i.e. this page), and the backends' `stt.json`, `pipertts.json`
and `training.json`. They are generated from the
running apps (`scripts/sync_openapi.py`) and checked against them in CI, so they cannot describe
routes that do not exist.

---

## Network exposure and access control

**Ports.** `docker-compose.yml` publishes the gateway on `FRONTEND_BIND_ADDR:FRONTEND_PORT`
(default `0.0.0.0:3000`) and every backend on `BACKEND_BIND_ADDR` (default **`127.0.0.1`**). The
backends have no authentication, so by default they answer on the host's loopback only; the UI
never needs them, because the gateway proxies every call, the live microphone WebSocket included,
over the internal Docker network. Set `BACKEND_BIND_ADDR=0.0.0.0` only on a network you trust.
Services never reach each other through published ports.

**Browsers.** `ALLOWED_ORIGINS` is **empty by default**: the gateway answers its own origin only,
sends no CORS headers, and refuses a state-changing request (`POST`, `PUT`, `PATCH`, `DELETE`)
that carries a foreign `Origin` with **403** (`cross_origin_blocked`). Opening the UI by another
hostname needs nothing. List origins to let other pages call the API; `*` opens it to any page and
is logged as a warning. Behind a reverse proxy that rewrites the `Host` header, add the public
URL to `TRUSTED_ORIGINS` (the symptom of forgetting it is a 403 on every button); set
`TRUST_PROXY_HEADERS=true` only if the proxy overwrites `X-Forwarded-Host`. Requests without an
`Origin` header (curl, SDKs) are unaffected.

**API key.** Optional. With `API_KEY` set, `/v1/*` needs `Authorization: Bearer <key>` and so
does every state-changing `/api/*` call, except requests the bundled UI itself makes (a
same-origin browser request). Reads (`GET /api/...`), CORS preflights, `/health` and the pages
stay open; a wrong or missing key is **401** with `WWW-Authenticate: Bearer`. The check is
constant-time. This protects an exposed API port; it is not user authentication, because a
script can forge a browser's headers, and it does not guard `/ws/stt` (browsers cannot send an
`Authorization` header on a WebSocket; the origin rule applies there instead).

**Limits.** Request bodies larger than `MAX_UPLOAD_MB` (default 512, sized for the UI's
multi-file training upload) get **413**, as do JSON bodies over 1 MiB; `/v1/audio/transcriptions`
takes at most 25 MB per file. Text fields for TTS and voice design are capped at `MAX_TTS_CHARS`
(default 20000, 422 beyond it). Each backend also enforces its own, usually smaller, limit; see
[Limits per service](#limits-per-service).

---

## Why `/v1` exists

The backend differs per device: the RK3588S and Strix Halo run `whisper-cpp`, the NVIDIA boxes
run `faster-whisper`. Those speak different native contracts.

**`/v1` makes that invisible.** The same request returns the same response shape on every device.
There is a test asserting exactly this (`tests/test_openai_v1_api.py::test_identical_shape_across_providers`) —
if it ever fails, this surface has lost the only property it exists for.

---

## Using it with an OpenAI client

No custom client needed. Point the official SDK at the gateway.

```python
from openai import OpenAI

client = OpenAI(base_url="http://your-host:3000/v1", api_key="unused")

with open("audio.wav", "rb") as f:
    print(client.audio.transcriptions.create(model="whisper-1", file=f).text)

speech = client.audio.speech.create(model="tts-1", voice="de_DE-thorsten-medium",
                                    input="Guten Tag.", response_format="wav")
speech.write_to_file("out.wav")
```

> `api_key` is required by the SDK itself — it raises at construction without one. The value is
> ignored unless the gateway was started with `API_KEY`, in which case pass that key. Without
> `API_KEY` there is no authentication: keep port 3000 on a network you trust.

```bash
curl http://your-host:3000/v1/audio/transcriptions \
  -F file=@audio.wav -F model=whisper-1 -F language=de

curl http://your-host:3000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model":"tts-1","voice":"de_DE-thorsten-medium","input":"Guten Tag.","response_format":"mp3"}' \
  --output out.mp3
```

---

## `POST /v1/audio/transcriptions`

`multipart/form-data`.

| Field | Default | Notes |
|---|---|---|
| `file` | — | **Required.** Max 25 MB (413 `file_too_large`), must not be empty. |
| `model` | — | Accepted but **advisory** — this deployment serves whatever backend it has. Any string works. |
| `language` | auto-detect | ISO-639-1 (`de`, `en`, …). Omit or send `auto` to detect. |
| `prompt` | — | Forwarded as `initial_prompt` where the backend supports it. |
| `response_format` | `json` | `json`, `text`, `verbose_json`, `srt` or `vtt`. |
| `temperature` | `0` | Forwarded where supported. |
| `stream` | `false` | Accepted and ignored (spec-legal for `whisper-1`). |

Any other spec field (`timestamp_granularities`, `chunking_strategy`, `keywords`, `include`, …)
is **accepted and ignored** rather than rejected — the spec grows, and a 422 on an unrecognised
field breaks clients that send newer parameters harmlessly.

```jsonc
// response_format=json
{ "text": "guten tag" }

// response_format=verbose_json (segment fields abridged)
{ "task": "transcribe", "language": "de", "duration": 3.2, "text": "guten tag",
  "segments": [ { "id": 0, "start": 0.0, "end": 3.2, "text": "guten tag", "avg_logprob": -0.2, "no_speech_prob": 0.01 } ] }
```

`response_format=text`, `srt` and `vtt` return a **raw body**, not JSON (`text/plain;
charset=utf-8`). `verbose_json` fills the fields openai-python validates (`seek`, `tokens`,
`compression_ratio`, …) with neutral values where a backend cannot measure them. Segments come
from the backend; a backend that reports none gets one segment spanning the file. `diarized_json` is rejected (400 `unsupported_value`) rather
than silently downgraded, and an unknown value is 400 `invalid_value`.

**Which provider answered.** The response carries `X-Provider: <id>`. If the *default* provider
could not be reached at all (connection refused or timed out) and exactly one other STT provider is
healthy, the request is retried there once and the response also carries
`X-Provider-Fallback: <default>-><used>` (and a warning is logged). A backend that answers with an
error, or times out while working, is **not** retried: that would run the job twice. `/api/stt`
never falls back; it does what you asked.

---

## `POST /v1/audio/speech`

JSON body (**not** multipart).

| Field | Default | Notes |
|---|---|---|
| `model` | — | **Required** by the spec; advisory here. |
| `input` | — | **Required.** Max 4096 characters (400 `string_above_max_length`). |
| `voice` | — | **Required** by the spec. Never validated against a list — see below. |
| `response_format` | `mp3` | `mp3`, `wav` or `pcm`. |
| `speed` | `1.0` | 0.25–4.0. |

Returns raw audio bytes: `audio/mpeg`, `audio/wav`, or headerless 16-bit mono PCM (`audio/pcm`).
The response carries `X-Provider`.

**On `voice`:** OpenAI's own spec is internally inconsistent here (its prose names 13 voices, its
`VoiceIdsShared` enum has 10, and the schema accepts any string), so this deployment never 404s on
a voice name. OpenAI's placeholder names (`alloy`, `nova`, …) are recognised and mapped to the
deployment default; anything else is passed through as one of *your* voices, e.g.
`de_DE-thorsten-medium`. List them at `GET /api/providers/piper/voices`.

**On `mp3`:** it is the spec default, and the TTS backends emit WAV, so the gateway transcodes with
ffmpeg (asynchronously, killed after 120 s). If ffmpeg is missing the endpoint returns **501**
naming `wav` as the alternative rather than silently returning a WAV labelled as MP3.

**On `pcm`:** the WAV header is stripped and stereo is mixed down to mono at the backend's **native
sample rate**, reported in `X-Sample-Rate` (Piper is 22050 Hz). OpenAI's PCM is 24 kHz; a client that
hard-codes that must read the header, or resample. It is not resampled here so the format does not
depend on ffmpeg being installed.

---

## `GET /v1/models`

```jsonc
{ "object": "list",
  "data": [ { "id": "whisper-1", "object": "model", "created": 1677610602, "owned_by": "tts-stt" },
            { "id": "tts-1",     "object": "model", "created": 1677610602, "owned_by": "tts-stt" } ] }
```

`GET /v1/models/{id}` returns a single entry, or 404 (`model_not_found`) in the error envelope.

The ids are deliberately the classic OpenAI ones. Advertising `whisper-1` makes *ignoring* `stream`
and *honouring* `timestamp_granularities` both spec-legal, which matches this project's actual
capabilities.

---

## Errors

Every `/v1` error uses the OpenAI envelope, because clients branch on these fields — including
framework errors (a missing `file`, `temperature=abc`, an unknown route, a wrong method, a crash),
which FastAPI would otherwise answer with `{"detail": …}`:

```jsonc
{ "error": { "message": "…", "type": "invalid_request_error", "param": "response_format", "code": "invalid_value" } }
```

| Status | `type` | Typical `code` |
|---|---|---|
| 400, 404, 413, 415, 422 | `invalid_request_error` | `invalid_value`, `unsupported_value`, `file_too_large`, `request_too_large`, `string_above_max_length`, `model_not_found` |
| 401 | `authentication_error` | `invalid_api_key` (with `WWW-Authenticate: Bearer`) |
| 403 | `permission_error` | `cross_origin_blocked` |
| 429 | `rate_limit_error` | — |
| 5xx | `server_error` | — |

`/api/*` keeps FastAPI's `{"detail": …}`. A backend that fails is reported by name:
**503** when it cannot be reached, **502** when it answers 200 with something that is not the JSON
it promised, and any status a backend answers with (400, 409, 413, 422, 503, …) is relayed with its
own `detail`. Path ids (`job_id`, `voice_id`) are restricted to `[A-Za-z0-9_-]{1,128}`; anything
else is 422 and never reaches a backend.

---

## Language handling

German is a hard requirement of this project, and language handling is where it is easiest to get
a plausible-looking wrong answer. Behaviours worth knowing:

**Auto-detection is backend-specific and the gateway normalises it.** whisper.cpp defaults to
English when no language is supplied, while faster-whisper rejects the literal string `auto`. The
gateway sends `auto` explicitly to the former and omits the field for the latter. Sending
`language=auto` therefore genuinely auto-detects on every device. `STT_DEFAULT_LANGUAGE` sets what
`stt-service` uses when a request names none (empty = detect); the web UI always sends `auto`, so
it affects API callers that omit the field.

**Unsupported languages are refused, not guessed.** `stt-service` answers 400 for an unknown
language code, Canary answers 422 for a language its model cannot decode (it used to decode
everything as German), and Qwen3-TTS answers 400.

**TTS defaults to German.** A request that names no language (or `auto`, in any capitalisation) is
spoken in `PIPER_DEFAULT_LANGUAGE` (`de`) after Piper has guessed German versus English from the
text (`PIPER_AUTO_DETECT`); Qwen3-TTS uses `QWEN3_DEFAULT_LANGUAGE` (`German`), Chatterbox
`CHATTERBOX_DEFAULT_LANGUAGE` (`de`).

**A TTS voice that cannot serve the requested language is reported.** If you ask for German on a
deployment with no German voice, Piper substitutes another one. That is signalled rather than
silent:

| Header | Meaning |
|---|---|
| `X-Language-Requested` | What you asked for (`auto` for any spelling of auto, `invalid` for a value that is not a language tag) |
| `X-Language` | The language of the voice actually used |
| `X-Language-Fallback` | `true` if a substitution happened |

Set `PIPER_STRICT_LANGUAGE=true` to get a **400** listing the available languages instead.

### Which providers can actually detect language

Some backends expose a `/detect_language` route that always returns `null`. The registry reports
the truth in a machine-readable `language_detect` field:

| Provider | Detects? |
|---|---|
| `whisper` (faster-whisper) | yes |
| `qwen3-asr` | yes (from the first piece of the recording only, 30 s by default) |
| `whisper-cpp` | yes (via `language=auto`) |
| `parakeet` | **no** — route exists, always returns null |
| `canary` | **no** — no language identification at all |

---

## Discovery and health

`GET /providers` returns the full registry: which backends this deployment has, their
capabilities, contracts and settings. Use it to find voice names and to check whether a capability
is present before relying on it.

The `canary` entry's language list follows the checkpoint the Canary service runs: the gateway reads
`supported_languages` from the service's `GET /status` (refreshed every five minutes), so
`CANARY_ASR_MODEL=nvidia/canary-1b-v2` shows 25 languages and the flash models four. While the
service cannot be reached the entry falls back to `de`, `en`, `es`, `fr` (asked again after 20
seconds), and after a first answer it keeps the last one. An entry you replace through
`PROVIDER_REGISTRY_JSON` is left as you wrote it.

`GET /health` on the gateway is liveness only (the Docker healthcheck).

`GET /api/health` probes every backend concurrently and reports `healthy`, `latency_ms`,
`model_resident` and `device` per provider. The result is reused for `HEALTH_CACHE_TTL` seconds
(default 2; 0 = always probe) and concurrent callers share one probe round.

> `model_resident: false` is **not** an error. Idle models are unloaded to free VRAM and the next
> request reloads them. See `*_MODEL_TTL` in [`.env.example`](../.env.example).

### `/health` versus `/ready` on the backends

Every model service answers two questions separately:

| | `GET /health` | `GET /ready` |
|---|---|---|
| Asks | Is the process alive? | Can it serve a request now? |
| While the first model download or load runs | 200 | **503** `{"ready": false, "reason": "loading"}` (with `Retry-After`) |
| After a failed load | 200 | **503** `{"ready": false, "reason": "load_failed", "detail": …}` |
| Model unloaded by the idle TTL | 200 | **200** — it will reload on the next request |
| Loads a model or waits for one | never | never |

The two model-less services answer `/ready` about their own preconditions instead: Piper is 503
(`problems`: no voice installed in the models directory, no `piper` binary, an output directory it
cannot write) and the training service is 503 with the list of what is wrong.

Docker healthchecks and `depends_on` use `/health` only: a first model download takes minutes and
optional backends may be absent, so gating on readiness would keep the gateway (and with it the
whole UI) down for as long as the slowest backend. Use `/ready` in setup scripts and operator
checks, for example `curl -fsS http://127.0.0.1:5001/ready` after `docker compose up`. The
training service's `/ready` names the things that usually break a first run: a data directory the
container cannot write, a PyTorch build without kernels for the GPU, and low disk. Backends whose
ports are published on loopback are reachable there; otherwise use `docker compose exec`.
`whisper-cpp` is the unmodified upstream server: it has no `/ready`, and its compose healthcheck
only requires an HTTP answer.

---

## Freeing VRAM on demand

`POST /api/providers/{provider}/unload` releases a model and its memory immediately. The idle TTL
already does this on its own schedule; this is for when you are about to run something else on the
same GPU and want the memory back now. The next request reloads transparently, so it is always safe
to call.

| Status | Meaning |
|---|---|
| `200` `{"unloaded": true, "reason": "ok"}` | Released |
| `200` `{"unloaded": false, "reason": "not_resident"}` | Already unloaded — not an error |
| `409` `{"reason": "busy", ...}` | A request is in flight (or a load is running). Retry when idle. |

The **409 is the point**. Freeing memory a running decode is still reading would crash the worker,
so every service refuses while a reference is outstanding rather than unloading and hoping. The
response carries the outstanding count (`model_refs`, or `inflight` on qwen3-tts) so you know
whether a retry is worthwhile. A client that disconnects mid-request no longer leaves a reference
behind that would answer 409 forever.

Supported where the provider lists the `model_unload` capability — currently `whisper`,
`qwen3-asr`, `qwen3` (TTS), `chatterbox`, `parakeet` and `canary`. Asking any other provider
returns 400. The two that cannot: `piper` is CPU-only ONNX with no VRAM to reclaim, and
`whisper-cpp` is the upstream whisper-server binary with no Python layer to add a route to.

```bash
curl -X POST http://your-host:3000/api/providers/whisper/unload
```

On qwen3-tts this also clears a model selected via `/load_model`; the next request reloads whichever
model is currently desired, not the environment default. `POST /load_model` waits up to 180 s for
in-flight requests to drain before answering 409.

---

## Limits per service

Each backend bounds its own input and answers **413** beyond it. The variables are per service on
purpose (the container-side name `MAX_UPLOAD_MB` means something different in each), so compose
gives each its own host variable:

| Service | Upload | Length or text | Host variables |
|---|---|---|---|
| gateway | 512 MB body, 1 MiB JSON, 25 MB per `/v1` file | 20000 characters of TTS text | `MAX_UPLOAD_MB`, `MAX_TTS_CHARS` |
| stt-service, qwen3-asr, parakeet, canary | 200 MB per file | 7200 s (stt, qwen3-asr) or 1500 s (NeMo) per recording | `ASR_MAX_UPLOAD_MB`, `ASR_MAX_AUDIO_SECONDS`, `NEMO_MAX_AUDIO_S` |
| piper-tts | 100 MB model upload, 25 MB analysed audio | 20000 characters | `PIPER_MAX_UPLOAD_MB`, `PIPER_MAX_ANALYZE_UPLOAD_MB`, `PIPER_MAX_TEXT_CHARS` |
| qwen3-tts | 20 MB reference audio | 5000 characters | `QWEN3_TTS_MAX_UPLOAD_MB`, `QWEN3_TTS_MAX_TEXT_CHARS` |
| chatterbox | 20 MB reference audio | 5000 characters | `CHATTERBOX_MAX_UPLOAD_MB`, `CHATTERBOX_MAX_TEXT_CHARS` |
| piper-training | 500 MB per `/train` request | 1000 characters per segment | `TRAINING_MAX_UPLOAD_MB`, `TRAINING_MAX_TEXT_CHARS` |

The gateway accepts more text than Qwen3-TTS and Chatterbox, so a long text can pass the gateway
and be refused by the backend; its 413 is relayed with the backend's own message. Chatterbox used
to cut such audio off at about 40 s without saying so. The upload limits bound what a service
copies to disk and decodes; a multipart body is already spooled by the framework before a handler
runs, so put a reverse-proxy limit in front if wire bytes matter.

---

## Not implemented

Deliberately, because no mainstream client exercises them — the research checked
openai-python, openai-node, LangChain, Home Assistant and Open WebUI:

`timestamp_granularities`, `POST /v1/audio/translations`, `opus`/`aac`/`flac` speech output,
`stream=true`, `stream_format=sse`, diarization, and token `usage` accounting.

The `/api/*` surface still offers **segment timestamps** (`/api/stt`) and **streaming TTS**
(`/api/tts` against a provider declaring `tts_stream`) if you need them today.
