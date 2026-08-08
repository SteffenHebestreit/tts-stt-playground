# API reference

This project is used primarily as a speech API. There are two surfaces:

| Surface | For | Stability |
|---|---|---|
| **`/v1/*`** | OpenAI-compatible. Use this for anything new. | Follows the OpenAI spec |
| `/api/*` | The web UI's own contract. Richer, but project-specific. | Stable, unchanged |

Everything is served by the **frontend gateway on port 3000**. Backend ports do not need
publishing — see [`device-deployment-matrix.md`](./device-deployment-matrix.md).

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

> `api_key` is required by the SDK itself — it raises at construction without one — but this
> deployment ignores the value. There is no authentication yet; do not expose port 3000 to an
> untrusted network.

```bash
curl http://your-host:3000/v1/audio/transcriptions \
  -F file=@audio.wav -F model=whisper-1 -F language=de

curl http://your-host:3000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model":"tts-1","voice":"de_DE-thorsten-medium","input":"Guten Tag.","response_format":"wav"}' \
  --output out.mp3
```

---

## `POST /v1/audio/transcriptions`

`multipart/form-data`.

| Field | Default | Notes |
|---|---|---|
| `file` | — | **Required.** Max 25 MB. |
| `model` | — | Accepted but **advisory** — this deployment serves whatever backend it has. Any string works. |
| `language` | auto-detect | ISO-639-1 (`de`, `en`, …). Omit or send `auto` to detect. |
| `prompt` | — | Forwarded as `initial_prompt` where the backend supports it. |
| `response_format` | `json` | `json` or `text`. |
| `temperature` | `0` | Forwarded where supported. |
| `stream` | `false` | Accepted and ignored (spec-legal for `whisper-1`). |

Any other spec field (`timestamp_granularities`, `chunking_strategy`, `keywords`, `include`, …)
is **accepted and ignored** rather than rejected — the spec grows, and a 422 on an unrecognised
field breaks clients that send newer parameters harmlessly.

```jsonc
// response_format=json
{ "text": "guten tag" }
```

`response_format=text` returns a **raw body**, not JSON — `text/plain; charset=utf-8`.

`verbose_json`, `srt` and `vtt` return **400** with a clear message. `diarized_json` is rejected
explicitly rather than silently downgraded.

---

## `POST /v1/audio/speech`

JSON body (**not** multipart).

| Field | Default | Notes |
|---|---|---|
| `model` | — | **Required** by the spec; advisory here. |
| `input` | — | **Required.** Max 4096 characters. |
| `voice` | — | **Required** by the spec. Never validated against a list — see below. |
| `response_format` | `mp3` | `mp3` or `wav`. |
| `speed` | `1.0` | 0.25–4.0. |

Returns raw audio bytes: `audio/mpeg` or `audio/wav`.

**On `voice`:** OpenAI's own spec is internally inconsistent here (its prose names 13 voices, its
`VoiceIdsShared` enum has 10, and the schema accepts any string), so this deployment never 404s on
a voice name. OpenAI's placeholder names (`alloy`, `nova`, …) are recognised and mapped to the
deployment default; anything else is passed through as one of *your* voices, e.g.
`de_DE-thorsten-medium`. List them at `GET /api/providers/piper/voices`.

**On `mp3`:** it is the spec default, and the TTS backends emit WAV, so the gateway transcodes with
ffmpeg. If ffmpeg is missing the endpoint returns **501** naming `wav` as the alternative rather
than silently returning a WAV labelled as MP3.

---

## `GET /v1/models`

```jsonc
{ "object": "list",
  "data": [ { "id": "whisper-1", "object": "model", "created": 1677610602, "owned_by": "tts-stt" },
            { "id": "tts-1",     "object": "model", "created": 1677610602, "owned_by": "tts-stt" } ] }
```

`GET /v1/models/{id}` returns a single entry, or 404 in the error envelope.

The ids are deliberately the classic OpenAI ones. Advertising `whisper-1` makes *ignoring* `stream`
and *honouring* `timestamp_granularities` both spec-legal, which matches this project's actual
capabilities.

---

## Errors

Every `/v1` error uses the OpenAI envelope, because clients branch on these fields:

```jsonc
{ "error": { "message": "…", "type": "invalid_request_error", "param": "response_format", "code": "invalid_value" } }
```

`type` is one of `invalid_request_error` (4xx), `authentication_error` (401),
`rate_limit_error` (429), `server_error` (5xx).

`/api/*` keeps FastAPI's `{"detail": …}` — unchanged.

---

## Language handling

German is a hard requirement of this project, and language handling is where it is easiest to get
a plausible-looking wrong answer. Two behaviours worth knowing:

**Auto-detection is backend-specific and the gateway normalises it.** whisper.cpp defaults to
English when no language is supplied, while faster-whisper rejects the literal string `auto`. The
gateway sends `auto` explicitly to the former and omits the field for the latter. Sending
`language=auto` therefore genuinely auto-detects on every device.

**A TTS voice that cannot serve the requested language is reported.** If you ask for German on a
deployment with no German voice, Piper substitutes an English one. That is signalled rather than
silent:

| Header | Meaning |
|---|---|
| `X-Language-Requested` | What you asked for |
| `X-Language` | The language of the voice actually used |
| `X-Language-Fallback` | `true` if a substitution happened |

Set `PIPER_STRICT_LANGUAGE=true` to get a **400** listing the available languages instead.

### Which providers can actually detect language

Some backends expose a `/detect_language` route that always returns `null`. The registry reports
the truth in a machine-readable `language_detect` field:

| Provider | Detects? |
|---|---|
| `whisper` (faster-whisper) | yes |
| `qwen3-asr` | yes |
| `whisper-cpp` | yes (via `language=auto`) |
| `parakeet` | **no** — route exists, always returns null |
| `canary` | **no** — no language identification at all |

---

## Discovery

`GET /providers` returns the full registry: which backends this deployment has, their
capabilities, contracts and settings. Use it to find voice names and to check whether a capability
is present before relying on it.

`GET /api/health` probes every backend concurrently and reports `healthy`, `latency_ms`,
`model_resident` and `device` per provider.

> `model_resident: false` is **not** an error. Idle models are unloaded to free VRAM and the next
> request reloads them. See `*_MODEL_TTL` in [`.env.example`](../.env.example).

---

## Not implemented

Deliberately, because no mainstream client exercises them — the research checked
openai-python, openai-node, LangChain, Home Assistant and Open WebUI:

`verbose_json` / `srt` / `vtt`, `timestamp_granularities`, `POST /v1/audio/translations`,
`opus`/`aac`/`flac`/`pcm`, `stream=true`, `stream_format=sse`, diarization, and token `usage`
accounting.

The `/api/*` surface still offers **segment timestamps** (`/api/stt`) and **streaming TTS**
(`/api/tts` against a provider declaring `tts_stream`) if you need them today.
