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
| GET | `/health` | Liveness, residency, device, and multilingual status. Never loads or waits for the model |
| GET | `/ready` | Can it serve a request now? `200` resident or reloadable, `503` while loading or after a failed load |
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
| `task` | `transcribe` | `translate` is supported, but only to English and only by models that can (see [Model Notes](#model-notes)) |
| `language` | `STT_DEFAULT_LANGUAGE`, else auto-detect | A code like `en`, `de`, `fr`; locale tags such as `de-DE` are reduced to `de`. `auto` always auto-detects. An unknown code is a `400` |
| `beam_size` | `5` | Higher is slower but can improve quality |
| `vad_filter` | `true` | Disable for short/noisy clips if speech is dropped |
| `vad_threshold` | `0.5` | Lower for more permissive speech detection |
| `no_speech_threshold` | `0.6` | Raise for compressed browser audio |

### Limits

Uploads are copied to disk in chunks, never read into memory whole.

| Limit | Default | Result |
|-------|---------|--------|
| `MAX_UPLOAD_MB` per file | `200` | `413` |
| `MAX_AUDIO_SECONDS` per file | `7200` | `413`, checked from the container header before the file is decoded (and again after decoding for files without one). An abuse guard, not a tuning knob |

The framework has already spooled a multipart request to disk by the time the
handler runs, so the upload limit bounds what is copied and decoded, not the
bytes that crossed the wire. Put a request-size limit in front of the service if
that matters. The gateway has its own `MAX_UPLOAD_MB` (default 512), so a file
between the two limits is accepted there and refused here.

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `WHISPER_MODEL_SIZE` | `large-v3-turbo` | A built-in size (`tiny`, `base`, `small`, `medium`, `large-v3`, `large-v3-turbo`), a Hugging Face repo id, or the path of a CTranslate2 export — see [German fine-tunes](#using-a-german-fine-tune) |
| `WHISPER_COMPUTE_TYPE` | unset | Pin the CT2 compute type. Empty probes the GPU and picks the most memory-efficient supported option. |
| `WHISPER_OOM_FALLBACK` | `small` | Smaller multilingual model tried before abandoning the GPU |
| `WHISPER_NUM_WORKERS` | `2` | CT2 workers on the shared model; gives live and batch traffic independent slots |
| `WHISPER_MODEL_DIR` | unset | Where models are downloaded. Unset keeps the Hugging Face cache (`/root/.cache/huggingface`), where existing installs already have their weights; setting it moves the cache and re-downloads once |
| `HF_HUB_OFFLINE` | `0` | `1` once the model is cached: no revision check against the Hub on each load, and no network needed. Loading an uncached model then fails, and says why |
| `STT_DEFAULT_LANGUAGE` | unset | Language used when a request does not name one (`de` for a German-first setup). Unset auto-detects, as before. An explicit `auto` in a request still auto-detects |
| `STT_MODEL_TTL` / `MODEL_TTL` | `300` | Seconds idle before the weights are released. `0` releases the moment the service falls idle, `-1` pins them resident. |
| `STT_BATCH_WORKERS` | `min(4, CPUs)` | Threads for file transcription. Each holds one fully decoded file (about 230 MB per hour of audio), so this is also the memory ceiling |
| `STT_BATCHED_INFERENCE` | `false` | Decode `/transcribe` files with faster-whisper's `BatchedInferencePipeline`: several times faster on long files, but with a single temperature and no previous-text conditioning, so output differs. Needs `vad_filter=true`. Never used for live transcription |
| `STT_BATCH_SIZE` | `8` | Batch size for the above; the pipeline costs about 1.5 GB extra VRAM at 8 |
| `MAX_UPLOAD_MB` | `200` | Largest single upload; see [Limits](#limits) |
| `MAX_AUDIO_SECONDS` | `7200` | Longest single recording |
| `FORCE_ACCELERATION` | unset | Force backend: `cuda`, `rocm`, or `cpu` |
| `USE_CUDA` | `true` | Set to `false` to force CPU mode |
| `ALLOWED_ORIGINS` | `*` | Comma-separated CORS origins |
| `ALLOW_CREDENTIALS` | `false` | Enables CORS credentials when origins are explicit |

Live-transcription tuning is documented with its reasoning in
[`.env.example`](../.env.example): `WS_WINDOW_S`, `WS_MIN_NEW_AUDIO_S`,
`WS_MAX_SESSIONS`, `WS_MAX_BUFFER_S` and the two hallucination filters
(`WS_NO_SPEECH_THRESHOLD`, `WS_MIN_AVG_LOGPROB`). Added with the changes below:

| Variable | Default | Description |
|----------|---------|-------------|
| `WS_MAX_INTERIM_ERRORS` | `3` | Interim decodes that must fail in a row before the client is sent an `error` message. The session stays open |
| `WS_IDLE_TIMEOUT_S` | `60` | A socket that sends nothing for this long is closed (code `4408`). `0` disables it. It otherwise holds a `WS_MAX_SESSIONS` slot and the model indefinitely |
| `WS_LANGUAGE_LOCK_PROB` | `0.8` | With no language fixed, the first decode that detects one at least this confidently fixes it for the rest of the session |

### Live transcription (`/ws/transcribe`)

Send an optional first text frame `{"language": "de"}` (`de-DE` is accepted and
reduced to `de`; `"auto"` detects; omitted uses `STT_DEFAULT_LANGUAGE`), then
binary PCM16 mono 16 kHz frames, then `{"event": "stop"}`.

The server sends:

- `{"type": "partial", "confirmed", "pending", ...}` after each interim decode.
  `confirmed` is the **whole session so far** — words two consecutive decodes
  agreed on (LocalAgreement on absolute word times) — and only ever grows;
  `pending` is the tentative tail. A client can simply overwrite its display with
  the two fields. Also `language`, `decode_ms`, `lag_ms`, `pending_seconds`.
- `{"type": "final", "text", "language", "duration", ...}` after `stop`: an
  accurate whole-buffer decode that replaces the partials.
- `{"type": "error", "code", "message"}` for what would otherwise only be
  logged: an unknown language (`invalid_language`; the previous language stays),
  repeated decode failures (`decode_failed`), a model that cannot load
  (`model_unavailable`), `too_many_sessions`, `idle_timeout`. `error` repeats
  `message` for older clients.
- `{"type": "warning", "code": "buffer_rolled"}` once the session outgrows
  `WS_MAX_BUFFER_S`.

`WS_WINDOW_S` is now the *target* length of the audio each interim decode looks
at: the window starts at the last committed sentence end and is cut there once
it outgrows the target, and it may stretch to twice that (30 s at most) while no
sentence has settled. A session holds the model resident for as long as it is
open, so an idle-unload can never land under it.

### Residency

The model is reference counted and released after `STT_MODEL_TTL` seconds idle,
then reloaded on demand — which is what lets Whisper share a card with the ASR
and TTS services instead of holding 1.6–3.1 GB for the container's lifetime. Two
consequences worth knowing:

- `/health` returns **200 with `model_resident: false`** for an unloaded model.
  That is idle, not down; 503 is reserved for "every load attempt failed".
- An unload never interrupts work in progress. `POST /unload` answers `409` with
  the outstanding `model_refs` while a decode is running.

The model preloads in the background at start (not with `STT_MODEL_TTL=0`), so
the port is open and `/health` answers during a first-time download. Use `/ready`
to wait for it: `503` with `reason: "loading"` until the model is in, `200` after,
and `503` with `reason: "load_failed"` and the error if every attempt failed.
`/ready` stays `200` for a model that was unloaded on purpose — it means "can
serve", not "resident now". A request that arrives during the preload waits for
it rather than loading a second copy.

If a load does not fit (a CUDA out-of-memory because another service is holding
the card), the service falls back — a smaller multilingual model, then CPU — and
`/health` reports what is actually running with `degraded: true`. The fallback is
not permanent: every later load, including the one after an idle unload, starts
again from the preferred device and model.

## Model Notes

| Model | Multilingual | Typical Use |
|-------|--------------|-------------|
| `small` | yes | Balanced for modest GPUs / CPU; also the OOM fallback |
| `medium` | yes | Better multilingual accuracy |
| `large-v3-turbo` | yes | Default. 4 decoder layers against large-v3's 32 — fastest multilingual option, but **cannot translate** |
| `large-v3` | yes | Highest quality, more VRAM; the only choice that supports `task=translate` |
| `distil-large-v3`, `distil-large-v3.5` | no | Fast English-only inference (`.5` needs faster-whisper 1.2). Never auto-selected: no `.en` suffix, but would silently drop German. |

`task=translate` is refused with a `400` for models that cannot do it: the turbo
and distil families and English-only models. A custom model is judged by its
name, so a fine-tune of the turbo model (`...turbo-german`) is refused too.

Models download automatically on first use and are cached in the container volume mounted at `.cache/`.

### Using a German fine-tune

`WHISPER_MODEL_SIZE` accepts a Hugging Face repo id of a CTranslate2 model or a
local directory, so a German fine-tune such as
[`primeline/whisper-large-v3-turbo-german`](https://huggingface.co/primeline/whisper-large-v3-turbo-german)
(Apache-2.0) can replace the default without touching code. The default stays
`large-v3-turbo`.

The original checkpoint is a Transformers model and has to be converted once:

```bash
pip install "transformers[torch]>=4.23" ctranslate2
ct2-transformers-converter \
  --model primeline/whisper-large-v3-turbo-german \
  --output_dir whisper-large-v3-turbo-german-ct2 \
  --copy_files tokenizer.json preprocessor_config.json \
  --quantization float16
```

Put the resulting directory in the models volume (mounted at `/app/models`) and
point the service at it:

```
WHISPER_MODEL_SIZE=/app/models/whisper-large-v3-turbo-german-ct2
STT_DEFAULT_LANGUAGE=de
```

- `--copy_files tokenizer.json preprocessor_config.json` is not optional: without
  the tokenizer next to the weights faster-whisper falls back to downloading
  `openai/whisper-tiny`'s from the Hub, which breaks offline use.
- Convert it yourself rather than pulling someone else's CTranslate2 build. Third
  party conversions on the Hub frequently declare no licence of their own, so what
  you may do with them is unclear; the original's Apache-2.0 carries over to your
  own conversion.
- A custom model is trusted to exist: a failed load steps down (`small`, then
  CPU) and is never "corrected" to `large-v3`, which is larger. `float16` weights
  still run as `int8_float16` on the GPU when `WHISPER_COMPUTE_TYPE` says so.
- `STT_DEFAULT_LANGUAGE=de` skips language detection for requests that do not
  name a language, which is where a German fine-tune is most likely to go wrong
  (a short clip detected as English).

## Requirements

- NVIDIA CUDA, AMD ROCm, and CPU are supported
- Internal service port is `8000` (mapped to host port `5001` by Compose)
- For multilingual transcription, avoid English-only distilled models
- `faster-whisper` 1.2.1 with `ctranslate2` 4.x. The AMD image runs on CPU: PyPI's CTranslate2 wheels are CUDA-only
