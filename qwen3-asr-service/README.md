# Qwen3-ASR Service

Fast multilingual speech recognition using [Qwen3-ASR](https://huggingface.co/Qwen/Qwen3-ASR-1.7B). This service complements the `faster-whisper` backend with stronger Qwen-family ASR quality and a simpler response schema for the Qwen3 voice workflows.

## Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/transcribe` | Transcribe one audio file (`audio`) |
| POST | `/transcribe-batch` | Transcribe multiple files in one request (`audios`) |
| POST | `/detect_language` | Detect spoken language from the first piece of a file (`file`) |
| GET | `/health` | Liveness: always 200; reports whether the model is resident |
| GET | `/ready` | Readiness: 200, or 503 while the first load runs (`loading`) or failed (`load_failed`) |
| GET | `/status` | Detailed GPU, model and limit information |
| POST | `/unload` | Free the model's VRAM now (409 while a request is running) |

## Usage

```bash
curl -X POST "http://localhost:5002/transcribe" \
	-F "audio=@sample.wav" \
	-F "language=auto"
```

`language` takes `auto`, an ISO code or locale tag (`de`, `de-DE`) or the English name (`German`). The 30 languages qwen-asr 0.0.6 supports are accepted; anything else is transcribed with automatic language detection and the response says so in `warnings`.

### Response

`stt-form-v1` (`text`, `segments`, `language`, `duration`) plus:

| Field | Meaning |
|-------|---------|
| `segments[]` | One per audio piece: `start`, `end` (seconds), `text`, `language`. Pieces with no speech make no segment. There is no per-segment confidence. |
| `language` | The requested language, or the one most of the audio was detected in (an English name such as `German`). |
| `chunks` | How many pieces the audio was cut into. |
| `truncated` | `true` when the model hit its output limit in some piece; that piece's segment has `truncated: true` too. |
| `warnings` | Human-readable notes (output limit reached, language not supported). |

## How long recordings are handled

qwen-asr generates at most `max_new_tokens` per piece and splits audio only at 20 minutes, so a recording longer than about two minutes used to come back as HTTP 200 with the end missing and one segment for the whole file. The service now decodes every upload to 16 kHz mono (ffmpeg, falling back to librosa) and cuts it into pieces of at most `QWEN3_ASR_CHUNK_S` seconds at the quietest point of the last few seconds before the limit, so a cut falls between words. The pieces are transcribed in batches of `QWEN3_ASR_BATCH_SIZE` and joined in order.

Timestamps are per piece. The forced aligner (word timestamps) is not loaded.

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `QWEN3_ASR_MODEL` | `Qwen/Qwen3-ASR-1.7B` | Hugging Face model ID |
| `QWEN3_ASR_CHUNK_S` | `30` | Length of the pieces the audio is cut into, seconds (5 to 600) |
| `QWEN3_ASR_BATCH_SIZE` | `4` | Pieces per forward pass. Lower it if the GPU runs out of memory (503), raise it on a card with headroom for speed |
| `QWEN3_ASR_MAX_NEW_TOKENS` | `max(512, 20 x QWEN3_ASR_CHUNK_S)` | Output limit per piece. Real speech needs about a third of the default |
| `QWEN3_ASR_ATTN_IMPLEMENTATION` | unset | `sdpa`, `eager` or `flash_attention_2` (needs flash-attn, which the image does not ship) |
| `MAX_UPLOAD_MB` | `200` | Largest accepted upload; 413 above it |
| `MAX_AUDIO_SECONDS` | `7200` | Longest accepted recording; 413 above it, `0` disables |
| `ASR_MODEL_TTL` | `300` | Seconds idle before the model is unloaded; `0` unloads after every request (and skips the start-up preload), `-1` never |
| `ASR_MAX_CONCURRENCY` | `1` | Transcriptions running on the GPU at once |
| `ASR_MAX_QUEUE` | `4` x `ASR_MAX_CONCURRENCY` | How many requests may wait beyond the ones running. The next one is refused at once with `503` + `Retry-After: 5`, before its upload is copied or decoded. `0` = nobody waits. See [Queue](#queue) |
| `ASR_QUEUE_TIMEOUT_S` | `60` | Longest a request waits for its turn before it is refused with `503` + `Retry-After: 5` (must be at least `0.1`) |
| `ALLOWED_ORIGINS` | *(empty)* | Comma-separated origins a browser page may call this service from. **Unset or empty: no CORS headers at all.** `*` opens it to every page and must be written down (it logs a warning). Independently of this, a state-changing request (anything but `GET`/`HEAD`/`OPTIONS`) that carries an `Origin` header for another host than the one it was sent to is refused with `403` unless the origin is listed; requests without an `Origin` (the gateway, curl) are not affected |
| `ALLOW_CREDENTIALS` | `false` | Enables CORS credentials when origins are explicit |

The model preloads in the background at start-up, so the port opens at once and `/ready` answers 503 `loading` until the weights are in.

`/ready` answers `503` with `reason` `load_failed` after a failed load; its `detail` is a short category (`model_files_unavailable`, `out_of_memory`, `missing_dependency`, `load_error`), never the exception text, which can hold paths: that is in the log.

## Queue

At most `ASR_MAX_CONCURRENCY` transcriptions run at once. Every request that got past that point used to have copied its upload to disk and decoded the audio into memory (a 2 h recording is about 460 MB of float32) and then waited for the ones ahead of it, with no limit on how many or how long, so a burst of uploads piled up temp files and RAM and held every client until the slowest one was done. Now a request is admitted only if fewer than `ASR_MAX_CONCURRENCY + ASR_MAX_QUEUE` are in progress (otherwise `503` + `Retry-After: 5` at once, having cost nothing), and one that waits more than `ASR_QUEUE_TIMEOUT_S` for its turn is refused the same way. Whichever way a request ends, refused, timed out, cancelled or done, its model reference is handed back. Requests that wait keep the model resident, so with `ASR_MODEL_TTL=0` a queue does not reload it once per request. `GET /status` reports `queue` (the limits, `in_flight`, `running`, `waiting`).

`/transcribe-batch` holds one place for the whole request however many files it carries and transcribes them one after another. If a file cannot get a turn in time, the answer is `503` when nothing was done yet; otherwise the files already done are returned and the rest are reported as `503` (each would only wait as long again).

## Errors

An unexpected failure is a `500` whose `detail` says `Transcription failed (internal error). Request id: <id>.`; the exception (which can hold file paths and library internals) is in the service log under that id. A GPU out of memory is a `503` with advice, and an error the service wrote for the caller (`413`, `422`) is passed on as written. In `/transcribe-batch` a failed file carries the same message in its `error`.

## Requirements

- CUDA or ROCm GPU strongly recommended
- CPU mode works, but is too slow for practical long-form usage
- The 1.7B model typically needs about 4 GB of VRAM, plus activations for `QWEN3_ASR_BATCH_SIZE` pieces
- Model weights are downloaded automatically on first start and cached in the container volume
- The CUDA image installs a pinned torch (`--build-arg TORCH_VERSION=...`) and fails the build if it has no `sm_120` (Blackwell) kernels
