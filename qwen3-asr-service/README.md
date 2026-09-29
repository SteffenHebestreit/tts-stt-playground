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
| `ALLOWED_ORIGINS` | `*` | Comma-separated CORS origins |
| `ALLOW_CREDENTIALS` | `false` | Enables CORS credentials when origins are explicit |

The model preloads in the background at start-up, so the port opens at once and `/ready` answers 503 `loading` until the weights are in.

## Requirements

- CUDA or ROCm GPU strongly recommended
- CPU mode works, but is too slow for practical long-form usage
- The 1.7B model typically needs about 4 GB of VRAM, plus activations for `QWEN3_ASR_BATCH_SIZE` pieces
- Model weights are downloaded automatically on first start and cached in the container volume
- The CUDA image installs a pinned torch (`--build-arg TORCH_VERSION=...`) and fails the build if it has no `sm_120` (Blackwell) kernels
