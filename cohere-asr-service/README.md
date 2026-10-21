# Cohere-ASR Service

Speech-to-text with Cohere Transcribe (`cohere-transcribe-03-2026`): a 2B-parameter
conformer encoder with a transformer decoder, Apache-2.0, trained on 14 languages:
**Arabic, German, Greek, English, Spanish, French, Italian, Japanese, Korean, Dutch,
Polish, Portuguese, Vietnamese and Chinese**. It is loaded with transformers' own
`CohereAsrForConditionalGeneration` and `CohereAsrProcessor` (transformers 5.4 and
later), never with remote code. bfloat16 on a CUDA GPU, float32 on the CPU (slow).

Provider id `cohere-asr` in the gateway (`ENABLE_COHERE_ASR=true`), compose profile
`cohere-asr`, port 5009, image `ghcr.io/steffenhebestreit/tts-stt-cohere-asr`.

## Where the weights come from

`CohereLabs/cohere-transcribe-03-2026` is gated on Hugging Face: an anonymous download
is a `401`, so every install would need a personal access token first. The default is
`evewashere/cohere-transcribe-03-2026-ungated`, a public copy of it, **pinned to commit
`29b9036c65620e1a148127c6147543b52358da6a`**. In that commit every file has the same git
blob id as in the official repository, except `README.md`, and `model.safetensors` has
the official LFS SHA-256 `987bd3e141c7bfdb5a78f5db11397ee7737308357e6cc0a3f36a4979b158137a`
(4,131,862,976 bytes).

Before anything reads the weights, the service checks the size and hashes the file, and
refuses to load anything else (`/ready` answers `503 load_failed`, `/status` shows
`weights_check: failed`, the log says what it found). So a mirror that changed its
files could not change what runs here. The hash takes about 13 s and is done once per
process: a reload after an idle unload does not read the 4 GB again.

Only the files a checkpoint needs are fetched (configs, tokenizer, `model.safetensors`):
not the repository's remote-code `.py` files, its demo audio or its images.

| Setting | Default | Meaning |
|---|---|---|
| `COHERE_ASR_MODEL` | `evewashere/cohere-transcribe-03-2026-ungated` | A Hugging Face id, or the path of a checkpoint directory on a mounted volume (shown to callers as its last component) |
| `COHERE_ASR_REVISION` | *(empty)* | Empty: the pinned commit above for the mirror, `main` for anything else |
| `COHERE_ASR_SHA256` | *(empty)* | The SHA-256 `model.safetensors` must have. Empty: the official digest for the mirror, no check for anything else. A 64-digit hex digest: that one. `off`: no check. Anything else stops every load (a typo must not switch the check off) |
| `HF_TOKEN` | *(empty)* | Only for the official repository. Never logged, never in a response |

To use the official repository: accept its terms on Hugging Face, then set
`COHERE_ASR_MODEL=CohereLabs/cohere-transcribe-03-2026` and `HF_TOKEN=hf_...`. Its weights
are not held to a digest by default; set `COHERE_ASR_SHA256` to the one above to make
them so.

## Language

The model has **no language identification**: the language is part of its decoder
prompt, and it transcribes whatever it hears as that language.

- `auto`, an empty value and no `language` field take `COHERE_ASR_DEFAULT_LANGUAGE`
  (default `de`). An explicit `auto` adds a line to `warnings` saying so.
- Locale tags reduce to the language (`de-DE`, `de_DE`, ` DE `).
- A language outside the 14 is refused with **`422`** and the list, before anything is
  loaded: transcribing it as German instead would be a confident wrong answer with a `200`.

## What the model hears, and how a recording is cut

Like most attention encoder-decoders, the model writes words for silence and room noise
(its model card says so and asks for a noise gate or VAD in front). So a voice activity
detector decides what reaches it, `COHERE_ASR_VAD`:

- `silero` (default): Silero VAD 6 (`silero-vad`, MIT; the model ships inside the
  wheel, nothing is downloaded). About 1.2 s of CPU per 4 minutes of audio (~30 s per
  2 hours), outside the GPU turn. `COHERE_ASR_VAD_THRESHOLD` is its threshold.
- `energy`: a level gate against the file's own noise floor (10th-percentile frame
  + 12 dB, and at least -60 dBFS). Steady noise at any level stays out. Used
  automatically, and named in `/status` (`vad: "energy (silero unavailable)"`), when
  Silero cannot be loaded.
- `off`: the whole file, cut at quiet points only (qwen3-asr's behaviour).

The stretches of speech are then grouped into chunks of at most `COHERE_ASR_CHUNK_S`
seconds (5-30, default 30; the feature extractor would split anything longer itself):

- Neighbouring stretches share a chunk while the pause between them is at most 2 s,
  and the pause stays in. A longer pause starts the next chunk and is never transcribed.
- A chunk is closed in a pause, never in a word. One stretch longer than the limit is
  cut at its quietest 50 ms near the limit (`audio_pipeline.plan_chunks`, the code
  qwen3-asr uses).
- Speech within 1 s of the start or the end of the file is extended to it: a recording
  that starts mid-word gives the VAD no run-up (on an MLS clip Silero started at 0.71 s
  and the first word was lost).
- A file without speech comes back with an empty `text`, no segments, `"No speech was
  detected in the audio."` in `warnings`, and no model call. Checked with 10 s of
  digital silence and of white noise at -50 and -26 dBFS.

Chunks go through `generate()` `COHERE_ASR_BATCH_SIZE` at a time (default 4; lower it on
out-of-memory). A chunk whose output reaches `COHERE_ASR_MAX_NEW_TOKENS` (empty: 20 per
second of chunk, 600 for 30 s; at most 1000) is flagged `truncated`, with a warning.

## Timestamps

The model has none. Each segment is one chunk: its `start` and `end` are where the chunk
begins and ends in the file (a sentence or a few, up to 30 s), so `srt`, `vtt` and
`verbose_json` through the gateway's `/v1/audio/transcriptions` are chunk-level, not
word-level.

## Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/transcribe` | `audio` (file), optional `language`, optional `initial_prompt` (ignored, with a warning: the model takes no prompt). `stt-form-v1` shape |
| GET | `/status` | Model, revision, `weights_sha256` and `weights_check` (`pending`, `ok`, `failed`, `off`), `load_seconds`, `supported_languages`, `default_language`, `vad`, chunk and batch settings, queue, GPU memory |
| GET | `/health` | Liveness, `200` whatever the model is doing; `model_loaded`, `model_resident`, `current_model`, `default_language`, `device` |
| GET | `/ready` | `200` once the model has loaded; `503` `loading` (with `Retry-After`) or `load_failed` (a short category, never the exception text) |
| POST | `/unload` | Release the model and its VRAM now; `409` while a request is in flight |

`/transcribe` answers `text`, `segments` (`start`, `end`, `text`, and `truncated` when
cut off), `language` (the one used), `language_probability` (`null`), `duration`,
`speech_seconds` (what the VAD passed on), `processing_time`, `chunks`, `truncated`,
`warnings`, `task` and `model`.

| Answer | When |
|---|---|
| `400` | Empty upload |
| `413` | Over `MAX_UPLOAD_MB` (200) or longer than `MAX_AUDIO_SECONDS` (7200, `0` = unlimited) |
| `422` | A language the model does not know; audio no decoder can open |
| `503` + `Retry-After` | `ASR_MAX_CONCURRENCY + ASR_MAX_QUEUE` requests already admitted, or one waited `ASR_QUEUE_TIMEOUT_S` |
| `503` | GPU out of memory; or the model could not be loaded (weights that fail the check among them: `/ready` says `load_failed` and a short category, the log has the reason) |
| `500` | Anything else (the response names a request id; the log has the reason) |

The shared settings work as in the other ASR services: `ASR_MODEL_TTL` / `MODEL_TTL`
(idle seconds before the weights are released; `0` at once, `-1` never),
`ASR_MAX_CONCURRENCY`, `ASR_MAX_QUEUE`, `ASR_QUEUE_TIMEOUT_S`, `ALLOWED_ORIGINS`,
`ALLOW_CREDENTIALS`, `PYTORCH_CUDA_ALLOC_CONF`. The body limit and the origin guard are
the same modules as everywhere else (`body_limit.py`, `origin_guard.py`).

## Measured

RTX 4080 16 GB (Ada), Docker Desktop, bf16, transformers 5.20.2, torch 2.11 (cu128):

| | |
|---|---|
| First-start download | 4.1 GB |
| Load from a warm cache | 30-32 s, 13 s of it the SHA-256 check |
| VRAM | +4.2 GB at the card once loaded (torch: 3.86 GiB allocated); +4.6 GB peak during a 4-minute file (4.25 GiB reserved) |
| 100 MLS German clips (seed 1234), through the gateway's `/api/stt` | WER 5.12 %, CER 2.65 %; median 0.42 s, p95 0.62 s per clip; RTF 0.028 |
| The same with `COHERE_ASR_VAD=off` | WER 5.20 %, CER 2.76 % (no significant difference; the gate costs ~0.1 s per request) |
| A 4-minute German recording (`conv_000/clean.wav`) | WER 3.66 %, CER 1.23 %, 11 chunks, 4.7 s through the gateway |

On read speech like MLS the output is mostly lower case without punctuation, although
punctuation is requested: the model mirrors the style of that kind of audio.
`benchmarks/run_german_eval.py --provider cohere-asr` measures it on your own data.

## Image

`nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04`, python3.11 from deadsnakes, torch and
torchaudio 2.11.0 from the cu128 index (the build fails unless they carry sm_120
kernels, so it runs on Blackwell cards), `transformers==5.20.2` (exact: the model code
lives in transformers) and `silero-vad==6.2.3`. The build checks that the Cohere classes
import and the VAD model loads. It is the same recipe as `granite-asr-service` up to the
service's own code (`requirements.txt` and `transformers_asr.py` are identical copies,
`tests/test_transformers_asr_shared.py`), so a host that pulls both downloads the heavy
layers once. About 10 GB unpacked. NVIDIA only: there is no ROCm, Vulkan or ARM64 image.

Local helpers next to `app.py`: `cohere_support.py` (the source, the digest, the
languages), `transformers_asr.py` (weights check, languages, speech gate, chunks,
segments), `audio_pipeline.py` (qwen3-asr-service's upload, decode and quiet-point
chunking, unchanged), `model_lifecycle.py`, `body_limit.py`, `origin_guard.py`.
