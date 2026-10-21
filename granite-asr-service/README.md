# Granite-ASR Service

Speech-to-text with IBM Granite Speech 5.0 (Apache-2.0), loaded with transformers' own
classes (never remote code). bfloat16 on a CUDA GPU, float32 on the CPU (slow). Two
checkpoints, chosen with `GRANITE_ASR_MODEL`:

| Model | What it is | Languages | Keywords | VRAM (measured) |
|---|---|---|---|---|
| `ibm-granite/granite-speech-5.0-2b` (default) | Speech LLM: conformer encoder and projector in front of granite-4.0-1b. Punctuated and cased output, German nouns capitalised | en, fr, de, es, pt, ja | yes, up to 2000 | 4.7 GB |
| `ibm-granite/granite-speech-5.0-470m-turboctc` | 470M conformer with a CTC head, one greedy non-autoregressive pass. Punctuated and cased | en, fr, de, es, pt, it, ja | no | 1.2 GB |

The kind of checkpoint is read from its `config.json` (`model_type` `granite_speech` or
`granite_speech5_ctc`), so a fine-tune of either works too (its languages come from
`GRANITE_SUPPORTED_LANGUAGES`, or from its kind).

Provider id `granite-asr` in the gateway (`ENABLE_GRANITE_ASR=true`), compose profile
`granite-asr`, port 5010, image `ghcr.io/steffenhebestreit/tts-stt-granite-asr`.

## Weights

Both IBM repositories are public; no token is needed. Both were changed on 2026-10-21
(the 470m became multilingual that day), so each is **pinned to the commit this
repository measured**: `f4e68169fa2a5751acc4b5da9957855e5e8fc3f1` for the 2b,
`c85a3307b6abec4a169daaf11e8dfa80ac27ad81` for the 470m. `GRANITE_ASR_REVISION` moves
it (`main` follows the repository; empty is the pin, or `main` for an unknown model).
Only the files a checkpoint needs are fetched: not the 2b's second checkpoint
(`out_llm.safetensors`, 411 MB), its sample audio or signatures.

`GRANITE_ASR_SHA256` (a hex digest of `model.safetensors`) makes the service refuse
other weights, as `cohere-asr-service` does by default; empty means no check. `HF_TOKEN`
is passed on for a private checkpoint.

## Language

Neither model identifies the language.

- `auto`, an empty value and no `language` field take `GRANITE_ASR_DEFAULT_LANGUAGE`
  (default `de`); an explicit `auto` adds a line to `warnings` saying so.
- The 2b is told the language in its prompt, the model card's language-ID prompting,
  which also keeps it from translating: `The speech is in German. Transcribe it exactly;
  do not translate. transcribe the speech with proper punctuation and capitalization.`
- The 470m takes no prompt: the language is checked and reported but does not steer it.
- A language the model does not know is refused with **`422`** and the list, before
  anything is loaded. `GRANITE_SUPPORTED_LANGUAGES` replaces the list (comma or space
  separated) for a fine-tune.

`GET /status` reports `supported_languages`, `default_language` and `current_model`; the
gateway builds Granite's language selector from it (refreshed every 5 minutes, like
Canary's), so switching to the 470m adds Italian there.

The prompt was chosen on 40 MLS German clips: punctuation prompt alone 5.80 % WER,
language-ID + punctuation 5.94 %, language-ID + the raw-transcript prompt 6.09 %, the
raw prompt alone 6.16 %. All of that is within noise; the language-ID variant is kept
because it is what stops the model translating.

## Keyword biasing (2b)

`POST /transcribe` takes `keywords`, and `initial_prompt` is read the same way: the gateway
sends OpenAI's `prompt` as `initial_prompt`, from `/v1/audio/transcriptions` (when Granite
is the default STT engine) and from `/api/stt`. Both are split at commas, semicolons and
line breaks, trimmed, de-duplicated ignoring case, and appended to the prompt as
`Keywords: a, b, c`. More than 2000 keywords, or one longer than 100 characters, is a
`422`: silently using part of a list would bias towards a list nobody asked for. So a
prose prompt without commas becomes one keyword. The response's `keywords` says how many
were applied; the 470m applies none and says so in `warnings`.

Example: an MLS clip with "Oberst Bersdorf" came back as "Bahrsdorf", and as "Bersdorf"
with `keywords=Bersdorf, Derbyit, Norwich`.

## What the model hears, how a recording is cut, timestamps

The same speech gate and chunk planner as `cohere-asr-service` (shared
`transformers_asr.py`; see its README): Silero VAD by default (`GRANITE_ASR_VAD`
`silero` / `energy` / `off`, `GRANITE_ASR_VAD_THRESHOLD`), chunks of at most
`GRANITE_ASR_CHUNK_S` seconds (5-120, default 30; the 2b's model card advises at most
2 minutes per call) cut in pauses, pauses over 2 s never transcribed, speech near the
file's edges extended to them, and no model call for a file without speech.
`GRANITE_ASR_BATCH_SIZE` (default 4) chunks go through the model at a time;
`GRANITE_ASR_MAX_NEW_TOKENS` (empty: 20 per second of chunk) caps the 2b's output per
chunk, and a chunk that reaches it is flagged `truncated`.

Neither model gives timestamps: each segment is one chunk with the chunk's `start` and
`end`, so `srt`, `vtt` and `verbose_json` through the gateway are chunk-level.

## Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/transcribe` | `audio` (file), optional `language`, `keywords`, `initial_prompt`. `stt-form-v1` shape plus `speech_seconds`, `chunks`, `keywords`, `truncated`, `warnings` |
| GET | `/status` | Model, revision, `model_family` (`speech-llm` / `ctc`), `supported_languages`, `default_language`, `keywords`, `vad`, `weights_check`, `load_seconds`, chunk and batch settings, queue, GPU memory |
| GET | `/health` | Liveness, `200` whatever the model is doing |
| GET | `/ready` | `200` once loaded; `503` `loading` or `load_failed` |
| POST | `/unload` | Release the model and its VRAM now; `409` while a request is in flight |

Answers as in `cohere-asr-service`: `400` empty upload, `413` too large or too long
(`MAX_UPLOAD_MB` 200, `MAX_AUDIO_SECONDS` 7200), `422` unsupported language, keyword list
over the limit or undecodable audio, `503` + `Retry-After` when busy, `503` on GPU
out-of-memory and when the model could not be loaded (a checkpoint of another
`model_type`, or weights that fail `GRANITE_ASR_SHA256`; `/ready` says `load_failed`),
`500` for anything else, with a request id. The shared
`ASR_*`, `MODEL_TTL`, `ALLOWED_ORIGINS` settings work as everywhere else.

## Measured

RTX 4080 16 GB (Ada), Docker Desktop, bf16, transformers 5.20.2, torch 2.11 (cu128);
100 MLS German clips (seed 1234) through the gateway's `/api/stt`, and a 4-minute German
recording (`conv_000/clean.wav`):

| | granite-speech-5.0-2b | granite-speech-5.0-470m-turboctc |
|---|---|---|
| First-start download | 4.6 GB | 0.95 GB |
| Load from a warm cache | 18-23 s | 6 s |
| VRAM once loaded (at the card) | +4.7 GB (torch: 4.32 GiB) | +1.15 GB (torch: 0.89 GiB) |
| VRAM peak on the 4-minute file | +4.8 GB (4.74 GiB reserved) | +1.4 GB (1.04 GiB reserved) |
| 100 MLS clips | WER 6.55 %, CER 2.71 % | WER 5.69 %, CER 2.96 % |
| Latency per clip | median 1.68 s, p95 2.49 s, RTF 0.108 | median 0.16 s, p95 0.19 s, RTF 0.010 |
| 4-minute file | WER 4.54 %, CER 1.54 %, 15.8 s | WER 3.66 %, CER 1.17 %, 2.1 s |

## Image

Identical to `cohere-asr-service` up to its own code (CUDA 12.8 runtime, python3.11 from
deadsnakes, torch and torchaudio 2.11.0 cu128 with a build-time sm_120 check,
`transformers==5.20.2`, `silero-vad==6.2.3`); the build checks that
`GraniteSpeechForConditionalGeneration` and `GraniteSpeech5ForCTC` import (the latter
needs transformers 5.16) and that the VAD model loads. torchaudio is there because
`GraniteSpeechFeatureExtractor` refuses to run without it. About 10 GB unpacked, the
heavy layers shared with the Cohere image. NVIDIA only: no ROCm, Vulkan or ARM64 image.

Local helpers next to `app.py`: `granite_support.py` (models, pins, languages, prompt,
keywords), `transformers_asr.py`, `audio_pipeline.py`, `model_lifecycle.py`,
`body_limit.py`, `origin_guard.py`.
