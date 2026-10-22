# vLLM-ASR Service

The playground's speech-to-text candidates under **vLLM's continuous batching**, to
measure what they do at production-like load. A thin FastAPI adapter speaks the
`stt-form-v1` contract of every backend here (`/transcribe`, `/health`, `/ready`,
`/status`, `/unload`, body limit, origin guard) and runs `vllm serve <model>` as a
subprocess on 127.0.0.1 inside the same container. A recording is cut in its pauses into
chunks of at most 28 s, and **every chunk of every request goes to the engine at once**,
so vLLM batches them with everything else in flight.

Provider id `vllm-asr` in the gateway (`ENABLE_VLLM_ASR=true`), compose profile
`vllm-asr`, port 5011, image `ghcr.io/steffenhebestreit/tts-stt-vllm-asr` (built on the
host, not published: see [Image](#image)).

The base is `vllm/vllm-openai:v0.31.0`, the exact release Steffen's luna-stt app runs on
the NAS's RTX 5060 Ti (Blackwell, driver 580.173): the engine is proven on that card. It
is a CUDA 13 build and needs an NVIDIA driver 580 or newer.

## Models

One model per service, `VLLM_ASR_MODEL`. Each of these was started under vLLM 0.31 on an
RTX 4080 and is pinned to the commit that was measured (`VLLM_ASR_REVISION` moves it;
empty is the pin, `main` for anything else):

| `VLLM_ASR_MODEL` | Family | How a chunk is asked | Languages | What vLLM 0.31 needs |
|---|---|---|---|---|
| `ibm-granite/granite-speech-5.0-2b` (default, `f4e68169`) | `granite` | `/v1/chat/completions`, audio as `input_audio`, then granite-asr-service's prompt (language, punctuation, keywords) | de en fr es pt ja | `--max-model-len 2048` (granite-4.0-1b's 16k context does not fit the KV cache at 0.45) |
| `evewashere/cohere-transcribe-03-2026-ungated` (`29b9036c`) | `cohere` | `/v1/audio/transcriptions` with the language | 14 (cohere-asr's) | nothing: vLLM's own `CohereAsrForConditionalGeneration`, no remote code; the language must be sent (it defaults to English otherwise) |
| `Qwen/Qwen3-ASR-1.7B` (`7278e1e7`) | `qwen3-asr` | `/v1/audio/transcriptions`; `prompt` is the model's context (system turn), which it biases towards | 28 (Qwen3-ASR's, as far as vLLM's ISO list has them; not Cantonese) | `--max-model-len 2048` (its 65536 would need 7 GB of KV cache to start) |
| `primeline/whisper-large-v3-turbo-german` (`9e7012da`) | `whisper` | `/v1/audio/transcriptions`; `prompt` is Whisper's initial prompt | de | nothing |

All four answer on vLLM's `/v1/audio/transcriptions`. Granite does too, but there vLLM
asks it a fixed "can you transcribe the speech into a written format?" without the
language, and the 2b answers in lower case without any punctuation (MLS: "als bettler
war ich nicht geboren ein bettler..."). Through the chat endpoint with granite-asr's
prompt it writes "Die zweite Ursache lag in der Zusammensetzung unserer Gesellschaft. Ich
habe erwähnt, daß ich in St. Louis...", the same text granite-asr-service produces, and
keywords work. So the adapter supports both endpoints and uses the chat one for Granite.

Cohere's weights come from the same ungated mirror as cohere-asr-service, at the same
commit, and `model.safetensors` is checked against Cohere's official SHA-256 and size
(`cohere_support.py`, the same file) before the engine is started; a mismatch means the
engine never starts (`/ready` `load_failed`, `weights_check: failed`). `VLLM_ASR_SHA256`
overrides it (`off` disables it) or sets a digest for another model;
`VLLM_ASR_MODEL=CohereLabs/cohere-transcribe-03-2026` with `HF_TOKEN` uses the gated
original. Every checkpoint is fetched by the adapter with the same allow-list as the
transformers services (configs, tokenizer, the weights; no remote code, no second
checkpoint such as Granite's `out_llm.safetensors`) and the engine is pointed at the
local directory with `HF_HUB_OFFLINE=1`, so it fetches nothing itself.

Any other checkpoint of these four kinds works too: the family is read from its
`config.json` (`model_type` `whisper`, `cohere_asr`, `qwen3_asr` or `granite_speech`),
its languages are the family's (`VLLM_ASR_SUPPORTED_LANGUAGES` overrides them), and
`VLLM_ASR_MAX_MODEL_LEN` / `VLLM_ASR_EXTRA_ARGS` take what it needs.

**Whisper.** Plain `openai/whisper-large-v3-turbo` runs as well, but under vLLM (greedy,
no temperature fallback) it answered whole German parliament speeches with just "Vielen
Dank." (measured 2026-10-22 on VoxPopuli: 16.6 % WER against 11.5 % for the primeline
fine-tune), which is why the German fine-tune is the Whisper option here. For Whisper
quality the `whisper` provider (faster-whisper, with its temperature fallback) remains
the reference; vLLM 0.31 has `use_beam_search`, which this service does not use yet.

## Language and prompt

- A language the model does not know is refused with **422**, before anything is started.
- No language, and `auto` for a model that does not identify languages (Granite, Cohere,
  the German Whisper), mean `VLLM_ASR_DEFAULT_LANGUAGE` (`de`); an explicit `auto` then
  adds a line to `warnings`.
- An explicit `auto` with Qwen3-ASR (or a multilingual Whisper) is passed on: the model
  identifies the language itself. vLLM does not report which one it found, so `language`
  is `null` in the answer.
- `initial_prompt` (what the gateway sends for OpenAI's `prompt`) and `keywords` become
  Granite's keyword list (at most 2000, as in granite-asr-service), Qwen3-ASR's context
  or Whisper's initial prompt; Cohere takes none and says so in `warnings`.

## What the model hears, chunks, segments

The speech gate and chunk planner of cohere-/granite-asr-service (the shared
`transformers_asr.py`): Silero VAD (`VLLM_ASR_VAD` `silero`/`energy`/`off`,
`VLLM_ASR_VAD_THRESHOLD`), chunks of at most `VLLM_ASR_CHUNK_S` seconds (5-30, default
28: under the 30 s window every family takes whole, so vLLM never splits a chunk itself)
cut in pauses, pauses over 2 s never transcribed, no engine call for a file without
speech. Each chunk is one segment with the chunk's start and end; the texts are joined in
order with `audio_pipeline.join_texts`. The service never joins segments the engine cut
itself: if that is ever added (Whisper's `verbose_json` segments), a segment whose raw
text starts without a space continues the previous word, the convention whose neglect
turned "europäischen" into "europ äischen" in luna-cloud.

`VLLM_ASR_MAX_NEW_TOKENS` (empty: 20 per second of chunk, 560 at 28 s) caps each chunk; a
Granite chunk that reaches it is flagged `truncated` (the transcription endpoint does not
say why a chunk ended, so the other families cannot be flagged).

The VAD runs in `VLLM_ASR_VAD_WORKERS` worker processes (default 4). Silero walks a
recording in 32 ms windows from a Python loop, about 60 ms of one core per 15 s of audio,
holding the GIL: with one gate in the adapter, 32 requests at once capped the service at
about 200x real time while the engine alone did about 450x (Cohere); four gates in
threads reached 250x, four processes 390-405x. `0` runs it in the adapter process.

## GPU memory, idle release

`VLLM_ASR_GPU_MEMORY_UTILIZATION` (default **0.45**) is the share of the **whole card**
the engine takes at start-up, weights, activations, CUDA graphs and KV cache together,
and keeps while it is awake: about 7.2 GiB of a 16 GB card. That fits next to Whisper
(about 2.3 GB), Magpie (about 3.3 GB) and Steffen's luna-stt vLLM app (about 3.3 GB) on
the 5060 Ti. The share minus weights and activations is the KV cache, which decides how
many chunks are decoded at once (`/status` reports `engine.kv_cache_tokens`):

| Model | Weights | KV cache at 0.45 | Chunks of 28 s at once (about) |
|---|---|---|---|
| Granite 5.0 2b | 4.32 GiB | 0.98 GiB, 12,864 tokens | 23 |
| Cohere Transcribe | 3.87 GiB | 2.52 GiB, 27,514 tokens | 27 (vLLM's own bound at 1024 tokens each) |
| Qwen3-ASR 1.7B | 3.90 GiB | 1.49 GiB, 13,936 tokens | 25 |
| Whisper turbo German | 1.51 GiB | 5.27 GiB, 49,624 tokens | 110 |

Lower values leave more room for others and decode fewer chunks at once; at about 0.37
Granite and Qwen3-ASR no longer start (negative KV cache), which is an `out_of_memory`
start failure, not a crash. vLLM checks at start-up that the card has that much free;
on a busy card a start therefore fails until another service gives memory back.

After `ASR_MODEL_TTL` idle seconds (300; `0` at once, `-1` never) the engine is released,
as `VLLM_ASR_IDLE_MODE` says:

- `sleep` (default): vLLM's sleep mode level 1 (`--enable-sleep-mode`,
  `VLLM_SERVER_DEV_MODE=1`, `POST /sleep?level=1`): the weights go to host RAM (up to
  4.7 GB of it) and the KV cache is dropped. Measured: all but 0.4-0.9 GB of the card is
  freed in 2-4 s, and the next request wakes the engine in 0.3-0.9 s (first answer
  0.8-1.9 s after the release). Level 2 (weights discarded) is not used: woken from it the
  engine answered "!!!!!!" or "other other other", with or without `reload_weights`.
- `stop`: the engine process ends, freeing everything; the next request starts it again,
  22-40 s with vLLM's compile cache warm (the first start ever also compiles: up to 22 s
  more for Qwen3-ASR).

`POST /unload` does the same at once (`?mode=stop` ends the process even in sleep mode),
`409` while a request is in flight. A start that fails for any reason (out of memory
while sizing the KV cache, a crash, no answer within `VLLM_ASR_START_TIMEOUT_S`) kills
the engine's whole process group (vLLM runs its engine core in a child process) and
reaps it before the error is raised, so nothing keeps GPU memory: the lesson of #93. An
engine that dies while serving is a 503 for the requests in flight, and the next request
starts a new one.

The adapter process never opens a CUDA context: it hides the GPU from itself
(`CUDA_VISIBLE_DEVICES=""`) before anything imports torch and hands the original value to
the engine only. The VAD runs on the CPU.

## Concurrency knobs

What the adapter admits must not serialise what vLLM could batch:

| Knob | Default | What it bounds |
|---|---|---|
| `VLLM_ASR_MAX_CONCURRENCY` (`ASR_MAX_CONCURRENCY` in the container) | 32 | Requests whose chunks are sent to the engine at the same time. Not the shared `ASR_MAX_CONCURRENCY` (1 for the one-model-call services) |
| `VLLM_ASR_MAX_QUEUE` | 4 x the above | Requests that may wait for a turn; beyond it, or after `ASR_QUEUE_TIMEOUT_S` (60), 503 + `Retry-After` |
| `VLLM_ASR_MAX_PENDING_CHUNKS` | 256 | Chunk requests in flight to the engine across all requests (the adapter's sockets and buffers); vLLM queues what does not fit its KV cache itself |
| `VLLM_ASR_VAD_WORKERS` | 4 | VAD processes (above) |
| `VLLM_ASR_MAX_NUM_SEQS`, `VLLM_ASR_MAX_NUM_BATCHED_TOKENS` | vLLM's | The engine's own scheduler; 2048 batched tokens made no difference for Granite |
| gateway `MAX_CONCURRENT_UPLOADS` | 4 per worker | Through `/api/stt`, uploads beyond it are refused with 503 before they reach this service: raise it (the measurements below used 40) or call the service directly |

## Measured

RTX 4080 16 GB (Ada), Docker Desktop on Windows (WDDM), vLLM 0.31.0, bf16, the default
settings (0.45, sleep mode, 28 s chunks, 4 VAD workers); everything through the gateway's
`/api/stt` with `provider=vllm-asr`. Accuracy: `benchmarks/run_german_eval.py` on 100 MLS
German clips (seed 1234), and the 4-minute VoxPopuli-based conversation
`addon-vox/conv_000/clean.wav` (13 chunks). Throughput: `benchmarks/run_throughput.py` on
the first 200 MLS clips (3,039 s of audio), 1, 8 and 32 requests at once. VRAM is the
whole card's `nvidia-smi` reading with the engine awake, asleep and stopped.

| | Granite 5.0 2b | Cohere Transcribe | Qwen3-ASR 1.7B | Whisper turbo German |
|---|---|---|---|---|
| 100 MLS clips: WER / CER | 6.55 % / 2.68 % | 5.04 % / 2.66 % | 6.85 % / 2.49 % | 7.04 % / 2.52 % |
| Latency per clip, one at a time (median / p95) | 0.60 / 0.83 s | 0.31 / 0.43 s | 0.67 / 0.95 s | 0.29 / 0.36 s |
| Throughput, 1 request at a time | 25x | 50x | 24x | 53x |
| 8 at once | 146x | 224x | 151x | 225x |
| 32 at once (latency p95) | 293x (2.5 s) | 392x (1.8 s) | 343x (2.0 s) | 326x (2.1 s) |
| 4-min MLS-based conversation (12 chunks): WER, wall | 5.12 %, 3.4 s | 3.51 %, 4.6 s | 5.42 %, 2.9 s | 6.00 %, 2.4 s |
| 4 of them at once | 323x | 355x | 339x | 291x |
| 4-min VoxPopuli-based conversation (13 chunks): WER, wall | 13.8 %, 3.0 s | 14.5 %, 2.5 s | 14.0 %, 2.6 s | 13.2 %, 2.4 s |
| Container start to `/ready` (warm caches) | 54 s | 48 s | 47-52 s | 35-37 s |
| Restart of a stopped engine (first answer) | 40 s | 26 s | 36 s | 22 s |
| Put to sleep / woken / first answer after sleep | 3.9 / 0.81 / 1.8 s | 3.6 / 0.71 / 1.1 s | 3.7 / 0.88 / 1.9 s | 1.8 / 0.31 / 0.8 s |
| VRAM awake (card, after use) / asleep | 7.4 / 0.86 GB | 9.0 / 0.53 GB | 9.3 / 0.77 GB | 7.6 / 0.39 GB |

Cohere and Qwen3-ASR read about 1.5 GB above the 0.45 share after the throughput runs (the
same in two runs); vLLM's share bounds the KV cache, not every activation an encoder
needs at 32 requests at once. Not yet understood (WDDM accounting or real growth): budget
9-9.5 GB for them on a shared card until it is measured on Linux.

The transformers services on the same 100 clips, one request at a time: cohere-asr 5.12 %
WER at 0.42 s median, granite-asr (2b) 6.55 % at 1.68 s. vLLM gives the same accuracy
(the same prompts, greedy) and 1.4-2.8x lower latency for one request; what it adds is
the batching: 32 requests at once run 6-14x faster than one at a time. MLS is
contaminated for Cohere and the primeline Whisper (both write MLS-style text, primeline
was fine-tuned on it): rank the models on VoxPopuli, FLEURS and the conversation sets,
not on these clips.

**For the NAS (RTX 5060 Ti 16 GB), scale by about 0.63** (the factor used between the
two cards): roughly 16-31x real time for one request, 90-150x at 8 and 185-250x at 32.
The 5060 Ti's 16 GB give the same 0.45 share as here; with Whisper, Magpie and luna-stt
resident there are about 7.5 GB free, so 0.45 (7.3 GiB requested) is at the edge: if the
start fails with `out_of_memory`, use 0.42 (Granite then keeps about 0.5 GiB of KV cache).

## Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/transcribe` | `audio` (file), optional `language`, `initial_prompt`, `keywords`. `stt-form-v1` shape plus `speech_seconds`, `chunks`, `keywords`, `prompt_applied`, `truncated`, `warnings`, `engine: "vllm"` |
| GET | `/status` | Model, `revision`, `model_family`, `offered_models`, `weights_check`, `supported_languages`, `default_language`, `language_detection`, `prompt`, VAD, chunking, queue, and `engine`: `state`, `pid`, `start_seconds`, `wake_seconds`, `sleep_seconds`, `kv_cache_tokens`, `idle_mode`, `gpu_memory_utilization`, `max_model_len`, extra arguments |
| GET | `/health` | Liveness, `200` whatever the engine is doing; `engine_state` |
| GET | `/ready` | `200` once the engine has started; `503` `loading`, or `load_failed` with `detail` (`out_of_memory`, `model_files_unavailable`, `load_error`) |
| POST | `/unload` | Engine to sleep now (`?mode=stop`: end it); `409` while a request is in flight |

Answers as in the transformers services: `400` empty upload, `413` too large or too long
(`MAX_UPLOAD_MB` 200, `MAX_AUDIO_SECONDS` 7200), `422` unsupported language, a keyword
list over the limit or undecodable audio, `503` + `Retry-After` when busy or when the
engine died mid-request, `503` when it could not be started, `500` for anything else
(an engine error answer included), with a request id and nothing from the engine's text.

## Image

`FROM vllm/vllm-openai:v0.31.0`, plus `silero-vad==6.2.3` installed with `--no-deps` (it
needs only the base's torch and packaging; a build-time check loads its model and makes
sure that did not initialise CUDA) and the adapter's code. vLLM, torch, fastapi, uvicorn,
httpx, huggingface_hub and ffmpeg are the base's. The entrypoint (`vllm serve`) is
replaced by the adapter's uvicorn. `/root/.cache` is the cache volume: the checkpoint in
`huggingface/`, vLLM's compile cache in `vllm/` (a restart reuses both).

**Not published.** The base is about 9 GB to download and 32 GB unpacked, more than a
GitHub runner has free, so CI builds and pushes no image for this service
(`scripts/plan_publish.py`, `LOCAL_BUILD_ONLY`); `tests/test_docker_image_contents.py`
checks statically that every module `app.py` imports is copied, as for the ROCm variants,
and `tests/test_vllm_asr_shared.py` the rest of the recipe. Build it where it runs; with
the base already there (any host that runs vLLM) that takes seconds:

```bash
docker compose --profile vllm-asr build vllm-asr-service       # from a checkout
# compose also builds it by itself when the host does not have the image (PULL_POLICY=missing)
```

On TrueNAS (the Custom App file pulls images, it does not build them), once per release,
with the tag the app runs (`IMAGE_TAG`), in the TrueNAS shell as root:

```bash
docker build -t ghcr.io/steffenhebestreit/tts-stt-vllm-asr:0.2.0 \
  https://github.com/SteffenHebestreit/tts-stt-playground.git#develop:vllm-asr-service
```

then delete the `profiles: ["vllm-asr"]` line of `vllm-asr-service` and offer it in
Settings -> Engines (or `ENABLE_VLLM_ASR:-true`). `PULL_POLICY` must stay `missing`. The
service shares the app's `cache/` with cohere-asr and granite-asr, so their downloads of
Granite and Cohere are reused. NVIDIA only: no ROCm, Vulkan or ARM64 image.

Local helpers next to `app.py`: `vllm_support.py` (models, families, pins, the engine's
command line, one chunk as a request), `vllm_engine.py` (the subprocess: start, sleep,
wake, stop), `vad_pool.py` (the VAD workers); copied unchanged from the other services:
`transformers_asr.py`, `cohere_support.py`, `granite_support.py`, `audio_pipeline.py`,
`model_lifecycle.py`, `body_limit.py`, `origin_guard.py` (tests keep the copies equal).

**Security.** The engine listens on 127.0.0.1 inside the container only; its sleep
endpoints need vLLM's development mode, whose other routes are equally unreachable from
outside. No remote code is ever trusted (`--trust-remote-code` is refused in
`VLLM_ASR_EXTRA_ARGS`), `HF_TOKEN` is not passed to the engine, and vLLM's usage
statistics are off (`VLLM_NO_USAGE_STATS`, `DO_NOT_TRACK`).
