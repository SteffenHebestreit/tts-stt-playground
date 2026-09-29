# Canary-ASR Service

Speech-to-text built on NVIDIA `canary-180m-flash` — a 182M-parameter multilingual ASR model with punctuation/capitalisation that runs at RTFx >1000 on GPU, making it the lowest-latency transcription backend in this stack. Supports **English, German, Spanish, and French**; `CANARY_ASR_MODEL=nvidia/canary-1b-v2` switches to the 25-language checkpoint.

Canary has no language identification: the request `language` selects the decoder language.

- `auto` or an empty value takes `CANARY_DEFAULT_LANGUAGE` (default `de`).
- Locale tags reduce to the language (`de-DE`, `de_DE`, `DE` all mean `de`).
- A language the configured model does not decode is refused with **`422`** and the list of supported ones. It used to be silently decoded as the default language, so Italian audio came back as garbled German.

### Languages per model

| `CANARY_ASR_MODEL` | Languages |
|--------------------|-----------|
| `nvidia/canary-180m-flash` (default), `nvidia/canary-1b-flash`, `nvidia/canary-1b` | `en`, `de`, `es`, `fr` |
| `nvidia/canary-1b-v2` | `bg` `hr` `cs` `da` `nl` `en` `et` `fi` `fr` `de` `el` `hu` `it` `lv` `lt` `mt` `pl` `pt` `ro` `sk` `sl` `es` `sv` `ru` `uk` |
| anything else (local `.nemo`, fine-tune) | `en`, `de`, `es`, `fr` until `CANARY_SUPPORTED_LANGUAGES` says otherwise (a warning is logged at startup) |

The 25-language row follows NeMo's checkpoint table (`docs/source/asr/asr_checkpoints.rst`, "EU25"),
which additionally names Belarusian; only the 25 both sources agree on are listed.
The set is a table, not read from the checkpoint's tokenizer (which cannot be
inspected without downloading it), so set `CANARY_SUPPORTED_LANGUAGES` if you
have verified another language. `GET /status` shows the effective set.

## Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/transcribe` | Transcribe one file (`audio`, optional `language`) — `stt-form-v1` shape |
| POST | `/transcribe-batch` | Batch transcription (`audios[]`, optional `language`); failures are reported per file |
| POST | `/v1/audio/transcriptions` | OpenAI-compatible endpoint (`file`, `language`, `response_format`; `text` is returned as `text/plain`) |
| POST | `/detect_language` | Returns a transcript sample (Canary has no LID — no language/confidence) |
| GET | `/status` | Device, GPU memory, model, supported languages, compute dtype, limits |
| GET | `/health` | Liveness. Always `200` while the process is up, whatever the model is doing |
| GET | `/ready` | Can it serve a request now? `200`, or `503` with `reason` `loading` / `load_failed` (see [Readiness](#readiness)) |
| POST | `/unload` | Release the model and its VRAM now (`409` while a request is in flight) |

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `CANARY_ASR_MODEL` | `nvidia/canary-180m-flash` | NeMo model name (`nvidia/canary-1b-v2` for the 25-language, higher-accuracy variant), or the path to a `.nemo` file |
| `CANARY_DEFAULT_LANGUAGE` | `de` | Language used when the request says `auto` or leaves it empty. If the model cannot decode it, `en` is used and a warning is logged at startup |
| `CANARY_SUPPORTED_LANGUAGES` | *(derived from the model)* | Comma or space separated language codes to accept instead of the table above |
| `MAX_UPLOAD_MB` | `200` | Largest single upload; see [Limits](#limits) |
| `NEMO_MAX_AUDIO_S` | `1500` | Longest single recording in seconds (`0` = unlimited); see [Limits](#limits) |
| `NEMO_MATMUL_PRECISION` | `high` | `torch.set_float32_matmul_precision` at model load (`highest`, `high` = TF32, `medium`). CUDA/ROCm only |
| `NEMO_BF16` | `false` | Run inference in bfloat16 (less VRAM, faster). Needs an NVIDIA GPU with compute capability 8.0+, ignored otherwise. **German WER has not been re-measured under bf16**, so it is opt-in |
| `ASR_MODEL_TTL` / `MODEL_TTL` | `300` | Seconds idle before the ~2 GB model is released. `0` releases as soon as it falls idle, `-1` pins it resident. |
| `ASR_MAX_CONCURRENCY` | `1` | Concurrent inferences on the shared model |
| `ASR_MAX_QUEUE` | `4` × `ASR_MAX_CONCURRENCY` | How many requests may wait beyond the ones running. The next one is refused at once with `503` + `Retry-After: 5`, before its upload is copied or converted. `0` = nobody waits. See [Queue](#queue) |
| `ASR_QUEUE_TIMEOUT_S` | `60` | Longest a request waits for its turn before it is refused with `503` + `Retry-After: 5` (must be at least `0.1`) |
| `ALLOWED_ORIGINS` | *(empty)* | Comma-separated origins a browser page may call this service from. **Unset or empty: no CORS headers at all.** `*` opens it to every page and must be written down (it logs a warning). Independently of this, a state-changing request (anything but `GET`/`HEAD`/`OPTIONS`) that carries an `Origin` header for another host than the one it was sent to is refused with `403` unless the origin is listed; requests without an `Origin` (the gateway, curl) are not affected |
| `ALLOW_CREDENTIALS` | `false` | Enables CORS credentials when origins are explicit |

## Limits

| Limit | Default | Answer past it |
|-------|---------|----------------|
| `MAX_UPLOAD_MB` per file | `200` | `413`, also from the `Content-Length` header before the body is read (single-file routes). Uploads are streamed to disk in 1 MiB chunks, never held in memory |
| `NEMO_MAX_AUDIO_S` per file | `1500` (25 min) | `413` with a message. Checked from the container header before ffmpeg runs, and again on the converted file; ffmpeg is told to stop at the limit (`-t`), so a file whose header lies cannot be decoded in full first |
| Unsupported `language` | | `422`, before the model is touched |
| More than `ASR_MAX_CONCURRENCY + ASR_MAX_QUEUE` requests in progress | `1 + 4` | `503` + `Retry-After: 5` at once, before the upload is copied (see [Queue](#queue)) |
| Waiting longer than `ASR_QUEUE_TIMEOUT_S` for a turn | `60` | `503` + `Retry-After: 5` |
| Empty upload | | `400` |
| Audio no decoder can open | | `422` |
| GPU out of memory | | `503` (the CUDA cache is released), not a generic `500` |

In `/transcribe-batch` the size and length limits apply to each file: an offending
file comes back as `{"filename", "error", "status"}` and the others are still
transcribed. An unsupported `language` fails the whole request (`422`) since it
applies to every file. The gateway has its own upload cap (`MAX_UPLOAD_MB`,
default 512), so a file between the two limits is refused here.

Timestamps are requested from models whose `transcribe()` takes a `timestamps`
argument and whose prompt format has a timestamp slot (canary-180m-flash,
canary-1b-flash, canary-1b-v2); for others the text comes back as one segment
spanning the file. Inference runs with TF32 matmuls (`NEMO_MATMUL_PRECISION=high`).

## NeMo version and torch build

The image installs `nemo_toolkit[asr]` **3.0.x** on a pinned `torch` / `torchaudio` **2.11.0** from PyTorch's CUDA 12.8 index. `cu128` is the first build with Blackwell (`sm_120`, RTX 50 series) kernels, and PyTorch publishes it for 2.7 through 2.11 only (2.12 and later ship CUDA 12.6 / 13.x), so the torch pin cannot simply follow NeMo's own tested stack (torch 2.12, CUDA 12.6 / 13.2). Two build arguments move the versions, nothing needs editing:

| Build argument | Default | Meaning |
|----------------|---------|---------|
| `NEMO_TOOLKIT_SPEC` | `>=3.0.0,<3.1` | pip specifier for `nemo_toolkit[asr]`. Replaces the line in `requirements.txt` (which carries the same default) |
| `TORCH_VERSION` / `TORCHAUDIO_VERSION` | `2.11.0` | Must exist as a `+cu128` wheel; torch and torchaudio are released in lockstep |

```bash
# Roll back to the 2.x line (no file edits; the image tag can stay the same)
docker compose build --build-arg NEMO_TOOLKIT_SPEC='>=2.7.3,<3' canary-asr-service
# Accept 3.1 and later 3.x releases once you have tested them
docker compose build --build-arg NEMO_TOOLKIT_SPEC='>=3.0.0,<4' canary-asr-service
```

The build **fails on purpose** when the installed torch is not a CUDA 12.8 build with `sm_120` kernels, both right after the torch install and again after NeMo is installed (a resolver must not swap the wheel). `GET /status` reports what a running image was built with: `runtime.nemo_toolkit`, `runtime.torch`, `runtime.torch_cuda` and `runtime.cuda_arch_list`.

Why 3.0.x is the default (nemo_toolkit 3.0.0 was published 2026-08-07, and the old `>=2.0.0` line already resolved to it on a fresh build):

- The API this service calls is unchanged from 2.7.3: the `transcribe()` signatures of the RNNT and multi-task models are identical (3.0 raises `ValueError` where 2.7 asserted, and passes `verbose` through), `Hypothesis` keeps `text` / `timestamp` (`segment`, `word`, `char`), and `ASRModel.from_pretrained`, the `.cpu()` unload hook and the `use_cuda_graph_decoder` option are the same. For Canary this rests on reading the two wheels (the multi-task `transcribe()` differs only in `ValueError` for `assert` and the `verbose` pass-through); a Canary model was **not** instantiated (no checkpoint could be downloaded, and a random one needs a full tokenizer and prompt-format set-up), so the Canary path is source-inspected only.
- A randomly initialised Parakeet-TDT (CPU, torch 2.11, Python 3.11) run through the real `transcribe(..., timestamps=True)` on 2.7.3 and on 3.0.0 gave identical text and timestamps, also with numpy 2.4. **No GPU run was possible**, so the rewritten TDT decoder and the CUDA-graph path are covered by the checklist below.
- 3.0.0 needs Python 3.11 or later (its sources use 3.11 syntax; its README says 3.12+, its metadata says 3.10+). The image's Python 3.11 works.
- It pulls `transformers` 5 and `huggingface_hub` 1 (2.7.x pinned 4.57 and 0.36). The `numpy<2` cap in `requirements.txt` was NeMo 2.3's; both releases ran on numpy 2.4 and it is gone. `cuda-bindings` stays below 13 to match the CUDA 12.8 torch.
- `PYTORCH_JIT=0` stays: 3.0.0 still contains the `@torch.jit.script` functions (`online_clustering.py`, `vad_utils.py`, ...) it was set for.
- NeMo 3 hardened the `_target_` allow-list used when a `.nemo` is restored. Every `_target_` such a configuration names (preprocessor, encoder, decoder and joint or transformer decoder, token classifier, loss; written down from the NeMo sources, not read from the checkpoint, which could not be downloaded) passes the check when tested directly. A third-party checkpoint with an unusual `_target_` would fail to load with "Instantiation of unsafe target".

### GPU smoke test after a rebuild

```bash
docker compose build canary-asr-service && docker compose up -d canary-asr-service          # 1. the build fails on a wrong torch wheel
curl -s localhost:5006/health; curl -s localhost:5006/status | jq '.runtime'        # 2. 200 while loading; nemo_toolkit 3.0.x, torch 2.11.0+cu128, sm_120 listed
until curl -sf localhost:5006/ready >/dev/null; do sleep 5; done; curl -s localhost:5006/ready   # 3. 503 "loading", then 200
curl -s -F "audio=@german.wav" localhost:5006/transcribe | jq '{text, segments, processing_time}'  # 4. a German wav you know the transcript of
docker compose build --build-arg NEMO_TOOLKIT_SPEC='>=2.7.3,<3' canary-asr-service && docker compose up -d canary-asr-service    # 5. rerun step 4 on 2.x and diff text and segment times
```

### Unloading a checkpoint with a timestamp aligner

Canary checkpoints that ship a forced-alignment model for timestamps (`timestamps_asr_model`) keep it outside the module tree and move it to the GPU on the first timestamped request. `POST /unload` now moves it off the GPU together with the model; before, only the model moved and the aligner kept its memory while the endpoint reported the model released.

## Running

Opt-in profile (heavy NeMo image, like Parakeet):

```bash
ENABLE_CANARY_ASR=true docker compose --profile canary-asr --profile frontend up -d
```

Set `ENABLE_CANARY_ASR=true` on the frontend service so the provider appears in the browser UI.

### Residency

The model is loaded on demand and released after `ASR_MODEL_TTL` seconds idle,
then reloaded on the next request. This service is opt-in and bursty — selected
for a batch of files, then idle — so holding the card between bursts is the
worst trade available on a VRAM-bound host.

Unloading is reference counted: `POST /unload` answers `409` with the
outstanding `active_requests` while a forward pass is running, and `/health`
returns **200 with `model_resident: false`** for an idle-unloaded model. That is
idle, not down.

### Readiness

`/health` is liveness and stays `200` while the model is idle, loading or even
failing to load, so Docker never restarts a container that is doing its job.
`/ready` answers the other question:

| Answer | `reason` | Meaning |
|--------|----------|---------|
| `200` | `ok` | The model has loaded at least once (an idle unload or one failed reload does not flip it), or nothing is known to be wrong yet (lazy start, `ASR_MODEL_TTL=0`) |
| `503` + `Retry-After: 5` | `loading` | The first load, or the startup preload, is still running |
| `503` | `load_failed` | The last load failed and none ever succeeded; `detail` is a short category (`model_files_unavailable`, `out_of_memory`, `missing_dependency`, `load_error`), never the exception text, which can hold paths: that is in the log |

The startup preload runs in the background, so the port answers (`/health` 200,
`/ready` 503 `loading`) during the first-time model download instead of staying
closed. A request that arrives meanwhile waits for the load. Gate anything that
needs a usable model on `/ready`, not `/health`.
### Queue

At most `ASR_MAX_CONCURRENCY` forward passes run at once. Every request that got
past that point used to have copied its upload to disk and converted the audio and
then waited for the ones ahead of it, with no limit on how many or how long, so a
burst of uploads piled up temp files and held every client until the slowest one
was done. Now a request is admitted only if fewer than `ASR_MAX_CONCURRENCY +
ASR_MAX_QUEUE` are in progress (otherwise `503` + `Retry-After: 5` at once, having
cost nothing), and one that waits more than `ASR_QUEUE_TIMEOUT_S` for its turn is
refused the same way. Whichever way a request ends, refused, timed out, cancelled or
done, its temp files are removed and its model reference is handed back. Requests
that wait keep the model resident, so with `ASR_MODEL_TTL=0` a queue does not
reload it once per request. `GET /status` reports `queue` (the limits, `in_flight`,
`running`, `waiting`).

`/transcribe-batch` holds one place for the whole request however many files it
carries and decodes them one after another. If a file cannot get a turn in time,
the answer is `503` when nothing was done yet; otherwise the files already done are
returned and the rest are reported as `503` (each would only wait as long again).

There is no `ASR_MAX_BATCH` here: this service decodes one file per forward pass
(that setting belongs to `parakeet-asr-service`, which batches).

### Errors

An unexpected failure is a `500` whose `detail` says `Transcription failed (internal
error). Request id: <id>.`; the exception (which can hold file paths and library
internals) is in the service log under that id. A GPU out of memory is a `503` with
advice, and an error the service wrote for the caller (`413`, `422`, "no
transcription") is passed on as written. In `/transcribe-batch` a failed file
carries the same message in its `error`.

