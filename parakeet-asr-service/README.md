# Parakeet-ASR Service

Fast multilingual speech recognition using NVIDIA [parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) (FastConformer-TDT). Covers **25 European languages including German** with automatic language detection, and is among the fastest open ASR models (real-time factor in the thousands on GPU, fast even on CPU) — a strong realtime/streaming-friendly complement to the `faster-whisper` and `Qwen3-ASR` backends.

It exposes the project's native `stt-form-v1` contract (`/transcribe` with segment timestamps, so it works in the UI segmentation flow **and** as a training-pipeline STT backend) plus an OpenAI-compatible `/v1/audio/transcriptions` endpoint.

## Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/transcribe` | Transcribe one audio file (`audio`); returns text + segments |
| POST | `/v1/audio/transcriptions` | OpenAI-compatible transcription (`file`) |
| POST | `/transcribe-batch` | Transcribe multiple files in one batched pass; failures are reported per file |
| POST | `/detect_language` | Return a transcript sample (Parakeet auto-detects language internally) |
| GET | `/health` | Liveness and model/device status. Always `200` while the process is up, whatever the model is doing |
| GET | `/ready` | Can it serve a request now? `200`, or `503` with `reason` `loading` / `load_failed` (see [Readiness](#readiness)) |
| GET | `/status` | Detailed GPU and model information, compute dtype, limits |
| POST | `/unload` | Release the model and its VRAM now (`409` while a request is in flight) |

## Usage

```bash
curl -X POST "http://localhost:5005/transcribe" \
	-F "audio=@sample.wav"
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `PARAKEET_ASR_MODEL` | `nvidia/parakeet-tdt-0.6b-v3` | Hugging Face model ID, or the path to a `.nemo` file (see [Local checkpoints](#local-checkpoints)) |
| `MAX_UPLOAD_MB` | `200` | Largest single upload; see [Limits](#limits) |
| `NEMO_MAX_AUDIO_S` | `1500` | Longest single recording in seconds (`0` = unlimited); see [Limits](#limits) |
| `NEMO_MATMUL_PRECISION` | `high` | `torch.set_float32_matmul_precision` at model load (`highest`, `high` = TF32, `medium`). CUDA/ROCm only |
| `NEMO_BF16` | `false` | Run inference in bfloat16 (less VRAM, faster). Needs an NVIDIA GPU with compute capability 8.0+, ignored otherwise. **German WER has not been re-measured under bf16**, so it is opt-in |
| `ASR_MODEL_TTL` / `MODEL_TTL` | `300` | Seconds idle before the ~3 GB model is released. `0` releases as soon as it falls idle, `-1` pins it resident. |
| `ASR_MAX_CONCURRENCY` | `1` | Concurrent inferences on the shared model |
| `ASR_MAX_BATCH` | `8` | Batch size for `/transcribe-batch`; peak VRAM scales with it |
| `ALLOWED_ORIGINS` | `*` | Comma-separated CORS origins |
| `ALLOW_CREDENTIALS` | `false` | Enables CORS credentials when origins are explicit |

## Limits

| Limit | Default | Answer past it |
|-------|---------|----------------|
| `MAX_UPLOAD_MB` per file | `200` | `413`, also from the `Content-Length` header before the body is read (single-file routes). Uploads are streamed to disk in 1 MiB chunks, never held in memory |
| `NEMO_MAX_AUDIO_S` per file | `1500` (25 min) | `413` with a message. Checked from the container header before ffmpeg runs, and again on the converted file; ffmpeg is told to stop at the limit (`-t`), so a file whose header lies cannot be decoded in full first |
| Empty upload | | `400` |
| Audio no decoder can open | | `422` |
| GPU out of memory | | `503` (the CUDA cache is released), not a generic `500` |

NeMo encodes a file in one pass and parakeet-tdt-0.6b-v3 uses full attention,
which NeMo documents as good for about 24 minutes; longer files ran the GPU out
of memory and came back as a generic `500`. `NEMO_MAX_AUDIO_S=0` removes the
guard for a model configured with local attention or a bigger card. The gateway has its own upload cap (`MAX_UPLOAD_MB`,
default 512), so a file between the two limits is refused here.

In `/transcribe-batch` the limits apply to each file: an offending file comes back
as `{"filename", "error", "status"}` (`status` is the HTTP code it would have had)
and the others are still transcribed. If the batched forward pass fails as a
whole, the files are retried one at a time so one unreadable file cannot sink the
rest; a batch that returns fewer results than files reports the missing ones as
errors rather than `null`. Peak VRAM of a batch grows with `ASR_MAX_BATCH` times
the length of its files.

`response_format=text` on `/v1/audio/transcriptions` returns the transcript as
`text/plain`, not a JSON-quoted string.

## NeMo version and torch build

The image installs `nemo_toolkit[asr]` **3.0.x** on a pinned `torch` / `torchaudio` **2.11.0** from PyTorch's CUDA 12.8 index. `cu128` is the first build with Blackwell (`sm_120`, RTX 50 series) kernels, and PyTorch publishes it for 2.7 through 2.11 only (2.12 and later ship CUDA 12.6 / 13.x), so the torch pin cannot simply follow NeMo's own tested stack (torch 2.12, CUDA 12.6 / 13.2). Two build arguments move the versions, nothing needs editing:

| Build argument | Default | Meaning |
|----------------|---------|---------|
| `NEMO_TOOLKIT_SPEC` | `>=3.0.0,<3.1` | pip specifier for `nemo_toolkit[asr]`. Replaces the line in `requirements.txt` (which carries the same default) |
| `TORCH_VERSION` / `TORCHAUDIO_VERSION` | `2.11.0` | Must exist as a `+cu128` wheel; torch and torchaudio are released in lockstep |

```bash
# Roll back to the 2.x line (no file edits; the image tag can stay the same)
docker compose build --build-arg NEMO_TOOLKIT_SPEC='>=2.7.3,<3' parakeet-asr-service
# Accept 3.1 and later 3.x releases once you have tested them
docker compose build --build-arg NEMO_TOOLKIT_SPEC='>=3.0.0,<4' parakeet-asr-service
```

The build **fails on purpose** when the installed torch is not a CUDA 12.8 build with `sm_120` kernels, both right after the torch install and again after NeMo is installed (a resolver must not swap the wheel). `GET /status` reports what a running image was built with: `runtime.nemo_toolkit`, `runtime.torch`, `runtime.torch_cuda` and `runtime.cuda_arch_list`.

Why 3.0.x is the default (nemo_toolkit 3.0.0 was published 2026-08-07, and the old `>=2.0.0` line already resolved to it on a fresh build):

- The API this service calls is unchanged from 2.7.3: the `transcribe()` signatures of the RNNT and multi-task models are identical (3.0 raises `ValueError` where 2.7 asserted, and passes `verbose` through), `Hypothesis` keeps `text` / `timestamp` (`segment`, `word`, `char`), and `ASRModel.from_pretrained`, the `.cpu()` unload hook and the `use_cuda_graph_decoder` option are the same. The ROCm image (`Dockerfile.rocm`) stays on `>=2.7.3,<3`: the `rocm6.2` index ends at torch 2.5.1, below NeMo 3's floor (torch 2.6, README: 2.7), and 3.0 was not tried on ROCm.
- A randomly initialised Parakeet-TDT (CPU, torch 2.11, Python 3.11) run through the real `transcribe(..., timestamps=True)` on 2.7.3 and on 3.0.0 gave identical text and timestamps, also with numpy 2.4. **No GPU run was possible**, so the rewritten TDT decoder and the CUDA-graph path are covered by the checklist below.
- 3.0.0 needs Python 3.11 or later (its sources use 3.11 syntax; its README says 3.12+, its metadata says 3.10+). The image's Python 3.11 works.
- It pulls `transformers` 5 and `huggingface_hub` 1 (2.7.x pinned 4.57 and 0.36). The `numpy<2` cap in `requirements.txt` was NeMo 2.3's; both releases ran on numpy 2.4 and it is gone. `cuda-bindings` stays below 13 to match the CUDA 12.8 torch.
- `PYTORCH_JIT=0` stays: 3.0.0 still contains the `@torch.jit.script` functions (`online_clustering.py`, `vad_utils.py`, ...) it was set for.
- NeMo 3 hardened the `_target_` allow-list used when a `.nemo` is restored. Every `_target_` such a configuration names (preprocessor, encoder, decoder and joint or transformer decoder, token classifier, loss; written down from the NeMo sources, not read from the checkpoint, which could not be downloaded) passes the check when tested directly. A third-party checkpoint with an unusual `_target_` would fail to load with "Instantiation of unsafe target".

### GPU smoke test after a rebuild

```bash
docker compose build parakeet-asr-service && docker compose up -d parakeet-asr-service          # 1. the build fails on a wrong torch wheel
curl -s localhost:5005/health; curl -s localhost:5005/status | jq '.runtime'        # 2. 200 while loading; nemo_toolkit 3.0.x, torch 2.11.0+cu128, sm_120 listed
until curl -sf localhost:5005/ready >/dev/null; do sleep 5; done; curl -s localhost:5005/ready   # 3. 503 "loading", then 200
curl -s -F "audio=@german.wav" localhost:5005/transcribe | jq '{text, segments, processing_time}'  # 4. a German wav you know the transcript of
docker compose build --build-arg NEMO_TOOLKIT_SPEC='>=2.7.3,<3' parakeet-asr-service && docker compose up -d parakeet-asr-service    # 5. rerun step 4 on 2.x and diff text and segment times
```

### Local checkpoints

`ASRModel.from_pretrained` reads every string containing `/` as a Hugging Face repo id, on 2.7.x and 3.x alike, so a path to a `.nemo` file used to fail with `HFValidationError`. When `PARAKEET_ASR_MODEL` names an existing file the service now restores it (`ASRModel.restore_from`, which is what `from_pretrained` calls after downloading). Useful for a German fine-tune such as [parakeet-primeline](https://huggingface.co/primeline/parakeet-primeline) kept on a volume:

```yaml
    volumes: ["./models:/models:ro"]
    environment: {PARAKEET_ASR_MODEL: /models/parakeet-primeline.nemo}
```

A repo id such as `primeline/parakeet-primeline` needs the repo to contain a file named exactly `<repo name>.nemo` (NeMo looks for that name and otherwise treats the repo as an unpacked model directory); this could not be checked from the build environment, which has no access to huggingface.co. A checkpoint that fails to load with a `weights_only` error needs `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` in the container (only for files you trust).

## Notes

- Input audio is auto-converted to 16 kHz mono via `ffmpeg` (handles WAV/MP3/M4A/FLAC). A file that is already a 16 kHz mono WAV is read as is.
- Inference runs with TF32 matmuls (`NEMO_MATMUL_PRECISION=high`), the setting NVIDIA's own `transcribe_speech.py` uses and the one the published RTFx figures were measured with. Duration-sorted batching is not implemented.
- License: NVIDIA Open Model License (see the model card). Heavier dependency footprint than Whisper (pulls in `nemo_toolkit`); model weights download on first start and are cached in the `parakeet-asr-cache` volume.
- CUDA or ROCm GPU recommended for realtime; CPU works but is slower (still competitive vs. Whisper on CPU).

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
| `503` | `load_failed` | The last load failed and none ever succeeded; `detail` carries the error |

The startup preload runs in the background, so the port answers (`/health` 200,
`/ready` 503 `loading`) during the first-time model download instead of staying
closed. A request that arrives meanwhile waits for the load. Gate anything that
needs a usable model on `/ready`, not `/health`.
