# TrueNAS Custom App Checklist

Use this checklist when turning the repository into a TrueNAS SCALE Custom App deployment.

> For a full walkthrough (prerequisites, GPU setup, verification, troubleshooting) see
> [`truenas-installation-guide.md`](./truenas-installation-guide.md).

## Before You Start

- Confirm the NVIDIA GPU is visible in TrueNAS SCALE
- Confirm the repo is stored on a persistent dataset, for example `/mnt/pool/apps/tts-stt` (needed to build the images)
- Copy `.env.truenas.example` to `.env` and set `APP_DATA_DIR`
- Decide whether you need backend ports published at all — the web UI does not need them
  (everything is proxied through the frontend on port 3000)

## Dataset Layout

Set **one** variable and every data directory is created under it:

```env
APP_DATA_DIR=/mnt/pool/apps/tts-stt
```

That produces `models/`, `output/`, `.cache/`, and `piper-training-service/{data,checkpoints,models}/` under the dataset. Optionally relocate the big Hugging Face caches with `QWEN3_TTS_CACHE_DIR`, `QWEN3_ASR_CACHE_DIR`, `PARAKEET_ASR_CACHE_DIR`, `WHISPER_CPP_MODELS_DIR`.

## TrueNAS App Setup Steps

1. Create or select a dataset, for example `/mnt/pool/apps/tts-stt`.
2. Clone or copy the repository into that dataset (required to build images).
3. Copy `.env.truenas.example` to `.env`; set `APP_DATA_DIR`.
4. If reached over the network, set `ALLOWED_ORIGINS` to the origin the browser loads.
   `BROWSER_*_URL` is not needed — see the note under *First Startup Validation*.
5. Launch the stack with:

```bash
docker compose -f docker-compose.yml -f docker-compose.truenas.yml --profile all up -d
```

(The base + `truenas` overlay run from the built images — do **not** add `docker-compose.dev.yml`, which is for local source editing only.)

Optional STT backends, each opt-in via its own profile + frontend flag:

- `whisper-cpp`: `ENABLE_WHISPER_CPP=true` + `--profile whisper-cpp`
- `parakeet-asr` (25 EU langs incl. German): `ENABLE_PARAKEET_ASR=true` + `--profile parakeet-asr`
- `canary-asr` (fastest German STT): `ENABLE_CANARY_ASR=true` + `--profile canary-asr`
- `chatterbox-tts` (streaming German TTS): `ENABLE_CHATTERBOX_TTS=true` + `--profile chatterbox-tts`

## First Startup Validation

Run these checks after the first deployment:

```bash
docker compose ps

# Gateway liveness
curl http://localhost:3000/health

# Every backend at once, probed concurrently over the internal network.
# Reports healthy / latency_ms / model_loaded / model_size / device per provider.
curl http://localhost:3000/api/health | jq
```

Backend ports (5000-5007, 8080) only answer directly if you chose to publish
them; the web UI never uses them.

If using Apps -> Discover -> Install via YAML instead of a shell compose
command, pin `qwen3-asr-service` and `qwen3-tts-service` to the same GPU with
`device_ids: ["0"]`, `NVIDIA_VISIBLE_DEVICES=0`, and `CUDA_VISIBLE_DEVICES=0`.
Do not combine `device_ids` with `count` in the same reservation block.

## Common Decisions

- Always-on inference only: enable `frontend`, `piper-tts`, `stt`, `qwen3-asr`, `qwen3-tts`
- Training sessions: stop `qwen3-tts-service` before enabling `piper-training-service`
- Lowest-memory mode: run `frontend`, `piper-tts`, and `stt` only
- Remote browser access: set `ALLOWED_ORIGINS`; no backend ports or `BROWSER_*_URL` needed
- Microphone from another machine: requires HTTPS (browsers demand a secure context)

## Cleanup From Older Repo States

Only if these paths are not in active use:

```bash
rm -rf cache/tts cache/xtts-cache
rm -f models/luna.json models/luna.onnx
```