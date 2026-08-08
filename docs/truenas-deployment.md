# TrueNAS SCALE Deployment Guide

> **Installing for the first time?** Use
> **[`docs/truenas-installation-guide.md`](./truenas-installation-guide.md)** instead — it is the
> complete step-by-step walkthrough. This page is the reference for building from source,
> storage layout, and VRAM planning.

This guide prepares the repository for use as a dedicated app stack on a TrueNAS SCALE host with one NVIDIA RTX 5060 Ti (12 GB).

## Deployment Goal

The recommended default is a balanced configuration that can run:

- `stt-service`
- `piper-tts-service`
- `qwen3-asr-service`
- `qwen3-tts-service`
- `frontend-service`

Training is supported, but should be treated as a scheduled workload rather than something that always runs next to every inference service.

## Two ways to deploy

1. **Prebuilt images (no clone/build)** — pull published GHCR images and either
   paste [`docker-compose.truenas-app.yml`](../docker-compose.truenas-app.yml)
   into Apps → Custom App → *Install via YAML*, or add the repo as a custom
   catalog for a guided form. See [`truenas/README.md`](../truenas/README.md).
   Best for most users.
2. **Build from source (this guide)** — clone the repo onto a dataset and build
   the images locally with the compose overlay below. Use this to customize code
   or when you cannot pull from GHCR.

## How the compose files fit together

| File | Role |
|------|------|
| `docker-compose.yml` | Base stack. Runs **from the built images** and persists data through configurable host paths. This is what you deploy. |
| `docker-compose.truenas.yml` | Single-GPU host overlay: pins all GPU services to card `0` and picks 12 GB-friendly models. |
| `docker-compose.dev.yml` | **Local development only** — live-mounts source files over the images. Do **not** use it on TrueNAS. |

So the base file is deployment-ready on its own; you no longer need the repo's source files mounted into the containers at runtime (only to build the images).

## Recommended Files

Copy the TrueNAS preset and launch with the overlay:

```bash
cp .env.truenas.example .env        # then edit APP_DATA_DIR + hostnames
docker compose \
  -f docker-compose.yml \
  -f docker-compose.truenas.yml \
  --profile all up -d
```

`.env.truenas.example` ships in the repo and already sets `APP_DATA_DIR`, balanced models, and GPU pinning. (For local development with live code reload, add `-f docker-compose.dev.yml` and use `.env.example` instead.)

To include the optional `whisper-cpp` backend in the browser UI as well, set `ENABLE_WHISPER_CPP=true` in the frontend environment and start it explicitly:

```bash
ENABLE_WHISPER_CPP=true docker compose \
  --env-file .env.example \
  -f docker-compose.yml \
  -f docker-compose.truenas.yml \
  --profile all --profile whisper-cpp up -d
```

If you rename `.env.example` to `.env`, you can omit `--env-file`.

When the frontend is opened from another machine, edit `ALLOWED_ORIGINS` to the
origin the browser loads.

> **`BROWSER_*_URL` is no longer needed for the web UI.** The browser now reaches
> every backend through the frontend gateway on port 3000 (`/api/*`,
> `/api/health`, `/ws/stt`), so backend ports do not have to be published or
> reachable. Set the `BROWSER_*_URL` values only if you deliberately publish
> backend ports to call them directly from your own tools.

Additional deployment aids:

- `docs/truenas-custom-app-checklist.md`
- `docs/truenas-service-profiles.md`
- `.env.example`

## Storage Layout

All persistent data is controlled by a single variable, **`APP_DATA_DIR`**. Point
it at a dataset and every data directory is created underneath it — you do not
have to keep data inside the repo. Set it in `.env`:

```env
APP_DATA_DIR=/mnt/pool/apps/tts-stt
```

Resulting layout on the pool:

```text
/mnt/pool/apps/tts-stt/
  models/                              # shared Piper voices + exported custom voices
  output/                              # generated audio
  .cache/                             # Whisper / Python cache
  piper-training-service/data/         # prepared datasets
  piper-training-service/checkpoints/  # resumable training state
  piper-training-service/models/       # training-side model exports
```

Each path can be overridden individually (`MODELS_DIR`, `OUTPUT_DIR`, `CACHE_DIR`,
`TRAINING_DATA_DIR`, `TRAINING_CHECKPOINTS_DIR`, `TRAINING_MODELS_DIR`,
`TRAINING_CONFIGS_DIR`) if you want to spread data across datasets.

### Model-download caches

The large Hugging Face downloads (Qwen3-TTS/ASR, Parakeet) and the whisper.cpp
GGUF models default to **named Docker volumes** (they live in the docker root).
To keep these multi-GB downloads on your pool instead, set them to dataset
paths: `QWEN3_TTS_CACHE_DIR`, `QWEN3_ASR_CACHE_DIR`, `PARAKEET_ASR_CACHE_DIR`,
`QWEN3_TTS_VOICES_DIR`, `WHISPER_CPP_MODELS_DIR`.

The repo only needs to live on the host so the images can be **built**; runtime
data is wherever `APP_DATA_DIR` points.

## CUDA / Driver Requirements

The Dockerfiles use `nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04` and PyTorch cu128 wheels.

| Requirement | Minimum |
|---|---|
| NVIDIA driver | 550 (570+ recommended) |
| CUDA base image | 12.8.1 |
| PyTorch wheel | cu128 |
| GPU architecture | Ampere (sm_80) through Blackwell (sm_120) |

This was validated on an **RTX 5060 Ti (Blackwell, sm_120)** on TrueNAS SCALE.
Earlier CUDA 12.1 images produced `cudaErrorNoKernelImageForDevice` on Blackwell GPUs because the cu121 PyTorch wheels only include kernels up to sm_90.
The cu128 wheels are backward-compatible with Ampere and Ada Lovelace cards.

## VRAM Planning

Approximate GPU memory usage on this hardware class:

| Service | Expected VRAM | Notes |
|---------|---------------|-------|
| `stt-service` (`large-v3-turbo`) | ~1.6 GB | Default. Best latency/accuracy trade; cannot translate |
| `stt-service` (`large-v3`) | ~3.1 GB | Only option that supports `task=translate` |
| `qwen3-asr-service` | ~4 GB | Practical multilingual ASR |
| `qwen3-tts-service` (`0.6B`) | 2.5 to 3 GB | The default. Required on a 12 GB card; the 1.7B does not fit alongside STT |
| `piper-training-service` | 2 to 4 GB | Depends on batch pressure and dataset |
| `canary-asr-service` (optional) | ~2 GB | Fastest German STT (180M). Opt-in (`--profile canary-asr` + `ENABLE_CANARY_ASR=true`) |
| `parakeet-asr-service` (optional) | ~2 to 3 GB | 25 EU langs incl. German. Opt-in (`--profile parakeet-asr` + `ENABLE_PARAKEET_ASR=true`) |
| `chatterbox-tts-service` (optional) | ~4 GB | Streaming German TTS — lowest time-to-first-audio. Opt-in (`--profile chatterbox-tts` + `ENABLE_CHATTERBOX_TTS=true`) |

### Recommended Operating Modes

1. Always-on inference: `frontend`, `piper-tts`, `stt`, `qwen3-asr`, `qwen3-tts`
2. Training window: stop `qwen3-tts` first, then start `piper-training-service`
3. Low-memory mode: run `frontend`, `piper-tts`, and `stt` only

## Compose Profiles

### Full stack

```bash
docker compose -f docker-compose.yml -f docker-compose.truenas.yml --profile all up -d
```

### Inference-focused stack

```bash
docker compose -f docker-compose.yml -f docker-compose.truenas.yml \
  --profile frontend \
  --profile piper-tts \
  --profile stt \
  --profile qwen3-asr \
  --profile qwen3-tts up -d
```

### Training session

```bash
docker compose stop qwen3-tts-service
docker compose -f docker-compose.yml -f docker-compose.truenas.yml --profile training up -d
```

## Custom App YAML GPU Pinning

If you deploy through TrueNAS Apps -> Discover -> Install via YAML instead of a
shell `docker compose` command, pin both Qwen services to the same GPU
explicitly. For a single NVIDIA card shared by `qwen3-asr-service` and
`qwen3-tts-service`, use the same `device_ids` entry on both services:

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          device_ids: ["0"]
          capabilities: [gpu]
environment:
  NVIDIA_VISIBLE_DEVICES: "0"
  CUDA_VISIBLE_DEVICES: "0"
```

Apply that block to both Qwen services if they should share GPU `0`.
Use either `device_ids` or `count`, never both in the same device reservation.
If you keep using `docker-compose.truenas.yml`, the overlay already pins the
GPU-facing environment variables to `0` for the single-GPU host profile.

## Reverse Proxy / Hostname

All browser traffic goes through the frontend on port 3000, so a reverse proxy
needs exactly one upstream. Set only the CORS origin:

```env
ALLOWED_ORIGINS=https://voice.example.com
ALLOW_CREDENTIALS=false
```

Two proxy settings are load-bearing:

- Forward the `Upgrade`/`Connection` headers — live transcription is a WebSocket
  (`/ws/stt`) and silently fails without them.
- Disable response buffering — streaming TTS starts playing before synthesis
  finishes, and a buffering proxy restores full-text latency.

A worked nginx config is in
[`truenas-installation-guide.md`](./truenas-installation-guide.md#7-reverse-proxy--https).

HTTPS is **required** to use the microphone from any machine other than the NAS
itself: browsers only grant microphone access in a secure context.

Leave `ALLOW_CREDENTIALS=false` unless you introduce cookie-based auth.

## Health Checks

One call probes every backend concurrently over the internal network:

```bash
curl http://localhost:3000/api/health | jq
```

Each entry reports `healthy`, `latency_ms`, `model_loaded`, `model_size` and
`device` — so `"device": "cpu"` immediately tells you the GPU was not allocated.

The gateway's own liveness probe stays lightweight:

```bash
curl http://localhost:3000/health
```

Backend ports only answer directly if you chose to publish them.

## Maintenance

### Update images / rebuild services

```bash
docker compose -f docker-compose.yml -f docker-compose.truenas.yml build
docker compose -f docker-compose.yml -f docker-compose.truenas.yml up -d
```

### Clean obsolete local leftovers from older XTTS versions

```bash
rm -rf cache/tts cache/xtts-cache
rm -f models/luna.json models/luna.onnx
```

Only remove those paths if they are not in active use on your host.