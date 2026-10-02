# TrueNAS SCALE: build from source (reference)

> **Installing or updating?** Use **[`truenas-installation-guide.md`](./truenas-installation-guide.md)**:
> it is the complete guide for the prebuilt-image route. This page covers only what is specific to
> building the images on the NAS: the compose files, the storage layout and the GPU driver.

Training is supported, but treat it as a scheduled workload rather than something that always
runs next to every inference service.

## When to use this route

Use it to change the code or when the NAS cannot pull from GHCR. Everyone else should paste
[`docker-compose.truenas-app.yml`](../docker-compose.truenas-app.yml) into Apps -> Custom App
(prebuilt images, no clone, no build).

## How the compose files fit together

| File | Role |
|---|---|
| `docker-compose.truenas-app.yml` | Standalone, image-only stack for **Apps -> Custom App -> Install via YAML**. Never merged with another file. |
| `docker-compose.yml` | Base stack. Runs from the built images and persists data through configurable host paths. |
| `docker-compose.truenas.yml` | Overlay for a single-GPU host: pins every image to one release and every GPU service to card `0`. |
| `docker-compose.dev.yml` | **Local development only.** It live-mounts source over the images: never use it on a NAS. |

## Launch

```bash
cd /mnt/tank/apps/tts-stt            # your clone
git checkout v0.2.0                  # the release IMAGE_TAG in the overlay names
cp .env.truenas.example .env         # set APP_DATA_DIR; the file explains the rest
docker compose -f docker-compose.yml -f docker-compose.truenas.yml --profile all up -d --build
```

The build takes 20 to 40 minutes. Without `--build`, `up` pulls the pinned release from GHCR
instead. `--profile all` is the default set; add `--profile training`, `--profile canary-asr`,
`--profile parakeet-asr`, `--profile chatterbox-tts`, `--profile magpie-tts` or `--profile whisper-cpp` for the opt-in
services, and set the matching `ENABLE_*` flag so the UI shows them
([profiles and modes](./truenas-service-profiles.md)).

Update: `git fetch --tags && git checkout vX.Y.Z`, set `IMAGE_TAG=X.Y.Z` in `.env`, run the same
`up -d --build`. Roll back by checking out the previous tag and setting the previous `IMAGE_TAG`.

## Storage layout

One variable, **`APP_DATA_DIR`**, points every data directory at a dataset:

```env
APP_DATA_DIR=/mnt/tank/apps/tts-stt
```

```text
/mnt/tank/apps/tts-stt/
  models/                              Piper voices + exported custom voices
  output/                              generated audio
  .cache/                              Whisper / Python cache
  settings/                            what the web UI's Settings page saves
  backend-settings/                    backend settings saved there (later releases)
  piper-training-service/{data,checkpoints,models,configs}/
```

Each path can be overridden on its own (`MODELS_DIR`, `OUTPUT_DIR`, `CACHE_DIR`, `SETTINGS_DIR`,
`BACKEND_SETTINGS_DIR`, `TRAINING_DATA_DIR`, `TRAINING_CHECKPOINTS_DIR`, `TRAINING_MODELS_DIR`,
`TRAINING_CONFIGS_DIR`).
The large Hugging Face downloads (Qwen3, Parakeet, Canary, Chatterbox, Magpie) and the whisper.cpp models
default to **named Docker volumes** in the Docker root. To keep them on your pool set
`QWEN3_TTS_CACHE_DIR`, `QWEN3_ASR_CACHE_DIR`, `PARAKEET_ASR_CACHE_DIR`, `CANARY_ASR_CACHE_DIR`,
`CHATTERBOX_TTS_CACHE_DIR`, `MAGPIE_TTS_CACHE_DIR`, `QWEN3_TTS_VOICES_DIR` and `WHISPER_CPP_MODELS_DIR`. (The Custom App
file uses one shared `cache/` directory for all of them instead.)

## GPU driver

The Dockerfiles use `nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04` and PyTorch cu128 wheels, which
carry kernels from Ampere (sm_80) up to Blackwell (sm_120). Earlier CUDA 12.1 images failed on
Blackwell with `cudaErrorNoKernelImageForDevice`.

- **Verified** (read from the published config of `nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04`, and
  from NVIDIA's `libnvidia-container` source for how it is evaluated): the image only starts when the
  driver reports CUDA >= 12.8 (R570 or newer), **or** the GPU is a data-centre or workstation model
  (brands `tesla`, `quadro`, `nvidia`, `nvidiartx`, ...) on the branch 470, 535, 550, 560 or 565
  (CUDA forward compatibility). `geforce` is not in the list: **a GeForce card needs R570 or newer**,
  whatever its branch. The failure looks like `unsatisfied condition: cuda>=12.8, please update your
  driver to a newer version`.
- **Not verified:** NVIDIA's own minimum for Blackwell cards. R570 is the first branch with CUDA
  12.8 and is what `scripts/truenas/preflight.sh` requires for a Blackwell GPU.

`scripts/truenas/preflight.sh` reads your driver and GPU and says which of these applies.

## Pinning the GPU in a hand-written Custom App YAML

If you merge your own YAML instead of using the file above, give every GPU service the same
reservation. Use `device_ids` or `count`, never both:

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          device_ids: ["0"]
          capabilities: [gpu]
```

## Health checks

```bash
curl -s http://localhost:3000/health       # the gateway, liveness only
curl -s http://localhost:3000/api/health   # every backend, probed concurrently
```

Each backend entry reports `healthy`, `latency_ms`, `model_loaded`, `model_size` and `device`, so
`"device": "cpu"` tells you the GPU did not reach the container. Backend ports answer directly only
from the NAS itself (`BACKEND_BIND_ADDR`, default `127.0.0.1`).

## Clean up leftovers from older versions

In the repository checkout (not the dataset), only if these paths are not in use:

```bash
rm -rf cache/tts cache/xtts-cache
rm -f models/luna.json models/luna.onnx
```
