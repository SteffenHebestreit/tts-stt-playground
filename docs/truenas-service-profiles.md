# TrueNAS: which services to run when

Sizes (VRAM, downloads) are in one place: the table in
[`truenas-installation-guide.md`](./truenas-installation-guide.md#6-vram-and-disk-planning).
Budget against the VRAM `nvidia-smi` reports for your card.

## Always on

- `frontend-service`, `piper-tts-service`
- `stt-service`, `qwen3-asr-service`, `qwen3-tts-service` (the `0.6B` model)

This gives the broadest feature coverage at reasonable GPU pressure. All GPU services share the
card you select (`GPU_DEVICE_ID`).

## Start only when needed

- `piper-training-service`: competes for VRAM with the Qwen services. Run it in a training window.
- `parakeet-asr-service`: 25 European languages including German, heavy NeMo image.
- `canary-asr-service`: the fastest German STT, also NeMo.
- `cohere-asr-service`: Cohere Transcribe (2B, 14 languages including German), about 4.2 GB of VRAM.
- `granite-asr-service`: IBM Granite Speech 5.0, the 2b with keyword biasing (about 4.7 GB of VRAM) or
  the 470m TurboCTC (about 1.2 GB).
- `vllm-asr-service`: one of the STT candidates (`VLLM_ASR_MODEL`: Granite 2b, Cohere, Qwen3-ASR or
  Whisper turbo German) under vLLM's batching, for throughput. 45 % of the card (about 7.2 GB of
  16 GB) while awake, 0.4-0.9 GB asleep. Its image is built on the NAS (vllm-asr-service/README.md).
- `chatterbox-tts-service`: streaming German TTS.
- `magpie-tts-service`: NVIDIA TTS with five built-in voices that reads numbers and dates correctly.
- `whisper-cpp`: CPU-only STT alternative; not needed while `stt-service` runs.

The web UI only shows a backend that is offered: ticked under **Settings -> Engines** in the web
UI, or its `ENABLE_*` flag true on `frontend-service` (`ENABLE_PARAKEET_ASR`, `ENABLE_CANARY_ASR`,
`ENABLE_COHERE_ASR`, `ENABLE_GRANITE_ASR`, `ENABLE_VLLM_ASR`, `ENABLE_CHATTERBOX_TTS`, `ENABLE_MAGPIE_TTS`,
`ENABLE_WHISPER_CPP`, and `ENABLE_TRAINING` for the
Voice Training tab, which the Custom App file sets to false).

## Modes

| Mode | Services |
|---|---|
| **A. Daily use** | `frontend`, `piper-tts`, `stt`, `qwen3-asr`, `qwen3-tts` |
| **B. Training window** | Mode A without `qwen3-tts-service`, plus `piper-training-service` |
| **C. Minimal footprint** | `frontend`, `piper-tts`, `stt`; add `whisper-cpp` instead of `stt` for no GPU at all |

## Switching modes

**Custom App YAML** (`docker-compose.truenas-app.yml`): optional services are dormant behind a
`profiles:` line. Delete the line to enable one, put it back to disable it; the exact edits are
in [the guide](./truenas-installation-guide.md#optional-services). For a training window, free
the VRAM of the Qwen3-TTS model without touching the app (it reloads on its next use; the idle
container keeps about 0.5 GB for its CUDA context):

```bash
curl -X POST http://localhost:3000/api/providers/qwen3/unload    # add -H "Authorization: Bearer <API_KEY>" if you set one
```

**Compose from source:**

```bash
# Mode A
docker compose --env-file .env -f docker-compose.yml -f docker-compose.truenas.yml --profile all up -d

# Mode B: free the VRAM first, then start training
docker compose stop qwen3-tts-service
docker compose --env-file .env -f docker-compose.yml -f docker-compose.truenas.yml --profile training up -d

# Mode C
docker compose --env-file .env -f docker-compose.yml -f docker-compose.truenas.yml \
  --profile frontend --profile piper-tts --profile stt up -d
```

## Browser access

The browser talks only to the frontend on port 3000, which proxies everything else internally
(`/api/*`, `/api/health`, `/ws/stt`). Leave `ALLOWED_ORIGINS` empty: it is only for scripts in
other web pages. Open the UI by IP address, `*.local` or a plain name; a real domain or a
Tailscale name must be listed in `TRUSTED_HOSTS` (`403 host_not_allowed` otherwise). The microphone
from another machine needs HTTPS
([reverse proxy](./truenas-installation-guide.md#8-reverse-proxy-and-https)).
