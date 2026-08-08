# TrueNAS Service Profile Recommendations

These recommendations assume one NVIDIA RTX 5060 Ti with 12 GB of VRAM dedicated to this repository.

## Always-On Services

Recommended for day-to-day operation:

- `frontend-service`
- `piper-tts-service`
- `stt-service`
- `qwen3-asr-service`
- `qwen3-tts-service` using `Qwen/Qwen3-TTS-12Hz-0.6B-Base`

This mode gives the broadest feature coverage while keeping GPU pressure reasonable.
On the single-GPU TrueNAS profile, `qwen3-asr-service` and `qwen3-tts-service`
both target GPU `0`.

## On-Demand Services

Recommended to start only when needed:

- `piper-training-service`
- `whisper-cpp`
- `parakeet-asr-service`

Reasons:

- `piper-training-service` competes for VRAM with the Qwen services
- `whisper-cpp` is useful as an alternative STT backend, but not required if `stt-service` is already running
- if you do enable `whisper-cpp`, set `ENABLE_WHISPER_CPP=true` in the frontend environment so it appears in the browser UI
- `parakeet-asr-service` is the fastest STT (25 EU languages incl. German) but pulls a heavy NeMo image and competes for VRAM; enable with `--profile parakeet-asr` and `ENABLE_PARAKEET_ASR=true`

## Recommended Modes

### Mode A: Daily Use

- `frontend`
- `piper-tts`
- `stt`
- `qwen3-asr`
- `qwen3-tts`

### Mode B: Training Window

Stop:

- `qwen3-tts-service`

Then start or keep running:

- `frontend-service`
- `piper-tts-service`
- `stt-service`
- `qwen3-asr-service`
- `piper-training-service`

### Mode C: Minimal Footprint

- `frontend-service`
- `piper-tts-service`
- `stt-service`

Use this when the host should remain responsive and you do not need Qwen-based inference.

## Suggested Commands

### Daily Use

```bash
docker compose --env-file .env -f docker-compose.yml -f docker-compose.truenas.yml \
  --profile frontend \
  --profile piper-tts \
  --profile stt \
  --profile qwen3-asr \
  --profile qwen3-tts up -d
```

### Training Window

```bash
docker compose stop qwen3-tts-service
docker compose --env-file .env -f docker-compose.yml -f docker-compose.truenas.yml --profile training up -d
```

## Browser Access Strategy

The browser talks only to the frontend on port 3000, which proxies everything
else over the internal Docker network (`/api/*`, `/api/health`, `/ws/stt`).

So when the frontend is accessed from another machine, there is exactly one
thing to set:

```env
ALLOWED_ORIGINS=http://truenas.example.local:3000
```

You do **not** need to publish backend ports, and the `BROWSER_*_URL` variables
are not used by the web UI. Set them only if you deliberately publish backend
ports to call those APIs directly from your own tools.

To use the **microphone** from another machine you need HTTPS — browsers only
grant microphone access in a secure context. See
[`truenas-installation-guide.md`](./truenas-installation-guide.md#7-reverse-proxy--https).