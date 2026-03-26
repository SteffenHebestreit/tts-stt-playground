# TrueNAS SCALE Deployment Guide

This guide prepares the repository for use as a dedicated app stack on a TrueNAS SCALE host with one NVIDIA RTX 5060 Ti (16 GB).

## Deployment Goal

The recommended default is a balanced configuration that can run:

- `stt-service`
- `piper-tts-service`
- `qwen3-asr-service`
- `qwen3-tts-service`
- `frontend-service`

Training is supported, but should be treated as a scheduled workload rather than something that always runs next to every inference service.

## Recommended Files

Use these files together:

```bash
docker compose \
  --env-file .env.truenas.example \
  -f docker-compose.yml \
  -f docker-compose.truenas.yml \
  --profile all up -d
```

To include the optional `whisper-cpp` backend in the browser UI as well, set `ENABLE_WHISPER_CPP=true` in the frontend environment and start it explicitly:

```bash
ENABLE_WHISPER_CPP=true docker compose \
  --env-file .env.truenas.example \
  -f docker-compose.yml \
  -f docker-compose.truenas.yml \
  --profile all --profile whisper-cpp up -d
```

If you rename `.env.truenas.example` to `.env`, you can omit `--env-file`.

Use `.env.truenas.production.example` instead when the frontend is opened from a
different machine and must call the backend services through a hostname rather
than `localhost`.

Additional deployment aids:

- `docs/truenas-custom-app-checklist.md`
- `docs/truenas-service-profiles.md`
- `.env.truenas.production.example`

## Storage Layout

Recommended dataset layout on TrueNAS:

```text
/mnt/pool/apps/tts-stt/
  .env
  docker-compose.yml
  docker-compose.truenas.yml
  models/
  output/
  .cache/
  piper-training-service/data/
  piper-training-service/checkpoints/
  piper-training-service/models/
```

### What Should Persist

- `models/`: shared Piper voice models and exported custom voices
- `output/`: generated audio output files
- `.cache/`: Whisper and Python cache data
- `piper-training-service/data/`: prepared datasets
- `piper-training-service/checkpoints/`: resumable training state

## VRAM Planning

Approximate GPU memory usage on this hardware class:

| Service | Expected VRAM | Notes |
|---------|---------------|-------|
| `stt-service` (`medium`) | 2 to 3 GB | Good default for always-on STT |
| `qwen3-asr-service` | ~4 GB | Practical multilingual ASR |
| `qwen3-tts-service` (`0.6B`) | 2.5 to 3 GB | Better fit than 1.7B for 16 GB hosts |
| `piper-training-service` | 2 to 4 GB | Depends on batch pressure and dataset |

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

## Reverse Proxy / Hostname

If TrueNAS exposes this stack behind a hostname instead of `localhost`, set explicit CORS origins and browser-facing backend URLs:

```env
ALLOWED_ORIGINS=http://truenas.example.local:3000
ALLOW_CREDENTIALS=false
BROWSER_TTS_URL=http://truenas.example.local:5000
BROWSER_STT_URL=http://truenas.example.local:5001
BROWSER_TRAINING_URL=http://truenas.example.local:8080
BROWSER_QWEN3_ASR_URL=http://truenas.example.local:5002
BROWSER_QWEN3_TTS_URL=http://truenas.example.local:5004
```

Only set `ALLOWED_ORIGINS` to the frontend origin that the browser will load,
not to every backend URL. Leave `ALLOW_CREDENTIALS=false` unless you introduce
cookie-based auth.

## Health Checks

Once deployed, verify the stack:

```bash
curl http://localhost:3000/health
curl http://localhost:5000/health
curl http://localhost:5001/health
curl http://localhost:5002/health
curl http://localhost:5004/health
curl http://localhost:8080/health
```

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