# TrueNAS Custom App Checklist

Use this checklist when turning the repository into a TrueNAS SCALE Custom App deployment.

## Before You Start

- Confirm the NVIDIA GPU is visible in TrueNAS SCALE
- Confirm the repo is stored on a persistent dataset, for example `/mnt/pool/apps/tts-stt`
- Copy `.env.example` to `.env`
- Decide whether the frontend will access services via `localhost` ports or a TrueNAS hostname

## Dataset Layout

Recommended persistent paths:

- `/mnt/pool/apps/tts-stt/models`
- `/mnt/pool/apps/tts-stt/output`
- `/mnt/pool/apps/tts-stt/.cache`
- `/mnt/pool/apps/tts-stt/piper-training-service/data`
- `/mnt/pool/apps/tts-stt/piper-training-service/checkpoints`
- `/mnt/pool/apps/tts-stt/piper-training-service/models`

## TrueNAS App Setup Steps

1. Create or select a dataset, for example `/mnt/pool/apps/tts-stt`.
2. Clone or copy the repository into that dataset.
3. Copy `.env.example` to `.env`.
4. If using a hostname, edit `ALLOWED_ORIGINS` and the `BROWSER_*_URL` variables in `.env`.
5. Launch the stack with:

```bash
docker compose --env-file .env -f docker-compose.yml -f docker-compose.truenas.yml --profile all up -d
```

If you want `whisper-cpp` visible in the frontend as an alternative STT backend, set `ENABLE_WHISPER_CPP=true` and add `--profile whisper-cpp` to that command.

## First Startup Validation

Run these checks after the first deployment:

```bash
docker compose ps
curl http://localhost:3000/health
curl http://localhost:5000/health
curl http://localhost:5001/health
curl http://localhost:5002/health
curl http://localhost:5004/health
```

If `piper-training-service` is enabled, also validate:

```bash
curl http://localhost:8080/health
```

If using Apps -> Discover -> Install via YAML instead of a shell compose
command, pin `qwen3-asr-service` and `qwen3-tts-service` to the same GPU with
`device_ids: ["0"]`, `NVIDIA_VISIBLE_DEVICES=0`, and `CUDA_VISIBLE_DEVICES=0`.
Do not combine `device_ids` with `count` in the same reservation block.

## Common Decisions

- Always-on inference only: enable `frontend`, `piper-tts`, `stt`, `qwen3-asr`, `qwen3-tts`
- Training sessions: stop `qwen3-tts-service` before enabling `piper-training-service`
- Lowest-memory mode: run `frontend`, `piper-tts`, and `stt` only
- Remote browser access: start from `.env.example` and edit the `BROWSER_*_URL` values

## Cleanup From Older Repo States

Only if these paths are not in active use:

```bash
rm -rf cache/tts cache/xtts-cache
rm -f models/luna.json models/luna.onnx
```