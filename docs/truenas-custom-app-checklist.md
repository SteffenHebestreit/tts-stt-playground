# TrueNAS: first-run checklist

One page to tick off. The steps and the reasons are in
[`truenas-installation-guide.md`](./truenas-installation-guide.md).

## Before

- [ ] TrueNAS SCALE 24.10 or newer.
- [ ] NVIDIA driver installed (Apps -> Settings -> Install NVIDIA Drivers) with CUDA 12.8 support
      (R570 or newer: every GeForce card needs it, and an RTX 50 series card cannot run on less).
      `nvidia-smi` works in the shell.
- [ ] Dataset created, e.g. `/mnt/tank/apps/tts-stt`.
- [ ] `scripts/truenas/preflight.sh --data-dir <dataset> --create` ends with `READY`.
- [ ] The release you are about to pin is published: `scripts/truenas/update-check.sh --tag <release>`.
- [ ] Optional: `scripts/truenas/pull-images.sh --tag <release>` (TrueNAS aborts an Apps job after 20 minutes).

## Install

- [ ] Apps -> Discover Apps -> Custom App -> Install via YAML, name `tts-stt`.
- [ ] Paste `docker-compose.truenas-app.yml`; replace `${APP_DATA_DIR:?set dataset path}` with the dataset path.
- [ ] Optional services: delete the `profiles:` line **and** set the `ENABLE_*` flag (training has none).
- [ ] Install. The app shows **Running** within about a minute; model downloads continue in the background.

## After

- [ ] `curl -s http://localhost:3000/api/health`: every backend `healthy`, GPU services `"device": "cuda"`.
- [ ] Web UI at `http://<truenas-ip>:3000`: generate speech (Text-to-Speech tab).
- [ ] Microphone test over `https://` or `localhost` (Live Transcription).
- [ ] Snapshot the dataset (Datasets -> Data Protection) or add a periodic snapshot task.
- [ ] Optional: `diun` for release notifications, `update-check.sh` in a cron job.

## Decisions

- Remote browser access: leave `ALLOWED_ORIGINS` empty; behind a proxy that rewrites `Host`, set `TRUSTED_ORIGINS`.
- Training: free `qwen3-tts-service`'s VRAM first on a small card (`POST /api/providers/qwen3/unload`).
- Lowest memory: run `frontend`, `piper-tts` and `stt` only.
