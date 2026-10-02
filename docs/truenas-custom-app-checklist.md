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
- [ ] Optional services: delete the `profiles:` line, then offer the service under **Settings -> Engines**
      in the web UI (or set its `ENABLE_*` flag; voice training's is `ENABLE_TRAINING`).
- [ ] Install. The app shows **Running** within about a minute; model downloads continue in the background.

## After

- [ ] `curl -s http://localhost:3000/api/health`: every backend `healthy`, GPU services `"device": "cuda"`.
- [ ] Web UI at `http://<truenas-ip>:3000`: generate speech (Text-to-Speech tab).
- [ ] Microphone test over `https://` or `localhost` (Live Transcription).
- [ ] Settings (`http://<truenas-ip>:3000/settings`) loads. Without `API_KEY`, claim it: ask for a
      one-time code and read it under Workloads -> frontend-service -> View Logs.
- [ ] Snapshot the dataset (Datasets -> Data Protection) or add a periodic snapshot task.
- [ ] Optional: `diun` for release notifications, `update-check.sh` in a cron job.

## Decisions

- Remote browser access: leave `ALLOWED_ORIGINS` empty. An IP address, `*.local` or a plain name (`truenas`) needs nothing; a real domain or reverse proxy needs `TRUSTED_HOSTS` (else `403 host_not_allowed`), and `TRUSTED_ORIGINS` too if the proxy rewrites `Host`.
- `API_KEY` (optional): protects `/v1/*`, mutating `/api/*` and `/ws/stt`; the web UI asks for it once per tab, there is no exemption for it.
- Training: free `qwen3-tts-service`'s VRAM first on a small card (`POST /api/providers/qwen3/unload`).
- Lowest memory: run `frontend`, `piper-tts` and `stt` only.
