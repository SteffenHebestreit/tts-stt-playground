# TrueNAS SCALE — Installation Guide

A complete, start-to-finish install of the TTS-STT stack on TrueNAS SCALE. Follow it top to
bottom; nothing is assumed except a working TrueNAS box and, for GPU use, an NVIDIA card.

Reading time ~10 minutes. Actual install ~15 minutes plus model downloads.

- **New here? Use [Path A](#path-a--install-via-yaml-recommended).** It is the supported route.
- Already running an older version? Jump to [Upgrading](#upgrading-from-an-earlier-install).

---

## 1. What you are installing

| Component | Purpose | GPU |
|---|---|---|
| `frontend-service` | Web UI **and the gateway every browser request goes through** | no |
| `piper-tts-service` | Fast CPU text-to-speech, 40+ voices incl. German | no |
| `stt-service` | Whisper speech-to-text + **live microphone transcription** | yes |
| `qwen3-asr-service` | Multilingual speech recognition | yes |
| `qwen3-tts-service` | Voice cloning / high-quality TTS | yes |
| *optional* `chatterbox-tts` | Streaming German TTS — speech starts before the text is finished | yes |
| *optional* `canary-asr` | Fastest German STT (180M params) | yes |
| *optional* `parakeet-asr` | 25 EU languages incl. German | yes |
| *optional* `piper-training` | Train your own voice from recordings | yes |
| *optional* `whisper-cpp` | CPU-only STT, no GPU required | no |

### Only one port is published

The browser talks **only** to the frontend on port `3000`. It proxies everything else over the
internal Docker network:

```
browser  ──►  :3000  ──┬──►  /api/*        TTS, STT, voices, training
                       ├──►  /api/health   all backend probes, run concurrently
                       └──►  /ws/stt       live microphone (WebSocket relay)
```

This matters more than it sounds:

- You do **not** need to publish ports 5000–5007 or 8080.
- You do **not** need to set any `BROWSER_*_URL` variable, even when reaching the NAS from
  another machine. (Older versions of this project required it. That is now obsolete.)
- It works behind an HTTPS reverse proxy. Browsers block `ws://` from an `https://` page as
  mixed content, so the old direct-to-port design could not do live microphone over HTTPS at all.

---

## 2. Prerequisites

### TrueNAS version

TrueNAS SCALE **Electric Eel (24.10) or newer**. These releases run apps on Docker. Earlier
Kubernetes-based releases (Bluefin, Cobia, Dragonfish) are **not** supported by this guide.

Check under **System → Update**.

### For GPU services

| Requirement | Minimum |
|---|---|
| NVIDIA driver | 550 (570+ recommended) |
| GPU architecture | Ampere (sm_80) → Blackwell (sm_120) |
| VRAM | 8 GB minimum, 16 GB comfortable |

Verify the GPU is visible to TrueNAS:

```bash
# In the TrueNAS shell
nvidia-smi
```

If that fails, install the NVIDIA drivers under **Apps → Configuration → Settings** and reboot
before continuing. TrueNAS will not expose a GPU it cannot see.

> **No GPU?** The stack still runs. Install `frontend` + `piper-tts` (both CPU-only) and
> optionally `whisper-cpp` for CPU speech-to-text. Skip every GPU service.

### Disk space

| What | Size |
|---|---|
| Container images | ~25 GB (torch + CUDA are large) |
| Whisper `large-v3-turbo` | ~1.6 GB |
| Qwen3-ASR 1.7B | ~4 GB |
| Qwen3-TTS 0.6B | ~2.5 GB |
| Piper voices | ~500 MB |
| **Recommended free space** | **60 GB** |

---

## 3. Create the dataset

Everything persistent lives under one directory. Create it first — TrueNAS will not create a
missing host path for you, and the app will fail to start if it is absent.

**Datasets → your pool → Add Dataset**

- Name: `tts-stt` (e.g. under an existing `apps` dataset)
- Everything else: defaults are fine

Note the resulting path, e.g. `/mnt/tank/apps/tts-stt`. You need it in the next step.

<details>
<summary>Shell alternative</summary>

```bash
zfs create tank/apps/tts-stt
```
</details>

### What ends up there

```text
/mnt/tank/apps/tts-stt/
  models/                  # Piper voices + your exported custom voices
  output/                  # generated audio (auto-pruned, default 24 h)
  .cache/                  # Whisper model cache
  hf-cache/                # HuggingFace downloads, one subdir per service
  qwen3-voices/            # saved voice-clone profiles
  piper-training-service/  # datasets + checkpoints (only if training is enabled)
```

> **Put this on an SSD if you can.** Model loading is read-heavy, and cold start on spinning
> rust adds tens of seconds per service.

---

## Path A — Install via YAML (recommended)

### A1. Open the installer

**Apps → Discover Apps → Custom App** (top right) **→ Install via YAML**

### A2. Paste the stack definition

Copy the entire contents of
[`docker-compose.truenas-app.yml`](../docker-compose.truenas-app.yml) and paste it in.

### A3. Set your data directory

This is **the only edit most people need**. In the pasted YAML, replace every occurrence of the
fallback path with your dataset:

```
/mnt/pool/apps/tts-stt   →   /mnt/tank/apps/tts-stt
```

A search-and-replace in the editor is fine. Alternatively leave the YAML untouched and set
`APP_DATA_DIR=/mnt/tank/apps/tts-stt` in the app's environment.

### A4. Allocate the GPU

In the same install form, find the **GPU Configuration** section and allocate your NVIDIA card
to the app.

> If you skip this, the GPU services will start but fall back to CPU — Whisper will still work,
> just far slower, and `/health` will report `"device": "cpu"` so you can tell.

### A5. Install

Click **Install**. TrueNAS pulls the images (~25 GB — expect 5–15 minutes on a decent link).

### A6. Wait for the models

**The app will look unhealthy for several minutes after the containers start. This is normal.**
Each GPU service downloads its model on first run. The health checks deliberately report *not
ready* until the model is actually loaded, so "healthy" means genuinely usable.

Watch progress:

```bash
docker logs -f ix-tts-stt-stt-service-1
```

First start typically takes 3–10 minutes. Subsequent starts take seconds — the models are cached
on your dataset.

### A7. Open the UI

```
http://<truenas-host>:3000
```

The status row at the top shows one indicator per backend. Hover any indicator to see its loaded
model, device and probe latency.

---

## Path B — Custom catalog (guided form)

The [`truenas/tts-stt/`](../truenas/tts-stt/) directory is a catalog scaffold that gives you a
point-and-click form (ports, GPU, dataset, models, optional backends) instead of raw YAML.

Add this repository as a **custom catalog** under **Apps → Discover Apps → Manage Catalogs → Add
Catalog**, then install "TTS-STT Studio" from it.

> **Caveat, stated plainly:** TrueNAS renders catalog templates with its internal `ix-lib` Jinja2
> helpers, and that contract changes between releases. This scaffold uses plain Jinja2 so it is
> readable and adaptable, but it is **not guaranteed to render unchanged on every TrueNAS
> version**. If it fails to render, use Path A — it has no templating and cannot break this way.

---

## Path C — Build from source

Only needed if you want to modify the code or cannot pull from GHCR.

```bash
cd /mnt/tank/apps
git clone https://github.com/steffenhebestreit/tts-stt.git
cd tts-stt

cp .env.truenas.example .env
nano .env                      # set APP_DATA_DIR and review the model choices

docker compose \
  -f docker-compose.yml \
  -f docker-compose.truenas.yml \
  --profile all up -d --build
```

The build takes 20–40 minutes. `docker-compose.truenas.yml` pins every GPU service to card `0`
and picks 16 GB-friendly model sizes.

> Do **not** add `docker-compose.dev.yml` on a NAS. It live-mounts source over the images and is
> for local development only.

---

## 4. Verify the install

```bash
# The gateway
curl http://localhost:3000/health

# Every backend at once, probed concurrently — the useful one
curl http://localhost:3000/api/health | jq
```

A healthy backend looks like:

```json
{
  "providers": {
    "whisper": {
      "healthy": true,
      "status_code": 200,
      "latency_ms": 3.1,
      "model_loaded": true,
      "model_size": "large-v3-turbo",
      "device": "cuda"
    }
  }
}
```

Check these three things:

1. `"healthy": true` for every backend you enabled
2. `"device": "cuda"` — if it says `cpu`, the GPU was not allocated (see A4)
3. `"model_loaded": true` — if false, the model is still downloading

### End-to-end test

1. Open `http://<truenas-host>:3000`
2. **Text-to-Speech** tab → type `Guten Tag, wie geht es Ihnen?` → Generate. Audio should play
   automatically.
3. **Live Transcription** panel → *Start Live Transcription* → allow microphone access → speak.
   Words should appear within about a second, with a `decode NNN ms` readout underneath.

> The microphone requires a **secure context**. `http://` works on `localhost` only. To use the
> microphone from another machine you need HTTPS — see
> [Reverse proxy](#7-reverse-proxy--https) below.

---

## 5. Choosing your services

Everything past the four defaults is opt-in, because VRAM is the binding constraint.

### VRAM budget

| Service | VRAM | Notes |
|---|---|---|
| `stt-service` (`large-v3-turbo`) | ~1.6 GB | Best latency/accuracy trade; cannot translate |
| `stt-service` (`large-v3`) | ~3.1 GB | Only option that supports speech translation |
| `qwen3-asr-service` | ~4 GB | Multilingual recognition |
| `qwen3-tts-service` (`0.6B`) | ~2.5 GB | Voice cloning |
| `qwen3-tts-service` (`1.7B`) | ~4.5 GB | Better quality |
| `chatterbox-tts-service` | ~4 GB | Streaming German TTS |
| `canary-asr-service` | ~2 GB | Fastest German STT |
| `parakeet-asr-service` | ~3 GB | 25 EU languages |
| `piper-training-service` | 2–4 GB | On demand only |

### Suggested configurations

**16 GB card, everyday use** *(the default)*
`frontend` + `piper-tts` + `stt` + `qwen3-asr` + `qwen3-tts (0.6B)` ≈ 8 GB

**16 GB card, tuned for realtime German**
`frontend` + `piper-tts` + `stt (turbo)` + `canary-asr` + `chatterbox-tts` ≈ 8 GB
Lowest latency in both directions: Canary for recognition, Chatterbox for streaming speech.

**8 GB card**
`frontend` + `piper-tts` + `stt (turbo)` ≈ 2 GB. Add one more service at most.

**Training**
Stop `qwen3-tts-service` first, then start `piper-training-service`. Training wants several GB
and is not something to leave running.

### Enabling an optional backend

Two steps, both required:

1. Uncomment its service block in the app YAML.
2. Set the matching `ENABLE_*` flag on `frontend-service` (`ENABLE_CHATTERBOX_TTS=true`, etc.).

Set the flag without the service and the UI shows a permanently red indicator. Add the service
without the flag and it runs but never appears in the UI.

---

## 6. Tuning for lower latency

Defaults are already tuned for realtime. These are the knobs that matter, set on `stt-service`:

| Variable | Default | Effect |
|---|---|---|
| `WHISPER_MODEL_SIZE` | `large-v3-turbo` | Dominant quality/speed lever |
| `WS_WINDOW_S` | `8.0` | How much trailing audio each interim decode re-transcribes. Whisper pads its input to a fixed 30 s before the encoder, so this does **not** change encoder cost — it bounds the decoder tokens regenerated per tick. A real saving, but a moderate one; raising it buys more context for punctuation than it costs. |
| `WS_MIN_NEW_AUDIO_S` | `0.5` | Floor on how often a partial transcript can be emitted |
| `WS_MAX_SESSIONS` | `4` | Concurrent live microphone sessions before new ones are refused |
| `WHISPER_NUM_WORKERS` | `2` | Gives live and batch traffic independent slots |

The live panel prints `decode NNN ms` under the transcript. **Tune against that number, not
against feel.** If decode time approaches `WS_MIN_NEW_AUDIO_S × 1000`, move to a smaller model
first — that changes encoder cost, which the window length does not. Lowering `WS_WINDOW_S`
helps second, and costs you punctuation context.

On the GPU services generally:

| Variable | Default | Effect |
|---|---|---|
| `ASR_MAX_CONCURRENCY` / `TTS_MAX_CONCURRENCY` | `1` | Bounds concurrent inference. Raise only with VRAM to spare — overlapping requests on one model make each one slower. |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` | Reduces allocator fragmentation when several models share a card |

---

## 7. Reverse proxy / HTTPS

**You need this if you want the microphone to work from any machine other than the NAS itself.**
Browsers only grant microphone access in a secure context.

Because everything is proxied through port 3000, the config is a single upstream. Nginx:

```nginx
server {
    listen 443 ssl;
    server_name voice.example.com;

    ssl_certificate     /path/to/fullchain.pem;
    ssl_certificate_key /path/to/privkey.pem;

    location / {
        proxy_pass http://truenas.lan:3000;
        proxy_http_version 1.1;

        # Required — live transcription is a WebSocket
        proxy_set_header Upgrade    $http_upgrade;
        proxy_set_header Connection "upgrade";

        proxy_set_header Host              $host;
        proxy_set_header X-Real-IP         $remote_addr;
        proxy_set_header X-Forwarded-For   $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Streaming TTS starts playing before synthesis finishes; buffering here
        # would throw that away and restore full-text latency.
        proxy_buffering off;

        # Long-running jobs (training, large transcriptions)
        proxy_read_timeout 600s;
    }
}
```

Then set on `frontend-service`:

```env
ALLOWED_ORIGINS=https://voice.example.com
```

Traefik users: enable the WebSocket-capable HTTP router (the default) and disable response
buffering for this service.

---

## 8. Maintenance

### Update

**Apps → Installed → tts-stt → Edit**, then re-deploy. To pin a version, set `IMAGE_TAG` to a
release tag instead of `latest`.

Shell (Path C):

```bash
cd /mnt/tank/apps/tts-stt
git pull
docker compose -f docker-compose.yml -f docker-compose.truenas.yml --profile all up -d --build
```

### Back up

Snapshot the dataset. **Datasets → tts-stt → Snapshots → Add**, or set a periodic task.

Worth keeping: `models/` (your trained voices) and `qwen3-voices/` (saved clone profiles).
`hf-cache/` and `.cache/` are re-downloadable — exclude them if snapshot size matters.

### Free disk space

```bash
docker image prune -a     # old images after an update
```

Generated audio in `output/` is pruned automatically (`OUTPUT_RETENTION_HOURS`, default 24).

---

## 9. Troubleshooting

### The app installs but never goes healthy

Almost always a model still downloading. Health checks intentionally report not-ready until the
model is loaded.

```bash
docker logs -f ix-tts-stt-stt-service-1
```

Look for `Model loaded successfully`. If you instead see a HuggingFace network error, the NAS
cannot reach `huggingface.co` — check DNS and any egress firewall.

### `"device": "cpu"` when you have a GPU

The GPU was not allocated to the app. Edit the app, allocate the NVIDIA card, redeploy. Confirm
the host sees it with `nvidia-smi` first.

### Everything worked, then a service started failing after adding another

VRAM exhaustion. Check with `nvidia-smi` while the stack runs. Use a smaller Whisper model, drop
to `Qwen3-TTS-12Hz-0.6B-Base`, or disable a backend. See the [VRAM budget](#vram-budget).

### The microphone button does nothing

1. Are you on `https://` or `http://localhost`? Browsers refuse microphone access otherwise.
2. Browser console: a WebSocket error to `/ws/stt` means the reverse proxy is not forwarding
   `Upgrade`/`Connection` headers (see [section 7](#7-reverse-proxy--https)).
3. `curl http://<host>:3000/api/health` — is the `whisper` provider healthy?

### Live transcription lags behind my speech

Read the `decode NNN ms` readout under the transcript:

- **Under ~500 ms** — working as intended.
- **Approaching 1000 ms** — try a smaller `WHISPER_MODEL_SIZE` first; then `WS_WINDOW_S` `5.0`.
- **Over 1500 ms** — you are likely on CPU (check `/api/health` for `"device"`), or the GPU is
  contended. Reduce concurrent services or use a smaller model.

The system deliberately **skips** audio rather than queueing it when decoding falls behind, so
lag stays bounded instead of growing through a session. Dropped blocks are reported when you
stop.

### `permission denied` on the dataset

The containers run as root and need write access to `APP_DATA_DIR`. Confirm the path exists and
that no restrictive ACL is applied. Simplest fix from the shell:

```bash
chmod -R u+rwX /mnt/tank/apps/tts-stt
```

### A backend shows red but its container is running

`curl http://localhost:3000/api/health | jq` and read the `error` field. `ConnectError` means the
frontend cannot resolve the service — usually the service name in `*_SERVICE_URL` does not match
the service key in the YAML.

---

## Upgrading from an earlier install

This release changed how the browser reaches the backends. If you installed a previous version:

1. **You can stop publishing backend ports.** Ports 5000–5007 and 8080 no longer need to be
   reachable from the browser. Remove them from your app config to reduce exposure.
2. **`BROWSER_*_URL` variables are obsolete.** They are ignored by the web UI. Remove them.
   (They still apply if you deliberately publish backend ports for your own scripts.)
3. **`ALLOWED_ORIGINS` can be tightened.** With everything same-origin, set it to just your
   frontend origin instead of `*`.
4. **The default Whisper model changed** to `large-v3-turbo`. It is faster and roughly as
   accurate — but it **cannot do speech translation**. If you use `task=translate`, set
   `WHISPER_MODEL_SIZE=large-v3` explicitly. Requests for translation on a turbo model now return
   a clear HTTP 400 instead of silently returning untranslated text.
5. **Health checks now report 503 until the model loads.** A service that shows unhealthy during
   startup is correct behaviour, not a regression.

---

## Related documentation

- [`truenas/README.md`](../truenas/README.md) — catalog vs YAML install, image registry
- [`docs/truenas-service-profiles.md`](./truenas-service-profiles.md) — which services to run when
- [`docs/truenas-custom-app-checklist.md`](./truenas-custom-app-checklist.md) — first-run checklist
- [`docs/truenas-deployment.md`](./truenas-deployment.md) — build-from-source detail, storage layout
- [`docs/provider-contracts.md`](./provider-contracts.md) — the API contracts each backend implements
