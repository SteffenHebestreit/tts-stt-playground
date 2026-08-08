# TrueNAS SCALE — installing TTS-STT as an app

> **Full step-by-step walkthrough:
> [`docs/truenas-installation-guide.md`](../docs/truenas-installation-guide.md)** — prerequisites,
> dataset setup, GPU allocation, verification, VRAM budgets and troubleshooting. Start there.
> This page is the short version.

There are two ways to run this stack on TrueNAS SCALE (Electric Eel / Fangtooth
and newer, which use Docker for apps). **Both use the prebuilt images published
to GHCR**, so no repository or build step is required on the NAS.

Only the web UI port (3000) needs to be published: the frontend proxies every
backend call, including live microphone transcription over WebSocket.

## Option A — Install via YAML (recommended, fully supported)

This is the simplest and most robust path.

1. Create a dataset for data, e.g. `/mnt/tank/apps/tts-stt`.
2. In the UI: **Apps → Discover Apps → Custom App → Install via YAML**.
3. Paste [`docker-compose.truenas-app.yml`](../docker-compose.truenas-app.yml).
4. Edit `APP_DATA_DIR` (top of the file) to your dataset path.
5. Allocate the NVIDIA GPU to the app, then deploy.
6. Open `http://<truenas-host>:3000`. Nothing else to configure — backend ports
   do not need publishing and `BROWSER_*_URL` is not used by the web UI.

Expect the app to report unhealthy for a few minutes on first start while each
service downloads its model; health checks stay red until the model is loaded.

Reconfigure later by editing the app's YAML/environment in the Apps UI. Optional
backends (training, Parakeet, whisper.cpp) are included commented-out at the
bottom of that file.

## Option B — Custom catalog (point-and-click form)

The [`tts-stt/`](./tts-stt/) directory is a **catalog scaffold** providing an
`app.yaml`, a `questions.yaml` configuration form, an `app-readme.md`, and a
Jinja2 `templates/docker-compose.yaml`. Add this repository as a **custom
catalog** (community train) in TrueNAS to get a guided install form (ports, GPU,
dataset, models, optional backends).

> **Caveat:** TrueNAS renders catalog templates with its internal `ix-lib`
> Jinja2 helpers, and the exact contract changes between releases. This scaffold
> uses plain Jinja2 so it is readable and adaptable, but it is **not guaranteed
> to render unchanged on every TrueNAS version** — validate it against your
> target release. If in doubt, use **Option A**, which has no templating.

## Images

Published by CI (`.github/workflows/publish-images.yml`) to:

```
ghcr.io/steffenhebestreit/tts-stt-<service>:<tag>
```

Override the registry/tag with `IMAGE_REGISTRY` / `IMAGE_TAG` (or the
`image_registry` / `image_tag` form fields) to use a fork.

## More

- **Step-by-step installation: [`docs/truenas-installation-guide.md`](../docs/truenas-installation-guide.md)**
- Build from source, storage layout, VRAM planning: [`docs/truenas-deployment.md`](../docs/truenas-deployment.md)
- First-run checklist: [`docs/truenas-custom-app-checklist.md`](../docs/truenas-custom-app-checklist.md)
- Which services to run when: [`docs/truenas-service-profiles.md`](../docs/truenas-service-profiles.md)
