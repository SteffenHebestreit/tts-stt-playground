# TrueNAS SCALE — installing TTS-STT as an app

There are two ways to run this stack on TrueNAS SCALE (Electric Eel / Fangtooth
and newer, which use Docker for apps). **Both use the prebuilt images published
to GHCR**, so no repository or build step is required on the NAS.

## Option A — Install via YAML (recommended, fully supported)

This is the simplest and most robust path.

1. Create a dataset for data, e.g. `/mnt/tank/apps/tts-stt`.
2. In the UI: **Apps → Discover Apps → Custom App → Install via YAML**.
3. Paste [`docker-compose.truenas-app.yml`](../docker-compose.truenas-app.yml).
4. Edit `APP_DATA_DIR` (top of the file) to your dataset path.
5. Allocate the NVIDIA GPU to the app, then deploy.
6. Open `http://<truenas-host>:3000`. If you reach it from another machine,
   set the `BROWSER_*_URL` values to the TrueNAS host.

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

- Full guide, storage layout, VRAM planning: [`docs/truenas-deployment.md`](../docs/truenas-deployment.md)
- First-run checklist: [`docs/truenas-custom-app-checklist.md`](../docs/truenas-custom-app-checklist.md)
- Which services to run when: [`docs/truenas-service-profiles.md`](../docs/truenas-service-profiles.md)
