# TrueNAS SCALE: TTS-STT as an app

> **Everything step by step, with update, rollback and troubleshooting:
> [`docs/truenas-installation-guide.md`](../docs/truenas-installation-guide.md).** This page is the
> map of what lives in this repository for TrueNAS.

All routes use the prebuilt images published to GHCR (`ghcr.io/steffenhebestreit/tts-stt-<service>`)
except building from source. Only the web UI port (3000) is published: the frontend proxies every
backend call, live microphone transcription included.

## Which route

| Route | Effort | Status |
|---|---|---|
| **A. Install via YAML**: paste [`docker-compose.truenas-app.yml`](../docker-compose.truenas-app.yml), replace one path | 5 minutes | **Supported.** The recommended route |
| **A2. Settings file**: the same compose file plus a `settings.env` on the dataset, wrapped by [`custom-app-include.yml`](./custom-app-include.yml) | 10 minutes, then updates are one line | **Not needed on TrueNAS**: most settings live in the web UI's Settings page (below). Kept for running the file with `docker compose` from a shell; standard Compose (`include` with `env_file`), not yet run on a TrueNAS |
| **B. Custom catalog**: a form instead of YAML, [`tts-stt/`](./tts-stt/) | - | **Scaffold, not loadable as it stands** (see below) |
| **C. Build from source** on the NAS | 30+ minutes | Supported: [`docs/truenas-deployment.md`](../docs/truenas-deployment.md) |

## Settings in the web UI

TrueNAS gives a YAML app no settings form, so the app has its own: the gear in the web UI, or
`http://<truenas-ip>:3000/settings` (a new install also gets a **Settings** button in the Apps
screen). Host names, the reverse-proxy URL, API access and named API keys, which optional engines
are offered, the default engines and the gateway limits change there at once, without a restart.
They are stored in the dataset (`settings/`, mounted into the gateway only), so Edit, Update,
Stop/Start and a re-paste keep them; a value saved there wins over the YAML, setting by setting.
What needs a container to be recreated stays in the YAML: the release, the GPU, the port, the
dataset and the `profiles:` lines that install optional services. Changes need an admin key: the
YAML's `API_KEY`, or, on an install without one, a key created with a one-time code that only the
gateway's log shows. Details and recovery: the guide's
[Settings in the web UI](../docs/truenas-installation-guide.md#settings-in-the-web-ui).

## Setup and updates

```text
scripts/truenas/preflight.sh     before installing: Docker, driver, GPU, VRAM budget, disk, port
scripts/truenas/pull-images.sh   pre-pull a release (TrueNAS aborts an Apps job after 20 minutes)
scripts/truenas/update-check.sh  read-only: is a newer release, or a newer image for your tag, out?
scripts/truenas/set_version.py   maintainers: move every place that names the release
scripts/truenas/render_catalog.py  maintainers: render the catalog template from the form's defaults
```

The compose file **pins a release** (`IMAGE_TAG`), so nothing changes until you change it. To follow
the newest build instead, set `IMAGE_TAG:-latest` and `PULL_POLICY:-always`: then every Stop/Start
or Save pulls. Update, rollback (ZFS snapshot plus the previous tag) and data compatibility are
in the guide. Optional release notifications: the `diun` service in the compose file (notify-only,
off by default).

## Route B: the catalog scaffold

[`tts-stt/`](./tts-stt/) holds `app.yaml`, `questions.yaml`, `app-readme.md` and a
`templates/docker-compose.yaml` in plain Jinja2. It is kept equivalent to the compose file by
`tests/test_truenas_catalog.py`: the template renders on its own, passes `docker compose config`,
and at the default answers produces the same services, images, environment, volumes and health
checks. What it is **not**:

- TrueNAS renders official catalog apps with its ix-lib helpers (`ix_lib.base.render.Render` in
  `truenas/apps`) and reads a catalog in the `trains/<train>/<app>/<version>/` layout. This
  directory has neither. Whether a hand-written plain template is accepted was not checked.
- The `ix_volume` storage answer assumes how TrueNAS hands the dataset path to the template.

Treat it as a starting point for a real catalog app, not as an install route.

## The release version

One place decides it: `app_version` in [`tts-stt/app.yaml`](./tts-stt/app.yaml). Everything that
defaults to it (both TrueNAS compose files, both `.env.truenas*.example`, the catalog form, the
commands in the docs) is checked by `tests/test_truenas_release_consistency.py`.

To release, in this order:

1. `scripts/truenas/set_version.py X.Y.Z` and review the diff.
2. `pytest tests/test_truenas_*.py`, commit.
3. Tag `vX.Y.Z` and push the tag. `publish-images.yml` runs the test suite first (which fails if the
   tag differs from `app_version`), then publishes `ghcr.io/steffenhebestreit/tts-stt-*:X.Y.Z` and
   `:X.Y` for every image, moves `:latest`, and adds `:X.Y.Z-nemo2` rollback images for canary and
   parakeet. Each image is published on its own: one failed CUDA build does not hold the rest back,
   and you re-run the failed jobs.
4. Only when all the images exist, tell people to install: `scripts/truenas/update-check.sh --tag X.Y.Z`
   says whether they do. Until then a pinned install fails on `manifest unknown`.

## CI

`.github/workflows/truenas.yml` runs on changes to `truenas/`, the TrueNAS compose files,
`.env.truenas*.example`, `scripts/truenas/`, `docs/truenas-*` and the TrueNAS tests: shellcheck on
the scripts, `docker compose config` on every TrueNAS file with the example settings (including a
pasted file that still has the placeholder path, which must be refused), the catalog render for
several answer sets, and `tests/test_truenas_*.py` with a guard that none of them was skipped. The
same tests run in `ci.yml` on every push.

What none of it can show is TrueNAS itself: the guide's last section lists what to check on a
real system.

## More

- [`docs/truenas-installation-guide.md`](../docs/truenas-installation-guide.md): install, update, roll back, VRAM, troubleshooting
- [`docs/truenas-deployment.md`](../docs/truenas-deployment.md): build from source, storage layout, GPU driver
- [`docs/truenas-service-profiles.md`](../docs/truenas-service-profiles.md): which services to run when
- [`docs/truenas-custom-app-checklist.md`](../docs/truenas-custom-app-checklist.md): first-run checklist
