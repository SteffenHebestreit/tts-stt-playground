# TrueNAS SCALE: install, update, roll back

The one guide for running TTS-STT on TrueNAS SCALE. The other TrueNAS pages in this folder are
short and point back here.

- **Install** in about 5 minutes: [section 2](#2-install-in-5-minutes)
- **Change settings** in the web UI, no YAML: [Settings in the web UI](#settings-in-the-web-ui)
- **Update** in about 2 minutes: [section 4](#4-update-in-2-minutes)
- **Roll back**: [section 5](#5-roll-back)
- **VRAM planning**: [section 6](#6-vram-and-disk-planning)
- Something is wrong: [section 7](#7-troubleshooting)

Commands in a `code block` run in the TrueNAS shell (System -> Shell, or SSH) as root.

---

## 1. What you are installing

| Service | Purpose | GPU | Runs by default |
|---|---|---|---|
| `frontend-service` | Web UI and the gateway every browser request goes through | no | yes |
| `piper-tts-service` | Fast CPU text-to-speech, German voices included | no | yes |
| `stt-service` | Whisper speech-to-text and live microphone transcription | yes | yes |
| `qwen3-asr-service` | Multilingual speech recognition | yes | yes |
| `qwen3-tts-service` | Voice cloning and high-quality TTS | yes | yes |
| `chatterbox-tts-service` | Streaming German TTS: speech starts before the text is finished | yes | opt-in |
| `magpie-tts-service` | NVIDIA TTS with five built-in voices; reads numbers and dates correctly | yes | opt-in |
| `canary-asr-service` | Fastest German speech recognition (180M parameters) | yes | opt-in |
| `parakeet-asr-service` | Speech recognition for 25 European languages | yes | opt-in |
| `piper-training-service` | Train your own voice from recordings | yes | opt-in |
| `whisper-cpp` | CPU-only speech recognition, no GPU needed | no | opt-in |
| `diun` | Notifies you about new releases; never updates anything | no | opt-in |

**Only port 3000 is published.** The browser talks to the frontend, which proxies every backend
call, the live-microphone WebSocket (`/ws/stt`) included, over the internal network. You do not
publish backend ports and you do not set any `BROWSER_*_URL` variable. It also works behind an
HTTPS reverse proxy ([section 8](#8-reverse-proxy-and-https)).

**Nothing updates by itself.** The compose file pins a release (`IMAGE_TAG`), so a fresh image
reaches you only when you change that value.

---

## 2. Install in 5 minutes

### Before you start

- TrueNAS SCALE **24.10 (Electric Eel) or newer**: these releases run Apps on Docker.
- For the GPU services: an NVIDIA driver installed from **Apps -> Settings -> Install NVIDIA
  Drivers**, and the GPU **not** isolated for a VM. The images are built on CUDA 12.8: use a
  driver with CUDA 12.8 support (R570 or newer). A GeForce card (RTX 30, 40, 50 series) needs
  exactly that; only data-centre and workstation cards also start on the older branches 470 to 565.
- About 60 GB free: 25 GB of images, the models ([section 6](#6-vram-and-disk-planning)) and room
  for generated audio.

### Step 1: create the dataset (1 minute)

**Datasets -> your pool -> Add Dataset**, name `tts-stt`, defaults are fine. Note the path, for
example `/mnt/tank/apps/tts-stt`. Everything persistent goes there:

```text
/mnt/tank/apps/tts-stt/
  models/                  Piper voices (image voices are copied in) and your trained voices
  output/                  generated audio, pruned after 24 h
  cache/                   ONE model cache for all services: a model downloads once
  qwen3-voices/            saved voice-clone profiles
  settings/                what the web UI's Settings page saves (host names, engines, API keys)
  backend-settings/        backend settings saved in the web UI (kept for later releases)
  piper-training-service/  datasets and checkpoints (training only)
  whisper-cpp-models/      whisper.cpp models (whisper-cpp only)
```

Use an SSD-backed pool if you can: model loading is read-heavy.

### Step 2: check the host (1 minute, recommended)

Get the helper scripts once (they are read-only unless noted):

```bash
mkdir -p /mnt/tank/apps/tts-stt/scripts && cd /mnt/tank/apps/tts-stt/scripts
for f in lib.sh preflight.sh update-check.sh pull-images.sh; do
  curl -fsSLO "https://raw.githubusercontent.com/SteffenHebestreit/tts-stt-playground/v0.2.0/scripts/truenas/$f"
done
chmod +x ./*.sh
./preflight.sh --data-dir /mnt/tank/apps/tts-stt --create
```

`preflight.sh` checks Docker and Compose, the driver against the CUDA 12.8 requirement, the GPU
and its VRAM against the services you plan to run, free disk on the dataset and the Docker root,
and that port 3000 is free. `--create` makes the subdirectories, `--with canary-asr,training`
plans for optional services, `--with-docker` proves a container can use the GPU. It ends with
`READY` or lists what to fix.

### Step 3: pre-pull the images (optional)

```bash
./pull-images.sh --tag 0.2.0
```

Takes 5 to 15 minutes on a decent link. TrueNAS stops an Apps job after **20 minutes**. The default images are about 25 GB and need
roughly 21 MB/s to finish in that time. If you skip this step and the install job times out,
press Install (or Save) again: layers already downloaded are kept.

### Step 4: paste the YAML (1 minute)

1. **Apps -> Discover Apps -> Custom App** (three-dots menu) **-> Install via YAML**.
2. Name it `tts-stt`. Paste the complete
   [`docker-compose.truenas-app.yml`](../docker-compose.truenas-app.yml).
3. Search-and-replace `${APP_DATA_DIR:?set dataset path}` (every occurrence) with your dataset path,
   for example `/mnt/tank/apps/tts-stt`. **This is the only edit most installs need.**
4. **Install.**

The path has no default on purpose: if you forget it, the editor rejects the YAML with
`required variable APP_DATA_DIR is missing a value` instead of writing to a wrong place.

No search-and-replace in the editor (not checked whether it has one)? Make the edit in the shell,
print the result and paste that instead:

```bash
curl -fsSL https://raw.githubusercontent.com/SteffenHebestreit/tts-stt-playground/v0.2.0/docker-compose.truenas-app.yml \
  | sed 's|${APP_DATA_DIR:?set dataset path}|/mnt/tank/apps/tts-stt|g' > /mnt/tank/apps/tts-stt/app.yml
cat /mnt/tank/apps/tts-stt/app.yml
```

Most settings are changed later in the web UI ([Settings in the web UI](#settings-in-the-web-ui)).
The ones that need a container to be recreated sit in the same file as `${NAME:-default}`; you
change the default in place:

| Setting | Text to search for | Meaning |
|---|---|---|
| `IMAGE_TAG` | `IMAGE_TAG:-0.2.0` | The release to run. Change it to [update or roll back](#4-update-in-2-minutes). |
| `PULL_POLICY` | `PULL_POLICY:-missing` | `missing` pulls only absent images (pinned release). `always` pulls on every Start/Save. |
| `GPU_DEVICE_ID` | `GPU_DEVICE_ID:-0` | Index or `GPU-...` UUID from `nvidia-smi -L`. |
| `FRONTEND_PORT` | `FRONTEND_PORT:-3000` | Web UI port. Also change both `port:` lines under `x-portals`. |

There is **no separate GPU step**: the file itself asks Docker for the GPU
(`deploy.resources.reservations.devices`) on every GPU service.

### Step 5: open it (30 seconds)

`http://<truenas-ip>:3000`. The app reaches **Running** within about a minute. Models download in
the background on the first start (several GB): the status row in the UI shows each backend, and

```bash
docker logs -f ix-tts-stt-stt-service-1
```

shows the download. Check everything at once:

```bash
curl -s http://localhost:3000/api/health | python3 -m json.tool
```

Every enabled backend should report `"healthy": true`, and the GPU services `"device": "cuda"`
(`cpu` means the GPU did not reach the container). `"model_loaded": false` on a fresh install is
the download still running.

**End-to-end test:** Text-to-Speech tab, type `Guten Tag, wie geht es Ihnen?`, Generate. Then
Live Transcription, start, allow the microphone, speak: words appear within about a second. The
microphone needs a secure context: `http://` works on `localhost` only, so from another machine
use [HTTPS](#8-reverse-proxy-and-https).

### Settings in the web UI

Most settings are changed in the app itself, not in the YAML: the gear next to **API
Documentation** in the web UI, or `http://<truenas-ip>:3000/settings` (an app installed from this
file also gets a **Settings** button next to **Web UI** in the Apps screen). A change applies from
the next request on, in every gateway worker, with no restart, and is kept in the dataset's
`settings/` folder, so it survives Edit, Update, Stop/Start and a re-paste of a newer YAML.

| Part of the page | What it changes |
|---|---|
| Access | The host names the UI may be reached by (`TRUSTED_HOSTS`), the public URL behind a proxy that rewrites `Host` (`TRUSTED_ORIGINS`), whether to believe `X-Forwarded-Host` (`TRUST_PROXY_HEADERS`), other web pages that may call the API (`ALLOWED_ORIGINS`) |
| API access & keys | Open on the network or a key required; what the YAML's `API_KEY` may do (admin or client); named API keys, created and revoked one by one |
| Engines | Which optional engines are offered (Canary, Parakeet, Chatterbox, Magpie, whisper.cpp, voice training), each with whether its container answers; the default text-to-speech and speech-to-text engine |
| Limits | `MAX_UPLOAD_MB`, `MAX_TTS_CHARS`, `MAX_CONCURRENT_UPLOADS`, `MAX_CONCURRENT_FFMPEG` |
| History & recovery | The last 20 versions with Restore, **Back to the app YAML** (drops every value saved here), and **Print a recovery code** for a lost admin key |

**Who may change them.** Every change needs an admin key.

- `API_KEY` set in the YAML: enter it when the page asks. It is an admin key unless you make it a
  client key on the page (which needs an admin key made there first).
- No `API_KEY`: the page shows the values read-only. Press **Print a one-time code**, read the code
  in the gateway's log (**Apps -> Installed -> tts-stt -> Workloads -> frontend-service -> View
  Logs**, or `docker logs ix-tts-stt-frontend-service-1`), and enter it with a name for your key.
  The browser creates the key and shows it once: store it in a password manager. The same step can
  switch the API to "a key is required". A code is valid for 30 minutes, works once and appears
  nowhere but the log.
- Keys made on the page are stored only as SHA-256 digests. Give scripts and Home Assistant their
  own **client** keys: those use the API but cannot change settings.

**Example: a new host name.** `http://speach.k2o:3000` answers `403 host_not_allowed`. Open
`http://<truenas-ip>:3000/settings` (an IP address is always accepted), add `speach.k2o` under
**Access**, Save. The next request by that name works; nothing restarts.

**Safety nets.** A change that would refuse the address you are saving from is refused
(`would_lock_out`) and says why. A new proxy URL or a change of `X-Forwarded-Host` is confirmed by
the page through the new rules, and undone after 60 seconds if that confirmation never arrives.
Every save is a version: Restore brings an earlier one back. A hand edit of `settings/gateway.json`
with an invalid value is ignored and shown in a banner.

**What stays in the YAML:** the release (`IMAGE_TAG`, `PULL_POLICY`), the GPU, the port, the
dataset, which optional containers are installed (the `profiles:` lines), `API_KEY`, the switches
that would open the gateway completely (`ALLOWED_HOSTS:-*`, `ALLOWED_ORIGINS:-*`), and the two
switches of the page itself: `ENABLE_SETTINGS_UI` (`false` makes the page read-only and its saved
values ignored; API keys made there stay in force) and `SETTINGS_LOCKED_KEYS` (settings that always
come from the YAML, comma separated, for example `TRUSTED_HOSTS,MAX_UPLOAD_MB`). Where the YAML and
the page both set something, the page wins ([section 3](#3-what-truenas-does-with-this-app)); every
field shows where its value comes from and has **Reset to app YAML**.

**An install pasted before the page existed** has no `settings/` mount. The page then shows the
values read-only, says why, and shows the two lines to add under `frontend-service:` ->
`volumes:`. Pasting the current file with your edits carried over adds them, together with the
`backend-settings` line of every backend and the three lines `ENABLE_SETTINGS_UI`,
`SETTINGS_LOCKED_KEYS` and `ENABLE_TRAINING`. Snapshot the dataset first.

### Optional services

Two steps per service:

1. **Install it:** delete the service's `profiles:` line in the YAML (**Apps -> Installed -> tts-stt
   -> Edit**) and Save. The image is pulled and the container starts.
2. **Offer it:** **Settings -> Engines**, tick its **Offer** box, Save. The row says whether the
   container answers, or that it is not installed, with the line to delete. Or set its flag on
   `frontend-service` in the YAML instead, for example `ENABLE_CANARY_ASR:-false` to
   `ENABLE_CANARY_ASR:-true`.

An engine offered without its container is a permanently red indicator; a container that is not
offered runs but never shows in the UI.

| Service | Delete the `profiles` line of | Offer it in Settings, or the flag |
|---|---|---|
| Canary (fast German STT) | `canary-asr-service` | `ENABLE_CANARY_ASR` |
| Parakeet (25 languages) | `parakeet-asr-service` | `ENABLE_PARAKEET_ASR` |
| Chatterbox (streaming TTS) | `chatterbox-tts-service` | `ENABLE_CHATTERBOX_TTS` |
| Magpie (NVIDIA TTS, 5 voices) | `magpie-tts-service` | `ENABLE_MAGPIE_TTS` |
| whisper.cpp (CPU STT) | `whisper-cpp` | `ENABLE_WHISPER_CPP` |
| Voice training | `piper-training-service` | `ENABLE_TRAINING` (off in this file, which hides the Voice Training tab) |

Check the VRAM first: `./preflight.sh --data-dir /mnt/tank/apps/tts-stt --with canary-asr`.

`whisper-cpp` downloads its model (about 0.6 GB) before its server starts, so with it enabled the
app shows **Deploying** until that finishes. Every other service answers `/health` at once and
downloads in the background.

**No GPU?** Delete `stt-service`, `qwen3-asr-service` and `qwen3-tts-service`, install and offer
`whisper-cpp` as above, and make it the default speech-to-text engine in **Settings -> Engines**
(or set `DEFAULT_STT_PROVIDER:-whisper` to `whisper-cpp` in the YAML).

### Other ways to install

- **Settings file** (`include:` with `env_file:`). Not needed on TrueNAS: most settings live in the
  web UI's Settings page now, and what is left in the YAML changes rarely. It stays for people who
  run the stack with `docker compose` from a shell. TrueNAS keeps only the parsed YAML, so comments
  and your edit history are gone after Save, and a change of release means editing the tag on every
  `image:` line. With a settings file the pasted YAML is fixed and the release is one line:

  ```bash
  cd /mnt/tank/apps/tts-stt && mkdir -p app && cd app
  BASE=https://raw.githubusercontent.com/SteffenHebestreit/tts-stt-playground/v0.2.0
  curl -fsSLO "$BASE/docker-compose.truenas-app.yml"
  curl -fsSL "$BASE/.env.truenas.example" -o settings.env
  nano settings.env     # APP_DATA_DIR, IMAGE_TAG, ...
  ```

  Then paste [`truenas/custom-app-include.yml`](../truenas/custom-app-include.yml) into Install via
  YAML after changing its two paths. Optional services still need their `profiles:` line deleted in
  the downloaded compose file. **Not yet run on a real TrueNAS** (see
  [section 9](#9-what-has-and-has-not-been-verified)).
- **Custom catalog** (a form instead of YAML): a scaffold in [`truenas/tts-stt/`](../truenas/tts-stt/),
  not loadable by TrueNAS as it stands. See [`truenas/README.md`](../truenas/README.md).
- **Build from source on the NAS:** [`truenas-deployment.md`](./truenas-deployment.md).

---

## 3. What TrueNAS does with this app

Read from the TrueNAS middleware source (`truenas/middleware`, plugins/apps), because it decides
how updates behave:

- **Install, Edit and Save** run `docker compose up -d --force-recreate --remove-orphans`.
  Compose pulls an image when it is missing locally (`pull_policy: missing`) or on every run
  (`always`). It does **not** pass `--pull=always` for you.
- **Stop** runs `docker compose down` (containers removed, images and data kept). **Start** runs
  `up --force-recreate`. Both take the `stop_grace_period` of each service into account.
- The Apps UI stores the YAML as **parsed data**: comments and anchors are dropped. What has to
  survive lives in `x-notes` and this guide.
- The app state is **Deploying** while any container with a health check is not healthy yet, and
  **Running** once all are. Here `/health` is a liveness check that answers while a model is
  still loading, so the state settles within about a minute.
- Compose calls are killed after **20 minutes** (see [Step 3](#step-3-pre-pull-the-images-optional)).
- TrueNAS checks the registries for newer images of a running app itself and can pull and
  redeploy them. That covers floating tags; a *pinned* tag never changes, which is why
  [`update-check.sh`](#check-for-a-new-release) exists.
- Every container runs as **root** (no image sets `USER`). Root can write anywhere in the
  dataset: no `chown` or ACL entry is needed. If you also share the dataset over SMB, set the
  share's ACL for your users on the folders you want to edit.
- **What the Settings page saves survives all of the above.** It lives in the dataset's
  `settings/` folder, a bind mount of `frontend-service` only, and Install, Edit/Save, Update and
  Stop/Start only recreate containers. A re-paste of a newer YAML, a delete and reinstall on the
  same dataset, and a ZFS snapshot rollback (data and settings together) keep it too. The gateway
  refuses to save unless `settings/` really is a mount, so a save cannot land inside a container and
  vanish with it. The folder holds `gateway.json` (only the values changed in the page),
  `keys.json` (API keys as SHA-256 digests, never the keys), `history/` (the last 20 versions) and
  `audit.jsonl` (who changed what, also written to the gateway's log). `backend-settings/` is
  mounted into every Python backend read-only and stays empty until a release uses it.
- **Which value wins**, setting by setting, from highest to lowest: a name in
  `SETTINGS_LOCKED_KEYS` (the YAML value, and the field is read-only); a valid value saved in the
  page; the YAML's `${NAME:-default}`; the built-in default. A setting never touched in the page
  keeps following the YAML and the defaults of new releases. `ENABLE_SETTINGS_UI:-false` in the
  YAML, or a file named `SAFE-MODE` in `settings/`, makes the gateway ignore what the page saved;
  API keys made in the page stay in force in every case, so recovery never opens the API. The
  gateway's log names the settings that come from the page when it starts.

---

## 4. Update in 2 minutes

### Check for a new release

```bash
./update-check.sh
```

Read-only. It compares the images your containers run with the registry and the newest published
release, and prints `UPDATE AVAILABLE` (exit status 10) or `Up to date`. Use `--quiet` in a cron
job. `./update-check.sh --tag <new-release>` tells you whether a release's images exist yet.

Prefer a push notification? Enable the optional `diun` service ([section 10](#10-optional-update-notifications-diun)).

### Apply it

1. **Read the release notes** on GitHub (Releases). Note anything under "breaking" or "data".
2. **Snapshot the dataset:** `zfs snapshot tank/apps/tts-stt@pre-update` (or Datasets -> your
   dataset -> Data Protection -> Snapshots -> Create). `preflight.sh` prints your dataset name.
3. **Pre-pull (optional):** `./pull-images.sh --tag <new-release>` (add `--with canary-asr,...` for
   the optional services you run).
4. **Change the release:** **Apps -> Installed -> tts-stt -> Edit**. Replace the old release in
   the YAML with the new one, everywhere (search for `IMAGE_TAG:-`; the version follows it). With a settings file:
   edit `IMAGE_TAG=` in `settings.env`, then Stop and Start the app.
5. **Save.** TrueNAS recreates the containers; images that are missing are pulled first.
6. **Check:** `curl -s http://localhost:3000/api/health` and open the UI.

Your models, voices, output and what the Settings page saved stay in the dataset; nothing is
downloaded again. Once you are sure of the new version, free the old images:
`docker image prune -a --filter "until=720h"` removes
unused images older than 30 days (keep the previous release's images until then, they make a
rollback instant).

### Follow the newest build instead (`latest`)

Set `IMAGE_TAG:-latest` **and** `PULL_POLICY:-always`. Then **every Stop/Start or Save pulls**, so
an update is: Stop, Start. The Apps UI also flags images that changed on the registry. This
trades reproducibility for convenience: a `latest` build can change behaviour. `:latest` moves
when a release `vX.Y.Z` is tagged and on every push to `master`; a prerelease tag and a manual
run never move it (the rules are in `scripts/plan_publish.py`, run by
`.github/workflows/publish-images.yml`, which runs the test suite first).

### From the shell

Both read from the middleware source, not run by the author:
`midclt call -job app.redeploy tts-stt` re-applies the current config (a Start does the same), and
`midclt call -job app.pull_images tts-stt '{"redeploy": true}'` pulls and then redeploys.

### Data compatibility between releases

The data folders are only ever added to: new releases add files (a new `vocab.json` in a training
checkpoint, more fields in a saved voice's metadata, new voice files copied into `models/default`)
and read what older releases wrote. A rollback to the previous release therefore keeps working
with the same dataset. What can go wrong is a release that *changes the meaning* of something it
already stored; the release notes call that out, and the snapshot in step 2 covers it.

---

## 5. Roll back

**Usual case: only the software is the problem.**

1. **Apps -> Installed -> tts-stt -> Edit**, put the previous release back in `IMAGE_TAG:-`,
   Save. Its images are still on the host if you did not prune them, so this takes seconds. If
   they are gone, run `./pull-images.sh --tag <previous-release>` first.
2. Check `curl -s http://localhost:3000/api/health`.

**Back to a release from before the Settings page:** it ignores `settings/` and
`backend-settings/` and runs on the YAML values alone, until you update again (then the saved
values apply once more). **Keep `API_KEY` in the YAML** if the API must need a key: a key made in
the page, and "a key is required" switched on there, mean nothing to an older image.

**A NeMo service misbehaves after an update (canary, parakeet):** the images are built on NeMo
3.x. Every release also publishes both services built against NeMo 2.x as `<release>-nemo2`
(for example `0.2.0-nemo2`). Set `NEMO_IMAGE_SUFFIX:-` to `NEMO_IMAGE_SUFFIX:--nemo2` on those two
services (or `NEMO_IMAGE_SUFFIX=-nemo2` in the settings file); nothing else changes.

**The dataset is damaged (a release changed stored data):**

```bash
zfs list -t snapshot -o name,creation tank/apps/tts-stt        # find the snapshot
# Stop the app first: Apps -> Installed -> tts-stt -> Stop
zfs rollback tank/apps/tts-stt@pre-update                       # add -r to drop newer snapshots
```

then put the previous release back as above and Start. `zfs rollback` discards everything written
since the snapshot (voices you created, downloads). To look before you commit, clone it:
`zfs clone tank/apps/tts-stt@pre-update tank/apps/tts-stt-restore`.

---

## 6. VRAM and disk planning

Budget against the VRAM `nvidia-smi` reports for **your** card:

```bash
nvidia-smi --query-gpu=name,memory.total --format=csv
```

These are **planning figures** (resident weights with the default settings, and what the first
start downloads), not measurements on your hardware. `preflight.sh` uses this table.

| Service | VRAM | First-start download | Notes |
|---|---|---|---|
| `frontend-service` | 0 GB | 0 GB | Web UI and gateway |
| `piper-tts-service` | 0 GB | 0.5 GB | CPU only; voices ship in the image |
| `stt-service` (`large-v3-turbo`) | 1.6 GB | 1.6 GB | Default. Best latency/accuracy trade; cannot translate |
| `stt-service` (`large-v3`) | 3.1 GB | - | The only Whisper choice that translates |
| `qwen3-asr-service` | 4 GB | 4 GB | Multilingual recognition |
| `qwen3-tts-service` (`0.6B`) | 2.5 GB | 2.5 GB | Default. Voice cloning |
| `qwen3-tts-service` (`1.7B`) | 4.5 GB | - | Better quality |
| `canary-asr-service` | 2 GB | 1 GB | Estimate: 182M parameters at 4 bytes |
| `parakeet-asr-service` | 3 GB | 2.5 GB | Estimate: 0.6B parameters at 4 bytes |
| `chatterbox-tts-service` | 4 GB | 3 GB | Streaming German TTS |
| `magpie-tts-service` | 4.2 GB | 2.5 GB | NVIDIA TTS, five built-in voices. Measured on an RTX 4080: 4.2 GB of the card with the model loaded, 2.5 GB downloaded, and about 7 GB of host RAM |
| `piper-training-service` | 4 GB | 2 GB | On demand only. Download column is a data and checkpoint reserve |
| `whisper-cpp` | 0 GB | 0.6 GB | CPU only; estimate for the q5_0 model |

The default set (`stt` turbo + `qwen3-asr` + `qwen3-tts` 0.6B) is about **8.1 GB** resident. Images
are about **25 GB** for that set, and each optional GPU image adds several GB more.

Resident figures are peaks, and none of them is permanent: every GPU service releases its weights
after `MODEL_TTL` seconds idle (300) and reloads on the next request, so what has to fit is close
to the models in *use*, not the sum. `STT_MODEL_TTL`, `ASR_MODEL_TTL`, `TTS_MODEL_TTL` override it
per service: `>0` seconds, `0` to free the moment a service is idle, `-1` to keep it resident.
Unloading is reference counted, so a request in flight is never freed underneath itself.
`POST /api/providers/{id}/unload` frees a model early and answers `409` while it is busy.

**Suggested sets**

| Card | Set | Resident |
|---|---|---|
| 8 GB | `frontend`, `piper`, `stt` (turbo) and at most one more service | 2 to 6 GB |
| 12 GB | the default set, with `MODEL_TTL` at 120 if you add one optional service | 8.1 GB, up to about 10 GB |
| 16 GB | the default set plus `chatterbox`, `magpie` or `canary` | about 10 to 12 GB |
| Training | free the VRAM of `qwen3-tts-service` first (`POST /api/providers/qwen3/unload`, or stop the service), then enable `piper-training-service` | training wants several GB |

**Lower latency** (set on `stt-service`; tune against the `decode NNN ms` readout in the live
panel, not against feel):

| Variable | Default | Effect |
|---|---|---|
| `WHISPER_MODEL_SIZE` | `large-v3-turbo` | Dominant quality/speed lever. Move to a smaller model first if decode time nears `WS_MIN_NEW_AUDIO_S x 1000` |
| `WS_WINDOW_S` | `8.0` | Target amount of trailing audio each interim decode re-transcribes; cut at the last finished sentence, never above min(30, 2x). Bounds decoder work, not encoder cost |
| `WS_MIN_NEW_AUDIO_S` | `0.5` | Floor on how often a partial transcript is sent |
| `WS_MAX_SESSIONS` | `4` | Concurrent live microphone sessions |
| `WHISPER_COMPUTE_TYPE` | `float16` | Pinned because INT8 is not available on Blackwell (RTX 50). Empty probes the device |
| `STT_DEFAULT_LANGUAGE` | `de` | Language for API requests that name none (empty = auto-detect). The web UI sends its own choice |

---

## 7. Troubleshooting

| What you see | Cause | Fix |
|---|---|---|
| YAML rejected: `required variable APP_DATA_DIR is missing a value` | The placeholder is still in the file | Replace `${APP_DATA_DIR:?set dataset path}` everywhere with your dataset path |
| Install job fails after about 20 minutes, or `Timed out` in the job log | 25 GB of images did not finish inside TrueNAS's 20 minute limit | Press Install/Save again (layers are kept), or `./pull-images.sh --tag <release>` first |
| `manifest unknown` / `not found` while pulling | The release you pinned has not been published yet | `./update-check.sh --tag <release>`; CI publishes the images after the `vX.Y.Z` tag is pushed |
| `mkdir ...: read-only file system` at start | The data path is not a real dataset path (still `/mnt/pool/...`?) | Fix `APP_DATA_DIR`; `./preflight.sh --data-dir ...` |
| App stays **Deploying** | A container's health check is not passing | `docker ps --filter label=com.docker.compose.project=ix-tts-stt --format 'table {{.Names}}\t{{.Status}}'`, then `docker logs ix-tts-stt-<service>-1` |
| A backend is red in the UI | Not reachable or unhealthy | `curl -s http://localhost:3000/api/health` and read its `error`. `ConnectError` means the service name in `*_SERVICE_URL` does not match a running service |
| `"device": "cpu"` on a GPU service | The container does not see the GPU | `nvidia-smi` on the host; `./preflight.sh --with-docker`; reinstall the driver under Apps -> Settings; check the GPU is not isolated for a VM |
| `could not select device driver "nvidia"` | No NVIDIA runtime in Docker | Install the driver from the Apps settings and restart the Apps service |
| `unsatisfied condition: cuda>=12.8, please update your driver` | The driver is older than R570 (a GeForce card has no other way in) | Update under Apps -> Settings -> Install NVIDIA Drivers; `./preflight.sh` says which case applies |
| A service fails after you enabled another | Out of VRAM | `nvidia-smi` while it runs; smaller Whisper or `0.6B` Qwen3-TTS, shorter `MODEL_TTL`, or drop a service |
| No Piper voices | The models directory is empty and the seed job did not run | `docker logs ix-tts-stt-piper-voices-seed-1`; voices are files `*.onnx` + `*.onnx.json` in `models/default` |
| Microphone button does nothing | Browsers grant the microphone only in a secure context | Use `https://` (or `http://localhost`); check the proxy forwards the WebSocket upgrade |
| `403 Host '...' is not allowed` (`host_not_allowed`) on every request, the page does not even load | The web UI is reached by a name the gateway does not accept without setup: a real domain (a reverse proxy) or a Tailscale MagicDNS name. IP addresses, `localhost`, `*.local`, `*.lan`, `*.internal`, `*.home.arpa` and plain names such as `truenas` are always accepted | Open `http://<truenas-ip>:3000/settings` (by IP address) and add the name under **Access**; it works on the next request. Or set `TRUSTED_HOSTS:-voice.example.com` (comma separated, `*.tail1234.ts.net` for a whole domain) on `frontend-service`. `ALLOWED_HOSTS:-*` switches the check off; avoid it |
| 403 on every button behind a reverse proxy (the page loads) | The proxy changes the `Host` header, so the browser's `Origin` no longer matches it | Add the proxy's public URL under **Settings -> Access** (or `TRUSTED_ORIGINS`); it also trusts that host name |
| The UI asks for an API key, or a script gets `401` | `API_KEY` is set, or **Settings -> API access & keys** requires a key. There is no exemption for the web UI | Type the key into the prompt (once per browser tab); scripts send `Authorization: Bearer <key>`. Keys made in the page work everywhere the YAML key does |
| The Settings page is read-only: "Settings cannot be saved here" | The YAML was pasted before the page existed and has no `settings/` mount | Add the two lines the page shows under `frontend-service:` -> `volumes:` (or paste the current file with your edits), Save |
| The Settings page shows the values but saves nothing; "Nobody can change these settings yet" | No admin key exists: no `API_KEY` in the YAML and none made in the page | Ask the page for a one-time code and read it in the log: **Workloads -> frontend-service -> View Logs**, or `docker logs ix-tts-stt-frontend-service-1` |
| Lost the admin key; Settings answers `401` to everything | The key made in the page is gone from your browser and password manager | **Print a recovery code** under the page's key prompt: a one-time code from the log, as above, creates a new admin key (revoke the old one afterwards). Or delete `settings/keys.json`: every key made in the page is gone, and API access follows the YAML again (open without `API_KEY`) |
| Settings answers `would_lock_out` | The change would refuse the address you are saving from (a host name or proxy you reach it by) | Keep that name, or make the change from `http://<truenas-ip>:3000/settings` |
| A saved value broke something (host name, engine, limit) | The page saved it and it wins over the YAML | **Settings -> History & recovery**: Restore an earlier version, or **Back to the app YAML**. Without the page: delete `settings/gateway.json` (the YAML applies from the next request), create an empty `settings/SAFE-MODE`, or set `ENABLE_SETTINGS_UI:-false` |
| A value you changed in the YAML does nothing | The Settings page saved that setting, and a saved value wins | The field says where its value comes from: press **Reset to app YAML**, or list the setting in `SETTINGS_LOCKED_KEYS` |
| A banner says a saved value was ignored | A hand edit of `settings/gateway.json` with a value the page would refuse; the rest of the file still applies | Save the field again in the page, or fix or delete that line |
| `503` with `Retry-After` | A limit was reached, not a fault: too many uploads or `ffmpeg` conversions at once, or a full or timed-out queue at a model service | Retry after the delay. Limits: `MAX_CONCURRENT_UPLOADS`, `MAX_CONCURRENT_FFMPEG`, `ASR_MAX_QUEUE`, `TTS_MAX_QUEUE`, `ASR_QUEUE_TIMEOUT_S`, `TTS_QUEUE_TIMEOUT_S` |
| Live transcription lags | Decode time exceeds the update interval | Read `decode NNN ms`: under 500 ms is fine; near 1000 ms use a smaller model; over 1500 ms you are probably on CPU or the GPU is contended. Audio is skipped rather than queued, so lag stays bounded |
| Permission problems on the dataset | ACL denies root, or the folder is read-only | Containers run as root; check the dataset's ACL type and that no restrictive ACL blocks root. `preflight.sh` reports a dataset it cannot write |

---

## 8. Reverse proxy and HTTPS

You need HTTPS for the microphone from any machine other than the NAS. Everything is proxied
through port 3000, so the proxy has one upstream. nginx:

```nginx
server {
    listen 443 ssl;
    server_name voice.example.com;
    ssl_certificate     /path/to/fullchain.pem;
    ssl_certificate_key /path/to/privkey.pem;

    location / {
        proxy_pass http://truenas.lan:3000;
        proxy_http_version 1.1;

        # Required: live transcription is a WebSocket.
        proxy_set_header Upgrade    $http_upgrade;
        proxy_set_header Connection "upgrade";

        proxy_set_header Host              $host;
        proxy_set_header X-Real-IP         $remote_addr;
        proxy_set_header X-Forwarded-For   $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Streaming TTS plays before synthesis finishes; buffering would undo that.
        proxy_buffering off;
        proxy_read_timeout 600s;
    }
}
```

`ALLOWED_ORIGINS` stays empty: the UI and the API are the same origin, also over
`http://<ip>:3000`. The gateway refuses a `Host` header that is not an IP address, `localhost`,
`*.local`, `*.lan`, `*.internal`, `*.home.arpa` or a plain name (it is the guard against DNS
rebinding), so the proxy's public name has to be listed: add it under **Settings -> Access** (open
the page by the NAS's IP address to do that), or set `TRUSTED_HOSTS:-voice.example.com` on
`frontend-service` (comma separated; `*.example.com` covers a whole domain; symptom without it:
`403 host_not_allowed` on every request). With `proxy_set_header Host $host` that is all. A proxy
that does not pass the original `Host` needs its public URL as well, `https://voice.example.com`
under **Settings -> Access** or in `TRUSTED_ORIGINS:-` (symptom without it: 403 on every button; the
host name of a trusted origin counts as trusted, so this one setting also satisfies the Host check).
If the proxy sets `X-Forwarded-Host` instead, turn on believing it (`TRUST_PROXY_HEADERS`, only when
the proxy overwrites what the client sent) and list that name as a host name. The page applies a
new proxy URL and the `X-Forwarded-Host` switch for 60 seconds and keeps them only once it has
reached the gateway through the new rules, so a wrong value undoes itself.

To lock a script-facing API, set `API_KEY`, or switch **Settings -> API access & keys** to "a key
is required" and create named keys there: a Bearer token for `/v1/*`, mutating `/api/*` and the
live-transcription socket. There is **no exemption for the web UI**: it asks for the key once per
browser tab and keeps it in `sessionStorage`. A key is a shared secret, not user authentication,
and it crosses the network in clear over plain `http://`: use it behind the HTTPS proxy.

---

## 9. What has and has not been verified

Verified while writing this (no TrueNAS machine or GPU was available), and re-checked by
`tests/test_truenas_*.py` and `.github/workflows/truenas.yml` on every change: the compose files
pass `docker compose config` and resolve as described here; the catalog template renders and
matches the compose file; the release number is the same in every file; the scripts run against
fake `docker`, `nvidia-smi`, `curl`, `skopeo`, `df` and `stat`, which proves their logic and not
what the real tools print. The TrueNAS behaviour in section 3 was read from the TrueNAS source,
not observed.

Not verified, in the order to check them after your first install:

1. `./preflight.sh --data-dir ... --with-docker` reports `READY` on the real host, including the
   GPU test container and the driver verdict.
2. The YAML installs, the app reaches **Running** within a few minutes, and `/api/health` is green.
   Note whether the **Web UI** button and the notes appear (`x-portals` and `x-notes` are read
   from the source, not observed) and whether "Deploying" ends before the model downloads do.
3. Change `IMAGE_TAG` to a second release and Save: it pulls only the new images and the data
   is untouched. Put it back: instant.
4. `./update-check.sh` prints a sensible table, and `UPDATE AVAILABLE` after a newer tag exists.
5. Stop and Start with `PULL_POLICY:-always` and `IMAGE_TAG:-latest` pulls a moved tag.
6. The settings-file route (`include:` with `env_file:`) is accepted by Install via YAML.
7. The NVIDIA driver rule. Verified (from the published config of the `nvidia/cuda:12.8.1` base
   image and NVIDIA's `libnvidia-container` source): the image starts on a driver that reports
   CUDA >= 12.8 (R570 or newer), or on a data-centre/workstation GPU with driver branch 470, 535,
   550, 560 or 565; the list has no `geforce` brand, so a GeForce card needs R570 or newer. Not
   verified: NVIDIA's own minimum for Blackwell (expected R570), and that your card's brand is
   what `preflight.sh` guesses from its name.
8. The menu paths named in this guide (Datasets -> Add Dataset, Apps -> Discover Apps -> Custom
   App -> Install via YAML, Apps -> Settings -> Install NVIDIA Drivers, Data Protection ->
   Snapshots) are from memory of the UI, and whether the YAML editor has search-and-replace.
9. That an exited `piper-voices-seed` counts as normal for the app state. The middleware source
   counts an exited container with a normal exit code as fine while another one runs; the list
   of normal exit codes was not in the copy that was read.
10. The Settings page on the NAS: the **Settings** button of a new install (read from the source,
    like the Web UI button), a host name added in the page working on the next request, the saved
    values surviving Edit/Save, Update and Stop/Start, and the one-time code showing up under
    **Workloads -> View Logs**.

---

## 10. Optional update notifications (Diun)

[Diun](https://crazymax.dev/diun/) watches the images of your containers and **notifies** you when
a newer release is published. It never pulls, restarts or changes anything. The `tts-stt` services
already carry the labels it reads (`diun.enable`, `diun.watch_repo`, `diun.include_tags`,
`diun.sort_tags`, `diun.max_tags`), so only the watcher is missing:

1. Delete the `profiles:` line of the `diun` service in the YAML.
2. Set `DIUN_NTFY_TOPIC:-` to a topic you subscribe to in the [ntfy](https://ntfy.sh/) app (any
   hard-to-guess name), or replace the two `DIUN_NOTIF_NTFY_*` variables with another notifier
   (mail, Telegram, webhook: <https://crazymax.dev/diun/notif/>).
3. Save. It checks once a day at 06:00 (`DIUN_WATCH_SCHEDULE`) and does not notify about the tags
   that exist when it first starts.

It needs read access to the Docker socket, which is root-equivalent on the host: enable it only if
you want it. What's up Docker (WUD) is a comparable alternative; Watchtower is archived and
updates on its own, which is why it is not used here.

---

## 11. Upgrading from an installation before 0.2

- **One model cache.** Models used to live in `.cache/` (Whisper) and `hf-cache/<service>/`.
  Now all services share `cache/`. Reuse what you downloaded instead of downloading it again,
  with the app stopped:

  ```bash
  cd /mnt/tank/apps/tts-stt
  mkdir -p cache
  [ -d .cache ] && cp -an .cache/. cache/
  for d in hf-cache/*/; do [ -d "$d" ] && cp -an "$d". cache/; done
  ```

  Check that the app works, then delete `.cache/` and `hf-cache/`.
- **Images are pinned.** They used to follow `latest`. To keep that, set `IMAGE_TAG:-latest`
  with `PULL_POLICY:-always`.
- **`BROWSER_*_URL` is gone** and backend ports are no longer published. Remove them from your config.
- **CORS is closed by default.** `ALLOWED_ORIGINS` used to default to `*`; the web UI needs none
  of it. List origins only for scripts in other web pages. The backends are closed the same way
  (`BACKEND_ALLOWED_ORIGINS` empty), and a state-changing request that carries a foreign `Origin`
  gets `403`.
- **Host names are validated.** A request must address an IP address, `localhost`, `*.local`,
  `*.lan`, `*.internal`, `*.home.arpa` or a plain name; a real domain or a Tailscale name has to be
  listed in `TRUSTED_HOSTS` or under Settings -> Access, or every request answers
  `403 host_not_allowed` (see Troubleshooting).
- **`API_KEY` has no UI exemption any more.** With a key set, the web UI prompts for it once per
  browser tab; `/ws/stt` needs it too.
- **`MAX_TTS_CHARS` defaults to 5000** (it was 20000): what Qwen3-TTS, Chatterbox and Magpie accept. A
  Piper-only install can raise it.
- **`pcm` speech is 24 kHz** (`X-Sample-Rate: 24000`), as OpenAI documents, instead of the voice's
  own rate.
- **Whisper defaults to `large-v3-turbo`,** which cannot translate. For `task=translate` set
  `WHISPER_MODEL_SIZE:-large-v3`; a translation request on a turbo model returns HTTP 400.

---

## Related

- [`truenas/README.md`](../truenas/README.md): the install routes and the catalog scaffold
- [`truenas-deployment.md`](./truenas-deployment.md): build from source, storage layout, GPU driver
- [`truenas-service-profiles.md`](./truenas-service-profiles.md): which services to run when
- [`truenas-custom-app-checklist.md`](./truenas-custom-app-checklist.md): one-page first-run checklist
- [`provider-contracts.md`](./provider-contracts.md): the API each backend implements
