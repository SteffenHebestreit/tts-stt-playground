#!/usr/bin/env bash
# Pre-install readiness check for the TTS-STT TrueNAS app. Run it in the TrueNAS shell
# (System -> Shell, or SSH) before pasting docker-compose.truenas-app.yml into the Apps UI.
#
# Read-only, with two exceptions you ask for explicitly: --create makes the data
# subdirectories, and --with-docker runs one throw-away container to prove the GPU is visible
# from inside Docker.
#
# Exit status: 0 = ready (warnings allowed), 1 = at least one check failed, 2 = bad arguments.
#
#   ./preflight.sh --data-dir /mnt/tank/apps/tts-stt
#   ./preflight.sh --data-dir /mnt/tank/apps/tts-stt --with canary-asr,chatterbox-tts --with-docker
#   ./preflight.sh --data-dir /mnt/tank/apps/tts-stt --no-gpu --with whisper-cpp

set -u

HERE=$(cd "$(dirname "$(readlink -f "$0")")" && pwd)
# shellcheck source-path=SCRIPTDIR source=lib.sh
. "$HERE/lib.sh"

# What can be verified about the NVIDIA driver (read on 2026-09-29 from the published image config of
# nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04 on Docker Hub, the base the GPU images are built on):
#  * The image carries
#      NVIDIA_REQUIRE_CUDA="cuda>=12.8 brand=unknown,driver>=470,driver<471 brand=grid,... brand=tesla,...
#                           (the same brands again for the branches 535, 550, 560 and 565)"
#    NVIDIA's container toolkit refuses to start the container unless ONE alternative holds:
#      - the driver reports CUDA >= 12.8, which is R570 or newer, or
#      - the GPU's brand is in the list AND its driver branch is 470, 535, 550, 560 or 565 (CUDA
#        forward compatibility, which NVIDIA supports on data-centre and workstation GPUs only).
#    The list names unknown, grid, tesla, nvidia, quadro, quadrortx, nvidiartx and the virtual-GPU
#    brands: NOT geforce. A GeForce card (the RTX 30, 40 and 50 series) therefore needs a driver that
#    reports CUDA >= 12.8, whatever its branch. Any branch outside the list (545, 555, ...) is refused.
#  * NOT verified from here: NVIDIA's own minimum driver for Blackwell (RTX 50 series, compute
#    capability 12.0). It is expected to be the R570 branch, the first with CUDA 12.8, so anything
#    older is reported as a failure.
DRIVER_BRANCHES_FORWARD_COMPAT="470 535 550 560 565"
CUDA_MIN="12.8"
BLACKWELL_MIN_DRIVER=570

COMPOSE_MIN="2.20.0" # depends_on.required and include, both used by the compose files

usage() {
  cat <<'USAGE'
Usage: preflight.sh --data-dir PATH [options]

  --data-dir PATH         the dataset you will use as APP_DATA_DIR (default: $APP_DATA_DIR)
  --create                create the data subdirectories (the dataset itself must already exist)
  --with LIST             optional services to plan for, comma separated:
                          canary-asr, parakeet-asr, chatterbox-tts, magpie-tts, training, whisper-cpp
  --whisper-model NAME    stt-service model (default: large-v3-turbo)
  --qwen3-tts-model NAME  qwen3-tts model (default: Qwen/Qwen3-TTS-12Hz-0.6B-Base)
  --gpu-id ID             GPU index or UUID the app will use (default: 0)
  --port N                web UI port (default: 3000)
  --no-gpu                plan a CPU-only install (frontend, piper-tts and whisper-cpp)
  --with-docker           start a throw-away container that must see the GPU
  --gpu-test-image IMAGE  image for --with-docker (default: nvidia/cuda:12.8.1-base-ubuntu22.04)
  --list-vram             print "service<TAB>GB" for every service and exit
  -h, --help              this text
USAGE
}

DATA_DIR=${APP_DATA_DIR:-}
CREATE=0
WITH=""
WHISPER_MODEL="large-v3-turbo"
QWEN3_TTS_MODEL="Qwen/Qwen3-TTS-12Hz-0.6B-Base"
GPU_ID="0"
PORT=$TS_DEFAULT_PORT
NO_GPU=0
WITH_DOCKER=0
GPU_TEST_IMAGE="nvidia/cuda:12.8.1-base-ubuntu22.04"

while [ $# -gt 0 ]; do
  case "$1" in
    --data-dir) DATA_DIR=${2:-}; shift 2 ;;
    --create) CREATE=1; shift ;;
    --with) WITH=${2:-}; shift 2 ;;
    --whisper-model) WHISPER_MODEL=${2:-}; shift 2 ;;
    --qwen3-tts-model) QWEN3_TTS_MODEL=${2:-}; shift 2 ;;
    --gpu-id) GPU_ID=${2:-}; shift 2 ;;
    --port) PORT=${2:-}; shift 2 ;;
    --no-gpu) NO_GPU=1; shift ;;
    --with-docker) WITH_DOCKER=1; shift ;;
    --gpu-test-image) GPU_TEST_IMAGE=${2:-}; shift 2 ;;
    --list-vram)
      ts_catalogue | awk -F'|' '{ printf "%s\t%s\n", $1, $6 }'
      exit 0
      ;;
    -h | --help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

case "$PORT" in '' | *[!0-9]*) echo "--port must be a number" >&2; exit 2 ;; esac

FAILS=0
WARNS=0
ok() { printf '  %s %s\n' "$(ts_paint 32 '[ OK ]')" "$*"; }
warn() { WARNS=$((WARNS + 1)); printf '  %s %s\n' "$(ts_paint 33 '[WARN]')" "$*"; }
fail() { FAILS=$((FAILS + 1)); printf '  %s %s\n' "$(ts_paint 31 '[FAIL]')" "$*"; }
info() { printf '  [INFO] %s\n' "$*"; }
section() { printf '\n%s\n' "$*"; }

# ---------------------------------------------------------------------------------------------
# Which services are we planning for?
# ---------------------------------------------------------------------------------------------
SELECTED=()
if [ "$NO_GPU" = 1 ]; then
  SELECTED+=(frontend-service piper-tts-service)
else
  while IFS= read -r svc; do SELECTED+=("$svc"); done < <(ts_services core)
fi
if [ -n "$WITH" ]; then
  IFS=',' read -r -a WANT <<<"$WITH"
  for item in "${WANT[@]}"; do
    item=${item// /}
    [ -z "$item" ] && continue
    svc=$(ts_service_for_profile "$item")
    [ -z "$svc" ] && [ -n "$(ts_field "$item" 1)" ] && svc=$item
    if [ -z "$svc" ]; then
      echo "unknown optional service: $item (try: canary-asr, parakeet-asr, chatterbox-tts, magpie-tts, training, whisper-cpp)" >&2
      exit 2
    fi
    SELECTED+=("$svc")
  done
fi
# de-duplicate, keep order
mapfile -t SELECTED < <(printf '%s\n' "${SELECTED[@]}" | awk '!seen[$0]++')

selected_has() {
  local s
  for s in "${SELECTED[@]}"; do [ "$s" = "$1" ] && return 0; done
  return 1
}

wants_gpu() {
  local s
  [ "$NO_GPU" = 1 ] && return 1
  for s in "${SELECTED[@]}"; do
    ts_ge "$(ts_field "$s" 6)" 0.1 && return 0
  done
  return 1
}

printf 'TTS-STT TrueNAS preflight\n'
printf 'Planning for: %s\n' "${SELECTED[*]}"

# ---------------------------------------------------------------------------------------------
# Docker and Compose
# ---------------------------------------------------------------------------------------------
section "Docker"
DOCKER_OK=0
if ! ts_have docker; then
  fail "docker not found. TrueNAS SCALE 24.10 (Electric Eel) or newer runs Apps on Docker."
elif ! docker info >/dev/null 2>&1; then
  fail "the Docker daemon is not reachable. Run this as root (sudo -i) and check Apps -> Settings -> Choose Pool."
else
  DOCKER_OK=1
  ok "Docker $(docker version --format '{{.Server.Version}}' 2>/dev/null)"
  compose_version=$(docker compose version --short 2>/dev/null | tr -d 'v')
  if [ -z "$compose_version" ]; then
    fail "the 'docker compose' plugin is missing."
  elif [ "$(ts_version_cmp "$compose_version" "$COMPOSE_MIN")" = "-1" ]; then
    fail "Docker Compose $compose_version is older than $COMPOSE_MIN, which the compose files need (depends_on required, include)."
  else
    ok "Docker Compose $compose_version"
  fi
fi

# ---------------------------------------------------------------------------------------------
# GPU
# ---------------------------------------------------------------------------------------------
GPU_VRAM_GB=""
if wants_gpu; then
  section "NVIDIA GPU"
  if ! ts_have nvidia-smi; then
    fail "nvidia-smi not found. Install the driver under Apps -> Settings -> Install NVIDIA Drivers, reboot, or use --no-gpu."
  else
    query="index,uuid,name,driver_version,memory.total"
    rows=$(nvidia-smi --query-gpu="$query,compute_cap" --format=csv,noheader,nounits 2>/dev/null) || rows=""
    has_cc=1
    if [ -z "$rows" ]; then
      has_cc=0
      rows=$(nvidia-smi --query-gpu="$query" --format=csv,noheader,nounits 2>/dev/null) || rows=""
    fi
    if [ -z "$rows" ]; then
      fail "nvidia-smi is installed but reports no GPU. Is the card isolated for a VM (Settings -> Isolated GPU Device(s)), or is the driver not loaded?"
    else
      picked=""
      while IFS= read -r row; do
        IFS=',' read -r idx uuid name drv mem cc <<<"$row"
        idx=${idx// /}; uuid=${uuid// /}; drv=${drv// /}; mem=${mem// /}; cc=${cc// /}
        name=${name# }
        info "GPU $idx: $name, driver $drv, ${mem} MiB${cc:+, compute capability $cc}"
        if [ "$idx" = "$GPU_ID" ] || [ "$uuid" = "$GPU_ID" ]; then
          picked="$idx|$name|$drv|$mem|$cc"
        fi
      done <<<"$rows"
      if [ -z "$picked" ]; then
        fail "no GPU with index or UUID '$GPU_ID' (see the list above; set GPU_DEVICE_ID to match)."
      else
        IFS='|' read -r idx name drv mem cc <<<"$picked"
        GPU_VRAM_GB=$(awk -v m="$mem" 'BEGIN { printf "%.1f", m / 1024 }')
        drv_major=${drv%%.*}
        cuda_seen=$(nvidia-smi 2>/dev/null | grep -o 'CUDA Version: *[0-9][0-9.]*' | head -n 1 | awk '{ print $NF }')
        blackwell=0
        if [ "$has_cc" = 1 ] && [ -n "$cc" ]; then
          [ "${cc%%.*}" -ge 12 ] 2>/dev/null && blackwell=1
        elif [[ "$name" =~ RTX\ 50[0-9][0-9] ]]; then
          blackwell=1
        fi

        cuda_ok=0
        if [ -n "$cuda_seen" ] && [ "$(ts_version_cmp "$cuda_seen" "$CUDA_MIN")" != "-1" ]; then
          cuda_ok=1
        fi
        branch_ok=0
        for b in $DRIVER_BRANCHES_FORWARD_COMPAT; do [ "$drv_major" = "$b" ] && branch_ok=1; done
        # The forward-compatibility alternatives of the image do not list the geforce brand.
        geforce=0
        [[ "$name" =~ GeForce|TITAN ]] && geforce=1

        if [ "$blackwell" = 1 ] && [ "$drv_major" -lt "$BLACKWELL_MIN_DRIVER" ] 2>/dev/null; then
          fail "driver $drv is too old for a Blackwell GPU ($name); the R$BLACKWELL_MIN_DRIVER branch (CUDA 12.8) or newer is required. Update under Apps -> Settings -> Install NVIDIA Drivers."
        elif [ "$cuda_ok" = 1 ]; then
          ok "driver $drv supports CUDA ${cuda_seen} (the images need >= $CUDA_MIN)"
        elif [ "$branch_ok" = 1 ] && [ "$geforce" = 0 ]; then
          warn "driver $drv reports CUDA ${cuda_seen:-unknown}: the CUDA $CUDA_MIN base image accepts it only through CUDA forward compatibility (data-centre and workstation GPUs, branch R$drv_major). A driver with CUDA >= $CUDA_MIN (R570 or newer) is the safe choice."
        elif [ "$geforce" = 1 ]; then
          fail "driver $drv reports CUDA ${cuda_seen:-unknown}, and the CUDA $CUDA_MIN base image accepts a GeForce card only on a driver that reports CUDA >= $CUDA_MIN (R570 or newer), whatever its branch. Update under Apps -> Settings -> Install NVIDIA Drivers."
        else
          fail "driver $drv (branch $drv_major, CUDA ${cuda_seen:-unknown}) is refused by the CUDA $CUDA_MIN base image, which accepts CUDA >= $CUDA_MIN or the branches: $DRIVER_BRANCHES_FORWARD_COMPAT (data-centre and workstation GPUs)."
        fi

        if [ "$blackwell" = 1 ]; then
          info "Blackwell GPU: the images ship cu128 kernels for compute capability 12.0."
        fi
        ok "GPU $idx will be used: ${GPU_VRAM_GB} GB VRAM"
      fi
    fi
  fi

  if [ "$DOCKER_OK" = 1 ]; then
    runtimes=$(docker info --format '{{range $k, $v := .Runtimes}}{{$k}} {{end}}' 2>/dev/null)
    if [[ " $runtimes " == *" nvidia "* ]]; then
      ok "Docker has the nvidia runtime"
    else
      warn "Docker does not list an nvidia runtime (found: ${runtimes:-none}). TrueNAS registers it when the NVIDIA driver is installed from the Apps settings."
    fi

    if [ "$WITH_DOCKER" = 1 ]; then
      if out=$(docker run --rm --gpus "device=${GPU_ID}" "$GPU_TEST_IMAGE" nvidia-smi -L 2>&1); then
        if grep -q '^GPU ' <<<"$out"; then
          ok "a container sees the GPU: $(grep -m1 '^GPU ' <<<"$out")"
        else
          fail "the test container ran but printed no GPU: $out"
        fi
      else
        fail "could not run a GPU container from $GPU_TEST_IMAGE: $(tail -n 3 <<<"$out" | tr '\n' ' ')"
      fi
    else
      info "add --with-docker to prove a container can use the GPU (pulls $GPU_TEST_IMAGE once)."
    fi
  fi
fi

# ---------------------------------------------------------------------------------------------
# Data directory
# ---------------------------------------------------------------------------------------------
section "Data directory"
DATA_FS_DEV=""
if [ -z "$DATA_DIR" ]; then
  fail "no data directory given. Pass --data-dir /mnt/<pool>/<dataset> (the value you will use as APP_DATA_DIR)."
else
  if [ "$DATA_DIR" = "/mnt/pool/apps/tts-stt" ]; then
    warn "$DATA_DIR is the placeholder from the compose file. Use the real path of your dataset."
  fi
  if [ ! -d "$DATA_DIR" ]; then
    fail "$DATA_DIR does not exist. Create it as a dataset first (Datasets -> Add Dataset), so it can be snapshotted; --create only makes the subdirectories."
  else
    if probe=$(mktemp "$DATA_DIR/.preflight.XXXXXX" 2>/dev/null); then
      rm -f "$probe"
      ok "$DATA_DIR exists and is writable"
    else
      fail "$DATA_DIR is not writable for this user. The containers run as root, so run this script as root too."
    fi
    DATA_FS_DEV=$(stat -c %d "$DATA_DIR" 2>/dev/null)

    fstype=$(stat -f -c %T "$DATA_DIR" 2>/dev/null)
    if [ "$fstype" = "zfs" ]; then
      dataset=$(findmnt -no SOURCE --target "$DATA_DIR" 2>/dev/null)
      if [ -n "$dataset" ]; then
        ok "ZFS dataset $dataset"
        info "before every update: zfs snapshot ${dataset}@pre-update-\$(date +%Y%m%d)"
      else
        ok "ZFS filesystem"
      fi
    else
      warn "$DATA_DIR is on '${fstype:-unknown}', not ZFS: there will be no dataset snapshots to roll back to."
    fi

    # settings: what the web UI's Settings page saves (frontend-service, read-write);
    # backend-settings: the backends' settings folder (frontend read-write, backends read-only).
    subdirs=(models output cache qwen3-voices settings backend-settings)
    selected_has piper-training-service && subdirs+=(piper-training-service/data piper-training-service/checkpoints piper-training-service/models piper-training-service/configs)
    selected_has whisper-cpp && subdirs+=(whisper-cpp-models)
    missing=()
    for sub in "${subdirs[@]}"; do
      [ -d "$DATA_DIR/$sub" ] || missing+=("$sub")
    done
    if [ "${#missing[@]}" -eq 0 ]; then
      ok "all data subdirectories exist"
    elif [ "$CREATE" = 1 ]; then
      if mkdir -p "${missing[@]/#/$DATA_DIR/}"; then
        ok "created: ${missing[*]}"
      else
        fail "could not create the data subdirectories under $DATA_DIR"
      fi
    else
      info "docker creates these on first start: ${missing[*]} (or run again with --create)"
    fi
    info "the containers run as root (no USER in any Dockerfile), so no chown or ACL entry is needed for them."
  fi
fi

# ---------------------------------------------------------------------------------------------
# Disk space
# ---------------------------------------------------------------------------------------------
section "Disk space"
free_gb() { # free_gb PATH -> free GiB with one decimal
  df -Pk "$1" 2>/dev/null | awk 'NR == 2 { printf "%.1f", $4 / 1048576 }'
}
models_gb=0
for svc in "${SELECTED[@]}"; do
  models_gb=$(ts_sum "$models_gb" "$(ts_field "$svc" 7)")
done
if [ -n "$DATA_DIR" ] && [ -d "$DATA_DIR" ]; then
  free=$(free_gb "$DATA_DIR")
  # 1.5x for a second model version during an update, plus room for generated audio and voices.
  recommended=$(awk -v m="$models_gb" 'BEGIN { printf "%.1f", m * 1.5 + 5 }')
  if [ -z "$free" ]; then
    warn "could not read the free space of $DATA_DIR"
  elif ! ts_ge "$free" "$models_gb"; then
    fail "$free GB free on $DATA_DIR, but the selected models alone need about $models_gb GB."
  elif ! ts_ge "$free" "$recommended"; then
    warn "$free GB free on $DATA_DIR; the models need about $models_gb GB and $recommended GB is comfortable."
  else
    ok "$free GB free on $DATA_DIR (models need about $models_gb GB)"
  fi
fi
if [ "$DOCKER_OK" = 1 ] && wants_gpu; then
  docker_root=$(docker info --format '{{.DockerRootDir}}' 2>/dev/null)
  if [ -n "$docker_root" ] && [ -d "$docker_root" ]; then
    free=$(free_gb "$docker_root")
    need=$TS_IMAGES_CORE_GB
    # Same filesystem as the data dir: the two needs add up.
    if [ -n "$DATA_FS_DEV" ] && [ "$(stat -c %d "$docker_root" 2>/dev/null)" = "$DATA_FS_DEV" ]; then
      need=$(ts_sum "$need" "$models_gb")
    fi
    if [ -z "$free" ]; then
      warn "could not read the free space of the Docker root $docker_root"
    elif ! ts_ge "$free" "$need"; then
      fail "$free GB free under the Docker root $docker_root; the images need about $need GB."
    elif ! ts_ge "$free" "$(awk -v n="$need" 'BEGIN { print n * 1.4 }')"; then
      warn "$free GB free under the Docker root $docker_root; the images need about $need GB and an update briefly holds two versions."
    else
      ok "$free GB free under the Docker root $docker_root (images need about $need GB)"
    fi
    info "optional NeMo, Chatterbox, Magpie and training images are large as well; leave extra room for each."
  fi
fi

# ---------------------------------------------------------------------------------------------
# Port
# ---------------------------------------------------------------------------------------------
section "Network"
port_in_use() {
  if ts_have ss; then
    ss -H -ltn "sport = :$1" 2>/dev/null | grep -q .
  elif ts_have netstat; then
    netstat -ltn 2>/dev/null | awk -v p=":$1" '$4 ~ p"$" { found = 1 } END { exit !found }'
  else
    (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null
  fi
}
if port_in_use "$PORT"; then
  ours=""
  if [ "$DOCKER_OK" = 1 ]; then
    ours=$(docker ps --filter "label=com.docker.compose.project=ix-${TS_DEFAULT_APP}" --format '{{.Ports}}' 2>/dev/null | grep -E "[:.]${PORT}->" | head -n 1)
  fi
  if [ -n "$ours" ]; then
    info "port $PORT is already published by the running $TS_DEFAULT_APP app: this is an update, not a fresh install (see update-check.sh)."
  else
    fail "port $PORT is already in use ($(ss -H -ltnp "sport = :$PORT" 2>/dev/null | awk '{ print $NF }' | head -n 1)). Pick another FRONTEND_PORT, or stop what listens there."
  fi
else
  ok "port $PORT is free"
fi

# ---------------------------------------------------------------------------------------------
# VRAM budget
# ---------------------------------------------------------------------------------------------
if wants_gpu; then
  section "VRAM budget (resident models, docs/truenas-installation-guide.md)"
  total="0"
  largest="0"
  largest_name=""
  for svc in "${SELECTED[@]}"; do
    gb=$(ts_field "$svc" 6)
    case "$svc" in
      stt-service)
        case "$WHISPER_MODEL" in large-v3-turbo) : ;; *) gb=3.1 ;; esac ;;
      qwen3-tts-service)
        case "$QWEN3_TTS_MODEL" in *1.7B*) gb=4.5 ;; esac ;;
    esac
    ts_ge "$gb" 0.1 || continue
    printf '         %-24s %5s GB\n' "$svc" "$gb"
    total=$(ts_sum "$total" "$gb")
    if ! ts_ge "$largest" "$gb"; then largest=$gb; largest_name=$svc; fi
  done
  printf '         %-24s %5s GB\n' "all resident at once" "$total"
  if [ -z "$GPU_VRAM_GB" ]; then
    info "no GPU was read above, so the total cannot be compared with your card."
  elif ! ts_ge "$GPU_VRAM_GB" "$largest"; then
    fail "the largest service ($largest_name, $largest GB) does not fit in ${GPU_VRAM_GB} GB of VRAM."
  elif ts_ge "$(awk -v v="$GPU_VRAM_GB" 'BEGIN { print v * 0.9 }')" "$total"; then
    ok "$total GB resident of ${GPU_VRAM_GB} GB VRAM: fits with headroom"
  else
    warn "$total GB resident is more than 90% of ${GPU_VRAM_GB} GB VRAM. It works because idle models unload after MODEL_TTL (300 s), but simultaneous use can run out of memory. Drop a service or shorten the TTLs."
  fi
  info "figures are the guide's resident sizes; check the live number with: nvidia-smi --query-gpu=memory.used,memory.total --format=csv"
fi

# ---------------------------------------------------------------------------------------------
section "Summary"
if [ "$FAILS" -gt 0 ]; then
  printf '  %s %d check(s) failed, %d warning(s). Fix the failures before installing.\n' "$(ts_paint 31 'NOT READY')" "$FAILS" "$WARNS"
  exit 1
fi
printf '  %s %d warning(s).\n' "$(ts_paint 32 'READY')" "$WARNS"
printf '  Next: paste docker-compose.truenas-app.yml into Apps -> Discover Apps -> Custom App -> Install via YAML.\n'
exit 0
