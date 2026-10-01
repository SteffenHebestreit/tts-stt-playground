#!/usr/bin/env bash
# Pre-pull the images of one release from the TrueNAS shell.
#
# Why this exists: the Apps UI pulls images inside a job that TrueNAS kills after 20 minutes
# (compose_utils.py in truenas/middleware runs `docker compose up` with timeout=1200). The
# default stack is about 25 GB of images, which needs roughly 21 MB/s to finish in time. Layers
# that arrived before the kill are kept, so a retry resumes, but pulling here first turns the
# install or update into a start-up of images that are already on disk.
#
#   ./pull-images.sh --tag 0.2.0
#   ./pull-images.sh --tag 0.2.0 --with canary-asr,chatterbox-tts
#
# It only runs `docker pull`; it does not touch the app. Exit status: 0 = all pulled, 1 = a pull
# failed, 2 = bad arguments.

set -u

HERE=$(cd "$(dirname "$(readlink -f "$0")")" && pwd)
# shellcheck source-path=SCRIPTDIR source=lib.sh
. "$HERE/lib.sh"

usage() {
  cat <<'USAGE'
Usage: pull-images.sh --tag TAG [options]

  --tag TAG         release to pull, e.g. 0.2.0 (default: the version of this checkout, if any)
  --registry REG    image registry/owner (default: ghcr.io/steffenhebestreit)
  --with LIST       also pull optional services: canary-asr, parakeet-asr, chatterbox-tts,
                    magpie-tts, training, whisper-cpp
  --only LIST       pull just these services (names as in the compose file)
  --dry-run         print what would be pulled
  -h, --help        this text
USAGE
}

TAG=$(ts_default_tag)
REGISTRY=$TS_DEFAULT_REGISTRY
WITH=""
ONLY=""
DRY=0

while [ $# -gt 0 ]; do
  case "$1" in
    --tag) TAG=${2:-}; shift 2 ;;
    --registry) REGISTRY=${2:-}; shift 2 ;;
    --with) WITH=${2:-}; shift 2 ;;
    --only) ONLY=${2:-}; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    -h | --help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [ -z "$TAG" ]; then
  echo "no release given: pass --tag 0.2.0 (the value of IMAGE_TAG in the compose file)." >&2
  exit 2
fi

services=()
if [ -n "$ONLY" ]; then
  IFS=',' read -r -a want <<<"$ONLY"
  for item in "${want[@]}"; do
    item=${item// /}
    if [ -z "$(ts_field "$item" 1)" ]; then
      echo "unknown service: $item" >&2
      exit 2
    fi
    services+=("$item")
  done
else
  while IFS= read -r svc; do services+=("$svc"); done < <(ts_services core)
  if [ -n "$WITH" ]; then
    IFS=',' read -r -a want <<<"$WITH"
    for item in "${want[@]}"; do
      item=${item// /}
      svc=$(ts_service_for_profile "$item")
      [ -z "$svc" ] && [ -n "$(ts_field "$item" 1)" ] && svc=$item
      if [ -z "$svc" ]; then
        echo "unknown optional service: $item" >&2
        exit 2
      fi
      services+=("$svc")
    done
  fi
fi

if [ "$DRY" = 0 ] && ! ts_docker_usable; then
  echo "cannot talk to Docker. Run this as root (sudo -i) on the TrueNAS host." >&2
  exit 1
fi

failed=()
for svc in "${services[@]}"; do
  ref=$(ts_image_ref "$REGISTRY" "$svc" "$TAG")
  if [ "$DRY" = 1 ]; then
    echo "would pull $ref"
    continue
  fi
  echo "==> $ref"
  if ! docker pull "$ref"; then
    failed+=("$ref")
  fi
done

if [ "$DRY" = 1 ]; then
  exit 0
fi
if [ "${#failed[@]}" -gt 0 ]; then
  echo
  echo "These pulls failed:"
  printf '  %s\n' "${failed[@]}"
  echo "A failed pull with 'manifest unknown' means the release is not published yet: ./update-check.sh --tag $TAG"
  exit 1
fi
echo
echo "All ${#services[@]} images for $TAG are on this host. Now change IMAGE_TAG in the app and save."
exit 0
