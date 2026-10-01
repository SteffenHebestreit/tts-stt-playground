#!/usr/bin/env bash
# Is a newer TTS-STT release (or a newer image behind your floating tag) available?
#
# READ-ONLY. It never pulls, restarts or edits anything: it compares what your containers run
# with what the registry publishes, and tells you. Run it in the TrueNAS shell, or from cron.
#
#   ./update-check.sh                       # inspect the running app "tts-stt"
#   ./update-check.sh --tag 0.3.0           # have the images for 0.3.0 been published yet?
#   ./update-check.sh --quiet && echo fine  # silent unless there is something to do
#
# Exit status: 0 = up to date, 10 = update available, 2 = could not check (no network, private
# registry), 1 = usage error or the app was not found.
#
# Registry access tries `docker buildx imagetools`, then `skopeo`, then plain curl against the
# registry API, so it works on a host that has none of the plugins.

set -u

HERE=$(cd "$(dirname "$(readlink -f "$0")")" && pwd)
# shellcheck source-path=SCRIPTDIR source=lib.sh
. "$HERE/lib.sh"

usage() {
  cat <<'USAGE'
Usage: update-check.sh [options]

  --app NAME        TrueNAS app name (default: tts-stt); its compose project is ix-NAME
  --project NAME    compose project name, if it does not follow the ix-NAME rule
  --registry REG    image registry/owner (default: ghcr.io/steffenhebestreit)
  --tag TAG         do not inspect containers; check that TAG exists for every image
  --with LIST       with --tag: also check optional services (canary-asr, parakeet-asr,
                    chatterbox-tts, magpie-tts, training, whisper-cpp)
  --offline         list what runs, contact no registry
  --quiet           print only updates and problems
  -h, --help        this text
USAGE
}

APP=$TS_DEFAULT_APP
PROJECT=""
REGISTRY=$TS_DEFAULT_REGISTRY
TAG=""
WITH=""
OFFLINE=0
QUIET=0

while [ $# -gt 0 ]; do
  case "$1" in
    --app) APP=${2:-}; shift 2 ;;
    --project) PROJECT=${2:-}; shift 2 ;;
    --registry) REGISTRY=${2:-}; shift 2 ;;
    --tag) TAG=${2:-}; shift 2 ;;
    --with) WITH=${2:-}; shift 2 ;;
    --offline) OFFLINE=1; shift ;;
    --quiet) QUIET=1; shift ;;
    -h | --help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done
[ -z "$PROJECT" ] && PROJECT="ix-${APP}"

say() { [ "$QUIET" = 1 ] || printf '%s\n' "$*"; }

# ---------------------------------------------------------------------------------------------
# --tag: have the images for a release been published?
# ---------------------------------------------------------------------------------------------
if [ -n "$TAG" ]; then
  services=()
  while IFS= read -r svc; do services+=("$svc"); done < <(ts_services core)
  if [ -n "$WITH" ]; then
    IFS=',' read -r -a want <<<"$WITH"
    for item in "${want[@]}"; do
      svc=$(ts_service_for_profile "${item// /}")
      [ -z "$svc" ] && [ -n "$(ts_field "${item// /}" 1)" ] && svc=${item// /}
      if [ -z "$svc" ]; then
        echo "unknown optional service: $item" >&2
        exit 2
      fi
      services+=("$svc")
    done
  fi
  missing=0
  unknown=0
  for svc in "${services[@]}"; do
    ref=$(ts_image_ref "$REGISTRY" "$svc" "$TAG")
    if digest=$(ts_registry_digest "$ref"); then
      say "  published  $ref  ${digest:0:19}"
    else
      # Could not resolve: either the tag does not exist or the registry is unreachable. A second,
      # tag-independent request tells the two apart.
      ts_split_ref "$ref"
      if ts_registry_tags "$TS_REF_REGISTRY/$TS_REF_NAME" >/dev/null 2>&1; then
        printf '  MISSING    %s\n' "$ref"
        missing=$((missing + 1))
      else
        printf '  UNKNOWN    %s (registry not reachable, or the package is private)\n' "$ref"
        unknown=$((unknown + 1))
      fi
    fi
  done
  if [ "$missing" -gt 0 ]; then
    echo "Tag $TAG is not (fully) published: do not deploy it yet. CI publishes the images after a vX.Y.Z tag is pushed."
    exit 10
  fi
  if [ "$unknown" -gt 0 ]; then
    echo "Could not verify tag $TAG."
    exit 2
  fi
  say "Tag $TAG is published for all ${#services[@]} checked images."
  exit 0
fi

# ---------------------------------------------------------------------------------------------
# Inspect the running app
# ---------------------------------------------------------------------------------------------
if ! ts_docker_usable; then
  echo "cannot talk to Docker. Run this as root (sudo -i) on the TrueNAS host." >&2
  exit 1
fi

mapfile -t NAMES < <(docker ps --filter "label=com.docker.compose.project=${PROJECT}" --format '{{.Names}}' 2>/dev/null)
if [ "${#NAMES[@]}" -eq 0 ]; then
  echo "no running containers for compose project '${PROJECT}'. Is the app installed under another name (--app), or stopped?" >&2
  exit 1
fi

UPDATES=0
UNKNOWN=0
CURRENT_TAG=""
MIXED_TAGS=0
# Assigned, not just declared: an empty declared array is "unbound" under `set -u` in bash 5.2.
ROW_SERVICE=()
ROW_REF=()
ROW_STATUS=()
ROW_DETAIL=()

for name in "${NAMES[@]}"; do
  line=$(docker inspect --format '{{.Config.Image}}|{{.Image}}|{{index .Config.Labels "com.docker.compose.service"}}' "$name" 2>/dev/null) || continue
  IFS='|' read -r ref image_id service <<<"$line"
  [[ "${ref##*/}" == tts-stt-* ]] || continue # ignore anything that is not one of our images

  ts_split_ref "$ref"
  repo="$TS_REF_REGISTRY/$TS_REF_NAME"
  tag=$TS_REF_TAG
  if ts_is_semver "$tag"; then
    if [ -z "$CURRENT_TAG" ]; then CURRENT_TAG=$tag; elif [ "$CURRENT_TAG" != "$tag" ]; then MIXED_TAGS=1; fi
  fi

  local_digest=$(docker image inspect --format '{{range .RepoDigests}}{{println .}}{{end}}' "$image_id" 2>/dev/null |
    awk -F@ -v r="$repo" '$1 == r { print $2; exit }')

  status="" detail=""
  if [ "$OFFLINE" = 1 ]; then
    status="not checked"
    detail="offline"
  elif remote_digest=$(ts_registry_digest "$ref"); then
    if [ -z "$local_digest" ]; then
      status="unknown"
      detail="no registry digest on the local image (built or loaded locally?)"
      UNKNOWN=$((UNKNOWN + 1))
    elif [ "$local_digest" = "$remote_digest" ]; then
      status="up to date"
      detail="${remote_digest:0:19}"
    else
      status="UPDATE"
      detail="registry has ${remote_digest:0:19}, running ${local_digest:0:19}"
      UPDATES=$((UPDATES + 1))
    fi
  else
    status="unknown"
    detail="registry did not answer"
    UNKNOWN=$((UNKNOWN + 1))
  fi

  ROW_SERVICE+=("${service:-$name}")
  ROW_REF+=("$ref")
  ROW_STATUS+=("$status")
  ROW_DETAIL+=("$detail")
done

if [ "${#ROW_SERVICE[@]}" -eq 0 ]; then
  echo "project '${PROJECT}' runs no ${REGISTRY##*/}/tts-stt-* images. Nothing to check." >&2
  exit 1
fi

# ---------------------------------------------------------------------------------------------
# A newer release, for installs pinned to a version
# ---------------------------------------------------------------------------------------------
NEWER=""
NEWER_PENDING=""
if [ "$OFFLINE" = 0 ] && [ -n "$CURRENT_TAG" ]; then
  if tags=$(ts_registry_tags "$REGISTRY/tts-stt-frontend"); then
    newest=$(printf '%s\n' "$tags" | ts_highest_semver)
    if [ -n "$newest" ] && [ "$(ts_version_cmp "$newest" "$CURRENT_TAG")" = "1" ]; then
      # CI publishes the images of one release in parallel. Only call it available once every
      # image this app runs exists under the new tag, or the update would fail on the lagging one.
      lagging=()
      for i in "${!ROW_REF[@]}"; do
        ts_split_ref "${ROW_REF[i]}"
        if ! ts_registry_digest "$TS_REF_REGISTRY/$TS_REF_NAME:$newest" >/dev/null 2>&1; then
          lagging+=("${ROW_SERVICE[i]}")
        fi
      done
      if [ "${#lagging[@]}" -eq 0 ]; then
        NEWER=$newest
        UPDATES=$((UPDATES + 1))
      else
        NEWER_PENDING="$newest (still publishing: ${lagging[*]})"
      fi
    fi
  else
    UNKNOWN=$((UNKNOWN + 1))
    say "Could not list release tags for $REGISTRY/tts-stt-frontend, so a newer release cannot be detected."
  fi
fi

# ---------------------------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------------------------
if [ "$QUIET" = 0 ] || [ "$UPDATES" -gt 0 ] || [ "$UNKNOWN" -gt 0 ]; then
  printf '%-26s %-46s %s\n' "SERVICE" "IMAGE" "STATUS"
  for i in "${!ROW_SERVICE[@]}"; do
    if [ "$QUIET" = 1 ] && [ "${ROW_STATUS[i]}" = "up to date" ]; then continue; fi
    printf '%-26s %-46s %s (%s)\n' "${ROW_SERVICE[i]}" "${ROW_REF[i]##*/}" "${ROW_STATUS[i]}" "${ROW_DETAIL[i]}"
  done
fi
[ "$MIXED_TAGS" = 1 ] && echo "Note: the services run different release tags. Set one IMAGE_TAG for all of them."
[ -n "$NEWER_PENDING" ] && echo "Release $NEWER_PENDING: try again in a few minutes."

# The optional services this app runs, as the names pull-images.sh takes with --with.
running_optional=""
for svc in "${ROW_SERVICE[@]}"; do
  profile=$(ts_field "$svc" 5)
  [ -n "$profile" ] && running_optional="${running_optional:+$running_optional,}$profile"
done

if [ -n "$NEWER" ]; then
  echo
  echo "UPDATE AVAILABLE: release $NEWER is published (you run $CURRENT_TAG)."
  echo "  1. Snapshot the dataset, then change the release in the app and save (see docs/truenas-installation-guide.md, 'Update')."
  echo "  2. Pre-pull the images to stay clear of the 20 minute Apps job limit:"
  echo "     ./pull-images.sh --tag $NEWER${running_optional:+ --with $running_optional}"
fi
if [ "$UPDATES" -gt 0 ] && [ -z "$NEWER" ]; then
  # Which tag, and which services, moved: one pull command per tag.
  mapfile -t stale_tags < <(
    for i in "${!ROW_REF[@]}"; do
      if [ "${ROW_STATUS[i]}" = "UPDATE" ]; then
        ts_split_ref "${ROW_REF[i]}"
        echo "$TS_REF_TAG"
      fi
    done | sort -u
  )
  echo
  echo "UPDATE AVAILABLE: the registry holds a newer image for a tag you follow."
  echo "  1. Pull it:"
  for tag in "${stale_tags[@]}"; do
    stale_services=""
    for i in "${!ROW_REF[@]}"; do
      ts_split_ref "${ROW_REF[i]}"
      if [ "${ROW_STATUS[i]}" = "UPDATE" ] && [ "$TS_REF_TAG" = "$tag" ]; then
        stale_services="${stale_services:+$stale_services,}${ROW_SERVICE[i]}"
      fi
    done
    echo "     ./pull-images.sh --tag $tag --only $stale_services"
  done
  echo "  2. Apps -> $APP -> Stop, then Start: the containers are recreated from the pulled images."
  echo "     (With PULL_POLICY set to always, a Stop and Start pulls by itself and step 1 is not needed.)"
fi

if [ "$OFFLINE" = 1 ]; then
  echo "Offline: nothing was compared with a registry."
  exit 0
fi
if [ "$UPDATES" -gt 0 ]; then exit 10; fi
if [ "$UNKNOWN" -gt 0 ]; then
  echo "Could not check everything (see above); this is not a confirmation that you are up to date."
  exit 2
fi
if [ -n "$NEWER_PENDING" ]; then
  say "Up to date with what is published so far."
else
  say "Up to date."
fi
exit 0
