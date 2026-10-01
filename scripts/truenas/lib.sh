# shellcheck shell=bash
# The settings below are read by the scripts that source this file.
# shellcheck disable=SC2034
# Shared definitions for the TrueNAS helper scripts (preflight.sh, update-check.sh,
# pull-images.sh). Source it; do not run it.
#
# One service catalogue, so the three scripts cannot disagree about which images exist,
# which port a service listens on or how much VRAM it needs. tests/test_truenas_scripts.py
# checks this table against docker-compose.truenas-app.yml and against the VRAM table in
# docs/truenas-installation-guide.md, so a change in either place fails a test instead of
# drifting silently.

TS_DEFAULT_REGISTRY="ghcr.io/steffenhebestreit"
TS_DEFAULT_APP="tts-stt"
TS_DEFAULT_PORT=3000

# name | image suffix (tts-stt-<suffix>) | container port | group | compose profile | VRAM GB | model disk GB
# VRAM is the resident figure with the default model (docs/truenas-installation-guide.md, "VRAM
# planning"). Disk is what the first start downloads: measured where the repo says so, otherwise a
# planning estimate (parameters x bytes), which the guide labels as such.
ts_catalogue() {
  cat <<'CATALOGUE'
frontend-service|frontend|3000|core||0|0
piper-tts-service|piper-tts|5000|core||0|0.5
stt-service|stt|8000|core||1.6|1.6
qwen3-asr-service|qwen3-asr|5002|core||4|4
qwen3-tts-service|qwen3-tts|5004|core||2.5|2.5
canary-asr-service|canary-asr|5006|optional|canary-asr|2|1
parakeet-asr-service|parakeet-asr|5005|optional|parakeet-asr|3|2.5
chatterbox-tts-service|chatterbox-tts|5007|optional|chatterbox-tts|4|3
magpie-tts-service|magpie-tts|5008|optional|magpie-tts|4.2|2.5
piper-training-service|piper-training|8080|optional|training|4|2
whisper-cpp|whisper-cpp|8080|optional|whisper-cpp|0|0.6
CATALOGUE
}

# Docker root free space the default image set needs (docs: "Container images ~25 GB").
TS_IMAGES_CORE_GB=25

# ts_field NAME INDEX -> one field (1-based) of NAME's catalogue row.
ts_field() {
  ts_catalogue | awk -F'|' -v n="$1" -v i="$2" '$1 == n { print $i; exit }'
}

# ts_services core|optional|all -> service names, one per line.
ts_services() {
  case "$1" in
    all) ts_catalogue | awk -F'|' '{ print $1 }' ;;
    *) ts_catalogue | awk -F'|' -v g="$1" '$4 == g { print $1 }' ;;
  esac
}

# ts_service_for_profile PROFILE -> service name owning that compose profile.
ts_service_for_profile() {
  ts_catalogue | awk -F'|' -v p="$1" '$5 == p { print $1; exit }'
}

# ts_image_ref REGISTRY SERVICE TAG -> full image reference.
ts_image_ref() {
  printf '%s/tts-stt-%s:%s\n' "$1" "$(ts_field "$2" 2)" "$3"
}

# ---------------------------------------------------------------------------------------------
# Version helpers
# ---------------------------------------------------------------------------------------------

# ts_is_semver X -> success for exactly MAJOR.MINOR.PATCH (a pinned release tag).
ts_is_semver() {
  [[ "$1" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]
}

# ts_version_cmp A B -> prints -1, 0 or 1 comparing dotted numeric versions ("v" prefix and any
# pre-release/build suffix ignored; missing components count as 0).
ts_version_cmp() {
  local a=${1#v} b=${2#v} i x y
  a=${a%%[-+]*}
  b=${b%%[-+]*}
  local -a pa pb
  IFS=. read -r -a pa <<<"$a"
  IFS=. read -r -a pb <<<"$b"
  for i in 0 1 2 3; do
    x=${pa[i]:-0}
    y=${pb[i]:-0}
    x=$((10#${x:-0}))
    y=$((10#${y:-0}))
    if ((x > y)); then
      echo 1
      return 0
    elif ((x < y)); then
      echo -1
      return 0
    fi
  done
  echo 0
}

# ts_highest_semver -> reads tags on stdin, prints the highest MAJOR.MINOR.PATCH (nothing if none).
ts_highest_semver() {
  local best="" tag
  while IFS= read -r tag; do
    ts_is_semver "$tag" || continue
    if [ -z "$best" ] || [ "$(ts_version_cmp "$tag" "$best")" = "1" ]; then
      best=$tag
    fi
  done
  [ -n "$best" ] && printf '%s\n' "$best"
  return 0
}

# ts_default_tag -> the release this checkout ships (truenas/tts-stt/app.yaml app_version), when the
# scripts run from a git checkout. Empty otherwise; callers then ask for --tag.
ts_default_tag() {
  local here yaml
  here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd) || return 0
  yaml="$here/../../truenas/tts-stt/app.yaml"
  [ -r "$yaml" ] || return 0
  sed -n 's/^app_version:[[:space:]]*"\{0,1\}\([0-9][0-9.]*\)"\{0,1\}[[:space:]]*$/\1/p' "$yaml" | head -n 1
}

# ts_split_ref REF -> sets TS_REF_REGISTRY, TS_REF_NAME, TS_REF_TAG (tag defaults to latest).
ts_split_ref() {
  local ref=$1 rest
  TS_REF_TAG=latest
  rest=${ref%%@*}
  if [[ "${rest##*/}" == *:* ]]; then
    TS_REF_TAG=${rest##*:}
    rest=${rest%:*}
  fi
  TS_REF_REGISTRY=${rest%%/*}
  TS_REF_NAME=${rest#*/}
}

# ---------------------------------------------------------------------------------------------
# Registry access (read-only). Tries docker buildx, then skopeo, then plain curl against the
# registry HTTP API, so it degrades on hosts that lack the plugin. TS_REGISTRY_METHOD pins one.
# ---------------------------------------------------------------------------------------------

TS_ACCEPT_MANIFESTS="application/vnd.oci.image.index.v1+json, application/vnd.docker.distribution.manifest.list.v2+json, application/vnd.oci.image.manifest.v1+json, application/vnd.docker.distribution.manifest.v2+json"

ts_have() {
  command -v "$1" >/dev/null 2>&1
}

# Anonymous pull token (GHCR and Docker Hub style). Empty output means "try without a token".
ts_curl_token() {
  local registry=$1 name=$2 body
  body=$(curl -fsS --max-time 20 "https://${registry}/token?service=${registry}&scope=repository:${name}:pull" 2>/dev/null) || return 0
  printf '%s' "$body" | sed -n 's/.*"token"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p'
}

ts_curl_digest() {
  local registry=$1 name=$2 tag=$3 token headers
  local -a auth=()
  token=$(ts_curl_token "$registry" "$name")
  [ -n "$token" ] && auth=(-H "Authorization: Bearer $token")
  headers=$(curl -fsSI --max-time 20 ${auth[@]+"${auth[@]}"} -H "Accept: ${TS_ACCEPT_MANIFESTS}" \
    "https://${registry}/v2/${name}/manifests/${tag}" 2>/dev/null) || return 1
  # Not -F': *': the value itself contains a colon (sha256:<hex>), so split at the first one only.
  printf '%s\n' "$headers" | tr -d '\r' | awk 'tolower($0) ~ /^docker-content-digest:/ { sub(/^[^:]*:[ \t]*/, ""); print; exit }'
}

# Pull the tag names out of a {"tags":["a","b"]} or {"Tags":[...]} document on stdin.
ts_tags_from_json() {
  tr -d '\n' | grep -io '"tags"[[:space:]]*:[[:space:]]*\[[^]]*\]' | sed 's/^[^[]*\[//; s/\]$//' | tr ',' '\n' | tr -d '" '
}

ts_curl_tags() {
  local registry=$1 name=$2 token body
  local -a auth=()
  token=$(ts_curl_token "$registry" "$name")
  [ -n "$token" ] && auth=(-H "Authorization: Bearer $token")
  # One page of up to 1000 tags is enough: registries list tags in lexical order, so release numbers
  # (digits) come before latest, master and sha-<commit>, which are the tags that pile up.
  body=$(curl -fsS --max-time 20 ${auth[@]+"${auth[@]}"} "https://${registry}/v2/${name}/tags/list?n=1000" 2>/dev/null) || return 1
  printf '%s' "$body" | ts_tags_from_json
}

ts_registry_methods() {
  case "${TS_REGISTRY_METHOD:-auto}" in
    auto) echo "buildx skopeo curl" ;;
    *) echo "$TS_REGISTRY_METHOD" ;;
  esac
}

# ts_registry_digest REF -> manifest digest of REF on the registry (sha256:...). Fails when no
# method can answer, which callers report as "could not check" rather than "up to date".
ts_registry_digest() {
  local ref=$1 method digest=""
  ts_split_ref "$ref"
  for method in $(ts_registry_methods); do
    case "$method" in
      buildx)
        ts_have docker || continue
        digest=$(docker buildx imagetools inspect "$ref" 2>/dev/null | awk '/^Digest:/ { print $2; exit }')
        ;;
      skopeo)
        ts_have skopeo || continue
        digest=$(skopeo inspect --format '{{.Digest}}' "docker://${ref}" 2>/dev/null)
        ;;
      curl)
        ts_have curl || continue
        digest=$(ts_curl_digest "$TS_REF_REGISTRY" "$TS_REF_NAME" "$TS_REF_TAG")
        ;;
    esac
    if [[ "$digest" == sha256:* ]]; then
      printf '%s\n' "$digest"
      return 0
    fi
  done
  return 1
}

# ts_registry_tags REPO_REF -> every tag of a repository, one per line (REPO_REF has no tag).
ts_registry_tags() {
  local ref=$1 method tags=""
  ts_split_ref "$ref"
  for method in $(ts_registry_methods); do
    case "$method" in
      skopeo)
        ts_have skopeo || continue
        tags=$(skopeo list-tags "docker://${TS_REF_REGISTRY}/${TS_REF_NAME}" 2>/dev/null | ts_tags_from_json)
        ;;
      curl)
        ts_have curl || continue
        tags=$(ts_curl_tags "$TS_REF_REGISTRY" "$TS_REF_NAME")
        ;;
      *) continue ;;
    esac
    if [ -n "$tags" ]; then
      printf '%s\n' "$tags"
      return 0
    fi
  done
  return 1
}

# ---------------------------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------------------------

# ts_sum A B ... -> decimal sum with one place after the point.
ts_sum() {
  printf '%s\n' "$@" | awk '{ s += $1 } END { printf "%.1f\n", s }'
}

# ts_ge A B -> success when decimal A >= decimal B.
ts_ge() {
  awk -v a="$1" -v b="$2" 'BEGIN { exit !(a + 0 >= b + 0) }'
}

ts_docker_usable() {
  ts_have docker && docker info >/dev/null 2>&1
}

TS_COLOR=""
if [ -t 1 ] && [ -z "${NO_COLOR:-}" ]; then
  TS_COLOR=1
fi

ts_paint() { # ts_paint CODE TEXT
  if [ -n "$TS_COLOR" ]; then
    printf '\033[%sm%s\033[0m' "$1" "$2"
  else
    printf '%s' "$2"
  fi
}
