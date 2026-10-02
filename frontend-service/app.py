"""Frontend service for the TTS-STT platform.

Serves the web UI, static assets, and API documentation pages.
Acts as a gateway that provides browser-facing URLs for all backend services.
"""

from fastapi import FastAPI, HTTPException, Path as PathParam, Request, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, Response, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exception_handlers import http_exception_handler, request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from starlette.background import BackgroundTask
from starlette.datastructures import Headers, MutableHeaders
from starlette.exceptions import HTTPException as StarletteHTTPException

import openai_router
import settings_schema
import settings_store
from openai_router import (
    UpstreamUnavailable,
    build_router as build_openai_router,
    http_exception_response as openai_http_exception_response,
    is_v1_path,
    openai_error,
    validation_error_response as openai_validation_error_response,
)
from pydantic import BaseModel, Field, field_validator
from pydantic_core import PydanticCustomError
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Any, Mapping, Optional
from urllib.parse import urlsplit
import asyncio
import base64
import binascii
import collections
import copy
import dataclasses
import hashlib
import hmac
import html
import httpx
import ipaddress
import json
import logging
import math
import os
import re
import time
import uuid
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def _asset_version() -> str:
    """Cache-busting token derived from the static assets themselves.

    It has to satisfy two things at once:
    - identical across uvicorn workers, or they hand out different asset URLs
      for the same files and thrash the browser cache;
    - different whenever the assets actually change, or browsers keep serving a
      stale app.js after an update.

    A restart timestamp fails the first, and a pinned image tag fails the second
    (deployments on `:latest` would never bust). Hashing the files satisfies both.
    """
    explicit = os.getenv("APP_VERSION")
    if explicit:
        return explicit

    digest = hashlib.sha256()
    static_dir = Path(__file__).parent / "static"
    try:
        for path in sorted(static_dir.rglob("*")):
            if path.is_file():
                digest.update(path.name.encode())
                digest.update(str(path.stat().st_mtime_ns).encode())
                digest.update(str(path.stat().st_size).encode())
    except OSError:
        # Fall back to a per-process token rather than failing to start.
        return str(int(time.time()))
    return digest.hexdigest()[:12]


APP_VERSION = _asset_version()


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    """Close the shared pooled HTTP client when the service stops."""
    yield
    global _http_client
    client, _http_client = _http_client, None
    aclose = getattr(client, "aclose", None) if client is not None else None
    if aclose is not None:
        try:
            await aclose()
        except Exception:
            pass


app = FastAPI(title="TTS-STT Frontend Service", version="2.0.0", lifespan=_lifespan)


# --- request hardening --------------------------------------------------------
#
# This service is the only one a browser talks to, and it can delete trained
# models, unload backends and start multi-hour training jobs. With the old
# default (`ALLOWED_ORIGINS=*`, every method allowed) any web page open in a
# LAN browser could drive all of that: a preflight from an arbitrary origin for
# DELETE was answered 200 with `Access-Control-Allow-Origin: *`.
#
# So the default is now same-origin only. Cross-origin access is opt-in, either
# for browsers (ALLOWED_ORIGINS) or for hosts that sit in front of the UI under a
# different name (TRUSTED_ORIGINS).
#
# Every setting below is read from the environment (the app YAML) once, here. The
# Settings page can override the ones in settings_schema.PREFERENCE_KEYS: the
# "live settings" section at the end of this module overlays the saved values and
# sets the module attributes (MAX_UPLOAD_MB, TRUST_PROXY_HEADERS, ENABLE_*, ...)
# and the access policy from the result, on the next request in every worker.
# With nothing saved they keep exactly the values read here.

_TRUTHY = frozenset({"1", "true", "yes", "on"})


def _env_flag(name: str, default: bool) -> bool:
    """An on/off switch from the environment; unset or empty means `default`."""
    raw = os.getenv(name, "").strip().lower()
    return default if not raw else raw in _TRUTHY


def _env_number(name: str, default, cast):
    """A positive number from the environment, or `default` when unset or unusable."""
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = cast(raw)
    except ValueError:
        logger.warning("%s=%r is not a number; using %s", name, raw, default)
        return default
    if not 0 < value < float("inf"):        # also rejects nan, which compares false to everything
        logger.warning("%s=%r must be a positive number; using %s", name, raw, default)
        return default
    return value


def _normalize_origin(origin: str) -> str:
    return origin.strip().rstrip("/").lower()


def _split_origins(raw: str) -> list[str]:
    return [_normalize_origin(o) for o in (raw or "").split(",") if o.strip()]


# Empty (or unset) means "no CORS at all", NOT "*": the old code turned an empty
# value into a wildcard, so the one setting that should have meant "closed" opened
# everything. A '*' can only come from here, never from the Settings page.
_ENV_ALLOWED_ORIGINS = tuple(_split_origins(os.getenv("ALLOWED_ORIGINS", "")))
# Only matters for cookies, which the app sets none of; always off next to a '*'.
ALLOW_CREDENTIALS = _env_flag("ALLOW_CREDENTIALS", False)
if "*" in _ENV_ALLOWED_ORIGINS:
    logger.warning(
        "ALLOWED_ORIGINS contains '*': any web page open in a browser that can reach this "
        "service may call it, including the delete/unload/training endpoints. Leave it empty "
        "for same-origin only, or list the origins that need access.")

# Extra origins that are this UI under another name (a reverse proxy that does
# not preserve Host). They pass the state-changing Origin check but get no CORS
# headers: a page served from them is same-origin from the browser's point of
# view, so none are needed.
_ENV_TRUSTED_ORIGINS = tuple(dict.fromkeys(_split_origins(os.getenv("TRUSTED_ORIGINS", ""))))

# X-Forwarded-Host is only believed when a proxy we control sets it; on an
# exposed port anyone can send it, which would let a forged request name itself
# same-origin. (The guard reads the access policy; this mirrors its value.)
TRUST_PROXY_HEADERS = _env_flag("TRUST_PROXY_HEADERS", False)

# Optional shared secret, "the deployment key". Unset keeps the service open on
# the LAN as before (unless the Settings page requires a key). Set, it is
# required on every /v1 call and on every state-changing /api call, from the
# bundled UI as well (which asks for it once per browser tab, see app.js). It is
# only ever compared as a SHA-256 digest, next to the keys made in the Settings
# page, and the page can neither change nor remove it.
API_KEY = os.getenv("API_KEY", "").strip()

# Upload cap for anything that is not JSON or the OpenAI transcription route.
# The largest legitimate body is the training-audio upload (`audio_files`, many
# WAVs at once); 512 MB is ~48 minutes of 44.1 kHz mono 16-bit, which is well
# past the "10+ minutes" the UI recommends. Every byte of it is held in RAM
# while it is forwarded, so this is also the worst-case memory cost of one
# request.
MAX_UPLOAD_MB = _env_number("MAX_UPLOAD_MB", 512.0, float)
MAX_REQUEST_BYTES = int(MAX_UPLOAD_MB * 1024 * 1024)
# Nothing in the JSON API is anywhere near this; it only bounds what a hostile
# caller can make the gateway buffer.
MAX_JSON_BODY_BYTES = 1024 * 1024
# multipart framing and the non-file fields around the 25 MB file itself
_MULTIPART_SLACK_BYTES = 1024 * 1024

# How many uploads one worker process forwards at the same time. Each holds its
# whole body in RAM until the backend has answered, so without a cap the memory
# bound is "MAX_UPLOAD_MB times however many clients connect". Requests over the
# cap are answered 503 + Retry-After without reading their body.
MAX_CONCURRENT_UPLOADS = _env_number("MAX_CONCURRENT_UPLOADS", 4, int)
UPLOAD_RETRY_AFTER_S = 5

# Longest text a TTS request may carry (Piper and Qwen3 both synthesise a whole
# request in one go; an unbounded string is an unbounded GPU job). The default is
# what the smallest backend accepts (chatterbox, magpie-tts and qwen3-tts: MAX_TEXT_CHARS=5000);
# a larger value only moves the failure from an early, clear 422 here to a 413
# from the backend after the request has already been queued. Keep it <= the
# smallest MAX_TEXT_CHARS of the TTS backends in use. Checked per request (see
# FrontendTTSRequest), so a value saved in the Settings page applies at once.
MAX_TTS_CHARS = _env_number("MAX_TTS_CHARS", 5000, int)

# Which optional engines the UI and the API offer. Offering one does not start
# it: its container is installed separately (a `profiles:` line in the YAML).
ENABLE_WHISPER_CPP = _env_flag("ENABLE_WHISPER_CPP", False)
ENABLE_PARAKEET_ASR = _env_flag("ENABLE_PARAKEET_ASR", False)
ENABLE_CANARY_ASR = _env_flag("ENABLE_CANARY_ASR", False)
ENABLE_CHATTERBOX_TTS = _env_flag("ENABLE_CHATTERBOX_TTS", False)
ENABLE_MAGPIE_TTS = _env_flag("ENABLE_MAGPIE_TTS", False)
# Voice training is on unless switched off: installs without the training
# container (the TrueNAS app, the RK3588) would otherwise show a permanently red
# status dot and a tab that cannot work.
ENABLE_TRAINING = _env_flag("ENABLE_TRAINING", True)
_TRAINING_PROVIDER = os.getenv("TRAINING_PROVIDER", "piper-training")

# The Settings page's own switches; only the app YAML holds them. Off (or a
# SAFE-MODE file in the settings folder) means the saved gateway.json is ignored.
# The keys in keys.json stay enforced either way, so recovery never opens the API.
ENABLE_SETTINGS_UI = _env_flag("ENABLE_SETTINGS_UI", True)
# Keys whose app YAML value wins over a saved one (the page shows them read-only).
SETTINGS_LOCKED_KEYS, _unknown_locked_keys = settings_schema.parse_locked_keys(
    os.getenv("SETTINGS_LOCKED_KEYS", ""))
if _unknown_locked_keys:
    logger.warning(
        "SETTINGS_LOCKED_KEYS names %s, which the Settings page does not have; ignored.",
        ", ".join(repr(name[:64]) for name in _unknown_locked_keys[:10]))

_SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})


def _default_port(scheme: str) -> int:
    return 443 if scheme in ("https", "wss") else 80


def _origin_key(origin: str) -> Optional[tuple[str, int]]:
    """(host, port) of an Origin value, or None for `null` and anything malformed."""
    try:
        parts = urlsplit(origin.strip())
        if parts.scheme not in ("http", "https", "ws", "wss") or not parts.hostname:
            return None
        return parts.hostname.lower(), parts.port or _default_port(parts.scheme)
    except ValueError:
        return None


def _host_key(host: str, scheme: str) -> Optional[tuple[str, int]]:
    """(host, port) of a Host header; a missing port means the scheme's default."""
    try:
        parts = urlsplit(f"//{host.strip()}")
        if not parts.hostname:
            return None
        return parts.hostname.lower(), parts.port or _default_port(scheme)
    except ValueError:
        return None


# The functions below read the access policy in force (`_policy`, swapped as a
# whole when a setting changes) unless they are handed one: the guard passes the
# one it started with, and the Settings page's lockout check a candidate.


def _request_host(headers: Headers, policy: Optional["_AccessPolicy"] = None) -> Optional[str]:
    policy = _policy if policy is None else policy
    if policy.trust_proxy_headers:
        forwarded = headers.get("x-forwarded-host")
        if forwarded:
            return forwarded.split(",")[0]
    return headers.get("host")


def _is_same_origin(origin: str, headers: Headers, policy: Optional["_AccessPolicy"] = None) -> bool:
    """Does this Origin name the host the request was addressed to?"""
    policy = _policy if policy is None else policy
    key = _origin_key(origin)
    if key is None:
        return False
    if _normalize_origin(origin) in policy.trusted_origins:
        return True
    host = _request_host(headers, policy)
    if not host:
        return False
    return _host_key(host, urlsplit(origin.strip()).scheme) == key


def _origin_permitted(origin: str, headers: Headers, policy: Optional["_AccessPolicy"] = None) -> bool:
    """Same origin as this service, or one the operator explicitly allowed."""
    policy = _policy if policy is None else policy
    allowed = policy.allowed_origins
    if "*" in allowed or _normalize_origin(origin) in allowed:
        return True
    return _is_same_origin(origin, headers, policy)


# --- host validation (DNS rebinding) --------------------------------------------
#
# The Origin check above compares `Origin` with `Host`, and under DNS rebinding
# both come from the attacker: a page on evil.example whose name is re-pointed at
# this service's LAN address sends `Host: evil.example:3000` together with
# `Origin: http://evil.example:3000`, which match, and the browser even labels the
# request `Sec-Fetch-Site: same-origin`. So the Host itself has to be one that
# names this service. A rebound name is always a public DNS name, and public DNS
# names have a dot and are not on the list below, so they are refused.
#
# Accepted without configuration (none of these can be rebound by a third party):
#   - any IP literal: the browser does no DNS lookup, so there is nothing to rebind;
#   - `localhost`, `*.localhost`, `*.local` (mDNS), `*.localdomain`, `*.lan`,
#     `*.internal` and `*.home.arpa`: not resolvable through public DNS;
#   - single-label names (`truenas`, `frontend-service`): a public name always has
#     a dot, and these are what a Docker network or a LAN search domain provides.
# Anything else (a real domain in front of a reverse proxy, a Tailscale MagicDNS
# name) is listed in TRUSTED_HOSTS, or comes in through TRUSTED_ORIGINS.

_LOCAL_HOST_SUFFIXES = (".local", ".localhost", ".localdomain", ".lan", ".internal", ".home.arpa")
_HOST_PATTERN = re.compile(
    r"^(?:\[(?P<v6>[0-9A-Fa-f:.]+(?:%[0-9A-Za-z._~-]+)?)\]|(?P<name>[A-Za-z0-9._-]+))"
    r"(?::(?P<port>[0-9]{1,5}))?$"
)


def _parse_host_header(value: str) -> Optional[str]:
    """The lower-cased hostname of a Host header, or None when it is not a plain host[:port].

    Strict on purpose: userinfo, paths, backslashes and whitespace never occur in
    a genuine Host header, and a lenient URL parser would read some of them
    differently from the browser that (did not) send them.
    """
    match = _HOST_PATTERN.match(value.strip())
    if not match:
        return None
    host = (match.group("v6") or match.group("name")).lower().rstrip(".")
    return host or None


def _normalize_host_entry(entry: str) -> Optional[str]:
    """A TRUSTED_HOSTS entry as a bare lower-case host, keeping a leading `*.`."""
    text = entry.strip().lower()
    if not text:
        return None
    wildcard = text.startswith("*.")
    if wildcard:
        text = text[2:]
    parsed = _parse_host_header(urlsplit(text).netloc if "://" in text else text.split("/")[0])
    if not parsed:
        logger.warning("TRUSTED_HOSTS entry %r is not a hostname; ignored", entry)
        return None
    return f"*.{parsed}" if wildcard else parsed


def _split_hosts(raw: str) -> list[str]:
    return [h for h in (_normalize_host_entry(e) for e in (raw or "").split(",")) if h]


# `*` here (and only here) switches the check off, for setups where every
# hostname reaches the service anyway. It is not a list: names go in TRUSTED_HOSTS.
_allowed_hosts_setting = [e.strip() for e in os.getenv("ALLOWED_HOSTS", "").split(",") if e.strip()]
ALLOW_ANY_HOST = "*" in _allowed_hosts_setting
if _allowed_hosts_setting and not ALLOW_ANY_HOST:
    logger.warning(
        "ALLOWED_HOSTS only understands '*' (turn the Host check off); list hostnames in "
        "TRUSTED_HOSTS instead. Ignoring ALLOWED_HOSTS=%r.", ",".join(_allowed_hosts_setting))
if ALLOW_ANY_HOST:
    logger.warning(
        "ALLOWED_HOSTS='*': the Host header is not validated, so a web page that re-points "
        "its own DNS name at this service (DNS rebinding) is treated as same-origin. Prefer "
        "listing the names you use in TRUSTED_HOSTS.")

_ENV_TRUSTED_HOSTS = tuple(dict.fromkeys(_split_hosts(os.getenv("TRUSTED_HOSTS", ""))))


def _host_allowed(host_header: Optional[str], policy: Optional["_AccessPolicy"] = None) -> bool:
    """May a request addressed to this Host reach the service at all?"""
    policy = _policy if policy is None else policy
    if policy.allow_any_host or not host_header:
        # No Host header at all is an HTTP/1.0 client, never a rebinding browser.
        return True
    host = _parse_host_header(host_header)
    if host is None:
        return False
    try:
        ipaddress.ip_address(host.split("%", 1)[0])
        return True
    except ValueError:
        pass
    if "." not in host:
        return True
    return (
        host in policy.trusted_host_names
        or host.endswith(_LOCAL_HOST_SUFFIXES)
        or host.endswith(policy.trusted_host_suffixes)
    )


# --- the refusal of a host name ---------------------------------------------------
#
# What a person who opened the UI under a new name needs to read: the IP address
# always passes the check (above), and from there the name can be added in the
# Settings page. API clients keep the JSON refusal; a browser navigation gets the
# same text as a small HTML page. That page has no script and no "trust this name"
# button: under DNS rebinding the refused name is the attacker's choice, and one
# click must never be enough to trust it.


def _settings_can_trust_hosts() -> bool:
    """Can the Settings page add a host name (rather than only the app YAML)?"""
    return ENABLE_SETTINGS_UI and "TRUSTED_HOSTS" not in SETTINGS_LOCKED_KEYS


def _host_refusal_message(host: Optional[str]) -> str:
    shown = f"Host {(host or '')[:100]!r} is not allowed. If this is the name you use to reach this service, "
    if _settings_can_trust_hosts():
        return shown + ("open it by its IP address and add the name under Settings -> Access (TRUSTED_HOSTS), "
                        "or add it to TRUSTED_HOSTS in the app YAML.")
    return shown + "add its hostname to TRUSTED_HOSTS in the app YAML."


def _is_navigation(method: str, path: str, headers: Headers) -> bool:
    """Is this a browser loading a page (rather than a script calling the API)?

    Sec-Fetch-Mode says so where browsers send it (secure contexts only); over plain
    http on a LAN they do not, and the Accept header of a navigation names text/html
    while fetch() asks for */*.
    """
    if method not in ("GET", "HEAD") or is_v1_path(path) or path == "/api" or path.startswith("/api/"):
        return False
    mode = headers.get("sec-fetch-mode")
    if mode is not None:
        return mode.strip().lower() == "navigate"
    return "text/html" in headers.get("accept", "").lower()


# Nothing on the refusal page may load or run anything.
_REFUSAL_PAGE_CSP = "default-src 'none'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'"


def _host_refusal_page(host: Optional[str]) -> HTMLResponse:
    """The host refusal for a browser: the name, escaped, and how to get it accepted."""
    raw = (host or "")[:100]
    match = _HOST_PATTERN.match(raw.strip())
    port = match.group("port") if match and match.group("port") else "3000"
    shown, address = html.escape(raw, quote=True), html.escape(f"http://<IP address>:{port}/settings", quote=True)
    if _settings_can_trust_hosts():
        steps = (
            "<ol>\n"
            f"<li>Open this server by its IP address instead, for example <code>{address}</code>. "
            "IP addresses always work.</li>\n"
            "<li>In Settings, under Access, add the name to <em>Host names of this server</em> and save.</li>\n"
            "</ol>\n"
            "<p>Or add the name to <code>TRUSTED_HOSTS</code> in the app YAML.</p>\n")
    else:
        steps = "<p>Add the name to <code>TRUSTED_HOSTS</code> in the app YAML.</p>\n"
    body = (
        "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
        "<title>Host name not allowed</title>\n</head>\n<body>\n"
        "<h1>This host name is not allowed</h1>\n"
        f"<p>This server was opened as <strong>{shown}</strong>, a name it does not know as its own, so it "
        "refuses the request. This protects it against DNS rebinding.</p>\n"
        "<p>If this is the name you use for this server:</p>\n"
        f"{steps}</body>\n</html>\n")
    return HTMLResponse(body, status_code=403, headers={
        "Content-Security-Policy": _REFUSAL_PAGE_CSP, "Cache-Control": "no-store"})


# --- the access policy and the keys ----------------------------------------------
#
# Everything the guard decides on, built from the effective settings and replaced
# as a whole (`_policy`, `_keyring`) when one of them changes. A change is applied
# without an `await` in between, so no request ever sees half of one.


@dataclass(frozen=True)
class _AccessPolicy:
    allowed_origins: tuple[str, ...]          # CORS and the Origin check; '*' only from the app YAML
    allow_credentials: bool
    trusted_origins: frozenset                # this UI under another name (normalized origins)
    trust_proxy_headers: bool                 # believe X-Forwarded-Host
    trusted_host_names: frozenset
    trusted_host_suffixes: tuple[str, ...]    # ".example.com" for "*.example.com"
    allow_any_host: bool                      # ALLOWED_HOSTS='*' (the app YAML only)
    require_key: bool                         # /v1, mutating /api and /ws/stt need a key


def _build_access_policy(values: Mapping[str, Any]) -> _AccessPolicy:
    """The policy for a set of effective values (settings_schema's keys)."""
    allowed = tuple(dict.fromkeys(_normalize_origin(o) for o in values["ALLOWED_ORIGINS"] if o.strip()))
    trusted_origins = frozenset(_normalize_origin(o) for o in values["TRUSTED_ORIGINS"] if o.strip())
    hosts = {h.strip().lower() for h in values["TRUSTED_HOSTS"] if h.strip()}
    # A TRUSTED_ORIGINS entry says "this UI is also reachable as https://tts.example.com",
    # which is also a statement about the Host it arrives with.
    for origin in trusted_origins:
        key = _origin_key(origin)
        if key:
            hosts.add(key[0])
    return _AccessPolicy(
        allowed_origins=allowed,
        allow_credentials=ALLOW_CREDENTIALS and "*" not in allowed,
        trusted_origins=trusted_origins,
        trust_proxy_headers=bool(values["TRUST_PROXY_HEADERS"]),
        trusted_host_names=frozenset(h for h in hosts if not h.startswith("*.")),
        trusted_host_suffixes=tuple(sorted(f".{h[2:]}" for h in hosts if h.startswith("*."))),
        allow_any_host=ALLOW_ANY_HOST,
        require_key=bool(values["require_key"]),
    )


@dataclass(frozen=True)
class _Credential:
    """Whose key a request presented: the app YAML's API_KEY, or a key made in the Settings page."""

    name: str                       # settings_store.DEPLOYMENT_KEY, or the page key's name
    role: str                       # "admin" (API and settings) or "client" (API only)
    key_id: Optional[str] = None    # a page key's id; None for the deployment key


def _key_digest(key: str) -> bytes:
    return hashlib.sha256(key.encode("utf-8")).digest()


@dataclass(frozen=True)
class _KeyRing:
    """The SHA-256 digest of every accepted key, and whose key it is."""

    entries: tuple[tuple[bytes, _Credential], ...] = ()

    def match(self, candidate: str) -> Optional[_Credential]:
        # Digests of one length, compared in constant time and against every entry:
        # neither the time taken nor an early exit tells a guess how close it came.
        # (Comparing the plain strings, as this used to, leaked the key's length.)
        digest = _key_digest(candidate)
        found = None
        for expected, credential in self.entries:
            if hmac.compare_digest(digest, expected) and found is None:
                found = credential
        return found


_API_KEY_DIGEST = _key_digest(API_KEY) if API_KEY else None


def _build_key_ring(page_keys, deployment_role: str) -> _KeyRing:
    """The deployment key (when the YAML sets one) in `deployment_role`, then the page keys."""
    entries = []
    if _API_KEY_DIGEST is not None:
        entries.append((_API_KEY_DIGEST, _Credential(settings_store.DEPLOYMENT_KEY, deployment_role)))
    for record in page_keys:
        entries.append((bytes.fromhex(record.sha256), _Credential(record.name, record.role, record.id)))
    return _KeyRing(tuple(entries))


def _key_matches(candidate: str) -> Optional[_Credential]:
    """Whose key `candidate` is (truthy), or None."""
    return _keyring.match(candidate)


def _bearer_credential(headers: Headers) -> Optional[_Credential]:
    """The credential an `Authorization: Bearer <key>` header carries, or None."""
    scheme, _, token = headers.get("authorization", "").partition(" ")
    if scheme.lower() != "bearer":
        return None
    return _key_matches(token.strip())


def _bearer_key_ok(headers: Headers) -> bool:
    return _bearer_credential(headers) is not None


# The app YAML's value of every setting the Settings page has, in the page's
# representation (lists as tuples): what applies while nothing is saved, and what
# "Reset to YAML" goes back to.
_ENV_VALUES: Mapping[str, Any] = MappingProxyType({
    "TRUSTED_HOSTS": _ENV_TRUSTED_HOSTS,
    "TRUSTED_ORIGINS": _ENV_TRUSTED_ORIGINS,
    "TRUST_PROXY_HEADERS": TRUST_PROXY_HEADERS,
    "ALLOWED_ORIGINS": _ENV_ALLOWED_ORIGINS,
    "ENABLE_CANARY_ASR": ENABLE_CANARY_ASR,
    "ENABLE_PARAKEET_ASR": ENABLE_PARAKEET_ASR,
    "ENABLE_CHATTERBOX_TTS": ENABLE_CHATTERBOX_TTS,
    "ENABLE_MAGPIE_TTS": ENABLE_MAGPIE_TTS,
    "ENABLE_WHISPER_CPP": ENABLE_WHISPER_CPP,
    "ENABLE_TRAINING": ENABLE_TRAINING,
    "DEFAULT_TTS_PROVIDER": os.getenv("DEFAULT_TTS_PROVIDER", "piper"),
    "DEFAULT_STT_PROVIDER": os.getenv("DEFAULT_STT_PROVIDER", "whisper"),
    "MAX_UPLOAD_MB": MAX_UPLOAD_MB,
    "MAX_TTS_CHARS": MAX_TTS_CHARS,
    "MAX_CONCURRENT_UPLOADS": MAX_CONCURRENT_UPLOADS,
    # openai_router reads this one itself (once, at its import) for its ffmpeg slots.
    "MAX_CONCURRENT_FFMPEG": openai_router.MAX_CONCURRENT_FFMPEG,
})
# The settings the environment actually names (the rest are code defaults).
_ENV_SET = frozenset(key for key in settings_schema.PREFERENCE_KEYS if os.getenv(key, "").strip())

# Until the live-settings section at the end of this module overlays what the
# Settings page saved, this is exactly the app YAML's policy.
_policy = _build_access_policy({**_ENV_VALUES, "require_key": bool(API_KEY)})
_keyring = _build_key_ring((), settings_schema.ROLE_ADMIN)


def _guard_verdict(method: str, path: str, headers: Headers) -> Optional[tuple[int, str, str]]:
    """(status, message, code) if the request must be refused, else None."""
    if method == "OPTIONS":
        # A CORS preflight: carries no credentials and changes nothing.
        return None

    policy = _policy
    host = _request_host(headers, policy)
    if not _host_allowed(host, policy):
        return 403, _host_refusal_message(host), "host_not_allowed"

    if method not in _SAFE_METHODS:
        origin = headers.get("origin")
        if origin is not None and not _origin_permitted(origin, headers, policy):
            return (
                403,
                f"Cross-origin request from {origin!r} refused. Add it to ALLOWED_ORIGINS "
                "(or TRUSTED_ORIGINS if it is this UI behind a proxy) to allow it.",
                "cross_origin_blocked",
            )

    if policy.require_key:
        # No exemption for "the request looks like it came from the UI": Origin,
        # Host and Sec-Fetch-Site are all under a rebinding page's control, so
        # none of them says who is calling. The UI sends the key like any client.
        needs_key = is_v1_path(path) or (path.startswith("/api/") and method not in _SAFE_METHODS)
        if needs_key and not _bearer_key_ok(headers):
            return 401, "A valid API key is required (Authorization: Bearer <key>).", "invalid_api_key"
    return None


# Browsers cannot put a header on a WebSocket handshake, so the key travels as a
# subprotocol next to the real one: `new WebSocket(url, ["tts-stt.v1", "bearer.<b64url(key)>"])`.
# base64url keeps it inside the token alphabet subprotocol names must use. Scripts
# can use `Authorization: Bearer` on the upgrade request instead. The real
# protocol name (`tts-stt.v1`) is whatever the client offers first; the server
# only has to echo one back that is not the credential.
_WS_KEY_PROTOCOL_PREFIX = "bearer."


def _websocket_key_ok(websocket: WebSocket) -> bool:
    if _bearer_key_ok(websocket.headers):
        return True
    for protocol in websocket.scope.get("subprotocols") or []:
        if not protocol.startswith(_WS_KEY_PROTOCOL_PREFIX):
            continue
        encoded = protocol[len(_WS_KEY_PROTOCOL_PREFIX):]
        try:
            candidate = base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4)).decode("utf-8")
        except (binascii.Error, ValueError):
            continue
        if _key_matches(candidate):
            return True
    return False


def _websocket_subprotocol(websocket: WebSocket) -> Optional[str]:
    """The subprotocol to echo: a browser fails the handshake if it offered some and none is chosen.

    Never the `bearer.` one, which is the credential and must not be reflected.
    """
    for protocol in websocket.scope.get("subprotocols") or []:
        if not protocol.startswith(_WS_KEY_PROTOCOL_PREFIX):
            return protocol
    return None


def _refusal(path: str, status: int, message: str, code: str) -> JSONResponse:
    """A refusal in the shape the caller expects: OpenAI envelope on /v1, `detail` elsewhere.

    The Settings API's refusals also carry the `code`, which its page switches on.
    """
    if is_v1_path(path):
        response = openai_error(status, message, code=code)
    elif _is_settings_path(path):
        response = JSONResponse(status_code=status, content={"detail": message, "code": code})
    else:
        response = JSONResponse(status_code=status, content={"detail": message})
    if status == 401:
        response.headers["WWW-Authenticate"] = "Bearer"
    return response


def _guard_response(scope) -> Optional[Response]:
    """The refusal for a request the guard stops, or None to let it through."""
    method, path = scope["method"], scope["path"]
    headers = Headers(scope=scope)
    if _is_settings_path(path):
        return _settings_guard(method, path, headers, _client_address(scope))
    verdict = _guard_verdict(method, path, headers)
    if verdict is None:
        return None
    if verdict[2] == "host_not_allowed" and _is_navigation(method, path, headers):
        return _host_refusal_page(_request_host(headers))
    return _refusal(path, *verdict)


class _RequestGuardMiddleware:
    """Cross-origin and API-key checks. Pure ASGI: no body buffering, no task per request."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            refusal = _guard_response(scope)
            if refusal is not None:
                await refusal(scope, receive, send)
                return
        await self.app(scope, receive, send)


class _SecurityHeadersMiddleware:
    """nosniff / referrer / framing headers on every response.

    There is deliberately no Content-Security-Policy: the template still uses
    inline event handlers, and a policy that forbids them would break the UI.
    """

    _HEADERS = (
        ("X-Content-Type-Options", "nosniff"),
        ("Referrer-Policy", "same-origin"),
        ("X-Frame-Options", "SAMEORIGIN"),
    )

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_with_headers(message):
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                for name, value in self._HEADERS:
                    headers.setdefault(name, value)
            await send(message)

        await self.app(scope, receive, send_with_headers)


class _BodyTooLarge(Exception):
    """Raised into the app from `receive()` once the body passes its limit."""


def _body_limit_for(path: str, content_type: str) -> int:
    if _is_settings_path(path):
        # Small JSON documents only; never a setting of its own, so no saved
        # value can lock the Settings page out of its own API.
        return SETTINGS_BODY_LIMIT
    if path == "/v1/audio/transcriptions":
        # The route enforces the exact 25 MB on the file and answers in the
        # OpenAI shape; this is only the backstop for the framing around it.
        return min(MAX_REQUEST_BYTES, openai_router.MAX_UPLOAD_BYTES + _MULTIPART_SLACK_BYTES)
    if content_type.split(";")[0].strip().lower() == "application/json":
        return MAX_JSON_BODY_BYTES
    return MAX_REQUEST_BYTES


class _BodyLimitMiddleware:
    """Refuse oversized request bodies with a 413 before the app buffers them.

    Starlette spools a multipart upload and the handlers then read it whole into
    memory, and uvicorn has no body limit of its own. Two checks, because a
    client controls both: `Content-Length` rejects an honest oversized request
    without reading a byte, and a running count catches chunked uploads and
    understated lengths.

    Once the cap is passed the response is owned here: the app is fed an error
    from `receive()`, and whatever it answers with (FastAPI turns a parse
    failure into a 400, `create_speech` catches everything) is replaced by the
    413, so no handler can swallow it.
    """

    def __init__(self, app):
        self.app = app

    @staticmethod
    async def _reject(scope, receive, send, limit: int):
        size = f"{limit // 1024} KiB" if limit < 1024 * 1024 else f"{limit / (1024 * 1024):g} MB"
        message = f"Request body too large. Maximum is {size}."
        response = _refusal(scope["path"], 413, message, "request_too_large")
        # The unread remainder of the body is still on the wire.
        response.headers["Connection"] = "close"
        await response(scope, receive, send)

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] in _SAFE_METHODS:
            await self.app(scope, receive, send)
            return

        headers = Headers(scope=scope)
        limit = _body_limit_for(scope["path"], headers.get("content-type", ""))
        try:
            declared = int(headers.get("content-length", ""))
        except ValueError:
            declared = None
        if declared is not None and declared > limit:
            await self._reject(scope, receive, send, limit)
            return

        received = 0
        exceeded = False
        started = False

        async def limited_receive():
            nonlocal received, exceeded
            if exceeded:
                raise _BodyTooLarge()
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > limit:
                    exceeded = True
                    raise _BodyTooLarge()
            return message

        async def guarded_send(message):
            nonlocal started
            if exceeded and not started:
                # Swallow the app's own answer; the 413 goes out once, below.
                return
            if message["type"] == "http.response.start":
                started = True
            await send(message)

        try:
            await self.app(scope, limited_receive, guarded_send)
        except _BodyTooLarge:
            pass
        if exceeded and not started:
            await self._reject(scope, receive, send, limit)


_upload_slots = openai_router.Slots(MAX_CONCURRENT_UPLOADS)


class _UploadSlotMiddleware:
    """Cap how many large request bodies one worker buffers at the same time.

    Only requests that *can* carry a large body count: JSON is capped at 1 MiB
    (`MAX_JSON_BODY_BYTES`) and never needs a slot, so a burst of small API
    calls cannot be locked out by a few big uploads. Everything else - multipart
    uploads, and any body whose Content-Type says it is not JSON, since FastAPI
    reads it before it decides it does not want it - takes a slot for as long as
    the request is being handled.

    Innermost of the middlewares, so a request the guard or the size limit
    refuses never occupies one.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] in _SAFE_METHODS:
            await self.app(scope, receive, send)
            return
        content_type = Headers(scope=scope).get("content-type", "")
        if _body_limit_for(scope["path"], content_type) <= MAX_JSON_BODY_BYTES:
            await self.app(scope, receive, send)
            return

        if not _upload_slots.try_acquire():
            response = _refusal(
                scope["path"], 503,
                f"The server is already handling {_upload_slots.limit} uploads. "
                f"Try again in {UPLOAD_RETRY_AFTER_S} seconds.",
                "server_busy",
            )
            response.headers["Retry-After"] = str(UPLOAD_RETRY_AFTER_S)
            # The body was never read; it is still on the wire.
            response.headers["Connection"] = "close"
            await response(scope, receive, send)
            return
        try:
            await self.app(scope, receive, send)
        finally:
            _upload_slots.release()


# --- the Settings page and its API: their own request rules ---------------------------
#
# /settings (the page) and /api/settings* (its API) decide who may use this server
# and how, so they get stricter rules than every other path:
#   - The Host check always applies, even with ALLOWED_HOSTS='*'.
#   - Once the server is claimed (an admin credential exists: the app YAML key in
#     role admin, or an admin key made in the page), every API request needs an
#     admin key, GET included: none or a wrong one is 401, a client key 403.
#     Unclaimed, the API can be read (the page shows the values read-only), and the
#     only write is the claim with a one-time code from the container log.
#   - 10 wrong keys or codes from one address within 10 minutes give 429, per worker.
#   - A changing request must come from this UI: an Origin, when sent, is this
#     server or a TRUSTED_ORIGINS entry (ALLOWED_ORIGINS does not count, `null` is
#     refused), Sec-Fetch-Site, when sent, is same-origin, and the body is JSON
#     (415 otherwise) of at most 64 KiB without a duplicate key.
#   - No CORS at all: a preflight gets 403, and no Access-Control-* header leaves.
#   - Nothing is cached (Cache-Control: no-store).

SETTINGS_BODY_LIMIT = 64 * 1024
SETTINGS_AUTH_FAILURES = 10
SETTINGS_AUTH_WINDOW_S = 600.0
# Reached without a key: the claim takes the one-time code instead, and anyone may
# ask for a code, because it only ever reaches the container log.
_SETTINGS_OPEN_POSTS = frozenset({"/api/settings/claim-code", "/api/settings/claim"})


def _is_settings_path(path: str) -> bool:
    """The Settings page and its API, which never get CORS headers."""
    return path in ("/settings", "/api/settings") or path.startswith(("/settings/", "/api/settings/"))


def _is_settings_api_path(path: str) -> bool:
    return path == "/api/settings" or path.startswith("/api/settings/")


def _client_address(scope) -> str:
    client = scope.get("client")
    return str(client[0]) if client else ""


class _FailureLimiter:
    """Wrong keys and codes per client address on the Settings API, in this worker.

    `limit` failures within `window` seconds refuse the address until the oldest of
    them is `window` old. A request with a working admin key clears its record.
    """

    def __init__(self, limit: int, window: float, *, clock=time.monotonic, max_clients: int = 4096):
        self.limit, self.window, self.clock, self.max_clients = limit, window, clock, max_clients
        self._failures: dict[str, collections.deque] = {}

    def _recent(self, client: str, now: float) -> Optional[collections.deque]:
        times = self._failures.get(client)
        if times is None:
            return None
        while times and now - times[0] >= self.window:
            times.popleft()
        if not times:
            del self._failures[client]
            return None
        return times

    def retry_after(self, client: str) -> Optional[int]:
        """Seconds until the address may try again, or None when it is not blocked."""
        now = self.clock()
        times = self._recent(client, now)
        if times is None or len(times) < self.limit:
            return None
        return max(1, math.ceil(times[0] + self.window - now))

    def failed(self, client: str) -> bool:
        """Count a failure; True when it is the one that blocks the address."""
        now = self.clock()
        times = self._recent(client, now)
        if times is None:
            if len(self._failures) >= self.max_clients:
                oldest = min(self._failures, key=lambda address: self._failures[address][-1])
                del self._failures[oldest]
            times = self._failures[client] = collections.deque(maxlen=self.limit)
        before = len(times)
        times.append(now)
        return before < self.limit <= len(times)

    def succeeded(self, client: str) -> None:
        self._failures.pop(client, None)


_settings_limiter = _FailureLimiter(SETTINGS_AUTH_FAILURES, SETTINGS_AUTH_WINDOW_S)


def _settings_error(status: int, message: str, code: str, *, headers: Optional[Mapping[str, str]] = None,
                    **extra: Any) -> JSONResponse:
    """A Settings API answer that is not a success: a message, a stable code and any details."""
    response = JSONResponse(status_code=status, content={"detail": message, "code": code, **extra},
                            headers=dict(headers or {}))
    if status == 401:
        response.headers["WWW-Authenticate"] = "Bearer"
    return response


def _presented_key(headers: Headers) -> bool:
    scheme, _, token = headers.get("authorization", "").partition(" ")
    return scheme.lower() == "bearer" and bool(token.strip())


def _audit_refusal(result: str, method: str, path: str, headers: Headers, client: str,
                   credential: Optional["_Credential"] = None) -> None:
    """An audit line for a refused Settings request (never raises)."""
    actor = settings_store.Actor(ip=client, host=headers.get("host", ""),
                                 credential=credential.name if credential else "")
    _settings_store.audit("refused", actor=actor, result=result, method=method, path=path[:200])


def _settings_guard(method: str, path: str, headers: Headers, client: str) -> Optional[Response]:
    """The rules above for one request to /settings or /api/settings*; None lets it through."""
    policy = _policy
    if policy.allow_any_host:
        policy = dataclasses.replace(policy, allow_any_host=False)
    host = _request_host(headers, policy)
    if not _host_allowed(host, policy):
        if _is_navigation(method, path, headers):
            return _host_refusal_page(host)
        return _refusal(path, 403, _host_refusal_message(host), "host_not_allowed")
    if method == "OPTIONS":
        return _settings_error(403, "The Settings page answers no cross-origin requests.", "cors_refused")
    if not _is_settings_api_path(path):
        return None                     # the page itself: an empty shell that asks the API
    retry_after = _settings_limiter.retry_after(client)
    if retry_after is not None:
        return _settings_error(429, "Too many wrong keys or codes from this address; try again later.",
                               "too_many_attempts", headers={"Retry-After": str(retry_after)})
    safe = method in _SAFE_METHODS
    if not safe:
        origin = headers.get("origin")
        if origin is not None and not _is_same_origin(origin, headers, policy):
            _audit_refusal("403 cross_origin_blocked", method, path, headers, client)
            return _settings_error(403, "Settings can only be changed from this server's own page.",
                                   "cross_origin_blocked")
        site = headers.get("sec-fetch-site")
        if site is not None and site.strip().lower() != "same-origin":
            _audit_refusal("403 cross_site_blocked", method, path, headers, client)
            return _settings_error(403, "Settings can only be changed from this server's own page.",
                                   "cross_site_blocked")
    if not (method == "POST" and path in _SETTINGS_OPEN_POSTS):
        if _settings_claimed():
            credential = _bearer_credential(headers)
            if credential is None:
                if _presented_key(headers):
                    blocked = _settings_limiter.failed(client)
                    _audit_refusal("401 invalid_api_key" + (", address blocked" if blocked else ""),
                                   method, path, headers, client)
                return _settings_error(401, "An admin key is required (Authorization: Bearer <key>).",
                                       "invalid_api_key")
            if credential.role != settings_schema.ROLE_ADMIN:
                _audit_refusal("403 admin_key_required", method, path, headers, client, credential)
                return _settings_error(403, "This key may use the API but not the Settings; an admin key "
                                            "is required.", "admin_key_required")
            _settings_limiter.succeeded(client)
        elif not safe:
            _audit_refusal("403 not_claimed", method, path, headers, client)
            return _settings_error(403, "Nobody may change settings yet: this server has no admin key. Print a "
                                        "one-time code to the container log and claim the server with it.",
                                   "not_claimed")
    if method in ("POST", "PUT", "PATCH"):
        content_type = headers.get("content-type", "").split(";")[0].strip().lower()
        if content_type != "application/json":
            return _settings_error(415, "The Settings API takes application/json.", "unsupported_media_type")
    return None


def _settings_send(send):
    """`send` for a Settings response: never cached, and never with a CORS header.

    Works on the raw header list, whatever case a layer further in used for a name.
    """

    async def send_settings(message):
        if message["type"] == "http.response.start":
            kept = [(name, value) for name, value in message.get("headers", [])
                    if not name.lower().startswith(b"access-control-") and name.lower() != b"cache-control"]
            message["headers"] = [*kept, (b"cache-control", b"no-store")]
        await send(message)

    return send_settings


class _SettingsMiddleware:
    """Outermost: picks up saved settings, then applies CORS.

    For every HTTP and WebSocket scope it first stats the settings files and
    applies a change before anything else looks at the request (`_settings_tick`,
    in the live-settings section below). There is deliberately no time throttle:
    every worker applies a change at the start of its very next request, so a
    client never meets a worker that still has the old rules (a just-replaced key
    answered 401 by a stale worker would make the page forget the new key).

    CORS is a starlette CORSMiddleware around the rest of the stack, rebuilt when
    the allowed origins change; with none (the default) the request passes through
    untouched, exactly as when no CORSMiddleware was installed. The Settings page
    and its API never get CORS headers, and none of their answers may be cached
    (`_settings_send`); their other rules are `_settings_guard`'s.
    """

    def __init__(self, app):
        self.app = app
        self._cors_for = None
        self._cors = None

    def _cors_handler(self, policy: _AccessPolicy):
        key = (policy.allowed_origins, policy.allow_credentials)
        if key != self._cors_for:
            self._cors = CORSMiddleware(
                self.app,
                allow_origins=list(policy.allowed_origins),
                allow_credentials=policy.allow_credentials,
                allow_methods=["*"],
                allow_headers=["*"],
            ) if policy.allowed_origins else None
            self._cors_for = key
        return self._cors

    async def __call__(self, scope, receive, send):
        if scope["type"] in ("http", "websocket"):
            _settings_tick()
        if scope["type"] == "http":
            if _is_settings_path(scope["path"]):
                await self.app(scope, receive, _settings_send(send))
                return
            cors = self._cors_handler(_policy)
            if cors is not None:
                await cors(scope, receive, send)
                return
        await self.app(scope, receive, send)


# Outermost last: CORS must wrap everything so even a 403/413 carries the CORS
# headers the calling page needs in order to read it.
app.add_middleware(_UploadSlotMiddleware)
app.add_middleware(_BodyLimitMiddleware)
app.add_middleware(_RequestGuardMiddleware)
app.add_middleware(_SecurityHeadersMiddleware)
app.add_middleware(_SettingsMiddleware)


# `/v1` promises the OpenAI error envelope for every failure, but FastAPI answers
# a missing form field, an unknown route or an unhandled crash in its own shapes.
# The handlers below translate only for /v1 and delegate everywhere else, so the
# browser-facing /api errors keep the `{"detail": ...}` the UI already reads.
@app.exception_handler(RequestValidationError)
async def _validation_error_handler(request: Request, exc: RequestValidationError):
    if is_v1_path(request.url.path):
        return openai_validation_error_response(exc)
    return await request_validation_exception_handler(request, exc)


@app.exception_handler(StarletteHTTPException)
async def _http_error_handler(request: Request, exc: StarletteHTTPException):
    if is_v1_path(request.url.path):
        return openai_http_exception_response(request, exc)
    return await http_exception_handler(request, exc)


@app.exception_handler(Exception)
async def _unhandled_error_handler(request: Request, exc: Exception):
    if is_v1_path(request.url.path):
        return openai_error(500, "The server had an error while processing your request.",
                            code="internal_error")
    return PlainTextResponse("Internal Server Error", status_code=500)


BASE_DIR = Path(__file__).resolve().parent

app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
# `tojson` sorts keys by default. The registry is an ordered mapping — the UI
# builds its provider selectors from Object.entries() — so sorting would quietly
# reorder them alphabetically.
templates.env.policies["json.dumps_kwargs"] = {"sort_keys": False}

# Internal Docker-network URLs (container-to-container communication)
TTS_SERVICE_URL = os.getenv("TTS_SERVICE_URL", "http://piper-tts-service:5000")
STT_SERVICE_URL = os.getenv("STT_SERVICE_URL", "http://stt-service:8000")
VOICE_TRAINING_URL = os.getenv("VOICE_TRAINING_URL", "http://piper-training-service:8080")
QWEN3_TTS_SERVICE_URL = os.getenv("QWEN3_TTS_SERVICE_URL", "http://qwen3-tts-service:5004")
QWEN3_ASR_SERVICE_URL = os.getenv("QWEN3_ASR_SERVICE_URL", "http://qwen3-asr-service:5002")
PARAKEET_ASR_SERVICE_URL = os.getenv("PARAKEET_ASR_SERVICE_URL", "http://parakeet-asr-service:5005")
CANARY_ASR_SERVICE_URL = os.getenv("CANARY_ASR_SERVICE_URL", "http://canary-asr-service:5006")
CHATTERBOX_TTS_SERVICE_URL = os.getenv("CHATTERBOX_TTS_SERVICE_URL", "http://chatterbox-tts-service:5007")
MAGPIE_TTS_SERVICE_URL = os.getenv("MAGPIE_TTS_SERVICE_URL", "http://magpie-tts-service:5008")
WHISPER_CPP_SERVICE_URL = os.getenv("WHISPER_CPP_SERVICE_URL", "http://whisper-cpp:8080")


def _build_basic_tts_messages() -> dict:
    """Return shared metadata-driven messages for the generic TTS flow."""
    return {
        "validation_text": "Please enter some text to synthesize",
        "start": "Generating speech with {provider}...",
        "success": "Speech generated successfully!",
        "error": "Generation failed: {error}",
        "voice_auto_option": "Auto-Select Best Voice",
    }


def _build_stt_messages() -> dict:
    """Return shared metadata-driven messages for the generic STT flow."""
    return {
        "validation_file": "Please select an audio file",
        "start": "Processing audio with {provider}...",
        "success": "Audio processed successfully!",
        "error": "Processing failed: {error}",
        "segmented_heading": "Transcription with Segmentation",
        "result_heading": "Transcription Result",
        "copy_action": "Copy Text",
        "copy_success": "Transcription copied to clipboard",
        "language_label": "Language",
        "duration_label": "Duration",
        "segments_label": "Segments",
        "unknown": "Unknown",
        "not_available": "N/A",
    }


_BUILT_ENTRIES: dict = {}


_NO_OVERRIDE = object()


def _parse_registry_override(raw: str) -> Any:
    """PROVIDER_REGISTRY_JSON, parsed once at start; malformed JSON stops the start.

    Anything but an object (`null` too) stops it as well, at the first build below.
    """
    raw = (raw or "").strip()
    return json.loads(raw) if raw else _NO_OVERRIDE


_REGISTRY_OVERRIDE = _parse_registry_override(os.getenv("PROVIDER_REGISTRY_JSON", ""))


def _registry_override_providers() -> frozenset:
    """Provider ids PROVIDER_REGISTRY_JSON defines: the operator's entries, which no flag removes."""
    if not isinstance(_REGISTRY_OVERRIDE, dict) or not isinstance(_REGISTRY_OVERRIDE.get("providers"), dict):
        return frozenset()
    return frozenset(_REGISTRY_OVERRIDE["providers"])


def _build_provider_registry(flags: Mapping[str, Any], defaults: Mapping[str, Any]) -> dict:
    """Build the browser-facing provider registry for the frontend UI.

    `flags` holds the effective ENABLE_* values and `defaults` the effective
    DEFAULT_TTS_PROVIDER / DEFAULT_STT_PROVIDER (both: settings_schema keys). Every
    call builds new objects, so a registry handed out earlier is never changed by a
    later build.
    """
    providers = {
        "piper": {
            "kind": "tts",
            "display_name": "PiperTTS (Local Training)",
            "short_name": "PiperTTS",
            "internal_url": TTS_SERVICE_URL,
            "health_endpoint": "/health",
            "capabilities": ["tts", "voice_catalog", "custom_models", "training_target"],
            "contracts": {
                "tts": "simple-json-tts-v1",
                "voice_catalog": "voice-catalog-v1",
                "managed_voices": "custom-voice-library-v1",
            },
            "settings": {
                "defaults": {
                    "language": "auto",
                    "quality": "medium",
                    "gender": "any",
                    "speed": 1.0,
                },
                "languages": [
                    {"value": "auto", "label": "Auto-Detect"},
                    {"value": "en", "label": "English"},
                    {"value": "de", "label": "German"},
                    {"value": "fr", "label": "French"},
                    {"value": "es", "label": "Spanish"},
                    {"value": "it", "label": "Italian"},
                    {"value": "nl", "label": "Dutch"},
                ],
                "qualities": [
                    {"value": "medium", "label": "Medium Quality"},
                    {"value": "high", "label": "High Quality"},
                    {"value": "low", "label": "Low Quality (Faster)"},
                    {"value": "x_low", "label": "Ultra Low (Fastest)"},
                ],
                "genders": [
                    {"value": "any", "label": "Any Gender"},
                    {"value": "male", "label": "Male Voice"},
                    {"value": "female", "label": "Female Voice"},
                ],
                "speed": {
                    "min": 0.5,
                    "max": 2.0,
                    "step": 0.1,
                    "default": 1.0,
                },
            },
            "ui": {
                "family": "piper",
                "selectable_as_engine": True,
                "show_status": True,
                "tab_label": "Text-to-Speech",
                "messages": {
                    "tts_generation": _build_basic_tts_messages(),
                    "custom_voice_library": {
                        "loading": "Loading voices...",
                        "empty_invalid": "No voices available.",
                        "empty": "No custom trained voices found. Train a voice model and it will appear here.",
                        "unavailable": "Failed to load voices. Is the PiperTTS service running?",
                        "action_test": "Test",
                        "action_delete": "Delete",
                        "test_start": "Generating test audio...",
                        "test_error": "Test failed: {error}",
                        "delete_confirm": "Delete custom voice \"{voice_id}\"? This cannot be undone.",
                        "delete_success": "Voice \"{voice_id}\" deleted",
                        "delete_error": "Failed to delete voice: {error}",
                    },
                },
                "sections": {
                    "tts": {
                        "title": "Text-to-Speech (PiperTTS)",
                        "description": "Generate high-quality speech using PiperTTS with intelligent voice selection based on language, quality, and gender preferences.",
                        "text_placeholder": "Enter the text you want to convert to speech...",
                        "text_sample": "Hello! This is a test of the PiperTTS neural text-to-speech system.",
                        "custom_voices_title": "Custom Voices",
                        "custom_voices_description": "Manage custom trained voices loaded in PiperTTS. Delete voices you no longer need.",
                    }
                },
            },
        },
        "qwen3": {
            "kind": "tts",
            "display_name": "Qwen3-TTS (Voice Cloning)",
            "short_name": "Qwen3-TTS",
            "internal_url": QWEN3_TTS_SERVICE_URL,
            "health_endpoint": "/health",
            "capabilities": ["tts", "voice_clone", "saved_voices", "model_switching", "model_unload"],
            "contracts": {
                "tts": "simple-json-tts-v1",
                "voice_catalog": "speaker-catalog-v1",
                "model_catalog": "model-catalog-v1",
                "model_selection": "model-selection-v1",
                "runtime_status": "runtime-status-v1",
                "saved_voices": "saved-voice-library-v1",
                "voice_clone": "voice-clone-tts-v1",
                "voice_design": "voice-design-tts-v1",
            },
            "settings": {
                "defaults": {
                    # "auto" reaches qwen3-tts-service as-is and resolves there
                    # to QWEN3_DEFAULT_LANGUAGE (German unless the operator says
                    # otherwise). A language named here would override that
                    # setting on every request the UI makes.
                    "language": "auto",
                    "speaker": "Vivian",
                },
                "languages": [
                    {"value": "auto", "label": "Automatic (service default)"},
                    {"value": "English", "label": "English"},
                    {"value": "German", "label": "German"},
                    {"value": "French", "label": "French"},
                    {"value": "Spanish", "label": "Spanish"},
                    {"value": "Italian", "label": "Italian"},
                    {"value": "Portuguese", "label": "Portuguese"},
                    {"value": "Russian", "label": "Russian"},
                    {"value": "Japanese", "label": "Japanese"},
                    {"value": "Korean", "label": "Korean"},
                    {"value": "Chinese", "label": "Chinese"},
                ],
            },
            "ui": {
                "family": "qwen3",
                "selectable_as_engine": True,
                "show_status": True,
                "tab_label": "Qwen3 TTS",
                "clone_tab_label": "Voice Cloning",
                "messages": {
                    "tts_generation": _build_basic_tts_messages(),
                    "model_switching": {
                        "start": "Switching model... This may take a while if the model needs to download.",
                        "success": "Model switched to {model}",
                        "error": "Failed to switch model: {error}",
                    },
                    "model_catalog": {
                        "current_description": "Current: {model} | Capabilities: {capabilities}",
                        "unavailable_option": "Service unavailable",
                    },
                    "voice_library": {
                        "save_start": "Saving voice \"{name}\" (transcribing + extracting embedding)...",
                        "save_success": "Voice \"{name}\" saved! Use it from \"Saved Voices\" for fast TTS.",
                    },
                    "saved_voice_library": {
                        "empty_option": "No saved voices - upload a sample first",
                        "error_option": "Error loading voices",
                        "info_ref": "Ref: \"{ref_text}\"",
                        "info_saved": "Saved: {created_at}",
                        "no_selection_delete": "No voice selected to delete.",
                        "delete_confirm": "Delete saved voice \"{voice_name}\"?",
                        "delete_success": "Voice \"{voice_name}\" deleted.",
                        "delete_error": "Failed to delete voice: {error}",
                    },
                    "builtin_tts": {
                        "validation_text": "Please enter some text to synthesize",
                        "start": "Generating speech with {speaker}...",
                        "success": "Speech generated in {duration}s (Speaker: {speaker})",
                        "error": "Generation failed: {error}",
                    },
                    "saved_voice_tts": {
                        "validation_text": "Please enter some text.",
                        "validation_voice": "No saved voice selected. Upload a sample first.",
                        "start": "Generating speech with saved voice...",
                        "progress": "Generating speech... {elapsed}s",
                        "success": "Speech generated in {duration}s",
                        "success_with_audio": "Speech generated in {duration}s ({audio_duration}s audio)",
                        "error": "Generation failed: {error}",
                        "action_busy": "Generating...",
                    },
                    "voice_clone": {
                        "validation_text": "Please enter some text.",
                        "validation_voice_file": "Please select a voice sample file.",
                        "validation_ref_text": "Please enter the reference audio transcript or uncheck the option.",
                        "start_auto_transcribe": "Auto-transcribing reference audio via Qwen3-ASR, then cloning...",
                        "start_manual_ref": "Cloning and generating speech...",
                        "progress_auto_transcribe": "Auto-transcribing + cloning... {elapsed}s",
                        "progress_generate": "Generating voice clone... {elapsed}s",
                        "success": "Voice cloning completed in {duration}s",
                        "success_with_save": "Voice cloning completed in {duration}s (voice \"{name}\" saved for fast reuse)",
                        "error": "Generation failed: {error}",
                        "action_busy": "Processing...",
                    },
                    "voice_design": {
                        "validation_text": "Please enter some text to synthesize.",
                        "validation_description": "Please describe the voice you want.",
                        "start": "Designing voice and generating speech...",
                        "progress": "Designing voice... {elapsed}s",
                        "success": "Voice design completed in {duration}s",
                        "error": "Generation failed: {error}",
                        "action_busy": "Generating...",
                    },
                    "runtime_status": {
                        "unavailable": "Qwen3-TTS Service Unavailable",
                        "online": "Qwen3-TTS Service Online",
                        "device_label": "Device",
                        "model_label": "Model",
                        "not_loaded": "Not Loaded",
                        "unknown": "Unknown",
                        "gpu_suffix": "GPU",
                        "cpu_suffix": "CPU",
                        "gpu_memory_label": "GPU Memory",
                        "speakers_label": "Speakers",
                    },
                },
                "forms": {
                    "builtin_tts": {
                        "fields": {
                            "text": {
                                "label": "Text to synthesize:",
                                "placeholder": "Enter the text you want to convert to speech...",
                                "sample": "Hello! This is a demonstration of Qwen3-TTS neural text-to-speech.",
                            },
                            "language": {"label": "Language:"},
                            "speaker": {"label": "Speaker Voice:"},
                            "instruction": {
                                "label": "Voice Instruction (Optional):",
                                "placeholder": "e.g., Speak slowly and calmly, with a warm tone",
                                "hint": "Describe how the voice should sound. Leave empty for default style.",
                            },
                        },
                        "actions": {
                            "generate": "Generate Speech",
                        },
                    },
                    "voice_clone": {
                        "fields": {
                            "text": {
                                "label": "Text to synthesize:",
                                "placeholder": "Enter the text you want to synthesize...",
                                "sample": "Hello! This is a demonstration of Qwen3-TTS voice cloning.",
                            },
                            "language": {"label": "Output Language:"},
                            "model": {"label": "Model:"},
                            "voice_source": {"label": "Voice Source:"},
                            "saved_voice": {"label": "Select Saved Voice:"},
                            "voice_file": {
                                "label": "Voice Sample Audio:",
                                "drop_text": "Drop audio file here or click to browse",
                                "hint": "Supports: MP3, WAV, M4A, FLAC (recommended: 3-10 seconds, clear speech)",
                            },
                            "save_voice_name": {
                                "label": "Save this voice as (optional):",
                                "placeholder": "e.g., My Voice, John, Customer Support",
                                "hint": "Name this voice to save it for fast reuse. Leave empty to clone without saving.",
                            },
                            "ref_text_toggle": {
                                "label": "Provide reference text manually (auto-transcribed via Qwen3-ASR if unchecked)",
                            },
                            "ref_text": {
                                "label": "Reference Audio Transcript:",
                                "placeholder": "Type the exact words spoken in the voice sample audio...",
                                "hint": "Providing the transcript of the reference audio improves voice cloning quality.",
                            },
                            "voice_description": {
                                "label": "Voice Description:",
                                "placeholder": "Describe the voice you want, e.g.: A deep male voice with a warm, calm British accent and slow speaking pace",
                                "hint": "Describe the characteristics of the voice: gender, pitch, accent, tempo, tone, emotion, etc.",
                            },
                        },
                        "actions": {
                            "refresh_saved": "Refresh",
                            "delete_saved": "Delete",
                            "generate": "Generate Speech",
                            "generate_design": "Design & Generate",
                        },
                    },
                },
                "sections": {
                    "builtin_tts": {
                        "title": "Text-to-Speech (Qwen3-TTS)",
                        "description": "Generate speech using Qwen3-TTS with built-in neural voices. Supports multiple languages with high-quality synthesis.",
                        "text_placeholder": "Enter the text you want to convert to speech...",
                        "text_sample": "Hello! This is a demonstration of Qwen3-TTS neural text-to-speech.",
                        "instruction_placeholder": "e.g., Speak slowly and calmly, with a warm tone",
                        "instruction_hint": "Describe how the voice should sound. Leave empty for default style.",
                    },
                    "cloning": {
                        "title": "Voice Cloning (Qwen3-TTS)",
                        "description": "Upload a voice sample and generate speech in that voice across multiple languages using Qwen3-TTS.",
                        "text_placeholder": "Enter the text you want to synthesize...",
                        "text_sample": "Hello! This is a demonstration of Qwen3-TTS voice cloning.",
                        "saved_source_label": "Saved Voices (fast)",
                        "upload_source_label": "Upload New Sample",
                        "save_voice_placeholder": "e.g., My Voice, John, Customer Support",
                        "ref_text_placeholder": "Type the exact words spoken in the voice sample audio...",
                        "ref_text_hint": "Providing the transcript of the reference audio improves voice cloning quality.",
                        "voice_description_placeholder": "Describe the voice you want, e.g.: A deep male voice with a warm, calm British accent and slow speaking pace",
                        "voice_description_hint": "Describe the characteristics of the voice: gender, pitch, accent, tempo, tone, emotion, etc.",
                        "modes": {
                            "design": {
                                "title": "Voice Design (Qwen3-TTS)",
                                "description": "Describe the voice you want using text and generate speech with that designed voice.",
                                "button_label": "Design & Generate"
                            },
                            "unsupported": {
                                "title": "Voice Cloning (Qwen3-TTS)",
                                "description": "The CustomVoice model uses built-in speakers only and does not support voice cloning or voice design. Switch to a Base model (1.7B or 0.6B) for cloning, or use VoiceDesign for text-described voices.",
                                "button_label": "Generate Speech"
                            },
                            "saved": {
                                "title": "Voice Cloning (Qwen3-TTS)",
                                "description": "Use a saved voice for fast TTS, or upload a new sample to clone.",
                                "button_label": "Generate Speech"
                            }
                        }
                    }
                },
            },
        },
        "whisper": {
            "kind": "stt",
            "display_name": "Whisper (faster-whisper)",
            "short_name": "Whisper STT",
            "internal_url": STT_SERVICE_URL,
            "health_endpoint": "/health",
            # "streaming" is the SSE route /transcribe-stream (a whole file,
            # segments pushed as they decode). "live_transcribe" is the
            # WebSocket route /ws/transcribe (a microphone, partials as you
            # speak). They are different things and only this provider has
            # either — see the gate in /ws/stt.
            "capabilities": [
                "transcribe", "segments", "detect_language",
                "streaming", "live_transcribe", "model_unload",
            ],
            # `language_detect` is the machine-readable truth for API clients.
            # Some backends expose a /detect_language route that always returns
            # null, so route-exists is not the same as capability-exists.
            "language_detect": True,
            "contracts": {
                "transcribe": "stt-form-v1",
                "detect_language": "stt-detect-language-v1",
            },
            "settings": {
                "defaults": {
                    "language": "auto",
                    "enable_segmentation": True,
                },
                "languages": [
                    {"value": "auto", "label": "Auto-Detect"},
                    {"value": "en", "label": "English"},
                    {"value": "de", "label": "German"},
                    {"value": "fr", "label": "French"},
                    {"value": "es", "label": "Spanish"},
                    {"value": "it", "label": "Italian"},
                    {"value": "nl", "label": "Dutch"},
                ],
            },
            "ui": {
                "selectable_as_stt": True,
                "show_status": True,
                "messages": {
                    "transcription": _build_stt_messages(),
                },
            },
        },
        "qwen3-asr": {
            "kind": "stt",
            "display_name": "Qwen3-ASR (multilingual)",
            "short_name": "Qwen3-ASR",
            "internal_url": QWEN3_ASR_SERVICE_URL,
            "health_endpoint": "/health",
            "capabilities": ["transcribe", "segments", "detect_language", "model_unload"],
            "language_detect": True,
            "contracts": {
                "transcribe": "stt-form-v1",
                "detect_language": "stt-detect-language-v1",
            },
            "settings": {
                "defaults": {
                    "language": "auto",
                    "enable_segmentation": True,
                },
                "languages": [
                    {"value": "auto", "label": "Auto-Detect"},
                    {"value": "en", "label": "English"},
                    {"value": "de", "label": "German"},
                    {"value": "fr", "label": "French"},
                    {"value": "es", "label": "Spanish"},
                    {"value": "it", "label": "Italian"},
                    {"value": "nl", "label": "Dutch"},
                ],
            },
            "ui": {
                "selectable_as_stt": True,
                "show_status": True,
                "messages": {
                    "transcription": _build_stt_messages(),
                },
            },
        },
        "piper-training": {
            "kind": "training",
            "display_name": "Piper Training",
            "short_name": "Voice Training",
            "internal_url": VOICE_TRAINING_URL,
            "health_endpoint": "/health",
            "capabilities": ["dataset_preparation", "voice_training", "model_export"],
            "contracts": {
                "training": "voice-training-job-v1",
            },
            "settings": {
                "defaults": {
                    "language": "en",
                    "gender": "female",
                    "epochs": 1000,
                    "batch_size": "32",
                },
                "languages": [
                    {"value": "en", "label": "English"},
                    {"value": "de", "label": "German"},
                    {"value": "fr", "label": "French"},
                    {"value": "es", "label": "Spanish"},
                    {"value": "it", "label": "Italian"},
                    {"value": "nl", "label": "Dutch"},
                ],
                "genders": [
                    {"value": "female", "label": "Female"},
                    {"value": "male", "label": "Male"},
                    {"value": "neutral", "label": "Neutral"},
                ],
                "batch_sizes": [
                    {"value": "16", "label": "16 (Lower memory)"},
                    {"value": "32", "label": "32 (Recommended)"},
                    {"value": "64", "label": "64 (Higher memory)"},
                ],
                "epochs": {
                    "min": 100,
                    "max": 5000,
                    "step": 100,
                },
            },
            "ui": {
                "family": "piper",
                "show_status": True,
                "tab_label": "Voice Training",
                "messages": {
                    "start_training": {
                        "validation_name": "Please enter a voice model name",
                        "validation_files": "Please select training audio files",
                        "start": "Starting VITS training pipeline for {deployment_target}...",
                        "success": "Training started successfully!",
                        "error": "Training failed: {error}",
                        "completed": "Training completed. Deployment target: {deployment_target}.",
                        "failed": "Training failed. Check training jobs for details.",
                        "progress": "Progress: {progress}% (Epoch {current_epoch}/{total_epochs})",
                    },
                    "train_from_dataset": {
                        "validation_name": "Please enter a voice model name",
                        "confirm": "Start training \"{voice_name}\" from the existing prepared dataset (train.json / val.json)?",
                        "start": "Starting training for \"{voice_name}\" from existing dataset...",
                        "success_status": "Training started! Job ID: {job_id}",
                        "success_notification": "Training started for \"{voice_name}\" with target {deployment_target}",
                        "error": "Failed: {error}",
                    },
                    "resume_training": {
                        "validation_name": "Please enter a voice model name",
                        "confirm": "Resume training for voice \"{voice_name}\" from the last checkpoint?",
                        "start_notification": "Resuming training for \"{voice_name}\"...",
                        "success_notification": "Training resumed for \"{voice_name}\" with target {deployment_target}",
                        "success_status": "Resumed training for \"{voice_name}\" — monitoring progress...",
                        "error_status": "Resume failed: {error}",
                        "error_notification": "Resume failed: {error}",
                    },
                    "model_management": {
                        "deploy_start": "Deploying \"{model_name}\" to {deployment_target}...",
                        "deploy_success": "Model \"{model_name}\" deployment status: {status} on {deployment_target}.",
                        "deploy_error": "Export failed: {error}",
                        "download_success": "Model download started",
                        "download_error": "Failed to download model",
                        "delete_confirm": "Delete model \"{job_id}\" and all training data? This cannot be undone.",
                        "delete_success": "Model deleted successfully",
                        "delete_error": "Failed to delete model",
                        "cancel_confirm": "Cancel this training job?",
                        "cancel_success": "Training job cancelled",
                        "cancel_error": "Failed to cancel job",
                    },
                    "model_list": {
                        "loading": "Loading trained models...",
                        "empty": "No trained models found. Start training to create your first model!",
                        "error": "Failed to load models. Is the training service running?",
                        "deploy_action": "Deploy",
                        "download_action": "Download",
                        "delete_action": "Delete",
                    },
                    "job_list": {
                        "loading": "Loading training jobs...",
                        "empty": "No training jobs found.",
                        "error": "Failed to load training jobs.",
                        "details_action": "Details",
                        "resume_action": "Resume",
                        "cancel_action": "Cancel",
                    },
                    "job_details": {
                        "fetch_error": "Failed to fetch job details",
                        "job_label": "Job",
                        "status_label": "Status",
                        "deployment_target_label": "Deployment Target",
                        "progress_label": "Progress",
                        "current_epoch_label": "Current Epoch",
                        "configuration_heading": "Configuration",
                        "epochs_label": "Epochs",
                        "batch_size_label": "Batch Size",
                        "learning_rate_label": "Learning Rate",
                        "best_loss_label": "Best Loss",
                        "recent_logs_heading": "Recent Logs",
                        "na": "N/A",
                    },
                },
                "forms": {
                    "start_training": {
                        "fields": {
                            "voice_name": {
                                "label": "Voice Model Name:",
                                "placeholder": "Enter a unique name for this voice model",
                            },
                            "language": {"label": "Target Language:"},
                            "gender": {"label": "Voice Gender:"},
                            "files": {
                                "label": "Training Audio Files:",
                                "drop_text": "Drop multiple audio files here or click to browse",
                                "hint": "Recommended: 10+ minutes of high-quality speech data",
                            },
                            "epochs": {"label": "Training Epochs:"},
                            "batch_size": {"label": "Batch Size:"},
                            "deployment_target": {"label": "Post-Export Deployment Target:"},
                        },
                        "actions": {
                            "submit": "Start Training",
                        },
                    },
                    "continue_training": {
                        "title": "Continue Existing Training",
                        "description": "If a training job was interrupted (e.g. after a restart) and a checkpoint exists, use <strong>Resume from Checkpoint</strong>. If no checkpoint was saved yet but the dataset (train.json / val.json) is already prepared, use <strong>Train from Dataset</strong>.",
                        "fields": {
                            "voice_name": {
                                "label": "Voice Model Name:",
                                "placeholder": "e.g. luna",
                            },
                            "epochs": {"label": "Epochs:"},
                            "deployment_target": {"label": "Deployment Target:"},
                        },
                        "actions": {
                            "resume": "Resume from Checkpoint",
                            "train_from_dataset": "Train from Dataset",
                        },
                    },
                    "model_management": {
                        "fields": {
                            "deployment_target": {"label": "Manual Deployment Target:"},
                        },
                    },
                },
                "sections": {
                    "training": {
                        "title": "Voice Training",
                        "description": "Train custom voice models using VITS neural networks. Upload multiple audio files with transcripts for high-quality voice cloning.",
                        "voice_name_placeholder": "Enter a unique name for this voice model",
                        "continue_voice_name_placeholder": "e.g. luna",
                        "models_description": "Manage your trained voice models. Export to TTS to use them, download, or delete.",
                    }
                },
            },
        },
    }

    # Voice training is offered unless ENABLE_TRAINING says otherwise. Without the
    # entry the training routes answer 404 and the UI shows no training status.
    if not flags["ENABLE_TRAINING"]:
        del providers["piper-training"]

    if flags["ENABLE_PARAKEET_ASR"]:
        providers["parakeet"] = {
            "kind": "stt",
            "display_name": "Parakeet-TDT (realtime, 25 EU langs)",
            "short_name": "Parakeet ASR",
            "internal_url": PARAKEET_ASR_SERVICE_URL,
            "health_endpoint": "/health",
            # /detect_language exists but is a stub that always returns null,
            # so the capability is NOT declared. Parakeet auto-detects
            # internally during transcription but does not report it.
            "capabilities": ["transcribe", "segments", "model_unload"],
            "language_detect": False,
            "contracts": {
                "transcribe": "stt-form-v1",
            },
            "settings": {
                "defaults": {
                    "language": "auto",
                    "enable_segmentation": True,
                },
                "languages": [
                    {"value": "auto", "label": "Auto-Detect"},
                    {"value": "en", "label": "English"},
                    {"value": "de", "label": "German"},
                    {"value": "fr", "label": "French"},
                    {"value": "es", "label": "Spanish"},
                    {"value": "it", "label": "Italian"},
                    {"value": "nl", "label": "Dutch"},
                ],
            },
            "ui": {
                "selectable_as_stt": True,
                "show_status": True,
                "messages": {
                    "transcription": _build_stt_messages(),
                },
            },
        }

    if flags["ENABLE_CANARY_ASR"]:
        providers["canary"] = {
            "kind": "stt",
            "display_name": "Canary-180M (realtime, en/de/es/fr)",
            "short_name": "Canary ASR",
            "internal_url": CANARY_ASR_SERVICE_URL,
            "health_endpoint": "/health",
            "capabilities": ["transcribe", "segments", "model_unload"],
            # Canary has no language identification at all — its /detect_language
            # route transcribes with the default language and returns null. The
            # contract is therefore NOT declared: a client must not be told it
            # can detect language here.
            "language_detect": False,
            "contracts": {
                "transcribe": "stt-form-v1",
            },
            "settings": {
                "defaults": {
                    "language": "de",
                    "enable_segmentation": True,
                },
                # Canary has no auto-detection — the language picks the decoder
                "languages": [
                    {"value": "de", "label": "German"},
                    {"value": "en", "label": "English"},
                    {"value": "fr", "label": "French"},
                    {"value": "es", "label": "Spanish"},
                ],
            },
            "ui": {
                "selectable_as_stt": True,
                "show_status": True,
                "messages": {
                    "transcription": _build_stt_messages(),
                },
            },
        }

    if flags["ENABLE_CHATTERBOX_TTS"]:
        providers["chatterbox"] = {
            "kind": "tts",
            "display_name": "Chatterbox (Multilingual, MIT)",
            "short_name": "Chatterbox",
            "internal_url": CHATTERBOX_TTS_SERVICE_URL,
            "health_endpoint": "/health",
            "capabilities": ["tts", "voice_clone", "tts_stream", "model_unload"],
            "contracts": {
                "tts": "simple-json-tts-v1",
                "voice_clone": "voice-clone-tts-v1",
                # Sentence-chunked streaming: audio starts after the first
                # chunk instead of after the whole text. The gateway routes
                # here automatically when the contract key is present.
                "tts_stream": "chunked-wav-stream-v1",
            },
            "settings": {
                "defaults": {
                    "language": "de",
                },
                "languages": [
                    {"value": "de", "label": "German"},
                    {"value": "en", "label": "English"},
                    {"value": "fr", "label": "French"},
                    {"value": "es", "label": "Spanish"},
                    {"value": "it", "label": "Italian"},
                    {"value": "nl", "label": "Dutch"},
                    {"value": "pt", "label": "Portuguese"},
                    {"value": "pl", "label": "Polish"},
                ],
            },
            "ui": {
                # Uses the generic TTS panel (same family as Piper)
                "family": "piper",
                "selectable_as_engine": True,
                "show_status": True,
                "tab_label": "Text-to-Speech",
                "messages": {
                    "tts_generation": _build_basic_tts_messages(),
                },
                "sections": {
                    "tts": {
                        "title": "Text-to-Speech (Chatterbox)",
                        "description": "Generate multilingual speech with Resemble AI Chatterbox. Output is watermarked. Voice cloning is available via the API (/clone).",
                        "text_placeholder": "Enter the text you want to convert to speech...",
                        "text_sample": "Hallo! Dies ist ein Test von Chatterbox Multilingual.",
                    }
                },
            },
        }

    if flags["ENABLE_MAGPIE_TTS"]:
        providers["magpie"] = {
            "kind": "tts",
            "display_name": "Magpie TTS (NVIDIA, 5 voices)",
            "short_name": "Magpie",
            "internal_url": MAGPIE_TTS_SERVICE_URL,
            "health_endpoint": "/health",
            "capabilities": ["tts", "voice_catalog", "model_unload"],
            "contracts": {
                "tts": "simple-json-tts-v1",
                # Five built-in speakers, listed by the service's /speakers.
                "voice_catalog": "speaker-catalog-v1",
            },
            "settings": {
                "defaults": {
                    # "auto" reaches magpie-tts-service as-is and resolves there to
                    # MAGPIE_DEFAULT_LANGUAGE (German unless the operator says
                    # otherwise); naming a language here would override that setting
                    # on every request the UI makes.
                    "language": "auto",
                },
                # The languages NeMo 3.0.x can really route to a tokenizer of the
                # checkpoint (the service lists them at /languages). The checkpoint
                # holds more, but NeMo would read those with English rules.
                "languages": [
                    # No label on purpose: app.js then names the default the service
                    # reports (/speakers `default_language`), "Automatic - server
                    # decides (default: German)", instead of a fixed text.
                    {"value": "auto"},
                    {"value": "de", "label": "German"},
                    {"value": "en", "label": "English"},
                    {"value": "es", "label": "Spanish"},
                    {"value": "fr", "label": "French"},
                    {"value": "ja", "label": "Japanese"},
                    {"value": "zh", "label": "Chinese"},
                ],
            },
            "ui": {
                # Uses the generic TTS panel (same family as Piper and Chatterbox)
                "family": "piper",
                "selectable_as_engine": True,
                "show_status": True,
                "tab_label": "Text-to-Speech",
                "messages": {
                    # "auto" is MAGPIE_DEFAULT_SPEAKER, one fixed voice, not a choice
                    # made for the text the way Piper's "auto" is.
                    "tts_generation": {**_build_basic_tts_messages(), "voice_auto_option": "Service default voice"},
                },
                "sections": {
                    "tts": {
                        "title": "Text-to-Speech (Magpie)",
                        "description": "Generate multilingual speech with NVIDIA Magpie-TTS and one of its five built-in voices. There is no voice cloning, and the audio arrives when the whole text is done.",
                        "text_placeholder": "Enter the text you want to convert to speech...",
                        "text_sample": "Hallo! Dies ist ein Test von NVIDIA Magpie. Am 3. Mai 2026 kostet das Ticket 12,50 Euro.",
                    }
                },
            },
        }

    if flags["ENABLE_WHISPER_CPP"]:
        providers["whisper-cpp"] = {
            "kind": "stt",
            "display_name": "whisper.cpp (OpenAI-compatible)",
            "short_name": "whisper.cpp",
            "internal_url": WHISPER_CPP_SERVICE_URL,
            "health_endpoint": "/",
            "capabilities": ["transcribe", "openai_compatible"],
            # whisper.cpp auto-detects when language=auto is sent (which the
            # gateway now always does), but exposes no separate LID route.
            "language_detect": True,
            "contracts": {
                "transcribe": "openai-audio-transcriptions-v1",
            },
            # whisper-server (current whisper.cpp) only serves its native
            # /inference route; it accepts the same form fields as the OpenAI
            # endpoint and returns {"text": ...}.
            "transcribe_path": "/inference",
            "settings": {
                "defaults": {
                    "language": "auto",
                    "enable_segmentation": False,
                },
                "languages": [
                    {"value": "auto", "label": "Auto-Detect"},
                    {"value": "en", "label": "English"},
                    {"value": "de", "label": "German"},
                    {"value": "fr", "label": "French"},
                    {"value": "es", "label": "Spanish"},
                    {"value": "it", "label": "Italian"},
                    {"value": "nl", "label": "Dutch"},
                ],
            },
            "ui": {
                "selectable_as_stt": True,
                "show_status": True,
                "messages": {
                    "transcription": _build_stt_messages(),
                },
            },
        }

    registry = {
        "providers": providers,
        "ui": {
            "default_tts_provider": defaults["DEFAULT_TTS_PROVIDER"],
            "default_stt_provider": defaults["DEFAULT_STT_PROVIDER"],
            "training_provider": _TRAINING_PROVIDER,
            "enable_whisper_cpp": flags["ENABLE_WHISPER_CPP"],
            "enable_parakeet_asr": flags["ENABLE_PARAKEET_ASR"],
            "enable_canary_asr": flags["ENABLE_CANARY_ASR"],
            "enable_chatterbox_tts": flags["ENABLE_CHATTERBOX_TTS"],
            "enable_magpie_tts": flags["ENABLE_MAGPIE_TTS"],
            "enable_training": flags["ENABLE_TRAINING"],
            "copy": {
                "app_subtitle": "Neural Text-to-Speech with Voice Training & Cloning + Speech-to-Text",
                "stt_tab_label": "Speech-to-Text",
                "stt_title": "Speech-to-Text",
                "stt_description": "Convert speech to text. Supports transcription with optional audio segmentation for training data preparation.",
            },
        },
    }

    # Remembered before the override below can replace it: an entry the operator
    # supplied is theirs, and nothing may rewrite it from what a backend reports.
    _BUILT_ENTRIES["canary"] = providers.get("canary")

    if _REGISTRY_OVERRIDE is not _NO_OVERRIDE:
        # A copy per build: the operator's entries in one registry are never the
        # objects of another (the canary refresh and tests change entries in place).
        parsed = copy.deepcopy(_REGISTRY_OVERRIDE)
        registry["providers"].update(parsed.get("providers", {}))
        registry["ui"].update(parsed.get("ui", {}))

    return registry


# The one registry object for the life of the process: the /v1 router holds it,
# and a settings change replaces its contents (`_apply`), never the object.
PROVIDER_REGISTRY = _build_provider_registry(_ENV_VALUES, _ENV_VALUES)
# The engine settings PROVIDER_REGISTRY was last built from.
_ENGINE_KEYS = (*settings_schema.ENGINE_FLAGS, "DEFAULT_TTS_PROVIDER", "DEFAULT_STT_PROVIDER")
_registry_built_from = {key: _ENV_VALUES[key] for key in _ENGINE_KEYS}


# --- settings the backends report ---------------------------------------------------
#
# Canary decodes only the languages its checkpoint was trained on, and which those
# are depends on CANARY_ASR_MODEL: en/de/es/fr for the flash models, 25 European
# languages for canary-1b-v2 (or whatever CANARY_SUPPORTED_LANGUAGES says). The
# list in the registry above is only the fallback for a service that cannot be
# asked; when it answers, its own /status (`supported_languages`, `default_language`,
# `current_model`) replaces it, so the selector offers exactly what the service will
# accept instead of a copy that goes stale.

_LANGUAGE_NAMES = {
    "ar": "Arabic", "bg": "Bulgarian", "cs": "Czech", "da": "Danish", "de": "German",
    "el": "Greek", "en": "English", "es": "Spanish", "et": "Estonian", "fi": "Finnish",
    "fr": "French", "hr": "Croatian", "hu": "Hungarian", "it": "Italian", "ja": "Japanese",
    "ko": "Korean", "lt": "Lithuanian", "lv": "Latvian", "mt": "Maltese", "nl": "Dutch",
    "pl": "Polish", "pt": "Portuguese", "ro": "Romanian", "ru": "Russian", "sk": "Slovak",
    "sl": "Slovenian", "sv": "Swedish", "tr": "Turkish", "uk": "Ukrainian", "zh": "Chinese",
}
_CANARY_LANGUAGE_REFRESH_S = 300.0    # how long a discovered list is trusted
_CANARY_LANGUAGE_RETRY_S = 20.0       # how soon to ask again after a failed attempt
_canary_languages = {"next_at": None}
# Injectable so tests can move time instead of sleeping.
_language_clock = time.monotonic


def _apply_canary_status(entry: dict, payload: Any) -> bool:
    """Put the languages (and model) a canary /status reports into its registry entry."""
    if not isinstance(payload, dict) or not isinstance(payload.get("supported_languages"), list):
        return False
    codes = []
    for item in payload["supported_languages"]:
        code = item.strip().lower() if isinstance(item, str) else ""
        if re.fullmatch(r"[a-z]{2,3}", code) and code not in codes:
            codes.append(code)
    if not codes:
        return False

    settings = entry.setdefault("settings", {})
    default = (_reported_default_language(payload) or "").lower()
    if default not in codes:
        current = (settings.get("defaults") or {}).get("language")
        default = current if current in codes else codes[0]
    ordered = sorted(codes, key=lambda c: (c != default, _LANGUAGE_NAMES.get(c, c.upper())))
    settings["languages"] = [{"value": c, "label": _LANGUAGE_NAMES.get(c, c.upper())} for c in ordered]
    settings["defaults"] = {**(settings.get("defaults") or {}), "language": default}

    model = payload.get("current_model")
    model = model.rsplit("/", 1)[-1] if isinstance(model, str) and model.strip() else "realtime"
    shown = "/".join(ordered) if len(ordered) <= 6 else f"{len(ordered)} languages"
    entry["display_name"] = f"Canary ({model}, {shown})"
    return True


async def _refresh_canary_languages() -> None:
    """Ask the canary service which languages it decodes, at most once per interval.

    Never raises and never waits long: the page that calls it must render even
    when canary is down (the built-in list is then used, and the next attempt is
    held back for `_CANARY_LANGUAGE_RETRY_S` so a dead service costs one short
    connect attempt per interval rather than one per page load).
    """
    entry = PROVIDER_REGISTRY["providers"].get("canary")
    # An operator who replaced the entry through PROVIDER_REGISTRY_JSON chose those
    # languages on purpose and is left alone.
    if entry is None or entry is not _BUILT_ENTRIES.get("canary"):
        return
    now = _language_clock()
    if _canary_languages["next_at"] is not None and now < _canary_languages["next_at"]:
        return
    # Provisional, so concurrent page loads do not each probe a dead service.
    _canary_languages["next_at"] = now + _CANARY_LANGUAGE_RETRY_S
    try:
        response = await _get_http_client().get(
            f"{entry['internal_url']}/status", timeout=_timeout(2.0))
        payload = response.json() if response.status_code == 200 else None
    except Exception as exc:
        logger.info("canary /status could not be read (%s); keeping the built-in language list", type(exc).__name__)
        return
    if _apply_canary_status(entry, payload):
        _canary_languages["next_at"] = now + _CANARY_LANGUAGE_REFRESH_S


def _template_provider_lists() -> tuple[list[tuple[str, dict]], list[tuple[str, dict]], list[tuple[str, dict]]]:
    """Return provider lists used to render the UI template."""
    providers = PROVIDER_REGISTRY["providers"]
    tts_providers = [
        (provider_id, provider)
        for provider_id, provider in providers.items()
        if provider.get("kind") == "tts" and provider.get("ui", {}).get("selectable_as_engine")
    ]
    stt_providers = [
        (provider_id, provider)
        for provider_id, provider in providers.items()
        if provider.get("kind") == "stt" and provider.get("ui", {}).get("selectable_as_stt")
    ]
    status_providers = [
        (provider_id, provider)
        for provider_id, provider in providers.items()
        if provider.get("ui", {}).get("show_status")
    ]
    return tts_providers, stt_providers, status_providers


def _get_provider(provider_id: str, kind: Optional[str] = None) -> dict:
    """Return a registered provider and validate its kind when requested."""
    provider = PROVIDER_REGISTRY["providers"].get(provider_id)
    if not provider:
        raise HTTPException(status_code=404, detail=f"Unknown provider: {provider_id}")
    if kind and provider.get("kind") != kind:
        raise HTTPException(status_code=400, detail=f"Provider {provider_id} is not a {kind} provider")
    return provider


def _normalize_qwen3_language(language: Optional[str]) -> str:
    """The ``lang`` to send to qwen3-tts-service for a normalized request language.

    Codes the model speaks become the labels it expects. "auto" and an empty
    value stay "auto": the service resolves that itself, to QWEN3_DEFAULT_LANGUAGE
    (German unless the operator changed it). Anything else is forwarded untouched,
    so a language the model cannot speak reaches the service and comes back as its
    400 naming the supported ones. Falling back to English here, as this used to
    for "auto" and for languages the model lacks (Dutch among them), overrode the
    operator's choice and spoke unsupported text with an English accent, HTTP 200.
    """
    language_map = {
        "en": "English",
        "en_us": "English",
        "en_gb": "English",
        "de": "German",
        "de_de": "German",
        "fr": "French",
        "fr_fr": "French",
        "es": "Spanish",
        "es_es": "Spanish",
        "it": "Italian",
        "it_it": "Italian",
        "pt": "Portuguese",
        "ru": "Russian",
        "ja": "Japanese",
        "ko": "Korean",
        "zh": "Chinese",
        "zh_cn": "Chinese",
    }
    requested = (language or "").strip()
    normalized = requested.lower().replace("-", "_")
    if not normalized or normalized == "auto":
        return "auto"
    return language_map.get(normalized, requested)


def _normalize_piper_voice_catalog(payload: dict) -> list[dict]:
    """Convert the Piper /voices response into a normalized voice list."""
    voices = payload.get("voices", {})
    if not isinstance(voices, dict):
        return []
    normalized = []
    for voice_id, voice in voices.items():
        if not isinstance(voice, dict):
            continue
        normalized.append({
            "id": voice_id,
            "name": voice.get("name") or voice.get("speaker") or voice_id,
            "language": voice.get("language"),
            "description": voice.get("quality"),
            "kind": voice.get("model_type", "default"),
            "raw": voice,
        })
    return normalized


def _normalize_qwen3_voice_catalog(payload: dict) -> list[dict]:
    """Convert the Qwen3 speaker list into a normalized voice list."""
    speakers = payload.get("speakers", [])
    languages = payload.get("languages", [])
    if not isinstance(speakers, list):
        speakers = []
    if not isinstance(languages, list):
        languages = []
    normalized = []
    for speaker in speakers:
        normalized.append({
            "id": speaker,
            "name": speaker,
            "language": "multilingual",
            "description": f"Built-in speaker. Languages: {', '.join(languages[:4])}{'...' if len(languages) > 4 else ''}",
            "kind": "builtin",
            "raw": {"speaker": speaker, "languages": languages},
        })
    return normalized


def _normalize_qwen3_model_catalog(payload: dict) -> list[dict]:
    """Convert the Qwen3 model list into a normalized model catalog."""
    models = payload.get("models", {})
    if not isinstance(models, dict):
        return []
    current_model = payload.get("current_model")
    normalized = []
    for model_id, info in models.items():
        if not isinstance(info, dict):
            continue
        capabilities = info.get("capabilities", [])
        normalized.append({
            "id": model_id,
            "name": info.get("name", model_id),
            "description": info.get("description", ""),
            "capabilities": capabilities,
            "capabilities_text": ", ".join(capabilities),
            "is_current": model_id == current_model,
            "raw": info,
        })
    return normalized


def _normalize_qwen3_runtime_status(payload: dict, provider_id: str) -> dict:
    """Convert provider-native Qwen3 runtime data into a stable frontend status shape."""
    model_info = payload.get("current_model_info") or {}
    gpu_memory_allocated = payload.get("gpu_memory_allocated")
    gpu_memory_gb = None
    if gpu_memory_allocated not in (None, ""):
        try:
            gpu_memory_gb = round(float(gpu_memory_allocated) / (1024 ** 3), 1)
        except (TypeError, ValueError):
            gpu_memory_gb = None

    speakers = payload.get("builtin_speakers")
    if not isinstance(speakers, list):
        speakers = []

    return {
        "provider": provider_id,
        "device_name": str(payload.get("device") or "").upper() or None,
        "device_type": "gpu" if payload.get("cuda_available") else "cpu",
        "model_loaded": bool(payload.get("model_loaded")),
        "model_name": model_info.get("name") or payload.get("current_model") or None,
        "gpu_memory_gb": gpu_memory_gb,
        "speakers": speakers,
    }


def _truncate_text(value: Optional[str], limit: int = 80) -> Optional[str]:
    """Return a compact preview string for longer free-form text fields."""
    text = str(value or "").strip()
    if not text:
        return None
    if len(text) <= limit:
        return text
    return f"{text[:limit].rstrip()}..."


def _format_frontend_timestamp(value: Optional[str]) -> Optional[str]:
    """Convert upstream timestamps into a stable, human-readable label."""
    if value in (None, ""):
        return None

    text = str(value).strip()
    if not text:
        return None

    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return text

    suffix = ""
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(timezone.utc)
        suffix = " UTC"

    return f"{parsed.strftime('%Y-%m-%d %H:%M')}{suffix}"


def _normalize_training_target_label(target_id: Optional[str]) -> str:
    """Map deployment target ids to stable browser-facing labels."""
    labels = {
        "none": "Manual download only",
        "piper-volume": "Piper shared volume",
        "piper-http": "Piper upload API",
    }
    if not target_id:
        return "Default"
    return labels.get(target_id, str(target_id).replace("-", " ").title())


def _normalize_qwen3_saved_voice_library(payload: dict, provider_id: str) -> dict:
    """Normalize saved-voice library entries for browser consumers."""
    voices_payload = payload.get("voices") if isinstance(payload, dict) else []
    normalized_voices = []

    if isinstance(voices_payload, list):
        for voice in voices_payload:
            if not isinstance(voice, dict):
                continue

            ref_text = voice.get("ref_text") or voice.get("reference_text")
            created_at = voice.get("created_at")
            normalized_voice = dict(voice)
            normalized_voice.update({
                "id": voice.get("id") or voice.get("voice_id") or voice.get("name"),
                "name": voice.get("name") or voice.get("id") or voice.get("voice_id"),
                "language": voice.get("language") or voice.get("lang"),
                "reference_text": ref_text or None,
                "reference_preview": _truncate_text(ref_text, 80),
                "created_at": created_at or None,
                "created_at_display": _format_frontend_timestamp(created_at),
            })
            normalized_voices.append(normalized_voice)

    return {
        "provider": provider_id,
        "voices": normalized_voices,
    }


def _normalize_training_logs(logs_payload) -> list[dict]:
    """Normalize training log entries into a stable text-oriented structure."""
    normalized_logs = []
    if not isinstance(logs_payload, list):
        return normalized_logs

    for entry in logs_payload[-5:]:
        if isinstance(entry, dict):
            timestamp = entry.get("timestamp") or entry.get("time")
            message = entry.get("message") or entry.get("text") or json.dumps(entry)
        else:
            timestamp = None
            message = str(entry)

        timestamp_display = _format_frontend_timestamp(timestamp)
        display = f"{timestamp_display}: {message}" if timestamp_display else message
        normalized_logs.append({
            "timestamp": timestamp or None,
            "timestamp_display": timestamp_display,
            "message": message,
            "display": display,
        })

    return normalized_logs


def _normalize_training_job(payload: dict) -> dict:
    """Normalize training job payloads for job tables and detail views."""
    if not isinstance(payload, dict):
        return {}

    config = payload.get("config") if isinstance(payload.get("config"), dict) else {}
    created_at = payload.get("created_at") or config.get("created_at")
    voice_name = (
        payload.get("voice_name")
        or payload.get("model_name")
        or config.get("voice_name")
        or config.get("speaker_name")
        or payload.get("job_id")
    )
    deployment_target = payload.get("deployment_target")
    best_loss = payload.get("best_loss")
    if best_loss in (None, ""):
        best_loss = payload.get("loss")

    try:
        normalized_best_loss = float(best_loss) if best_loss not in (None, "") else None
    except (TypeError, ValueError):
        normalized_best_loss = None

    config_summary = {
        "epochs": config.get("epochs") or payload.get("total_epochs"),
        "batch_size": config.get("batch_size"),
        "learning_rate": config.get("learning_rate"),
    }

    normalized_job = dict(payload)
    normalized_job.update({
        "voice_name": voice_name,
        "model_name": payload.get("model_name") or voice_name,
        "deployment_target_label": _normalize_training_target_label(deployment_target),
        "created_at": created_at or None,
        "created_at_display": _format_frontend_timestamp(created_at) or "N/A",
        "config_summary": config_summary,
        "best_loss": normalized_best_loss,
        "best_loss_display": f"{normalized_best_loss:.4f}" if normalized_best_loss is not None else None,
        "recent_logs": _normalize_training_logs(payload.get("logs")),
    })
    return normalized_job


def _normalize_training_jobs_payload(payload):
    """Normalize training job list responses while preserving top-level shape."""
    if isinstance(payload, list):
        return [_normalize_training_job(job) for job in payload if isinstance(job, dict)]

    if isinstance(payload, dict) and isinstance(payload.get("jobs"), list):
        normalized_payload = dict(payload)
        normalized_payload["jobs"] = [
            _normalize_training_job(job)
            for job in payload.get("jobs", [])
            if isinstance(job, dict)
        ]
        return normalized_payload

    return payload


def _normalize_training_export_response(payload: dict) -> dict:
    """Add stable deployment labels to training export responses."""
    if not isinstance(payload, dict):
        return payload

    normalized_payload = dict(payload)
    deployment = normalized_payload.get("deployment")
    if isinstance(deployment, dict):
        target = deployment.get("target")
        normalized_payload["deployment"] = {
            **deployment,
            "target_label": _normalize_training_target_label(target),
        }
    return normalized_payload


# --- what a client is told when a backend fails ----------------------------------
#
# A backend's error body is written for the operator: a 500 carries `str(exception)`
# (file paths under /app/models, a whole traceback), and a connection error names
# the internal URL. Echoing those to every caller hands out the deployment layout
# for free, so a client gets a status-appropriate sentence and a request id, and
# the detail goes to the log under the same id.
#
# 4xx answers are different: a backend uses them to explain what is wrong with the
# request ("Language 'xx' is not supported ... Supported: ...", "text is 5001
# characters; the limit is 5000"), and the UI shows exactly that. Those are
# forwarded when they are one short line of prose; anything that looks like a
# traceback, a filesystem path or a URL is treated like a 5xx.

_MAX_CLIENT_DETAIL_CHARS = 500
_UNSAFE_DETAIL = re.compile(
    r"Traceback \(most recent call last\)"
    r"|File \""
    r"|://"
    r"|(?<![\w.-])/(?:app|home|usr|tmp|root|opt|var|etc|models?|data|mnt|srv|proc|sys|run|lib|workspace)\b"
    r"|\b[A-Za-z]:\\"
)


def _new_request_id() -> str:
    return uuid.uuid4().hex[:12]


def _client_safe_detail(detail: Any) -> Optional[str]:
    """`detail` as one short, plain sentence a client may see, or None when it may not."""
    if isinstance(detail, list):
        # FastAPI's own validation errors: [{"loc": [...], "msg": "...", "input": ...}].
        # `input` echoes the request and `ctx` can hold anything, so only loc + msg.
        parts = []
        for item in detail[:3]:
            if isinstance(item, dict) and isinstance(item.get("msg"), str):
                loc = ".".join(str(p) for p in item.get("loc", ()) if p not in ("body", "query", "path"))
                parts.append(f"{loc}: {item['msg']}" if loc else item["msg"])
        detail = "; ".join(parts)
    if not isinstance(detail, str):
        return None
    text = detail.strip()
    if not text or len(text) > _MAX_CLIENT_DETAIL_CHARS or "\n" in text or "\r" in text:
        return None
    if _UNSAFE_DETAIL.search(text):
        return None
    return text


def _generic_upstream_detail(status: int, request_id: str) -> str:
    if status >= 500:
        return f"The backend service failed to handle the request (HTTP {status}). Request id: {request_id}."
    return f"The backend service rejected the request (HTTP {status}). Request id: {request_id}."


def _upstream_error_detail(payload: Any) -> Any:
    """The message field of a backend's JSON error, whichever convention it follows."""
    if not isinstance(payload, dict):
        return None
    detail = payload.get("detail") or payload.get("error") or payload.get("message")
    if isinstance(detail, dict):
        detail = detail.get("message")
    return detail


def _numeric_retry_after(response: httpx.Response) -> Optional[str]:
    """The backend's Retry-After when it is a number of seconds, else None."""
    value = (response.headers.get("retry-after") or "").strip()
    return value if value.isascii() and value.isdigit() else None


def _build_error_from_response(response: httpx.Response) -> HTTPException:
    """Convert an upstream HTTP error into a frontend HTTPException.

    The status is the backend's own; the message is what `_client_safe_detail`
    lets through (4xx, and a designed 503, below) or a generic sentence with a
    request id. The raw body is logged, never returned.

    A 503 that carries a numeric Retry-After is a backend's designed answer, not
    a crash: the queue is full, the request waited its time for a turn, or the GPU
    ran out of memory. Its sentence is written for the caller ("The service is
    busy ... Retry shortly."), so it is passed on with the Retry-After, which the
    UI and OpenAI clients act on. Every other 5xx stays generic.
    """
    status = response.status_code
    raw = response.text or ""
    detail: Any = None
    try:
        detail = _upstream_error_detail(response.json())
    except Exception:
        pass

    retry_after = _numeric_retry_after(response) if status == 503 else None
    safe = _client_safe_detail(detail) if status < 500 or retry_after else None
    if safe is not None:
        headers = {"Retry-After": retry_after} if retry_after else None
        return HTTPException(status_code=status, detail=safe, headers=headers)

    request_id = _new_request_id()
    try:
        url = response.request.url
    except RuntimeError:        # a Response built by hand has no request attached
        url = "unknown url"
    logger.warning(
        "backend %s answered HTTP %s [request id %s]: %s", url, status, request_id, raw[:2000])
    return HTTPException(status_code=status, detail=_generic_upstream_detail(status, request_id))


def _request_of(exc: httpx.RequestError) -> Optional[httpx.Request]:
    """The request an httpx error belongs to (httpx raises RuntimeError when none is attached)."""
    try:
        return exc.request
    except RuntimeError:
        return None


def _build_upstream_request_error(service_name: str, exc: httpx.RequestError) -> HTTPException:
    """Convert an upstream transport failure into a 503 frontend HTTPException.

    The client learns which service is down, not where it lives: the internal URL
    and the transport error are logged under the request id in the message.

    A read timeout is not a service that is down: the backend accepted the
    request and has not answered within the read budget, and it may well still be
    generating. Calling that "unavailable" sent people looking for a dead
    container, so it is said as it is, with the budget that ran out.
    """
    request_id = _new_request_id()
    request = _request_of(exc)
    request_url = getattr(request, "url", None)
    timed_out = isinstance(exc, httpx.ReadTimeout)
    logger.warning(
        "%s %s [request id %s]: %s: %s (%s)",
        service_name, "timed out" if timed_out else "is unreachable", request_id,
        type(exc).__name__, exc, request_url or "no url")
    if timed_out:
        read_budget = ((getattr(request, "extensions", None) or {}).get("timeout") or {}).get("read")
        budget = f" (read timeout {read_budget:g} s)" if isinstance(read_budget, (int, float)) else ""
        detail = (f"{service_name} did not finish in time{budget}; it may still be working on the "
                  f"request. Request id: {request_id}.")
    else:
        detail = f"{service_name} is unavailable. Request id: {request_id}."
    # /v1 may only swap providers when the request never got to a backend. A
    # read timeout means one accepted the audio and is still working on it, so
    # repeating the job elsewhere would run it twice (minutes of GPU time).
    unreachable = isinstance(exc, (httpx.ConnectError, httpx.ConnectTimeout))
    return (UpstreamUnavailable if unreachable else HTTPException)(status_code=503, detail=detail)


def _passthrough_headers(response: httpx.Response) -> dict:
    """Return a filtered set of upstream headers safe to forward.

    Content-Length is intentionally NOT forwarded: adapters often re-serialize
    the body (e.g. normalized STT JSON), and a stale upstream length makes
    uvicorn fail with "Response content longer than Content-Length". Starlette
    recomputes the correct length from the actual body.
    """
    keep = {"content-type", "content-disposition"}
    return {
        key: value
        for key, value in response.headers.items()
        if key.lower() in keep or key.lower().startswith("x-")
    }


async def _extract_form_payload(request: Request) -> tuple[dict, Optional[list[tuple[str, tuple[str, bytes, str]]]]]:
    """Normalize a Starlette form request into httpx-compatible data/files payloads.

    Text fields are returned as a dict (repeated keys become lists): httpx
    multipart encoding only accepts dict-shaped ``data`` — passing a list of
    tuples makes AsyncClient raise "Attempted to send an sync request".
    """
    form = await request.form()

    data: dict = {}
    files: list[tuple[str, tuple[str, bytes, str]]] = []
    for key, value in form.multi_items():
        if hasattr(value, "filename"):
            content = await value.read()
            files.append((
                key,
                (value.filename or "upload.bin", content, value.content_type or "application/octet-stream"),
            ))
        else:
            text = str(value)
            if key in data:
                existing = data[key]
                if isinstance(existing, list):
                    existing.append(text)
                else:
                    data[key] = [existing, text]
            else:
                data[key] = text

    return data, files or None


# Shared, connection-pooled HTTP client. Reused across requests so we don't pay
# pool/connection setup on every proxied call. Recreated transparently if
# httpx.AsyncClient is swapped out (e.g. patched in unit tests).
_http_client: Optional[httpx.AsyncClient] = None
_http_client_factory = None

# Upstream uvicorn workers are started with --timeout-keep-alive 120; the client
# expiry must stay <= that, or the pool hands out sockets the server has already
# closed and every such request pays a silent retry.
_KEEPALIVE_EXPIRY_S = float(os.getenv("UPSTREAM_KEEPALIVE_EXPIRY", "115"))
_POOL_LIMITS = httpx.Limits(
    max_connections=int(os.getenv("UPSTREAM_MAX_CONNECTIONS", "100")),
    max_keepalive_connections=int(os.getenv("UPSTREAM_MAX_KEEPALIVE", "20")),
    keepalive_expiry=_KEEPALIVE_EXPIRY_S,
)


# Health probes must stay well under the UI's refresh cadence; a slow backend
# should show as unhealthy quickly rather than stalling the status row.
PROVIDER_HEALTH_TIMEOUT_S = float(os.getenv("PROVIDER_HEALTH_TIMEOUT", "6"))


def _timeout(read: float) -> httpx.Timeout:
    """Per-phase timeouts.

    A bare float applies the *whole* budget to each phase including connect, so a
    600 s synthesis budget also meant a 600 s wait for a dead upstream. Connect
    should fail fast; only the read phase needs the long budget.

    Every upstream call in this module goes through here. Passing a bare float
    to httpx directly is the bug this function exists to prevent, and it was
    still doing exactly that on nine call sites — including voice cloning and
    saved-voice TTS at 600 s, where stopping the qwen3 container left the
    browser's spinner running for ten minutes before the "unavailable" arrived.
    """
    return httpx.Timeout(connect=3.0, read=read, write=120.0, pool=10.0)


def _get_http_client() -> httpx.AsyncClient:
    """Return the process-wide pooled AsyncClient (per-call timeouts are passed explicitly)."""
    global _http_client, _http_client_factory
    if _http_client is None or _http_client_factory is not httpx.AsyncClient:
        _http_client = httpx.AsyncClient(limits=_POOL_LIMITS)
        _http_client_factory = httpx.AsyncClient
    return _http_client


async def _stream_upstream(
    method: str,
    url: str,
    *,
    display_name: str,
    extra_headers: Optional[dict] = None,
    read_timeout: float = 600.0,
    **request_kwargs,
) -> StreamingResponse:
    """Proxy an upstream response body through without buffering it.

    Reading `response.content` here would defeat the whole point of the
    providers' sentence-streaming endpoints: the backend would stream, and the
    gateway would sit on the bytes until the last one arrived. Instead the
    upstream response is opened in streaming mode and its raw chunks are handed
    straight to the client as they land.
    """
    client = _get_http_client()
    req = client.build_request(method, url, timeout=_timeout(read_timeout), **request_kwargs)
    try:
        upstream = await client.send(req, stream=True)
    except httpx.RequestError as exc:
        raise _build_upstream_request_error(display_name, exc) from exc

    if upstream.status_code >= 400:
        # Error bodies are small: read it, close the connection, and report.
        # The read itself can fail mid-body (upstream died while sending its
        # error); that must still surface as a 502 with the provider named,
        # not as an unhandled 500 from a raw httpx exception.
        try:
            await upstream.aread()
        except httpx.RequestError as exc:
            await upstream.aclose()
            raise _build_upstream_request_error(display_name, exc) from exc
        finally:
            await upstream.aclose()
        raise _build_error_from_response(upstream)

    headers = {k: v for k, v in upstream.headers.items() if k.lower().startswith("x-")}
    if extra_headers:
        headers.update(extra_headers)
    # Tell any intermediary not to buffer this, which would re-introduce the
    # exact latency the streaming path exists to remove.
    headers["Cache-Control"] = "no-cache"
    headers["X-Accel-Buffering"] = "no"

    return StreamingResponse(
        upstream.aiter_raw(),
        status_code=upstream.status_code,
        media_type=upstream.headers.get("content-type", "audio/wav"),
        headers=headers,
        # Mandatory: the client is process-wide, so without this every streamed
        # request leaks its pooled connection.
        background=BackgroundTask(upstream.aclose),
    )


async def _upstream_call(provider: dict, provider_id: str, method: str, path: str,
                         timeout: float, **request_kwargs) -> httpx.Response:
    """One request to a provider's internal URL, with the shared failure mapping.

    A transport failure becomes a 503 naming the provider and any >=400 answer
    is re-raised with the upstream's own status and `detail`. Every public helper
    below is a thin binding of this; they used to be eight near-identical copies,
    which is how one of them ended up mapping errors differently from the rest.
    """
    client = _get_http_client()
    url = f"{provider['internal_url']}{path}"
    try:
        if method == "GET":
            response = await client.get(url, timeout=_timeout(timeout))
        elif method == "DELETE":
            response = await client.delete(url, timeout=_timeout(timeout))
        else:
            response = await client.post(url, timeout=_timeout(timeout), **request_kwargs)
    except httpx.RequestError as exc:
        raise _build_upstream_request_error(provider.get("display_name", provider_id), exc) from exc
    if response.status_code >= 400:
        raise _build_error_from_response(response)
    return response


def _form_kwargs(data: dict, files: Optional[list]) -> dict:
    """Request kwargs for a multipart POST. An empty `files` must be omitted, not
    sent as `[]`, or httpx switches a data-only form to a multipart body."""
    kwargs: dict = {"data": data}
    if files:
        kwargs["files"] = files
    return kwargs


def _upstream_json(response: httpx.Response, provider: dict, expect: Optional[type] = None) -> Any:
    """Parse a successful upstream body as JSON, or raise a 502 naming the backend.

    A proxy that answers 200 with an HTML page, or a worker that dies mid-body,
    left every `response.json()` in this module to escape as an unhandled
    JSONDecodeError: a bare 500 that does not say which service misbehaved.
    `expect` also rejects a valid document of the wrong shape (a list where an
    object is read), which fails the same way one line later.
    """
    name = provider.get("display_name") or "Upstream service"
    try:
        payload = response.json()
    except ValueError as exc:
        raise HTTPException(status_code=502, detail=f"{name} returned an invalid response (not JSON)") from exc
    if expect is not None and not isinstance(payload, expect):
        raise HTTPException(
            status_code=502,
            detail=f"{name} returned an unexpected response ({type(payload).__name__}, expected {expect.__name__})",
        )
    return payload


async def _provider_get(provider_id: str, path: str, timeout: float = 30.0) -> httpx.Response:
    """Run a GET against a registered provider's internal URL."""
    return await _upstream_call(_get_provider(provider_id), provider_id, "GET", path, timeout)


async def _provider_delete(provider_id: str, path: str, timeout: float = 30.0) -> httpx.Response:
    """Run a DELETE against a registered provider's internal URL."""
    return await _upstream_call(_get_provider(provider_id), provider_id, "DELETE", path, timeout)


async def _provider_json_post(provider_id: str, path: str, payload: dict, timeout: float = 120.0) -> httpx.Response:
    """Run a JSON POST against a registered provider's internal URL."""
    return await _upstream_call(_get_provider(provider_id), provider_id, "POST", path, timeout, json=payload)


async def _provider_form_post_raw(provider_id: str, path: str, *, data: dict,
                                  files: list, timeout: float = 300.0) -> httpx.Response:
    """Multipart POST with an already-built payload.

    `_provider_form_post` re-parses the incoming Request, which suits the
    `/api/*` adapters that forward a browser form verbatim. The `/v1` router has
    already translated the request into OpenAI-shaped fields, so it needs to
    hand the payload over directly instead.
    """
    return await _upstream_call(
        _get_provider(provider_id), provider_id, "POST", path, timeout, **_form_kwargs(data, files))


async def _provider_form_post(provider_id: str, path: str, request: Request, timeout: float = 300.0) -> httpx.Response:
    """Run a multipart form POST against a registered provider's internal URL."""
    provider = _get_provider(provider_id)
    data, files = await _extract_form_payload(request)
    return await _upstream_call(provider, provider_id, "POST", path, timeout, **_form_kwargs(data, files))


def _training_provider() -> dict:
    return _get_provider("piper-training", kind="training")


def _training_json(response: httpx.Response, expect: Optional[type] = None) -> Any:
    return _upstream_json(response, _training_provider(), expect)


async def _proxy_training_get(path: str, timeout: float = 30.0):
    """Proxy a GET request to the training service."""
    return await _upstream_call(_training_provider(), "training service", "GET", path, timeout)


async def _proxy_training_delete(path: str, timeout: float = 30.0):
    """Proxy a DELETE request to the training service."""
    return await _upstream_call(_training_provider(), "training service", "DELETE", path, timeout)


async def _proxy_training_form_post(path: str, request: Request, timeout: float = 300.0):
    """Proxy a multipart form POST request to the training service."""
    provider = _training_provider()
    data, files = await _extract_form_payload(request)
    return await _upstream_call(provider, "training service", "POST", path, timeout, **_form_kwargs(data, files))


def _tts_text_schema(schema: dict) -> None:
    """The OpenAPI `maxLength` of a TTS text: the limit in force when the document is made."""
    schema["maxLength"] = MAX_TTS_CHARS


def _tts_text_within_limit(text: str) -> str:
    """`Field(max_length=MAX_TTS_CHARS)`, but with the limit in force now, not at import.

    Raises exactly what pydantic raises for max_length (type, message with its
    singular for 1, context), so the 422 body is the same byte for byte, and
    `_tts_text_schema` keeps the OpenAPI document as it was.
    """
    limit = MAX_TTS_CHARS
    if len(text) > limit:
        raise PydanticCustomError(
            "string_too_long",
            "String should have at most {max_length} character" + ("" if limit == 1 else "s"),
            {"max_length": limit},
        )
    return text


class FrontendTTSRequest(BaseModel):
    """Normalized text-to-speech request accepted by the frontend adapter.

    Every free-text field is bounded: these are forwarded to a GPU service, and
    an unbounded string is an unbounded job (a 2 MB `text` was accepted before).
    """

    provider: str = Field(max_length=64)
    text: str = Field(json_schema_extra=_tts_text_schema)
    voice: Optional[str] = Field(default=None, max_length=256)
    language: str = Field(default="auto", max_length=64)
    quality: Optional[str] = Field(default=None, max_length=32)
    gender: Optional[str] = Field(default=None, max_length=32)
    speed: Optional[float] = None
    instructions: Optional[str] = Field(default=None, max_length=4000)
    output_format: str = Field(default="wav", max_length=16)

    @field_validator("text")
    @classmethod
    def text_within_limit(cls, value: str) -> str:
        return _tts_text_within_limit(value)


_CJK_LANGUAGE_CODES = frozenset({"zh", "ja"})
_CJK_LANGUAGE_NAMES = frozenset({"chinese", "mandarin", "japanese"})


def _is_cjk_language(language: Optional[str]) -> bool:
    """True for a Chinese or Japanese tag or name ("zh", "zh-CN", "ja_JP", "Chinese").

    Magpie needs about three times the seconds per character for these. "auto" is
    not one of them: it means MAGPIE_DEFAULT_LANGUAGE, which the gateway does not
    know, so a deployment whose default is zh gets the shorter budget for text sent
    without a language.
    """
    value = (language or "").strip().lower().replace("_", "-")
    return value.split("-", 1)[0] in _CJK_LANGUAGE_CODES or value in _CJK_LANGUAGE_NAMES


def _build_tts_payload(
    provider_id: str,
    provider: dict,
    *,
    text: str,
    voice: Optional[str] = None,
    language: str = "auto",
    quality: Optional[str] = None,
    gender: Optional[str] = None,
    speed: Optional[float] = None,
    instructions: Optional[str] = None,
    output_format: str = "wav",
) -> tuple[dict, float]:
    """Translate a normalized TTS request into one provider's native body.

    Shared by `/api/tts` and the OpenAI-compatible `/v1/audio/speech`, and that
    sharing is the point: `/v1` used to build Piper's body unconditionally, so
    on a deployment with DEFAULT_TTS_PROVIDER=qwen3 it sent `language` and
    `voice` to a service whose fields are `lang` and `speaker`. Pydantic ignores
    unknown keys, so both were dropped in silence and every request came back as
    the default speaker reading German text in English — HTTP 200, wrong audio,
    no signal. The same failure mode the whisper.cpp language default had.

    Returns ``(payload, read_timeout_seconds)``.
    """
    contract = provider.get("contracts", {}).get("tts")
    if contract != "simple-json-tts-v1":
        raise HTTPException(
            status_code=400, detail=f"Unsupported TTS contract for provider {provider_id}"
        )

    if provider_id == "piper":
        payload = {
            "text": text,
            "output_format": output_format,
            "speed": speed if speed is not None else 1.0,
        }
        if voice:
            payload["voice"] = voice
        # "auto" means "the service decides": the gateway must not fill in a
        # language itself. Compared case-insensitively so /api/tts and /v1 agree
        # on every spelling of it (the old exact match forwarded "AUTO" as if it
        # were a locale).
        requested_language = (language or "").strip()
        if requested_language and requested_language.lower() != "auto":
            payload["language"] = requested_language
        if quality:
            payload["quality"] = quality
        if gender:
            payload["gender"] = gender
        return payload, 120.0

    if provider_id == "chatterbox":
        # No speed control in the chatterbox API; "auto" resolves to
        # CHATTERBOX_DEFAULT_LANGUAGE service-side. /api/tts streams from its
        # /tts-stream, so these 300 s are the longest wait between two chunks there,
        # not a budget for the whole text.
        return {"text": text, "language": language or "auto"}, 300.0

    if provider_id == "magpie":
        # No speed or instruction control in the Magpie API. "auto" (and a missing
        # voice, which is what /v1 sends for one of OpenAI's own voice names) resolves
        # to MAGPIE_DEFAULT_LANGUAGE / MAGPIE_DEFAULT_SPEAKER service-side; the gateway
        # must not fill either in itself. An unknown speaker or an unsupported
        # language is the service's 400, listing what it can do.
        payload = {"text": text, "language": language or "auto", "speaker": voice or "auto"}
        # Magpie declares no `tts_stream`: its /tts answers once every group of
        # sentences is done, so this read timeout bounds the whole job. A fixed 300 s
        # cut long texts off while the GPU was still on them (and was answered as
        # "unavailable"). 180 s cover the wait for a turn (TTS_QUEUE_TIMEOUT_S, 60 s),
        # a model reload (about 40-50 s) and a language's first-use text normalizer
        # (up to 48 s); then 0.1 s per character for de/en/es/fr (measured 0.04-0.05
        # on an RTX 4080, with headroom for a 5060 Ti) and 0.25 s for zh/ja (measured
        # 0.12-0.16). Never below the 600 s /v1 always gave it. /v1 takes
        # max(this, 600 s), so both routes get the same number.
        per_char = 0.25 if _is_cjk_language(language) else 0.1
        return payload, max(600.0, 180.0 + per_char * len(text))

    if provider_id == "qwen3":
        return {
            "text": text,
            "lang": _normalize_qwen3_language(language),
            "speaker": voice or "Vivian",
            "instruct": instructions or "",
        }, 120.0

    raise HTTPException(
        status_code=400, detail=f"Unsupported TTS contract for provider {provider_id}"
    )


class ProviderModelSelectionRequest(BaseModel):
    """Request body for selecting a provider model variant."""

    model: str = Field(max_length=256)


class ProviderVoiceDesignRequest(BaseModel):
    """Request body for provider-scoped voice design."""

    text: str = Field(json_schema_extra=_tts_text_schema)
    voice_description: str = Field(max_length=4000)
    # Forwarded verbatim; "auto" is resolved by the service (QWEN3_DEFAULT_LANGUAGE).
    lang: str = Field(default="auto", max_length=64)

    @field_validator("text")
    @classmethod
    def text_within_limit(cls, value: str) -> str:
        return _tts_text_within_limit(value)


async def _build_frontend_stt_payload(provider_id: str, form, contract: str) -> tuple[str, dict, list[tuple[str, tuple[str, bytes, str]]]]:
    """Translate normalized frontend STT form data into a provider-specific request.

    ``data`` is dict-shaped because httpx multipart encoding rejects sequence
    payloads when files are present.
    """
    audio = form.get("audio")
    if not hasattr(audio, "filename"):
        raise HTTPException(status_code=400, detail="Audio file not provided")

    filename = audio.filename or "audio.bin"
    content_type = audio.content_type or "application/octet-stream"
    content = await audio.read()
    raw_language = form.get("language")
    language = str(raw_language).strip() if raw_language is not None else ""
    if language.lower() == "auto":
        language = "auto"

    data: dict = {}
    if contract == "stt-form-v1":
        files = [("audio", (filename, content, content_type))]
        # "auto" is an answer, not an omission, and the services treat the two
        # differently: an explicit "auto" means detect the language, while a
        # missing field means STT_DEFAULT_LANGUAGE (German in the TrueNAS
        # profile). Dropping "auto" here, as this used to, forced every
        # Auto-Detect upload to the operator's default language while the live
        # microphone path (which sends "auto") detected. All stt-form-v1
        # backends accept it: whisper detects, qwen3-asr and parakeet detect,
        # canary (no language identification) takes its default. A request that
        # names no language still names none.
        if language:
            data["language"] = language
        return "/transcribe", data, files

    if contract == "openai-audio-transcriptions-v1":
        files = [("file", (filename, content, content_type))]
        data["response_format"] = "json"
        # Send "auto" EXPLICITLY rather than omitting the field. whisper.cpp's
        # server defaults to `std::string language = "en"` and only overrides it
        # when the form field is present, so omitting it here meant "auto"
        # silently transcribed German audio as English on every whisper-cpp
        # deployment. "auto" is an explicitly supported value there
        # (`-l LANG ... 'auto' for auto-detect`, and it is special-cased in the
        # server's language validation), so this restores real auto-detection.
        data["language"] = language if language else "auto"
        backend_path = _get_provider(provider_id).get("transcribe_path") or "/v1/audio/transcriptions"
        return backend_path, data, files

    raise HTTPException(status_code=400, detail=f"Unsupported STT contract for provider {provider_id}")


def _finite_float(value: Any) -> Optional[float]:
    """A finite float, or None for missing, non-numeric and NaN/inf values.

    Backends serialise these fields loosely (whisper.cpp sends `"1.75"` as a
    string), and a stray `"n/a"` used to turn a successful transcription into a
    500 at the very last step.
    """
    if value in (None, ""):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if result == result and result not in (float("inf"), float("-inf")) else None


def _normalize_frontend_stt_response(payload: dict, contract: str) -> dict:
    """Normalize provider transcription payloads into the shared browser-facing STT shape."""
    if not isinstance(payload, dict):
        raise HTTPException(status_code=502, detail="Speech-to-text backend returned an unexpected response")
    segments_payload = payload.get("segments")
    segments = []
    if isinstance(segments_payload, list):
        for segment in segments_payload:
            if not isinstance(segment, dict):
                continue
            segments.append({
                "start": _finite_float(segment.get("start")) or 0.0,
                "end": _finite_float(segment.get("end")) or 0.0,
                "text": segment.get("text", "") or "",
            })

    if contract == "openai-audio-transcriptions-v1":
        text = payload.get("text") or payload.get("transcript") or ""
    else:
        text = payload.get("text") or ""

    if not text and segments:
        text = " ".join(segment["text"] for segment in segments).strip()

    language = payload.get("language")
    normalized_duration = _finite_float(payload.get("duration"))

    return {
        "text": text,
        "segments": segments,
        "language": language or None,
        "duration": normalized_duration,
    }


# Path parameters are interpolated straight into the upstream URL, and Starlette
# hands them over percent-DECODED: `a%3Fb=1` arrives as `a?b=1` and became a query
# string on the backend, `x%23y` became a fragment, and `%2e%2e` a `..` segment
# (DELETE /model/.. is DELETE /). Job ids are uuid4 and voice ids are
# `[A-Za-z0-9_-]+` at the backends (piper `SAFE_NAME_RE`, qwen3
# `_SAFE_VOICE_RE`), so anything outside that set cannot name a real resource —
# and a set without `.` rules out `.` and `..` by construction.
_RESOURCE_ID_PATTERN = r"^[A-Za-z0-9_-]{1,128}$"


def _job_id_param():
    return PathParam(..., pattern=_RESOURCE_ID_PATTERN, max_length=128)


def _voice_id_param():
    return PathParam(..., pattern=_RESOURCE_ID_PATTERN, max_length=128)


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Render the main web UI page with service URLs injected into the template."""
    await _refresh_canary_languages()
    tts_providers, stt_providers, status_providers = _template_provider_lists()
    # Request-first signature. The legacy TemplateResponse(name, context) form
    # is not merely deprecated in Starlette 1.x, it is gone: the name slot takes
    # the request, so the context dict lands where the template name belongs and
    # the loader raises "unhashable type: 'dict'". Supported since Starlette
    # 0.29, so this works on the pinned 0.115.8 and on anything newer.
    return templates.TemplateResponse(request, "index.html", {
        # The object, not a pre-rendered string: the template encodes it with
        # `tojson`, which escapes `</script>`. A `json.dumps(...) | safe` string
        # did not, and PROVIDER_REGISTRY_JSON is operator-supplied.
        "provider_registry": PROVIDER_REGISTRY,
        "tts_provider_options": tts_providers,
        "stt_provider_options": stt_providers,
        "status_provider_options": status_providers,
        "default_tts_provider": PROVIDER_REGISTRY["ui"]["default_tts_provider"],
        "default_stt_provider": PROVIDER_REGISTRY["ui"]["default_stt_provider"],
        "app_version": APP_VERSION,
    })


@app.get("/api-docs", response_class=HTMLResponse)
async def api_docs():
    """Serve the interactive API documentation page."""
    with open(BASE_DIR / "static" / "api_docs.html", "r", encoding="utf-8") as f:
        content = f.read()
    return HTMLResponse(content=content)


@app.get("/health")
async def health():
    """Return service health status and configured backend URLs.

    Kept lightweight: Docker probes this every 30s, so it returns only the
    small service map. The full provider registry is available at /providers.
    """
    return {
        "status": "healthy",
        "services": {
            provider_id: provider["internal_url"]
            for provider_id, provider in PROVIDER_REGISTRY["providers"].items()
        },
    }


@app.get("/providers")
async def providers():
    """Return the UI provider registry and provider contracts."""
    await _refresh_canary_languages()
    return PROVIDER_REGISTRY


# --- provider health ----------------------------------------------------------
#
# The UI polls /api/health and every call fans out to every backend. Cached for a
# couple of seconds so N open tabs (and the /v1 fallback lookup) cost one round of
# probes rather than N, without making a recovering service look down for long.
HEALTH_CACHE_TTL_S = float(os.getenv("HEALTH_CACHE_TTL", "2"))
# Injectable so tests can move time instead of sleeping.
_health_clock = time.monotonic
# `epoch` counts registry changes: a round that started before one is not stored.
_health_cache: dict = {"at": None, "value": None, "inflight": None, "epoch": 0}


def _reset_provider_health() -> None:
    """Forget the cached round and stop joining a running one (the engines changed)."""
    _health_cache.update(at=None, value=None, inflight=None, epoch=_health_cache["epoch"] + 1)


def _reported_default_language(payload: Any) -> Optional[str]:
    """`default_language` from a backend body, when it is a short plain code or name."""
    if not isinstance(payload, dict):
        return None
    value = payload.get("default_language")
    if isinstance(value, str) and 0 < len(value.strip()) <= 64:
        return value.strip()
    return None


async def _probe_all_providers() -> dict:
    client = _get_http_client()
    providers_map = PROVIDER_REGISTRY.get("providers", {})

    async def probe(provider_id: str, provider: dict) -> tuple[str, dict]:
        url = f"{provider['internal_url']}{provider.get('health_endpoint', '/health')}"
        started = time.monotonic()
        try:
            response = await client.get(url, timeout=_timeout(PROVIDER_HEALTH_TIMEOUT_S))
            body = {}
            try:
                body = response.json()
            except Exception:
                pass
            if not isinstance(body, dict):
                body = {}
            result = {
                "healthy": response.status_code < 400,
                "status_code": response.status_code,
                "latency_ms": round((time.monotonic() - started) * 1000, 1),
                # Surfacing these lets the UI show *why* a service is not ready
                # instead of a bare red dot.
                "model_loaded": body.get("model_loaded"),
                # A service can be healthy with no model in memory once idle
                # unloading is on. "idle" is not "down" — the next request
                # loads it. Absent on providers that do not report residency.
                "model_resident": body.get("model_resident"),
                "model_size": body.get("model_size") or body.get("current_model"),
                "device": body.get("device"),
            }
            # The language the backend falls back to for "auto"; the UI names it
            # in the language selector. Only what a backend actually reports.
            default_language = _reported_default_language(body)
            if default_language:
                result["default_language"] = default_language
            return provider_id, result
        except Exception as exc:
            return provider_id, {
                "healthy": False,
                "status_code": None,
                "latency_ms": round((time.monotonic() - started) * 1000, 1),
                "error": type(exc).__name__,
            }

    results = await asyncio.gather(
        *(probe(pid, p) for pid, p in providers_map.items() if p.get("internal_url"))
    )
    return {"providers": dict(results)}


async def _cached_provider_health() -> dict:
    """Aggregate provider health, at most one probe round per TTL.

    Concurrent callers that arrive while a round is running join it instead of
    starting their own. The round runs as its own task and is awaited through
    `shield`, so a caller that disconnects mid-probe does not cancel it for the
    others.
    """
    cache = _health_cache
    now = _health_clock()
    if (cache["value"] is not None and cache["at"] is not None
            and 0 <= now - cache["at"] < HEALTH_CACHE_TTL_S):
        return cache["value"]

    loop = asyncio.get_running_loop()
    task = cache["inflight"]
    # A task from another event loop (tests, worker restarts) cannot be awaited here.
    if task is None or task.done() or task.get_loop() is not loop:
        task = loop.create_task(_probe_all_providers())
        cache["inflight"] = task
        epoch = cache["epoch"]

        def _store(finished: "asyncio.Task") -> None:
            if cache["inflight"] is finished:
                cache["inflight"] = None
            if cache["epoch"] != epoch:
                return  # probed the engines offered before a settings change
            if not finished.cancelled() and finished.exception() is None:
                cache["value"] = finished.result()
                cache["at"] = _health_clock()

        task.add_done_callback(_store)
    return await asyncio.shield(task)


@app.get("/api/health")
async def provider_health():
    """Probe every provider's health endpoint concurrently.

    The browser used to do this itself, one cross-origin request per provider,
    sequentially — so a single unreachable backend stalled the whole status row
    for its full timeout, and every backend port had to be reachable from the
    browser just to render an indicator.

    Doing it here means the browser makes one same-origin call, the probes run in
    parallel, and the stack no longer needs its backend ports published.
    """
    return await _cached_provider_health()


@app.post("/api/auth/check", status_code=204)
async def auth_check():
    """204 when the caller may use the state-changing API, 401 when `API_KEY` is set and was not sent.

    The request guard does the checking (this is a POST under /api, so it needs
    the key exactly when every other mutating call does); the handler only
    answers. The UI calls it before opening the live-transcription WebSocket,
    where a browser cannot react to a 401 the way it does for fetch(), so that
    the key is known before the socket is dialled.
    """
    return Response(status_code=204)


@app.get("/api/providers/{provider_id}/voices")
async def provider_voices(provider_id: str):
    """Return a normalized voice catalog for a TTS provider."""
    provider = _get_provider(provider_id, kind="tts")
    contract = provider.get("contracts", {}).get("voice_catalog")

    if contract == "voice-catalog-v1":
        response = await _provider_get(provider_id, "/voices", timeout=15.0)
        payload = _upstream_json(response, provider, dict)
        result = {
            "provider": provider_id,
            "contract": contract,
            "voices": _normalize_piper_voice_catalog(payload),
        }
        # What "auto" resolves to on this backend (PIPER_DEFAULT_LANGUAGE); the UI
        # names it in the language selector instead of guessing.
        default_language = _reported_default_language(payload)
        if default_language:
            result["default_language"] = default_language
        return result

    if contract == "speaker-catalog-v1":
        response = await _provider_get(provider_id, "/speakers", timeout=15.0)
        payload = _upstream_json(response, provider, dict)
        result = {
            "provider": provider_id,
            "contract": contract,
            "voices": _normalize_qwen3_voice_catalog(payload),
        }
        default_language = _reported_default_language(payload)
        if default_language:
            result["default_language"] = default_language
        return result

    raise HTTPException(status_code=400, detail=f"Provider {provider_id} does not expose a normalized voice catalog")


@app.get("/api/providers/{provider_id}/models")
async def provider_models(provider_id: str):
    """Return a normalized model catalog for providers that support model variants."""
    provider = _get_provider(provider_id)
    contract = provider.get("contracts", {}).get("model_catalog")
    if contract != "model-catalog-v1":
        raise HTTPException(status_code=400, detail=f"Provider {provider_id} does not expose a model catalog")

    response = await _provider_get(provider_id, "/models")
    models = _normalize_qwen3_model_catalog(_upstream_json(response, provider, dict))
    return {
        "provider": provider_id,
        "models": models,
        "current_model": next((model for model in models if model.get("is_current")), None),
    }


@app.post("/api/providers/{provider_id}/models/select")
async def provider_select_model(provider_id: str, request: ProviderModelSelectionRequest):
    """Switch the active model for a provider that supports model variants."""
    provider = _get_provider(provider_id)
    contract = provider.get("contracts", {}).get("model_selection")
    if contract != "model-selection-v1":
        raise HTTPException(status_code=400, detail=f"Provider {provider_id} does not support model switching")

    response = await _provider_json_post(provider_id, "/load_model", {"model": request.model}, timeout=300.0)
    payload = _upstream_json(response, provider, dict)
    model_info = payload.get("model_info")
    return {
        "provider": provider_id,
        "message": payload.get("message", ""),
        "model": {
            "id": request.model,
            "name": (model_info.get("name") if isinstance(model_info, dict) else None) or request.model,
        },
    }


@app.get("/api/providers/{provider_id}/status")
async def provider_status(provider_id: str):
    """Return provider status for providers that expose runtime metadata."""
    provider = _get_provider(provider_id)
    contract = provider.get("contracts", {}).get("runtime_status")
    if contract != "runtime-status-v1":
        raise HTTPException(status_code=400, detail=f"Provider {provider_id} does not expose a status endpoint")

    response = await _provider_get(provider_id, "/status")
    return _normalize_qwen3_runtime_status(_upstream_json(response, provider, dict), provider_id)


@app.post("/api/providers/{provider_id}/unload")
async def provider_unload(provider_id: str):
    """Ask a provider to release its model and free the memory now.

    Passes the upstream status and body through verbatim rather than going via
    `_provider_json_post`, which raises on any >=400 and would flatten the
    upstream body to its `detail`. A 409 here is a normal, actionable answer —
    the model is busy — and the caller needs the reference count that comes with
    it to know when retrying is worthwhile.
    """
    provider = _get_provider(provider_id)
    if not provider.get("capabilities") or "model_unload" not in provider["capabilities"]:
        raise HTTPException(
            status_code=400,
            detail=f"Provider {provider_id} does not support unloading",
        )

    client = _get_http_client()
    try:
        response = await client.post(f"{provider['internal_url']}/unload", timeout=_timeout(30.0))
    except httpx.RequestError as exc:
        raise _build_upstream_request_error(provider.get("display_name", provider_id), exc) from exc

    try:
        body = response.json()
    except ValueError:
        body = {"detail": response.text}
    if not isinstance(body, dict):
        # `**body` on a list or a bare string raised TypeError: a backend that
        # answered "ok" turned a successful unload into a 500.
        body = {"detail": body}
    if response.status_code >= 500:
        # A failure body is written for the operator (tracebacks, paths): log it
        # and give the caller the same sentence every other route gives.
        body = {"detail": _build_error_from_response(response).detail}
    elif response.status_code >= 400 and "detail" in body and _client_safe_detail(body["detail"]) is None:
        # A 409 "busy" with its reference count is a designed answer and passes
        # intact; only a `detail` that is not a plain sentence is replaced.
        body = {**body, "detail": _build_error_from_response(response).detail}
    return JSONResponse(status_code=response.status_code, content={"provider": provider_id, **body})


@app.get("/api/providers/{provider_id}/saved-voices")
async def provider_saved_voices(provider_id: str):
    """List saved voices for providers that support a voice library."""
    provider = _get_provider(provider_id)
    contract = provider.get("contracts", {}).get("saved_voices")
    if contract != "saved-voice-library-v1":
        raise HTTPException(status_code=400, detail=f"Provider {provider_id} does not support saved voices")

    response = await _provider_get(provider_id, "/voices")
    return _normalize_qwen3_saved_voice_library(_upstream_json(response, provider, dict), provider_id)


@app.get("/api/providers/{provider_id}/custom-voices")
async def provider_custom_voices(provider_id: str):
    """List managed custom voices for providers that support custom model management."""
    provider = _get_provider(provider_id)
    contract = provider.get("contracts", {}).get("managed_voices")
    if contract != "custom-voice-library-v1":
        raise HTTPException(status_code=400, detail=f"Provider {provider_id} does not support custom voice management")

    response = await _provider_get(provider_id, "/voices")
    normalized_voices = _normalize_piper_voice_catalog(_upstream_json(response, provider, dict))
    custom_voices = [voice for voice in normalized_voices if voice.get("kind") == "custom"]
    return {
        "provider": provider_id,
        "contract": contract,
        "voices": custom_voices,
    }


@app.post("/api/providers/{provider_id}/saved-voices")
async def provider_save_voice(provider_id: str, request: Request):
    """Create a saved voice entry for providers that support voice libraries."""
    provider = _get_provider(provider_id)
    contract = provider.get("contracts", {}).get("saved_voices")
    if contract != "saved-voice-library-v1":
        raise HTTPException(status_code=400, detail=f"Provider {provider_id} does not support saved voices")

    response = await _provider_form_post(provider_id, "/voices/save", request, timeout=300.0)
    return _upstream_json(response, provider)


@app.delete("/api/providers/{provider_id}/saved-voices/{voice_id}")
async def provider_delete_saved_voice(provider_id: str, voice_id: str = _voice_id_param()):
    """Delete a saved voice entry for providers that support voice libraries."""
    provider = _get_provider(provider_id)
    contract = provider.get("contracts", {}).get("saved_voices")
    if contract != "saved-voice-library-v1":
        raise HTTPException(status_code=400, detail=f"Provider {provider_id} does not support saved voices")

    response = await _provider_delete(provider_id, f"/voices/{voice_id}")
    return _upstream_json(response, provider)


@app.delete("/api/providers/{provider_id}/custom-voices/{voice_id}")
async def provider_delete_custom_voice(provider_id: str, voice_id: str = _voice_id_param()):
    """Delete a managed custom voice for providers that support custom model management."""
    provider = _get_provider(provider_id)
    contract = provider.get("contracts", {}).get("managed_voices")
    if contract != "custom-voice-library-v1":
        raise HTTPException(status_code=400, detail=f"Provider {provider_id} does not support custom voice management")

    response = await _provider_delete(provider_id, f"/voice/{voice_id}")
    return _upstream_json(response, provider)


@app.post("/api/providers/{provider_id}/saved-voices/{voice_id}/tts")
async def provider_saved_voice_tts(provider_id: str, request: Request, voice_id: str = _voice_id_param()):
    """Synthesize speech with a saved voice profile through the frontend adapter."""
    provider = _get_provider(provider_id)
    contract = provider.get("contracts", {}).get("saved_voices")
    if contract != "saved-voice-library-v1":
        raise HTTPException(status_code=400, detail=f"Provider {provider_id} does not support saved voices")

    response = await _provider_form_post(provider_id, f"/voices/{voice_id}/tts", request, timeout=600.0)
    headers = _passthrough_headers(response)
    headers["X-Provider"] = provider_id
    return Response(content=response.content, media_type=response.headers.get("content-type", "audio/wav"), headers=headers)


@app.post("/api/providers/{provider_id}/voice-clone")
async def provider_voice_clone(provider_id: str, request: Request):
    """Run provider-scoped voice cloning through a frontend adapter."""
    provider = _get_provider(provider_id)
    contract = provider.get("contracts", {}).get("voice_clone")
    if contract != "voice-clone-tts-v1":
        raise HTTPException(status_code=400, detail=f"Provider {provider_id} does not support voice cloning")

    form = await request.form()
    use_ref_text = bool(str(form.get("ref_text", "")).strip())
    backend_path = "/clone-with-ref-text" if use_ref_text else "/clone"

    response = await _provider_form_post(provider_id, backend_path, request, timeout=600.0)
    headers = _passthrough_headers(response)
    headers["X-Provider"] = provider_id
    return Response(content=response.content, media_type=response.headers.get("content-type", "audio/wav"), headers=headers)


@app.post("/api/providers/{provider_id}/voice-design")
async def provider_voice_design(provider_id: str, request: ProviderVoiceDesignRequest):
    """Run provider-scoped voice design through a frontend adapter."""
    provider = _get_provider(provider_id)
    contract = provider.get("contracts", {}).get("voice_design")
    if contract != "voice-design-tts-v1":
        raise HTTPException(status_code=400, detail=f"Provider {provider_id} does not support voice design")

    response = await _provider_json_post(provider_id, "/voice_design", request.model_dump(), timeout=600.0)
    headers = _passthrough_headers(response)
    headers["X-Provider"] = provider_id
    return Response(content=response.content, media_type=response.headers.get("content-type", "audio/wav"), headers=headers)


@app.websocket("/ws/stt")
async def frontend_ws_stt(websocket: WebSocket):
    """Relay the live-transcription WebSocket to the STT provider.

    Without this the browser has to dial the STT container's published port
    directly. getUserMedia requires a secure context, so any real deployment is
    HTTPS — and there a `ws://localhost:5001` handshake is hard-blocked as mixed
    content. Behind single-port ingress that port is not routable at all.

    Query string is forwarded, so ?provider=whisper selects the upstream.
    """
    import websockets

    # Neither CORSMiddleware nor the request guard runs on WebSocket routes, and
    # browsers do not apply the same-origin policy to WebSockets at all: any page
    # can dial this endpoint and stream the microphone through it. The same rule
    # as for state-changing HTTP requests applies — same origin, or listed.
    # Clients that send no Origin (scripts) are not a browser-borne risk.
    # The Host check applies here too: under DNS rebinding the page's Origin
    # equals its Host, so the Origin rule below cannot tell it from the real UI.
    policy = _policy
    if not _host_allowed(_request_host(websocket.headers, policy), policy):
        await websocket.close(code=1008)
        return

    origin = websocket.headers.get("origin")
    if origin and not _origin_permitted(origin, websocket.headers, policy):
        await websocket.close(code=1008)  # policy violation
        return

    # The HTTP middleware only sees HTTP scopes, so the API key has to be checked
    # here. The handshake is completed first and then closed with a reason: a
    # close before accept becomes a bare HTTP 403, which a browser reports as
    # code 1006 with no reason, and the UI could not tell "needs the key" from
    # "server down".
    if policy.require_key and not _websocket_key_ok(websocket):
        await websocket.accept(subprotocol=_websocket_subprotocol(websocket))
        await websocket.close(code=1008, reason="A valid API key is required")
        return

    provider_id = websocket.query_params.get("provider", "whisper")
    try:
        provider = _get_provider(provider_id, kind="stt")
    except HTTPException as exc:
        await websocket.close(code=1008, reason=str(exc.detail)[:120])
        return

    # Only stt-service implements /ws/transcribe. The other STT backends are
    # request/response only — qwen3-asr, parakeet and canary expose no
    # WebSocket at all, and whisper-cpp is the upstream whisper-server binary
    # with no Python layer. Without this check the relay happily dialled
    # ws://<backend>/ws/transcribe, and the caller got an opaque connection
    # failure instead of being told the provider cannot do this.
    if "live_transcribe" not in (provider.get("capabilities") or []):
        await websocket.close(
            code=1008,
            reason=f"provider '{provider_id}' does not support live transcription"[:120],
        )
        return

    upstream_url = provider["internal_url"].replace("http://", "ws://").replace("https://", "wss://") + "/ws/transcribe"

    await websocket.accept(subprotocol=_websocket_subprotocol(websocket))
    try:
        async with websockets.connect(
            upstream_url,
            max_size=None,        # audio frames are not size-bounded by the protocol
            ping_interval=20,
            open_timeout=10,
            # No Origin header, like every other server-to-server call. The browser's
            # Origin was validated above, against this service's own Host; forwarding
            # it made the STT service compare a foreign Origin with *its* Host
            # (stt-service:8000) and refuse the handshake, which is precisely what
            # its origin guard is for.
        ) as upstream:
            async def client_to_upstream():
                while True:
                    message = await websocket.receive()
                    if message.get("type") == "websocket.disconnect":
                        return
                    if message.get("bytes") is not None:
                        await upstream.send(message["bytes"])
                    elif message.get("text") is not None:
                        await upstream.send(message["text"])

            async def upstream_to_client():
                async for message in upstream:
                    if isinstance(message, bytes):
                        await websocket.send_bytes(message)
                    else:
                        await websocket.send_text(message)

            done, pending = await asyncio.wait(
                [asyncio.create_task(client_to_upstream()),
                 asyncio.create_task(upstream_to_client())],
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in pending:
                task.cancel()
            for task in done:
                # Surface a relay-side failure rather than closing silently.
                exc = task.exception()
                if exc:
                    raise exc
    except WebSocketDisconnect:
        pass
    except Exception as e:
        # The handshake with the browser already succeeded, so a plain close
        # here would reach the page as a normal 1000 — no `error` event fires
        # and the UI would keep claiming it is listening. Send an explicit
        # error frame, which the client already knows how to render.
        logger.warning(f"WebSocket relay to {provider_id} failed: {e}")
        try:
            await websocket.send_json({
                "type": "error",
                "error": f"Live transcription unavailable: {provider_id} could not be reached.",
            })
        except Exception:
            pass
        try:
            await websocket.close(code=1011)
        except Exception:
            pass
        return
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


@app.post("/api/stt")
async def frontend_stt(request: Request):
    """Transcribe audio through a normalized frontend STT adapter."""
    form = await request.form()
    provider_id = str(form.get("provider", "")).strip()
    if not provider_id:
        raise HTTPException(status_code=400, detail="Provider not provided")

    provider = _get_provider(provider_id, kind="stt")
    contract = provider.get("contracts", {}).get("transcribe") or "stt-form-v1"
    backend_path, data, files = await _build_frontend_stt_payload(provider_id, form, contract)

    client = _get_http_client()
    try:
        response = await client.post(f"{provider['internal_url']}{backend_path}", data=data, files=files, timeout=_timeout(300.0))
    except httpx.RequestError as exc:
        raise _build_upstream_request_error(provider.get("display_name", provider_id), exc) from exc

    if response.status_code >= 400:
        raise _build_error_from_response(response)

    headers = _passthrough_headers(response)
    headers["X-Provider"] = provider_id
    return JSONResponse(
        content=_normalize_frontend_stt_response(_upstream_json(response, provider, dict), contract),
        headers=headers,
    )


@app.post("/api/tts")
async def frontend_tts(request: FrontendTTSRequest, http_request: Request):
    """Synthesize speech through a normalized frontend TTS adapter."""
    provider = _get_provider(request.provider, kind="tts")

    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text not provided")

    payload, read_timeout = _build_tts_payload(
        request.provider,
        provider,
        text=request.text,
        voice=request.voice,
        language=request.language,
        quality=request.quality,
        gender=request.gender,
        speed=request.speed,
        instructions=request.instructions,
        output_format=request.output_format,
    )

    # Prefer the provider's chunked streaming endpoint when it declares one:
    # time-to-first-audio then tracks the first sentence instead of the whole
    # text. The body is proxied without buffering (see _stream_upstream) —
    # buffering here would throw the entire benefit away.
    path = "/tts-stream" if "tts_stream" in provider.get("contracts", {}) else "/tts"

    # Raced against the browser leaving (a closed tab, a reload): until the backend
    # answers, nothing else would notice, and a non-streaming backend generates the
    # whole text first. Cancelling closes the upstream connection.
    return await openai_router.unless_client_gone(http_request, _stream_upstream(
        "POST",
        f"{provider['internal_url']}{path}",
        display_name=provider.get("display_name", request.provider),
        json=payload,
        read_timeout=read_timeout,
        extra_headers={"X-Provider": request.provider},
    ))


@app.get("/api/training/deployment-targets")
async def frontend_training_deployment_targets():
    """Return training deployment targets through the frontend adapter."""
    response = await _proxy_training_get("/deployment-targets")
    return _training_json(response)


@app.post("/api/training/train")
async def frontend_training_start(request: Request):
    """Start a training job through the frontend adapter."""
    response = await _proxy_training_form_post("/train", request, timeout=300.0)
    return _training_json(response)


@app.post("/api/training/train-from-dataset")
async def frontend_training_from_dataset(request: Request):
    """Start a dataset-backed training job through the frontend adapter."""
    response = await _proxy_training_form_post("/train-from-dataset", request, timeout=120.0)
    return _training_json(response)


@app.post("/api/training/resume")
async def frontend_training_resume(request: Request):
    """Resume a training job through the frontend adapter."""
    response = await _proxy_training_form_post("/resume-training", request, timeout=120.0)
    return _training_json(response)


@app.get("/api/training/jobs")
async def frontend_training_jobs():
    """List training jobs through the frontend adapter."""
    response = await _proxy_training_get("/jobs")
    return _normalize_training_jobs_payload(_training_json(response))


@app.get("/api/training/status/{job_id}")
async def frontend_training_status(job_id: str = _job_id_param()):
    """Get a single training job status through the frontend adapter."""
    response = await _proxy_training_get(f"/status/{job_id}")
    return _normalize_training_job(_training_json(response, dict))


@app.post("/api/training/export/{job_id}")
async def frontend_training_export(request: Request, job_id: str = _job_id_param()):
    """Export and optionally deploy a model bundle through the frontend adapter."""
    response = await _proxy_training_form_post(f"/export/{job_id}", request, timeout=120.0)
    return _normalize_training_export_response(_training_json(response, dict))


@app.get("/api/training/download/{job_id}")
async def frontend_training_download(job_id: str = _job_id_param()):
    """Download an exported model bundle through the frontend adapter."""
    response = await _proxy_training_get(f"/download/{job_id}", timeout=120.0)
    return Response(content=response.content, media_type=response.headers.get("content-type", "application/octet-stream"), headers=_passthrough_headers(response))


@app.delete("/api/training/model/{job_id}")
async def frontend_training_delete_model(job_id: str = _job_id_param()):
    """Delete a trained model through the frontend adapter."""
    response = await _proxy_training_delete(f"/model/{job_id}")
    return _training_json(response)


@app.delete("/api/training/job/{job_id}")
async def frontend_training_cancel_job(job_id: str = _job_id_param()):
    """Cancel a training job through the frontend adapter."""
    response = await _proxy_training_delete(f"/job/{job_id}")
    return _training_json(response)


# --- live settings (the Settings page's files) ---------------------------------------
#
# settings_store.py reads the folder /app/settings (TTS_STT_SETTINGS_DIR in tests):
# gateway.json holds the values saved in the Settings page, keys.json the API keys
# made there and whether a key is required. Per setting, the first that applies wins:
#   1. SETTINGS_LOCKED_KEYS names it: the app YAML value;
#   2. a valid value in gateway.json, unless ENABLE_SETTINGS_UI=false or a SAFE-MODE
#      file in the folder says to ignore that file;
#   3. the app YAML (`_ENV_VALUES`), which falls back to the code default.
# keys.json counts in every mode, and a damaged one fails closed: a key is required
# and only the YAML key is accepted, as a client.
#
# `_settings_tick()` runs first on every HTTP and WebSocket request
# (_SettingsMiddleware), so a change saved by any worker, or by hand, is in force
# from the next request on, in every worker, without a restart.

# The store's wall clock (commit-confirm deadlines, one-time codes); injectable so
# tests can move time instead of sleeping.
_settings_clock = time.time
_settings_store = settings_store.SettingsStore.from_environment(clock=lambda: _settings_clock())
# What `_apply` last put in force: the store state it was given and its effective values.
_applied_state: Optional[settings_store.SettingsState] = None
_applied_values: Mapping[str, Any] = MappingProxyType({})


def _overrides_in_force(values: Mapping[str, Any]) -> dict[str, Any]:
    """The saved values of gateway.json that win over the app YAML: all but the locked keys."""
    return {key: value for key, value in values.items() if key not in SETTINGS_LOCKED_KEYS}


def _file_values_in_force(state: settings_store.SettingsState) -> dict[str, Any]:
    """The saved gateway.json values that apply now: none while the file is to be ignored."""
    if not ENABLE_SETTINGS_UI or state.safe_mode:
        return {}
    return _overrides_in_force(state.preferences.values)


def _access_in_force(keys: settings_store.KeysState) -> tuple[bool, str]:
    """(whether a key is required, the deployment key's role), from keys.json and the YAML."""
    if keys.fail_closed:
        return True, settings_schema.ROLE_CLIENT
    require = keys.require_key and "require_key" not in SETTINGS_LOCKED_KEYS
    role = (settings_schema.ROLE_ADMIN if "deployment_key_role" in SETTINGS_LOCKED_KEYS
            else keys.deployment_key_role)
    return require or bool(API_KEY), role


def _effective(state: settings_store.SettingsState) -> dict[str, Any]:
    """Every setting's value in force: the 16 of gateway.json, require_key and deployment_key_role."""
    values = dict(_ENV_VALUES)
    values.update(_file_values_in_force(state))
    values["require_key"], values["deployment_key_role"] = _access_in_force(state.keys)
    return values


def _apply(state: settings_store.SettingsState, *, initial: bool = False) -> None:
    """Put the values of `state` in force in this worker.

    Everything is built first and then assigned without an `await` in between, so
    no request sees half of a change. A request already running keeps what it
    started with: the policy and the registry's entries are replaced, never
    changed in place, and the registry object itself (which the /v1 router holds)
    has its contents swapped.
    """
    global _policy, _keyring, _applied_state, _applied_values, _registry_built_from
    global TRUST_PROXY_HEADERS, MAX_UPLOAD_MB, MAX_REQUEST_BYTES, MAX_TTS_CHARS, MAX_CONCURRENT_UPLOADS
    global ENABLE_WHISPER_CPP, ENABLE_PARAKEET_ASR, ENABLE_CANARY_ASR, ENABLE_CHATTERBOX_TTS
    global ENABLE_MAGPIE_TTS, ENABLE_TRAINING

    values = _effective(state)
    policy = _build_access_policy(values)
    keyring = _build_key_ring(state.keys.keys, values["deployment_key_role"])
    engines = {key: values[key] for key in _ENGINE_KEYS}
    rebuild = any(not settings_schema.same_value(engines[key], _registry_built_from[key]) for key in _ENGINE_KEYS)
    registry = _build_provider_registry(values, values) if rebuild else None
    previous, previous_keyring, previous_tts_chars = _applied_values, _keyring, MAX_TTS_CHARS

    _policy, _keyring = policy, keyring
    TRUST_PROXY_HEADERS = policy.trust_proxy_headers
    MAX_UPLOAD_MB = float(values["MAX_UPLOAD_MB"])
    MAX_REQUEST_BYTES = int(MAX_UPLOAD_MB * 1024 * 1024)
    MAX_TTS_CHARS = int(values["MAX_TTS_CHARS"])
    MAX_CONCURRENT_UPLOADS = int(values["MAX_CONCURRENT_UPLOADS"])
    _upload_slots.limit = MAX_CONCURRENT_UPLOADS
    openai_router._ffmpeg_slots.limit = int(values["MAX_CONCURRENT_FFMPEG"])
    ENABLE_WHISPER_CPP = values["ENABLE_WHISPER_CPP"]
    ENABLE_PARAKEET_ASR = values["ENABLE_PARAKEET_ASR"]
    ENABLE_CANARY_ASR = values["ENABLE_CANARY_ASR"]
    ENABLE_CHATTERBOX_TTS = values["ENABLE_CHATTERBOX_TTS"]
    ENABLE_MAGPIE_TTS = values["ENABLE_MAGPIE_TTS"]
    ENABLE_TRAINING = values["ENABLE_TRAINING"]
    if registry is not None:
        PROVIDER_REGISTRY.clear()
        PROVIDER_REGISTRY.update(registry)
        _registry_built_from = engines
        _canary_languages["next_at"] = None
        _reset_provider_health()
    if MAX_TTS_CHARS != previous_tts_chars:
        app.openapi_schema = None   # regenerated with the new maxLength when next asked for
    _applied_state, _applied_values = state, MappingProxyType(values)

    if not initial:
        changed = [key for key in values if not settings_schema.same_value(previous.get(key), values[key])]
        if keyring != previous_keyring:
            changed.append("API keys")
        if changed:
            logger.info("settings: now in force in this worker: %s", ", ".join(changed))


def _settings_tick() -> None:
    """Apply what changed in the settings folder, and undo an overdue unconfirmed change.

    Stats the files and re-reads only one that changed (settings_store). Never
    raises: if anything fails, the settings in force stay as they are.
    """
    try:
        _settings_store.refresh()
        _settings_store.expire_pending()
        state = _settings_store.state
        if state is not _applied_state:
            _apply(state)
    except Exception:
        logger.exception("settings: could not apply the saved settings; the ones in force stay")


def _log_settings_at_start(state: settings_store.SettingsState) -> None:
    """Say which values come from the Settings page instead of the app YAML, and why any are ignored."""
    folder = state.mount.directory
    saved = list(state.preferences.values)
    in_force = list(_file_values_in_force(state))
    if saved and not ENABLE_SETTINGS_UI:
        logger.warning("ENABLE_SETTINGS_UI=false: %s saved in the Settings page %s ignored; the app YAML "
                       "applies. API keys made there stay in force.", ", ".join(saved),
                       "is" if len(saved) == 1 else "are")
    elif saved and state.safe_mode:
        logger.warning("%s exists: %s saved in the Settings page %s ignored; the app YAML applies. API keys "
                       "made there stay in force.", os.path.join(folder, settings_store.SAFE_MODE_FILE),
                       ", ".join(saved), "is" if len(saved) == 1 else "are")
    if in_force:
        logger.info("settings: %s from %s (revision %d), not from the app YAML",
                    ", ".join(in_force), os.path.join(folder, settings_schema.FILE_PREFERENCES),
                    state.preferences.revision)
    locked = [key for key in saved if key in SETTINGS_LOCKED_KEYS]
    if locked and ENABLE_SETTINGS_UI and not state.safe_mode:
        logger.info("settings: SETTINGS_LOCKED_KEYS keeps the app YAML value of %s", ", ".join(locked))
    if state.keys.keys:
        logger.info("settings: %d API key(s) from %s", len(state.keys.keys),
                    os.path.join(folder, settings_schema.FILE_KEYS))
    if not state.mount.writable:
        logger.info("settings: the Settings page cannot save here (%s)", state.mount.detail)


# One read at start, an unconfirmed change whose time is up undone (as on every
# request), and everything put in force before the first request.
_settings_store.expire_pending()
_apply(_settings_store.state, initial=True)
_log_settings_at_start(_settings_store.state)


# --- the Settings API ------------------------------------------------------------------
#
# Internal: not in the OpenAPI document, and only for the page at /settings. Every
# route here is behind `_settings_guard` (an admin key once the server is claimed,
# JSON of at most 64 KiB, this UI's own origin). Writes go through settings_store: a
# lock the workers share, a check of the revision the change is based on, atomic
# files, history and audit. The answering worker puts a write in force before it
# answers; the others do on their next request.
#
#   GET    /settings                 the page (an empty shell; settings.js asks the API)
#   GET    /api/settings             everything the page shows
#   PUT    /api/settings             {base_revision, set, reset, acknowledge, dry_run}
#   POST   /api/settings/confirm     {revision}: keep a TRUSTED_ORIGINS / TRUST_PROXY_HEADERS change
#   POST   /api/settings/restore     {history_id, base_revision, acknowledge, dry_run}
#   POST   /api/settings/discard     {base_revision, dry_run}: back to the app YAML
#   GET    /api/settings/engines     which optional engines run, answer or are not installed
#   PUT    /api/settings/access      {base_revision, require_key, deployment_key_role}
#   POST   /api/settings/keys        {name, role, key} -> 201
#   DELETE /api/settings/keys/{id}
#   POST   /api/settings/claim-code  {} -> 202; the code goes to the container log only
#   POST   /api/settings/claim       {code, name, key, require_key} -> 201

_SETTINGS_PAGE_CSP = ("default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
                      "connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'")
# The one place a one-time code is written: the container log. Not the audit logger,
# whose lines also go to audit.jsonl.
_claim_logger = logging.getLogger("tts_stt.settings.claim")
_LOCKED_MESSAGE = "Locked by SETTINGS_LOCKED_KEYS in the app YAML."
_REGISTRY_JSON_MESSAGE = "Defined by PROVIDER_REGISTRY_JSON in the app YAML; it cannot be switched here."
_API_KEY_FORCES_MESSAGE = "The app YAML sets API_KEY, so a key is always required."
# The lines that mount the settings folder, for the page to show while it is missing.
_SETTINGS_MOUNT_LINES = {
    "truenas": ['- "${APP_DATA_DIR:?set dataset path}/settings:/app/settings"',
                '- "${APP_DATA_DIR:?set dataset path}/backend-settings:/app/backend-settings"'],
    "compose": ["- ${SETTINGS_DIR:-${APP_DATA_DIR:-.}/settings}:/app/settings",
                "- ${BACKEND_SETTINGS_DIR:-${APP_DATA_DIR:-.}/backend-settings}:/app/backend-settings"],
}
_STORE_ERROR_STATUS = {
    "settings_not_mounted": 409, "settings_write_failed": 500, "settings_busy": 503,
    "revision_conflict": 409, "invalid_input": 400, "history_not_found": 404, "key_not_found": 404,
    "key_limit_reached": 409, "last_admin_credential": 409, "no_key_configured": 409,
    "keys_file_damaged": 409, "invalid_code": 403, "rate_limited": 429,
}


class _SettingsRefused(Exception):
    """Raised by a store `check` to stop a write; `response` is the answer."""

    def __init__(self, response: JSONResponse, code: str):
        super().__init__(code)
        self.response, self.code = response, code


def _settings_claimed() -> bool:
    """Is there an admin credential: an admin key made in the page, or the app YAML key as admin?"""
    state = _applied_state
    if state is not None and any(record.role == settings_schema.ROLE_ADMIN for record in state.keys.keys):
        return True
    return bool(API_KEY) and _applied_values.get("deployment_key_role") == settings_schema.ROLE_ADMIN


def _settings_actor(request: Request, credential: Optional[_Credential], name: Optional[str] = None) -> settings_store.Actor:
    """Who asks, for the audit trail and the history: address, Host and the credential's name."""
    return settings_store.Actor(
        ip=_client_address(request.scope), host=request.headers.get("host", ""),
        credential=name if name is not None else (credential.name if credential else ""))


def _store_refusal(exc: settings_store.SettingsStoreError) -> JSONResponse:
    extra: dict[str, Any] = {}
    headers: dict[str, str] = {}
    if isinstance(exc, settings_store.InvalidInput):
        extra["errors"] = exc.errors
    if isinstance(exc, settings_store.RevisionConflict):
        extra["revision"] = exc.current_revision
    if isinstance(exc, settings_store.RateLimited):
        headers["Retry-After"] = str(exc.retry_after)
    if isinstance(exc, settings_store.LockTimeout):
        headers["Retry-After"] = "1"
    return _settings_error(_STORE_ERROR_STATUS.get(exc.code, 400), exc.message, exc.code, headers=headers, **extra)


def _apply_saved() -> None:
    """Put a write in force in this worker before answering it (never raises: the write is
    done, and the next request's `_settings_tick` tries again)."""
    try:
        _apply(_settings_store.state)
    except Exception:
        logger.exception("settings: could not apply the saved settings; the ones in force stay")


def _refused_write(refused: _SettingsRefused, request: Request, actor: settings_store.Actor) -> JSONResponse:
    if refused.code == "would_lock_out":
        _settings_store.audit("refused", actor=actor, result="409 would_lock_out", method=request.method,
                              path=request.url.path)
    return refused.response


def _preferences_locked_out(state: settings_store.SettingsState) -> Optional[JSONResponse]:
    """Why gateway.json cannot be changed now (as the answer), or None."""
    if not ENABLE_SETTINGS_UI:
        return _settings_error(403, "ENABLE_SETTINGS_UI=false in the app YAML: the Settings page is read-only.",
                               "settings_disabled")
    if state.safe_mode:
        return _settings_error(403, "The settings folder holds a SAFE-MODE file: saved values are ignored and "
                                    "cannot be changed until it is deleted.", "settings_disabled")
    if not state.mount.writable:
        return _store_refusal(settings_store.NotWritable())
    return None


def _keys_locked_out(state: settings_store.SettingsState) -> Optional[JSONResponse]:
    """Why keys.json cannot be changed now (as the answer), or None. SAFE-MODE does not stop it."""
    if not ENABLE_SETTINGS_UI:
        return _settings_error(403, "ENABLE_SETTINGS_UI=false in the app YAML: the Settings page is read-only.",
                               "settings_disabled")
    if not state.mount.writable:
        return _store_refusal(settings_store.NotWritable())
    return None


_FIELD_CHECKS: dict[str, tuple[Any, str]] = {
    "base_revision": (lambda v: isinstance(v, int) and not isinstance(v, bool) and v >= 0,
                      "Expected the revision the change is based on."),
    "revision": (lambda v: isinstance(v, int) and not isinstance(v, bool) and v >= 0, "Expected a revision."),
    "set": (lambda v: isinstance(v, dict), "Expected an object of settings."),
    "reset": (lambda v: isinstance(v, list) and all(isinstance(i, str) for i in v), "Expected a list of settings."),
    "acknowledge": (lambda v: isinstance(v, list) and all(isinstance(i, str) for i in v),
                    "Expected a list of warnings."),
    "dry_run": (lambda v: isinstance(v, bool), "Expected true or false."),
    "require_key": (lambda v: isinstance(v, bool), "Expected true or false."),
    "deployment_key_role": (lambda v: isinstance(v, str), "Expected admin or client."),
    "history_id": (lambda v: isinstance(v, str), "Expected the id of a saved version."),
    # checked by the store, with messages that never repeat a key or a code
    "name": (lambda v: True, ""), "role": (lambda v: True, ""), "key": (lambda v: True, ""),
    "code": (lambda v: True, ""),
}


async def _settings_body(request: Request, *, required: tuple[str, ...] = (),
                         optional: tuple[str, ...] = ()) -> Any:
    """The request's JSON object with its fields checked, or the refusal (a JSONResponse)."""
    raw = await request.body()
    try:
        body = settings_schema.parse_json_strict(raw) if raw.strip() else {}
    except settings_schema.DuplicateKeyError:
        return _settings_error(400, "A name appears twice in the request.", "duplicate_key")
    except ValueError:
        return _settings_error(400, "The request is not valid JSON.", "invalid_json")
    if not isinstance(body, dict):
        return _settings_error(400, "The request has to be a JSON object.", "invalid_json")
    errors: dict[str, str] = {}
    for name, value in body.items():
        if name not in required and name not in optional:
            errors[name[:64]] = "Unknown field."
            continue
        check, message = _FIELD_CHECKS[name]
        if not check(value):
            errors[name] = message
    for name in required:
        if name not in body:
            errors[name] = "Required."
    if errors:
        return _settings_error(400, "The request is not valid.", "invalid_input", errors=errors)
    return body


def _override_engine_kinds() -> dict[str, str]:
    """{provider id: kind} of PROVIDER_REGISTRY_JSON's entries, which are offered whatever a flag says."""
    providers = _REGISTRY_OVERRIDE.get("providers") if isinstance(_REGISTRY_OVERRIDE, dict) else None
    if not isinstance(providers, dict):
        return {}
    return {provider: entry["kind"] for provider, entry in providers.items()
            if isinstance(entry, dict) and isinstance(entry.get("kind"), str)}


def _registry_toggle_errors(keys) -> dict[str, str]:
    overridden = _registry_override_providers()
    return {key: _REGISTRY_JSON_MESSAGE for key in keys
            if key in settings_schema.ENGINE_FLAGS and settings_schema.ENGINE_FLAGS[key][0] in overridden}


def _candidate_values(overrides: Mapping[str, Any]) -> dict[str, Any]:
    """The values in force if `overrides` were what gateway.json holds (keys.json as it is)."""
    values = dict(_ENV_VALUES)
    values.update(_overrides_in_force(overrides))
    values["require_key"], values["deployment_key_role"] = _access_in_force(_settings_store.state.keys)
    return values


def _lockout_reason(candidate: _AccessPolicy, headers: Headers) -> Optional[str]:
    """Why `candidate` would refuse the request that asks for it, or None.

    The request's own Host, X-Forwarded-Host (believed only under the candidate's
    TRUST_PROXY_HEADERS) and Origin are judged by the Settings rules of the candidate.
    """
    candidate = dataclasses.replace(candidate, allow_any_host=False)
    host = _request_host(headers, candidate)

    def name(value: Optional[str]) -> str:
        return repr(_parse_host_header(value or "") or (value or "")[:100])

    match = _HOST_PATTERN.match((headers.get("host") or "").strip())
    port = match.group("port") if match and match.group("port") else "3000"
    elsewhere = f"make the change from the server's IP address (http://<IP address>:{port}/settings)"
    if not _host_allowed(host, candidate):
        return f"You are connected as {name(host)}; this change would refuse that name. Keep it, or {elsewhere}."
    origin = headers.get("origin")
    if origin is not None and not _is_same_origin(origin, headers, candidate):
        now = _request_host(headers)
        if now != host:
            return (f"This page reaches the server as {name(now)} (X-Forwarded-Host); after this change the server "
                    f"would see {name(host)} and refuse the page's changes. Keep it, or {elsewhere}.")
        return (f"This page runs at {origin[:100]!r}; after this change the server would refuse its changes. "
                f"Keep it, or {elsewhere}.")
    return None


def _write_check(headers: Headers, *, engines: bool = True):
    """The store's `check` for a gateway.json write: runs under the lock with the exact new values."""

    def check(values: Mapping[str, Any], changed: tuple[str, ...]) -> None:
        candidate = _candidate_values(values)
        if engines and any(key in _ENGINE_KEYS for key in changed):
            offered = settings_schema.offered_engines(candidate, _override_engine_kinds())
            errors = settings_schema.check_engine_defaults(candidate, offered)
            if errors:
                raise _SettingsRefused(_settings_error(400, "Some values are not valid.", "invalid_input",
                                                       errors=errors), "invalid_input")
        if any(key in settings_schema.GUARD_KEYS for key in changed):
            reason = _lockout_reason(_build_access_policy(candidate), headers)
            if reason:
                raise _SettingsRefused(_settings_error(409, reason, "would_lock_out"), "would_lock_out")

    return check


def _confirm_view(pending: Optional[settings_store.Pending]) -> Optional[dict]:
    if pending is None:
        return None
    return {"revision": pending.revision, "deadline": pending.deadline,
            "seconds_left": max(0, math.ceil(pending.deadline - _settings_clock())),
            "keys": sorted(pending.previous)}


def _write_answer(result: settings_store.WriteResult, before: Mapping[str, Any], warnings=(),
                  notes: Optional[Mapping[str, Any]] = None) -> dict:
    """What a gateway.json write (or its dry run) answers."""
    after = _candidate_values(result.values)
    started = result.pending is not None and result.pending.revision == result.revision
    return {
        "revision": result.revision,
        "changed": list(result.changed),
        "dry_run": result.dry_run,
        "written": result.written,
        # the engines on offer changed: the main page shows them after a reload
        "reload_main_ui": any(not settings_schema.same_value(after[key], before.get(key)) for key in _ENGINE_KEYS),
        # a TRUSTED_ORIGINS / TRUST_PROXY_HEADERS change: POST /api/settings/confirm {revision} in time
        "confirm": {**_confirm_view(result.pending), "window": settings_store.CONFIRM_WINDOW_S} if started else None,
        "warnings": [warning.as_dict() for warning in warnings],
        "notes": {key: list(texts) for key, texts in (notes or {}).items()},
        "dropped": dict(result.dropped),
    }


def _jsonable(value: Any) -> Any:
    return list(value) if isinstance(value, tuple) else value


def _frontend_workers() -> int:
    """FRONTEND_WORKERS as entrypoint.sh reads it, for the page's worst-case memory hint."""
    raw = os.getenv("FRONTEND_WORKERS", "").strip()
    return int(raw) if raw.isdigit() and int(raw) > 0 else 2


def _setting_view(item: dict, state: settings_store.SettingsState, in_force: Mapping[str, Any],
                  overridden: frozenset) -> dict:
    """One field of the page: its value in force, where that comes from, and what else is known."""
    key = item["key"]
    preferences, keys = state.preferences, state.keys
    locked = key in SETTINGS_LOCKED_KEYS
    read_only = _LOCKED_MESSAGE if locked else None
    if key in settings_schema.PREFERENCE_KEYS:
        saved = key in preferences.values
        source = "locked" if locked else "saved" if key in in_force else "yaml" if key in _ENV_SET else "default"
        if read_only is None and key in settings_schema.ENGINE_FLAGS and settings_schema.ENGINE_FLAGS[key][0] in overridden:
            read_only = _REGISTRY_JSON_MESSAGE
        extra = {"yaml_value": _jsonable(_ENV_VALUES[key]), "yaml_set": key in _ENV_SET, "saved": saved,
                 "saved_value": _jsonable(preferences.values[key]) if saved else None,
                 "ignored": saved and key not in in_force, "dropped": preferences.dropped.get(key)}
    else:
        stored = keys.revision > 0 and not keys.fail_closed
        if locked:
            source = "locked"
        elif keys.fail_closed:
            source, read_only = "fail_closed", keys.problem
        elif key == "require_key" and API_KEY:
            source, read_only = "yaml", _API_KEY_FORCES_MESSAGE
        else:
            source = "saved" if stored else "default"
        yaml_value = bool(API_KEY) if key == "require_key" else settings_schema.ROLE_ADMIN
        saved_value = (keys.require_key if key == "require_key" else keys.deployment_key_role) if stored else None
        extra = {"yaml_value": yaml_value, "yaml_set": key == "require_key" and bool(API_KEY), "saved": stored,
                 "saved_value": saved_value, "ignored": False, "dropped": None}
    return {**item, "value": _jsonable(_applied_values[key]), "source": source, "locked": locked,
            "read_only": read_only, **extra}


def _settings_banners(state: settings_store.SettingsState, claimed: bool) -> list[dict]:
    banners: list[dict] = []

    def add(banner_id: str, level: str, text: str) -> None:
        banners.append({"id": banner_id, "level": level, "text": text})

    if not state.mount.writable:
        add("not_mounted", "error", f"Settings cannot be saved here: {state.mount.detail} Add the settings folder "
                                    "to frontend-service's volumes in the app YAML (lines below) and redeploy.")
    if not ENABLE_SETTINGS_UI:
        add("disabled", "warning", "ENABLE_SETTINGS_UI=false in the app YAML: this page is read-only and saved "
                                   "values are ignored. API keys made here stay in force.")
    elif state.safe_mode:
        add("safe_mode", "warning", "The settings folder holds a SAFE-MODE file: saved values are ignored and the "
                                    "app YAML applies. Delete the file to use them again. API keys stay in force.")
    if state.keys.fail_closed:
        add("keys_damaged", "error", state.keys.problem or "keys.json is damaged.")
    for problem in state.preferences.problems:
        add("problem", "warning", problem)
    if not claimed:
        add("unclaimed", "warning", "Nobody can change these settings yet: this server has no admin key. Print a "
                                    "one-time code to the container log and claim the server with it.")
    if not _applied_values.get("require_key"):
        add("open_api", "warning", "No API key: anyone on your network can use the API.")
    pending = state.preferences.pending
    if pending is not None:
        add("pending", "warning", f"The change of {', '.join(sorted(pending.previous))} is undone at "
                                  f"{datetime.fromtimestamp(pending.deadline, timezone.utc):%H:%M:%S} UTC "
                                  "unless it is confirmed.")
    if _file_values_in_force(state):
        add("stored", "info", "Stored on this server, applied at once.")
    elif ENABLE_SETTINGS_UI and not state.safe_mode:
        add("yaml", "info", "Not saved yet: showing the app YAML.")
    return banners


def _settings_view(state: settings_store.SettingsState, credential: Optional[_Credential]) -> dict:
    """GET /api/settings: every field, the keys without their secrets, banners and history."""
    preferences, keys = state.preferences, state.keys
    in_force = _file_values_in_force(state)
    overridden = _registry_override_providers()
    claimed = _settings_claimed()
    admin = credential is not None and credential.role == settings_schema.ROLE_ADMIN
    keys_open = ENABLE_SETTINGS_UI and state.mount.writable
    return {
        "revision": preferences.revision,
        "keys_revision": keys.revision,
        "claimed": claimed,
        "credential": None if credential is None else {
            "name": credential.name, "role": credential.role, "id": credential.key_id,
            "deployment_key": credential.key_id is None},
        "can_write": admin and keys_open and not state.safe_mode,
        "can_manage_keys": admin and keys_open and not keys.fail_closed,
        "can_claim": keys_open,
        "enabled": ENABLE_SETTINGS_UI,
        "safe_mode": state.safe_mode,
        "mount": {"state": state.mount.state, "writable": state.mount.writable, "detail": state.mount.detail,
                  "lines": _SETTINGS_MOUNT_LINES},
        "source": preferences.source,
        "saved_at": preferences.saved_at,
        "saved_by": dict(preferences.saved_by) if preferences.saved_by else None,
        "pending": _confirm_view(preferences.pending),
        "groups": [{"id": group, "label": label} for group, label in settings_schema.GROUPS],
        "settings": [_setting_view(item, state, in_force, overridden) for item in settings_schema.describe()],
        "access": {"require_key": _applied_values["require_key"],
                   "deployment_key_role": _applied_values["deployment_key_role"],
                   "deployment_key_present": bool(API_KEY), "fail_closed": keys.fail_closed},
        "keys": [record.public() for record in keys.keys],
        "max_keys": settings_store.MAX_KEYS,
        "engines": {
            "offered": {provider: entry.get("kind") for provider, entry in PROVIDER_REGISTRY["providers"].items()},
            "registry_json": sorted(overridden),
            "builtin": dict(settings_schema.BUILTIN_ENGINES),
        },
        "deployment": {
            "api_key_set": bool(API_KEY), "allow_any_host": ALLOW_ANY_HOST,
            "allowed_origins_wildcard": "*" in _ENV_ALLOWED_ORIGINS, "allow_credentials": ALLOW_CREDENTIALS,
            "settings_ui_enabled": ENABLE_SETTINGS_UI, "locked_keys": sorted(SETTINGS_LOCKED_KEYS),
            "workers": _frontend_workers(),
        },
        "banners": _settings_banners(state, claimed),
        "problems": list(preferences.problems) + ([keys.problem] if keys.problem else []),
        "history": [entry.as_dict() for entry in _settings_store.history()],
    }


# --- which optional engines are installed and answering -------------------------------
#
# Probed only for the Settings page (never on a request of the main UI or the API):
# a name that does not resolve can take seconds to fail on Docker Desktop. All at
# once, within a 4 s budget, cached for 30 s per worker. A Compose service that is
# not installed has no DNS name, which tells "not installed" from "not answering".

_ENGINE_PROBE_TTL_S = 30.0
_ENGINE_PROBE_BUDGET_S = 4.0
# Injectable so tests can move time instead of sleeping.
_engine_probe_clock = time.monotonic
_engine_probe_cache: dict = {"at": None, "value": None, "inflight": None}
# The profile that installs each engine's container, and what installing it downloads.
_ENGINE_PROFILES = {
    "canary": ("canary-asr", "about 5 GB"),
    "parakeet": ("parakeet-asr", "about 5 GB"),
    "chatterbox": ("chatterbox-tts", "about 5 GB"),
    "magpie": ("magpie-tts", "about 5 GB"),
    "whisper-cpp": ("whisper-cpp", "a small image and a model of about 0.6 GB"),
    "piper-training": ("training", None),
}


def _engine_targets() -> dict[str, tuple[str, str]]:
    """{provider id: (base URL, health path)} of the engines the Settings page offers or hides."""
    targets = {
        "canary": (CANARY_ASR_SERVICE_URL, "/health"),
        "parakeet": (PARAKEET_ASR_SERVICE_URL, "/health"),
        "chatterbox": (CHATTERBOX_TTS_SERVICE_URL, "/health"),
        "magpie": (MAGPIE_TTS_SERVICE_URL, "/health"),
        "whisper-cpp": (WHISPER_CPP_SERVICE_URL, "/"),
        "piper-training": (VOICE_TRAINING_URL, "/health"),
    }
    providers = _REGISTRY_OVERRIDE.get("providers") if isinstance(_REGISTRY_OVERRIDE, dict) else None
    for provider, entry in (providers if isinstance(providers, dict) else {}).items():
        # An operator's entry (PROVIDER_REGISTRY_JSON) is probed where it says it is.
        if provider in targets and isinstance(entry, dict) and isinstance(entry.get("internal_url"), str):
            health = entry.get("health_endpoint")
            targets[provider] = (entry["internal_url"], health if isinstance(health, str) else "/health")
    return targets


async def _resolves(host: str) -> bool:
    """Does the name resolve? The service name of a container that is not installed does not."""
    try:
        await asyncio.get_running_loop().getaddrinfo(host, None)
    except (OSError, UnicodeError):
        return False
    return True


# Injectable: tests answer from a table instead of asking DNS.
_engine_resolver = _resolves


async def _probe_engine(url: str, path: str) -> str:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + _ENGINE_PROBE_BUDGET_S
    try:
        host = urlsplit(url).hostname
    except ValueError:
        host = None
    if not host:
        return settings_schema.ENGINE_NOT_REACHABLE
    try:
        resolved = await asyncio.wait_for(_engine_resolver(host), _ENGINE_PROBE_BUDGET_S)
    except asyncio.TimeoutError:
        # Docker's DNS answers for its own containers at once; only an absent name takes this long.
        return settings_schema.ENGINE_NOT_INSTALLED
    except Exception:
        return settings_schema.ENGINE_NOT_REACHABLE
    if not resolved:
        return settings_schema.ENGINE_NOT_INSTALLED
    remaining = max(0.1, deadline - loop.time())
    try:
        response = await asyncio.wait_for(
            _get_http_client().get(f"{url}{path}", timeout=_timeout(remaining)), remaining)
    except Exception:
        return settings_schema.ENGINE_NOT_REACHABLE
    return settings_schema.ENGINE_RUNNING if response.status_code < 500 else settings_schema.ENGINE_NOT_REACHABLE


async def _probe_engines() -> dict:
    targets = _engine_targets()
    states = await asyncio.gather(*(_probe_engine(url, path) for url, path in targets.values()))
    return {"states": dict(zip(targets, states, strict=True)),
            "checked_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}


async def _engine_states() -> dict:
    """{"states": {provider id: state}, "checked_at"}: one probe round per TTL, shared by concurrent callers."""
    cache = _engine_probe_cache
    now = _engine_probe_clock()
    if (cache["value"] is not None and cache["at"] is not None
            and 0 <= now - cache["at"] < _ENGINE_PROBE_TTL_S):
        return cache["value"]
    loop = asyncio.get_running_loop()
    task = cache["inflight"]
    # A task from another event loop (tests, worker restarts) cannot be awaited here.
    if task is None or task.done() or task.get_loop() is not loop:
        task = loop.create_task(_probe_engines())
        cache["inflight"] = task

        def _store(finished: "asyncio.Task") -> None:
            if cache["inflight"] is finished:
                cache["inflight"] = None
            if not finished.cancelled() and finished.exception() is None:
                cache["value"], cache["at"] = finished.result(), _engine_probe_clock()

        task.add_done_callback(_store)
    return await asyncio.shield(task)


async def _engine_status_for(changes: Mapping[str, Any]) -> Optional[dict[str, str]]:
    """The engines' states when a change offers an optional engine (its warning depends on them)."""
    if not any(changes.get(key) is True and _applied_values.get(key) is not True
               for key in settings_schema.OPTIONAL_ENGINE_FLAGS):
        return None
    return (await _engine_states())["states"]


def _install_hint(profile: Optional[str], download: Optional[str]) -> Optional[dict]:
    if profile is None:
        return None
    size = f" ({download} download)" if download else ""
    return {"profile": profile,
            "truenas": f'Apps -> tts-stt -> Edit -> delete the line profiles: ["{profile}"] -> Save{size}',
            "compose": f"docker compose --profile {profile} up -d"}


# --- the routes ---------------------------------------------------------------------------


@app.get("/settings", response_class=HTMLResponse, include_in_schema=False)
async def settings_page(request: Request):
    """The Settings page: an empty shell; settings.js asks /api/settings for everything."""
    response = templates.TemplateResponse(request, "settings.html", {"app_version": APP_VERSION})
    response.headers["Content-Security-Policy"] = _SETTINGS_PAGE_CSP
    # The global middleware only sets SAMEORIGIN where a route has not set its own.
    response.headers["X-Frame-Options"] = "DENY"
    return response


@app.get("/api/settings", include_in_schema=False)
async def settings_read(request: Request):
    """Everything the page shows; read-only for anyone while the server is unclaimed."""
    return _settings_view(_settings_store.state, _bearer_credential(request.headers))


@app.put("/api/settings", include_in_schema=False)
async def settings_update(request: Request):
    """Save values and reset others to the app YAML: {base_revision, set, reset, acknowledge, dry_run}."""
    refusal = _preferences_locked_out(_settings_store.state)
    if refusal is not None:
        return refusal
    body = await _settings_body(request, required=("base_revision",),
                                optional=("set", "reset", "acknowledge", "dry_run"))
    if isinstance(body, Response):
        return body
    changes, reset, dry_run = body.get("set", {}), body.get("reset", []), body.get("dry_run", False)
    # What the app YAML fixes: SETTINGS_LOCKED_KEYS, and the engines PROVIDER_REGISTRY_JSON defines.
    locked = {**_registry_toggle_errors((*changes, *reset)),
              **{key: _LOCKED_MESSAGE for key in (*changes, *reset) if key in SETTINGS_LOCKED_KEYS}}
    if locked:
        return _settings_error(403, "Some settings are fixed by the app YAML.", "locked_by_deployment",
                               errors=locked)
    validation = settings_schema.validate_changes(
        changes, current=_applied_values, engine_status=await _engine_status_for(changes))
    errors = dict(validation.errors)
    for key in reset:
        if key not in settings_schema.PREFERENCE_KEYS:
            errors[key[:64]] = "This setting is not changed here." if key in settings_schema.BY_KEY else "Unknown setting."
        elif key in changes:
            errors[key] = "Set and reset at the same time."
    if errors:
        return _settings_error(400, "Some values are not valid.", "invalid_input", errors=errors)
    unconfirmed = settings_schema.unacknowledged(validation.warnings, body.get("acknowledge"))
    if unconfirmed and not dry_run:
        return _settings_error(409, "Some changes need to be confirmed first.", "needs_confirmation",
                               warnings=[warning.as_dict() for warning in unconfirmed])
    actor = _settings_actor(request, _bearer_credential(request.headers))
    before = _applied_values
    try:
        result = _settings_store.update_preferences(
            validation.values, reset, base_revision=body["base_revision"], actor=actor,
            check=_write_check(request.headers), dry_run=dry_run)
    except _SettingsRefused as refused:
        return _refused_write(refused, request, actor)
    except settings_store.SettingsStoreError as exc:
        return _store_refusal(exc)
    if result.written:
        _apply_saved()
    return _write_answer(result, before, validation.warnings, validation.notes)


@app.post("/api/settings/confirm", include_in_schema=False)
async def settings_confirm(request: Request):
    """Keep a TRUSTED_ORIGINS / TRUST_PROXY_HEADERS change; the page sends it through the new rules."""
    state = _settings_store.state
    refusal = _preferences_locked_out(state)
    if refusal is not None:
        return refusal
    body = await _settings_body(request, required=("revision",))
    if isinstance(body, Response):
        return body
    pending = state.preferences.pending
    if pending is None or pending.revision != body["revision"]:
        return _settings_error(409, "No change of this revision waits for a confirmation; it may have been undone.",
                               "nothing_to_confirm")
    try:
        confirmed = _settings_store.confirm(
            body["revision"], actor=_settings_actor(request, _bearer_credential(request.headers)))
    except settings_store.SettingsStoreError as exc:
        return _store_refusal(exc)
    _apply_saved()
    if not confirmed:
        return _settings_error(409, "Too late: the change was undone.", "confirm_expired")
    return {"confirmed": True, "revision": body["revision"]}


@app.post("/api/settings/restore", include_in_schema=False)
async def settings_restore(request: Request):
    """Make a saved version current again: {history_id, base_revision, acknowledge, dry_run}."""
    refusal = _preferences_locked_out(_settings_store.state)
    if refusal is not None:
        return refusal
    body = await _settings_body(request, required=("history_id", "base_revision"),
                                optional=("acknowledge", "dry_run"))
    if isinstance(body, Response):
        return body
    actor = _settings_actor(request, _bearer_credential(request.headers))
    before = _applied_values
    check = _write_check(request.headers)
    try:
        preview = _settings_store.restore(body["history_id"], base_revision=body["base_revision"], actor=actor,
                                          check=check, dry_run=True)
    except _SettingsRefused as refused:
        return _refused_write(refused, request, actor)
    except settings_store.SettingsStoreError as exc:
        return _store_refusal(exc)
    restored = {key: preview.values[key] for key in preview.changed
                if key in preview.values and key not in SETTINGS_LOCKED_KEYS}
    validation = settings_schema.validate_changes(
        restored, current=_applied_values, engine_status=await _engine_status_for(restored))
    if body.get("dry_run", False):
        return _write_answer(preview, before, validation.warnings, validation.notes)
    unconfirmed = settings_schema.unacknowledged(validation.warnings, body.get("acknowledge"))
    if unconfirmed:
        return _settings_error(409, "Some changes need to be confirmed first.", "needs_confirmation",
                               warnings=[warning.as_dict() for warning in unconfirmed])
    try:
        result = _settings_store.restore(body["history_id"], base_revision=body["base_revision"], actor=actor,
                                         check=check)
    except _SettingsRefused as refused:
        return _refused_write(refused, request, actor)
    except settings_store.SettingsStoreError as exc:
        return _store_refusal(exc)
    if result.written:
        _apply_saved()
    return _write_answer(result, before, validation.warnings, validation.notes)


@app.post("/api/settings/discard", include_in_schema=False)
async def settings_discard(request: Request):
    """Drop every saved value, so the app YAML applies; the discarded version stays in the history."""
    refusal = _preferences_locked_out(_settings_store.state)
    if refusal is not None:
        return refusal
    body = await _settings_body(request, required=("base_revision",), optional=("dry_run",))
    if isinstance(body, Response):
        return body
    actor = _settings_actor(request, _bearer_credential(request.headers))
    before = _applied_values
    try:
        # Back to the app YAML is always allowed, even where its engine defaults disagree.
        result = _settings_store.discard(base_revision=body["base_revision"], actor=actor,
                                         check=_write_check(request.headers, engines=False),
                                         dry_run=body.get("dry_run", False))
    except _SettingsRefused as refused:
        return _refused_write(refused, request, actor)
    except settings_store.SettingsStoreError as exc:
        return _store_refusal(exc)
    if result.written:
        _apply_saved()
    return _write_answer(result, before)


@app.get("/api/settings/engines", include_in_schema=False)
async def settings_engines():
    """Per engine the page can offer: running, not_reachable or not_installed, and how to install it."""
    probe = await _engine_states()
    overridden = _registry_override_providers()
    engines = []
    for key, (provider, kind) in settings_schema.ENGINE_FLAGS.items():
        engines.append({
            "provider": provider, "key": key, "kind": kind, "label": settings_schema.BY_KEY[key].label,
            "state": probe["states"].get(provider, settings_schema.ENGINE_NOT_REACHABLE),
            "offered": provider in PROVIDER_REGISTRY["providers"],
            "registry_json": provider in overridden,
            "install": _install_hint(*_ENGINE_PROFILES.get(provider, (None, None))),
        })
    return {"engines": engines, "checked_at": probe["checked_at"], "max_age": int(_ENGINE_PROBE_TTL_S)}


@app.put("/api/settings/access", include_in_schema=False)
async def settings_access(request: Request):
    """Open or key-required API, and what the app YAML key may do: {base_revision, require_key, deployment_key_role}."""
    refusal = _keys_locked_out(_settings_store.state)
    if refusal is not None:
        return refusal
    body = await _settings_body(request, required=("base_revision",),
                                optional=("require_key", "deployment_key_role"))
    if isinstance(body, Response):
        return body
    errors = {key: _LOCKED_MESSAGE for key in settings_schema.ACCESS_KEYS
              if key in body and key in SETTINGS_LOCKED_KEYS}
    if API_KEY and body.get("require_key") is False and "require_key" not in errors:
        errors["require_key"] = _API_KEY_FORCES_MESSAGE
    if errors:
        return _settings_error(403, "Some settings are fixed by the app YAML.", "locked_by_deployment",
                               errors=errors)
    credential = _bearer_credential(request.headers)
    actor = _settings_actor(request, credential)
    if body.get("deployment_key_role") == settings_schema.ROLE_CLIENT and credential is not None \
            and credential.key_id is None:
        _settings_store.audit("refused", actor=actor, result="403 deployment_key_not_allowed",
                              method=request.method, path=request.url.path)
        return _settings_error(403, "The app YAML key cannot make itself a client: do it with an admin key "
                                    "created on this page.", "deployment_key_not_allowed")
    try:
        keys = _settings_store.set_access(
            base_revision=body["base_revision"], actor=actor, deployment_key_present=bool(API_KEY),
            require_key=body.get("require_key"), deployment_key_role=body.get("deployment_key_role"))
    except settings_store.SettingsStoreError as exc:
        return _store_refusal(exc)
    _apply_saved()
    return {"keys_revision": keys.revision,
            "access": {"require_key": _applied_values["require_key"],
                       "deployment_key_role": _applied_values["deployment_key_role"]}}


@app.post("/api/settings/keys", include_in_schema=False, status_code=201)
async def settings_key_create(request: Request):
    """Store a key the page generated, as its SHA-256 and a hint: {name, role, key} -> 201 {id, hint, ...}."""
    refusal = _keys_locked_out(_settings_store.state)
    if refusal is not None:
        return refusal
    body = await _settings_body(request, required=("name", "role", "key"))
    if isinstance(body, Response):
        return body
    try:
        record = _settings_store.add_key(name=body["name"], role=body["role"], key=body["key"],
                                         actor=_settings_actor(request, _bearer_credential(request.headers)))
    except settings_store.SettingsStoreError as exc:
        return _store_refusal(exc)
    _apply_saved()
    return JSONResponse(status_code=201, content={**record.public(),
                                                  "keys_revision": _settings_store.state.keys.revision})


@app.delete("/api/settings/keys/{key_id}", include_in_schema=False)
async def settings_key_revoke(request: Request, key_id: str = PathParam(..., pattern=r"^[A-Za-z0-9_-]{1,32}$")):
    """Revoke a key made in the page; the last admin credential cannot go (409)."""
    refusal = _keys_locked_out(_settings_store.state)
    if refusal is not None:
        return refusal
    try:
        record = _settings_store.revoke_key(key_id, actor=_settings_actor(request, _bearer_credential(request.headers)),
                                            deployment_key_present=bool(API_KEY))
    except settings_store.SettingsStoreError as exc:
        return _store_refusal(exc)
    _apply_saved()
    return {"revoked": record.public(), "keys_revision": _settings_store.state.keys.revision}


@app.post("/api/settings/claim-code", include_in_schema=False, status_code=202)
async def settings_claim_code(request: Request):
    """Print a one-time code to this container's log, the only place it ever appears."""
    if not ENABLE_SETTINGS_UI:
        return _settings_error(403, "ENABLE_SETTINGS_UI=false in the app YAML: the Settings page is read-only.",
                               "settings_disabled")
    body = await _settings_body(request)
    if isinstance(body, Response):
        return body
    actor = _settings_actor(request, None)
    try:
        code = _settings_store.issue_code(actor=actor)
    except settings_store.SettingsStoreError as exc:
        return _store_refusal(exc)
    minutes = settings_store.CODE_TTL_S // 60
    _claim_logger.warning(
        "Settings: one-time code %s (asked for from %s). It is valid for %d minutes and works once: enter it "
        "on the Settings page to create an admin key.", code, actor.ip or "an unknown address", minutes)
    return JSONResponse(status_code=202, content={
        "detail": (f"A one-time code was printed to the log of frontend-service, valid for {minutes} minutes "
                   "(TrueNAS: Apps -> tts-stt -> Workloads -> frontend-service -> View Logs; elsewhere: "
                   "docker logs <frontend container>)."),
        "expires_in": settings_store.CODE_TTL_S})


@app.post("/api/settings/claim", include_in_schema=False, status_code=201)
async def settings_claim(request: Request):
    """Create an admin key with a one-time code: {code, name, key, require_key} -> 201. Also recovers a lost one."""
    if not ENABLE_SETTINGS_UI:
        return _settings_error(403, "ENABLE_SETTINGS_UI=false in the app YAML: the Settings page is read-only.",
                               "settings_disabled")
    body = await _settings_body(request, required=("code", "name", "key"), optional=("require_key",))
    if isinstance(body, Response):
        return body
    require_key = body.get("require_key")
    if "require_key" in SETTINGS_LOCKED_KEYS or (API_KEY and require_key is False):
        require_key = None          # the app YAML decides; the claim itself still goes ahead
    client = _client_address(request.scope)
    try:
        record = _settings_store.claim(code=body["code"], name=body["name"], key=body["key"],
                                       require_key=require_key,
                                       actor=_settings_actor(request, None, name="one-time code"))
    except settings_store.InvalidCode as exc:
        _settings_limiter.failed(client)
        return _store_refusal(exc)
    except settings_store.SettingsStoreError as exc:
        return _store_refusal(exc)
    _settings_limiter.succeeded(client)
    _apply_saved()
    return JSONResponse(status_code=201, content={**record.public(),
                                                  "keys_revision": _settings_store.state.keys.revision})


# --- OpenAI-compatible /v1 surface ------------------------------------------
#
# Mounted last so every helper it depends on is defined. Dependencies are passed
# in rather than imported, so openai_router.py never imports this module and
# there is no circular dependency.
#
# /api/* is unchanged and remains the browser contract; /v1/* is the surface
# that lets an OpenAI client talk to any of the four devices identically.
app.include_router(
    build_openai_router(
        get_provider=_get_provider,
        registry=PROVIDER_REGISTRY,
        post_form=_provider_form_post_raw,
        post_json=_provider_json_post,
        provider_health=provider_health,
        build_tts_payload=_build_tts_payload,
    )
)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3000)
