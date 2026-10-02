"""Shared harness for the gateway (frontend-service) tests.

`frontend-service/app.py` reads most of its configuration at import time
(CORS origins, API key, size caps), so a test that needs a different
configuration has to import a fresh copy of the module. That, and a stand-in for
the pooled `httpx.AsyncClient` the gateway talks to its backends with, is all
this module provides.

`FRONTEND_SERVICE_DIR` points the loader at another checkout of the service,
which is how the new tests are run against the pre-fix code to prove they fail
there.
"""

from __future__ import annotations

import os
import sys
from importlib.util import module_from_spec, spec_from_file_location
from itertools import count
from pathlib import Path
from typing import Callable, Optional

import httpx

REPO = Path(__file__).resolve().parents[1]
SERVICE_DIR = Path(os.environ.get("FRONTEND_SERVICE_DIR") or REPO / "frontend-service")

# Captured before any test patches `httpx.AsyncClient` (the gateway resolves the
# class through the global `httpx` module, so patching it replaces it everywhere).
REAL_ASYNC_CLIENT = httpx.AsyncClient

# Settings the gateway reads from the environment. Anything not passed to
# `load_frontend_app` is removed while importing, so a variable exported in the
# developer's shell cannot change what a test measures.
MANAGED_ENV = (
    "ALLOWED_ORIGINS", "ALLOW_CREDENTIALS", "TRUSTED_ORIGINS", "TRUST_PROXY_HEADERS",
    "TRUSTED_HOSTS", "ALLOWED_HOSTS",
    "API_KEY", "MAX_UPLOAD_MB", "MAX_TTS_CHARS", "MAX_CONCURRENT_UPLOADS", "HEALTH_CACHE_TTL",
    "DEFAULT_STT_PROVIDER", "DEFAULT_TTS_PROVIDER", "PROVIDER_REGISTRY_JSON",
    "ENABLE_WHISPER_CPP", "ENABLE_PARAKEET_ASR", "ENABLE_CANARY_ASR",
    "ENABLE_CHATTERBOX_TTS", "ENABLE_MAGPIE_TTS", "ENABLE_TRAINING", "APP_VERSION",
    # The Settings page: its switches, and the folder it saves to. Without
    # TTS_STT_SETTINGS_DIR the store looks at /app/settings, which is not mounted
    # here, so nothing saved applies and nothing can be saved.
    "ENABLE_SETTINGS_UI", "SETTINGS_LOCKED_KEYS", "TTS_STT_SETTINGS_DIR",
)

_counter = count()


def load_frontend_app(env: Optional[dict] = None):
    """Import a fresh copy of the gateway's app.py under `env`."""
    env = env or {}
    app_path = SERVICE_DIR / "app.py"
    spec = spec_from_file_location(f"fe_harness_{next(_counter)}", app_path)
    module = module_from_spec(spec)

    previous_cwd = os.getcwd()
    # Anything the caller sets (a backend URL, say) is put back too, not only the
    # names the gateway is known to read.
    saved = {key: os.environ.get(key) for key in (*MANAGED_ENV, *env)}
    sys.path.insert(0, str(SERVICE_DIR))
    try:
        for key in MANAGED_ENV:
            os.environ.pop(key, None)
        os.environ.update(env)
        os.chdir(SERVICE_DIR)
        assert spec.loader is not None
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(SERVICE_DIR))
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        os.chdir(previous_cwd)
    return module


class _Streamable(httpx.Response):
    """A Response that also answers the streaming proxy (`aiter_raw`, `aread`)."""

    async def aiter_raw(self, chunk_size=None):
        yield self.content

    async def aread(self):
        return self.content

    async def aclose(self):
        return None


def stub_client_class(handler: Callable):
    """Build an `httpx.AsyncClient` stand-in whose answers come from `handler`.

    `handler(method, url, kwargs)` returns an `httpx.Response`, a dict (sent as a
    JSON 200) or `bytes` (sent as a 200 body), or raises an httpx error to
    simulate a dead backend. Every call is recorded on the class as
    `Stub.calls == [(method, url, kwargs), ...]`.
    """

    class Stub:
        calls: list = []

        def __init__(self, *args, **kwargs):
            pass

        async def _dispatch(self, method: str, url: str, kwargs: dict) -> httpx.Response:
            type(self).calls.append((method, url, kwargs))
            result = handler(method, url, kwargs)
            if hasattr(result, "__await__"):
                result = await result
            if isinstance(result, dict) or isinstance(result, list):
                result = httpx.Response(200, json=result)
            elif isinstance(result, bytes):
                result = httpx.Response(200, content=result)
            result.request = httpx.Request(method, url)
            return result

        async def get(self, url, **kwargs):
            return await self._dispatch("GET", url, kwargs)

        async def post(self, url, **kwargs):
            return await self._dispatch("POST", url, kwargs)

        async def delete(self, url, **kwargs):
            return await self._dispatch("DELETE", url, kwargs)

        def build_request(self, method, url, **kwargs):
            return (method, url, kwargs)

        async def send(self, request, stream=False):
            method, url, kwargs = request
            response = await self._dispatch(method, url, kwargs)
            return _Streamable(
                status_code=response.status_code, headers=response.headers,
                content=response.content, request=response.request)

        async def aclose(self):
            return None

    Stub.calls = []
    return Stub


def install_stub(monkeypatch, module, handler: Callable):
    """Route all of `module`'s upstream traffic through `handler`; returns the stub class."""
    stub = stub_client_class(handler)
    monkeypatch.setattr(module.httpx, "AsyncClient", stub)
    return stub


def asgi_client(module, **kwargs) -> httpx.AsyncClient:
    """A real async client that talks to the app in-process (no sockets)."""
    return REAL_ASYNC_CLIENT(
        transport=httpx.ASGITransport(app=module.app), base_url="http://testserver", **kwargs)


async def asgi_call(app, method: str, path: str, headers=(), body: bytes = b""):
    """One HTTP request straight into the ASGI app, with exactly the headers given.

    Unlike a client this adds nothing of its own - no Host, no Content-Length -
    which is what a test of "there is no Host header" or "the body is chunked"
    needs. Returns ``(status, headers_dict, body_bytes)``.
    """
    scope = {
        "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": method,
        "scheme": "http", "path": path, "raw_path": path.encode(), "query_string": b"",
        "root_path": "", "server": ("testserver", 80), "client": ("127.0.0.1", 1234),
        "headers": [(k.lower().encode(), v.encode()) for k, v in headers],
    }
    sent = []
    delivered = False

    async def receive():
        nonlocal delivered
        if not delivered:
            delivered = True
            return {"type": "http.request", "body": body, "more_body": False}
        return {"type": "http.disconnect"}

    async def send(message):
        sent.append(message)

    await app(scope, receive, send)
    start = next(m for m in sent if m["type"] == "http.response.start")
    payload = b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body")
    return start["status"], {k.decode().lower(): v.decode() for k, v in start["headers"]}, payload


def wav_bytes(samples: bytes = b"\x00\x00" * 160, *, rate: int = 22050,
              channels: int = 1, width: int = 2) -> bytes:
    """A real PCM WAV; the pcm and mp3 paths parse or transcode it."""
    import struct

    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", 36 + len(samples), b"WAVE",
        b"fmt ", 16, 1, channels, rate, rate * channels * width, channels * width, width * 8,
        b"data", len(samples),
    )
    return header + samples
