"""Load any of the eight backends in-process, for the request-hardening tests (no tests of its own).

`tests/test_body_limit_backends.py` and `tests/test_origin_guard_backends.py` ask the
same question of every backend (does a chunked upload get cut off? does a foreign
Origin get refused?), so they share this one adapter over the per-service harnesses
that already know how to import each app under stand-ins for torch, NeMo, soundfile
and friends. Nothing here starts a lifespan, so no model is ever loaded.

A loaded backend is a ``Backend``: the imported module, an ``httpx.AsyncClient``
factory speaking ASGI to it (no sockets, and chunked bodies work), the temp
directory the app was told to use (a file left in it is a leak), and the route
each kind of request goes to.
"""

from __future__ import annotations

import contextlib
import inspect
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import httpx

MIB = 1024 * 1024

# Captured at import (collection), before any test can patch it: the gateway tests
# replace `httpx.AsyncClient` on the httpx module itself, so a stub that outlived
# its test would otherwise become the client used here.
_REAL_ASYNC_CLIENT = httpx.AsyncClient

BACKENDS = (
    "stt-service",
    "piper-tts-service",
    "piper-training-service",
    "chatterbox-tts-service",
    "magpie-tts-service",
    "qwen3-tts-service",
    "qwen3-asr-service",
    "parakeet-asr-service",
    "canary-asr-service",
)

# What each backend calls a multipart upload it accepts (None: it takes no uploads at all,
# like magpie-tts-service), and a state-changing route
# that needs no body, so a request that gets past the origin guard visibly reaches
# the handler (any status but 403) without doing any work.
#   upload:  (method, path, file field name)
#   probe:   (method, path)
#   json:    (path) of a route that reads a JSON body, or None
ROUTES = {
    "stt-service": {"upload": ("POST", "/transcribe-stream", "audio"), "probe": ("POST", "/unload"),
                    "json": None},
    "piper-tts-service": {"upload": ("POST", "/upload_model", "model_file"), "probe": ("POST", "/refresh_voices"),
                          "json": "/tts"},
    "piper-training-service": {"upload": ("POST", "/train", "audio_files"), "probe": ("DELETE", "/job/nope"),
                               "json": "/prepare-dataset"},
    "chatterbox-tts-service": {"upload": ("POST", "/clone", "file"), "probe": ("POST", "/unload"),
                               "json": "/tts"},
    "magpie-tts-service": {"upload": None, "probe": ("POST", "/unload"), "json": "/tts"},
    "qwen3-tts-service": {"upload": ("POST", "/clone", "file"), "probe": ("POST", "/unload"),
                          "json": "/tts"},
    "qwen3-asr-service": {"upload": ("POST", "/transcribe", "audio"), "probe": ("POST", "/unload"),
                          "json": None},
    "parakeet-asr-service": {"upload": ("POST", "/transcribe", "audio"), "probe": ("POST", "/unload"),
                             "json": None},
    "canary-asr-service": {"upload": ("POST", "/transcribe", "audio"), "probe": ("POST", "/unload"),
                           "json": None},
}


@dataclass
class Backend:
    name: str
    module: object
    tmp: Path
    routes: dict

    @property
    def app(self):
        return self.module.app

    def client(self, base_url: str = "http://backend.test", **kwargs) -> httpx.AsyncClient:
        """An in-process client; ``raise_app_exceptions=False`` so a crash is a 500, not a test error."""
        transport = httpx.ASGITransport(app=self.module.app, raise_app_exceptions=False)
        return _REAL_ASYNC_CLIENT(transport=transport, base_url=base_url, timeout=30.0, **kwargs)

    def leftovers(self) -> list:
        return sorted(p.name for p in self.tmp.iterdir())


def load_backend(name: str, monkeypatch, tmp_path: Path, stack: contextlib.ExitStack, **env) -> Backend:
    """Import backend *name* with *env* as its whole configuration and its temp files under *tmp_path*."""
    tmp = tmp_path / "spool"
    tmp.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(tempfile, "tempdir", str(tmp))

    if name == "stt-service":
        from test_stt_support import load_stt_app
        module = load_stt_app({k: str(v) for k, v in env.items()}, name=f"stt_hardening_{next(_ids)}")
    elif name == "piper-tts-service":
        from test_piper_tts_harness import load_piper_app
        module = load_piper_app({
            "PIPER_DATA_DIR": str(tmp_path / "models"), "PIPER_OUTPUT_DIR": str(tmp_path / "out"),
            **{k: str(v) for k, v in env.items()}})
    elif name == "piper-training-service":
        from test_piper_training_service_support import load_training_service
        module = load_training_service(stack, monkeypatch, tmp_path, {k: str(v) for k, v in env.items()}).module
    elif name == "chatterbox-tts-service":
        from chatterbox_loader import load_app
        module = load_app(**env)
    elif name == "magpie-tts-service":
        import magpie_loader
        # Only what the service reads: the loader restores exactly its own managed variables, so a
        # setting meant for the upload-taking backends (MAX_UPLOAD_MB) must not be left in os.environ.
        module = magpie_loader.load_app(**{k: v for k, v in env.items() if k in magpie_loader.MANAGED_ENV})
    elif name == "qwen3-tts-service":
        from qwen3_tts_loader import load_app
        module = load_app(voices_dir=str(tmp_path / "voices"), **env)
    elif name == "qwen3-asr-service":
        from test_qwen3_asr_harness import load_app
        module = load_app(**env)
    elif name in ("parakeet-asr-service", "canary-asr-service"):
        from test_nemo_services_harness import load_app
        module = load_app(name.split("-")[0], **env)
    else:
        raise KeyError(name)
    return Backend(name, module, tmp, ROUTES[name])


class _Counter:
    def __init__(self):
        self.n = 0

    def __next__(self):
        self.n += 1
        return self.n


_ids = _Counter()


def spy_on_route(app, path: str, method: str = "POST") -> list:
    """Record every call of the handler behind (*method*, *path*); the list stays empty if it never runs.

    Wraps ``dependant.call``, which FastAPI reads on each request, and keeps the
    wrapper as sync or async as the handler is (FastAPI decided that when the
    route was built).
    """
    for route in app.router.routes:
        if getattr(route, "path", None) == path and method in (getattr(route, "methods", None) or ()):
            calls: list = []
            original = route.dependant.call
            if inspect.iscoroutinefunction(original):
                async def spy(*args, **kwargs):
                    calls.append(1)
                    return await original(*args, **kwargs)
            else:
                def spy(*args, **kwargs):
                    calls.append(1)
                    return original(*args, **kwargs)
            route.dependant.call = spy
            return calls
    raise AssertionError(f"no {method} {path} route on {app}")


def multipart_head(field: str, filename: str = "x.wav", extra_fields: Optional[dict] = None,
                   boundary: str = "b") -> bytes:
    """The start of a multipart body: the form fields, then the file part's header, no content yet."""
    out = b""
    for key, value in (extra_fields or {}).items():
        out += (f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"\r\n\r\n{value}\r\n').encode()
    out += (f'--{boundary}\r\nContent-Disposition: form-data; name="{field}"; filename="{filename}"\r\n'
            "Content-Type: application/octet-stream\r\n\r\n").encode()
    return out


class Offer:
    """A chunked request body of *total* bytes that counts how much of it the server took."""

    def __init__(self, head: bytes, total: int, chunk: int = 256 * 1024):
        self.head, self.total, self.chunk = head, total, chunk
        self.pulled = 0          # chunks the server consumed

    def __aiter__(self):
        return self._generate()

    async def _generate(self):
        if self.head:
            yield self.head
        sent = 0
        while sent < self.total:
            self.pulled += 1
            size = min(self.chunk, self.total - sent)
            sent += size
            yield b"\0" * size

    @property
    def pulled_bytes(self) -> int:
        return self.pulled * self.chunk


def multipart_content_type(boundary: str = "b") -> dict:
    return {"content-type": f"multipart/form-data; boundary={boundary}"}
