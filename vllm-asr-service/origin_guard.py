"""Cross-origin protection for the backends: the ALLOWED_ORIGINS rules, pure ASGI.

Why this exists. A backend port published on the host (compose binds it to
127.0.0.1 by default, to every interface with BACKEND_BIND_ADDR=0.0.0.0) can be
reached by any web page open in a browser on a machine that can route to it. The
old default, ``ALLOWED_ORIGINS=*`` with an empty value also meaning ``*``, let
such a page read every response, and CORS is not the only problem: a multipart
``POST`` is a "simple request" that a browser sends without any preflight, so a
page could start a training job, overwrite a voice or unload a model whether or
not it was allowed to read the answer.

Two rules, both applied here:

* ``parse_allowed_origins``: unset or empty means NO CORS headers (closed), ``*``
  only when it is written down (and it logs a warning), otherwise an explicit
  comma-separated list. The service adds Starlette's CORSMiddleware only when
  that list is not empty.
* ``OriginGuardMiddleware``: a request that changes state (anything except GET,
  HEAD, OPTIONS) and carries an ``Origin`` header is answered 403 unless the
  origin names the host the request was addressed to (its ``Host`` header) or is
  listed in ALLOWED_ORIGINS. A WebSocket handshake gets the same treatment,
  because browsers do not apply the same-origin policy to WebSockets at all.
  Callers that send no ``Origin`` (the gateway, curl, benchmarks, other
  containers) are not browsers and are not affected.

This is a cheap browser-borne-request check, not authentication: anything that
can reach the port and sets no ``Origin`` header is served as before, and a DNS
rebinding page (whose Origin and Host agree) is not stopped by it.

This file is deliberately dependency-free (it reads no environment: the service
passes the configured value in) and is copied byte for byte into every backend
directory, because each Docker build context is its own directory and cannot
share code. tests/test_origin_guard.py fails when the copies drift.
"""

from __future__ import annotations

import json
import logging
from typing import Iterable, List, Optional, Sequence, Tuple
from urllib.parse import urlsplit

logger = logging.getLogger("origin_guard")

SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})

_MAX_ECHO = 200  # how much of a client-supplied Origin an error message repeats


def normalize_origin(origin: str) -> str:
    """The form origins are compared in: no surrounding blanks, no trailing slash, lower case."""
    return origin.strip().rstrip("/").lower()


def parse_allowed_origins(raw: Optional[str], name: str = "ALLOWED_ORIGINS") -> List[str]:
    """The origins configured in *raw* (a comma-separated string), normalised.

    Unset, empty or only commas and blanks gives ``[]``, which means no
    cross-origin access at all (it used to mean ``*``). A ``*`` in the list is
    honoured, because the operator wrote it, but it is logged: it lets any page
    open in a browser that can reach the service call it.
    """
    origins: List[str] = []
    for part in (raw or "").split(","):
        origin = normalize_origin(part)
        if origin and origin not in origins:
            origins.append(origin)
    if "*" in origins:
        logger.warning(
            "%s contains '*': any web page open in a browser that can reach this service may "
            "call it, including the endpoints that upload, delete or start jobs. Leave it unset "
            "for no cross-origin browser access, or list the origins that need it.", name)
    return origins


def _default_port(scheme: str) -> int:
    return 443 if scheme in ("https", "wss") else 80


def _origin_key(origin: str) -> Optional[Tuple[str, int]]:
    """(host, port) of an Origin value; None for ``null`` and anything malformed."""
    try:
        parts = urlsplit(origin.strip())
        if parts.scheme not in ("http", "https", "ws", "wss") or not parts.hostname:
            return None
        return parts.hostname.lower(), parts.port or _default_port(parts.scheme)
    except ValueError:
        return None


def _host_key(host: str, scheme: str) -> Optional[Tuple[str, int]]:
    """(host, port) of a Host header; a missing port means the scheme's default."""
    try:
        parts = urlsplit(f"//{host.strip()}")
        if not parts.hostname:
            return None
        return parts.hostname.lower(), parts.port or _default_port(scheme)
    except ValueError:
        return None


def is_same_origin(origin: str, host: Optional[str]) -> bool:
    """Does this Origin name the host (and port) the request was addressed to?"""
    key = _origin_key(origin)
    if key is None or not host:
        return False
    return _host_key(host, urlsplit(origin.strip()).scheme) == key


def origin_permitted(origin: str, host: Optional[str], allowed_origins: Iterable[str]) -> bool:
    """Same origin as the service, or one the operator listed (or ``*``)."""
    allowed = tuple(allowed_origins)
    if "*" in allowed or normalize_origin(origin) in allowed:
        return True
    return is_same_origin(origin, host)


def _header_values(scope, name: bytes) -> List[str]:
    return [
        value.decode("latin-1")
        for key, value in (scope.get("headers") or ())
        if key.lower() == name
    ]


def _refusal_detail(origin: str) -> str:
    shown = origin if len(origin) <= _MAX_ECHO else origin[:_MAX_ECHO] + "..."
    return (
        f"Cross-origin request from {shown!r} refused. Add it to ALLOWED_ORIGINS "
        "to allow it; requests without an Origin header (curl, other services) are not affected."
    )


async def _refuse_http(send, origin: str) -> None:
    body = json.dumps({"detail": _refusal_detail(origin)}).encode("utf-8")
    try:
        await send({
            "type": "http.response.start",
            "status": 403,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode("ascii")),
                # the request body has not been read and will not be
                (b"connection", b"close"),
            ],
        })
        await send({"type": "http.response.body", "body": body})
    except OSError:
        logger.debug("client went away before the 403 could be sent")


class OriginGuardMiddleware:
    """403 a state-changing request (or refuse a WebSocket) from a foreign browser origin."""

    def __init__(self, app, allowed_origins: Sequence[str] = ()):
        self.app = app
        self.allowed_origins = tuple(normalize_origin(o) for o in allowed_origins if o and o.strip())

    def _refused_origin(self, scope) -> Optional[str]:
        """The first Origin of this request that is not permitted, or None."""
        hosts = _header_values(scope, b"host")
        host = hosts[0] if hosts else None
        for origin in _header_values(scope, b"origin"):
            if not origin_permitted(origin, host, self.allowed_origins):
                return origin
        return None

    async def __call__(self, scope, receive, send):
        kind = scope["type"]
        if kind == "http" and scope["method"] not in SAFE_METHODS:
            origin = self._refused_origin(scope)
            if origin is not None:
                logger.info("403 %s %s: cross-origin request from %r", scope["method"], scope["path"], origin[:_MAX_ECHO])
                await _refuse_http(send, origin)
                return
        elif kind == "websocket":
            origin = self._refused_origin(scope)
            if origin is not None:
                logger.info("refused WebSocket %s from origin %r", scope["path"], origin[:_MAX_ECHO])
                await receive()  # the handshake request, before it may be refused
                await send({"type": "websocket.close", "code": 1008})  # policy violation
                return
        await self.app(scope, receive, send)
