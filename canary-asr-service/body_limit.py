"""Request-body size limit for the backends: pure ASGI, streamed byte count.

Why this exists. Starlette parses a request body before a handler runs: a
multipart upload is spooled to a temp file, a JSON body is read into memory. So a
size check inside the handler comes after the disk was filled or the RAM taken,
and uvicorn has no body limit of its own. A middleware that only looks at the
``Content-Length`` header is not enough either: a chunked upload sends none, and
a client may declare less than it sends. This middleware does both checks:

* a declared ``Content-Length`` over the limit is answered with 413 without
  reading a byte (and without the app ever seeing the request);
* whatever is declared, the bytes that actually arrive are counted in
  ``receive()``, and the moment the count crosses the limit the app is fed an
  error instead of more data, so nothing past the limit is ever read, spooled or
  buffered.

Once the limit is crossed the response is owned here: whatever the app answers
to the aborted read (FastAPI turns a failed body parse into a 400) is dropped and
the 413 goes out instead, so no handler can swallow it. If the app had already
started its own response there is nothing left to replace, and it is left alone
(never two responses).

This file is deliberately dependency-free and is copied byte for byte into every
backend directory, because each Docker build context is its own directory and
cannot share code. tests/test_body_limit.py fails when the copies drift, so a fix
made in one place has to be made in all of them.

Usage::

    app.add_middleware(BodyLimitMiddleware, limit_for=my_limit_for_path)

``limit_for(path)`` returns the most bytes a request to that path may carry, or
``None`` for no limit. It is called for every request (not once at start-up), so
a limit that depends on configuration can be read when it is needed. GET, HEAD
and OPTIONS are never limited: they carry no body the app reads.
"""

from __future__ import annotations

import json
import logging
from typing import Callable, Iterable, Optional, Tuple, Union

logger = logging.getLogger("body_limit")

SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})

_KIB = 1024
_MIB = 1024 * 1024
# A Content-Length with more digits than this is over any limit anyone could set
# (10**18 bytes is an exabyte); refusing to int() it keeps a megabyte of digits
# from costing anything, and Python 3.11+ refuses very long digit strings anyway.
_MAX_DIGITS = 18
_HUGE = 1 << 62


class BodyTooLarge(Exception):
    """Raised into the app from ``receive()`` once the body has passed its limit."""


def declared_length(headers: Iterable[Tuple[bytes, bytes]]) -> Optional[int]:
    """The Content-Length the request declares, or None when there is no usable one.

    Absent, empty, negative, signed, fractional or otherwise not a plain run of
    ASCII digits all count as "not declared": the streamed count still bounds
    such a request, and the HTTP server rejects a malformed header before an app
    sees it anyway. With several Content-Length headers the largest wins, so
    repeating the header cannot be used to hide a big one behind a small one.
    """
    best: Optional[int] = None
    for name, value in headers:
        if name.lower() != b"content-length":
            continue
        value = value.strip()
        if not value or not value.isdigit():   # bytes.isdigit() is ASCII-only
            continue
        length = _HUGE if len(value) > _MAX_DIGITS else int(value)
        if best is None or length > best:
            best = length
    return best


def _size(limit: int) -> str:
    if limit >= _MIB:
        return f"{limit / _MIB:.3g} MB"
    if limit >= _KIB:
        return f"{limit / _KIB:.3g} KB"
    return f"{limit} bytes"


async def _refuse(send, limit: int, hint: str) -> None:
    """A small JSON 413. The unread rest of the body is still on the wire, so close."""
    detail = f"Request body is larger than {_size(limit)}."
    if hint:
        detail = f"{detail} {hint}"
    body = json.dumps({"detail": detail}).encode("utf-8")
    try:
        await send({
            "type": "http.response.start",
            "status": 413,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode("ascii")),
                (b"connection", b"close"),
            ],
        })
        await send({"type": "http.response.body", "body": body})
    except OSError:
        # The client is already gone; there is nobody to tell.
        logger.debug("client went away before the 413 could be sent")


class BodyLimitMiddleware:
    """Answer 413 as soon as a request body is known to exceed its limit."""

    def __init__(
        self,
        app,
        limit_for: Union[Callable[[str], Optional[int]], int, None],
        hint: str = "",
    ):
        self.app = app
        if callable(limit_for):
            self._limit_for = limit_for
        else:
            self._limit_for = lambda path: limit_for
        self.hint = hint

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] in SAFE_METHODS:
            await self.app(scope, receive, send)
            return

        limit = self._limit_for(scope["path"])
        if limit is None:
            await self.app(scope, receive, send)
            return

        declared = declared_length(scope.get("headers") or ())
        if declared is not None and declared > limit:
            logger.info("413 %s %s: declares %d bytes, limit %d", scope["method"], scope["path"], declared, limit)
            await _refuse(send, limit, self.hint)
            return

        received = 0
        exceeded = False   # the count crossed the limit
        started = False    # the app began its own response

        async def limited_receive():
            nonlocal received, exceeded
            if exceeded:
                raise BodyTooLarge()
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body") or b"")
                if received > limit:
                    exceeded = True
                    raise BodyTooLarge()
            return message

        async def guarded_send(message):
            nonlocal started
            if exceeded and not started:
                return  # the app's answer to the aborted read; the 413 replaces it
            if message["type"] == "http.response.start":
                started = True
            await send(message)

        try:
            await self.app(scope, limited_receive, guarded_send)
        except Exception:
            # Whatever the app made of the aborted read (BodyTooLarge itself, a
            # parser error, an exception group from a task group) is expected
            # once the limit has been crossed. Anything else is a real failure.
            if not exceeded:
                raise
        if exceeded and not started:
            logger.info("413 %s %s: body passed %d bytes", scope["method"], scope["path"], limit)
            await _refuse(send, limit, self.hint)
