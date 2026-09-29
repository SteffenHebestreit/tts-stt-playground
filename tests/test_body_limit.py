"""body_limit.py: the request-body limit every backend shares, tested on its own.

What the old per-service middlewares did: compare the ``Content-Length`` header
with a limit. A client that sends the body chunked has no such header, and a
client may declare less than it sends, so both walked straight past them; Starlette
then spooled the whole upload to disk (multipart) or RAM (JSON) before any handler
could look at it.

These tests drive the middleware as raw ASGI, with a fake app that reads its body
the way Starlette does, so every case (chunked, understated, garbage
Content-Length, an app that swallows read errors, an app that already answered,
a client that hangs up) is exact. tests/test_body_limit_backends.py then sends
real chunked requests through each backend's own app.
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest

from backend_apps import BACKENDS

REPO = Path(__file__).resolve().parents[1]
MODULE = REPO / "piper-tts-service" / "body_limit.py"


def _load(path: Path = MODULE):
    spec = spec_from_file_location(f"body_limit_under_test_{abs(hash(str(path)))}", path)
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


bl = _load()
MIB = 1024 * 1024


# --- the copies -----------------------------------------------------------------------

def test_every_backend_ships_an_identical_copy():
    """Each Docker build context is its own directory, so the file exists once per
    service. A fix made in one copy and not the others is the bug this guards."""
    copies = {name: (REPO / name / "body_limit.py").read_bytes() for name in BACKENDS}
    reference = copies["piper-tts-service"]
    diverged = sorted(name for name, data in copies.items() if data != reference)
    assert not diverged, (
        f"body_limit.py differs in {diverged}. The copies exist because each service builds its "
        f"image from its own directory, not because they may behave differently: edit one and "
        f"copy it over the others.")


def test_no_other_service_directory_carries_an_unlisted_copy():
    """A copy in a directory this test does not list would drift unnoticed."""
    found = sorted(p.parent.name for p in REPO.glob("*-service/body_limit.py"))
    assert found == sorted(BACKENDS), f"body_limit.py copies: {found}; expected exactly {sorted(BACKENDS)}"


@pytest.mark.parametrize("name", BACKENDS)
def test_every_backend_installs_it_and_no_weaker_middleware_is_left(name):
    source = (REPO / name / "app.py").read_text(encoding="utf-8")
    assert "from body_limit import BodyLimitMiddleware" in source
    assert re.search(r"app\.add_middleware\(\s*BodyLimitMiddleware\b", source), f"{name} does not install it"
    # The Content-Length-only middlewares this replaced, in any of their spellings.
    assert "class _BodyLimitMiddleware" not in source
    assert 'get(b"content-length"' not in source
    for helper in ("nemo_common.py", "audio_pipeline.py"):
        module = REPO / name / helper
        if module.exists():
            assert "class BodyLimitMiddleware" not in module.read_text(encoding="utf-8")


def test_the_module_needs_nothing_outside_the_standard_library():
    """It is copied into images that differ wildly (CUDA, ROCm, slim); it must not
    grow a dependency on any of them."""
    import ast

    tree = ast.parse(MODULE.read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert imported <= set(sys.stdlib_module_names), imported - set(sys.stdlib_module_names)


# --- a tiny ASGI rig ------------------------------------------------------------------

def scope(method="POST", path="/up", headers=(), kind="http"):
    return {"type": kind, "method": method, "path": path, "headers": list(headers),
            "query_string": b"", "http_version": "1.1", "scheme": "http"}


def length(n):
    return (b"content-length", str(n).encode())


class Rig:
    """Feeds *chunks* to the app as http.request messages and records what it sent.

    ``pulled`` counts the chunks the app actually read. After the last chunk the
    client "waits": a further receive() returns http.disconnect, as a server does
    once the client goes away.
    """

    def __init__(self, chunks):
        self.chunks = list(chunks)
        self.pulled = 0
        self.sent = []
        self.disconnect_after_chunk = None

    async def receive(self):
        if self.disconnect_after_chunk is not None and self.pulled >= self.disconnect_after_chunk:
            return {"type": "http.disconnect"}
        if self.pulled < len(self.chunks):
            chunk = self.chunks[self.pulled]
            self.pulled += 1
            return {"type": "http.request", "body": chunk, "more_body": self.pulled < len(self.chunks)}
        return {"type": "http.disconnect"}

    async def send(self, message):
        self.sent.append(message)

    async def call(self, app, sc):
        await app(sc, self.receive, self.send)
        return self

    @property
    def status(self):
        starts = [m for m in self.sent if m["type"] == "http.response.start"]
        assert len(starts) <= 1, f"two responses: {self.sent}"
        return starts[0]["status"] if starts else None

    @property
    def body(self):
        return b"".join(m.get("body", b"") for m in self.sent if m["type"] == "http.response.body")

    def header(self, name):
        for m in self.sent:
            if m["type"] == "http.response.start":
                return dict(m["headers"]).get(name.encode())
        return None


def run(coro):
    return asyncio.run(coro)


class Reader:
    """An app that reads its whole body through receive(), like request.body() does."""

    def __init__(self, answer=b"ok"):
        self.called = 0
        self.read = 0
        self.error = None
        self.answer = answer

    async def __call__(self, scope, receive, send):
        self.called += 1
        try:
            while True:
                message = await receive()
                if message["type"] == "http.disconnect":
                    raise ConnectionAbortedError("client went away")
                self.read += len(message.get("body", b""))
                if not message.get("more_body"):
                    break
        except Exception as exc:
            self.error = exc
            raise
        await send({"type": "http.response.start", "status": 200, "headers": [(b"content-type", b"text/plain")]})
        await send({"type": "http.response.body", "body": self.answer})


def limited(app, limit, **kwargs):
    return bl.BodyLimitMiddleware(app, limit_for=lambda path: limit, **kwargs)


# --- declared Content-Length: the fast path -------------------------------------------

def test_a_declared_length_over_the_limit_is_refused_without_reading_or_calling_the_app():
    app = Reader()
    rig = Rig([b"x" * 10])
    run(rig.call(limited(app, 1000), scope(headers=[length(10_000_000)])))

    assert rig.status == 413
    assert app.called == 0
    assert rig.pulled == 0, "the body was read although its declared length is already too big"
    detail = json.loads(rig.body)["detail"]
    assert detail.startswith("Request body is larger than")
    assert rig.header("content-type") == b"application/json"
    assert int(rig.header("content-length")) == len(rig.body)
    assert rig.header("connection") == b"close", "the unread body is still on the wire"


def test_the_refusal_names_the_limit_in_readable_units():
    for limit, expected in ((5 * MIB, "5 MB"), (1536 * 1024, "1.5 MB"), (2048, "2 KB"), (300, "300 bytes")):
        rig = Rig([])
        run(rig.call(limited(Reader(), limit), scope(headers=[length(limit + 1)])))
        assert f"larger than {expected}." in json.loads(rig.body)["detail"], (limit, rig.body)


def test_a_hint_is_appended_to_the_refusal():
    rig = Rig([])
    run(rig.call(limited(Reader(), 10, hint="Set MAX_UPLOAD_MB."), scope(headers=[length(11)])))
    assert json.loads(rig.body)["detail"].endswith("Set MAX_UPLOAD_MB.")


def test_a_declared_length_within_the_limit_reaches_the_app():
    app = Reader()
    rig = Rig([b"x" * 600, b"y" * 400])
    run(rig.call(limited(app, 1000), scope(headers=[length(1000)])))
    assert rig.status == 200 and app.read == 1000 and rig.pulled == 2


def test_the_limit_is_inclusive():
    """Exactly the limit is fine, one byte more is not: streamed and declared alike."""
    ok = Reader()
    rig = Rig([b"x" * 1000])
    run(rig.call(limited(ok, 1000), scope()))
    assert rig.status == 200 and ok.read == 1000

    over = Reader()
    rig = Rig([b"x" * 1001])
    run(rig.call(limited(over, 1000), scope()))
    assert rig.status == 413

    rig = Rig([b"x" * 1000])
    run(rig.call(limited(Reader(), 1000), scope(headers=[length(1000)])))
    assert rig.status == 200
    rig = Rig([b"x" * 1001])
    run(rig.call(limited(Reader(), 1000), scope(headers=[length(1001)])))
    assert rig.status == 413 and rig.pulled == 0


# --- no (usable) Content-Length: the streamed count -----------------------------------

def test_a_chunked_body_over_the_limit_is_cut_off_while_it_is_still_arriving():
    """The reviewer's case: no Content-Length, 200 MB on offer, a 2 MB limit."""
    app = Reader()
    chunks = [b"\0" * (256 * 1024)] * 800
    rig = Rig(chunks)
    run(rig.call(limited(app, 2 * MIB), scope()))

    assert rig.status == 413
    assert rig.pulled <= 9, f"kept reading after the limit: {rig.pulled} chunks of 800 consumed"
    assert app.read <= 2 * MIB, "bytes past the limit were handed to the app"
    assert isinstance(app.error, bl.BodyTooLarge)


def test_a_single_huge_message_is_dropped_not_passed_on():
    """A server may deliver a whole body in one message; the app must not get it."""
    app = Reader()
    rig = Rig([b"\0" * (50 * MIB)])
    run(rig.call(limited(app, MIB), scope()))
    assert rig.status == 413 and app.read == 0


def test_an_understated_content_length_does_not_buy_extra_room():
    app = Reader()
    rig = Rig([b"\0" * (256 * 1024)] * 40)
    run(rig.call(limited(app, MIB), scope(headers=[length(10)])))
    assert rig.status == 413
    assert rig.pulled <= 5


@pytest.mark.parametrize("value", [
    b"", b"   ", b"abc", b"-5", b"+5", b"1e6", b"12.5", b"1_000", b"0x10", b"10 20", b"\xd9\xa3",
])
def test_a_garbage_content_length_is_ignored_and_the_stream_is_counted_instead(value):
    small = Reader()
    rig = Rig([b"x" * 100])
    run(rig.call(limited(small, 1000), scope(headers=[(b"content-length", value)])))
    assert rig.status == 200 and small.read == 100, value

    big = Reader()
    rig = Rig([b"x" * 600, b"x" * 600])
    run(rig.call(limited(big, 1000), scope(headers=[(b"content-length", value)])))
    assert rig.status == 413, value


def test_a_content_length_with_thousands_of_digits_is_refused_and_does_not_crash():
    """Python 3.11+ refuses to int() more than 4300 digits; that must not become a 500."""
    rig = Rig([])
    run(rig.call(limited(Reader(), MIB), scope(headers=[(b"content-length", b"9" * 6000)])))
    assert rig.status == 413


def test_with_several_content_length_headers_the_largest_one_counts():
    rig = Rig([])
    run(rig.call(limited(Reader(), 1000), scope(headers=[length(10), length(5000), length(20)])))
    assert rig.status == 413


def test_declared_length_parsing():
    parse = bl.declared_length
    assert parse([]) is None
    assert parse([(b"host", b"x")]) is None
    assert parse([(b"content-length", b"0")]) == 0
    assert parse([(b"content-length", b" 42 ")]) == 42
    assert parse([(b"Content-Length", b"7")]) == 7
    assert parse([(b"content-length", b"-1")]) is None
    assert parse([(b"content-length", b"1,2")]) is None
    assert parse([(b"content-length", b"3"), (b"content-length", b"9")]) == 9
    assert parse([(b"content-length", b"9" * 40)]) >= 10 ** 18


# --- what the middleware leaves alone -------------------------------------------------

@pytest.mark.parametrize("method", ["GET", "HEAD", "OPTIONS"])
def test_bodyless_methods_are_never_limited(method):
    app = Reader()
    rig = Rig([b"x" * 5000])
    run(rig.call(limited(app, 10), scope(method=method, headers=[length(5000)])))
    assert rig.status == 200 and app.read == 5000


@pytest.mark.parametrize("method", ["POST", "PUT", "PATCH", "DELETE"])
def test_every_state_changing_method_is_limited(method):
    rig = Rig([b"x" * 5000])
    run(rig.call(limited(Reader(), 10), scope(method=method)))
    assert rig.status == 413


def test_no_limit_for_a_path_means_no_check_and_no_wrapping():
    seen = {}

    async def app(sc, receive, send):
        seen["receive"] = receive
        await send({"type": "http.response.start", "status": 204, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    rig = Rig([b"x" * MIB])
    mw = bl.BodyLimitMiddleware(app, limit_for=lambda path: None)
    run(rig.call(mw, scope(headers=[length(10 ** 12)])))
    assert rig.status == 204 and seen["receive"] == rig.receive


def test_a_plain_number_is_a_limit_for_every_path():
    rig = Rig([b"x" * 100])
    run(rig.call(bl.BodyLimitMiddleware(Reader(), 50), scope(path="/anything")))
    assert rig.status == 413
    rig = Rig([b"x" * 100])
    run(rig.call(bl.BodyLimitMiddleware(Reader(), None), scope(path="/anything")))
    assert rig.status == 200


def test_the_limit_is_looked_up_per_request_from_the_path():
    """Limits that depend on configuration are read when needed, not frozen at start-up."""
    limits = {"/small": 10, "/big": 10_000}
    mw = bl.BodyLimitMiddleware(Reader(), limit_for=lambda path: limits.get(path))
    for path, expected in (("/small", 413), ("/big", 200), ("/other", 200)):
        rig = Rig([b"x" * 100])
        run(rig.call(mw, scope(path=path)))
        assert rig.status == expected, path
    limits["/big"] = 50
    rig = Rig([b"x" * 100])
    run(rig.call(mw, scope(path="/big")))
    assert rig.status == 413


@pytest.mark.parametrize("kind", ["lifespan", "websocket"])
def test_other_scope_types_pass_through_untouched(kind):
    seen = []

    async def app(sc, receive, send):
        seen.append((sc, receive, send))

    rig = Rig([])
    run(rig.call(limited(app, 1), scope(kind=kind)))
    assert seen and seen[0][0]["type"] == kind and seen[0][1] == rig.receive


# --- the app misbehaves once the limit is crossed -------------------------------------

def test_an_app_that_turns_the_failed_read_into_its_own_400_still_gets_the_413():
    """FastAPI answers 400 to a body it could not parse; a handler may catch anything."""

    async def app(sc, receive, send):
        try:
            while True:
                message = await receive()
                if not message.get("more_body"):
                    break
        except Exception:
            body = b'{"detail": "There was an error parsing the body"}'
            await send({"type": "http.response.start", "status": 400, "headers": []})
            await send({"type": "http.response.body", "body": body})

    rig = Rig([b"x" * 600] * 5)
    run(rig.call(limited(app, 1000), scope()))
    assert rig.status == 413, "the app's 400 replaced the limit's answer"
    assert b"error parsing" not in rig.body


def test_an_app_that_raises_something_else_after_the_limit_still_gets_the_413():
    class Boom(Exception):
        pass

    async def app(sc, receive, send):
        try:
            while True:
                await receive()
        except Exception as exc:
            raise Boom("parser blew up") from exc

    rig = Rig([b"x" * 600] * 5)
    run(rig.call(limited(app, 1000), scope()))
    assert rig.status == 413


def test_an_app_error_unrelated_to_the_body_is_not_swallowed():
    class Boom(Exception):
        pass

    async def app(sc, receive, send):
        raise Boom("a bug")

    rig = Rig([b"x"])
    with pytest.raises(Boom):
        run(rig.call(limited(app, 1000), scope()))


def test_an_app_that_ignores_the_error_and_keeps_reading_gets_it_again_and_the_413():
    attempts = []

    async def app(sc, receive, send):
        for _ in range(5):
            try:
                await receive()
            except bl.BodyTooLarge:
                attempts.append("refused")

    rig = Rig([b"x" * 600] * 10)
    run(rig.call(limited(app, 1000), scope()))
    assert rig.status == 413
    assert len(attempts) >= 4, "after the limit every further read must fail, not resume the body"
    assert rig.pulled <= 2


def test_an_app_that_already_started_answering_gets_no_second_response():
    """Nothing can be replaced once a response has begun; sending a 413 as well would
    be two responses on one connection."""

    async def app(sc, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})
        try:
            while True:
                await receive()
        except bl.BodyTooLarge:
            pass
        await send({"type": "http.response.body", "body": b"early answer"})

    rig = Rig([b"x" * 600] * 5)
    run(rig.call(limited(app, 1000), scope()))
    assert rig.status == 200          # exactly one response.start, and it is the app's
    assert rig.body == b"early answer"


def test_an_app_that_answers_without_reading_the_body_is_left_alone():
    async def app(sc, receive, send):
        await send({"type": "http.response.start", "status": 401, "headers": []})
        await send({"type": "http.response.body", "body": b"no"})

    rig = Rig([b"x" * 5000])
    run(rig.call(limited(app, 100), scope()))
    assert rig.status == 401 and rig.body == b"no" and rig.pulled == 0


def test_an_app_that_never_reads_a_chunked_body_costs_nothing():
    rig = Rig([b"\0" * MIB] * 100)

    async def app(sc, receive, send):
        await send({"type": "http.response.start", "status": 204, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    run(rig.call(limited(app, 1000), scope()))
    assert rig.status == 204 and rig.pulled == 0


# --- clients that go away -------------------------------------------------------------

def test_a_client_that_hangs_up_mid_upload_reaches_the_app_as_a_disconnect():
    app = Reader()
    rig = Rig([b"x" * 100] * 10)
    rig.disconnect_after_chunk = 3
    with pytest.raises(ConnectionAbortedError):
        run(rig.call(limited(app, 10_000), scope()))
    assert rig.sent == [], "nothing may be sent to a client that is gone"


def test_a_refusal_to_a_client_that_is_already_gone_does_not_raise():
    class Gone(Rig):
        async def send(self, message):
            raise ConnectionResetError("peer closed")

    rig = Gone([])
    run(rig.call(limited(Reader(), 10), scope(headers=[length(11)])))       # declared
    rig = Gone([b"x" * 100])
    run(rig.call(limited(Reader(), 10), scope()))                           # streamed


def test_the_middleware_keeps_no_state_between_requests():
    mw = limited(Reader(), 1000)
    for _ in range(3):
        rig = Rig([b"x" * 2000])
        run(rig.call(mw, scope()))
        assert rig.status == 413
        rig = Rig([b"x" * 200])
        run(rig.call(mw, scope()))
        assert rig.status == 200


def test_concurrent_requests_are_counted_separately():
    async def scenario():
        mw = limited(Reader(), 1000)
        rigs = [Rig([b"x" * 400] * 2), Rig([b"x" * 400] * 4), Rig([b"x" * 100])]
        await asyncio.gather(*(rig.call(mw, scope()) for rig in rigs))
        return [rig.status for rig in rigs]

    assert run(scenario()) == [200, 413, 200]
