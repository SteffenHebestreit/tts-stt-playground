"""origin_guard.py: the ALLOWED_ORIGINS rules and the cross-origin check every backend shares.

The gateway was made same-origin by default; the backends behind it were not. A
published backend port (127.0.0.1 by default, every interface with
BACKEND_BIND_ADDR=0.0.0.0) let any web page open in a browser send a multipart POST
- a "simple request", no preflight - to /train-from-dataset, /resume-training,
/export/{id}, /upload_model or /voices/save, and, with ``ALLOWED_ORIGINS=*`` (the
default, and also what an empty value meant), read the answers and issue preflighted
DELETEs as well.

Tested here as raw ASGI, so every header shape is exact. tests/
test_origin_guard_backends.py runs the same questions through each backend's app.
"""

from __future__ import annotations

import asyncio
import ast
import logging
import re
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest

from backend_apps import BACKENDS

REPO = Path(__file__).resolve().parents[1]
MODULE = REPO / "piper-tts-service" / "origin_guard.py"


def _load(path: Path = MODULE):
    spec = spec_from_file_location(f"origin_guard_under_test_{abs(hash(str(path)))}", path)
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


og = _load()


# --- the copies -----------------------------------------------------------------------

def test_every_backend_ships_an_identical_copy():
    """One copy per Docker build context; a fix made in one and not the others is the bug."""
    copies = {name: (REPO / name / "origin_guard.py").read_bytes() for name in BACKENDS}
    reference = copies["piper-tts-service"]
    diverged = sorted(name for name, data in copies.items() if data != reference)
    assert not diverged, (
        f"origin_guard.py differs in {diverged}. The copies exist because each service builds its "
        f"image from its own directory, not because they may behave differently: edit one and "
        f"copy it over the others.")


def test_no_other_service_directory_carries_an_unlisted_copy():
    found = sorted(p.parent.name for p in REPO.glob("*-service/origin_guard.py"))
    assert found == sorted(BACKENDS), f"origin_guard.py copies: {found}; expected exactly {sorted(BACKENDS)}"


@pytest.mark.parametrize("name", BACKENDS)
def test_every_backend_installs_the_guard_and_configures_cors_from_it(name):
    source = (REPO / name / "app.py").read_text(encoding="utf-8")
    assert "from origin_guard import OriginGuardMiddleware, parse_allowed_origins" in source
    assert re.search(r"app\.add_middleware\(\s*OriginGuardMiddleware\b", source), f"{name} does not install it"
    assert 'parse_allowed_origins(os.getenv("ALLOWED_ORIGINS", ""))' in source
    # The old parse turned an unset or empty value into "*"; the default must never be a wildcard again.
    assert 'os.getenv("ALLOWED_ORIGINS", "*")' not in source
    assert '["*"]' not in source.split("allowed_origins = ", 1)[1].split("\n", 1)[0]
    # CORS headers only when there is something to allow.
    assert re.search(r"if allowed_origins:\s+app\.add_middleware\(\s*CORSMiddleware", source), (
        f"{name} adds CORSMiddleware unconditionally: with no origins it must not be there at all")


def test_the_module_needs_nothing_outside_the_standard_library_and_reads_no_environment():
    tree = ast.parse(MODULE.read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert imported <= set(sys.stdlib_module_names), imported - set(sys.stdlib_module_names)
    # The service passes the configured value in; tests/test_env_wiring.py looks for os.getenv("X")
    # calls per service directory, and a hidden read here would not be found.
    assert "os" not in imported


# --- parsing ALLOWED_ORIGINS ----------------------------------------------------------

@pytest.mark.parametrize("raw", [None, "", "   ", ",", " , ,, ", "\t"])
def test_unset_or_empty_means_closed_not_a_wildcard(raw):
    assert og.parse_allowed_origins(raw) == []


def test_an_explicit_list_is_normalised():
    assert og.parse_allowed_origins("https://ui.example, http://Other.Example:8080/ ,https://ui.example") == [
        "https://ui.example", "http://other.example:8080"]


def test_a_wildcard_is_honoured_but_logged(caplog):
    with caplog.at_level(logging.WARNING, logger="origin_guard"):
        assert og.parse_allowed_origins("*") == ["*"]
    assert any("ALLOWED_ORIGINS contains '*'" in r.getMessage() for r in caplog.records)


def test_a_wildcard_among_others_is_logged_too(caplog):
    with caplog.at_level(logging.WARNING, logger="origin_guard"):
        assert "*" in og.parse_allowed_origins("https://a.example, *")
    assert caplog.records


def test_an_explicit_list_or_no_value_logs_nothing(caplog):
    with caplog.at_level(logging.DEBUG, logger="origin_guard"):
        og.parse_allowed_origins("https://ui.example")
        og.parse_allowed_origins("")
        og.parse_allowed_origins(None)
    assert not caplog.records


# --- the same-origin rule -------------------------------------------------------------

@pytest.mark.parametrize("origin, host, same", [
    ("http://192.168.1.5:5000", "192.168.1.5:5000", True),
    ("http://localhost:5000", "localhost:5000", True),
    ("http://LOCALHOST:5000", "localhost:5000", True),
    ("http://backend", "backend", True),
    ("http://backend", "backend:80", True),            # the default port is implied
    ("https://backend", "backend:443", True),
    ("https://backend:443", "backend", True),
    ("http://[::1]:5000", "[::1]:5000", True),
    ("http://evil.example", "192.168.1.5:5000", False),
    ("http://192.168.1.5:5001", "192.168.1.5:5000", False),     # same host, other port: another site
    ("http://192.168.1.5", "192.168.1.5:5000", False),
    ("http://backend:80", "backend:5000", False),
    ("null", "backend", False),                        # sandboxed iframe, file://, some redirects
    ("", "backend", False),
    ("http://", "backend", False),
    ("ftp://backend", "backend", False),
    ("http://backend:99999", "backend", False),        # not a port
    ("http://evil.example", None, False),
    ("http://backend", "", False),
    ("garbage", "garbage", False),
])
def test_is_same_origin(origin, host, same):
    assert og.is_same_origin(origin, host) is same


def test_a_host_that_only_looks_similar_is_not_the_same():
    assert not og.is_same_origin("http://backend.evil.example", "backend")
    assert not og.is_same_origin("http://backend", "backend.evil.example")
    assert not og.is_same_origin("http://evil.example/backend", "backend")
    assert not og.is_same_origin("http://backend@evil.example", "backend")


def test_origin_permitted_combines_the_list_the_wildcard_and_the_host():
    assert og.origin_permitted("https://ui.example", "backend:5000", ["https://ui.example"])
    assert og.origin_permitted("HTTPS://UI.EXAMPLE/", "backend:5000", ["https://ui.example"])
    assert not og.origin_permitted("https://evil.example", "backend:5000", ["https://ui.example"])
    assert og.origin_permitted("https://evil.example", "backend:5000", ["*"])
    assert og.origin_permitted("http://backend:5000", "backend:5000", [])
    assert not og.origin_permitted("http://evil.example", "backend:5000", [])
    assert not og.origin_permitted("null", "backend:5000", ["https://ui.example"])
    assert og.origin_permitted("null", "backend:5000", ["*"])


# --- the middleware, as raw ASGI ------------------------------------------------------

def scope(method="POST", path="/x", origin=None, host=b"backend:5000", kind="http", extra=()):
    headers = [(b"host", host)] if host is not None else []
    for value in ([origin] if isinstance(origin, (str, bytes)) else (origin or [])):
        headers.append((b"origin", value if isinstance(value, bytes) else value.encode()))
    headers.extend(extra)
    sc = {"type": kind, "path": path, "headers": headers, "query_string": b""}
    if kind == "http":
        sc["method"] = method
    return sc


class Rig:
    def __init__(self):
        self.sent = []
        self.received = 0

    async def receive(self):
        self.received += 1
        return {"type": "websocket.connect"} if self.received == 1 else {"type": "websocket.disconnect"}

    async def send(self, message):
        self.sent.append(message)


def call(sc, allowed=(), app=None):
    reached = []

    async def default_app(s, receive, send):
        reached.append(s)
        if s["type"] == "http":
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"ok"})

    rig = Rig()
    mw = og.OriginGuardMiddleware(app or default_app, allowed_origins=allowed)
    asyncio.run(mw(sc, rig.receive, rig.send))
    rig.reached = bool(reached)
    starts = [m for m in rig.sent if m["type"] == "http.response.start"]
    rig.status = starts[0]["status"] if starts else None
    return rig


@pytest.mark.parametrize("method", ["POST", "PUT", "PATCH", "DELETE"])
def test_a_state_changing_request_from_a_foreign_origin_is_refused(method):
    rig = call(scope(method=method, origin="https://evil.example"))
    assert rig.status == 403 and not rig.reached


def test_the_refusal_is_a_small_json_403_that_closes_the_connection():
    import json

    rig = call(scope(origin="https://evil.example"))
    start = next(m for m in rig.sent if m["type"] == "http.response.start")
    body = next(m for m in rig.sent if m["type"] == "http.response.body")["body"]
    headers = dict(start["headers"])
    assert headers[b"content-type"] == b"application/json"
    assert int(headers[b"content-length"]) == len(body)
    assert headers[b"connection"] == b"close"
    detail = json.loads(body)["detail"]
    assert "ALLOWED_ORIGINS" in detail and "https://evil.example" in detail


def test_a_huge_origin_is_not_echoed_whole():
    import json

    rig = call(scope(origin="https://" + "a" * 5000 + ".example"))
    body = next(m for m in rig.sent if m["type"] == "http.response.body")["body"]
    assert len(body) < 1000 and "..." in json.loads(body)["detail"]


@pytest.mark.parametrize("method", ["GET", "HEAD", "OPTIONS"])
def test_safe_methods_are_not_checked(method):
    """Reading is CORS's business (no headers, so the page cannot read it), and a
    preflight has to reach the CORS layer."""
    rig = call(scope(method=method, origin="https://evil.example"))
    assert rig.status == 200 and rig.reached


def test_a_request_without_an_origin_is_not_a_browser_and_passes():
    """The gateway, curl, benchmarks and the other containers send no Origin."""
    for method in ("POST", "DELETE", "PUT"):
        rig = call(scope(method=method, origin=None))
        assert rig.status == 200 and rig.reached, method


def test_a_request_without_any_host_but_with_a_foreign_origin_is_refused():
    assert call(scope(origin="http://x", host=None)).status == 403


def test_the_same_origin_passes():
    assert call(scope(origin="http://backend:5000")).status == 200
    assert call(scope(origin="http://BACKEND:5000")).status == 200


def test_another_port_on_the_same_host_is_another_origin():
    assert call(scope(origin="http://backend:5001")).status == 403


def test_a_listed_origin_passes_and_others_do_not():
    allowed = ["https://ui.example"]
    assert call(scope(origin="https://ui.example"), allowed).status == 200
    assert call(scope(origin="https://ui.example/"), allowed).status == 200
    assert call(scope(origin="https://Other.example"), allowed).status == 403


def test_the_configured_list_is_normalised_by_the_middleware_too():
    assert call(scope(origin="https://ui.example"), ["HTTPS://UI.EXAMPLE/", " "]).status == 200


def test_a_wildcard_admits_every_origin():
    assert call(scope(origin="https://evil.example"), ["*"]).status == 200
    assert call(scope(origin="null"), ["*"]).status == 200


@pytest.mark.parametrize("origin", ["null", "", "garbage", "file://", "chrome-extension://abc"])
def test_opaque_and_malformed_origins_are_refused(origin):
    assert call(scope(origin=origin)).status == 403


def test_every_origin_header_of_a_request_must_be_permitted():
    """A duplicated header cannot smuggle a foreign origin past a check of the first one."""
    rig = call(scope(origin=["http://backend:5000", "https://evil.example"]))
    assert rig.status == 403


def test_a_client_that_is_gone_does_not_break_the_refusal():
    class Gone(Rig):
        async def send(self, message):
            raise ConnectionResetError("peer closed")

    rig = Gone()
    mw = og.OriginGuardMiddleware(lambda *a: None)
    asyncio.run(mw(scope(origin="https://evil.example"), rig.receive, rig.send))


def test_lifespan_scopes_pass_through():
    rig = call(scope(kind="lifespan", origin="https://evil.example"))
    assert rig.reached


# --- WebSocket handshakes --------------------------------------------------------------

def test_a_websocket_handshake_from_a_foreign_origin_is_closed_with_1008():
    """Browsers do not apply the same-origin policy to WebSockets: any page can dial one."""
    rig = call(scope(kind="websocket", origin="https://evil.example", path="/ws/transcribe"))
    assert not rig.reached
    assert rig.sent == [{"type": "websocket.close", "code": 1008}]
    assert rig.received == 1, "the handshake request was not read before being refused"


def test_a_websocket_without_an_origin_or_from_a_permitted_one_reaches_the_app():
    assert call(scope(kind="websocket", origin=None)).reached
    assert call(scope(kind="websocket", origin="http://backend:5000")).reached
    assert call(scope(kind="websocket", origin="https://ui.example"), ["https://ui.example"]).reached
    assert call(scope(kind="websocket", origin="https://evil.example"), ["*"]).reached
