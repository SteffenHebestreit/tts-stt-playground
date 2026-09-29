"""The ALLOWED_ORIGINS contract and the cross-origin check, through every backend's own app.

What the security review found: every backend defaulted ``ALLOWED_ORIGINS`` to ``*``
and read an empty value as ``*`` too. A backend port published on the host (compose
binds 127.0.0.1 by default, all interfaces with BACKEND_BIND_ADDR=0.0.0.0) could then
be driven by any web page in a browser that can reach it: multipart POSTs to
/train-from-dataset, /resume-training, /export/{id}, /upload_model, /voices/save are
"simple requests" that need no preflight, and a preflighted DELETE /model/{id} was
answered with ``Access-Control-Allow-Origin: *``.

The contract now, for all eight backends:

* ALLOWED_ORIGINS unset or empty: no CORS headers at all (closed);
* ``*`` only when written down (with a warning); otherwise an explicit list;
* independently of CORS, a state-changing request with an Origin header naming
  another host than the one it was addressed to is answered 403 without running the
  handler, unless that origin is listed. No Origin (gateway, curl, benchmarks,
  other containers) is never affected.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging

import pytest

from backend_apps import (
    BACKENDS, load_backend, multipart_content_type, multipart_head, spy_on_route,
)

EVIL = "https://evil.example"
UI = "https://ui.example"

# The routes the review named (and a few more that change state), as
# (method, route template, concrete path).
STATE_CHANGING = {
    "piper-training-service": [
        ("POST", "/train-from-dataset", "/train-from-dataset"),
        ("POST", "/resume-training", "/resume-training"),
        ("POST", "/export/{job_id}", "/export/abc"),
        ("POST", "/train", "/train"),
        ("DELETE", "/model/{job_id}", "/model/abc"),
        ("DELETE", "/job/{job_id}", "/job/abc"),
    ],
    "piper-tts-service": [
        ("POST", "/upload_model", "/upload_model"),
        ("POST", "/tts", "/tts"),
        ("DELETE", "/voice/{voice_name}", "/voice/luna"),
        ("POST", "/refresh_voices", "/refresh_voices"),
    ],
    "qwen3-tts-service": [
        ("POST", "/voices/save", "/voices/save"),
        ("DELETE", "/voices/{voice_id}", "/voices/abc"),
        ("POST", "/load_model", "/load_model"),
        ("POST", "/unload", "/unload"),
    ],
    "chatterbox-tts-service": [
        ("POST", "/clone", "/clone"),
        ("POST", "/unload", "/unload"),
    ],
    "stt-service": [
        ("POST", "/transcribe", "/transcribe"),
        ("POST", "/unload", "/unload"),
    ],
    "qwen3-asr-service": [
        ("POST", "/transcribe", "/transcribe"),
        ("POST", "/unload", "/unload"),
    ],
    "parakeet-asr-service": [
        ("POST", "/transcribe", "/transcribe"),
        ("POST", "/unload", "/unload"),
    ],
    "canary-asr-service": [
        ("POST", "/transcribe", "/transcribe"),
        ("POST", "/unload", "/unload"),
    ],
}


def _cases():
    return [(name, *route) for name in BACKENDS for route in STATE_CHANGING[name]]


def _run(coro):
    return asyncio.run(coro)


def _load(name, monkeypatch, tmp_path, stack, **env):
    return load_backend(name, monkeypatch, tmp_path, stack, **env)


@pytest.fixture(params=BACKENDS)
def closed(request, monkeypatch, tmp_path):
    """A backend with ALLOWED_ORIGINS unset: the default."""
    with contextlib.ExitStack() as stack:
        yield _load(request.param, monkeypatch, tmp_path, stack)


@pytest.fixture(params=_cases(), ids=lambda case: f"{case[0]}:{case[1]}{case[3]}")
def route_case(request, monkeypatch, tmp_path):
    name, method, template, path = request.param
    with contextlib.ExitStack() as stack:
        yield _load(name, monkeypatch, tmp_path, stack), method, template, path


def _small_form(backend, path):
    """A tiny multipart body: what a page can send as a "simple request", without any preflight."""
    body = multipart_head("file", extra_fields={"model_name": "voice", "job_id": "abc", "text": "hallo"}) \
        + b"\0" * 64 + b"\r\n--b--\r\n"
    return body, multipart_content_type()


# --- the default is closed --------------------------------------------------------------

def test_the_default_sends_no_cors_headers_at_all(closed):
    async def scenario():
        async with closed.client() as client:
            simple = await client.get("/health", headers={"Origin": EVIL})
            preflight = await client.options("/health", headers={
                "Origin": EVIL, "Access-Control-Request-Method": "DELETE",
                "Access-Control-Request-Headers": "content-type"})
            return simple, preflight

    simple, preflight = _run(scenario())

    for response in (simple, preflight):
        cors = {k: v for k, v in response.headers.items() if k.lower().startswith("access-control-")}
        assert cors == {}, f"{closed.name} sent CORS headers by default: {cors}"


def test_an_empty_value_is_closed_too_not_a_wildcard(monkeypatch, tmp_path):
    """The old parse turned "" into ["*"], so the one setting that should say "closed" opened everything."""
    for value in ("", "  ", ","):
        for name in BACKENDS:
            with contextlib.ExitStack() as stack:
                backend = _load(name, monkeypatch, tmp_path / f"{name}{len(value)}", stack, ALLOWED_ORIGINS=value)
                assert backend.module.allowed_origins == [], (name, value)

                async def scenario():
                    async with backend.client() as client:
                        return await client.get("/health", headers={"Origin": EVIL})

                assert "access-control-allow-origin" not in _run(scenario()).headers, (name, value)


def test_a_foreign_origin_cannot_send_a_state_changing_request(route_case):
    """The review's core finding: no preflight is needed for a multipart POST, so CORS never protected these."""
    backend, method, template, path = route_case
    handler = spy_on_route(backend.app, template, method)
    body, headers = _small_form(backend, path)

    async def scenario():
        async with backend.client() as client:
            return await client.request(method, path, content=body, headers={**headers, "Origin": EVIL})

    response = _run(scenario())

    assert response.status_code == 403, f"{backend.name} {method} {path}: {response.status_code} {response.text}"
    assert "ALLOWED_ORIGINS" in response.json()["detail"]
    assert handler == [], "the handler ran for a request from a foreign origin"
    assert backend.leftovers() == []
    assert "access-control-allow-origin" not in response.headers


def test_requests_without_an_origin_are_untouched(route_case):
    """The gateway, curl, benchmarks and the other containers send no Origin."""
    backend, method, template, path = route_case
    handler = spy_on_route(backend.app, template, method)
    body, headers = _small_form(backend, path)

    async def scenario():
        async with backend.client() as client:
            return await client.request(method, path, content=body, headers=headers)

    response = _run(scenario())

    assert response.status_code != 403, response.text


def test_the_same_origin_is_allowed_by_the_host_header(route_case):
    """A page served from the backend itself (its /docs, a reverse proxy) is not cross-origin."""
    backend, method, template, path = route_case
    body, headers = _small_form(backend, path)

    async def scenario():
        async with backend.client("http://backend.test:5000") as client:
            same = await client.request(
                method, path, content=body, headers={**headers, "Origin": "http://backend.test:5000"})
            other_port = await client.request(
                method, path, content=body, headers={**headers, "Origin": "http://backend.test:5001"})
            return same, other_port

    same, other_port = _run(scenario())

    assert same.status_code != 403, same.text
    assert other_port.status_code == 403


def test_the_preflighted_delete_is_no_longer_answered_with_a_wildcard(monkeypatch, tmp_path):
    """DELETE /model/{id} from any page: the preflight was answered ACAO: * and the DELETE went through."""
    with contextlib.ExitStack() as stack:
        backend = _load("piper-training-service", monkeypatch, tmp_path, stack)
        handler = spy_on_route(backend.app, "/model/{job_id}", "DELETE")

        async def scenario():
            async with backend.client() as client:
                preflight = await client.options("/model/abc", headers={
                    "Origin": EVIL, "Access-Control-Request-Method": "DELETE"})
                actual = await client.delete("/model/abc", headers={"Origin": EVIL})
                return preflight, actual

        preflight, actual = _run(scenario())

    assert "access-control-allow-origin" not in preflight.headers
    assert "access-control-allow-methods" not in preflight.headers
    assert actual.status_code == 403 and handler == []


# --- an explicit list -------------------------------------------------------------------

@pytest.fixture(params=BACKENDS)
def listed(request, monkeypatch, tmp_path):
    with contextlib.ExitStack() as stack:
        yield _load(request.param, monkeypatch, tmp_path, stack,
                    ALLOWED_ORIGINS=f"{UI}, http://Other.example:8080/ ")


def test_a_listed_origin_gets_cors_headers_and_may_change_state(listed):
    method, path = listed.routes["probe"]

    async def scenario():
        async with listed.client() as client:
            preflight = await client.options(path, headers={
                "Origin": UI, "Access-Control-Request-Method": method,
                "Access-Control-Request-Headers": "content-type"})
            actual = await client.request(method, path, headers={"Origin": UI})
            second = await client.request(method, path, headers={"Origin": "http://other.example:8080"})
            return preflight, actual, second

    preflight, actual, second = _run(scenario())

    assert preflight.status_code == 200
    assert preflight.headers["access-control-allow-origin"] == UI
    assert method in preflight.headers["access-control-allow-methods"] or "*" in preflight.headers[
        "access-control-allow-methods"]
    assert actual.status_code != 403
    assert actual.headers["access-control-allow-origin"] == UI
    assert second.status_code != 403, "the list is normalised: case and a trailing slash do not matter"
    assert second.headers["access-control-allow-origin"] == "http://other.example:8080"


def test_an_unlisted_origin_is_refused_everywhere(listed):
    method, path = listed.routes["probe"]

    async def scenario():
        async with listed.client() as client:
            preflight = await client.options(path, headers={
                "Origin": EVIL, "Access-Control-Request-Method": method})
            actual = await client.request(method, path, headers={"Origin": EVIL})
            simple = await client.get("/health", headers={"Origin": EVIL})
            return preflight, actual, simple

    preflight, actual, simple = _run(scenario())

    assert preflight.status_code == 400 and "access-control-allow-origin" not in preflight.headers
    assert actual.status_code == 403 and "access-control-allow-origin" not in actual.headers
    assert "access-control-allow-origin" not in simple.headers


def test_a_refusal_from_the_body_limit_still_carries_the_cors_headers_of_a_listed_origin(listed):
    """CORS is the outermost layer: without its headers the page reads a 413 as a network error."""
    path = listed.routes["upload"][1]

    async def scenario():
        async with listed.client() as client:
            return await client.post(
                path, content=b"x", headers={"Origin": UI, "content-length": str(10 * 1024 ** 3),
                                             **multipart_content_type()})

    response = _run(scenario())

    assert response.status_code == 413
    assert response.headers["access-control-allow-origin"] == UI


def test_credentials_are_allowed_only_with_an_explicit_list(monkeypatch, tmp_path):
    for origins, expected in ((UI, "true"), ("*", None)):
        for name in BACKENDS:
            with contextlib.ExitStack() as stack:
                backend = _load(name, monkeypatch, tmp_path / f"{name}{len(origins)}", stack,
                                ALLOWED_ORIGINS=origins, ALLOW_CREDENTIALS="true")
                origin = UI if origins == UI else EVIL

                async def scenario():
                    async with backend.client() as client:
                        return await client.options("/health", headers={
                            "Origin": origin, "Access-Control-Request-Method": "GET"})

                response = _run(scenario())
                assert response.headers.get("access-control-allow-credentials") == expected, (name, origins)


# --- an explicit wildcard ---------------------------------------------------------------

@pytest.mark.parametrize("name", BACKENDS)
def test_a_written_down_wildcard_still_works_and_warns(name, monkeypatch, tmp_path, caplog):
    with caplog.at_level(logging.WARNING):
        with contextlib.ExitStack() as stack:
            backend = _load(name, monkeypatch, tmp_path, stack, ALLOWED_ORIGINS="*")
    assert any("ALLOWED_ORIGINS contains '*'" in r.getMessage() for r in caplog.records), (
        f"{name}: opening the service to every origin must be logged")
    method, path = backend.routes["probe"]

    async def scenario():
        async with backend.client() as client:
            return await client.request(method, path, headers={"Origin": EVIL})

    response = _run(scenario())

    assert response.status_code != 403
    assert response.headers["access-control-allow-origin"] == "*"


@pytest.mark.parametrize("name", BACKENDS)
def test_a_list_or_the_default_does_not_warn(name, monkeypatch, tmp_path, caplog):
    with caplog.at_level(logging.WARNING):
        with contextlib.ExitStack() as stack:
            _load(name, monkeypatch, tmp_path / "a", stack, ALLOWED_ORIGINS=UI)
        with contextlib.ExitStack() as stack:
            _load(name, monkeypatch, tmp_path / "b", stack)
    assert not any("ALLOWED_ORIGINS" in r.getMessage() for r in caplog.records)


# --- the live-transcription WebSocket (stt-service only) -----------------------------------

def test_the_stt_websocket_refuses_a_foreign_origin_and_serves_the_rest(monkeypatch, tmp_path):
    """Browsers do not apply the same-origin policy to WebSockets and CORSMiddleware does
    not run on them; the guard is what stands between any page and this endpoint."""
    from starlette.testclient import TestClient
    from starlette.websockets import WebSocketDisconnect

    from test_stt_support import reset_model

    with contextlib.ExitStack() as stack:
        backend = _load("stt-service", monkeypatch, tmp_path, stack)
        reset_model(backend.module)
        backend.module._idle_unloader.cancel()
        client = TestClient(backend.app)

        with pytest.raises(WebSocketDisconnect) as refused:
            with client.websocket_connect("/ws/transcribe", headers={"origin": EVIL}):
                pass
        assert refused.value.code == 1008
        assert backend.module._live_sessions == 0

        # No Origin: the gateway's relay, scripts. And the same origin as the Host header.
        for headers in ({}, {"origin": "http://testserver"}):
            with client.websocket_connect("/ws/transcribe", headers=headers) as ws:
                ws.send_json({"event": "stop"})
                assert ws.receive_json()["type"] == "final", headers


def test_the_stt_websocket_admits_a_listed_origin(monkeypatch, tmp_path):
    from starlette.testclient import TestClient

    from test_stt_support import reset_model

    with contextlib.ExitStack() as stack:
        backend = _load("stt-service", monkeypatch, tmp_path, stack, ALLOWED_ORIGINS=UI)
        reset_model(backend.module)
        backend.module._idle_unloader.cancel()
        client = TestClient(backend.app)
        with client.websocket_connect("/ws/transcribe", headers={"origin": UI}) as ws:
            ws.send_json({"event": "stop"})
            assert ws.receive_json()["type"] == "final"
