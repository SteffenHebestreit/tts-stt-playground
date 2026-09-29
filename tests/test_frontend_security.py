"""Request hardening of the gateway: cross-origin access, API key, response headers.

The review found the gateway defaulting to `ALLOWED_ORIGINS=*` with every method
allowed, so any web page open in a browser on the LAN could delete trained
models, unload backends or start training jobs (a preflight from an arbitrary
origin for DELETE was answered 200 with `Access-Control-Allow-Origin: *`).

Each scenario needs a differently configured module, because the policy is read
at import time. `tests/frontend_loader.py` imports a fresh copy per configuration
and swaps the backend HTTP client for a recorder, so "the request was refused"
is asserted as "no backend was contacted", not just as a status code.
"""

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from frontend_loader import install_stub, load_frontend_app

EVIL = "https://evil.example"
DELETE_JOB = "/api/training/model/job-1"
UNLOAD = "/api/providers/qwen3/unload"


def _ok(method, url, kwargs):
    return {"status": "success"}


def _gateway(monkeypatch, env=None, **client_kwargs):
    module = load_frontend_app(env)
    stub = install_stub(monkeypatch, module, _ok)
    return module, stub, TestClient(module.app, **client_kwargs)


# --- default: same-origin only ------------------------------------------------


def test_default_sends_no_cors_headers(monkeypatch):
    _, _, client = _gateway(monkeypatch)
    r = client.get("/providers", headers={"Origin": EVIL})
    assert r.status_code == 200
    assert "access-control-allow-origin" not in r.headers


def test_default_preflight_from_a_foreign_origin_is_not_granted(monkeypatch):
    """The exact request the review used: a DELETE preflight from another site."""
    _, _, client = _gateway(monkeypatch)
    r = client.options(DELETE_JOB, headers={
        "Origin": EVIL, "Access-Control-Request-Method": "DELETE"})
    assert "access-control-allow-origin" not in r.headers
    assert "access-control-allow-methods" not in r.headers


@pytest.mark.parametrize("method,path", [
    ("DELETE", DELETE_JOB),
    ("DELETE", "/api/training/job/job-1"),
    ("POST", UNLOAD),
    ("POST", "/api/training/resume"),
    ("POST", "/api/tts"),
])
def test_default_refuses_state_changing_requests_from_a_foreign_origin(monkeypatch, method, path):
    _, stub, client = _gateway(monkeypatch)
    kwargs = {"json": {"provider": "piper", "text": "hi"}} if path == "/api/tts" else {}
    r = client.request(method, path, headers={"Origin": EVIL}, **kwargs)
    assert r.status_code == 403
    assert "ALLOWED_ORIGINS" in r.json()["detail"]
    assert stub.calls == [], "the request reached a backend although it was refused"


def test_same_origin_browser_request_is_allowed(monkeypatch):
    _, stub, client = _gateway(monkeypatch)
    r = client.delete(DELETE_JOB, headers={"Origin": "http://testserver"})
    assert r.status_code == 200
    assert len(stub.calls) == 1


def test_request_without_origin_is_allowed(monkeypatch):
    """curl, the OpenAI SDKs and other backends send no Origin at all."""
    _, stub, client = _gateway(monkeypatch)
    assert client.delete(DELETE_JOB).status_code == 200
    assert len(stub.calls) == 1


def test_null_origin_is_refused(monkeypatch):
    """Sandboxed iframes, file:// pages and some redirects send `Origin: null`."""
    _, stub, client = _gateway(monkeypatch)
    assert client.delete(DELETE_JOB, headers={"Origin": "null"}).status_code == 403
    assert stub.calls == []


def test_same_host_on_another_port_is_a_different_origin(monkeypatch):
    _, _, client = _gateway(monkeypatch)
    assert client.delete(DELETE_JOB, headers={"Origin": "http://testserver:9999"}).status_code == 403


def test_omitted_default_port_matches(monkeypatch):
    """`Origin: http://host` and `Host: host:80` are the same origin."""
    _, _, client = _gateway(monkeypatch)
    r = client.delete(DELETE_JOB, headers={"Host": "nas.local:80", "Origin": "http://nas.local"})
    assert r.status_code == 200
    r = client.delete(DELETE_JOB, headers={"Host": "nas.local", "Origin": "https://nas.local"})
    assert r.status_code == 200


def test_ipv6_and_mixed_case_hosts_compare_correctly(monkeypatch):
    _, _, client = _gateway(monkeypatch)
    same = client.delete(DELETE_JOB, headers={"Host": "[::1]:3000", "Origin": "http://[::1]:3000"})
    assert same.status_code == 200
    other = client.delete(DELETE_JOB, headers={"Host": "[::1]:3000", "Origin": "http://[::2]:3000"})
    assert other.status_code == 403
    mixed = client.delete(DELETE_JOB, headers={"Host": "NAS.Local:3000", "Origin": "http://nas.local:3000"})
    assert mixed.status_code == 200


def test_reads_are_not_subject_to_the_origin_check(monkeypatch):
    """A cross-origin GET is harmless without CORS: the page cannot read the answer."""
    _, _, client = _gateway(monkeypatch)
    assert client.get("/api/training/jobs", headers={"Origin": EVIL}).status_code == 200


def test_empty_allowed_origins_means_closed_not_wildcard(monkeypatch):
    """An empty value used to be turned into `["*"]`."""
    _, stub, client = _gateway(monkeypatch, {"ALLOWED_ORIGINS": ""})
    r = client.get("/providers", headers={"Origin": EVIL})
    assert "access-control-allow-origin" not in r.headers
    assert client.delete(DELETE_JOB, headers={"Origin": EVIL}).status_code == 403
    assert stub.calls == []


# --- opting in ----------------------------------------------------------------


def test_listed_origin_is_allowed_and_gets_cors_headers(monkeypatch):
    _, stub, client = _gateway(monkeypatch, {"ALLOWED_ORIGINS": "https://ui.example, https://other.example/"})

    r = client.delete(DELETE_JOB, headers={"Origin": "https://ui.example"})
    assert r.status_code == 200
    assert r.headers["access-control-allow-origin"] == "https://ui.example"

    # trailing slash in the setting is forgiven
    assert client.delete(DELETE_JOB, headers={"Origin": "https://other.example"}).status_code == 200

    n = len(stub.calls)
    assert client.delete(DELETE_JOB, headers={"Origin": EVIL}).status_code == 403
    assert len(stub.calls) == n


def test_listed_origin_preflight_succeeds_and_foreign_one_does_not(monkeypatch):
    _, _, client = _gateway(monkeypatch, {"ALLOWED_ORIGINS": "https://ui.example"})
    ok = client.options(DELETE_JOB, headers={
        "Origin": "https://ui.example", "Access-Control-Request-Method": "DELETE"})
    assert ok.status_code == 200
    assert ok.headers["access-control-allow-origin"] == "https://ui.example"
    bad = client.options(DELETE_JOB, headers={
        "Origin": EVIL, "Access-Control-Request-Method": "DELETE"})
    assert "access-control-allow-origin" not in bad.headers


def test_wildcard_only_when_configured_explicitly(monkeypatch):
    _, stub, client = _gateway(monkeypatch, {"ALLOWED_ORIGINS": "*"})
    r = client.delete(DELETE_JOB, headers={"Origin": EVIL})
    assert r.status_code == 200
    assert r.headers["access-control-allow-origin"] == "*"


def test_wildcard_is_called_out_in_the_log(monkeypatch, caplog):
    import logging
    with caplog.at_level(logging.WARNING):
        load_frontend_app({"ALLOWED_ORIGINS": "*"})
    assert any("ALLOWED_ORIGINS contains '*'" in r.getMessage() for r in caplog.records)

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        load_frontend_app({"ALLOWED_ORIGINS": "https://ui.example"})
    assert not any("ALLOWED_ORIGINS" in r.getMessage() for r in caplog.records)


def test_trusted_origin_passes_the_check_without_cors_headers(monkeypatch):
    """The UI behind a reverse proxy that rewrites Host: same site, other name."""
    env = {"TRUSTED_ORIGINS": "https://tts.example.com"}
    headers = {"Host": "frontend-service:3000", "Origin": "https://tts.example.com"}

    _, _, client = _gateway(monkeypatch)
    assert client.delete(DELETE_JOB, headers=headers).status_code == 403

    _, stub, client = _gateway(monkeypatch, env)
    r = client.delete(DELETE_JOB, headers=headers)
    assert r.status_code == 200
    assert "access-control-allow-origin" not in r.headers


def test_forwarded_host_is_ignored_unless_proxy_headers_are_trusted(monkeypatch):
    headers = {"Host": "frontend-service:3000", "X-Forwarded-Host": "tts.example.com",
               "Origin": "https://tts.example.com"}

    _, stub, client = _gateway(monkeypatch)
    assert client.delete(DELETE_JOB, headers=headers).status_code == 403, (
        "an attacker-supplied X-Forwarded-Host must not make a request same-origin")
    assert stub.calls == []

    _, _, client = _gateway(monkeypatch, {"TRUST_PROXY_HEADERS": "true"})
    assert client.delete(DELETE_JOB, headers=headers).status_code == 200


# --- optional API key ---------------------------------------------------------

KEY = "s3cret-key"
AUTH = {"Authorization": f"Bearer {KEY}"}


def test_v1_requires_the_key_in_the_openai_envelope(monkeypatch):
    _, _, client = _gateway(monkeypatch, {"API_KEY": KEY})
    for headers in ({}, {"Authorization": "Bearer wrong"}, {"Authorization": KEY},
                    {"Authorization": "Basic " + KEY}):
        r = client.get("/v1/models", headers=headers)
        assert r.status_code == 401, headers
        error = r.json()["error"]
        assert error["type"] == "authentication_error"
        assert error["code"] == "invalid_api_key"
        assert r.headers["www-authenticate"] == "Bearer"
    assert client.get("/v1/models", headers=AUTH).status_code == 200


def test_v1_speech_and_transcription_are_covered_too(monkeypatch):
    _, stub, client = _gateway(monkeypatch, {"API_KEY": KEY})
    assert client.post("/v1/audio/speech", json={"input": "hi"}).status_code == 401
    r = client.post("/v1/audio/transcriptions", files={"file": ("a.wav", b"RIFF", "audio/wav")})
    assert r.status_code == 401
    assert stub.calls == []


def test_mutating_api_calls_need_the_key_from_a_script(monkeypatch):
    _, stub, client = _gateway(monkeypatch, {"API_KEY": KEY})
    r = client.delete(DELETE_JOB)
    assert r.status_code == 401
    assert r.json() == {"detail": "A valid API key is required (Authorization: Bearer <key>)."}
    assert stub.calls == []
    assert client.delete(DELETE_JOB, headers=AUTH).status_code == 200


def test_mutating_api_calls_from_the_bundled_ui_need_no_key(monkeypatch):
    _, _, client = _gateway(monkeypatch, {"API_KEY": KEY})
    assert client.delete(DELETE_JOB, headers={"Sec-Fetch-Site": "same-origin"}).status_code == 200
    assert client.delete(DELETE_JOB, headers={"Sec-Fetch-Site": "none"}).status_code == 200
    assert client.delete(DELETE_JOB, headers={"Origin": "http://testserver"}).status_code == 200
    # ...but a page on another site does not get the exemption
    assert client.delete(DELETE_JOB, headers={"Sec-Fetch-Site": "cross-site"}).status_code == 401
    assert client.delete(DELETE_JOB, headers={"Origin": EVIL}).status_code == 403


def test_reads_and_probes_stay_open_with_a_key(monkeypatch):
    _, _, client = _gateway(monkeypatch, {"API_KEY": KEY})
    for path in ("/health", "/providers", "/api/training/jobs", "/"):
        assert client.get(path).status_code == 200, path


def test_preflight_is_never_asked_for_the_key(monkeypatch):
    """Preflights carry no Authorization header; demanding one would break every
    cross-origin browser client, listed origin or not."""
    _, _, client = _gateway(monkeypatch, {"API_KEY": KEY, "ALLOWED_ORIGINS": "https://ui.example"})
    r = client.options("/v1/audio/speech", headers={
        "Origin": "https://ui.example", "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "authorization,content-type"})
    assert r.status_code == 200


def test_key_is_compared_in_constant_time(monkeypatch):
    module, _, client = _gateway(monkeypatch, {"API_KEY": KEY})
    seen = []
    real = module.hmac.compare_digest

    def spy(a, b):
        seen.append((a, b))
        return real(a, b)

    monkeypatch.setattr(module.hmac, "compare_digest", spy)
    client.get("/v1/models", headers=AUTH)
    assert seen == [(KEY.encode(), KEY.encode())]


def test_no_key_configured_leaves_v1_open(monkeypatch):
    _, _, client = _gateway(monkeypatch)
    assert client.get("/v1/models").status_code == 200


# --- response headers ---------------------------------------------------------


def test_security_headers_on_every_kind_of_response(monkeypatch):
    _, _, client = _gateway(monkeypatch)
    responses = [
        client.get("/"),                                  # rendered page
        client.get("/health"),                            # JSON
        client.get("/nope"),                              # 404
        client.get("/static/js/app.js"),                  # static file
        client.delete(DELETE_JOB, headers={"Origin": EVIL}),   # refused by the guard
        client.get("/v1/models"),                         # openai router
    ]
    for r in responses:
        assert r.headers["x-content-type-options"] == "nosniff", r.request.url
        assert r.headers["referrer-policy"] == "same-origin", r.request.url
        assert r.headers["x-frame-options"] == "SAMEORIGIN", r.request.url


def test_no_content_security_policy_is_sent(monkeypatch):
    """The template still uses inline handlers; a CSP would break the UI."""
    _, _, client = _gateway(monkeypatch)
    assert "content-security-policy" not in client.get("/").headers


# --- the WebSocket relay ------------------------------------------------------
#
# CORS middleware never runs on WebSocket routes and browsers do not apply the
# same-origin policy to them, so the relay checks Origin itself. Both cases below
# use a provider that has no live transcription: a request that gets past the
# Origin check is then closed for THAT reason, which the close reason tells apart
# from the Origin refusal (empty).


def _ws_close_reason(client, origin):
    headers = {"Origin": origin} if origin else {}
    with pytest.raises(WebSocketDisconnect) as closed:
        with client.websocket_connect("/ws/stt?provider=qwen3-asr", headers=headers) as ws:
            ws.receive_text()
    return closed.value.code, closed.value.reason


def test_websocket_from_a_foreign_origin_is_refused_by_default(monkeypatch):
    _, _, client = _gateway(monkeypatch)
    assert _ws_close_reason(client, EVIL) == (1008, "")


def test_websocket_from_the_same_origin_gets_past_the_origin_check(monkeypatch):
    """The default used to be `*`; tightening it must not lock the UI's own mic out."""
    _, _, client = _gateway(monkeypatch)
    code, reason = _ws_close_reason(client, "http://testserver")
    assert code == 1008 and "does not support live transcription" in reason


def test_websocket_without_origin_is_not_a_browser_and_passes(monkeypatch):
    _, _, client = _gateway(monkeypatch)
    code, reason = _ws_close_reason(client, None)
    assert "does not support live transcription" in reason


def test_websocket_from_a_listed_origin_passes(monkeypatch):
    _, _, client = _gateway(monkeypatch, {"ALLOWED_ORIGINS": "https://ui.example"})
    code, reason = _ws_close_reason(client, "https://ui.example")
    assert "does not support live transcription" in reason
    assert _ws_close_reason(client, EVIL) == (1008, "")
