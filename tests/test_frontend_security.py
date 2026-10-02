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

    # Trusting the proxy makes the forwarded name the effective host, which then
    # has to be a name the operator has said belongs to this service.
    _, _, client = _gateway(monkeypatch, {"TRUST_PROXY_HEADERS": "true"})
    assert client.delete(DELETE_JOB, headers=headers).status_code == 403
    _, _, client = _gateway(
        monkeypatch, {"TRUST_PROXY_HEADERS": "true", "TRUSTED_HOSTS": "tts.example.com"})
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


@pytest.mark.parametrize("browser_headers", [
    {"Sec-Fetch-Site": "same-origin"},
    {"Sec-Fetch-Site": "none"},
    {"Origin": "http://testserver"},
    {"Origin": "http://testserver", "Sec-Fetch-Site": "same-origin"},
    {"Sec-Fetch-Site": "cross-site"},
])
def test_mutating_api_calls_need_the_key_however_much_they_look_like_the_ui(monkeypatch, browser_headers):
    """`Origin == Host` and `Sec-Fetch-Site: same-origin` used to exempt a request from
    the key. Both are exactly what a DNS-rebinding page sends, so neither says who is
    calling; the bundled UI now sends the key like every other client."""
    _, stub, client = _gateway(monkeypatch, {"API_KEY": KEY})
    r = client.delete(DELETE_JOB, headers=browser_headers)
    assert r.status_code == 401
    assert stub.calls == [], "the request reached a backend without the key"
    assert r.headers["www-authenticate"] == "Bearer"

    assert client.delete(DELETE_JOB, headers={**browser_headers, **AUTH}).status_code == 200
    assert len(stub.calls) == 1


def test_a_foreign_origin_is_still_refused_before_the_key_is_considered(monkeypatch):
    _, stub, client = _gateway(monkeypatch, {"API_KEY": KEY})
    assert client.delete(DELETE_JOB, headers={"Origin": EVIL, **AUTH}).status_code == 403
    assert stub.calls == []


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
    """As SHA-256 digests: equal length whatever was sent, so not even the key's length leaks."""
    import hashlib

    module, _, client = _gateway(monkeypatch, {"API_KEY": KEY})
    seen = []
    real = module.hmac.compare_digest

    def spy(a, b):
        seen.append((a, b))
        return real(a, b)

    monkeypatch.setattr(module.hmac, "compare_digest", spy)
    client.get("/v1/models", headers=AUTH)
    digest = hashlib.sha256(KEY.encode()).digest()
    assert seen == [(digest, digest)]

    seen.clear()
    assert client.get("/v1/models", headers={"Authorization": "Bearer x"}).status_code == 401
    assert seen == [(hashlib.sha256(b"x").digest(), digest)]


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


# --- the Host header: DNS rebinding -------------------------------------------
#
# The Origin check compares Origin with Host, and under DNS rebinding both come
# from the attacker: a page on evil.example whose name was re-pointed at this
# service sends `Host: evil.example:3000` and `Origin: http://evil.example:3000`,
# which match, and the browser labels the request `Sec-Fetch-Site: same-origin`.
# So every one of those defences said "same origin" and the request went through.

REBOUND = {"Host": "evil.example:3000", "Origin": "http://evil.example:3000",
           "Sec-Fetch-Site": "same-origin"}


@pytest.mark.parametrize("env", [{}, {"API_KEY": KEY}], ids=["no-key", "with-key"])
def test_dns_rebinding_request_is_refused_and_reaches_no_backend(monkeypatch, env):
    """The exact request of the review: it returned 200 and reached the training
    service, with and without an API key configured."""
    _, stub, client = _gateway(monkeypatch, env)
    r = client.delete(DELETE_JOB, headers=REBOUND)
    assert r.status_code == 403
    assert "TRUSTED_HOSTS" in r.json()["detail"]
    assert stub.calls == [], "a rebound request reached a backend"
    # holding the key does not help a page that is not on an allowed host either
    r = client.delete(DELETE_JOB, headers={**REBOUND, **AUTH})
    assert r.status_code == 403
    assert stub.calls == []


def test_a_rebound_page_cannot_read_the_api_either(monkeypatch):
    """After rebinding the attacker's page is same-origin with the service, so a GET is
    readable as well: the UI, the registry (internal URLs), job lists and downloads."""
    _, stub, client = _gateway(monkeypatch)
    for path in ("/", "/providers", "/health", "/api/training/jobs", "/api/health"):
        r = client.get(path, headers={"Host": "evil.example:3000"})
        assert r.status_code == 403, path
    assert stub.calls == []


def test_the_host_refusal_uses_the_openai_envelope_on_v1(monkeypatch):
    _, _, client = _gateway(monkeypatch)
    r = client.get("/v1/models", headers={"Host": "evil.example"})
    assert r.status_code == 403
    error = r.json()["error"]
    assert error["code"] == "host_not_allowed"
    assert error["type"] == "permission_error"


@pytest.mark.parametrize("host", [
    "localhost:3000", "LOCALHOST", "127.0.0.1:3000", "127.0.0.1", "[::1]:3000", "[fe80::1]:3000",
    "192.168.1.20:3000", "10.0.0.5", "172.16.9.9:3000", "169.254.10.10",
    "100.64.1.2:3000",                                   # CGNAT / Tailscale address
    "8.8.8.8:3000",                                      # any IP literal: nothing to rebind
    "truenas.local:3000", "TrueNAS.Local", "nas.lan", "router.home.arpa", "nas.internal",
    "app.localhost:3000",
    "truenas", "frontend-service:3000", "testserver", "nas.local.",
])
def test_lan_names_and_ip_literals_work_without_configuration(monkeypatch, host):
    _, stub, client = _gateway(monkeypatch)
    assert client.get("/providers", headers={"Host": host}).status_code == 200, host
    # the same host may also make state-changing calls from a page it served
    assert client.delete(DELETE_JOB, headers={"Host": host, "Origin": f"http://{host}"}).status_code == 200, host
    assert stub.calls


@pytest.mark.parametrize("host", [
    "evil.example", "evil.example:3000", "EVIL.example",
    "evil.example.",                                     # trailing dot is still that name
    "192.168.1.20.evil.example:3000",                    # looks like a LAN address, is a name
    "127.0.0.1.nip.io", "localhost.evil.example", "truenas.local.evil.example",
    "nas.lan.evil.example", "my.home.arpa.evil.example",
    "example.com", "tts.example.com",
    # not a host[:port] at all: userinfo, paths, backslashes, whitespace, junk
    "evil.example@127.0.0.1", "127.0.0.1@evil.example", "evil.example\\@127.0.0.1",
    "evil.example/127.0.0.1", "127.0.0.1#@evil.example", "a b", "evil.example:3000:1",
    "evil.example:abc", "[::1", "::1",
])
def test_public_names_and_malformed_hosts_are_refused(monkeypatch, host):
    _, stub, client = _gateway(monkeypatch)
    r = client.get("/providers", headers={"Host": host})
    assert r.status_code == 403, host
    assert client.delete(DELETE_JOB, headers={"Host": host, "Origin": f"http://{host}"}).status_code == 403
    assert stub.calls == []


def test_a_request_with_no_host_header_at_all_is_not_a_rebinding_browser(monkeypatch):
    """HTTP/1.0 clients omit Host; a browser never does."""
    import asyncio
    from frontend_loader import asgi_call

    module, stub, _ = _gateway(monkeypatch)
    status, _, _ = asyncio.run(asgi_call(module.app, "GET", "/providers"))
    assert status == 200
    status, _, _ = asyncio.run(asgi_call(module.app, "DELETE", DELETE_JOB))
    assert status == 200


def test_trusted_hosts_names_a_domain_in_front_of_a_reverse_proxy(monkeypatch):
    headers = {"Host": "tts.example.com", "Origin": "https://tts.example.com"}
    _, stub, client = _gateway(monkeypatch)
    assert client.delete(DELETE_JOB, headers=headers).status_code == 403

    _, stub, client = _gateway(monkeypatch, {"TRUSTED_HOSTS": "tts.example.com"})
    assert client.delete(DELETE_JOB, headers=headers).status_code == 200
    assert client.get("/", headers={"Host": "tts.example.com:8443"}).status_code == 200   # any port
    # exact names only: a neighbour or a parent is not the listed one
    for host in ("evil.tts.example.com", "example.com", "tts.example.com.evil.example"):
        assert client.get("/providers", headers={"Host": host}).status_code == 403, host


def test_trusted_hosts_accepts_wildcards_ports_schemes_and_case(monkeypatch):
    env = {"TRUSTED_HOSTS": " *.tail1234.ts.net , HTTPS://Voice.Example.org:8443/ , nas.example.net:3000 "}
    _, _, client = _gateway(monkeypatch, env)
    for host in ("truenas.tail1234.ts.net", "a.b.tail1234.ts.net:3000", "voice.example.org",
                 "NAS.example.net"):
        assert client.get("/providers", headers={"Host": host}).status_code == 200, host
    for host in ("tail1234.ts.net",                      # a wildcard means subdomains
                 "truenas.tail1234.ts.net.evil.example", "evil-tail1234.ts.net",
                 "other.example.org"):
        assert client.get("/providers", headers={"Host": host}).status_code == 403, host


def test_a_trusted_origin_also_makes_its_hostname_a_trusted_host(monkeypatch):
    """TRUSTED_ORIGINS says "this UI is also reached as https://tts.example.com"."""
    _, _, client = _gateway(monkeypatch, {"TRUSTED_ORIGINS": "https://tts.example.com"})
    headers = {"Host": "tts.example.com", "Origin": "https://tts.example.com"}
    assert client.delete(DELETE_JOB, headers=headers).status_code == 200


def test_forged_forwarded_host_is_ignored_without_proxy_trust(monkeypatch):
    """X-Forwarded-Host is client-controlled unless a proxy we run overwrites it."""
    _, _, client = _gateway(monkeypatch)
    r = client.get("/providers", headers={"Host": "truenas.local", "X-Forwarded-Host": "evil.example"})
    assert r.status_code == 200


def test_with_proxy_trust_the_forwarded_host_is_what_gets_validated(monkeypatch):
    """A rebound page reaches the service through the proxy, which tells it the name."""
    _, stub, client = _gateway(monkeypatch, {"TRUST_PROXY_HEADERS": "true"})
    r = client.delete(DELETE_JOB, headers={
        "Host": "frontend-service:3000", "X-Forwarded-Host": "evil.example",
        "Origin": "https://evil.example"})
    assert r.status_code == 403
    assert stub.calls == []
    # a proxy that forwards a LAN name is fine without any list
    r = client.get("/providers", headers={"Host": "frontend-service:3000", "X-Forwarded-Host": "nas.local"})
    assert r.status_code == 200


def test_allowed_hosts_star_switches_the_check_off_and_says_so(monkeypatch, caplog):
    import logging
    with caplog.at_level(logging.WARNING):
        _, stub, client = _gateway(monkeypatch, {"ALLOWED_HOSTS": "*"})
    assert any("ALLOWED_HOSTS='*'" in r.getMessage() for r in caplog.records)
    assert client.delete(DELETE_JOB, headers=REBOUND).status_code == 200


def test_allowed_hosts_is_not_a_list_and_a_name_in_it_opens_nothing(monkeypatch, caplog):
    import logging
    with caplog.at_level(logging.WARNING):
        _, _, client = _gateway(monkeypatch, {"ALLOWED_HOSTS": "evil.example"})
    assert any("TRUSTED_HOSTS" in r.getMessage() for r in caplog.records)
    assert client.get("/providers", headers={"Host": "evil.example"}).status_code == 403


def test_host_check_leaves_preflights_alone(monkeypatch):
    """A preflight carries no credentials and changes nothing; the real request after it is checked."""
    _, _, client = _gateway(monkeypatch)
    r = client.options(DELETE_JOB, headers={"Host": "evil.example"})
    assert r.status_code != 403


# --- /api/auth/check: how the UI learns whether it needs a key --------------------


def test_auth_check_is_204_when_no_key_is_configured(monkeypatch):
    _, stub, client = _gateway(monkeypatch)
    assert client.post("/api/auth/check").status_code == 204
    assert stub.calls == [], "the check must not touch a backend"


def test_auth_check_challenges_until_the_key_is_sent(monkeypatch):
    _, _, client = _gateway(monkeypatch, {"API_KEY": KEY})
    r = client.post("/api/auth/check")
    assert r.status_code == 401
    assert r.headers["www-authenticate"] == "Bearer"
    assert client.post("/api/auth/check", headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert client.post("/api/auth/check", headers=AUTH).status_code == 204


# --- the WebSocket relay and the API key ----------------------------------------------
#
# The request guard only sees HTTP scopes, so /ws/stt let anyone stream audio through
# the relay whatever API_KEY said. Browsers cannot set headers on a WebSocket, so the
# key also travels as a `bearer.<base64url>` subprotocol next to the real one.


def _bearer_protocol(key):
    import base64
    return "bearer." + base64.urlsafe_b64encode(key.encode()).decode().rstrip("=")


class _SilentUpstream:
    """A stand-in for the STT service's socket that says nothing and hangs up."""

    def __init__(self, seen, url, kwargs):
        seen.append((url, kwargs))

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def send(self, message):
        pass

    def __aiter__(self):
        return self

    async def __anext__(self):
        raise StopAsyncIteration


def _fake_upstream(monkeypatch):
    import websockets
    seen = []
    monkeypatch.setattr(websockets, "connect", lambda url, **kw: _SilentUpstream(seen, url, kw))
    return seen


def _ws_outcome(client, path="/ws/stt?provider=whisper", **kwargs):
    """('accepted', subprotocol) when the relay accepted, else ('closed', code, reason)."""
    try:
        with client.websocket_connect(path, **kwargs) as ws:
            protocol = ws.accepted_subprotocol
            try:
                ws.receive_text()
            except WebSocketDisconnect as closed:
                if closed.code == 1008:
                    return ("closed", closed.code, closed.reason)
            return ("accepted", protocol)
    except WebSocketDisconnect as closed:
        return ("closed", closed.code, closed.reason)


def test_websocket_without_the_key_is_refused_with_a_reason(monkeypatch):
    seen = _fake_upstream(monkeypatch)
    _, _, client = _gateway(monkeypatch, {"API_KEY": KEY})
    assert _ws_outcome(client) == ("closed", 1008, "A valid API key is required")
    assert seen == [], "an unauthenticated socket was relayed to the STT service"


def test_websocket_with_a_wrong_key_is_refused(monkeypatch):
    seen = _fake_upstream(monkeypatch)
    _, _, client = _gateway(monkeypatch, {"API_KEY": KEY})
    outcome = _ws_outcome(client, subprotocols=["tts-stt.v1", _bearer_protocol("nope")])
    assert outcome == ("closed", 1008, "A valid API key is required")
    outcome = _ws_outcome(client, headers={"Authorization": "Bearer nope"})
    assert outcome[:2] == ("closed", 1008)
    assert seen == []


def test_websocket_with_the_key_as_a_subprotocol_is_relayed_without_echoing_it(monkeypatch):
    """The browser way. Echoing the credential back would put it in the response headers."""
    seen = _fake_upstream(monkeypatch)
    _, _, client = _gateway(monkeypatch, {"API_KEY": KEY})
    outcome = _ws_outcome(client, subprotocols=["tts-stt.v1", _bearer_protocol(KEY)])
    assert outcome == ("accepted", "tts-stt.v1")
    assert len(seen) == 1
    # the subprotocol order does not matter
    assert _ws_outcome(client, subprotocols=[_bearer_protocol(KEY), "tts-stt.v1"]) == ("accepted", "tts-stt.v1")


def test_websocket_with_the_key_in_the_authorization_header_is_relayed(monkeypatch):
    """The script way (a non-browser client can set headers)."""
    seen = _fake_upstream(monkeypatch)
    _, _, client = _gateway(monkeypatch, {"API_KEY": KEY})
    assert _ws_outcome(client, headers=AUTH) == ("accepted", None)
    assert len(seen) == 1


def test_websocket_key_may_contain_any_character(monkeypatch):
    """base64url keeps a key with spaces, commas or non-ASCII inside the subprotocol token alphabet."""
    seen = _fake_upstream(monkeypatch)
    odd = "pässwörd, with spaces/and=signs+more"
    _, _, client = _gateway(monkeypatch, {"API_KEY": odd})
    assert _ws_outcome(client, subprotocols=["tts-stt.v1", _bearer_protocol(odd)]) == ("accepted", "tts-stt.v1")
    assert len(seen) == 1


def test_websocket_offering_only_a_key_gets_no_credential_echoed_back(monkeypatch):
    _fake_upstream(monkeypatch)
    _, _, client = _gateway(monkeypatch, {"API_KEY": KEY})
    assert _ws_outcome(client, subprotocols=[_bearer_protocol(KEY)]) == ("accepted", None)


def test_websocket_without_an_api_key_configured_still_echoes_the_real_protocol(monkeypatch):
    """A UI that still holds a key from an earlier session offers protocols to a server with none;
    a browser aborts the handshake when it offers some and none is selected."""
    seen = _fake_upstream(monkeypatch)
    _, _, client = _gateway(monkeypatch)
    assert _ws_outcome(client, subprotocols=["tts-stt.v1", _bearer_protocol("stale")]) == ("accepted", "tts-stt.v1")
    assert len(seen) == 1
    assert _ws_outcome(client) == ("accepted", None)


def test_websocket_on_a_rebound_host_is_refused_even_with_a_matching_origin(monkeypatch):
    seen = _fake_upstream(monkeypatch)
    _, _, client = _gateway(monkeypatch)
    outcome = _ws_outcome(client, headers={"Host": "evil.example:3000", "Origin": "http://evil.example:3000"})
    assert outcome == ("closed", 1008, "")
    assert seen == []


def test_websocket_relay_failure_does_not_name_the_internal_url(monkeypatch):
    """The browser gets a generic frame; the URL and the error are for the log."""
    import websockets

    def refuse(url, **kwargs):
        raise OSError(f"connect call failed ('10.1.2.3', 8000) while dialling {url}")

    monkeypatch.setattr(websockets, "connect", refuse)
    _, _, client = _gateway(monkeypatch)
    with client.websocket_connect("/ws/stt?provider=whisper") as ws:
        frame = ws.receive_json()
    assert frame["type"] == "error"
    assert "stt-service" not in frame["error"] and "10.1.2.3" not in frame["error"]
    assert "://" not in frame["error"]


# --- what the relay sends upstream ---------------------------------------------------------
#
# The gateway validates the browser's Origin against its own Host. The backends now do
# the same with theirs, so forwarding the browser's Origin would make the STT service
# compare `https://tts.example.com` with `Host: stt-service:8000` and refuse the
# handshake. The relay dials it like any other server-to-server call: no Origin.


def _upstream_that_records_handshakes():
    """A real WebSocket server on localhost that records each handshake's headers and hangs up."""
    import asyncio
    import threading

    from websockets.asyncio.server import serve

    handshakes, ready, state = [], threading.Event(), {}

    async def handler(connection):
        handshakes.append({k.lower(): v for k, v in connection.request.headers.raw_items()})
        await connection.close()

    async def main():
        async with serve(handler, "127.0.0.1", 0) as server:
            state["port"] = server.sockets[0].getsockname()[1]
            state["stop"] = asyncio.Event()
            state["loop"] = asyncio.get_running_loop()
            ready.set()
            await state["stop"].wait()

    thread = threading.Thread(target=lambda: asyncio.run(main()), daemon=True)
    thread.start()
    assert ready.wait(10), "the recording WebSocket server did not start"

    def stop():
        state["loop"].call_soon_threadsafe(state["stop"].set)
        thread.join(timeout=10)

    return handshakes, state["port"], stop


@pytest.mark.parametrize("client_headers", [
    {"Origin": "http://testserver"},
    {},
], ids=["from-a-browser", "from-a-script"])
def test_the_relay_does_not_forward_the_browsers_origin_to_the_stt_service(monkeypatch, client_headers):
    handshakes, port, stop = _upstream_that_records_handshakes()
    try:
        _, _, client = _gateway(monkeypatch, {"STT_SERVICE_URL": f"http://127.0.0.1:{port}"})
        with client.websocket_connect("/ws/stt?provider=whisper", headers=client_headers) as ws:
            try:
                ws.receive_text()
            except WebSocketDisconnect:
                pass
    finally:
        stop()
    assert len(handshakes) == 1, "the relay never reached the STT service"
    assert "origin" not in handshakes[0], f"an Origin header was sent upstream: {handshakes[0]}"
    assert handshakes[0]["host"].startswith("127.0.0.1:")
