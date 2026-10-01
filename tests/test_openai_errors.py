"""Error shapes of the /v1 surface, and the STT default-provider fallback.

docs/api.md promises that every /v1 error uses the OpenAI envelope, because
clients branch on `error.type`/`error.code`/`error.param`. That held for the
errors the router raises itself and failed for everything the framework raises:
a missing `file` or `temperature=abc` answered FastAPI's 422 `{"detail": [...]}`,
an unknown route `{"detail": "Not Found"}`, a crash a bare text body. The
browser-facing /api errors must stay exactly as they were.

The fallback half covers the whisper-cpp-only devices (rk3588, strixhalo): the
default STT provider is `whisper`, which does not run there, so
/v1/audio/transcriptions answered 502 although a working backend was up.
"""

import logging

import httpx
import pytest
from fastapi.testclient import TestClient

from frontend_loader import install_stub, load_frontend_app, wav_bytes

WAV = wav_bytes()


def _envelope_ok(body):
    assert set(body) == {"error"}, f"not an OpenAI envelope: {body}"
    assert set(body["error"]) == {"message", "type", "param", "code"}
    return body["error"]


def _client(monkeypatch, env=None, handler=None, **kwargs):
    module = load_frontend_app(env)
    stub = install_stub(monkeypatch, module, handler or (lambda m, u, k: {"text": "hallo"}))
    return module, stub, TestClient(module.app, **kwargs)


# --- framework errors on /v1 ---------------------------------------------------


def test_missing_file_is_an_envelope_naming_the_parameter(monkeypatch):
    _, stub, client = _client(monkeypatch)
    r = client.post("/v1/audio/transcriptions", data={"model": "whisper-1"})
    assert r.status_code == 400
    error = _envelope_ok(r.json())
    assert error["type"] == "invalid_request_error"
    assert error["param"] == "file"
    assert error["code"] == "missing_required_parameter"
    assert stub.calls == []


def test_unparseable_field_is_an_envelope_naming_the_parameter(monkeypatch):
    _, _, client = _client(monkeypatch)
    r = client.post("/v1/audio/transcriptions", data={"temperature": "abc"},
                    files={"file": ("a.wav", WAV, "audio/wav")})
    assert r.status_code == 400
    error = _envelope_ok(r.json())
    assert error["param"] == "temperature"
    assert error["code"] == "invalid_value"


def test_unknown_v1_route_is_an_envelope(monkeypatch):
    _, _, client = _client(monkeypatch)
    for path in ("/v1/nope", "/v1/audio/nope", "/v1"):
        r = client.get(path)
        assert r.status_code == 404, path
        error = _envelope_ok(r.json())
        assert error["type"] == "invalid_request_error"
        assert error["message"] == f"Invalid URL (GET {path})"


def test_wrong_method_is_an_envelope_and_keeps_allow(monkeypatch):
    _, _, client = _client(monkeypatch)
    r = client.get("/v1/audio/speech")
    assert r.status_code == 405
    _envelope_ok(r.json())
    assert "POST" in r.headers["allow"]


def test_a_crash_on_v1_is_an_envelope(monkeypatch):
    def boom(method, url, kwargs):
        raise RuntimeError("backend exploded")

    _, _, client = _client(monkeypatch, handler=boom, raise_server_exceptions=False)
    r = client.post("/v1/audio/speech", json={"input": "hi", "response_format": "wav"})
    assert r.status_code == 500
    error = _envelope_ok(r.json())
    assert error["type"] == "server_error"
    assert "backend exploded" not in r.text, "internal details must not leak to the caller"


# --- /api is unchanged ---------------------------------------------------------


def test_api_errors_keep_the_detail_shape(monkeypatch):
    _, _, client = _client(monkeypatch)

    not_found = client.get("/api/nope")
    assert not_found.status_code == 404
    assert not_found.json() == {"detail": "Not Found"}

    invalid = client.post("/api/tts", json={"provider": "piper"})
    assert invalid.status_code == 422
    assert isinstance(invalid.json()["detail"], list)

    missing = client.post("/api/stt", data={"provider": "whisper"})
    assert missing.status_code == 400
    assert missing.json() == {"detail": "Audio file not provided"}


def test_paths_that_only_start_with_v1_are_not_v1(monkeypatch):
    _, _, client = _client(monkeypatch)
    for path in ("/v1beta/x", "/v10"):
        assert client.get(path).json() == {"detail": "Not Found"}, path


def test_a_crash_on_api_keeps_the_plain_500(monkeypatch):
    def boom(method, url, kwargs):
        raise RuntimeError("backend exploded")

    _, _, client = _client(monkeypatch, handler=boom, raise_server_exceptions=False)
    r = client.post("/api/tts", json={"provider": "piper", "text": "hi"})
    assert r.status_code == 500
    assert r.text == "Internal Server Error"


# --- default STT provider fallback ---------------------------------------------


def _fleet(healthy_hosts, *, transcribe_status=200):
    """Backends where only `healthy_hosts` answer; the rest refuse connections."""
    def handler(method, url, kwargs):
        host = httpx.URL(url).host
        if host not in healthy_hosts:
            raise httpx.ConnectError("connection refused", request=httpx.Request(method, url))
        if method == "GET":
            return {"status": "healthy"}
        if transcribe_status != 200:
            return httpx.Response(transcribe_status, json={"detail": "model exploded"})
        if url.endswith("/inference"):
            return {"text": f"from {host}"}
        return {"text": f"from {host}", "language": "de", "duration": 2.0,
                "segments": [{"start": 0.0, "end": 2.0, "text": f"from {host}"}]}
    return handler


def _transcribe(client, **data):
    return client.post("/v1/audio/transcriptions", data={"model": "whisper-1", **data},
                       files={"file": ("a.wav", WAV, "audio/wav")})


def _posted_hosts(stub):
    return [httpx.URL(url).host for method, url, _ in stub.calls if method == "POST"]


def test_unreachable_default_falls_back_to_the_single_healthy_provider(monkeypatch, caplog):
    handler = _fleet({"qwen3-asr-service"})
    _, stub, client = _client(monkeypatch, handler=handler)
    with caplog.at_level(logging.WARNING):
        r = _transcribe(client)
    assert r.status_code == 200
    assert r.json() == {"text": "from qwen3-asr-service"}
    assert r.headers["x-provider-fallback"] == "whisper->qwen3-asr"
    assert r.headers["x-provider"] == "qwen3-asr"
    assert _posted_hosts(stub) == ["stt-service", "qwen3-asr-service"]
    assert any("falling back" in rec.getMessage() and "qwen3-asr" in rec.getMessage()
               for rec in caplog.records), "the swap must be visible in the log"


def test_fallback_speaks_the_fallback_providers_own_contract(monkeypatch):
    """whisper-cpp takes `file` + an explicit language=auto, not the native form."""
    handler = _fleet({"whisper-cpp"})
    _, stub, client = _client(monkeypatch, {"ENABLE_WHISPER_CPP": "true"}, handler=handler)
    r = _transcribe(client)
    assert r.status_code == 200
    assert r.headers["x-provider-fallback"] == "whisper->whisper-cpp"
    method, url, kwargs = stub.calls[-1]
    assert url.endswith("whisper-cpp:8080/inference")
    assert kwargs["data"]["language"] == "auto"
    assert kwargs["files"][0][0] == "file"


def test_fallback_serves_the_requested_format(monkeypatch):
    _, _, client = _client(monkeypatch, handler=_fleet({"qwen3-asr-service"}))
    r = _transcribe(client, response_format="srt")
    assert r.status_code == 200
    assert r.text == "1\n00:00:00,000 --> 00:00:02,000\nfrom qwen3-asr-service\n"
    assert r.headers["x-provider-fallback"] == "whisper->qwen3-asr"


def test_no_fallback_when_the_default_is_healthy(monkeypatch):
    _, stub, client = _client(monkeypatch, handler=_fleet({"stt-service", "qwen3-asr-service"}))
    r = _transcribe(client)
    assert r.status_code == 200
    assert "x-provider-fallback" not in r.headers
    assert _posted_hosts(stub) == ["stt-service"]


def test_no_fallback_when_several_alternatives_are_healthy(monkeypatch):
    """Choosing between models changes accuracy and language coverage; that is the
    operator's call, so with two candidates the request fails as before."""
    handler = _fleet({"qwen3-asr-service", "parakeet-asr-service"})
    _, stub, client = _client(monkeypatch, {"ENABLE_PARAKEET_ASR": "true"}, handler=handler)
    r = _transcribe(client)
    assert r.status_code == 502
    _envelope_ok(r.json())
    assert _posted_hosts(stub) == ["stt-service"]


def test_no_fallback_when_nothing_is_healthy(monkeypatch):
    _, stub, client = _client(monkeypatch, handler=_fleet(set()))
    r = _transcribe(client)
    assert r.status_code == 502
    assert "x-provider-fallback" not in r.headers
    assert _posted_hosts(stub) == ["stt-service"]


def test_no_fallback_when_the_default_answered_with_an_error(monkeypatch, caplog):
    """A backend that is up and says the request failed would say the same thing
    to any other provider; only 'could not be reached' justifies a swap."""
    handler = _fleet({"stt-service", "qwen3-asr-service"}, transcribe_status=500)
    _, stub, client = _client(monkeypatch, handler=handler)
    with caplog.at_level(logging.WARNING):
        r = _transcribe(client)
    assert r.status_code == 502
    message = r.json()["error"]["message"]
    # what the backend said is for the operator; the caller gets a request id that finds it
    assert "model exploded" not in message
    request_id = message.rsplit("Request id: ", 1)[1].rstrip(".")
    assert any(request_id in rec.getMessage() and "model exploded" in rec.getMessage()
               for rec in caplog.records)
    assert _posted_hosts(stub) == ["stt-service"]


def test_no_fallback_when_the_default_accepted_the_audio_and_timed_out(monkeypatch):
    """A read timeout means a backend is mid-transcription. Sending the same audio
    to another provider would run the job twice."""
    def handler(method, url, kwargs):
        host = httpx.URL(url).host
        if method == "GET":
            return {"status": "healthy"}
        if host == "stt-service":
            raise httpx.ReadTimeout("timed out", request=httpx.Request(method, url))
        return {"text": "from elsewhere"}

    _, stub, client = _client(monkeypatch, handler=handler)
    r = _transcribe(client)
    assert r.status_code == 502
    assert "x-provider-fallback" not in r.headers
    assert _posted_hosts(stub) == ["stt-service"]


def test_the_fallback_target_failing_too_is_reported(monkeypatch):
    def handler(method, url, kwargs):
        host = httpx.URL(url).host
        if method == "GET":
            return {"status": "healthy"} if host == "qwen3-asr-service" else (_ for _ in ()).throw(
                httpx.ConnectError("refused", request=httpx.Request(method, url)))
        raise httpx.ConnectError("refused", request=httpx.Request(method, url))

    _, _, client = _client(monkeypatch, handler=handler)
    r = _transcribe(client)
    assert r.status_code == 502
    _envelope_ok(r.json())
    assert r.headers["x-provider-fallback"] == "whisper->qwen3-asr"


def _stand_in_answering(status, detail, headers=None):
    """The default (whisper) cannot be reached; qwen3-asr, the one healthy stand-in, answers `status`."""
    def handler(method, url, kwargs):
        host = httpx.URL(url).host
        if host != "qwen3-asr-service":
            raise httpx.ConnectError("connection refused", request=httpx.Request(method, url))
        if method == "GET":
            return {"status": "healthy"}
        return httpx.Response(status, json={"detail": detail}, headers=headers or {})
    return handler


@pytest.mark.parametrize("status", [400, 413, 422])
def test_a_stand_ins_refusal_is_a_502_that_names_it_not_the_callers_error(monkeypatch, status):
    """The stand-in's limits (a language it cannot do, a shorter audio cap) are not a verdict on the request.

    The default it stood in for was never asked and may well take it once it is back, so
    the answer is the 502 the SDKs retry, as with no stand-in at all, not a 400/413 they
    give up on; and it says which provider refused, and that it was a stand-in.
    """
    detail = "Language 'it' is not supported by this model. Supported: de, en, es, fr."
    _, stub, client = _client(monkeypatch, handler=_stand_in_answering(status, detail))

    r = _transcribe(client, language="it")

    assert r.status_code == 502
    error = _envelope_ok(r.json())
    assert (error["type"], error["code"]) == ("server_error", None)
    assert error["message"] == (
        f"Transcription fallback 'qwen3-asr' (the default 'whisper' is unreachable) rejected the request: {detail}")
    assert r.headers["x-provider"] == "qwen3-asr"
    assert r.headers["x-provider-fallback"] == "whisper->qwen3-asr"
    assert _posted_hosts(stub) == ["stt-service", "qwen3-asr-service"]


def test_a_busy_stand_in_is_a_503_the_client_can_wait_out_and_says_it_stood_in(monkeypatch):
    busy = "The service is busy: 5 requests are already in progress or waiting. Retry shortly."
    _, _, client = _client(monkeypatch, handler=_stand_in_answering(503, busy, {"Retry-After": "5"}))

    r = _transcribe(client)

    assert r.status_code == 503 and r.headers["retry-after"] == "5"
    error = _envelope_ok(r.json())
    assert (error["type"], error["code"]) == ("server_error", "server_busy")
    assert r.headers["x-provider"] == "qwen3-asr"
    assert r.headers["x-provider-fallback"] == "whisper->qwen3-asr"


def test_explicit_provider_selection_stays_strict(monkeypatch):
    """/api/stt names its provider; silently answering from another model would
    misreport which one produced the text."""
    handler = _fleet({"qwen3-asr-service"})
    _, stub, client = _client(monkeypatch, handler=handler)
    r = client.post("/api/stt", data={"provider": "whisper"},
                    files={"audio": ("a.wav", WAV, "audio/wav")})
    assert r.status_code == 503
    assert "x-provider-fallback" not in r.headers
    assert _posted_hosts(stub) == ["stt-service"]


def test_a_configured_default_that_is_up_is_used_as_configured(monkeypatch):
    handler = _fleet({"whisper-cpp", "qwen3-asr-service"})
    _, stub, client = _client(
        monkeypatch, {"ENABLE_WHISPER_CPP": "true", "DEFAULT_STT_PROVIDER": "whisper-cpp"},
        handler=handler)
    r = _transcribe(client)
    assert r.status_code == 200 and "x-provider-fallback" not in r.headers
    assert _posted_hosts(stub) == ["whisper-cpp"]
