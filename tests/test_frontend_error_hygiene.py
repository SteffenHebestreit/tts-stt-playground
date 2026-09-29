"""What a client is told when a backend fails, and what stays in the log.

A backend's error body is written for its operator: a 500 carries `str(exception)`
(paths under /app/models, a whole traceback) and a connection failure names the
internal URL. The gateway used to hand both to every caller - `/v1/audio/speech`
echoed the upstream body verbatim inside its envelope, and every connect error
said `<service> is unavailable: http://piper-tts-service:5000/tts` - which is the
deployment layout for free.

Now a client gets a status-appropriate sentence and a request id, and the detail
is logged under that id. A 4xx that is one short line of prose is different: it is
how a backend explains what is wrong with the request, the UI shows it, and it
still passes through. `/health` and `/providers` keep showing provider URLs on
purpose.
"""

import logging
import re

import httpx
import pytest
from fastapi.testclient import TestClient

from frontend_loader import install_stub, load_frontend_app, wav_bytes

TRACEBACK = (
    'Traceback (most recent call last):\n  File "/app/app.py", line 88, in synth\n'
    "    model = load('/app/models/de_DE-thorsten-medium.onnx')\n"
    "FileNotFoundError: [Errno 2] No such file or directory: '/app/models/de_DE-thorsten-medium.onnx'"
)
LEAKS = ("Traceback", "/app/", "thorsten", "FileNotFoundError", "piper-tts-service", "qwen3-tts-service",
         "stt-service:8000", "://")


def _gateway(monkeypatch, handler, env=None):
    module = load_frontend_app(env)
    stub = install_stub(monkeypatch, module, handler)
    return module, stub, TestClient(module.app, raise_server_exceptions=False)


def _refuse(method, url, kwargs):
    raise httpx.ConnectError("[Errno 111] Connection refused", request=httpx.Request(method, url))


def _status(code, body):
    def handler(method, url, kwargs):
        if isinstance(body, (dict, list)):
            return httpx.Response(code, json=body)
        return httpx.Response(code, content=body, headers={"content-type": "text/html"})
    return handler


def _request_id(text):
    found = re.search(r"Request id: ([0-9a-f]{12})", text)
    assert found, f"no request id in {text!r}"
    return found.group(1)


def _assert_clean(text):
    for leak in LEAKS:
        assert leak not in text, f"{leak!r} reached the client: {text!r}"


def _logged(caplog, request_id, needle):
    return any(request_id in r.getMessage() and needle in r.getMessage() for r in caplog.records)


# --- /v1/audio/speech: the route the review reproduced -----------------------------


def test_speech_does_not_echo_the_upstream_error_body(monkeypatch, caplog):
    _, _, client = _gateway(monkeypatch, _status(500, {"detail": TRACEBACK}))
    with caplog.at_level(logging.WARNING):
        r = client.post("/v1/audio/speech", json={"input": "hello"})
    assert r.status_code == 502
    error = r.json()["error"]
    assert set(error) == {"message", "type", "param", "code"}
    _assert_clean(error["message"])
    assert "Speech backend failed" in error["message"]
    assert _logged(caplog, _request_id(error["message"]), "de_DE-thorsten-medium.onnx"), (
        "the traceback must be in the server log under the id the client was given")


@pytest.mark.parametrize("body,content_type", [
    (TRACEBACK.encode(), "text/plain"),
    (b"<html><body><h1>500</h1>/app/models/x</body></html>", "text/html"),
    (b"", "text/plain"),
])
def test_speech_with_a_non_json_error_body_is_generic_too(monkeypatch, body, content_type):
    def handler(method, url, kwargs):
        return httpx.Response(500, content=body, headers={"content-type": content_type})

    _, _, client = _gateway(monkeypatch, handler)
    message = client.post("/v1/audio/speech", json={"input": "hello"}).json()["error"]["message"]
    _assert_clean(message)
    assert "<html" not in message
    _request_id(message)


def test_speech_connect_error_names_the_service_not_its_url(monkeypatch, caplog):
    _, _, client = _gateway(monkeypatch, _refuse)
    with caplog.at_level(logging.WARNING):
        r = client.post("/v1/audio/speech", json={"input": "hello"})
    assert r.status_code == 502
    message = r.json()["error"]["message"]
    _assert_clean(message)
    assert "PiperTTS" in message, "the caller should still learn which service is down"
    assert _logged(caplog, _request_id(message), "http://piper-tts-service:5000/tts")


def test_transcription_errors_are_generic_on_v1(monkeypatch, caplog):
    files = {"file": ("a.wav", wav_bytes(), "audio/wav")}
    _, _, client = _gateway(monkeypatch, _status(500, {"detail": TRACEBACK}))
    message = client.post("/v1/audio/transcriptions", files=files).json()["error"]["message"]
    _assert_clean(message)
    _request_id(message)

    _, _, client = _gateway(monkeypatch, _refuse)
    r = client.post("/v1/audio/transcriptions", files=files)
    assert r.status_code == 502
    _assert_clean(r.json()["error"]["message"])


# --- /api/*: same rule, the shape the UI reads ----------------------------------------


@pytest.mark.parametrize("call", [
    lambda c: c.post("/api/tts", json={"provider": "piper", "text": "hi"}),
    lambda c: c.post("/api/stt", data={"provider": "whisper"},
                     files={"audio": ("a.wav", wav_bytes(), "audio/wav")}),
    lambda c: c.get("/api/training/jobs"),
    lambda c: c.delete("/api/training/model/job-1"),
    lambda c: c.get("/api/providers/qwen3/models"),
    lambda c: c.post("/api/providers/qwen3/voice-clone",
                     data={"text": "hi"}, files={"voice_file": ("v.wav", wav_bytes(), "audio/wav")}),
    lambda c: c.post("/api/providers/qwen3/unload"),
], ids=["tts", "stt", "training-jobs", "delete-model", "models", "voice-clone", "unload"])
def test_api_connect_errors_are_a_503_without_the_internal_url(monkeypatch, caplog, call):
    _, _, client = _gateway(monkeypatch, _refuse)
    with caplog.at_level(logging.WARNING):
        r = call(client)
    assert r.status_code == 503
    detail = r.json()["detail"]
    _assert_clean(detail)
    assert "unavailable" in detail
    assert _logged(caplog, _request_id(detail), "Connection refused")


@pytest.mark.parametrize("call", [
    lambda c: c.post("/api/tts", json={"provider": "piper", "text": "hi"}),
    lambda c: c.post("/api/stt", data={"provider": "whisper"},
                     files={"audio": ("a.wav", wav_bytes(), "audio/wav")}),
    lambda c: c.get("/api/training/jobs"),
    lambda c: c.get("/api/providers/qwen3/saved-voices"),
], ids=["tts", "stt", "training-jobs", "saved-voices"])
def test_api_5xx_keeps_the_status_but_not_the_body(monkeypatch, caplog, call):
    _, _, client = _gateway(monkeypatch, _status(500, {"detail": TRACEBACK}))
    with caplog.at_level(logging.WARNING):
        r = call(client)
    assert r.status_code == 500
    detail = r.json()["detail"]
    _assert_clean(detail)
    assert "HTTP 500" in detail
    assert _logged(caplog, _request_id(detail), "FileNotFoundError")


def test_streamed_tts_errors_are_generic_as_well(monkeypatch, caplog):
    """The chatterbox route streams; its error path reads the body separately."""
    _, _, client = _gateway(monkeypatch, _status(500, {"detail": TRACEBACK}), {"ENABLE_CHATTERBOX_TTS": "true"})
    with caplog.at_level(logging.WARNING):
        r = client.post("/api/tts", json={"provider": "chatterbox", "text": "hi"})
    assert r.status_code == 500
    _assert_clean(r.json()["detail"])
    assert _logged(caplog, _request_id(r.json()["detail"]), "FileNotFoundError")


def test_a_read_timeout_is_a_503_without_the_url_too(monkeypatch):
    def slow(method, url, kwargs):
        raise httpx.ReadTimeout("timed out", request=httpx.Request(method, url))

    _, _, client = _gateway(monkeypatch, slow)
    r = client.post("/api/tts", json={"provider": "piper", "text": "hi"})
    assert r.status_code == 503
    _assert_clean(r.json()["detail"])


# --- 4xx: the backend's explanation of a bad request still reaches the caller -----------


@pytest.mark.parametrize("status,detail", [
    (400, "Language 'xx' is not supported by Qwen3-TTS. Supported: German, English, French."),
    (413, "text is 5001 characters; the limit is 5000 (MAX_TEXT_CHARS)."),
    (404, "Voice 'anna' not found"),
    (409, "Model is in use; retry when idle"),
    (422, "Unsupported language 'xx'. Use a Whisper language code such as 'de' or 'en', or 'auto'."),
])
def test_a_short_plain_4xx_explanation_passes_through_unchanged(monkeypatch, status, detail):
    _, _, client = _gateway(monkeypatch, _status(status, {"detail": detail}))
    r = client.post("/api/tts", json={"provider": "piper", "text": "hi"})
    assert r.status_code == status
    assert r.json() == {"detail": detail}

    r = client.post("/v1/audio/speech", json={"input": "hi"})
    assert detail in r.json()["error"]["message"]


@pytest.mark.parametrize("detail", [
    TRACEBACK,
    "could not open /app/models/x.onnx",
    "failed to read /tmp/upload-1234.wav",
    "cannot reach http://qwen3-asr-service:5002/transcribe",
    "C:\\Users\\svc\\models\\x.onnx missing",
    "x" * 501,
    "first line\nsecond line",
    {"nested": {"trace": TRACEBACK}},
    None,
    "",
], ids=["traceback", "app-path", "tmp-path", "url", "windows-path", "too-long", "multiline", "dict", "none", "empty"])
def test_a_4xx_that_looks_like_an_internal_detail_is_treated_like_a_5xx(monkeypatch, caplog, detail):
    _, _, client = _gateway(monkeypatch, _status(400, {"detail": detail}))
    with caplog.at_level(logging.WARNING):
        r = client.post("/api/tts", json={"provider": "piper", "text": "hi"})
    assert r.status_code == 400
    text = r.json()["detail"]
    assert "HTTP 400" in text and "rejected" in text
    _assert_clean(text)
    assert "second line" not in text and "x" * 50 not in text
    _request_id(text)


def test_fastapi_validation_errors_from_a_backend_are_summarised_without_the_input(monkeypatch):
    body = {"detail": [{"type": "string_too_long", "loc": ["body", "text"],
                        "msg": "String should have at most 5000 characters",
                        "input": "the whole secret request text", "ctx": {"max_length": 5000}}]}
    _, _, client = _gateway(monkeypatch, _status(422, body))
    r = client.post("/api/tts", json={"provider": "piper", "text": "hi"})
    assert r.status_code == 422
    assert r.json() == {"detail": "text: String should have at most 5000 characters"}


def test_a_bare_error_field_is_read_like_detail(monkeypatch):
    """whisper.cpp answers {"error": "..."}."""
    _, _, client = _gateway(monkeypatch, _status(400, {"error": "failed to read audio data"}),
                            {"ENABLE_WHISPER_CPP": "true"})
    r = client.post("/api/stt", data={"provider": "whisper-cpp"},
                    files={"audio": ("a.wav", wav_bytes(), "audio/wav")})
    assert r.status_code == 400
    assert r.json() == {"detail": "failed to read audio data"}


# --- unload passes a designed answer through and only cleans a failure -----------------


def test_unload_busy_answer_keeps_its_fields(monkeypatch):
    body = {"detail": "Model is in use; retry when idle", "reason": "busy", "active_requests": 2}
    _, _, client = _gateway(monkeypatch, _status(409, body))
    r = client.post("/api/providers/qwen3/unload")
    assert r.status_code == 409
    assert r.json() == {"provider": "qwen3", **body}


def test_unload_failure_body_is_replaced_but_a_bad_detail_alone_is_not_the_whole_answer(monkeypatch):
    _, _, client = _gateway(monkeypatch, _status(500, {"detail": TRACEBACK, "extra": "/app/models"}))
    r = client.post("/api/providers/qwen3/unload")
    assert r.status_code == 500
    assert set(r.json()) == {"provider", "detail"}
    _assert_clean(r.json()["detail"])

    _, _, client = _gateway(monkeypatch, _status(409, {"detail": TRACEBACK, "reason": "busy"}))
    r = client.post("/api/providers/qwen3/unload")
    assert r.json()["reason"] == "busy"
    _assert_clean(r.json()["detail"])

    _, _, client = _gateway(monkeypatch, _status(502, b"<html>/app/models</html>"))
    r = client.post("/api/providers/qwen3/unload")
    _assert_clean(r.json()["detail"])
    assert "<html" not in r.json()["detail"]


# --- what is deliberately still visible --------------------------------------------------


def test_health_and_providers_still_show_the_provider_urls(monkeypatch):
    """They exist so an operator can see what the gateway is wired to."""
    _, _, client = _gateway(monkeypatch, lambda m, u, k: {"status": "ok"})
    assert client.get("/health").json()["services"]["piper"] == "http://piper-tts-service:5000"
    assert client.get("/providers").json()["providers"]["piper"]["internal_url"] == "http://piper-tts-service:5000"


def test_request_ids_differ_between_failures(monkeypatch):
    _, _, client = _gateway(monkeypatch, _refuse)
    ids = {_request_id(client.post("/api/tts", json={"provider": "piper", "text": "hi"}).json()["detail"])
           for _ in range(5)}
    assert len(ids) == 5
