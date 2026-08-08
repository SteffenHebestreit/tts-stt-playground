"""Tests for the OpenAI-compatible /v1 surface.

The deliverable this surface exists for: a client must get the SAME response
shape whether the deployment behind it is whisper-cpp on the ARM SBC or
faster-whisper on the workstation. `test_identical_shape_across_providers`
is that check; the rest guard the contract details that clients branch on.

Field names and defaults are from the OpenAI OpenAPI spec (info.version 2.3.0).
"""

import json
import struct
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

REPO = Path(__file__).resolve().parents[1]
SERVICE_DIR = REPO / "frontend-service"


def _load_app(monkeypatch_env: dict):
    """Load frontend app.py fresh with the given environment."""
    import os

    app_path = SERVICE_DIR / "app.py"
    spec = spec_from_file_location(f"fe_v1_{abs(hash(frozenset(monkeypatch_env.items())))}", app_path)
    module = module_from_spec(spec)

    prev_cwd = os.getcwd()
    saved = {k: os.environ.get(k) for k in monkeypatch_env}
    sys.path.insert(0, str(SERVICE_DIR))
    try:
        os.environ.update(monkeypatch_env)
        os.chdir(SERVICE_DIR)
        assert spec.loader is not None
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(SERVICE_DIR))
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        os.chdir(prev_cwd)
    return module



def _valid_wav(n_samples: int = 160) -> bytes:
    """A genuinely valid PCM16 mono WAV.

    A truncated RIFF header is enough for a passthrough test but ffmpeg rejects
    it, so the mp3 path needs real bytes to exercise the transcode rather than
    the error branch.
    """
    data = bytes(2 * n_samples)   # PCM16 silence, no escape sequences
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", 36 + len(data), b"WAVE",
        b"fmt ", 16, 1, 1, 16000, 32000, 2, 16,
        b"data", len(data),
    )
    return header + data


class _StubClient:
    """Stands in for httpx.AsyncClient, answering both backend contracts.

    whisper-cpp speaks the OpenAI transcription shape; stt-service speaks the
    project's native shape. Both are exercised so the gateway's normalisation is
    what is actually under test.
    """

    last_post: dict = {}

    def __init__(self, *a, **k):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def post(self, url, **kwargs):
        type(self).last_post = {"url": url, **kwargs}
        if "/tts" in url:
            # Must be a genuinely valid WAV: the mp3 path feeds this to ffmpeg,
            # and a truncated header would exercise the error branch instead.
            return httpx.Response(200, content=_valid_wav(),
                                  headers={"content-type": "audio/wav"},
                                  request=httpx.Request("POST", url))
        if "/v1/audio/transcriptions" in url or "/inference" in url:
            return httpx.Response(200, json={"text": "guten tag"},
                                  request=httpx.Request("POST", url))
        return httpx.Response(200, json={
            "text": "guten tag",
            "segments": [{"start": 0.0, "end": 1.0, "text": "guten tag"}],
            "language": "de",
            "duration": 1.0,
        }, request=httpx.Request("POST", url))

    async def get(self, url, **kwargs):
        return httpx.Response(200, json={"status": "ok"}, request=httpx.Request("GET", url))


@pytest.fixture(scope="module")
def whisper_app():
    """Deployment whose default STT is faster-whisper (native contract)."""
    return _load_app({"DEFAULT_STT_PROVIDER": "whisper"})


@pytest.fixture(scope="module")
def whispercpp_app():
    """Deployment whose default STT is whisper-cpp (OpenAI contract) — i.e. D1/D4."""
    return _load_app({"DEFAULT_STT_PROVIDER": "whisper-cpp", "ENABLE_WHISPER_CPP": "true"})


def _client(module, monkeypatch):
    monkeypatch.setattr(module.httpx, "AsyncClient", _StubClient)
    return TestClient(module.app)


WAV = b"RIFF$\x00\x00\x00WAVEfmt " + b"\x00" * 32


# --- models -----------------------------------------------------------------


def test_list_models_shape(whisper_app, monkeypatch):
    r = _client(whisper_app, monkeypatch).get("/v1/models")
    assert r.status_code == 200
    body = r.json()
    # Spec: ListModelsResponse.required == ['object','data'] — and notably NO
    # 'has_more' (that belongs to the paginated fine-tuning response).
    assert body["object"] == "list"
    assert "has_more" not in body
    for entry in body["data"]:
        assert set(entry) >= {"id", "object", "created", "owned_by"}
        assert entry["object"] == "model"
        assert isinstance(entry["created"], int), "created is unixtime, an integer"


def test_retrieve_unknown_model_uses_the_error_envelope(whisper_app, monkeypatch):
    r = _client(whisper_app, monkeypatch).get("/v1/models/nope")
    assert r.status_code == 404
    err = r.json()["error"]
    # openai-python reads code/param/type off body["error"]; FastAPI's default
    # {"detail": ...} would leave all three None.
    assert set(err) == {"message", "type", "param", "code"}
    assert err["type"] == "invalid_request_error"
    assert err["code"] == "model_not_found"


# --- transcriptions ---------------------------------------------------------


def test_transcription_json_shape(whisper_app, monkeypatch):
    r = _client(whisper_app, monkeypatch).post(
        "/v1/audio/transcriptions",
        files={"file": ("a.wav", WAV, "audio/wav")},
        data={"model": "whisper-1"},
    )
    assert r.status_code == 200
    # Spec: CreateTranscriptionResponseJson.required == ['text'].
    assert r.json() == {"text": "guten tag"}


def test_transcription_text_format_returns_a_raw_body(whisper_app, monkeypatch):
    """openai-python returns response.text verbatim for response_format=text,
    so a JSON-quoted body would reach the caller with literal quote characters."""
    r = _client(whisper_app, monkeypatch).post(
        "/v1/audio/transcriptions",
        files={"file": ("a.wav", WAV, "audio/wav")},
        data={"model": "whisper-1", "response_format": "text"},
    )
    assert r.status_code == 200
    assert r.text == "guten tag"
    assert not r.text.startswith('"')
    assert r.headers["content-type"].startswith("text/plain")


def test_unknown_model_id_is_accepted(whisper_app, monkeypatch):
    """`model` is advisory here. Clients hardcode ids; 400ing on them would
    break every one of them for no benefit."""
    r = _client(whisper_app, monkeypatch).post(
        "/v1/audio/transcriptions",
        files={"file": ("a.wav", WAV, "audio/wav")},
        data={"model": "gpt-4o-transcribe"},
    )
    assert r.status_code == 200


def test_unknown_extra_fields_are_ignored_not_rejected(whisper_app, monkeypatch):
    """The spec has 14 request fields and grows. A 422 on an unrecognised one
    breaks clients that send newer parameters harmlessly."""
    r = _client(whisper_app, monkeypatch).post(
        "/v1/audio/transcriptions",
        files={"file": ("a.wav", WAV, "audio/wav")},
        data={"model": "whisper-1", "chunking_strategy": "auto",
              "keywords": "foo", "include": "logprobs"},
    )
    assert r.status_code == 200


def test_diarized_json_is_refused_explicitly(whisper_app, monkeypatch):
    """Silently returning a different shape would be worse than an error."""
    r = _client(whisper_app, monkeypatch).post(
        "/v1/audio/transcriptions",
        files={"file": ("a.wav", WAV, "audio/wav")},
        data={"model": "whisper-1", "response_format": "diarized_json"},
    )
    assert r.status_code == 400
    assert r.json()["error"]["param"] == "response_format"


def test_invalid_response_format_is_rejected(whisper_app, monkeypatch):
    r = _client(whisper_app, monkeypatch).post(
        "/v1/audio/transcriptions",
        files={"file": ("a.wav", WAV, "audio/wav")},
        data={"model": "whisper-1", "response_format": "yaml"},
    )
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "invalid_value"


def test_empty_file_is_rejected(whisper_app, monkeypatch):
    r = _client(whisper_app, monkeypatch).post(
        "/v1/audio/transcriptions",
        files={"file": ("a.wav", b"", "audio/wav")},
        data={"model": "whisper-1"},
    )
    assert r.status_code == 400
    assert r.json()["error"]["param"] == "file"


# --- the language contract --------------------------------------------------


def test_auto_language_is_sent_explicitly_to_whisper_cpp(whispercpp_app, monkeypatch):
    """The D1/D4 bug: whisper.cpp defaults to English when the field is absent,
    so omitting it made auto-detect silently mean English."""
    client = _client(whispercpp_app, monkeypatch)
    _StubClient.last_post = {}
    client.post("/v1/audio/transcriptions",
                files={"file": ("a.wav", WAV, "audio/wav")},
                data={"model": "whisper-1"})
    assert _StubClient.last_post["data"]["language"] == "auto"


def test_auto_language_is_omitted_for_faster_whisper(whisper_app, monkeypatch):
    """The inverse: faster-whisper rejects the literal string 'auto', so the
    field must be absent for it to auto-detect."""
    client = _client(whisper_app, monkeypatch)
    _StubClient.last_post = {}
    client.post("/v1/audio/transcriptions",
                files={"file": ("a.wav", WAV, "audio/wav")},
                data={"model": "whisper-1", "language": "auto"})
    assert "language" not in _StubClient.last_post["data"]


def test_explicit_language_reaches_both_backends(whisper_app, whispercpp_app, monkeypatch):
    for app in (whisper_app, whispercpp_app):
        client = _client(app, monkeypatch)
        _StubClient.last_post = {}
        client.post("/v1/audio/transcriptions",
                    files={"file": ("a.wav", WAV, "audio/wav")},
                    data={"model": "whisper-1", "language": "de"})
        assert _StubClient.last_post["data"]["language"] == "de"


# --- THE deliverable --------------------------------------------------------


def test_identical_shape_across_providers(whisper_app, whispercpp_app, monkeypatch):
    """Same request, same response shape, regardless of which device answers.

    whisper-cpp (ARM SBC, Strix Halo) and faster-whisper (5060 Ti, 4080) speak
    different backend contracts. If this test fails, the /v1 surface has not
    achieved the one thing it exists for.
    """
    bodies = []
    for app in (whisper_app, whispercpp_app):
        r = _client(app, monkeypatch).post(
            "/v1/audio/transcriptions",
            files={"file": ("a.wav", WAV, "audio/wav")},
            data={"model": "whisper-1", "language": "de"},
        )
        assert r.status_code == 200
        bodies.append(r.json())

    assert bodies[0].keys() == bodies[1].keys(), (
        f"response shapes differ between backends: {bodies[0]} vs {bodies[1]}"
    )
    assert bodies[0] == bodies[1]


# --- speech -----------------------------------------------------------------


def test_speech_requires_input(whisper_app, monkeypatch):
    r = _client(whisper_app, monkeypatch).post(
        "/v1/audio/speech", json={"model": "tts-1", "voice": "alloy"})
    assert r.status_code == 400
    assert r.json()["error"]["param"] == "input"


def test_speech_rejects_input_over_the_limit(whisper_app, monkeypatch):
    r = _client(whisper_app, monkeypatch).post(
        "/v1/audio/speech",
        json={"model": "tts-1", "voice": "alloy", "input": "x" * 4097})
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "string_above_max_length"


def test_speech_wav_passthrough(whisper_app, monkeypatch):
    r = _client(whisper_app, monkeypatch).post(
        "/v1/audio/speech",
        json={"model": "tts-1", "voice": "alloy", "input": "Guten Tag",
              "response_format": "wav"})
    assert r.status_code == 200
    assert r.headers["content-type"] == "audio/wav"
    assert r.content.startswith(b"RIFF")


def test_speech_rejects_out_of_range_speed(whisper_app, monkeypatch):
    for bad in (0.1, 5.0):
        r = _client(whisper_app, monkeypatch).post(
            "/v1/audio/speech",
            json={"model": "tts-1", "voice": "alloy", "input": "hi", "speed": bad})
        assert r.status_code == 400, f"speed={bad} should be rejected"
        assert r.json()["error"]["param"] == "speed"


def test_speech_unknown_voice_falls_back_rather_than_404(whisper_app, monkeypatch):
    """The spec's own prose and its VoiceIdsShared enum disagree on the voice
    list, and the schema accepts any string — so a 404 here would be wrong."""
    r = _client(whisper_app, monkeypatch).post(
        "/v1/audio/speech",
        json={"model": "tts-1", "voice": "not-a-real-voice", "input": "hi",
              "response_format": "wav"})
    assert r.status_code == 200


def test_openai_voice_names_do_not_leak_to_the_backend(whisper_app, monkeypatch):
    """'alloy' is an OpenAI placeholder, not one of our voices — forwarding it
    would make the backend fail to resolve a voice."""
    client = _client(whisper_app, monkeypatch)
    _StubClient.last_post = {}
    client.post("/v1/audio/speech",
                json={"model": "tts-1", "voice": "alloy", "input": "hi",
                      "response_format": "wav"})
    assert "voice" not in _StubClient.last_post.get("json", {})


def test_our_own_voice_name_is_forwarded(whisper_app, monkeypatch):
    client = _client(whisper_app, monkeypatch)
    _StubClient.last_post = {}
    client.post("/v1/audio/speech",
                json={"model": "tts-1", "voice": "de_DE-thorsten-medium",
                      "input": "hi", "response_format": "wav"})
    assert _StubClient.last_post["json"]["voice"] == "de_DE-thorsten-medium"


# --- backwards compatibility ------------------------------------------------


def test_api_namespace_is_untouched(whisper_app, monkeypatch):
    """/api/* is the browser contract and must keep working unchanged."""
    r = _client(whisper_app, monkeypatch).get("/providers")
    assert r.status_code == 200
    assert "providers" in r.json()


def test_speech_mp3_is_the_default_format(whisper_app, monkeypatch):
    """mp3 is the spec default, so a client sending no response_format gets it.

    Skipped where ffmpeg is absent: there the endpoint correctly returns a
    documented 501 pointing at wav, which is asserted separately below.
    """
    import shutil
    r = _client(whisper_app, monkeypatch).post(
        "/v1/audio/speech",
        json={"model": "tts-1", "voice": "alloy", "input": "Guten Tag"})
    if shutil.which("ffmpeg"):
        assert r.status_code == 200
        assert r.headers["content-type"] == "audio/mpeg"
    else:
        assert r.status_code == 501
        assert r.json()["error"]["param"] == "response_format"


# --- registry truthfulness ---------------------------------------------------
#
# The registry is what an API client uses to decide what a deployment can do.
# A capability declared but not implemented is worse than one that is absent:
# the client branches on it and gets nulls. These keep the declarations honest.


@pytest.fixture(scope="module")
def full_registry_app():
    return _load_app({
        "ENABLE_WHISPER_CPP": "true",
        "ENABLE_PARAKEET_ASR": "true",
        "ENABLE_CANARY_ASR": "true",
    })


def _stt_providers(module):
    return {
        pid: p for pid, p in module.PROVIDER_REGISTRY["providers"].items()
        if p.get("kind") == "stt"
    }


def test_every_stt_provider_declares_language_detect(full_registry_app):
    for pid, p in _stt_providers(full_registry_app).items():
        assert isinstance(p.get("language_detect"), bool), (
            f"{pid} does not declare language_detect; an API client cannot tell "
            "whether asking for auto-detection will work"
        )


def test_language_detect_agrees_with_capability_and_contract(full_registry_app):
    """A provider may not advertise detection it cannot perform.

    parakeet and canary both expose a /detect_language route that always returns
    `detected_language: null` — the route exists, the capability does not.
    """
    for pid, p in _stt_providers(full_registry_app).items():
        detects = p.get("language_detect")
        has_cap = "detect_language" in (p.get("capabilities") or [])
        has_contract = "detect_language" in (p.get("contracts") or {})
        if has_cap:
            assert detects, f"{pid} claims the detect_language capability but cannot detect"
        if has_contract:
            assert detects, f"{pid} declares the detect_language contract but cannot detect"


def test_known_stub_providers_do_not_claim_detection(full_registry_app):
    """Pinned explicitly: these two are the ones that used to lie."""
    providers = _stt_providers(full_registry_app)
    for pid in ("parakeet", "canary"):
        if pid in providers:
            assert providers[pid]["language_detect"] is False, (
                f"{pid}'s /detect_language returns null; it must not claim detection"
            )
