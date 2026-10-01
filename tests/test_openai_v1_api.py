"""Tests for the OpenAI-compatible /v1 surface.

The deliverable this surface exists for: a client must get the SAME response
shape whether the deployment behind it is whisper-cpp on the ARM SBC or
faster-whisper on the workstation. `test_identical_shape_across_providers`
is that check; the rest guard the contract details that clients branch on.

Field names and defaults are from the OpenAI OpenAPI spec (info.version 2.3.0).
"""

import asyncio
import json
import os
import struct
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from frontend_loader import asgi_client, install_stub, load_frontend_app, wav_bytes

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


class _StreamableResponse(httpx.Response):
    """An httpx.Response that also answers the streaming-proxy interface.

    `/api/tts` proxies audio with `send(..., stream=True)` and forwards
    `aiter_raw()`; the content here is already materialised, so it goes over in
    one chunk.
    """

    async def aiter_raw(self, chunk_size=None):
        yield self.content

    async def aread(self):
        return self.content

    async def aclose(self):
        return None


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

    def build_request(self, method, url, **kwargs):
        kwargs.pop("timeout", None)
        return {"method": method, "url": url, "kwargs": kwargs}

    async def send(self, request, stream=False):
        """The `/api/tts` streaming path goes through build_request + send.

        Routed back through post() so a single stub answers both call styles and
        `last_post` records either one — which is what lets a test compare the
        body /v1 puts on the wire against the body /api/tts puts on the wire.
        """
        response = await self.post(request["url"], **request["kwargs"])
        return _StreamableResponse(
            status_code=response.status_code,
            headers=response.headers,
            content=response.content,
            request=response.request,
        )

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


def test_auto_language_is_sent_explicitly_to_faster_whisper(whisper_app, monkeypatch):
    """faster-whisper's service accepts the literal 'auto' and detects. Leaving it out is
    a different request: a missing field means STT_DEFAULT_LANGUAGE, which the TrueNAS
    profile sets to 'de', so 'auto' used to be answered with German transcription."""
    client = _client(whisper_app, monkeypatch)
    for spelling in ("auto", "AUTO", " Auto "):
        _StubClient.last_post = {}
        client.post("/v1/audio/transcriptions",
                    files={"file": ("a.wav", WAV, "audio/wav")},
                    data={"model": "whisper-1", "language": spelling})
        assert _StubClient.last_post["data"]["language"] == "auto", spelling


def test_an_omitted_language_stays_omitted_for_faster_whisper(whisper_app, monkeypatch):
    """No language at all still means "the operator's default", not "detect"."""
    client = _client(whisper_app, monkeypatch)
    _StubClient.last_post = {}
    client.post("/v1/audio/transcriptions",
                files={"file": ("a.wav", WAV, "audio/wav")}, data={"model": "whisper-1"})
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


# --- speech across TTS providers ---------------------------------------------
#
# The module's whole premise is that a client must not care which backend it is
# talking to. That held for STT and quietly failed for TTS: /v1/audio/speech
# built Piper's body regardless of the deployment's default provider, and every
# other backend names the same fields differently. Pydantic drops unknown keys
# silently, so on a qwen3 or chatterbox deployment the request succeeded with
# HTTP 200 and returned the default speaker reading the text in the wrong
# language. Nothing in the response said so.


@pytest.fixture(scope="module")
def qwen3_tts_app():
    """Deployment whose default TTS is Qwen3 — `lang` / `speaker`, not `language` / `voice`."""
    return _load_app({"DEFAULT_TTS_PROVIDER": "qwen3"})


@pytest.fixture(scope="module")
def chatterbox_tts_app():
    """Deployment whose default TTS is Chatterbox — the German-capable MIT option."""
    return _load_app({
        "DEFAULT_TTS_PROVIDER": "chatterbox",
        "ENABLE_CHATTERBOX_TTS": "true",
    })


@pytest.fixture(scope="module")
def magpie_tts_app():
    """Deployment whose default TTS is NVIDIA Magpie — `speaker`, five built-in voices, no speed."""
    return _load_app({
        "DEFAULT_TTS_PROVIDER": "magpie",
        "ENABLE_MAGPIE_TTS": "true",
    })


def test_speech_reaches_qwen3_in_its_own_field_names(qwen3_tts_app, monkeypatch):
    client = _client(qwen3_tts_app, monkeypatch)
    _StubClient.last_post = {}
    r = client.post("/v1/audio/speech", json={
        "model": "tts-1", "input": "Guten Tag", "voice": "Ethan",
        "language": "de", "response_format": "wav",
    })
    assert r.status_code == 200
    sent = _StubClient.last_post["json"]
    assert sent["lang"] == "German", (
        "language=de was dropped: qwen3 reads `lang` with English labels, so the "
        "request silently synthesised German text with the English decoder"
    )
    assert sent["speaker"] == "Ethan", "voice was dropped; qwen3 reads `speaker`"
    assert "language" not in sent and "voice" not in sent


def test_openai_placeholder_voice_does_not_become_a_qwen3_speaker(qwen3_tts_app, monkeypatch):
    """'alloy' is OpenAI's name for nothing we have; it must not override the default."""
    client = _client(qwen3_tts_app, monkeypatch)
    _StubClient.last_post = {}
    client.post("/v1/audio/speech", json={
        "model": "tts-1", "input": "hi", "voice": "alloy", "response_format": "wav"})
    assert _StubClient.last_post["json"]["speaker"] == "Vivian"


def test_speech_reaches_chatterbox_with_its_language_field(chatterbox_tts_app, monkeypatch):
    client = _client(chatterbox_tts_app, monkeypatch)
    _StubClient.last_post = {}
    r = client.post("/v1/audio/speech", json={
        "model": "tts-1", "input": "Guten Tag", "language": "de",
        "response_format": "wav"})
    assert r.status_code == 200
    sent = _StubClient.last_post["json"]
    assert sent == {"text": "Guten Tag", "language": "de"}, (
        "chatterbox takes text+language only; forwarding piper's speed/voice/"
        "output_format relies on Pydantic silently discarding them"
    )


def test_speech_reaches_magpie_with_its_speaker_field(magpie_tts_app, monkeypatch):
    client = _client(magpie_tts_app, monkeypatch)
    _StubClient.last_post = {}
    r = client.post("/v1/audio/speech", json={
        "model": "tts-1", "input": "Guten Tag", "voice": "Leo", "language": "de",
        "response_format": "wav"})
    assert r.status_code == 200
    sent = _StubClient.last_post["json"]
    assert sent == {"text": "Guten Tag", "language": "de", "speaker": "Leo"}, (
        "magpie reads text, language and speaker; piper's voice/speed/output_format would be "
        "dropped by Pydantic in silence and the default speaker would answer"
    )


def test_openai_placeholder_voice_does_not_become_a_magpie_speaker(magpie_tts_app, monkeypatch):
    """'alloy' is OpenAI's name for nothing we have; Magpie would refuse it as an unknown speaker."""
    client = _client(magpie_tts_app, monkeypatch)
    _StubClient.last_post = {}
    client.post("/v1/audio/speech", json={
        "model": "tts-1", "input": "hi", "voice": "alloy", "response_format": "wav"})
    assert _StubClient.last_post["json"] == {"text": "hi", "language": "auto", "speaker": "auto"}


def test_speech_body_matches_the_api_tts_body_for_the_same_request(qwen3_tts_app, monkeypatch):
    """/v1 and /api/tts must translate identically — one shared builder, one truth."""
    client = _client(qwen3_tts_app, monkeypatch)

    _StubClient.last_post = {}
    client.post("/v1/audio/speech", json={
        "model": "tts-1", "input": "Guten Tag", "voice": "Ethan",
        "language": "de", "response_format": "wav"})
    via_v1 = _StubClient.last_post["json"]

    _StubClient.last_post = {}
    client.post("/api/tts", json={
        "provider": "qwen3", "text": "Guten Tag", "voice": "Ethan", "language": "de"})
    via_api = _StubClient.last_post["json"]

    assert via_v1 == via_api


def test_speech_refuses_a_provider_with_no_tts_contract(whisper_app, monkeypatch):
    """A misconfigured DEFAULT_TTS_PROVIDER must say so, not 500."""
    module = whisper_app
    monkeypatch.setitem(module.PROVIDER_REGISTRY["ui"], "default_tts_provider", "piper")
    monkeypatch.setitem(
        module.PROVIDER_REGISTRY["providers"]["piper"], "contracts", {})
    r = _client(module, monkeypatch).post(
        "/v1/audio/speech", json={"model": "tts-1", "input": "hi", "response_format": "wav"})
    assert r.status_code == 503
    assert r.json()["error"]["type"] == "server_error"


# --- backwards compatibility ------------------------------------------------


def test_api_namespace_is_untouched(whisper_app, monkeypatch):
    """/api/* is the browser contract and must keep working unchanged."""
    r = _client(whisper_app, monkeypatch).get("/providers")
    assert r.status_code == 200
    assert "providers" in r.json()


def test_speech_mp3_is_the_default_format(whisper_app, monkeypatch):
    """mp3 is the spec default, so a client sending no response_format gets it.

    The transcoder is replaced rather than skipped when ffmpeg is absent: this
    used to branch on `shutil.which("ffmpeg")`, so on a CI image without it the
    success path never ran at all, and on one with it the test needed a real
    encoder and a decodable WAV.
    """
    received = []

    async def fake_transcoder(wav):
        received.append(wav)
        return b"ID3-transcoded"

    monkeypatch.setattr(whisper_app.openai_router, "_wav_to_mp3", fake_transcoder)
    r = _client(whisper_app, monkeypatch).post(
        "/v1/audio/speech",
        json={"model": "tts-1", "voice": "alloy", "input": "Guten Tag"})
    assert r.status_code == 200
    assert r.headers["content-type"] == "audio/mpeg"
    assert r.content == b"ID3-transcoded"
    assert received and received[0].startswith(b"RIFF"), "the backend's WAV is what gets transcoded"


def test_speech_mp3_without_a_transcoder_is_a_documented_501(whisper_app, monkeypatch):
    async def unavailable(wav):
        return None

    monkeypatch.setattr(whisper_app.openai_router, "_wav_to_mp3", unavailable)
    r = _client(whisper_app, monkeypatch).post(
        "/v1/audio/speech", json={"model": "tts-1", "input": "Guten Tag"})
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


# --- ffmpeg: asynchronous, bounded, and absent-tolerant ------------------------
#
# `_wav_to_mp3` used `subprocess.run` inside an `async def`. mp3 is the DEFAULT
# format, so every default request froze the whole worker for the length of the
# encode: /health, the /ws/stt relay and every other request queued behind it.
# These use a stand-in `ffmpeg` on PATH so the real subprocess path runs.


def _fake_ffmpeg(directory, body: str):
    script = directory / "ffmpeg"
    script.write_text("#!/bin/sh\n" + body + "\n")
    script.chmod(0o755)
    return script


def _path_with(monkeypatch, directory):
    monkeypatch.setenv("PATH", f"{directory}{os.pathsep}{os.environ.get('PATH', '')}")


def test_mp3_transcode_does_not_block_the_event_loop(tmp_path, monkeypatch):
    _fake_ffmpeg(tmp_path, "cat >/dev/null; sleep 0.5; printf 'ID3-real-subprocess'")
    _path_with(monkeypatch, tmp_path)
    module = load_frontend_app()
    install_stub(monkeypatch, module, lambda m, u, k: httpx.Response(
        200, content=wav_bytes(), headers={"content-type": "audio/wav"}))

    async def run():
        loop = asyncio.get_running_loop()
        worst_gap = 0.0

        async def heartbeat():
            nonlocal worst_gap
            last = loop.time()
            while True:
                await asyncio.sleep(0.01)
                now = loop.time()
                worst_gap = max(worst_gap, now - last)
                last = now

        beat = asyncio.create_task(heartbeat())
        await asyncio.sleep(0.05)              # let it take its first reading
        async with asgi_client(module) as client:
            response = await client.post("/v1/audio/speech", json={"input": "Guten Tag"})
        # One more reading after the request: a stall shows up as the gap that ends
        # at the first tick following it, so the heartbeat must get to take one.
        await asyncio.sleep(0.05)
        beat.cancel()
        return response, worst_gap

    response, worst_gap = asyncio.run(run())
    assert response.status_code == 200
    assert response.content == b"ID3-real-subprocess"
    assert worst_gap < 0.25, (
        f"the event loop was frozen for {worst_gap:.2f}s while ffmpeg ran; "
        "nothing else on the worker can be served in that time"
    )


def test_missing_ffmpeg_returns_none_instead_of_raising(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", str(tmp_path))          # an empty directory
    module = load_frontend_app()
    assert asyncio.run(module.openai_router._wav_to_mp3(wav_bytes())) is None


def test_failing_ffmpeg_returns_none(tmp_path, monkeypatch):
    _fake_ffmpeg(tmp_path, "cat >/dev/null; echo 'invalid data' >&2; exit 1")
    _path_with(monkeypatch, tmp_path)
    module = load_frontend_app()
    assert asyncio.run(module.openai_router._wav_to_mp3(wav_bytes())) is None


def test_hung_ffmpeg_is_killed_at_the_timeout(tmp_path, monkeypatch):
    _fake_ffmpeg(tmp_path, "cat >/dev/null; exec sleep 3")
    _path_with(monkeypatch, tmp_path)
    module = load_frontend_app()
    monkeypatch.setattr(module.openai_router, "FFMPEG_TIMEOUT_S", 0.3)

    async def run():
        loop = asyncio.get_running_loop()
        started = loop.time()
        result = await module.openai_router._wav_to_mp3(wav_bytes())
        return result, loop.time() - started

    result, elapsed = asyncio.run(run())
    assert result is None
    assert elapsed < 2.0, "the timeout did not fire"


# --- speech: pcm ---------------------------------------------------------------


def _speech_app(monkeypatch, wav):
    module = load_frontend_app()
    install_stub(monkeypatch, module, lambda m, u, k: httpx.Response(
        200, content=wav, headers={"content-type": "audio/wav"}))
    return TestClient(module.app)


def test_pcm_that_is_already_24khz_is_the_wav_without_its_header_and_needs_no_ffmpeg(tmp_path, monkeypatch):
    """Qwen3 and Chatterbox produce 24 kHz, which is what OpenAI's pcm is: a header strip."""
    monkeypatch.setenv("PATH", str(tmp_path))          # no ffmpeg anywhere
    samples = struct.pack("<hhh", 1, -2, 300)
    client = _speech_app(monkeypatch, wav_bytes(samples, rate=24000))
    r = client.post("/v1/audio/speech", json={"input": "hi", "response_format": "pcm"})
    assert r.status_code == 200
    assert r.content == samples
    assert r.headers["content-type"] == "audio/pcm"
    assert r.headers["x-sample-rate"] == "24000"
    assert r.headers["x-provider"] == "piper"


@pytest.mark.parametrize("rate", [22050, 16000, 44100])
def test_pcm_from_any_other_rate_is_resampled_to_24khz_with_ffmpeg(tmp_path, monkeypatch, rate):
    """Piper voices are 22.05 or 16 kHz, and stock OpenAI clients play `pcm` at 24 kHz, so
    unresampled samples came out slow and low. The stand-in ffmpeg records how it was
    called and what it was fed, and answers with bytes the WAV cannot have contained."""
    _fake_ffmpeg(tmp_path, 'printf "%s\\n" "$@" > "$0.args"; cat > "$0.stdin"; printf "RESAMPLED-24K"')
    _path_with(monkeypatch, tmp_path)
    wav = wav_bytes(struct.pack("<hhh", 1, -2, 300), rate=rate)
    client = _speech_app(monkeypatch, wav)

    r = client.post("/v1/audio/speech", json={"input": "hi", "response_format": "pcm"})

    assert r.status_code == 200
    assert r.content == b"RESAMPLED-24K"
    assert r.headers["content-type"] == "audio/pcm"
    assert r.headers["x-sample-rate"] == "24000", "the header must describe the bytes that are sent"
    args = (tmp_path / "ffmpeg.args").read_text().split()
    assert args[args.index("-ar") + 1] == "24000"
    assert args[args.index("-ac") + 1] == "1"
    last_format_flag = max(i for i, arg in enumerate(args) if arg == "-f")   # the input's `-f wav` comes first
    assert args[last_format_flag + 1] == "s16le"
    assert args[-1] == "pipe:1"
    assert (tmp_path / "ffmpeg.stdin").read_bytes() == wav, "ffmpeg must be fed the backend's WAV"


def test_pcm_that_needs_resampling_without_ffmpeg_is_a_documented_501(tmp_path, monkeypatch):
    """Header-stripping at the wrong rate would be the silent slow-audio bug again."""
    monkeypatch.setenv("PATH", str(tmp_path))
    client = _speech_app(monkeypatch, wav_bytes(struct.pack("<hhh", 1, -2, 300), rate=22050))
    r = client.post("/v1/audio/speech", json={"input": "hi", "response_format": "pcm"})
    assert r.status_code == 501
    error = r.json()["error"]
    assert error["code"] == "unsupported_value" and error["param"] == "response_format"
    assert "24 kHz" in error["message"] and "wav" in error["message"]


def test_failing_resample_is_reported_not_returned_as_audio(tmp_path, monkeypatch):
    _fake_ffmpeg(tmp_path, "cat >/dev/null; echo 'boom' >&2; exit 1")
    _path_with(monkeypatch, tmp_path)
    client = _speech_app(monkeypatch, wav_bytes(struct.pack("<hhh", 1, -2, 300), rate=22050))
    r = client.post("/v1/audio/speech", json={"input": "hi", "response_format": "pcm"})
    assert r.status_code == 501


def test_wav_format_keeps_the_backends_own_rate(monkeypatch):
    """Only `pcm` (a headerless format) has to agree with what clients assume."""
    wav = wav_bytes(struct.pack("<hhh", 1, -2, 300), rate=22050)
    client = _speech_app(monkeypatch, wav)
    r = client.post("/v1/audio/speech", json={"input": "hi", "response_format": "wav"})
    assert r.status_code == 200 and r.content == wav


def test_pcm_downmixes_to_mono(monkeypatch):
    stereo = struct.pack("<hhhh", 100, 300, -50, 50)          # two frames
    client = _speech_app(monkeypatch, wav_bytes(stereo, rate=24000, channels=2))
    r = client.post("/v1/audio/speech", json={"input": "hi", "response_format": "pcm"})
    assert r.status_code == 200
    assert struct.unpack("<hh", r.content) == (200, 0)
    assert r.headers["x-sample-rate"] == "24000"


@pytest.mark.parametrize("wav", [
    wav_bytes(b"\x00" * 6, width=3),          # 24-bit
    b"definitely not a wav file",
])
def test_pcm_from_audio_that_is_not_16_bit_wav_is_a_502_envelope(monkeypatch, wav):
    client = _speech_app(monkeypatch, wav)
    r = client.post("/v1/audio/speech", json={"input": "hi", "response_format": "pcm"})
    assert r.status_code == 502
    assert set(r.json()["error"]) == {"message", "type", "param", "code"}


def test_other_speech_formats_are_still_refused(monkeypatch):
    client = _speech_app(monkeypatch, wav_bytes())
    for fmt in ("opus", "aac", "flac"):
        r = client.post("/v1/audio/speech", json={"input": "hi", "response_format": fmt})
        assert r.status_code == 400 and r.json()["error"]["code"] == "unsupported_value", fmt


# --- transcription: verbose_json, srt, vtt -------------------------------------

NATIVE_PAYLOAD = {
    "text": "Guten Tag zusammen",
    "language": "de",
    "duration": 3.5,
    "segments": [
        {"start": 0.0, "end": 1.5, "text": "Guten Tag", "avg_logprob": -0.25, "no_speech_prob": 0.01},
        {"start": 1.5, "end": 3.5, "text": " zusammen"},
    ],
}


def _stt_client(monkeypatch, payload, env=None, capture=None):
    module = load_frontend_app(env)

    def handler(method, url, kwargs):
        if capture is not None:
            capture.append((url, kwargs))
        return payload

    install_stub(monkeypatch, module, handler)
    return TestClient(module.app)


def _transcribe_fmt(client, fmt, **data):
    return client.post("/v1/audio/transcriptions",
                       data={"model": "whisper-1", "response_format": fmt, **data},
                       files={"file": ("a.wav", wav_bytes(), "audio/wav")})


def test_verbose_json_matches_the_specs_required_fields(monkeypatch):
    r = _transcribe_fmt(_stt_client(monkeypatch, NATIVE_PAYLOAD), "verbose_json", temperature="0.2")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("application/json")
    body = r.json()
    # CreateTranscriptionResponseVerboseJson.required
    assert body["language"] == "de"
    assert body["duration"] == 3.5
    assert body["text"] == "Guten Tag zusammen"
    assert body["task"] == "transcribe"
    # TranscriptionSegment.required - openai-python validates every one of these
    required = {"id", "seek", "start", "end", "text", "tokens", "temperature",
                "avg_logprob", "compression_ratio", "no_speech_prob"}
    assert [seg["id"] for seg in body["segments"]] == [0, 1]
    for seg in body["segments"]:
        assert set(seg) == required
    first, second = body["segments"]
    assert (first["start"], first["end"], first["text"]) == (0.0, 1.5, "Guten Tag")
    assert first["avg_logprob"] == -0.25 and first["no_speech_prob"] == 0.01
    assert second["text"] == "zusammen", "leading whitespace is trimmed"
    assert first["temperature"] == 0.2


def test_srt_output_is_exact(monkeypatch):
    r = _transcribe_fmt(_stt_client(monkeypatch, NATIVE_PAYLOAD), "srt")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/plain")
    assert r.text == (
        "1\n00:00:00,000 --> 00:00:01,500\nGuten Tag\n"
        "\n"
        "2\n00:00:01,500 --> 00:00:03,500\nzusammen\n"
    )


def test_vtt_output_is_exact(monkeypatch):
    r = _transcribe_fmt(_stt_client(monkeypatch, NATIVE_PAYLOAD), "vtt")
    assert r.status_code == 200
    assert r.text == (
        "WEBVTT\n\n"
        "00:00:00.000 --> 00:00:01.500\nGuten Tag\n"
        "\n"
        "00:00:01.500 --> 00:00:03.500\nzusammen\n"
    )


def test_a_backend_without_segments_gets_one_spanning_segment(monkeypatch):
    payload = {"text": "hallo welt", "duration": 2.25}
    client = _stt_client(monkeypatch, payload)
    body = _transcribe_fmt(client, "verbose_json").json()
    assert len(body["segments"]) == 1
    assert (body["segments"][0]["start"], body["segments"][0]["end"]) == (0.0, 2.25)
    assert body["segments"][0]["text"] == "hallo welt"
    assert _transcribe_fmt(client, "srt").text == "1\n00:00:00,000 --> 00:00:02,250\nhallo welt\n"


def test_an_empty_transcript_renders_empty_documents(monkeypatch):
    client = _stt_client(monkeypatch, {"text": "", "segments": []})
    assert _transcribe_fmt(client, "srt").text == ""
    assert _transcribe_fmt(client, "vtt").text == "WEBVTT\n\n"
    body = _transcribe_fmt(client, "verbose_json").json()
    assert body["segments"] == [] and body["text"] == "" and body["duration"] == 0.0


def test_timestamps_use_the_right_separator_and_carry_hours(monkeypatch):
    payload = {"text": "spaet", "segments": [{"start": 3661.5, "end": 3662.0004, "text": "spaet"}]}
    client = _stt_client(monkeypatch, payload)
    assert "01:01:01,500 --> 01:01:02,000" in _transcribe_fmt(client, "srt").text
    assert "01:01:01.500 --> 01:01:02.000" in _transcribe_fmt(client, "vtt").text


def test_junk_in_a_backends_numeric_fields_does_not_fail_the_request(monkeypatch):
    # NaN is what whisper.cpp emits for a segment with no tokens; Python's JSON
    # parser accepts the bare token, so it has to be sent as raw text here.
    payload = httpx.Response(200, content=(
        b'{"text": "hi", "duration": "n/a", "segments": ['
        b'{"start": "abc", "end": null, "text": "hi", "avg_logprob": NaN}]}'),
        headers={"content-type": "application/json"})
    r = _transcribe_fmt(_stt_client(monkeypatch, payload), "verbose_json")
    assert r.status_code == 200
    seg = r.json()["segments"][0]
    assert (seg["start"], seg["end"], seg["avg_logprob"]) == (0.0, 0.0, 0.0)


def test_whisper_cpp_is_asked_for_timings_only_when_they_are_needed(monkeypatch):
    env = {"DEFAULT_STT_PROVIDER": "whisper-cpp", "ENABLE_WHISPER_CPP": "true"}
    cpp_payload = {
        "task": "transcribe", "language": "german", "duration": 2.0, "text": " Guten Tag",
        "segments": [{"id": 0, "text": " Guten Tag", "start": 0.0, "end": 2.0}],
    }
    sent = []
    client = _stt_client(monkeypatch, cpp_payload, env, capture=sent)

    body = _transcribe_fmt(client, "verbose_json").json()
    assert sent[-1][1]["data"]["response_format"] == "verbose_json"
    assert body["language"] == "german"
    assert body["segments"][0]["text"] == "Guten Tag"

    _transcribe_fmt(client, "srt")
    assert sent[-1][1]["data"]["response_format"] == "verbose_json"

    _transcribe_fmt(client, "json")
    assert sent[-1][1]["data"]["response_format"] == "json"
    _transcribe_fmt(client, "text")
    assert sent[-1][1]["data"]["response_format"] == "json"


def test_diarized_json_is_still_refused(monkeypatch):
    r = _transcribe_fmt(_stt_client(monkeypatch, NATIVE_PAYLOAD), "diarized_json")
    assert r.status_code == 400 and r.json()["error"]["code"] == "unsupported_value"


def test_non_object_backend_reply_is_a_502_envelope(monkeypatch):
    r = _transcribe_fmt(_stt_client(monkeypatch, ["not", "an", "object"]), "json")
    assert r.status_code == 502
    assert set(r.json()["error"]) == {"message", "type", "param", "code"}


# --- ffmpeg concurrency --------------------------------------------------------
#
# Every mp3 response (the default format) and every resampled pcm response starts
# an ffmpeg process with the WAV held beside it. A burst used to start one per
# request; now at most MAX_CONCURRENT_FFMPEG run per worker and the rest are told
# to retry, which stock OpenAI clients do on their own.


def _slow_ffmpeg_app(tmp_path, monkeypatch, slots):
    _fake_ffmpeg(tmp_path, "cat >/dev/null; sleep 0.4; printf 'ID3-encoded'")
    _path_with(monkeypatch, tmp_path)
    module = load_frontend_app()
    monkeypatch.setattr(module.openai_router, "_ffmpeg_slots", module.openai_router.Slots(slots))
    install_stub(monkeypatch, module, lambda m, u, k: httpx.Response(
        200, content=wav_bytes(), headers={"content-type": "audio/wav"}))
    return module


def test_ffmpeg_processes_are_capped_and_the_overflow_is_told_to_retry(tmp_path, monkeypatch):
    module = _slow_ffmpeg_app(tmp_path, monkeypatch, slots=2)

    async def burst():
        async with asgi_client(module) as client:
            responses = await asyncio.gather(*(
                client.post("/v1/audio/speech", json={"input": f"text {i}"}) for i in range(5)))
            later = await client.post("/v1/audio/speech", json={"input": "after the burst"})
            return responses, later

    responses, later = asyncio.run(burst())
    statuses = sorted(r.status_code for r in responses)
    assert statuses == [200, 200, 503, 503, 503], statuses
    for r in responses:
        if r.status_code == 503:
            assert r.headers["retry-after"] == "2"
            error = r.json()["error"]
            assert error["code"] == "server_busy" and error["type"] == "server_error"
    assert later.status_code == 200, "the slots were not given back"
    assert module.openai_router._ffmpeg_slots.active == 0


def test_pcm_resampling_shares_the_same_cap_but_24khz_pcm_and_wav_do_not_use_it(tmp_path, monkeypatch):
    module = _slow_ffmpeg_app(tmp_path, monkeypatch, slots=0)          # every slot is taken

    def speech(rate, fmt):
        install_stub(monkeypatch, module, lambda m, u, k: httpx.Response(
            200, content=wav_bytes(struct.pack("<hh", 5, 6), rate=rate),
            headers={"content-type": "audio/wav"}))
        return TestClient(module.app).post(
            "/v1/audio/speech", json={"input": "hi", "response_format": fmt})

    assert speech(22050, "pcm").status_code == 503
    assert speech(22050, "mp3").status_code == 503
    assert speech(24000, "pcm").status_code == 200
    assert speech(22050, "wav").status_code == 200


def test_ffmpeg_slot_is_released_after_a_failed_or_timed_out_run(tmp_path, monkeypatch):
    _fake_ffmpeg(tmp_path, "cat >/dev/null; exec sleep 3")
    _path_with(monkeypatch, tmp_path)
    module = load_frontend_app()
    router = module.openai_router
    monkeypatch.setattr(router, "_ffmpeg_slots", router.Slots(1))
    monkeypatch.setattr(router, "FFMPEG_TIMEOUT_S", 0.3)
    assert asyncio.run(router._wav_to_mp3(wav_bytes())) is None
    assert router._ffmpeg_slots.active == 0

    monkeypatch.setenv("PATH", str(tmp_path / "nowhere"))
    assert asyncio.run(router._wav_to_mp3(wav_bytes())) is None
    assert router._ffmpeg_slots.active == 0


@pytest.mark.parametrize("raw,expected", [
    (None, 4), ("", 4), ("7", 7), (" 3 ", 3), ("0", 4), ("-2", 4), ("many", 4), ("2.5", 4),
])
def test_max_concurrent_ffmpeg_setting_is_read_defensively(monkeypatch, raw, expected):
    router = load_frontend_app().openai_router
    if raw is None:
        monkeypatch.delenv("MAX_CONCURRENT_FFMPEG", raising=False)
    else:
        monkeypatch.setenv("MAX_CONCURRENT_FFMPEG", raw)
    assert router._env_positive_int("MAX_CONCURRENT_FFMPEG", 4) == expected
