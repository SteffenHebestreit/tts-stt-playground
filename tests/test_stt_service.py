"""Tests for the STT (Whisper) service."""

import time

import pytest

# How long a cold start may take. The model preloads in the background, so the
# port is open (and `service_healthy` passes on /health) while the first
# download and load are still running; /ready is what says the model can serve.
READY_TIMEOUT_S = 600


@pytest.fixture(scope="module", autouse=True)
def _model_ready(stt_client):
    """Wait for /ready before any test here decodes something."""
    deadline = time.monotonic() + READY_TIMEOUT_S
    while True:
        response = stt_client.get("/ready")
        if response.status_code == 200:
            return
        body = response.json()
        assert body.get("reason") != "load_failed", f"the STT model failed to load: {body}"
        assert time.monotonic() < deadline, f"the STT model was still not ready after {READY_TIMEOUT_S}s: {body}"
        time.sleep(3)


def test_health(stt_client):
    """Health endpoint returns a successful response."""
    r = stt_client.get("/health")
    assert r.status_code == 200


def test_ready(stt_client):
    """/ready says the model can serve (resident, or unloaded and reloadable)."""
    r = stt_client.get("/ready")
    assert r.status_code == 200
    body = r.json()
    assert body["ready"] is True
    assert body["reason"] in ("resident", "unloaded")


def test_transcribe_returns_text(stt_client, test_audio_bytes):
    """Transcribe a short audio clip and check response schema."""
    r = stt_client.post(
        "/transcribe",
        files={"audio": ("test.wav", test_audio_bytes, "audio/wav")},
    )
    assert r.status_code == 200
    data = r.json()
    assert "text" in data


def test_transcribe_with_language(stt_client, test_audio_bytes):
    """Transcribe with explicit language parameter."""
    r = stt_client.post(
        "/transcribe",
        files={"audio": ("test.wav", test_audio_bytes, "audio/wav")},
        data={"language": "en"},
    )
    assert r.status_code == 200
    data = r.json()
    assert "text" in data


def test_transcribe_accepts_a_locale_tag(stt_client, test_audio_bytes):
    """A browser-style `de-DE` is reduced to `de` instead of failing inside the decoder."""
    r = stt_client.post(
        "/transcribe",
        files={"audio": ("test.wav", test_audio_bytes, "audio/wav")},
        data={"language": "de-DE"},
    )
    assert r.status_code == 200


def test_transcribe_rejects_an_unknown_language(stt_client, test_audio_bytes):
    r = stt_client.post(
        "/transcribe",
        files={"audio": ("test.wav", test_audio_bytes, "audio/wav")},
        data={"language": "klingon"},
    )
    assert r.status_code == 400


def test_detect_language(stt_client, test_audio_bytes):
    """Detect language endpoint should return a language code."""
    r = stt_client.post(
        "/detect_language",
        files={"file": ("test.wav", test_audio_bytes, "audio/wav")},
    )
    assert r.status_code == 200
    data = r.json()
    assert "detected_language" in data
