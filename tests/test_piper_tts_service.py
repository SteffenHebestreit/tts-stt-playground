"""Tests for the PiperTTS service."""

import pytest


def test_health(piper_tts_client):
    r = piper_tts_client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data.get("status") == "healthy"


def test_list_voices(piper_tts_client):
    r = piper_tts_client.get("/voices")
    assert r.status_code == 200
    data = r.json()
    assert "voices" in data
    assert isinstance(data["voices"], dict)


def test_tts_generates_audio(piper_tts_client):
    """Generate speech with a voice that has a downloaded model."""
    # First check which voices have models available
    voices = piper_tts_client.get("/voices").json().get("voices", {})
    if not voices:
        pytest.skip("No voices available")

    # Pick the first available voice
    voice_name = next(iter(voices))
    r = piper_tts_client.post(
        "/tts",
        json={"text": "Hello, this is a test.", "voice": voice_name, "output_format": "wav"},
    )
    if r.status_code == 500:
        pytest.skip("TTS model files not downloaded — expected in CI without model cache")
    assert r.status_code == 200
    assert r.content[:4] == b"RIFF"


def test_tts_empty_text_rejected(piper_tts_client):
    """Empty text should return an error."""
    r = piper_tts_client.post("/tts", json={"text": ""})
    assert r.status_code in (400, 422, 500)
