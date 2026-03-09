"""Tests for the STT (Whisper) service."""

import pytest


def test_health(stt_client):
    r = stt_client.get("/health")
    assert r.status_code == 200


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


def test_detect_language(stt_client, test_audio_bytes):
    """Detect language endpoint should return a language code."""
    r = stt_client.post(
        "/detect_language",
        files={"file": ("test.wav", test_audio_bytes, "audio/wav")},
    )
    assert r.status_code == 200
    data = r.json()
    assert "detected_language" in data
