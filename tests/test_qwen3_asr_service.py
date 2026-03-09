"""Tests for the Qwen3-ASR service."""

import pytest


def test_health(qwen3_asr_client):
    r = qwen3_asr_client.get("/health")
    assert r.status_code == 200


def test_transcribe(qwen3_asr_client, test_audio_bytes):
    """Transcribe audio and check response schema."""
    r = qwen3_asr_client.post(
        "/transcribe",
        files={"audio": ("test.wav", test_audio_bytes, "audio/wav")},
    )
    assert r.status_code == 200
    data = r.json()
    assert "text" in data


def test_detect_language(qwen3_asr_client, test_audio_bytes):
    """Detect language from audio."""
    r = qwen3_asr_client.post(
        "/detect_language",
        files={"file": ("test.wav", test_audio_bytes, "audio/wav")},
    )
    assert r.status_code == 200
    data = r.json()
    assert "detected_language" in data
