"""Tests for the Qwen3-TTS service."""

import pytest


def test_health(qwen3_tts_client):
    r = qwen3_tts_client.get("/health")
    assert r.status_code == 200


def test_status(qwen3_tts_client):
    """Status endpoint returns device and model info."""
    r = qwen3_tts_client.get("/status")
    assert r.status_code == 200
    data = r.json()
    assert "device" in data
    assert "model_loaded" in data


def test_clone_requires_file(qwen3_tts_client):
    """Clone without a voice file should fail."""
    r = qwen3_tts_client.post(
        "/clone",
        data={"text": "test", "lang": "en"},
    )
    assert r.status_code in (400, 422)
