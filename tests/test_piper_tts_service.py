"""Live tests for the PiperTTS service (they skip when the stack is not running).

The behaviour of the service itself is covered without a container by
test_piper_tts_app.py and test_piper_tts_limits.py; these check the same
contract against the running image.
"""

import pytest


def _require_installed_voices(client):
    """Skip unless /ready says at least one voice model is installed."""
    if client.get("/ready").status_code != 200:
        pytest.skip("no voice models installed - the service reports itself not ready")


def test_health(piper_tts_client):
    """Health endpoint returns a healthy status payload."""
    r = piper_tts_client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data.get("status") == "healthy"


def test_ready_matches_its_status_code(piper_tts_client):
    """Readiness is 200 with voices installed and 503 with the reasons otherwise."""
    r = piper_tts_client.get("/ready")
    body = r.json()
    assert body["status"] == ("ready" if r.status_code == 200 else "not_ready")
    assert (r.status_code == 200) == (not body["problems"])


def test_list_voices(piper_tts_client):
    """Voices endpoint returns a dictionary of available voices."""
    r = piper_tts_client.get("/voices")
    assert r.status_code == 200
    data = r.json()
    assert "voices" in data
    assert isinstance(data["voices"], dict)
    assert data["default_language"]


def test_tts_generates_audio(piper_tts_client):
    """Generate speech with a voice that has a downloaded model."""
    _require_installed_voices(piper_tts_client)
    voices = piper_tts_client.get("/voices").json().get("voices", {})
    voice_name = next(iter(voices))
    r = piper_tts_client.post(
        "/tts",
        json={"text": "Hello, this is a test.", "voice": voice_name, "output_format": "wav"},
    )
    assert r.status_code == 200
    assert r.content[:4] == b"RIFF"


def test_tts_without_language_uses_the_default_language(piper_tts_client):
    """The UI omits `language` for Auto-Detect; that must not mean an English voice."""
    _require_installed_voices(piper_tts_client)
    listing = piper_tts_client.get("/voices").json()
    default = listing["default_language"].split("_")[0].split("-")[0].lower()
    if not any(v["language"].lower().startswith(default) for v in listing["voices"].values()):
        pytest.skip(f"no installed voice for the default language {default!r}")

    r = piper_tts_client.post("/tts", json={"text": "Test."})  # no language evidence in the text

    assert r.status_code == 200
    assert r.headers["X-Language"].lower().startswith(default)
    assert r.headers["X-Language-Fallback"] == "false"


def test_tts_empty_text_rejected(piper_tts_client):
    """Empty text should return an error."""
    r = piper_tts_client.post("/tts", json={"text": ""})
    assert r.status_code in (400, 422)
