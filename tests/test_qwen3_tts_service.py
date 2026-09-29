"""Tests for the Qwen3-TTS service."""


def test_health(qwen3_tts_client):
    """Health endpoint returns an OK status."""
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


def test_ready_answers_with_a_reason(qwen3_tts_client):
    """Readiness is separate from liveness: 503 while loading, 200 otherwise."""
    r = qwen3_tts_client.get("/ready")
    assert r.status_code in (200, 503)
    body = r.json()
    assert isinstance(body["ready"], bool) and body["reason"]


def test_speakers_say_where_their_list_came_from(qwen3_tts_client):
    """A Base model has no preset speakers; the list is the model's, not a constant."""
    r = qwen3_tts_client.get("/speakers")
    assert r.status_code == 200
    body = r.json()
    assert isinstance(body["speakers"], list)
    assert body["speakers_source"] in {"model", "registry", "fallback"}


def test_an_unsupported_language_is_rejected_before_any_model_work(qwen3_tts_client):
    r = qwen3_tts_client.post("/tts", json={"text": "Hallo", "lang": "nl"})
    assert r.status_code == 400
    assert "German" in r.json()["detail"]


def test_blank_text_is_rejected(qwen3_tts_client):
    r = qwen3_tts_client.post("/tts", json={"text": "   "})
    assert r.status_code == 400
