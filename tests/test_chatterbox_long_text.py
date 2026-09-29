"""Long texts must be synthesised completely, and over-long texts refused.

One ``generate()`` call in the chatterbox library stops at 1000 speech tokens
(~40 s of audio) and returns what it has: HTTP 200 with the end of the text
missing. The fake model reproduces that by dropping everything past 200
characters, so a service that sends the whole text in one call loses the tail.
"""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from chatterbox_loader import (
    CLONE_VOICE, DEFAULT_VOICE, MODEL_MAX_CHARS, SAMPLE_RATE, SAMPLES_PER_CHAR, decode_audio,
    decode_pcm16, install_chatterbox, load_app, patch_decoder, reference_clip, voices_in,
)

SENTENCE = "Das ist ein ganz gewöhnlicher deutscher Satz über das Wetter im Herbst."
LONG_TEXT = " ".join([SENTENCE] * 30)  # ~2200 characters, about eleven times the fake's limit


@pytest.fixture
def svc(monkeypatch):
    package = install_chatterbox(monkeypatch)
    app = load_app()
    patch_decoder(monkeypatch, app)
    return app, package, TestClient(app.app)


def _spoken_text(package) -> str:
    return " ".join(call["text"] for call in package.model.calls)


def _gap_samples(ms: int) -> int:
    return int(SAMPLE_RATE * ms / 1000)


def test_tts_speaks_the_whole_of_a_long_text(svc):
    app, package, client = svc
    response = client.post("/tts", json={"text": LONG_TEXT, "language": "de"})
    assert response.status_code == 200

    calls = package.model.calls
    assert len(calls) > 1, "a long text was sent to generate() in one piece"
    assert max(len(c["text"]) for c in calls) <= 180 < MODEL_MAX_CHARS
    # Nothing lost, nothing repeated.
    assert _spoken_text(package).split() == LONG_TEXT.split()

    audio = decode_audio(response.content)
    expected = sum(len(c["text"]) * SAMPLES_PER_CHAR for c in calls) + (len(calls) - 1) * _gap_samples(150)
    assert audio.size == expected
    assert response.headers["X-Chunk-Count"] == str(len(calls))


def test_chunks_are_joined_with_the_configured_gap(monkeypatch):
    package = install_chatterbox(monkeypatch)
    app = load_app(CHATTERBOX_CHUNK_GAP_MS="400")
    audio = decode_audio(TestClient(app.app).post("/tts", json={"text": LONG_TEXT, "language": "de"}).content)

    calls = package.model.calls
    assert audio.size == sum(len(c["text"]) * SAMPLES_PER_CHAR for c in calls) + (len(calls) - 1) * _gap_samples(400)
    # The gap is silence between two chunks of speech.
    first = len(calls[0]["text"]) * SAMPLES_PER_CHAR
    assert np.all(audio[first:first + _gap_samples(400)] == 0)
    assert audio[first + _gap_samples(400)] == DEFAULT_VOICE


def test_the_default_gap_is_150_ms_and_zero_disables_it(monkeypatch):
    install_chatterbox(monkeypatch)
    assert load_app().CHUNK_GAP_MS == 150
    package = install_chatterbox(monkeypatch)
    app = load_app(CHATTERBOX_CHUNK_GAP_MS="0")
    audio = decode_audio(TestClient(app.app).post("/tts", json={"text": LONG_TEXT, "language": "de"}).content)
    assert audio.size == sum(len(c["text"]) * SAMPLES_PER_CHAR for c in package.model.calls)


def test_a_short_text_is_still_a_single_generate_call(svc):
    _app, package, client = svc
    response = client.post("/tts", json={"text": "Guten Tag.", "language": "de"})
    assert len(package.model.calls) == 1
    assert decode_audio(response.content).size == len("Guten Tag.") * SAMPLES_PER_CHAR


def test_clone_speaks_the_whole_text_and_builds_the_voice_only_once(svc):
    app, package, client = svc
    response = client.post(
        "/clone",
        data={"text": LONG_TEXT, "lang": "de"},
        files={"file": ("ref.wav", reference_clip(), "audio/wav")},
    )
    assert response.status_code == 200

    calls = package.model.calls
    assert len(calls) > 1
    assert _spoken_text(package).split() == LONG_TEXT.split()
    # The reference is analysed for the first chunk only; the rest reuse that voice.
    assert calls[0]["audio_prompt_path"] is not None
    assert all(c["audio_prompt_path"] is None for c in calls[1:])
    assert len(package.model.prepared_samples) == 1
    assert voices_in(decode_audio(response.content)) == {CLONE_VOICE}


def test_the_stream_pauses_between_chunks_like_tts_does(svc):
    _app, package, client = svc
    body = client.post("/tts-stream", json={"text": LONG_TEXT, "language": "de"}).content

    calls = package.model.calls
    samples = decode_pcm16(body)
    assert samples.size == sum(len(c["text"]) * SAMPLES_PER_CHAR for c in calls) + (len(calls) - 1) * _gap_samples(150)


# --- Text length limit -------------------------------------------------------

@pytest.mark.parametrize("path", ["/tts", "/tts-stream"])
def test_json_endpoints_refuse_text_past_max_text_chars(monkeypatch, path):
    package = install_chatterbox(monkeypatch)
    app = load_app(MAX_TEXT_CHARS="100")
    client = TestClient(app.app)

    refused = client.post(path, json={"text": "a" * 101, "language": "de"})
    accepted = client.post(path, json={"text": "a" * 100, "language": "de"})

    assert refused.status_code == 413
    assert "MAX_TEXT_CHARS" in refused.json()["detail"]
    assert accepted.status_code == 200
    assert all(len(c["text"]) <= 100 for c in package.model.calls)


def test_clone_refuses_text_past_max_text_chars(monkeypatch):
    package = install_chatterbox(monkeypatch)
    app = load_app(MAX_TEXT_CHARS="100")
    patch_decoder(monkeypatch, app)
    client = TestClient(app.app)

    refused = client.post(
        "/clone", data={"text": "a" * 101}, files={"file": ("ref.wav", reference_clip(), "audio/wav")}
    )

    assert refused.status_code == 413
    assert package.loads == 0, "the model was loaded for a request that was refused up front"


def test_the_default_limit_is_5000_characters(monkeypatch):
    install_chatterbox(monkeypatch)
    app = load_app()
    client = TestClient(app.app)
    assert app.MAX_TEXT_CHARS == 5000
    assert client.post("/tts", json={"text": "Wort " * 1001, "language": "de"}).status_code == 413
    assert client.post("/tts", json={"text": ("Wort. " * 833).strip(), "language": "de"}).status_code == 200


@pytest.mark.parametrize("path", ["/tts", "/tts-stream"])
def test_blank_text_is_a_400(svc, path):
    _app, _package, client = svc
    assert client.post(path, json={"text": "  \n ", "language": "de"}).status_code == 400
