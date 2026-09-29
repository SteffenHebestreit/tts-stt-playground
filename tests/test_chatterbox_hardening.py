"""chatterbox-tts-service hardening: what errors reveal, how long a request may queue, the default language.

* Unexpected failures answer with a generic message and a request id; the exception
  text (library messages carry paths, URLs, shapes) is in the log under the same id (E1).
* The wait for the generation permit is bounded in time and in depth: a request that
  cannot get one in TTS_QUEUE_TIMEOUT_S, or finds TTS_MAX_QUEUE already waiting, is
  answered 503 with Retry-After, and nothing (permit, model reference, temp file, wait
  count) is left behind on any of those paths (E4).
* CHATTERBOX_DEFAULT_LANGUAGE is normalised to a language id the library accepts (E5).
"""

import asyncio
import logging
import sys
import tempfile
import threading
import time
import types

import pytest
from fastapi.testclient import TestClient

from chatterbox_loader import (
    DEFAULT_VOICE, install_chatterbox, load_app, make_upload, patch_decoder, reference_clip, wait_until,
    wait_until_async,
)



def _warm(app):
    TestClient(app.app).post("/tts", json={"text": "Wärmen.", "language": "de"})


def _tts(client, text="Hallo Welt.", **body):
    return client.post("/tts", json={"text": text, "language": "de", **body})


def _clone(client, text="Klon.", **form):
    return client.post(
        "/clone", data={"text": text, "lang": "de", **form},
        files={"file": ("ref.wav", reference_clip(), "audio/wav")},
    )


@pytest.fixture
def scratch(tmp_path, monkeypatch):
    """Temp files land in `tmp_path`, so a leaked one is visible."""
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    return tmp_path


# --- E1: generic errors with an id ---------------------------------------------------------

def _assert_generic(response, log, secret, status=500):
    assert response.status_code == status, response.text
    request_id = response.headers["X-Request-ID"]
    assert request_id in response.json()["detail"]
    assert secret not in response.text, "the exception text reached the client"
    assert any(request_id in r.getMessage() and secret in r.getMessage() for r in log.records), \
        "the detail is not in the log under the id the client was given"


def test_a_generation_failure_does_not_reveal_the_exception(monkeypatch, caplog):
    package = install_chatterbox(monkeypatch)
    app = load_app()
    patch_decoder(monkeypatch, app)
    _warm(app)
    client = TestClient(app.app)

    def explode(text, **kwargs):
        raise RuntimeError("CUDA error at /usr/lib/python3/site-packages/chatterbox/t3.py:212 (shape [1, 9999])")

    package.model.generate = explode

    with caplog.at_level(logging.ERROR):
        via_tts = _tts(client)
        via_stream = client.post("/tts-stream", json={"text": "Hallo Welt.", "language": "de"})
        via_clone = _clone(client)

    for response in (via_tts, via_stream, via_clone):
        _assert_generic(response, caplog, "site-packages/chatterbox/t3.py")
    assert len({r.headers["X-Request-ID"] for r in (via_tts, via_stream, via_clone)}) == 3


def test_a_model_that_cannot_be_loaded_is_generic_for_tts_and_ready(monkeypatch, caplog):
    package = install_chatterbox(monkeypatch)
    app = load_app()
    client = TestClient(app.app, raise_server_exceptions=False)

    def unreachable(cls, device, t3_model=None):
        raise OSError("[Errno 28] No space left on device: '/root/.cache/huggingface/hub/models--ResembleAI'")

    sys.modules["chatterbox.mtl_tts"].ChatterboxMultilingualTTS.from_pretrained = classmethod(unreachable)

    with caplog.at_level(logging.ERROR):
        response = _tts(client)
        ready = client.get("/ready")

    _assert_generic(response, caplog, "/root/.cache/huggingface")
    assert ready.status_code == 503 and ready.json()["reason"] == "load_failed"
    for secret in ("/root/.cache", "No space left", "OSError", "huggingface"):
        assert secret not in ready.text, f"/ready reveals {secret!r}"
    assert any("No space left" in r.getMessage() for r in caplog.records), "the operator lost the detail"
    assert package.loads == 0


def test_a_message_written_for_the_caller_is_kept(monkeypatch):
    install_chatterbox(monkeypatch, conds=None)  # a checkpoint with no built-in voice
    client = TestClient(load_app().app)

    response = _tts(client)

    assert response.status_code == 500 and "/clone" in response.json()["detail"]


def test_the_text_limit_error_says_how_to_raise_the_limit(monkeypatch):
    install_chatterbox(monkeypatch)
    client = TestClient(load_app(MAX_TEXT_CHARS="20").app)

    response = _tts(client, text="x" * 21)

    assert response.status_code == 413
    detail = response.json()["detail"]
    assert "MAX_TEXT_CHARS" in detail and "20" in detail and "/" not in detail


# --- E4: bounded waiting for a generation permit -------------------------------------------

@pytest.fixture
def busy(monkeypatch, scratch):
    """busy(**env) -> (app, package): a loaded model whose next generation blocks on `model.gate`."""
    def _busy(**env):
        package = install_chatterbox(monkeypatch)
        app = load_app(**env)
        _warm(app)
        model = package.model
        model.gate = threading.Event()
        model.entered.clear()
        model.calls.clear()
        return app, package

    return _busy


def _settled(app) -> bool:
    return app._model_slot.refs == 0 and not app._GEN_SEM.locked() and app._waiting == 0


def _tts_request(app, text):
    return app.TTSRequest(text=text, language="de")


def test_a_request_waits_at_most_the_queue_timeout_and_gives_everything_back(busy):
    app, package = busy(TTS_QUEUE_TIMEOUT_S="0.4", TTS_MAX_QUEUE="4")
    model = package.model

    async def main():
        first = asyncio.create_task(app.text_to_speech(_tts_request(app, "Erster.")))
        assert await asyncio.to_thread(model.entered.wait, 5), "no generation started"
        began = time.monotonic()
        with pytest.raises(app.HTTPException) as caught:
            await app.text_to_speech(_tts_request(app, "Zweiter."))
        waited = time.monotonic() - began
        waiting_after = app._waiting
        model.gate.set()
        await first
        assert await wait_until_async(lambda: _settled(app)), "a permit, a model reference or a wait was leaked"
        third = await app.text_to_speech(_tts_request(app, "Dritter."))
        return caught.value, waited, waiting_after, third

    error, waited, waiting_after, third = asyncio.run(main())

    assert error.status_code == 503
    assert int(error.headers["Retry-After"]) >= 1
    assert "TTS_QUEUE_TIMEOUT_S" in error.detail
    assert 0.3 < waited < 5
    assert waiting_after == 0
    assert third.status_code == 200
    assert [call["text"] for call in model.calls] == ["Erster.", "Dritter."], "the timed-out request was generated"
    assert app._GEN_SEM._value == 1


def test_requests_beyond_the_queue_depth_are_turned_away_at_once(busy):
    app, package = busy(TTS_QUEUE_TIMEOUT_S="30", TTS_MAX_QUEUE="1")
    model = package.model

    async def main():
        running = asyncio.create_task(app.text_to_speech(_tts_request(app, "Läuft.")))
        assert await asyncio.to_thread(model.entered.wait, 5)
        queued = asyncio.create_task(app.text_to_speech(_tts_request(app, "Wartet.")))
        assert await wait_until_async(lambda: app._waiting == 1)
        began = time.monotonic()
        with pytest.raises(app.HTTPException) as caught:
            await app.text_to_speech(_tts_request(app, "Zu viel."))
        turned_away_after = time.monotonic() - began
        model.gate.set()
        first, second = await running, await queued
        assert await wait_until_async(lambda: _settled(app))
        return caught.value, turned_away_after, first, second

    error, turned_away_after, first, second = asyncio.run(main())

    assert error.status_code == 503 and "TTS_MAX_QUEUE" in error.detail
    assert int(error.headers["Retry-After"]) >= 1
    assert turned_away_after < 1.0, "the excess request waited instead of being shed"
    assert first.status_code == 200 and second.status_code == 200, "queued requests within the depth must be served"
    assert [call["text"] for call in model.calls] == ["Läuft.", "Wartet."]


def test_a_queue_depth_of_zero_means_no_waiting_at_all(busy):
    app, package = busy(TTS_MAX_QUEUE="0", TTS_QUEUE_TIMEOUT_S="30")
    model = package.model

    async def main():
        running = asyncio.create_task(app.text_to_speech(_tts_request(app, "Läuft.")))
        assert await asyncio.to_thread(model.entered.wait, 5)
        with pytest.raises(app.HTTPException) as caught:
            await app.text_to_speech(_tts_request(app, "Nein."))
        model.gate.set()
        await running
        assert await wait_until_async(lambda: _settled(app))
        return caught.value

    assert asyncio.run(main()).status_code == 503


def test_a_full_queue_refuses_a_clone_before_its_upload_is_decoded(busy, monkeypatch, scratch):
    app, package = busy(TTS_MAX_QUEUE="0", TTS_QUEUE_TIMEOUT_S="30")
    decoded = []
    patch_decoder(monkeypatch, app, record=decoded)
    model = package.model

    async def main():
        running = asyncio.create_task(app.text_to_speech(_tts_request(app, "Läuft.")))
        assert await asyncio.to_thread(model.entered.wait, 5)
        with pytest.raises(app.HTTPException) as caught:
            await app.clone_voice(text="Klon.", lang="de", file=make_upload(), exaggeration=None, cfg_weight=None)
        model.gate.set()
        await running
        return caught.value

    error = asyncio.run(main())

    assert error.status_code == 503
    assert decoded == [], "the reference was decoded for a request that was going to be turned away"
    assert list(scratch.iterdir()) == []


def test_a_clone_that_times_out_in_the_queue_leaves_no_temp_file(busy, monkeypatch, scratch):
    app, package = busy(TTS_QUEUE_TIMEOUT_S="0.4", TTS_MAX_QUEUE="4")
    decoded = []
    patch_decoder(monkeypatch, app, record=decoded)
    model = package.model

    async def main():
        running = asyncio.create_task(app.text_to_speech(_tts_request(app, "Läuft.")))
        assert await asyncio.to_thread(model.entered.wait, 5)
        with pytest.raises(app.HTTPException) as caught:
            await app.clone_voice(text="Klon.", lang="de", file=make_upload(), exaggeration=None, cfg_weight=None)
        left_while_waiting = list(scratch.iterdir())
        model.gate.set()
        await running
        assert await wait_until_async(lambda: _settled(app))
        return caught.value, left_while_waiting

    error, left_while_waiting = asyncio.run(main())

    assert error.status_code == 503
    assert decoded, "the clone should have been staged before it waited"
    assert left_while_waiting == [] and list(scratch.iterdir()) == [], "the staged reference was not removed"


def test_a_stream_is_turned_away_before_it_takes_a_model_reference(busy):
    app, package = busy(TTS_MAX_QUEUE="0", TTS_QUEUE_TIMEOUT_S="30")
    model = package.model

    async def main():
        running = asyncio.create_task(app.text_to_speech(_tts_request(app, "Läuft.")))
        assert await asyncio.to_thread(model.entered.wait, 5)
        refs_before = app._model_slot.refs
        with pytest.raises(app.HTTPException) as caught:
            await app.text_to_speech_stream(_tts_request(app, "Ein Stream."))
        refs_after = app._model_slot.refs
        model.gate.set()
        await running
        assert await wait_until_async(lambda: _settled(app))
        return caught.value, refs_before, refs_after

    error, refs_before, refs_after = asyncio.run(main())

    assert error.status_code == 503
    assert refs_after == refs_before


def test_a_waiter_that_goes_away_leaves_no_trace(busy):
    app, package = busy(TTS_QUEUE_TIMEOUT_S="30", TTS_MAX_QUEUE="4")
    model = package.model

    async def main():
        running = asyncio.create_task(app.text_to_speech(_tts_request(app, "Läuft.")))
        assert await asyncio.to_thread(model.entered.wait, 5)
        gone = asyncio.create_task(app.text_to_speech(_tts_request(app, "Geht.")))
        assert await wait_until_async(lambda: app._waiting == 1)
        gone.cancel()  # the client disconnected while queued
        with pytest.raises(asyncio.CancelledError):
            await gone
        waiting_after = app._waiting
        model.gate.set()
        await running
        assert await wait_until_async(lambda: _settled(app))
        follow_up = await app.text_to_speech(_tts_request(app, "Danach."))
        return waiting_after, follow_up

    waiting_after, follow_up = asyncio.run(main())

    assert waiting_after == 0
    assert follow_up.status_code == 200
    assert app._GEN_SEM._value == 1
    assert [call["text"] for call in model.calls] == ["Läuft.", "Danach."]


def test_the_later_chunks_of_a_running_stream_may_join_a_full_queue(monkeypatch):
    """A stream that has begun is not cut off half way by the depth limit; a new request is."""
    install_chatterbox(monkeypatch)
    app = load_app(TTS_MAX_QUEUE="0", TTS_QUEUE_TIMEOUT_S="30")

    async def main():
        await app._GEN_SEM.acquire()  # someone else is generating
        with pytest.raises(app.HTTPException):
            await app._acquire_generation_slot()
        waiting = asyncio.create_task(app._acquire_generation_slot(continuation=True))
        await asyncio.sleep(0.05)
        still_waiting = not waiting.done()
        app._GEN_SEM.release()
        await asyncio.wait_for(waiting, 2)
        held = app._GEN_SEM.locked()
        app._GEN_SEM.release()
        return still_waiting, held, app._waiting

    still_waiting, held, waiting = asyncio.run(main())

    assert still_waiting and held and waiting == 0


def test_the_queue_limits_have_defaults_and_survive_junk(monkeypatch, caplog):
    install_chatterbox(monkeypatch)

    defaults = load_app()
    assert defaults.QUEUE_TIMEOUT_S == 60.0 and defaults.MAX_QUEUE == 4
    assert load_app(TTS_MAX_CONCURRENCY="3").MAX_QUEUE == 12, "the depth follows the concurrency"
    with caplog.at_level(logging.WARNING):
        junk = load_app(TTS_QUEUE_TIMEOUT_S="soon", TTS_MAX_QUEUE="-3")
    assert junk.QUEUE_TIMEOUT_S == 60.0 and junk.MAX_QUEUE == 4
    assert "TTS_QUEUE_TIMEOUT_S" in caplog.text and "TTS_MAX_QUEUE" in caplog.text


# --- E5: the default language is a language id --------------------------------------------

@pytest.mark.parametrize("configured, expected", [
    ("", "de"), ("   ", "de"), ("de", "de"), (" de ", "de"), ("DE", "de"), ("de-DE", "de"), ("de_AT", "de"),
    ("German", "de"), ("german", "de"), (" GERMAN ", "de"),
    ("en", "en"), ("English", "en"), ("en-GB", "en"), ("fr", "fr"), ("es_MX", "es"), ("Spanish", "es"),
])
def test_the_default_language_is_normalised_and_reaches_the_model_as_a_language_id(monkeypatch, configured, expected):
    package = install_chatterbox(monkeypatch)
    app = load_app(CHATTERBOX_DEFAULT_LANGUAGE=configured)
    patch_decoder(monkeypatch, app)
    client = TestClient(app.app)

    response = client.post("/tts", json={"text": "Hallo Welt.", "language": "auto"})
    cloned = client.post("/clone", data={"text": "Klon."},
                         files={"file": ("ref.wav", reference_clip(), "audio/wav")})

    assert app.DEFAULT_LANGUAGE == expected
    assert app._resolve_language("auto") == expected and app._resolve_language(None) == expected
    assert response.status_code == 200, response.text
    assert response.headers["X-Language"] == expected
    assert package.model.calls[0]["language_id"] == expected
    assert client.get("/languages").json()["default"] == expected
    assert cloned.status_code == 200, cloned.text
    assert package.model.calls[1]["language_id"] == expected, "/clone resolves 'auto' the same way"


@pytest.mark.parametrize("configured", ["klingon", "auto", "xx-YY", "de de", "12", "Deutsch"])
def test_an_unknown_default_language_falls_back_to_german_with_a_warning(monkeypatch, caplog, configured):
    package = install_chatterbox(monkeypatch)
    with caplog.at_level(logging.WARNING):
        app = load_app(CHATTERBOX_DEFAULT_LANGUAGE=configured)
    client = TestClient(app.app)

    response = client.post("/tts", json={"text": "Hallo Welt.", "language": "auto"})

    assert app.DEFAULT_LANGUAGE == "de"
    assert "CHATTERBOX_DEFAULT_LANGUAGE" in caplog.text and repr(configured) in caplog.text
    assert response.status_code == 200
    assert package.model.calls[0]["language_id"] == "de"


def test_an_unset_default_language_is_german_without_a_warning(monkeypatch, caplog):
    install_chatterbox(monkeypatch)
    with caplog.at_level(logging.WARNING):
        app = load_app()

    assert app.DEFAULT_LANGUAGE == "de"
    assert "CHATTERBOX_DEFAULT_LANGUAGE" not in caplog.text
