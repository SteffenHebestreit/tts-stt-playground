"""/tts-stream and /tts must never leak the model reference or the generation slot.

Two ways the old code leaked:

* The reference was released in the streaming generator's ``finally``. A generator
  that never starts (the client is already gone when Starlette begins to stream)
  never runs its ``finally``: the reference stayed at 1 forever, so the idle TTL
  never fired and ``POST /unload`` answered 409 for the life of the process.
* On cancellation ``asyncio.to_thread`` abandons the awaiting coroutine but not the
  worker thread. The generation slot and the reference were released at once while
  the thread was still inside ``generate()``: a second generation could start next
  to it, and ``/unload`` could free weights that were still being read.

The model is a fake with a gate: ``generate()`` blocks on it after reading the voice,
which is the window a worker thread spends inside the real library.
"""

import asyncio
import gc
import threading

import pytest
from fastapi.testclient import TestClient

from chatterbox_loader import (
    DEFAULT_VOICE, install_chatterbox, load_app, make_upload, patch_decoder, wait_until, wait_until_async,
)

TEXT = " ".join(
    f"Das ist Satz Nummer {i} in einem etwas längeren deutschen Text über das Wetter." for i in range(1, 7)
)


@pytest.fixture
def svc(monkeypatch):
    package = install_chatterbox(monkeypatch)
    app = load_app()
    patch_decoder(monkeypatch, app)
    TestClient(app.app).post("/tts", json={"text": "Wärmen.", "language": "de"})  # load the model
    assert package.loads == 1
    package.model.calls.clear()
    return app, package


def _request(app, text=TEXT):
    return app.TTSRequest(text=text, language="de")


def _refs(app) -> int:
    return app._model_slot.refs


def _all_released(app) -> bool:
    return _refs(app) == 0 and not app._GEN_SEM.locked()


def test_a_stream_that_runs_to_the_end_gives_everything_back(svc):
    app, package = svc

    async def main():
        response = await app.text_to_speech_stream(_request(app))
        chunks = [chunk async for chunk in response.body_iterator]
        await response.background()
        return chunks

    chunks = asyncio.run(main())

    assert len(chunks) > 2
    assert wait_until(lambda: _refs(app) == 0), f"{_refs(app)} reference(s) still held"
    assert not app._GEN_SEM.locked()


def test_a_generator_that_is_never_iterated_does_not_pin_the_model(svc):
    """The response is dropped without ever being sent (client gone before the first byte)."""
    app, _package = svc

    async def main():
        response = await app.text_to_speech_stream(_request(app))
        assert _refs(app) == 1
        del response
        gc.collect()

    asyncio.run(main())

    assert wait_until(lambda: _refs(app) == 0), "an unstarted stream kept the model pinned forever"
    assert app._model_slot.try_unload()["reason"] == "ok"


def test_the_response_background_task_releases_when_nothing_was_streamed(svc):
    app, _package = svc

    async def main():
        response = await app.text_to_speech_stream(_request(app))
        assert _refs(app) == 1
        await response.background()  # what Starlette runs after the response, sent or not
        assert wait_until(lambda: _refs(app) == 0), "the background task did not release the reference"
        return response

    response = asyncio.run(main())

    async def close_late():
        # The generator's own release path is idempotent: closing later cannot double-release.
        await response.body_iterator.aclose()

    asyncio.run(close_late())
    assert _refs(app) == 0


def test_a_client_that_is_already_gone_when_streaming_starts_releases_through_starlette(svc):
    """Drive the real StreamingResponse with a `receive` that reports a disconnect at once,
    the shape uvicorn presents when the peer closed before the response was sent."""
    app, _package = svc

    async def main():
        response = await app.text_to_speech_stream(_request(app))
        sent = []

        async def receive():
            return {"type": "http.disconnect"}

        async def send(message):
            sent.append(message)

        await response({"type": "http", "asgi": {"spec_version": "2.3"}}, receive, send)
        del response
        gc.collect()

    asyncio.run(main())

    assert wait_until(lambda: _refs(app) == 0), "a disconnected client left the model pinned"


def test_closing_the_stream_early_releases_the_reference(svc):
    app, _package = svc

    async def main():
        response = await app.text_to_speech_stream(_request(app))
        stream = response.body_iterator
        await stream.__anext__()  # header
        await stream.__anext__()  # first audio
        await stream.aclose()  # client disconnected mid-stream

    asyncio.run(main())

    assert wait_until(lambda: _refs(app) == 0)
    assert not app._GEN_SEM.locked()


def test_a_failure_before_the_first_byte_is_an_http_error_and_releases(svc):
    app, package = svc

    async def main():
        with pytest.raises(app.HTTPException) as caught:
            await app.text_to_speech_stream(_request(app, text="BOOM gleich zu Beginn."))
        return caught.value

    error = asyncio.run(main())

    assert error.status_code == 500
    assert _refs(app) == 0
    assert not app._GEN_SEM.locked()


def test_ttl_zero_unloads_after_a_stream_and_not_before(monkeypatch):
    """The end to end symptom of a leaked reference: with TTL 0 the model never unloads."""
    install_chatterbox(monkeypatch)
    app = load_app(TTS_MODEL_TTL="0")

    async def main():
        response = await app.text_to_speech_stream(_request(app))
        stream = response.body_iterator
        await stream.__anext__()
        await stream.__anext__()
        resident_mid_stream = app._model_slot.resident
        await stream.aclose()
        return resident_mid_stream

    resident_mid_stream = asyncio.run(main())

    assert resident_mid_stream, "the model was unloaded while the stream was still open"
    assert wait_until(lambda: not app._model_slot.resident), "the model stayed resident after the stream closed"


# --- The worker thread outlives the request ----------------------------------

def _block_next_generation(package):
    package.model.gate = threading.Event()
    package.model.entered.clear()
    return package.model.gate


def test_cancelling_a_stream_mid_chunk_keeps_slot_and_model_until_the_worker_is_done(svc):
    app, package = svc
    model = package.model

    async def main():
        response = await app.text_to_speech_stream(_request(app))
        stream = response.body_iterator
        await stream.__anext__()  # header
        await stream.__anext__()  # first chunk (already generated)
        gate = _block_next_generation(package)
        pending = asyncio.ensure_future(stream.__anext__())  # chunk 2 starts in a worker thread
        assert await asyncio.to_thread(model.entered.wait, 5), "chunk 2 never started"

        pending.cancel()  # the client disconnects: Starlette cancels the streaming task
        with pytest.raises(asyncio.CancelledError):
            await pending
        await stream.aclose()

        # The thread is still inside generate(): nothing it uses may have been given back.
        assert _refs(app) >= 1, "the model reference was released under a running thread"
        assert app._GEN_SEM.locked(), "the generation slot was released under a running thread"
        busy = await app.unload()
        assert busy.status_code == 409, "/unload freed weights a worker thread is still reading"

        # A new request must wait for the slot instead of starting a second generation.
        calls_before = len(model.calls)
        second = asyncio.ensure_future(app.text_to_speech(_request(app, text="Zweite Anfrage.")))
        # Either the second request parks for the slot (right) or it starts generating
        # next to the orphan (the bug). Wait for whichever happens first instead of for a
        # fixed 0.4 s, which also "passes" when the machine is too busy to start a second
        # generation in that time.
        assert await wait_until_async(lambda: app._waiting == 1 or model.running > 1), (
            "the second request neither queued for the slot nor started")
        queued = app._waiting == 1
        overlapped = model.max_running
        started_early = len(model.calls) > calls_before

        gate.set()
        await second
        # Release runs on the loop when the worker ends; do not let the loop close first.
        await wait_until_async(lambda: _all_released(app))
        return overlapped, started_early, queued

    overlapped, started_early, queued = asyncio.run(main())

    assert queued, "the second request did not wait for the slot the orphaned worker still holds"
    assert overlapped == 1 and not started_early, "a second generation ran next to the orphaned one"
    assert _all_released(app), "the reference or the slot leaked after the worker finished"


def test_cancelling_tts_stops_after_the_running_chunk_and_releases_afterwards(svc):
    app, package = svc
    model = package.model

    async def main():
        gate = _block_next_generation(package)
        task = asyncio.ensure_future(app.text_to_speech(_request(app)))
        assert await asyncio.to_thread(model.entered.wait, 5)

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        still_held = (_refs(app) >= 1, app._GEN_SEM.locked())

        gate.set()
        await wait_until_async(lambda: _all_released(app))
        return still_held

    still_held = asyncio.run(main())

    assert still_held == (True, True), "resources were released while the worker was still generating"
    assert _all_released(app)
    # The chunk that was running finishes; the rest of the text is not generated for nobody.
    assert len(package.model.calls) == 1


def test_a_cancelled_request_waiting_for_the_slot_takes_nothing(svc):
    app, package = svc
    model = package.model

    async def main():
        gate = _block_next_generation(package)
        running = asyncio.ensure_future(app.text_to_speech(_request(app, text="Läuft.")))
        assert await asyncio.to_thread(model.entered.wait, 5)

        waiting = asyncio.ensure_future(app.text_to_speech(_request(app, text="Wartet.")))
        await asyncio.sleep(0.05)
        waiting.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiting

        gate.set()
        await running
        await wait_until_async(lambda: _all_released(app))

    asyncio.run(main())

    assert _all_released(app)
    assert [c["text"] for c in package.model.calls] == ["Läuft."]


def test_a_failed_worker_still_gives_everything_back(svc):
    app, _package = svc

    async def main():
        with pytest.raises(app.HTTPException):
            await app.text_to_speech(_request(app, text="BOOM."))
        await wait_until_async(lambda: _all_released(app))

    asyncio.run(main())

    assert _all_released(app)


def test_concurrent_requests_each_get_their_own_answer_and_leave_nothing_behind(svc):
    app, package = svc

    async def main():
        clone_task = app.clone_voice(text="Klon.", lang="de", file=make_upload(), exaggeration=None, cfg_weight=None)
        return await asyncio.gather(
            app.text_to_speech(_request(app, text="Eins.")),
            clone_task,
            app.text_to_speech(_request(app, text="Drei.")),
        )

    responses = asyncio.run(main())

    assert len(responses) == 3
    assert package.model.max_running == 1
    assert wait_until(lambda: _all_released(app))
    assert {c["voice"] for c in package.model.calls if c["text"] in ("Eins.", "Drei.")} == {DEFAULT_VOICE}


def test_a_model_that_cannot_load_is_an_http_error_for_the_stream_too(monkeypatch):
    install_chatterbox(monkeypatch)
    app = load_app()

    def failing_load():
        raise OSError("no space left for the checkpoint")

    app._model_slot = app.ModelSlot(
        failing_load, ttl_seconds=-1, name="Chatterbox Multilingual", on_unload=app._forget_chatterbox
    )

    async def main():
        with pytest.raises(app.HTTPException) as caught:
            await app.text_to_speech_stream(_request(app))
        return caught.value

    error = asyncio.run(main())

    assert error.status_code == 500 and "no space left" not in error.detail
    assert error.headers["X-Request-ID"] in error.detail
    assert not app._GEN_SEM.locked()
