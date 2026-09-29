"""The built-in voice must survive clones, exaggeration overrides and failures.

The chatterbox library keeps the *voice* in ``model.conds`` on the one shared model
object: ``generate(audio_prompt_path=...)`` replaces it and a differing
``exaggeration`` rewrites it in place. Before the fix every ``/clone`` therefore
changed the voice of every later ``/tts`` and ``/tts-stream``, and a clone that
ran between two chunks of a stream changed the voice mid-sentence.

The model is a fake that follows those library semantics (tests/chatterbox_loader.py)
and answers with a constant equal to the voice that spoke.
"""

import asyncio
import threading

import numpy as np
import pytest
from fastapi.testclient import TestClient

from chatterbox_loader import (
    CLONE_VOICE, DEFAULT_VOICE, decode_audio, decode_pcm16, install_chatterbox, load_app,
    WatchedLock, make_upload, patch_decoder, reference_clip, voices_in, wait_until_async,
)

LONG_TEXT = " ".join(
    f"Das ist Satz Nummer {i} in einem etwas längeren deutschen Text über das Wetter." for i in range(1, 9)
)


@pytest.fixture
def svc(monkeypatch):
    package = install_chatterbox(monkeypatch)
    app = load_app()
    patch_decoder(monkeypatch, app)
    return app, package


def _tts(client, **body):
    body.setdefault("language", "de")
    return client.post("/tts", json=body)


def _clone(client, text="Hallo Welt.", **form):
    return client.post(
        "/clone",
        data={"text": text, "lang": "de", **form},
        files={"file": ("ref.wav", reference_clip(), "audio/wav")},
    )


def test_a_clone_does_not_change_the_voice_of_later_requests(svc):
    app, _package = svc
    client = TestClient(app.app)

    before = _tts(client, text="Hallo Welt.")
    cloned = _clone(client)
    after = _tts(client, text="Hallo Welt.")
    streamed = client.post("/tts-stream", json={"text": "Hallo Welt.", "language": "de"})

    assert voices_in(decode_audio(before.content)) == {DEFAULT_VOICE}
    assert voices_in(decode_audio(cloned.content)) == {CLONE_VOICE}
    assert voices_in(decode_audio(after.content)) == {DEFAULT_VOICE}, "/tts spoke in the cloned voice"
    assert voices_in(decode_pcm16(streamed.content)) == {round(DEFAULT_VOICE, 3)}, (
        "/tts-stream spoke in the cloned voice"
    )


def test_a_clone_between_two_stream_chunks_cannot_change_the_streamed_voice(svc):
    """/tts-stream lets go of the generation slot between chunks, so a /clone can run there."""
    app, package = svc
    TestClient(app.app).post("/tts", json={"text": "Wärmen.", "language": "de"})  # load the model

    async def main():
        response = await app.text_to_speech_stream(app.TTSRequest(text=LONG_TEXT, language="de"))
        stream = response.body_iterator
        await stream.__anext__()  # header
        await stream.__anext__()  # first chunk
        # The clone runs to completion while the stream is suspended between chunks.
        clone = await app.clone_voice(
            text="Zwischenruf.", lang="de", file=make_upload(), exaggeration=None, cfg_weight=None
        )
        assert voices_in(decode_audio(await _body(clone))) == {CLONE_VOICE}
        return [chunk async for chunk in stream]

    remaining = asyncio.run(main())
    assert remaining, "the stream should have had more than one chunk"

    stream_calls = [c for c in package.model.calls if c["audio_prompt_path"] is None and c["text"] != "Wärmen."]
    assert len(stream_calls) > 1
    assert {c["voice"] for c in stream_calls} == {DEFAULT_VOICE}, "a stream chunk was spoken in the cloned voice"


async def _body(response) -> bytes:
    """Read a StreamingResponse produced by a handler called directly."""
    return b"".join([chunk async for chunk in response.body_iterator])


def test_an_exaggeration_override_does_not_leak_into_the_next_request(svc):
    app, package = svc
    client = TestClient(app.app)

    _tts(client, text="Erster Satz.", exaggeration=1.5)
    _tts(client, text="Zweiter Satz.")

    first, second = package.model.calls[-2:]
    assert first["emotion"] == 1.5
    assert second["emotion"] == 0.5
    assert second["voice"] == DEFAULT_VOICE


def test_a_failed_generation_still_restores_the_built_in_voice(svc):
    app, package = svc
    client = TestClient(app.app)

    failed = _clone(client, text="BOOM beim Klonen.")
    assert failed.status_code == 500
    after = _tts(client, text="Hallo Welt.")

    assert voices_in(decode_audio(after.content)) == {DEFAULT_VOICE}


def test_default_voice_requests_work_from_a_fresh_copy_of_the_voice(svc):
    """The snapshot itself must never be handed to the library (it rewrites conds in place)."""
    app, package = svc
    client = TestClient(app.app)
    _tts(client, text="Hallo.", exaggeration=1.9)

    # If the snapshot had been handed out, the library's in-place rewrite would
    # have changed what every later request starts from.
    assert app._default_conds.t3.emotion_adv == 0.5
    assert app._default_conds.t3 is not package.model.conds.t3


def test_a_checkpoint_without_a_built_in_voice_explains_itself_and_clone_still_works(monkeypatch):
    package = install_chatterbox(monkeypatch, conds=None)
    app = load_app()
    patch_decoder(monkeypatch, app)
    client = TestClient(app.app)

    plain = _tts(client, text="Hallo Welt.")
    assert plain.status_code == 500
    assert "/clone" in plain.json()["detail"]

    cloned = _clone(client)
    assert cloned.status_code == 200
    assert voices_in(decode_audio(cloned.content)) == {CLONE_VOICE}

    # The clone must not turn into the new built-in voice.
    again = _tts(client, text="Hallo Welt.")
    assert again.status_code == 500, "the clone persisted as the default voice"


def test_generations_never_overlap_on_the_shared_model_even_with_concurrency_above_one(monkeypatch):
    """`model.conds` is instance state: TTS_MAX_CONCURRENCY=2 must not let a /tts and a
    /clone run at once, or the /tts speaks in the clone's voice."""
    package = install_chatterbox(monkeypatch)
    app = load_app(TTS_MAX_CONCURRENCY="2")
    patch_decoder(monkeypatch, app)
    TestClient(app.app).post("/tts", json={"text": "Wärmen.", "language": "de"})  # load the model
    model = package.model
    model.gate = threading.Event()
    model.entered.clear()
    model_lock = WatchedLock()
    monkeypatch.setattr(app, "_MODEL_LOCK", model_lock)

    async def main():
        tts = asyncio.create_task(app.text_to_speech(app.TTSRequest(text="Hallo Welt.", language="de")))
        assert await asyncio.to_thread(model.entered.wait, 5), "no generation started"
        clone = asyncio.create_task(app.clone_voice(
            text="Klon.", lang="de", file=make_upload(), exaggeration=None, cfg_weight=None
        ))
        # Either the clone's worker parks on the model lock (right) or a (buggy) second
        # generation starts next to the first. Wait for whichever comes first, not for a
        # fixed 0.4 s that also passes when the machine is too busy to start one.
        assert await wait_until_async(lambda: model_lock.contended.is_set() or model.running > 1), (
            "the clone neither waited for the model nor started")
        overlapped = model.max_running
        parked = model_lock.contended.is_set()
        model.gate.set()
        return overlapped, parked, await tts, await clone

    overlapped, parked, tts_response, clone_response = asyncio.run(main())

    assert parked, "the clone did not wait for the model lock the running generation holds"
    assert overlapped == 1, "two generate() calls ran on the shared model at the same time"
    assert voices_in(decode_audio(_first_body(tts_response))) == {DEFAULT_VOICE}
    assert voices_in(decode_audio(_first_body(clone_response))) == {CLONE_VOICE}


def _first_body(response) -> bytes:
    """Bytes of a StreamingResponse built from a BytesIO (the non-streaming endpoints)."""
    async def read():
        return await _body(response)
    return asyncio.run(read())
