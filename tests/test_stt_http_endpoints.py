"""HTTP behaviour of stt-service: upload limits, temp-file hygiene, JSON safety,
readiness, language handling, and how the streaming route holds the model.

Runs the real app under stubs (see test_stt_support.py) with a fake model.
"""

import asyncio
import io
import os
import tempfile
import threading
import time

import pytest
from fastapi.testclient import TestClient
from starlette.datastructures import UploadFile

from test_stt_support import (
    FakeBatchedPipeline, FakeWhisper, Info, LiveSession, Segment, load_stt_app,
    reset_model, wait_for_preload, wait_until,
)

WAV = b"RIFF" + b"\x00" * 60


@pytest.fixture(scope="module")
def stt_app():
    return load_stt_app({}, name="stt_app_http")


@pytest.fixture(scope="module")
def _client(stt_app):
    with TestClient(stt_app.app) as test_client:
        wait_for_preload(stt_app)
        yield test_client


@pytest.fixture
def scratch(tmp_path, monkeypatch):
    """Route every temp file of the request into a directory the test can inspect."""
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    return tmp_path


@pytest.fixture
def client(stt_app, _client, scratch):
    stt_app._idle_unloader.cancel()
    model = reset_model(stt_app)
    model.default = [Segment("hallo welt", start=0.0, end=1.5)]
    model.info = Info("de", 0.97, duration=2.0)
    _client.fake_model = model
    yield _client
    assert wait_until(lambda: stt_app._model_refs == 0), f"leaked a model reference: {stt_app._model_refs}"


def _post(client, path="/transcribe", data=None, files=None):
    return client.post(path, data=data or {}, files=files or {"audio": ("a.wav", WAV, "audio/wav")})


def _leftovers(scratch):
    return sorted(p.name for p in scratch.iterdir())


# --- /ready ------------------------------------------------------------------


def test_ready_is_200_when_the_model_is_resident(client):
    response = client.get("/ready")
    assert response.status_code == 200
    assert response.json()["ready"] is True
    assert response.json()["reason"] == "resident"


def test_ready_is_200_for_a_model_that_was_unloaded_and_can_reload(client, stt_app):
    stt_app.whisper_model = None
    stt_app.model_loaded = False
    response = client.get("/ready")
    assert response.status_code == 200
    assert response.json()["reason"] == "unloaded"
    assert response.json()["model_resident"] is False


def test_ready_is_503_after_every_load_attempt_failed(client, stt_app):
    stt_app.whisper_model = None
    stt_app.model_loaded = False
    stt_app.startup_error = "no such model"
    response = client.get("/ready")
    assert response.status_code == 503
    body = response.json()
    assert body["ready"] is False and body["reason"] == "load_failed"
    assert body["detail"] == "no such model"


def test_ready_is_503_while_a_load_is_running(client, stt_app, monkeypatch):
    monkeypatch.setattr(stt_app, "_loading", True)
    stt_app.whisper_model = None
    response = client.get("/ready")
    assert response.status_code == 503
    assert response.json()["reason"] == "loading"
    assert response.headers["retry-after"]


def test_ready_is_503_until_the_startup_preload_has_finished(client, stt_app, monkeypatch):
    monkeypatch.setattr(stt_app, "_preload_pending", True)
    stt_app.whisper_model = None
    assert client.get("/ready").status_code == 503
    stt_app.whisper_model = client.fake_model
    assert client.get("/ready").status_code == 200, "a resident model is ready whatever the preload says"


def test_health_and_ready_never_load_the_model(client, stt_app, monkeypatch):
    """Polled on a timer: a poll that loaded the model would defeat the idle TTL."""
    constructed = []

    class Counting(FakeWhisper):
        def __init__(self, *args, **kwargs):
            constructed.append(1)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(stt_app, "WhisperModel", Counting)
    stt_app.whisper_model = None
    stt_app.model_loaded = False
    for _ in range(3):
        assert client.get("/health").status_code == 200
        assert client.get("/ready").status_code == 200
    assert constructed == [] and stt_app.whisper_model is None and stt_app._model_refs == 0


# --- /transcribe: JSON safety, cleanup, limits (L1, H3) ------------------------


def test_a_nan_logprob_is_null_not_a_500_after_the_whole_decode(client):
    client.fake_model.default = [Segment("hallo", avg_logprob=float("nan"), start=0.0, end=1.0)]
    response = _post(client)
    assert response.status_code == 200, response.text
    assert response.json()["segments"][0]["avg_logprob"] is None


def test_a_nan_logprob_in_a_batch_is_null_too(client):
    client.fake_model.default = [Segment("hallo", avg_logprob=float("-inf"), start=0.0, end=1.0)]
    response = client.post("/transcribe", files=[
        ("audios", ("a.wav", WAV, "audio/wav")), ("audios", ("b.wav", WAV, "audio/wav"))])
    assert response.status_code == 200, response.text
    assert response.json()["results"][0]["segments"][0]["avg_logprob"] is None


def test_a_model_that_cannot_load_answers_503_and_leaves_no_upload_behind(client, stt_app, scratch, monkeypatch):
    class Broken:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("weights are corrupt")

    monkeypatch.setattr(stt_app, "WhisperModel", Broken)
    stt_app.whisper_model = None
    stt_app.model_loaded = False
    response = _post(client)
    assert response.status_code == 503
    assert "weights are corrupt" in response.json()["detail"]
    assert _leftovers(scratch) == [], "the upload of a refused request was left on disk"
    stt_app.startup_error = None


def test_a_failed_decode_leaves_no_upload_behind(client, scratch):
    client.fake_model.fail_with = RuntimeError("decoder exploded")
    assert _post(client).status_code == 500
    assert _leftovers(scratch) == []


def test_a_successful_request_leaves_nothing_behind(client, scratch):
    assert _post(client).status_code == 200
    assert _leftovers(scratch) == []


def test_an_oversized_upload_is_413_and_never_reaches_the_model(client, stt_app, scratch, monkeypatch):
    monkeypatch.setattr(stt_app, "MAX_UPLOAD_BYTES", 1024)
    response = _post(client, files={"audio": ("big.wav", b"\x00" * 4096, "audio/wav")})
    assert response.status_code == 413
    assert "MAX_UPLOAD_MB" in response.json()["detail"]
    assert client.fake_model.calls == []
    assert _leftovers(scratch) == []


def test_an_upload_exactly_at_the_limit_is_accepted(client, stt_app, monkeypatch):
    monkeypatch.setattr(stt_app, "MAX_UPLOAD_BYTES", 1024)
    assert _post(client, files={"audio": ("ok.wav", b"\x00" * 1024, "audio/wav")}).status_code == 200


def test_a_limit_of_zero_disables_the_upload_cap(client, stt_app, monkeypatch):
    monkeypatch.setattr(stt_app, "MAX_UPLOAD_BYTES", 0)
    assert _post(client, files={"audio": ("big.wav", b"\x00" * 4096, "audio/wav")}).status_code == 200


def test_over_long_audio_is_413_before_the_model_is_touched(client, stt_app, monkeypatch):
    monkeypatch.setattr(stt_app, "_probe_duration_s", lambda path: 3 * 3600.0)
    response = _post(client)
    assert response.status_code == 413
    assert "MAX_AUDIO_SECONDS" in response.json()["detail"]
    assert client.fake_model.calls == []


def test_over_long_audio_without_a_header_duration_is_caught_after_decoding(client, stt_app, monkeypatch):
    monkeypatch.setattr(stt_app, "_probe_duration_s", lambda path: None)
    client.fake_model.info = Info("de", 0.97, duration=9000.0)
    response = _post(client)
    assert response.status_code == 413


def test_audio_within_the_duration_cap_is_transcribed(client, stt_app, monkeypatch):
    monkeypatch.setattr(stt_app, "_probe_duration_s", lambda path: 60.0)
    assert _post(client).status_code == 200


# --- language handling on the HTTP routes ------------------------------------


def _language_of_last_call(client):
    return client.fake_model.calls[-1].get("language")


def test_a_locale_tag_is_reduced_to_the_language_code_on_transcribe(client):
    assert _post(client, data={"language": "de-DE"}).status_code == 200
    assert _language_of_last_call(client) == "de"


def test_an_unknown_language_is_a_400_before_any_decode(client):
    response = _post(client, data={"language": "klingon"})
    assert response.status_code == 400
    assert "klingon" in response.json()["detail"]
    assert client.fake_model.calls == []


def test_omitted_and_auto_language_both_auto_detect_by_default(client):
    assert _post(client).status_code == 200
    assert _language_of_last_call(client) is None
    assert _post(client, data={"language": "auto"}).status_code == 200
    assert _language_of_last_call(client) is None


def test_the_default_language_fills_in_only_what_the_request_omits():
    app = load_stt_app({"STT_DEFAULT_LANGUAGE": "de"}, name="stt_app_http_default_lang")
    with TestClient(app.app) as test_client:
        wait_for_preload(app)
        model = reset_model(app)
        model.default = [Segment("hallo")]

        assert _post(test_client).status_code == 200
        assert model.calls[-1]["language"] == "de"
        assert _post(test_client, data={"language": "en"}).status_code == 200
        assert model.calls[-1]["language"] == "en"
        assert _post(test_client, data={"language": "auto"}).status_code == 200
        assert model.calls[-1]["language"] is None, "an explicit auto must not be overridden"

        assert test_client.post(
            "/transcribe-stream", files={"audio": ("a.wav", WAV, "audio/wav")}
        ).status_code == 200
        assert model.calls[-1]["language"] == "de"


def test_an_unsupported_default_language_is_ignored_not_fatal():
    app = load_stt_app({"STT_DEFAULT_LANGUAGE": "klingon"}, name="stt_app_http_bad_default")
    assert app.STT_DEFAULT_LANGUAGE is None


def test_an_unknown_task_is_a_400_not_a_500_mid_decode(client):
    for path in ("/transcribe", "/transcribe-stream"):
        response = _post(client, path=path, data={"task": "summarise"})
        assert response.status_code == 400, path
        assert "summarise" in response.json()["detail"]
    assert client.fake_model.calls == []


# --- translate is judged by model name (I1) ----------------------------------


@pytest.mark.parametrize("model,allowed", [
    ("large-v3", True),
    ("large-v3-turbo", False),
    ("turbo", False),
    ("small", True),
    ("distil-large-v3", False),
    ("primeline/whisper-large-v3-turbo-german", False),
    ("/app/models/whisper-large-v3-turbo-german-ct2", False),
    ("org/whisper-large-v3-german-ct2", True),
    ("/app/models/my-small-model", True),
])
def test_translate_is_rejected_for_models_that_cannot_do_it(stt_app, monkeypatch, model, allowed):
    from fastapi import HTTPException

    monkeypatch.setattr(stt_app, "model_size_loaded", model)
    if allowed:
        stt_app._reject_unsupported_translate("translate")
    else:
        with pytest.raises(HTTPException) as excinfo:
            stt_app._reject_unsupported_translate("translate")
        assert excinfo.value.status_code == 400
        assert model in excinfo.value.detail
    stt_app._reject_unsupported_translate("transcribe")   # never affected


def test_translate_on_a_turbo_style_custom_model_is_a_400_over_http(client, stt_app, monkeypatch):
    monkeypatch.setattr(stt_app, "model_size_loaded", "primeline/whisper-large-v3-turbo-german")
    response = _post(client, data={"task": "translate"})
    assert response.status_code == 400
    assert "does not support translation" in response.json()["detail"]
    assert client.fake_model.calls == []


def test_the_rejection_names_the_configured_model_before_anything_is_loaded(stt_app, monkeypatch):
    from fastapi import HTTPException

    monkeypatch.setattr(stt_app, "model_size_loaded", None)
    monkeypatch.setenv("WHISPER_MODEL_SIZE", "large-v3-turbo")
    with pytest.raises(HTTPException) as excinfo:
        stt_app._reject_unsupported_translate("translate")
    assert "'large-v3-turbo'" in excinfo.value.detail and "None" not in excinfo.value.detail


# --- worker pool and the opt-in batched pipeline (H3, D1) --------------------


def test_the_batch_pool_is_bounded_by_default_and_by_env():
    default = load_stt_app({}, name="stt_app_pool_default")
    assert default.executor._max_workers == max(1, min(4, os.cpu_count() or 4))
    assert load_stt_app({"STT_BATCH_WORKERS": "2"}, name="stt_app_pool_two").executor._max_workers == 2
    assert load_stt_app({"STT_BATCH_WORKERS": "0"}, name="stt_app_pool_zero").executor._max_workers == 1
    assert load_stt_app({"STT_BATCH_WORKERS": "many"}, name="stt_app_pool_bad").executor._max_workers >= 1


def test_the_batched_pipeline_is_off_unless_asked_for(client):
    FakeBatchedPipeline.instances.clear()
    assert _post(client).status_code == 200
    assert len(client.fake_model.calls) == 1
    assert FakeBatchedPipeline.instances == []


def test_the_batched_pipeline_serves_files_when_enabled_but_never_the_live_path():
    app = load_stt_app(
        {"STT_BATCHED_INFERENCE": "true", "STT_BATCH_SIZE": "3", "WS_MIN_NEW_AUDIO_S": "0.1"},
        name="stt_app_batched",
    )
    with TestClient(app.app) as test_client:
        wait_for_preload(app)
        model = reset_model(app)
        model.default = [Segment("live text")]
        FakeBatchedPipeline.instances.clear()

        response = _post(test_client)
        assert response.status_code == 200
        assert response.json()["text"] == "batched text"
        pipeline = FakeBatchedPipeline.instances[-1]
        assert pipeline.calls[0]["batch_size"] == 3
        assert pipeline.calls[0]["vad_filter"] is True
        assert model.calls == [], "the file went through the plain path as well"

        # Batched decoding needs VAD to chunk long audio; without it the plain path runs.
        assert _post(test_client, data={"vad_filter": "false"}).status_code == 200
        assert len(model.calls) == 1 and len(FakeBatchedPipeline.instances) == 1

        FakeBatchedPipeline.instances.clear()
        with LiveSession(test_client, {"language": "de"}) as session:
            session.send_audio(0.3)
            session.wait_type("partial")
        assert FakeBatchedPipeline.instances == [], "the WebSocket decoder must never use the batched pipeline"


# --- /detect_language --------------------------------------------------------


def test_detect_language_enforces_the_upload_cap_and_cleans_up(client, stt_app, scratch, monkeypatch):
    monkeypatch.setattr(stt_app, "MAX_UPLOAD_BYTES", 512)
    response = client.post("/detect_language", files={"file": ("big.wav", b"\x00" * 2048, "audio/wav")})
    assert response.status_code == 413
    assert _leftovers(scratch) == []


def test_detect_language_leaves_nothing_behind(client, scratch):
    response = client.post("/detect_language", files={"file": ("a.wav", WAV, "audio/wav")})
    assert response.status_code == 200
    assert response.json()["detected_language"] == "de"
    assert _leftovers(scratch) == []


# --- /transcribe-stream ------------------------------------------------------


def test_the_stream_delivers_segments_and_releases_everything(client, stt_app, scratch):
    client.fake_model.default = [Segment("erster satz", start=0.0, end=1.0), Segment("zweiter satz", start=1.0, end=2.0)]
    response = client.post("/transcribe-stream", files={"audio": ("a.wav", WAV, "audio/wav")})
    assert response.status_code == 200
    body = response.text
    assert "erster satz" in body and "zweiter satz" in body and '"status":"completed"' in body.replace(" ", "")
    assert wait_until(lambda: stt_app._model_refs == 0)
    assert _leftovers(scratch) == []


def test_the_stream_answers_503_for_a_model_that_cannot_load(client, stt_app, scratch, monkeypatch):
    class Broken:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("weights are corrupt")

    monkeypatch.setattr(stt_app, "WhisperModel", Broken)
    stt_app.whisper_model = None
    stt_app.model_loaded = False
    response = client.post("/transcribe-stream", files={"audio": ("a.wav", WAV, "audio/wav")})
    assert response.status_code == 503, "a load failure must be a real status, not an in-band event"
    assert _leftovers(scratch) == []
    stt_app.startup_error = None


def test_the_stream_enforces_the_upload_cap(client, stt_app, scratch, monkeypatch):
    monkeypatch.setattr(stt_app, "MAX_UPLOAD_BYTES", 1024)
    response = client.post("/transcribe-stream", files={"audio": ("big.wav", b"\x00" * 4096, "audio/wav")})
    assert response.status_code == 413
    assert _leftovers(scratch) == []


def test_the_stream_enforces_the_duration_cap(client, stt_app, scratch, monkeypatch):
    monkeypatch.setattr(stt_app, "_probe_duration_s", lambda path: 5 * 3600.0)
    response = client.post("/transcribe-stream", files={"audio": ("a.wav", WAV, "audio/wav")})
    assert response.status_code == 413
    assert _leftovers(scratch) == []
    assert client.fake_model.calls == []


def test_a_model_loaded_for_the_stream_is_loaded_once_under_ttl_zero():
    """A probe that acquires and releases unloads the model at once under TTL=0,
    and the generator then loaded it a second time."""
    constructed = []

    class Counting(FakeWhisper):
        def __init__(self, *args, **kwargs):
            constructed.append(1)
            super().__init__(*args, **kwargs)
            self.default = [Segment("guten tag", start=0.0, end=1.0)]

    app = load_stt_app({"STT_MODEL_TTL": "0"}, name="stt_app_http_ttl0", model_cls=Counting)
    with TestClient(app.app) as test_client:
        wait_for_preload(app)
        response = test_client.post("/transcribe-stream", files={"audio": ("a.wav", WAV, "audio/wav")})
        assert response.status_code == 200 and "guten tag" in response.text
        assert wait_until(lambda: app.whisper_model is None), "the idle model was not released"
    assert constructed == [1], f"loaded {len(constructed)} times for one stream"


# --- the stream's model reference under disconnects (L1) ---------------------


def _upload(data=WAV, name="a.wav"):
    return UploadFile(file=io.BytesIO(data), filename=name)


async def _open_stream(app):
    return await app.transcribe_audio_stream(
        audio=_upload(), language=None, task="transcribe", target_language="english",
        beam_size=5, vad_filter=True, vad_threshold=0.5, no_speech_threshold=0.6,
    )


async def _eventually(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.01)
    return predicate()


class _BlockingSegments:
    """A segment iterator whose next() blocks in a worker thread until released."""

    def __init__(self):
        self.inside = threading.Event()
        self.release = threading.Event()

    def __iter__(self):
        return self

    def __next__(self):
        self.inside.set()
        self.release.wait(10)
        raise StopIteration


def test_the_reference_is_released_only_after_the_worker_thread_has_finished(stt_app, scratch):
    """Cancelling the await does not stop the thread inside next(). Releasing at
    once let STT_MODEL_TTL=0 unload the model, and the next request load a second
    copy, while that thread was still decoding on the first."""
    stt_app._idle_unloader.cancel()
    model = reset_model(stt_app)
    blocking = _BlockingSegments()
    model.transcribe = lambda audio, **kw: (blocking, Info("de", 0.9, 2.0))

    async def scenario():
        response = await _open_stream(stt_app)
        assert stt_app._model_refs == 1
        events = []

        async def pump():
            async for chunk in response.body_iterator:
                events.append(chunk)

        task = asyncio.create_task(pump())
        assert await _eventually(blocking.inside.is_set), "the worker never reached next()"
        task.cancel()                              # the client went away
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.sleep(0.15)
        assert stt_app._model_refs == 1, "the reference was released while a worker thread was still using the model"
        blocking.release.set()
        assert await _eventually(lambda: stt_app._model_refs == 0), "the reference was never released"

    try:
        asyncio.run(scenario())
    finally:
        blocking.release.set()
    assert _leftovers(scratch) == []


def test_a_reference_taken_for_a_stream_that_never_starts_is_given_back(stt_app, scratch):
    """The handler takes the reference so a load failure can be a 503. If the client
    is gone before the body is iterated the generator never runs its finally, and
    without a backstop /unload would answer 409 until the process restarts."""
    stt_app._idle_unloader.cancel()
    reset_model(stt_app)

    async def scenario():
        response = await _open_stream(stt_app)
        assert stt_app._model_refs == 1

        async def receive():
            await asyncio.sleep(3600)

        async def send(message):
            raise OSError("client already gone")

        scope = {"type": "http", "asgi": {"spec_version": "2.5"}, "method": "POST", "path": "/transcribe-stream"}
        with pytest.raises(Exception):
            await response(scope, receive, send)
        assert await _eventually(lambda: stt_app._model_refs == 0), "the reference leaked"

    asyncio.run(scenario())
    assert _leftovers(scratch) == []


def test_a_disconnect_mid_stream_releases_promptly(stt_app, scratch):
    stt_app._idle_unloader.cancel()
    model = reset_model(stt_app)
    model.default = [Segment("eins", start=0.0, end=1.0), Segment("zwei", start=1.0, end=2.0)]

    async def scenario():
        response = await _open_stream(stt_app)
        sent = []

        async def receive():
            await asyncio.sleep(3600)

        async def send(message):
            sent.append(message)
            if len(sent) >= 3:                      # start + two events, then the peer resets
                raise OSError("connection reset")

        scope = {"type": "http", "asgi": {"spec_version": "2.5"}, "method": "POST", "path": "/transcribe-stream"}
        with pytest.raises(Exception):
            await response(scope, receive, send)
        assert await _eventually(lambda: stt_app._model_refs == 0), "the reference leaked after the disconnect"

    asyncio.run(scenario())
    assert _leftovers(scratch) == []


# --- /unload never touches the loop with a blocking lock ---------------------


def test_unload_and_health_stay_responsive_while_a_load_holds_the_lock(client, stt_app, monkeypatch):
    entered, gate = threading.Event(), threading.Event()

    class Slow(FakeWhisper):
        def __init__(self, *args, **kwargs):
            entered.set()
            gate.wait(15)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(stt_app, "WhisperModel", Slow)
    stt_app.whisper_model = None
    stt_app.model_loaded = False

    loader = threading.Thread(target=lambda: _post(client), daemon=True)
    loader.start()
    unloader = None
    try:
        assert entered.wait(5)
        unloader = threading.Thread(target=lambda: client.post("/unload"), daemon=True)
        unloader.start()                    # blocks on the lock the load holds

        answered = {}

        def probe():
            started = time.monotonic()
            answered["status"] = client.get("/health").status_code
            answered["elapsed"] = time.monotonic() - started

        prober = threading.Thread(target=probe, daemon=True)
        prober.start()
        prober.join(3)
        assert not prober.is_alive() and answered["elapsed"] < 2.0, (
            "the event loop is blocked behind the unload"
        )
        assert answered["status"] == 200
    finally:
        gate.set()
        loader.join(10)
        if unloader is not None:
            unloader.join(10)
