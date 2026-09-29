"""What stt-service tells an anonymous caller when something fails.

Findings covered:

* /health carried ``startup_error`` (the exception text of the last failed load: file
  paths under the model cache, repo ids, URLs) and ``torch_version``; /ready repeated
  the text as ``detail``; the 500s of /transcribe, /detect_language, the SSE stream
  and the live WebSocket echoed ``str(exc)`` (temp-file paths, library internals).
  Now a probe sees a short category (``model_files_unavailable``, ``out_of_memory``,
  ``missing_dependency``, ``load_error``) and an error names a request id; the
  exception text is in the log under that id.
* a cold load holds the model lock for seconds to minutes, and every streaming or
  WebSocket request waiting behind it used to park a thread of the event loop's
  default executor: with a few dozen of them, uploads and /unload waited out the
  whole load.

What a probe and the gateway read from /health (``status``, ``model_loaded``,
``model_resident``, ``model_size``, ``device``, ``can_load``, the status code) and
the /ready reasons (``loading``, ``load_failed``, ``resident``, ``unloaded``) must not
change.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import tempfile
import threading
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from test_stt_support import (
    FakeWhisper, Info, LiveSession, Segment, load_stt_app, reset_model, wait_for_preload, wait_until,
)

WAV = b"RIFF" + b"\x00" * 60
REPO = Path(__file__).resolve().parents[1]
LEAK = "/root/.cache/huggingface/hub/models--Systran--faster-whisper-tiny/snapshots/0123abcd/model.bin"


@pytest.fixture(scope="module")
def stt_app():
    return load_stt_app({}, name="stt_app_hygiene")


@pytest.fixture(scope="module")
def _client(stt_app):
    with TestClient(stt_app.app) as test_client:
        wait_for_preload(stt_app)
        yield test_client


@pytest.fixture
def client(stt_app, _client, tmp_path, monkeypatch):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    stt_app._idle_unloader.cancel()
    model = reset_model(stt_app)
    model.default = [Segment("hallo welt", start=0.0, end=1.5)]
    model.info = Info("de", 0.97, duration=2.0)
    _client.fake_model = model
    yield _client
    stt_app.startup_error = None
    stt_app._idle_unloader.cancel()
    assert wait_until(lambda: stt_app._model_refs == 0), f"leaked a model reference: {stt_app._model_refs}"


def _post(client, path="/transcribe", files=None):
    return client.post(path, files=files or {"audio": ("a.wav", WAV, "audio/wav")})


def _broken_model(error):
    class Broken:
        def __init__(self, *args, **kwargs):
            raise error

    return Broken


def _fail_to_load(stt_app, monkeypatch, error):
    """Run the real load path against a WhisperModel that raises *error*."""
    monkeypatch.setattr(stt_app, "WhisperModel", _broken_model(error))
    monkeypatch.setattr(stt_app, "whisper_model", None)
    monkeypatch.setattr(stt_app, "model_loaded", False)
    stt_app.load_model()


def _request_id(text):
    match = re.search(r"Request id: ([0-9a-f]{12})\.", text)
    assert match, f"no request id in {text!r}"
    return match.group(1)


@pytest.fixture
def no_model(stt_app, monkeypatch):
    """Nothing resident and nothing known to be wrong: the next acquire loads."""
    stt_app._idle_unloader.cancel()
    monkeypatch.setattr(stt_app, "whisper_model", None)
    monkeypatch.setattr(stt_app, "model_loaded", False)
    monkeypatch.setattr(stt_app, "startup_error", None)
    yield
    stt_app._idle_unloader.cancel()


# --- /health and /ready ---------------------------------------------------------------------

def test_health_names_the_failure_and_keeps_the_exception_in_the_log(client, stt_app, monkeypatch, caplog):
    with caplog.at_level(logging.ERROR):
        _fail_to_load(stt_app, monkeypatch, OSError(f"Unable to open file {LEAK!r}"))

    response = client.get("/health")

    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "error" and body["can_load"] is False
    assert body["startup_error"] == "model_files_unavailable"
    assert "/root" not in response.text and "huggingface" not in response.text and "snapshots" not in response.text
    assert "torch_version" not in body
    assert LEAK in caplog.text, "the operator must still find the path in the log"


def test_health_keeps_the_fields_the_gateway_and_the_probes_read(client, stt_app):
    body = client.get("/health").json()

    assert client.get("/health").status_code == 200
    assert {"status", "model_loaded", "model_resident", "model_size", "device", "can_load",
            "compute_type", "degraded", "live_sessions", "model_refs", "model_ttl_seconds",
            "cuda_available"} <= set(body)
    assert body["status"] == "ok" and body["can_load"] is True and body["model_resident"] is True
    assert "startup_error" not in body and "torch_version" not in body


def test_ready_says_load_failed_with_a_category_and_recovers(client, stt_app, monkeypatch, caplog):
    with caplog.at_level(logging.ERROR):
        _fail_to_load(stt_app, monkeypatch, RuntimeError(f"CUDA failed: out of memory near {LEAK}"))

    failed = client.get("/ready")

    assert failed.status_code == 503
    assert failed.json()["reason"] == "load_failed" and failed.json()["ready"] is False
    assert failed.json()["detail"] == "out_of_memory"
    assert "/root" not in failed.text and "CUDA" not in failed.text

    monkeypatch.setattr(stt_app, "WhisperModel", FakeWhisper)
    stt_app.load_model()
    assert stt_app.startup_error is None
    assert client.get("/ready").json()["reason"] == "resident"


def test_the_ready_reasons_stay_the_documented_strings(client, stt_app, monkeypatch):
    loading = {}
    monkeypatch.setattr(stt_app, "_loading", True)
    monkeypatch.setattr(stt_app, "whisper_model", None)
    response = client.get("/ready")
    loading["status"], loading["reason"], loading["retry"] = (
        response.status_code, response.json()["reason"], response.headers.get("retry-after"))
    monkeypatch.setattr(stt_app, "_loading", False)
    monkeypatch.setattr(stt_app, "startup_error", "load_error")
    failed = client.get("/ready")
    monkeypatch.setattr(stt_app, "startup_error", None)
    unloaded = client.get("/ready")
    monkeypatch.setattr(stt_app, "whisper_model", client.fake_model)
    resident = client.get("/ready")

    assert loading == {"status": 503, "reason": "loading", "retry": "5"}
    assert (failed.status_code, failed.json()["reason"]) == (503, "load_failed")
    assert (unloaded.status_code, unloaded.json()["reason"]) == (200, "unloaded")
    assert (resident.status_code, resident.json()["reason"]) == (200, "resident")


@pytest.mark.parametrize("error, category", [
    (MemoryError(), "out_of_memory"),
    (RuntimeError("CUDA failed with error out of memory"), "out_of_memory"),
    (ModuleNotFoundError("No module named 'ctranslate2'"), "missing_dependency"),
    (ConnectionError(f"could not reach the hub while fetching {LEAK}"), "model_files_unavailable"),
    (RuntimeError(f"Unable to open file 'model.bin' in model '{LEAK}'"), "model_files_unavailable"),
    (ValueError("Invalid model size 'huge'"), "load_error"),
])
def test_a_failed_load_is_reported_as_its_category(client, stt_app, monkeypatch, error, category):
    _fail_to_load(stt_app, monkeypatch, error)

    assert client.get("/health").json()["startup_error"] == category
    response = _post(client)
    assert response.status_code == 503
    assert response.json()["detail"] == f"Model not available: {category}"


# --- request errors -------------------------------------------------------------------------

def test_a_crash_in_the_decode_is_a_500_with_a_request_id_and_the_log_has_the_rest(client, caplog):
    client.fake_model.fail_with = RuntimeError(f"decoder exploded reading {LEAK}")

    with caplog.at_level(logging.ERROR):
        response = _post(client)

    assert response.status_code == 500
    detail = response.json()["detail"]
    assert "decoder exploded" not in detail and "/root" not in detail
    request_id = _request_id(detail)
    assert request_id in caplog.text and f"decoder exploded reading {LEAK}" in caplog.text


def test_a_crash_in_language_detection_is_a_500_with_a_request_id(client, caplog):
    client.fake_model.fail_with = RuntimeError(f"detector exploded reading {LEAK}")

    with caplog.at_level(logging.ERROR):
        response = _post(client, "/detect_language", files={"file": ("a.wav", WAV, "audio/wav")})

    assert response.status_code == 500
    detail = response.json()["detail"]
    assert "detector exploded" not in detail and "/root" not in detail
    assert _request_id(detail) in caplog.text and "detector exploded" in caplog.text


def test_an_error_our_own_code_wrote_for_the_client_is_still_passed_on(client):
    response = client.post("/transcribe", data={"task": "sing"}, files={"audio": ("a.wav", WAV, "audio/wav")})
    assert response.status_code == 400 and "sing" in response.json()["detail"]


def test_a_crash_in_the_stream_is_an_event_with_a_request_id_not_the_exception(client, caplog):
    client.fake_model.fail_with = RuntimeError(f"stream exploded reading {LEAK}")

    with caplog.at_level(logging.ERROR):
        response = _post(client, "/transcribe-stream")

    events = [json.loads(line[len("data: "):]) for line in response.text.splitlines() if line.startswith("data: ")]
    errors = [e for e in events if e.get("status") == "error"]
    assert len(errors) == 1
    assert "stream exploded" not in errors[0]["error"] and "/root" not in errors[0]["error"]
    assert _request_id(errors[0]["error"]) in caplog.text and "stream exploded" in caplog.text


def test_a_crash_in_the_live_session_is_a_frame_with_a_request_id_not_the_exception(client, stt_app, monkeypatch, caplog):
    async def explode():
        raise RuntimeError(f"session exploded reading {LEAK}")

    monkeypatch.setattr(stt_app, "_hold_model_async", explode)

    with caplog.at_level(logging.ERROR):
        with LiveSession(client) as session:
            error = session.wait_type("error")

    assert error["code"] == "internal_error"
    assert "session exploded" not in error["message"] and "/root" not in error["message"]
    assert error["error"] == error["message"], "the shipped frontend reads the `error` key"
    assert _request_id(error["message"]) in caplog.text and "session exploded" in caplog.text


# --- model names ----------------------------------------------------------------------------

def test_a_local_model_path_is_shown_as_its_last_component_only(stt_app, no_model, monkeypatch):
    # Read at request time: nothing is loaded, so the configured name is what is reported.
    monkeypatch.setattr(stt_app, "model_size_loaded", None)
    monkeypatch.setenv("WHISPER_MODEL_SIZE", "/models/ct2/whisper-german")
    plain = TestClient(stt_app.app)

    health, ready, info, models = (plain.get(p).json() for p in ("/health", "/ready", "/info", "/models"))

    assert health["model_size"] == ready["model_size"] == info["model_size"] == "whisper-german"
    assert models["current_model"] == "whisper-german"
    assert "/models/ct2" not in json.dumps([health, ready, info, models])


@pytest.mark.parametrize("name", ["primeline/whisper-large-v3-turbo-german", "large-v3-turbo"])
def test_a_hugging_face_id_and_a_size_are_shown_as_they_are(stt_app, no_model, monkeypatch, name):
    monkeypatch.setattr(stt_app, "model_size_loaded", None)
    monkeypatch.setenv("WHISPER_MODEL_SIZE", name)
    assert TestClient(stt_app.app).get("/health").json()["model_size"] == name


# --- the two error_category implementations answer alike ----------------------------------------

def _load_module(path, name):
    spec = spec_from_file_location(name, path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_stt_and_the_lifecycle_module_categorise_every_failure_alike():
    """They live in different images and cannot share a file; this is what keeps them one behaviour."""
    residency = _load_module(REPO / "stt-service" / "residency.py", "residency_under_test")
    lifecycle = _load_module(REPO / "parakeet-asr-service" / "model_lifecycle.py", "lifecycle_under_test")

    class Hostile(Exception):
        def __str__(self):
            raise RuntimeError("no str")

    HFValidationError = type("HFValidationError", (ValueError,), {})
    OutOfMemoryError = type("OutOfMemoryError", (RuntimeError,), {})
    cases = [
        MemoryError(), OutOfMemoryError("x"), RuntimeError("CUDA out of memory"), ImportError("x"),
        ModuleNotFoundError("x"), FileNotFoundError(LEAK), PermissionError(13, "denied", LEAK),
        ConnectionError("hub"), TimeoutError("slow"), HFValidationError("repo id"),
        RuntimeError("Unable to open file 'model.bin'"), RuntimeError("No such file or directory"),
        RuntimeError("weights are corrupt"), ValueError("nope"), KeyError("k"), Hostile(),
    ]
    for error in cases:
        assert residency.error_category(error) == lifecycle.error_category(error), repr(error)


# --- waiters behind a cold load leave the default executor free ---------------------------------

class _SlowToLoad(FakeWhisper):
    """A model whose constructor parks until the test lets the load finish."""

    entered = threading.Event()
    release = threading.Event()
    constructed = 0

    def __init__(self, *args, **kwargs):
        type(self).constructed += 1
        type(self).entered.set()
        assert type(self).release.wait(5), "the test never let the load finish"
        super().__init__(*args, **kwargs)


@pytest.fixture
def slow_load(stt_app, no_model, monkeypatch):
    _SlowToLoad.entered, _SlowToLoad.release, _SlowToLoad.constructed = threading.Event(), threading.Event(), 0
    monkeypatch.setattr(stt_app, "WhisperModel", _SlowToLoad)
    yield _SlowToLoad
    _SlowToLoad.release.set()


def test_sixty_streams_waiting_for_a_cold_load_do_not_starve_the_default_executor(stt_app, slow_load):
    async def main():
        waiters = [asyncio.create_task(stt_app._hold_model_async()) for _ in range(60)]
        try:
            assert await asyncio.to_thread(slow_load.entered.wait, 5), "the load never started"
            await asyncio.sleep(0.2)  # let the rest queue up behind it
            assert await asyncio.wait_for(asyncio.to_thread(lambda: "free"), 3) == "free", (
                "an unrelated asyncio.to_thread waited behind the model load"
            )
            # One thread waits for the model lock, not sixty. (This app's own pool: other
            # test modules load their own copies of the app, each with a thread of its own.)
            assert len(stt_app._acquire_executor._threads) == 1
        finally:
            slow_load.release.set()  # a failure above must not leave 60 threads waiting on the load
        holds = await asyncio.gather(*waiters)
        assert slow_load.constructed == 1, "the waiters must share one load"
        assert stt_app._model_refs == 60
        for hold in holds:
            await stt_app._release_soon(hold)
        assert await asyncio.to_thread(wait_until, lambda: stt_app._model_refs == 0)

    asyncio.run(main())


def test_a_waiter_cancelled_before_its_acquire_started_takes_nothing(stt_app, no_model, monkeypatch):
    """Run for nobody, a withdrawn acquire would retry a failing load."""
    monkeypatch.setattr(stt_app, "WhisperModel", _broken_model(RuntimeError("no VRAM")))
    constructed = []
    real_load, real_acquire = stt_app.load_model, stt_app.acquire_model

    def counting():
        constructed.append(1)
        real_load()

    gate = threading.Event()

    def blocking_acquire():
        assert gate.wait(5)
        return real_acquire()

    monkeypatch.setattr(stt_app, "load_model", counting)
    monkeypatch.setattr(stt_app, "acquire_model", blocking_acquire)

    async def main():
        first = asyncio.create_task(stt_app._hold_model_async())
        await asyncio.sleep(0.05)
        second = asyncio.create_task(stt_app._hold_model_async())
        await asyncio.sleep(0.05)
        second.cancel()
        with pytest.raises(asyncio.CancelledError):
            await second
        gate.set()
        with pytest.raises(Exception):
            await first
        # One worker, first in first out: once this returns, the second acquire
        # would already have run had it not been withdrawn.
        await asyncio.wrap_future(stt_app._acquire_executor.submit(lambda: None))

    asyncio.run(main())

    assert len(constructed) == 1, "the withdrawn acquire ran anyway"
    assert stt_app._model_refs == 0
