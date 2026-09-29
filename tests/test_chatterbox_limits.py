"""Input bounds, reference-clip handling, temp files and /unload behaviour.

Before: /clone read the whole upload into memory with no cap, and the library then
decoded and embedded the whole clip although only 6-10 s condition the voice; the
temp file was deleted when the request ended even if a worker thread was still
about to read it; /unload ran the lock-taking unload on the event loop.
"""

import asyncio
import io
import os
import re
import tempfile
import threading
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from chatterbox_loader import (
    CLONE_VOICE, SAMPLE_RATE, decode_audio, install_chatterbox, load_app, make_upload, patch_decoder,
    reference_clip, voices_in, wait_until, wait_until_async,
)

REPO = Path(__file__).resolve().parents[1]


def _post_clone(client, content, text="Hallo Welt.", filename="ref.wav", **form):
    return client.post(
        "/clone", data={"text": text, "lang": "de", **form},
        files={"file": (filename, content, "audio/wav")},
    )


@pytest.fixture
def scratch(tmp_path, monkeypatch):
    """Redirect temp files so a leak is visible as a file left in `tmp_path`."""
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    return tmp_path


# --- Upload size -------------------------------------------------------------

def test_an_oversized_upload_is_refused_before_it_is_buffered(monkeypatch, scratch):
    package = install_chatterbox(monkeypatch)
    app = load_app(MAX_UPLOAD_MB="0.001")  # ~1 KB
    patch_decoder(monkeypatch, app)
    client = TestClient(app.app)

    response = _post_clone(client, b"\0" * (3 * 1024 * 1024))  # well past the Content-Length guard

    assert response.status_code == 413
    assert package.loads == 0
    assert list(scratch.iterdir()) == []


def test_an_upload_just_over_the_limit_is_refused_by_the_handler(monkeypatch, scratch):
    package = install_chatterbox(monkeypatch)
    app = load_app(MAX_UPLOAD_MB="0.001")
    patch_decoder(monkeypatch, app)
    client = TestClient(app.app)

    response = _post_clone(client, b"\0" * 5000)  # under the multipart slack, over MAX_UPLOAD_MB

    assert response.status_code == 413
    assert "MAX_UPLOAD_MB" in response.json()["detail"]
    assert package.loads == 0
    assert list(scratch.iterdir()) == []


def test_the_copy_is_bounded_even_when_the_upload_reports_no_size(monkeypatch, scratch):
    """Chunked clients: `UploadFile.size` can be None, so the copy loop itself must stop."""
    install_chatterbox(monkeypatch)
    app = load_app(MAX_UPLOAD_MB="0.001")
    upload = make_upload(b"\0" * 50_000)
    upload.size = None

    with pytest.raises(app.HTTPException) as caught:
        app._stage_reference(upload)

    assert caught.value.status_code == 413
    assert list(scratch.iterdir()) == []


def test_json_bodies_are_bounded_too(monkeypatch):
    install_chatterbox(monkeypatch)
    client = TestClient(load_app().app)
    assert client.post("/tts", content=b"x" * (2 * 1024 * 1024), headers={"content-type": "application/json"}).status_code == 413


def test_an_empty_or_undecodable_reference_is_a_400_and_leaves_nothing_behind(monkeypatch, scratch):
    package = install_chatterbox(monkeypatch)
    app = load_app()
    client = TestClient(app.app)

    def cannot_decode(path, seconds):
        raise ValueError("Format not recognised")

    monkeypatch.setattr(app, "_decode_reference", cannot_decode)

    empty = _post_clone(client, b"")
    broken = _post_clone(client, b"not audio at all")

    assert empty.status_code == 400 and "empty" in empty.json()["detail"].lower()
    assert broken.status_code == 400 and "decode" in broken.json()["detail"].lower()
    assert package.loads == 0
    assert list(scratch.iterdir()) == []


# --- Reference length --------------------------------------------------------

def test_only_the_first_seconds_of_a_long_reference_are_decoded(monkeypatch, scratch):
    package = install_chatterbox(monkeypatch)
    app = load_app()
    asked = []
    patch_decoder(monkeypatch, app, record=asked)
    client = TestClient(app.app)

    response = _post_clone(client, reference_clip(seconds=100))

    assert response.status_code == 200
    assert asked == [30.0]
    assert package.model.prepared_samples == [30 * SAMPLE_RATE], "the library was handed the whole clip"


def test_the_reference_cap_is_configurable(monkeypatch, scratch):
    package = install_chatterbox(monkeypatch)
    app = load_app(CHATTERBOX_REF_MAX_SECONDS="5")
    patch_decoder(monkeypatch, app)

    _post_clone(TestClient(app.app), reference_clip(seconds=20))

    assert package.model.prepared_samples == [5 * SAMPLE_RATE]


def test_a_short_reference_is_passed_through_whole(monkeypatch, scratch):
    package = install_chatterbox(monkeypatch)
    app = load_app()
    patch_decoder(monkeypatch, app)

    response = _post_clone(TestClient(app.app), reference_clip(seconds=3))

    assert package.model.prepared_samples == [3 * SAMPLE_RATE]
    assert voices_in(decode_audio(response.content)) == {CLONE_VOICE}


# --- Temp files --------------------------------------------------------------

def test_no_temp_file_is_left_after_success_or_failure(monkeypatch, scratch):
    install_chatterbox(monkeypatch)
    app = load_app()
    patch_decoder(monkeypatch, app)
    client = TestClient(app.app, raise_server_exceptions=False)

    ok = _post_clone(client, reference_clip(seconds=1))
    failed = _post_clone(client, reference_clip(seconds=1), text="BOOM beim Klonen.")

    assert ok.status_code == 200 and failed.status_code == 500
    assert list(scratch.iterdir()) == []


def test_the_staged_reference_outlives_a_cancelled_request_until_the_worker_is_done(monkeypatch, scratch):
    """The worker thread may not have opened the file yet when the client gives up."""
    package = install_chatterbox(monkeypatch)
    app = load_app()
    patch_decoder(monkeypatch, app)
    TestClient(app.app).post("/tts", json={"text": "Wärmen.", "language": "de"})  # load the model
    model = package.model
    model.gate = threading.Event()
    model.entered.clear()

    async def main():
        task = asyncio.ensure_future(app.clone_voice(
            text="Klon.", lang="de", file=make_upload(reference_clip(seconds=1)), exaggeration=None, cfg_weight=None
        ))
        assert await asyncio.to_thread(model.entered.wait, 5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        during = sorted(p.name for p in scratch.iterdir())
        model.gate.set()
        await wait_until_async(lambda: not list(scratch.iterdir()) and app._model_slot.refs == 0)
        return during

    during = asyncio.run(main())

    assert len(during) == 1 and during[0].endswith(".wav"), f"expected the staged reference to survive, got {during}"
    assert list(scratch.iterdir()) == []


# --- Endpoints ---------------------------------------------------------------

def test_the_ref_text_alias_ignores_ref_text_and_speaks_long_text(monkeypatch, scratch):
    package = install_chatterbox(monkeypatch)
    app = load_app()
    patch_decoder(monkeypatch, app)
    text = " ".join(["Das ist ein ganz gewöhnlicher deutscher Satz über das Wetter im Herbst."] * 12)

    response = TestClient(app.app).post(
        "/clone-with-ref-text",
        data={"text": text, "ref_text": "Ein ganz anderer Text.", "lang": "de"},
        files={"file": ("ref.wav", reference_clip(), "audio/wav")},
    )

    assert response.status_code == 200
    assert " ".join(c["text"] for c in package.model.calls).split() == text.split()
    assert "Ein ganz anderer Text." not in " ".join(c["text"] for c in package.model.calls)


@pytest.mark.parametrize("field, value", [
    ("exaggeration", -0.5), ("cfg_weight", -1), ("exaggeration", float("nan")), ("cfg_weight", float("inf")),
])
def test_meaningless_tuning_values_are_a_422_not_garbage_audio(monkeypatch, field, value):
    package = install_chatterbox(monkeypatch)
    client = TestClient(load_app().app)

    body = '{"text": "Hallo.", "language": "de", "%s": %s}' % (
        field, {"nan": "NaN", "inf": "Infinity"}.get(str(value), value)
    )
    response = client.post("/tts", content=body, headers={"content-type": "application/json"})

    assert response.status_code == 422
    assert package.loads == 0


def test_tuning_values_within_range_are_passed_through(monkeypatch):
    package = install_chatterbox(monkeypatch)
    client = TestClient(load_app().app)

    client.post("/tts", json={"text": "Hallo.", "language": "de", "exaggeration": 0.9, "cfg_weight": 0.0})

    call = package.model.calls[-1]
    assert (call["exaggeration"], call["cfg_weight"]) == (0.9, 0.0)


# --- /unload -----------------------------------------------------------------

def test_unload_does_not_block_the_event_loop(monkeypatch):
    """The unload takes the slot lock and runs the unload hooks (seconds on a real GPU)."""
    install_chatterbox(monkeypatch)
    app = load_app()
    started, release, blocked = threading.Event(), threading.Event(), []

    def slow_forget(model):
        started.set()
        if not release.wait(1.0):
            blocked.append(True)  # nothing could set `release`: the loop was stuck in here
        app._forget_chatterbox(model)

    app._model_slot = app.ModelSlot(
        app._load_chatterbox, ttl_seconds=-1, name="Chatterbox Multilingual", on_unload=slow_forget
    )
    TestClient(app.app).post("/tts", json={"text": "Wärmen.", "language": "de"})

    async def main():
        task = asyncio.ensure_future(app.unload())
        assert await asyncio.to_thread(started.wait, 5), "the unload never ran"
        release.set()
        return await task

    result = asyncio.run(main())

    assert not blocked, "POST /unload blocked the event loop"
    assert result["unloaded"] is True


def test_unload_answers_409_while_a_generation_is_in_flight_and_200_after(monkeypatch):
    package = install_chatterbox(monkeypatch)
    app = load_app()
    client = TestClient(app.app)
    client.post("/tts", json={"text": "Wärmen.", "language": "de"})
    model = package.model
    model.gate = threading.Event()
    model.entered.clear()
    outcome = {}

    worker = threading.Thread(
        target=lambda: outcome.update(tts=client.post("/tts", json={"text": "Läuft.", "language": "de"})), daemon=True
    )
    worker.start()
    assert model.entered.wait(5)

    busy = client.post("/unload")
    model.gate.set()
    worker.join(10)
    assert wait_until(lambda: app._model_slot.refs == 0)
    done = client.post("/unload")

    assert busy.status_code == 409
    assert outcome["tts"].status_code == 200
    assert done.status_code == 200 and done.json()["unloaded"] is True


# --- Requirements ------------------------------------------------------------

def test_requirements_leave_torch_alone_and_chatterbox_to_the_dockerfile():
    """chatterbox-tts pins torch==2.6.0 (the cu124 build, no sm_120 kernels) and gradio.

    Listing it, or torch/torchaudio, in requirements.txt makes the second pip install
    downgrade the cu128 torch the Dockerfile installed first.
    """
    text = (REPO / "chatterbox-tts-service" / "requirements.txt").read_text(encoding="utf-8")
    names = {
        re.match(r"[A-Za-z0-9_.\-]+", line.strip()).group(0).lower().replace("_", "-")
        for line in text.splitlines()
        if line.strip() and not line.strip().startswith("#")
    }

    assert not names & {"torch", "torchaudio", "torchvision", "gradio", "chatterbox-tts"}, names
    # What the package needs at import time and would otherwise have brought along.
    assert {"librosa", "transformers", "diffusers", "s3tokenizer", "resemble-perth", "conformer"} <= names
