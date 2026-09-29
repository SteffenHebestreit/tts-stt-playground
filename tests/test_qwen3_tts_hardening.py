"""qwen3-tts-service hardening: what errors reveal, how long a reference clip may be, how long a request may queue.

* Unexpected failures answer with a generic message and a request id; the exception
  text is in the log under the same id (E1).
* A reference clip is measured before it is used (header first, then a decode that
  stops at the limit) and refused above QWEN3_TTS_REF_MAX_SECONDS, so a few MB of
  highly compressible audio cannot expand into hours of samples (E3).
* Waiting for the model lock or a generation permit is bounded in time
  (TTS_QUEUE_TIMEOUT_S) and in depth (TTS_MAX_QUEUE); the in-flight accounting the idle
  reaper and /unload depend on comes back to zero on every path (E4).

The concurrency tests run every request on ONE event loop through httpx's ASGI
transport: the asyncio primitives are bound to the loop that contends for them.
"""

from __future__ import annotations

import asyncio
import io
import struct
import sys
import tempfile
import threading
import time
import types

import httpx
import pytest

np = pytest.importorskip("numpy", reason="qwen3-tts app.py requires numpy")

# The real soundfile, imported now, before the loader can register its stand-in (which has
# no SoundFile): None where it is not installed.
try:
    import soundfile as _real_soundfile
    if not hasattr(_real_soundfile, "SoundFile"):
        _real_soundfile = None
except ImportError:  # pragma: no cover
    _real_soundfile = None

needs_soundfile = pytest.mark.skipif(_real_soundfile is None, reason="needs the real soundfile")

from qwen3_tts_loader import (  # noqa: E402
    BASE_06, CUSTOM_17, DESIGN_17, FakeQwenModel, capture_wav, client, install_model, install_qwen_tts, load_app,
)

UPLOAD = b"RIFF" + b"\x01\x02\x03" * 40


def _audio(name="ref.wav", content=UPLOAD):
    return {"file": (name, content, "audio/wav")}


@pytest.fixture
def scratch(tmp_path, monkeypatch):
    """Temp files land in `tmp_path`, so a leaked one is visible."""
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    return tmp_path


def _leaked(scratch) -> list:
    """Temp files in `scratch`, apart from the voices directory the loader creates there."""
    return [p.name for p in scratch.iterdir() if not p.name.startswith("qwen3-tts-voices-")]


@pytest.fixture
def base_app(monkeypatch, scratch):
    m = load_app()
    capture_wav(monkeypatch, m)
    return m, install_model(m, FakeQwenModel("base"), BASE_06)


# --- E1: generic errors with an id ---------------------------------------------------------

def _assert_generic(response, log, secret, status=500):
    assert response.status_code == status, response.text
    request_id = response.headers["X-Request-ID"]
    assert request_id in response.json()["detail"]
    assert secret not in response.text, "the exception text reached the client"
    assert any(request_id in r.getMessage() and secret in r.getMessage() for r in log.records), \
        "the detail is not in the log under the id the client was given"


SECRET = "torch.OutOfMemoryError at /usr/local/lib/python3.12/site-packages/qwen_tts/model.py:88"


def test_no_generation_endpoint_reveals_the_exception(monkeypatch, scratch, caplog):
    m = load_app()
    capture_wav(monkeypatch, m)
    install_qwen_tts(monkeypatch)
    base = FakeQwenModel("base")
    install_model(m, base, BASE_06)
    base.fail_with = RuntimeError(SECRET)
    c = client(m)

    with caplog.at_level("ERROR"):
        responses = [
            c.post("/clone", data={"text": "Hallo."}, files=_audio()),
            c.post("/clone-with-ref-text", data={"text": "Hallo.", "ref_text": "Ref."}, files=_audio()),
            c.post("/voices/save", data={"name": "Anna", "ref_text": "Hallo."}, files=_audio()),
        ]
        design = FakeQwenModel("voice_design")
        design.fail_with = RuntimeError(SECRET)
        install_model(m, design, DESIGN_17)
        responses.append(c.post("/voice_design", json={"text": "Hallo.", "voice_description": "warm"}))
        speakers = FakeQwenModel("custom_voice")
        speakers.fail_with = RuntimeError(SECRET)
        install_model(m, speakers, CUSTOM_17)
        responses.append(c.post("/tts", json={"text": "Hallo.", "speaker": "Vivian"}))

    for response in responses:
        _assert_generic(response, caplog, SECRET)
    assert len({r.headers["X-Request-ID"] for r in responses}) == len(responses)


def test_a_saved_voice_playback_failure_does_not_reveal_the_exception(monkeypatch, scratch, caplog, tmp_path):
    m = load_app(str(tmp_path / "voices"))
    capture_wav(monkeypatch, m)
    install_qwen_tts(monkeypatch)
    base = FakeQwenModel("base")
    install_model(m, base, BASE_06)
    c = client(m)
    assert c.post("/voices/save", data={"name": "Anna", "ref_text": "Hallo."}, files=_audio()).status_code == 200
    base.fail_with = RuntimeError(SECRET)

    with caplog.at_level("ERROR"):
        r = c.post("/voices/anna/tts", data={"text": "Hallo."})

    _assert_generic(r, caplog, SECRET)


def test_a_failed_model_switch_and_a_failed_load_do_not_reveal_the_exception(monkeypatch, caplog):
    m = load_app()
    secret = "OSError: [Errno 28] No space left on device: '/root/.cache/huggingface/hub/models--Qwen'"

    def broken(name, **kw):
        raise OSError(secret)

    install_qwen_tts(monkeypatch, from_pretrained=broken)
    c = client(m)

    with caplog.at_level("ERROR"):
        switched = c.post("/load_model", json={"model": BASE_06})
        cold = c.post("/clone", data={"text": "Hallo."}, files=_audio())
        ready = c.get("/ready")

    _assert_generic(switched, caplog, "No space left on device")
    _assert_generic(cold, caplog, "No space left on device")
    assert ready.status_code == 503 and ready.json()["reason"] == "load_failed"
    for leaked in ("/root/.cache", "No space left", "OSError", "huggingface", "models--"):
        assert leaked not in ready.text, f"/ready reveals {leaked!r}"


def test_the_text_limit_error_says_how_to_raise_the_limit(scratch):
    m = load_app(MAX_TEXT_CHARS="20")

    r = client(m).post("/clone", data={"text": "x" * 21}, files=_audio())

    assert r.status_code == 413
    detail = r.json()["detail"]
    assert "MAX_TEXT_CHARS" in detail and "20" in detail and "/" not in detail


# --- E3: reference clips are measured, and bounded ---------------------------------------

def _measured(monkeypatch, m, result):
    """Make the service's clip measurement return `result` (or raise it); record what it was asked."""
    asked = []

    def measure(path, cap):
        asked.append(cap)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(m, "_measure_reference", measure)
    return asked


REF_ENDPOINTS = [
    ("/clone", {"text": "Hallo."}),
    ("/clone-with-ref-text", {"text": "Hallo.", "ref_text": "Ref."}),
    ("/voices/save", {"name": "Anna", "ref_text": "Hallo."}),
]


@pytest.mark.parametrize("path, data", REF_ENDPOINTS)
def test_a_clip_over_the_cap_is_refused_before_the_model_sees_it(base_app, monkeypatch, path, data, scratch):
    m, model = base_app
    asked = _measured(monkeypatch, m, (3600.0, True))

    r = client(m).post(path, data=data, files=_audio())

    assert r.status_code == 413
    assert "QWEN3_TTS_REF_MAX_SECONDS" in r.json()["detail"] and "60" in r.json()["detail"]
    assert asked == [60.0]
    assert model.calls == [], "the model was given a clip that is far over the cap"
    assert _leaked(scratch) == [], "the refused clip was left on disk"
    assert m._inflight == 0 and m._active_requests == 0


def test_a_clip_whose_decode_was_cut_off_is_refused_without_a_length(base_app, monkeypatch):
    m, model = base_app
    _measured(monkeypatch, m, (60.0 + 1 / 16000, False))

    r = client(m).post("/clone", data={"text": "Hallo."}, files=_audio())

    assert r.status_code == 413 and "longer than the limit" in r.json()["detail"]
    assert model.calls == []


@pytest.mark.parametrize("result, status", [
    (ValueError("Format not recognised"), 400),
    (ImportError("No module named 'librosa'"), 400),
    ((0.0, True), 400),
])
def test_a_clip_that_cannot_be_read_or_is_empty_is_a_400(base_app, monkeypatch, scratch, caplog, result, status):
    m, model = base_app
    _measured(monkeypatch, m, result)

    with caplog.at_level("WARNING"):
        r = client(m).post("/clone", data={"text": "Hallo."}, files=_audio())

    assert r.status_code == status
    assert "Format not recognised" not in r.text and "librosa" not in r.text
    assert model.calls == [] and _leaked(scratch) == []


def test_a_clip_within_the_cap_reaches_the_model_unchanged(base_app, monkeypatch):
    m, model = base_app
    _measured(monkeypatch, m, (59.9, True))

    r = client(m).post("/clone", data={"text": "Hallo."}, files=_audio())

    assert r.status_code == 200
    [(_, prompt_call), *_] = model.calls
    assert prompt_call["ref_audio_bytes"] == UPLOAD


def test_the_reference_cap_defaults_to_sixty_seconds_and_can_be_changed(monkeypatch, caplog):
    assert load_app().REF_MAX_SECONDS == 60.0
    assert load_app(QWEN3_TTS_REF_MAX_SECONDS="12.5").REF_MAX_SECONDS == 12.5
    with caplog.at_level("WARNING"):
        assert load_app(QWEN3_TTS_REF_MAX_SECONDS="long").REF_MAX_SECONDS == 60.0
    assert "QWEN3_TTS_REF_MAX_SECONDS" in caplog.text


class FakeSoundFile:
    """`soundfile.SoundFile` as far as the measurement uses it: a clip that reads zeros."""

    opened = []

    def __init__(self, path, *, rate=16000, frames=0, endless=False):
        self.samplerate, self.frames, self.endless = rate, frames, endless
        self.frames_read = 0
        FakeSoundFile.opened.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self, count, dtype="float32", always_2d=True):
        available = count if self.endless else max(0, min(count, self.frames - self.frames_read))
        self.frames_read += available
        return np.zeros((available, 1), dtype="float32")


def _clip(monkeypatch, m, **clip_args):
    FakeSoundFile.opened = []
    monkeypatch.setattr(m.sf, "SoundFile", lambda path: FakeSoundFile(path, **clip_args), raising=False)


def test_a_header_that_declares_an_hour_is_refused_without_reading_a_sample(base_app, monkeypatch):
    m, _ = base_app
    _clip(monkeypatch, m, rate=8000, frames=8000 * 3600)

    seconds, complete = m._real_measure_reference("ignored.flac", 60.0)

    assert (seconds, complete) == (3600.0, True)
    assert FakeSoundFile.opened[0].frames_read == 0


def test_a_clip_with_no_usable_header_is_decoded_only_up_to_the_cap(base_app, monkeypatch):
    """Frames unknown (0) and an endless body: the decode itself must stop."""
    m, _ = base_app
    _clip(monkeypatch, m, rate=16000, frames=0, endless=True)

    seconds, complete = m._real_measure_reference("ignored.wav", 30.0)

    assert not complete and seconds > 30.0
    read = FakeSoundFile.opened[0].frames_read
    assert 30 * 16000 < read <= 30 * 16000 + 1, f"read {read} frames for a 30 s cap"


def test_a_short_clip_is_measured_exactly(base_app, monkeypatch):
    m, _ = base_app
    _clip(monkeypatch, m, rate=16000, frames=16000 * 5)

    assert m._real_measure_reference("ignored.wav", 60.0) == (5.0, True)


def test_a_container_soundfile_does_not_read_goes_to_librosa_with_a_limited_decode(base_app, monkeypatch):
    m, _ = base_app

    def unreadable(path):
        raise RuntimeError("Format not recognised.")

    monkeypatch.setattr(m.sf, "SoundFile", unreadable, raising=False)
    calls = []
    librosa = types.ModuleType("librosa")

    def load(path, sr=None, mono=True, duration=None):
        calls.append({"sr": sr, "duration": duration})
        return np.zeros(int(min(duration, 200.0) * sr), dtype="float32"), sr

    librosa.load = load
    monkeypatch.setitem(sys.modules, "librosa", librosa)

    seconds, complete = m._real_measure_reference("clip.m4a", 30.0)

    assert calls == [{"sr": 16000, "duration": 31.0}], "the fallback decode was not limited"
    assert seconds == 31.0 and not complete


@needs_soundfile
def test_a_real_highly_compressible_file_is_refused_quickly(monkeypatch, scratch):
    """An hour of silence is a few dozen KB as FLAC; expanding it is the bomb."""
    sf = _real_soundfile
    m = load_app(QWEN3_TTS_REF_MAX_SECONDS="30")
    model = install_model(m, FakeQwenModel("base"), BASE_06)
    monkeypatch.setattr(m, "sf", sf)
    m._measure_reference = m._real_measure_reference
    buffer = io.BytesIO()
    with sf.SoundFile(buffer, "w", samplerate=8000, channels=1, format="FLAC", subtype="PCM_16") as out:
        silence = np.zeros(8000 * 60, dtype="int16")
        for _ in range(60):
            out.write(silence)  # 3600 s
    assert buffer.tell() < 2_000_000

    began = time.monotonic()
    r = client(m).post("/clone", data={"text": "Hallo."}, files=_audio("hour.flac", buffer.getvalue()))

    assert r.status_code == 413, r.text
    assert time.monotonic() - began < 20
    assert model.calls == []


@needs_soundfile
def test_a_real_short_wav_passes_the_real_measurement(monkeypatch, scratch):
    sf = _real_soundfile
    m = load_app()
    model = install_model(m, FakeQwenModel("base"), BASE_06)
    monkeypatch.setattr(m, "sf", sf)
    m._measure_reference = m._real_measure_reference
    buffer = io.BytesIO()
    sf.write(buffer, np.zeros(16000 * 3, dtype="float32"), 16000, format="WAV")

    r = client(m).post("/clone", data={"text": "Hallo."}, files=_audio("ref.wav", buffer.getvalue()))

    assert r.status_code == 200, r.text
    assert model.calls[0][1]["ref_audio_bytes"] == buffer.getvalue()


# --- E4: bounded waiting ---------------------------------------------------------------------

class Gate:
    """Makes the model's `generate_custom_voice` block until released (a worker thread inside the model)."""

    def __init__(self, model):
        self.release, self.entered = threading.Event(), threading.Event()
        original = model.generate_custom_voice

        def blocked(*args, **kwargs):
            self.entered.set()
            assert self.release.wait(20), "the test never released the gate"
            return original(*args, **kwargs)

        model.generate_custom_voice = blocked


def _speak(http, text="Hallo."):
    return http.post("/tts", json={"text": text, "speaker": "Vivian"})


def _idle(m) -> bool:
    return (m._admitted == 0 and m._inflight == 0 and m._active_requests == 0
            and m._GEN_SEM._value == m._GEN_CONCURRENCY and not m._LOAD_LOCK.locked())


async def _until(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.005)
    return bool(predicate())


def _http(m):
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=m.app), base_url="http://test")


@pytest.fixture
def busy(monkeypatch, scratch):
    """busy(**env) -> (app, model, gate): a resident CustomVoice model that blocks on `gate`."""
    def _busy(**env):
        m = load_app(**env)
        capture_wav(monkeypatch, m)
        model = install_model(m, FakeQwenModel("custom_voice"), CUSTOM_17)
        return m, model, Gate(model)

    return _busy


def test_a_request_waits_at_most_the_queue_timeout_and_the_accounting_returns_to_zero(busy):
    m, model, gate = busy(TTS_QUEUE_TIMEOUT_S="0.4")

    async def main():
        async with _http(m) as http:
            first = asyncio.create_task(_speak(http, "Erster."))
            assert await asyncio.to_thread(gate.entered.wait, 5)
            began = time.monotonic()
            second = await _speak(http, "Zweiter.")
            waited = time.monotonic() - began
            inflight_meanwhile = m._inflight
            gate.release.set()
            first_response = await first
            assert await _until(lambda: _idle(m)), "a permit, a hold on the model or a ticket was leaked"
            third = await _speak(http, "Dritter.")
            return second, waited, inflight_meanwhile, first_response, third

    second, waited, inflight_meanwhile, first_response, third = asyncio.run(main())

    assert second.status_code == 503
    assert int(second.headers["Retry-After"]) >= 1
    assert "TTS_QUEUE_TIMEOUT_S" in second.json()["detail"]
    assert 0.3 < waited < 5
    assert inflight_meanwhile >= 2, "the waiting request should have held the model until it gave up"
    assert first_response.status_code == 200 and third.status_code == 200
    assert [call["text"] for name, call in model.calls if name == "generate_custom_voice"] == ["Erster.", "Dritter."]
    assert _idle(m)


def test_requests_beyond_the_queue_depth_are_turned_away_at_once(busy):
    m, model, gate = busy(TTS_QUEUE_TIMEOUT_S="30", TTS_MAX_QUEUE="1")

    async def main():
        async with _http(m) as http:
            running = asyncio.create_task(_speak(http, "Läuft."))
            assert await asyncio.to_thread(gate.entered.wait, 5)
            queued = asyncio.create_task(_speak(http, "Wartet."))
            assert await _until(lambda: m._admitted == 2)
            began = time.monotonic()
            excess = await _speak(http, "Zu viel.")
            turned_away_after = time.monotonic() - began
            gate.release.set()
            first, second = await running, await queued
            assert await _until(lambda: _idle(m))
            return excess, turned_away_after, first, second

    excess, turned_away_after, first, second = asyncio.run(main())

    assert excess.status_code == 503 and "TTS_MAX_QUEUE" in excess.json()["detail"]
    assert int(excess.headers["Retry-After"]) >= 1
    assert turned_away_after < 1.0, "the excess request waited instead of being shed"
    assert first.status_code == 200 and second.status_code == 200
    assert [call["text"] for name, call in model.calls if name == "generate_custom_voice"] == ["Läuft.", "Wartet."]


def test_a_full_service_refuses_an_upload_before_copying_it(busy, monkeypatch, scratch):
    m, model, gate = busy(TTS_QUEUE_TIMEOUT_S="30", TTS_MAX_QUEUE="0")
    copies = []
    real_spool = m._spool_upload

    async def counting(upload, limit=None):
        copies.append(upload.filename)
        return await real_spool(upload, limit)

    monkeypatch.setattr(m, "_spool_upload", counting)

    async def main():
        async with _http(m) as http:
            running = asyncio.create_task(_speak(http))
            assert await asyncio.to_thread(gate.entered.wait, 5)
            refused = await http.post("/clone", data={"text": "Klon."}, files=_audio())
            gate.release.set()
            await running
            assert await _until(lambda: _idle(m))
            return refused

    refused = asyncio.run(main())

    assert refused.status_code == 503
    assert copies == [], "the reference was copied for a request that was going to be turned away"
    assert _leaked(scratch) == []


def test_a_waiter_that_goes_away_leaves_no_trace(busy):
    m, model, gate = busy(TTS_QUEUE_TIMEOUT_S="30", TTS_MAX_QUEUE="4")

    async def main():
        async with _http(m) as http:
            running = asyncio.create_task(_speak(http, "Läuft."))
            assert await asyncio.to_thread(gate.entered.wait, 5)
            gone = asyncio.create_task(_speak(http, "Geht."))
            assert await _until(lambda: m._admitted == 2)
            gone.cancel()  # the client disconnected while queued
            with pytest.raises(asyncio.CancelledError):
                await gone
            assert await _until(lambda: m._admitted == 1), "the cancelled waiter kept its ticket"
            holds_after_cancel = m._inflight
            gate.release.set()
            await running
            assert await _until(lambda: _idle(m))
            follow_up = await _speak(http, "Danach.")
            return holds_after_cancel, follow_up

    holds_after_cancel, follow_up = asyncio.run(main())

    assert holds_after_cancel >= 1  # the running request's own holds, none of the cancelled one's
    assert follow_up.status_code == 200
    assert [call["text"] for name, call in model.calls if name == "generate_custom_voice"] == ["Läuft.", "Danach."]
    assert _idle(m)


def test_a_request_waiting_for_a_model_load_gives_up_with_a_503_and_is_not_counted(monkeypatch, scratch):
    m = load_app(TTS_QUEUE_TIMEOUT_S="0.4")
    m._desired_model_name = CUSTOM_17  # what a cold start loads
    capture_wav(monkeypatch, m)
    loading, release = threading.Event(), threading.Event()

    def slow_load(name, **kw):
        loading.set()
        assert release.wait(20)
        return FakeQwenModel("custom_voice")

    install_qwen_tts(monkeypatch, from_pretrained=slow_load)

    async def main():
        async with _http(m) as http:
            first = asyncio.create_task(_speak(http, "Erster."))
            assert await asyncio.to_thread(loading.wait, 5)
            began = time.monotonic()
            second = await _speak(http, "Zweiter.")
            waited = time.monotonic() - began
            release.set()
            first_response = await first
            assert await _until(lambda: _idle(m))
            return second, waited, first_response

    second, waited, first_response = asyncio.run(main())

    assert second.status_code == 503 and "Retry-After" in second.headers
    assert 0.3 < waited < 5
    assert first_response.status_code == 200, "the request that did the load must not be affected"
    assert _idle(m)


def test_the_reaper_still_sees_a_waiting_request_as_holding_the_model(busy):
    """The idle reaper must not unload weights while a request that got the model waits for a permit."""
    m, model, gate = busy(TTS_QUEUE_TIMEOUT_S="30", TTS_MODEL_TTL="0.05")

    async def main():
        async with _http(m) as http:
            running = asyncio.create_task(_speak(http, "Läuft."))
            assert await asyncio.to_thread(gate.entered.wait, 5)
            queued = asyncio.create_task(_speak(http, "Wartet."))
            assert await _until(lambda: m._admitted == 2)
            await asyncio.sleep(0.15)  # far past the TTL
            unload_allowed = m._should_unload()
            resident = m.tts_model is not None
            gate.release.set()
            await running, await queued
            return unload_allowed, resident

    unload_allowed, resident = asyncio.run(main())

    assert not unload_allowed and resident


def test_the_queue_limits_have_defaults_that_follow_the_concurrency_and_survive_junk(caplog):
    defaults = load_app()
    assert defaults.QUEUE_TIMEOUT_S == 60.0 and defaults.MAX_QUEUE == 4
    assert load_app(TTS_MAX_CONCURRENCY="3").MAX_QUEUE == 12
    assert load_app(TTS_MAX_QUEUE="0").MAX_QUEUE == 0
    with caplog.at_level("WARNING"):
        junk = load_app(TTS_QUEUE_TIMEOUT_S="soon", TTS_MAX_QUEUE="-3")
    assert junk.QUEUE_TIMEOUT_S == 60.0 and junk.MAX_QUEUE == 4
    assert "TTS_QUEUE_TIMEOUT_S" in caplog.text and "TTS_MAX_QUEUE" in caplog.text
