"""piper-tts-service under load and bad input: timeouts, concurrency, upload limits.

The stand-in `piper` binary can hang, fail or run slowly, so the process handling
is exercised for real: what is killed, what is left on disk, how many run at once.
"""

from __future__ import annotations

import asyncio
import io
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import httpx
import numpy as np
import pytest

from test_piper_tts_harness import (
    WAV_BYTES,
    asgi_post_json,
    install_voice,
    process_alive,
    start,  # noqa: F401  (fixture)
)

CUSTOM_CONFIG = {
    "audio": {"sample_rate": 22050, "quality": "medium"},
    "model_card": {"language": "de", "speaker": "luna"},
    "phoneme_id_map": {"<pad>": 0, "a": 1},
}


def _upload(svc, name="luna", model=b"\x08\x07onnx", config=CUSTOM_CONFIG, **form):
    files = {"model_file": ("m.onnx", model, "application/octet-stream")}
    if config is not None:
        payload = config if isinstance(config, (bytes, str)) else json.dumps(config)
        files["config_file"] = ("m.json", payload, "application/json")
    return svc.client.post("/upload_model", files=files, data={"voice_name": name, **form})


# --- piper subprocess: timeout, failure, disconnect --------------------------------------

def test_a_piper_that_hangs_is_killed_at_the_timeout(start):
    svc = start(PIPER_TIMEOUT_S="1")
    svc.fake.mode = "hang"

    began = time.monotonic()
    r = svc.speak()
    elapsed = time.monotonic() - began

    assert r.status_code == 504
    assert "PIPER_TIMEOUT_S" in r.json()["detail"]
    assert 0.9 < elapsed < 10
    assert not process_alive(svc.fake.pid()), "the timed-out piper process was left running"
    assert list(svc.out.iterdir()) == [], "the partial WAV was not removed"

    svc.fake.mode = "ok"
    assert svc.speak().status_code == 200, "the timed-out request kept its slot"


def test_a_failing_piper_reports_its_message_and_leaves_no_file(start):
    svc = start()
    svc.fake.mode = "fail"

    r = svc.speak()

    assert r.status_code == 500
    assert "boom: model exploded" in r.json()["detail"]
    assert list(svc.out.iterdir()) == []


def test_a_successful_request_leaves_no_file_after_the_download(start):
    svc = start()

    r = svc.speak()

    assert r.status_code == 200 and r.content == WAV_BYTES
    assert list(svc.out.iterdir()) == []


def test_a_client_that_hangs_up_gets_its_piper_killed(start):
    svc = start(PIPER_TIMEOUT_S="30")
    svc.fake.mode = "hang"

    messages = asyncio.run(asgi_post_json(
        svc.module.app, "/tts", {"text": "Guten Tag"},
        disconnect_when=lambda: svc.fake.pid() is not None,
    ))

    pid = svc.fake.pid()
    assert pid is not None, "piper was never started"
    assert not process_alive(pid), "piper kept running for a client that was gone"
    assert list(svc.out.iterdir()) == []
    assert messages and messages[0]["status"] == 499


# --- concurrency -------------------------------------------------------------------------

def _speak_concurrently(svc, count):
    with ThreadPoolExecutor(max_workers=count) as pool:
        return list(pool.map(lambda _: svc.speak(), range(count)))


def test_synthesis_is_limited_to_the_configured_concurrency(start):
    svc = start(PIPER_MAX_CONCURRENCY="1", PIPER_TIMEOUT_S="30")
    svc.fake.mode = "slow"

    responses = _speak_concurrently(svc, 3)

    assert [r.status_code for r in responses] == [200, 200, 200]
    assert svc.fake.peak_concurrency() == 1


def test_a_higher_limit_really_runs_requests_in_parallel(start):
    svc = start(PIPER_MAX_CONCURRENCY="2", PIPER_TIMEOUT_S="30")
    svc.fake.mode = "slow"

    responses = _speak_concurrently(svc, 4)

    assert all(r.status_code == 200 for r in responses)
    assert svc.fake.peak_concurrency() == 2


def test_a_saturated_service_answers_503_with_retry_after(start):
    svc = start(PIPER_MAX_CONCURRENCY="1", PIPER_TIMEOUT_S="1")

    async def scenario():
        slot = svc.module._slots()
        await slot.acquire()  # the only slot is busy, and stays busy
        try:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=svc.module.app), base_url="http://test"
            ) as client:
                return await client.post("/tts", json={"text": "Hallo"})
        finally:
            slot.release()

    r = asyncio.run(scenario())

    assert r.status_code == 503
    assert int(r.headers["Retry-After"]) >= 1
    assert svc.fake.calls() == []


def _slow_custom_voice(svc, monkeypatch, seconds, seen):
    """Register a custom voice whose 'inference' takes a while, and count overlaps."""
    assert _upload(svc).status_code == 200
    state = {"running": 0, "peak": 0}
    lock = threading.Lock()

    def infer(*_args):
        with lock:
            state["running"] += 1
            state["peak"] = max(state["peak"], state["running"])
        try:
            time.sleep(seconds)
        finally:
            with lock:
                state["running"] -= 1
        return WAV_BYTES

    monkeypatch.setattr(svc.module, "_custom_onnx_infer", infer)
    seen.append(state)


def test_custom_voice_inference_shares_the_concurrency_limit(start, monkeypatch):
    svc = start(PIPER_MAX_CONCURRENCY="1", PIPER_TIMEOUT_S="30")
    seen: list = []
    _slow_custom_voice(svc, monkeypatch, 0.3, seen)

    with ThreadPoolExecutor(max_workers=3) as pool:
        responses = list(pool.map(
            lambda _: svc.client.post("/synthesize", json={"text": "Hallo", "voice_name": "luna"}), range(3)))

    assert [r.status_code for r in responses] == [200, 200, 200]
    assert seen[0]["peak"] == 1


def test_custom_voice_inference_that_overruns_answers_504(start, monkeypatch):
    svc = start(PIPER_TIMEOUT_S="1")
    _slow_custom_voice(svc, monkeypatch, 1.6, [])

    r = svc.client.post("/synthesize", json={"text": "Hallo", "voice_name": "luna"})

    assert r.status_code == 504
    assert "PIPER_TIMEOUT_S" in r.json()["detail"]


# --- /analyze_audio ----------------------------------------------------------------------

def test_audio_analysis_does_not_block_the_event_loop(start, monkeypatch):
    svc = start()
    threads = {}

    def slow_load(path, sr=None):
        threads["load"] = threading.get_ident()
        time.sleep(0.8)
        return np.zeros(100, dtype="float32"), 22050

    monkeypatch.setattr(svc.module.librosa, "load", slow_load)

    async def scenario():
        loop_thread = threading.get_ident()
        longest_stall = 0.0

        async def watchdog():
            nonlocal longest_stall
            last = time.monotonic()
            while True:
                await asyncio.sleep(0.02)
                now = time.monotonic()
                longest_stall = max(longest_stall, now - last - 0.02)
                last = now

        dog = asyncio.ensure_future(watchdog())
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=svc.module.app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/analyze_audio", files={"audio_file": ("a.wav", WAV_BYTES, "audio/wav")})
        dog.cancel()
        return response, loop_thread, longest_stall

    response, loop_thread, longest_stall = asyncio.run(scenario())

    assert response.status_code == 200
    assert response.json()["librosa"]["sample_rate"] == 22050
    assert threads["load"] != loop_thread, "librosa ran on the event loop thread"
    assert longest_stall < 0.4, f"the event loop stalled for {longest_stall:.2f}s during analysis"


def test_audio_analysis_reports_signal_statistics_from_real_librosa(start, monkeypatch, tmp_path):
    librosa = pytest.importorskip("librosa")
    sf = pytest.importorskip("soundfile")
    svc = start()
    monkeypatch.setattr(svc.module, "librosa", librosa)
    buf = io.BytesIO()
    sf.write(buf, (0.5 * np.sin(2 * np.pi * 440 * np.arange(22050) / 22050)).astype("float32"), 22050, format="WAV")

    r = svc.client.post("/analyze_audio", files={"audio_file": ("a.wav", buf.getvalue(), "audio/wav")})

    assert r.status_code == 200
    stats = r.json()["librosa"]
    assert stats["sample_rate"] == 22050
    assert stats["duration"] == pytest.approx(1.0, abs=0.01)
    assert 0.3 < stats["rms_energy"] < 0.4


def test_an_analysis_upload_over_the_limit_is_413(start):
    svc = start(MAX_ANALYZE_UPLOAD_MB="0.001")  # about 1 KB

    small = svc.client.post("/analyze_audio", files={"audio_file": ("a.wav", b"x" * 500, "audio/wav")})
    over = svc.client.post("/analyze_audio", files={"audio_file": ("a.wav", b"x" * 1500, "audio/wav")})
    far_over = svc.client.post("/analyze_audio", files={"audio_file": ("a.wav", b"x" * 3_000_000, "audio/wav")})

    assert small.status_code == 200
    assert over.status_code == 413      # caught while copying the file
    assert far_over.status_code == 413  # caught from Content-Length, before the body is parsed


# --- /upload_model -----------------------------------------------------------------------

def test_a_model_upload_over_the_limit_is_413_and_leaves_nothing_behind(start):
    svc = start(MAX_UPLOAD_MB="0.001")

    r = _upload(svc, model=b"m" * 2000)

    assert r.status_code == 413
    assert not (svc.models / "custom" / "luna").exists()
    assert svc.client.get("/voices").json()["custom_count"] == 0


def test_an_uploaded_voice_is_registered_and_synthesises(start, monkeypatch):
    svc = start()
    monkeypatch.setattr(svc.module, "_custom_onnx_infer", lambda *a: WAV_BYTES)

    r = _upload(svc)

    assert r.status_code == 200
    assert r.json()["voice_info"]["language"] == "de"
    voice_dir = svc.models / "custom" / "luna"
    assert (voice_dir / "luna.onnx").read_bytes() == b"\x08\x07onnx"
    assert sorted(p.name for p in voice_dir.iterdir()) == ["luna.json", "luna.onnx"]
    assert svc.client.get("/voices").json()["custom_count"] == 1
    synthesised = svc.client.post("/synthesize", json={"text": "Hallo", "voice_name": "luna"})
    assert synthesised.status_code == 200 and synthesised.content == WAV_BYTES


def test_an_upload_without_a_config_gets_a_generated_one(start):
    svc = start()

    r = _upload(svc, config=None)

    assert r.status_code == 200
    generated = json.loads((svc.models / "custom" / "luna" / "luna.json").read_text())
    assert generated["model_card"]["speaker"] == "luna"


def test_a_bad_config_upload_keeps_the_existing_voice_intact(start):
    svc = start()
    assert _upload(svc, model=b"good-model").status_code == 200

    for bad in ("{not json", "[1, 2]", {"audio": {"sample_rate": "fast"}}):
        r = _upload(svc, model=b"replacement", config=bad)
        assert r.status_code == 400, bad

    voice_dir = svc.models / "custom" / "luna"
    assert (voice_dir / "luna.onnx").read_bytes() == b"good-model"
    assert sorted(p.name for p in voice_dir.iterdir()) == ["luna.json", "luna.onnx"]
    assert svc.client.get("/voice/luna").status_code == 200


def test_a_failed_first_upload_does_not_leave_an_empty_voice_directory(start):
    svc = start()

    assert _upload(svc, config="{not json").status_code == 400

    assert not (svc.models / "custom" / "luna").exists()


def test_an_oversized_config_is_413(start):
    svc = start()
    huge = json.dumps({"padding": "x" * (5 * 1024 * 1024 + 10)})

    assert _upload(svc, config=huge).status_code == 413
    assert not (svc.models / "custom" / "luna").exists()


def test_upload_rejects_unsafe_voice_names(start):
    svc = start()

    for name in ("../evil", "a/b", "luna\n"):
        assert _upload(svc, name=name).status_code == 400, name
    assert not (svc.models / "custom").exists() or list((svc.models / "custom").iterdir()) == []


def test_refresh_voices_keeps_serving_custom_voices_while_it_rescans(start):
    svc = start()
    assert _upload(svc).status_code == 200

    r = svc.client.post("/refresh_voices")

    assert r.json()["custom_voices"] == 1
    assert "luna" in svc.client.get("/voices").json()["voices"]


def test_delete_removes_files_and_registration(start):
    svc = start()
    assert _upload(svc).status_code == 200

    assert svc.client.delete("/voice/luna").status_code == 200

    assert not (svc.models / "custom" / "luna").exists()
    assert svc.client.get("/voice/luna").status_code == 404


def test_default_voices_installed_next_to_custom_ones_stay_separate(start):
    svc = start()
    install_voice(svc.models, "en_GB-alan-medium")
    assert _upload(svc).status_code == 200

    listed = svc.client.post("/refresh_voices").json()

    assert listed["default_voices"] == 3 and listed["custom_voices"] == 1
