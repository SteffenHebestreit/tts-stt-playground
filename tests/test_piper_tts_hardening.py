"""piper-tts-service hardening: what errors reveal, what an upload may do, how long audio may be.

Three groups, each driven through the HTTP surface:

* error responses carry a generic message and a request id; the detail is in the log
  under the same id (E1);
* /upload_model refuses built-in ids, counts and weighs the custom voices, validates
  the model in a killable child process, and a synthesis that overruns gives its slot
  back (E2);
* audio for /analyze_audio is bounded in duration before and while it is decoded (E3).

The upload validator is a child process running onnxruntime. Most tests replace its
command with a small script (so they need nothing installed); the tests marked as
using the real stack build actual ONNX graphs and are skipped without `onnx` and
`onnxruntime`.
"""

from __future__ import annotations

import io
import json
import os
import signal
import sys
import tempfile
import textwrap
import threading
import time
import types
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

from test_piper_tts_harness import (
    WAV_BYTES,
    install_voice,
    process_alive,
    start,  # noqa: F401  (fixture)
)

CONFIG = {
    "audio": {"sample_rate": 22050, "quality": "medium"},
    "model_card": {"language": "de", "speaker": "luna"},
    "phoneme_id_map": {"<pad>": 0, "<start>": 1, "<end>": 2, "a": 3, "b": 4},
    "phonemizer_language": "de",
}


def _upload(svc, name="luna", model=b"\x08\x07onnx", config=CONFIG, **form):
    files = {"model_file": ("m.onnx", model, "application/octet-stream")}
    if config is not None:
        payload = config if isinstance(config, (bytes, str)) else json.dumps(config)
        files["config_file"] = ("m.json", payload, "application/json")
    return svc.client.post("/upload_model", files=files, data={"voice_name": name, **form})


def _custom_dir_listing(svc) -> list:
    custom = svc.models / "custom"
    return sorted(p.name for p in custom.iterdir()) if custom.exists() else []


# --- E1: errors say what happened in general terms, and give an id -----------------------

def _assert_generic(response, status, log, secret):
    """A 5xx that names no internals, carries an id, and whose detail is in the log."""
    assert response.status_code == status, response.text
    request_id = response.headers["X-Request-ID"]
    assert len(request_id) >= 8
    assert request_id in response.json()["detail"]
    assert secret not in response.text, "internal detail reached the client"
    assert any(request_id in r.getMessage() and secret in r.getMessage() for r in log.records), \
        "the detail is not in the log under the id the client was given"


def test_a_failing_custom_synthesis_does_not_reveal_the_exception(start, monkeypatch, caplog):
    svc = start()
    assert _upload(svc).status_code == 200

    def explode(*_args):
        raise RuntimeError("cannot open /app/models/custom/luna/luna.onnx: Traceback (most recent call last)")

    monkeypatch.setattr(svc.module, "_custom_onnx_infer", explode)

    with caplog.at_level("ERROR"):
        via_tts = svc.client.post("/tts", json={"text": "Hallo", "voice": "luna"})
        via_synthesize = svc.client.post("/synthesize", json={"text": "Hallo", "voice_name": "luna"})

    _assert_generic(via_tts, 500, caplog, "/app/models/custom/luna/luna.onnx")
    _assert_generic(via_synthesize, 500, caplog, "/app/models/custom/luna/luna.onnx")
    assert via_tts.headers["X-Request-ID"] != via_synthesize.headers["X-Request-ID"]


def test_a_missing_piper_binary_is_not_described_to_the_client(start, monkeypatch, caplog):
    svc = start()
    monkeypatch.setenv("PATH", "/nonexistent")

    with caplog.at_level("ERROR"):
        r = svc.speak()

    _assert_generic(r, 500, caplog, "piper binary")


def test_a_failed_upload_does_not_reveal_the_exception_and_leaves_nothing(start, monkeypatch, caplog):
    svc = start()

    def refuse(*_args, **_kwargs):
        raise OSError(28, "No space left on device: '/app/models/custom/luna/luna.onnx'")

    monkeypatch.setattr(svc.module.os, "replace", refuse)

    with caplog.at_level("ERROR"):
        r = _upload(svc)

    _assert_generic(r, 500, caplog, "No space left on device")
    assert _custom_dir_listing(svc) == [], "a failed upload left files behind"
    assert svc.client.get("/voices").json()["custom_count"] == 0


def test_a_failed_delete_does_not_reveal_the_exception(start, monkeypatch, caplog):
    svc = start()
    assert _upload(svc).status_code == 200

    def refuse(*_args, **_kwargs):
        raise PermissionError(13, "Permission denied: '/app/models/custom/luna'")

    monkeypatch.setattr(svc.module.shutil, "rmtree", refuse)

    with caplog.at_level("ERROR"):
        r = svc.client.delete("/voice/luna")

    _assert_generic(r, 500, caplog, "Permission denied")


def test_a_signal_analysis_failure_does_not_reveal_the_exception(start, monkeypatch, caplog):
    svc = start()

    def broken(path, sr=None, duration=None):
        raise ValueError(f"cannot decode {path}: format not recognised")

    monkeypatch.setattr(svc.module.librosa, "load", broken)

    with caplog.at_level("ERROR"):
        r = svc.client.post("/analyze_audio", files={"audio_file": ("a.wav", WAV_BYTES, "audio/wav")})

    assert r.status_code == 200
    body = r.json()
    assert "cannot decode" not in r.text and tempfile.gettempdir() not in r.text
    request_id = body["librosa_error"].split("request id ")[1].rstrip(").")
    assert any(request_id in rec.getMessage() and "cannot decode" in rec.getMessage() for rec in caplog.records)


def test_an_unreadable_file_analysis_names_no_temp_path(start):
    svc = start()

    r = svc.client.post("/analyze_audio", files={"audio_file": ("a.wav", b"not audio at all", "audio/wav")})

    assert r.status_code == 200
    assert tempfile.gettempdir() not in r.text and "tmp" not in r.json().get("error", "")


def test_ready_reasons_name_no_filesystem_path(start):
    svc = start(voices=())
    svc.module.OUTPUT_DIR = str(svc.out / "gone")  # not a directory any more

    r = svc.client.get("/ready")

    assert r.status_code == 503
    problems = " ".join(r.json()["problems"])
    assert "PIPER_DATA_DIR" in problems and "PIPER_OUTPUT_DIR" in problems, "the operator lost the hint"
    assert str(svc.models) not in r.text and str(svc.out) not in r.text and "/app/" not in r.text


def test_an_overlong_voice_name_is_a_400_not_a_500_with_a_path(start):
    svc = start()

    r = _upload(svc, name="v" * 300)

    assert r.status_code == 400
    assert "/" not in r.json()["detail"]


def test_the_text_limit_error_says_how_to_raise_the_limit(start):
    svc = start(MAX_TEXT_CHARS="50")

    r = svc.speak(text="x" * 51)

    assert r.status_code == 422
    message = json.dumps(r.json()["detail"])
    assert "MAX_TEXT_CHARS" in message and "50" in message
    assert "/app" not in message


# --- E2: reserved ids ---------------------------------------------------------------------

@pytest.mark.parametrize("name", [
    "de_DE-thorsten-medium",       # installed built-in
    "de_de-thorsten-medium",       # differs in case only
    "fr_FR-siwis-medium",          # in the catalogue, model not installed here
])
def test_an_upload_cannot_take_the_id_of_a_built_in_voice(start, name):
    svc = start()

    r = _upload(svc, name=name)

    assert r.status_code == 409
    assert "built-in" in r.json()["detail"]
    assert _custom_dir_listing(svc) == []
    assert svc.client.get("/voices").json()["custom_count"] == 0


def test_a_rejected_id_does_not_replace_the_built_in_voice(start):
    svc = start()
    before = svc.client.get("/voice/de_DE-thorsten-medium").json()

    _upload(svc, name="de_DE-thorsten-medium", config={**CONFIG, "model_card": {"language": "xx", "speaker": "evil"}})
    after = svc.client.get("/voice/de_DE-thorsten-medium").json()
    svc.speak(voice="de_DE-thorsten-medium")

    assert after == before and after["model_type"] == "default"
    assert svc.model_used() == "de_DE-thorsten-medium"


def test_a_custom_voice_already_on_disk_under_a_built_in_id_does_not_shadow_it(start):
    """Voices uploaded before the check existed: the built-in wins, the custom one stays reachable."""
    svc = start()
    legacy = svc.models / "custom" / "de_DE-thorsten-medium"
    legacy.mkdir(parents=True)
    (legacy / "de_DE-thorsten-medium.onnx").write_bytes(b"legacy")
    (legacy / "de_DE-thorsten-medium.json").write_text(json.dumps(CONFIG))

    assert svc.client.post("/refresh_voices").json()["custom_voices"] == 1

    listed = svc.client.get("/voices").json()["voices"]["de_DE-thorsten-medium"]
    assert listed["model_type"] == "default" and listed["speaker"] == "thorsten"
    assert svc.speak(voice="de_DE-thorsten-medium").status_code == 200
    assert svc.fake.calls()[-1][0].startswith(str(svc.models / "default")), "the custom model was used for /tts"
    assert svc.client.delete("/voice/de_DE-thorsten-medium").status_code == 200, "the operator cannot clean it up"


# --- E2: caps -----------------------------------------------------------------------------

def test_the_number_of_custom_voices_is_capped(start):
    svc = start(PIPER_MAX_CUSTOM_VOICES="2")
    assert _upload(svc, name="one").status_code == 200
    assert _upload(svc, name="two").status_code == 200

    third = _upload(svc, name="three")

    assert third.status_code == 409 and "PIPER_MAX_CUSTOM_VOICES" in third.json()["detail"]
    assert _custom_dir_listing(svc) == ["one", "two"]
    assert _upload(svc, name="two", model=b"replacement").status_code == 200, "a re-upload is not a new voice"
    assert svc.client.delete("/voice/one").status_code == 200
    assert _upload(svc, name="three").status_code == 200, "deleting frees a place"


def test_the_disk_used_by_custom_voices_is_capped(start):
    svc = start(PIPER_MAX_CUSTOM_MB="0.01")  # about 10 KB in total
    six_kb = b"m" * 6000
    assert _upload(svc, name="one", model=six_kb).status_code == 200

    second = _upload(svc, name="two", model=six_kb)

    assert second.status_code == 413 and "PIPER_MAX_CUSTOM_MB" in second.json()["detail"]
    assert _custom_dir_listing(svc) == ["one"], "the refused upload left files behind"
    # replacing a voice is measured against what is left without the old one
    assert _upload(svc, name="one", model=b"r" * 7000).status_code == 200
    assert (svc.models / "custom" / "one" / "one.onnx").read_bytes() == b"r" * 7000


def test_a_full_disk_budget_refuses_the_next_upload_outright(start):
    svc = start(PIPER_MAX_CUSTOM_MB="0.005")
    assert _upload(svc, name="one", model=b"m" * 3000).status_code == 200
    (svc.models / "custom" / "one" / "extra.bin").write_bytes(b"x" * 3000)  # already over budget

    r = _upload(svc, name="two")

    assert r.status_code == 413
    assert _custom_dir_listing(svc) == ["one"]


# --- E2: validation in a child process ----------------------------------------------------

@pytest.fixture
def validating(start, monkeypatch, tmp_path):
    """validating(body, **env): the real `_validate_onnx`, with `body` as the validation process.

    `body` is Python source run in the child (`request` is the parsed stdin); the
    child finds a directory to leave notes in in $CHECKER_DIR (`svc.record`).
    """
    def _start(body="print(json.dumps({'ok': True}))", **env):
        svc = start(validate_models=True, **env)
        record = tmp_path / "checker"
        record.mkdir(exist_ok=True)
        monkeypatch.setenv("CHECKER_DIR", str(record))
        svc.record = record
        script = "import json, os, sys, time\nrequest = json.load(sys.stdin)\n" + textwrap.dedent(body)
        monkeypatch.setattr(svc.module, "_ONNX_CHECK_COMMAND", [sys.executable, "-c", script])
        return svc

    return _start


@pytest.mark.parametrize("code, status, fragment", [
    ("not_loadable", 400, "not a loadable ONNX model"),
    ("wrong_inputs", 400, "'text' and 'text_lengths'"),
    ("inference_failed", 400, "test synthesis"),
    ("bad_output", 400, "not audio"),
    ("runtime_missing", 503, "not available"),
])
def test_a_verdict_of_the_checker_becomes_a_clear_refusal(validating, code, status, fragment):
    svc = validating(f"print(json.dumps({{'ok': False, 'code': '{code}'}}))")

    r = _upload(svc)

    assert r.status_code == status and fragment in r.json()["detail"]
    assert _custom_dir_listing(svc) == [], "a refused model was published"
    assert svc.client.get("/voices").json()["custom_count"] == 0


def test_a_valid_verdict_publishes_the_voice(validating):
    svc = validating()

    r = _upload(svc)

    assert r.status_code == 200
    assert _custom_dir_listing(svc) == ["luna"]
    assert sorted(p.name for p in (svc.models / "custom" / "luna").iterdir()) == ["luna.json", "luna.onnx"]


def test_the_checker_runs_beside_nothing_else_and_gets_ids_from_the_config(validating):
    svc = validating("""
        here = sorted(os.listdir("."))
        open(os.path.join(os.environ["CHECKER_DIR"], "seen.json"), "w").write(
            json.dumps({"cwd": here, "model": request["model"], "ids": request["ids"]}))
        print(json.dumps({"ok": True}))
    """)
    other = svc.models / "custom" / "other"
    other.mkdir(parents=True)
    (other / "other.onnx").write_bytes(b"someone else's model")

    assert _upload(svc).status_code == 200

    seen = json.loads((svc.record / "seen.json").read_text())
    assert set(seen["cwd"]) <= {"model.onnx", "config.json"}, "the model was validated next to other files"
    assert Path(seen["model"]).name == "model.onnx"
    assert seen["ids"] == [0, 1, 2, 3, 4]


def test_a_checker_that_never_answers_is_killed_and_the_upload_refused(validating):
    svc = validating("""
        open(os.path.join(os.environ["CHECKER_DIR"], "pid"), "w").write(str(os.getpid()))
        time.sleep(120)
    """, PIPER_ONNX_VALIDATE_TIMEOUT_S="1")

    began = time.monotonic()
    r = _upload(svc)
    elapsed = time.monotonic() - began

    assert r.status_code == 400 and "PIPER_ONNX_VALIDATE_TIMEOUT_S" in r.json()["detail"]
    assert 0.9 < elapsed < 15
    assert not process_alive(int((svc.record / "pid").read_text())), "the validation process was left running"
    assert _custom_dir_listing(svc) == []


def test_a_checker_that_crashes_refuses_the_upload(validating):
    svc = validating("os.kill(os.getpid(), 11)")  # what a crash inside a native library looks like

    r = _upload(svc)

    assert r.status_code == 400 and "could not be loaded safely" in r.json()["detail"]
    assert _custom_dir_listing(svc) == []
    assert svc.client.get("/health").status_code == 200, "the service itself is fine"


def test_a_second_upload_during_a_validation_is_turned_away(validating):
    svc = validating("""
        open(os.path.join(os.environ["CHECKER_DIR"], "started"), "w").write("1")
        time.sleep(1.5)
        print(json.dumps({"ok": True}))
    """)

    with ThreadPoolExecutor(max_workers=1) as pool:
        first = pool.submit(_upload, svc, "first")
        deadline = time.monotonic() + 10
        while not (svc.record / "started").exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        second = _upload(svc, "second")
        first_result = first.result(timeout=20)

    assert second.status_code == 503 and int(second.headers["Retry-After"]) >= 1
    assert first_result.status_code == 200
    assert _custom_dir_listing(svc) == ["first"]
    assert _upload(svc, "second").status_code == 200, "the busy flag was not cleared"


def test_the_busy_flag_is_cleared_after_a_failed_upload(validating):
    svc = validating("print(json.dumps({'ok': False, 'code': 'not_loadable'}))")

    assert _upload(svc).status_code == 400
    assert _upload(svc).status_code == 400  # would be 503 if the first had kept the flag


@pytest.mark.parametrize("config", [
    {k: v for k, v in CONFIG.items() if k != "phoneme_id_map"},                                  # no map
    {**CONFIG, "phoneme_id_map": {}},                                                             # empty map
    {**CONFIG, "phoneme_id_map": {"a": "one"}},                                                   # no usable id
    {**CONFIG, "phoneme_id_map": {"a": -1}},                                                      # id out of range
    {**CONFIG, "phoneme_id_map": {str(i): i for i in range(5001)}},                               # absurdly large
    {**CONFIG, "audio": {"sample_rate": 4}},                                                      # sample rate
    {**CONFIG, "audio": {"sample_rate": 22050, "quality": "x" * 500}},                            # metadata
    {**CONFIG, "phonemizer_language": "de; rm -rf /"},                                            # goes to espeak
    {**CONFIG, "model_card": "a string"},
    {**CONFIG, "limits": {"max_input_symbols": 0}},
])
def test_a_config_the_service_would_trip_over_is_refused(validating, config):
    svc = validating()

    r = _upload(svc, config=config)

    assert r.status_code == 400, r.text
    assert _custom_dir_listing(svc) == []


def test_a_deeply_nested_config_is_a_400_not_a_crash(validating):
    svc = validating()

    r = _upload(svc, config="[" * 200_000)

    assert r.status_code == 400
    assert _custom_dir_listing(svc) == []


def test_a_staging_directory_a_crash_left_behind_is_removed_at_startup(start):
    svc = start()
    stale = svc.models / "custom" / ".upload-abandoned"
    stale.mkdir(parents=True)
    (stale / "model.onnx").write_bytes(b"x" * 100)

    from fastapi.testclient import TestClient
    with TestClient(svc.module.app):
        pass  # runs the lifespan

    assert not stale.exists()


def test_a_staging_directory_is_not_mistaken_for_a_voice(start):
    svc = start()
    staging = svc.models / "custom" / ".upload-live"
    staging.mkdir(parents=True)
    (staging / ".upload-live.json").write_text(json.dumps(CONFIG))
    (staging / ".upload-live.onnx").write_bytes(b"x")

    assert svc.client.post("/refresh_voices").json()["custom_voices"] == 0


# --- E2: a synthesis that overruns gives its slot back --------------------------------------

def test_a_timed_out_custom_synthesis_stops_its_run_and_frees_the_slot(start, monkeypatch):
    """Without a way to stop the native call, PIPER_MAX_CONCURRENCY overruns take every slot for good."""
    svc = start(PIPER_MAX_CONCURRENCY="1", PIPER_TIMEOUT_S="1")
    assert _upload(svc).status_code == 200
    started = threading.Event()
    runs = []

    class Run:  # what onnxruntime.RunOptions is to the model call
        terminate = False

    def endless(model_path, text, voice, speed, run_options=None):
        runs.append(run_options)
        started.set()
        deadline = time.monotonic() + 25
        while not run_options.terminate and time.monotonic() < deadline:
            time.sleep(0.02)
        raise RuntimeError("Exiting due to terminate flag being set to true.")

    monkeypatch.setattr(svc.module, "_new_run_options", Run)
    monkeypatch.setattr(svc.module, "_custom_onnx_infer", endless)

    began = time.monotonic()
    overran = svc.client.post("/tts", json={"text": "Hallo", "voice": "luna"})
    assert overran.status_code == 504 and time.monotonic() - began < 10
    assert runs[0].terminate is True, "the overrun run was not told to stop"

    def wait_for_the_slot():
        deadline = time.monotonic() + 10
        while svc.module._SLOTS[1]._value < 1 and time.monotonic() < deadline:
            time.sleep(0.02)
        return svc.module._SLOTS[1]._value

    # the only slot is usable again as soon as the thread has noticed
    assert wait_for_the_slot() == 1, "the timed-out synthesis kept its slot"
    assert svc.speak().status_code == 200
    for _ in range(3):  # and it stays that way, however many overruns there are
        again = svc.client.post("/synthesize", json={"text": "Hallo", "voice_name": "luna"})
        assert again.status_code == 504
        assert wait_for_the_slot() == 1
    assert svc.speak().status_code == 200


# --- E3: analysis audio is bounded in duration --------------------------------------------

def _record_decodes(monkeypatch, svc, seconds_of_audio, rate=22050):
    """A librosa.load that honours `duration` the way the real one does, and records the requests."""
    calls = []

    def load(path, sr=None, duration=None, **_kwargs):
        calls.append({"duration": duration})
        frames = int(seconds_of_audio * rate)
        if duration is not None:
            frames = min(frames, int(duration * rate))
        return np.zeros(frames, dtype="float32"), rate

    monkeypatch.setattr(svc.module.librosa, "load", load)
    return calls


def _declare(monkeypatch, svc, analysis):
    async def probe(_path):
        return dict(analysis)

    monkeypatch.setattr(svc.module, "analyze_audio_with_ffmpeg", probe)


def test_audio_that_declares_more_than_the_cap_is_refused_without_being_decoded(start, monkeypatch):
    svc = start(PIPER_ANALYZE_MAX_SECONDS="60")
    calls = _record_decodes(monkeypatch, svc, 5000)
    _declare(monkeypatch, svc, {"duration": 5000.0, "sample_rate": 8000})

    r = svc.client.post("/analyze_audio", files={"audio_file": ("a.flac", b"tiny", "audio/flac")})

    assert r.status_code == 413
    assert "PIPER_ANALYZE_MAX_SECONDS" in r.json()["detail"] and "60" in r.json()["detail"]
    assert calls == [], "the file was decoded although its header already exceeded the cap"


def test_audio_whose_header_lies_is_cut_off_while_decoding(start, monkeypatch):
    """No usable duration from the probe (or a false one): the decode itself is limited."""
    svc = start(PIPER_ANALYZE_MAX_SECONDS="2")
    calls = _record_decodes(monkeypatch, svc, 1000)
    _declare(monkeypatch, svc, {"error": "The file could not be read as audio."})

    r = svc.client.post("/analyze_audio", files={"audio_file": ("a.flac", b"tiny", "audio/flac")})

    assert r.status_code == 413
    assert calls and calls[0]["duration"] is not None and calls[0]["duration"] <= 3.0, \
        "the decode was not limited to the cap"


def test_audio_within_the_cap_is_analysed(start, monkeypatch):
    svc = start(PIPER_ANALYZE_MAX_SECONDS="2")
    calls = _record_decodes(monkeypatch, svc, 2.0)
    _declare(monkeypatch, svc, {"duration": 2.0, "sample_rate": 22050})

    r = svc.client.post("/analyze_audio", files={"audio_file": ("a.wav", b"tiny", "audio/wav")})

    assert r.status_code == 200
    assert r.json()["librosa"]["duration"] == pytest.approx(2.0)
    assert len(calls) == 1


def test_the_analysis_cap_defaults_to_ten_minutes(start):
    assert start().module.ANALYZE_MAX_SECONDS == 600.0


def test_a_real_highly_compressible_file_is_refused_quickly(start, monkeypatch):
    """An hour of silence is a few dozen KB as FLAC; decoding it whole is the bomb."""
    sf = pytest.importorskip("soundfile")
    librosa = pytest.importorskip("librosa")
    svc = start(PIPER_ANALYZE_MAX_SECONDS="30")
    monkeypatch.setattr(svc.module, "librosa", librosa)
    buffer = io.BytesIO()
    with sf.SoundFile(buffer, "w", samplerate=8000, channels=1, format="FLAC", subtype="PCM_16") as out:
        silence = np.zeros(8000 * 60, dtype="int16")
        for _ in range(60):
            out.write(silence)  # 3600 s
    assert buffer.tell() < 2_000_000

    began = time.monotonic()
    r = svc.client.post("/analyze_audio", files={"audio_file": ("hour.flac", buffer.getvalue(), "audio/flac")})

    assert r.status_code == 413, r.text
    assert time.monotonic() - began < 20
