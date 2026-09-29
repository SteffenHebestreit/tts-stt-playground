"""Saved voices in qwen3-tts: /voices/save, playback, the library on disk.

Findings these pin down:

- `/voices/save` required the Qwen3-ASR service even in x-vector mode, where the
  transcript is never used, and answered "could not transcribe" when the service
  was merely down;
- the saved metadata overwrote the transcript with null, so every voice listed
  `ref_text: null`;
- `model_used` was read from a global that another request could blank mid-save;
- the voice-id check used ``$``, which matches before a trailing newline.

In-context (ICL) cloning is the opt-in `mode=icl`. The default stays x-vector, and
voices saved before it existed load exactly as they did.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import pytest

pytest.importorskip("numpy", reason="qwen3-tts app.py requires numpy")

from qwen3_tts_loader import (  # noqa: E402
    BASE_06, CUSTOM_17, FakeQwenModel, capture_wav, client, install_model,
    install_qwen_tts, load_app, make_tensor,
)

UPLOAD = b"RIFF" + b"\x01\x02\x03" * 40


def _audio(name="ref.wav"):
    return {"file": (name, UPLOAD, "audio/wav")}


def _calls(model, method):
    return [kwargs for name, kwargs in model.calls if name == method]


@pytest.fixture
def lib(monkeypatch, tmp_path):
    """(app, resident Base model, voices dir) with the qwen_tts item class available."""
    voices = tmp_path / "voices"
    m = load_app(str(voices))
    capture_wav(monkeypatch, m)
    install_qwen_tts(monkeypatch)
    return m, install_model(m, FakeQwenModel("base"), BASE_06), voices


class Asr:
    """Stand-in for the Qwen3-ASR service."""

    def __init__(self, monkeypatch, m, text="Das ist die Referenz.", segments=None, error=None):
        self.calls = []
        self.text, self.segments, self.error = text, segments or [], error
        monkeypatch.setattr(m, "_auto_transcribe", self)
        monkeypatch.setattr(m, "_trim_audio_segment", self._trim)

    async def __call__(self, audio_path, filename="audio.wav"):
        self.calls.append((audio_path, filename))
        if self.error is not None:
            raise self.error
        return {"text": self.text, "segments": self.segments, "duration": 8.0}

    @staticmethod
    def _trim(input_path, start, end, output_path):
        Path(output_path).write_bytes(b"TRIMMED")
        return True


def _meta(voices, voice_id):
    return json.loads((voices / voice_id / "metadata.json").read_text())


# --- ASR is optional in x-vector mode --------------------------------------------


def test_an_unreachable_asr_service_does_not_stop_an_x_vector_save(lib, monkeypatch):
    m, model, voices = lib
    asr = Asr(monkeypatch, m, error=m._AsrUnavailable("ConnectError: down"))

    r = client(m).post("/voices/save", data={"name": "Anna"}, files=_audio())

    assert r.status_code == 200, r.text
    body = r.json()
    assert body["voice_id"] == "anna" and body["mode"] == "xvector"
    assert body["ref_text"] == "" and body["ref_text_source"] == "none"
    (call,) = _calls(model, "create_voice_clone_prompt")
    assert call["x_vector_only_mode"] is True and call["ref_text"] is None
    assert call["ref_audio_bytes"] == UPLOAD, "with no ASR there is nothing to trim to"
    assert (voices / "anna" / "ref_spk_embedding.pt").exists()
    assert len(asr.calls) == 1


def test_a_really_unreachable_asr_service_is_survived_quickly(monkeypatch, tmp_path):
    """Through the real httpx client, not a stand-in: the discard port refuses."""
    m = load_app(str(tmp_path / "voices"), QWEN3_ASR_SERVICE_URL="http://127.0.0.1:9")
    capture_wav(monkeypatch, m)
    install_qwen_tts(monkeypatch)
    model = install_model(m, FakeQwenModel("base"), BASE_06)

    started = time.monotonic()
    r = client(m).post("/voices/save", data={"name": "Anna"}, files=_audio())

    assert r.status_code == 200, r.text
    assert time.monotonic() - started < 5, "an unreachable service must fail fast, not hang for a minute"
    assert _calls(model, "create_voice_clone_prompt")[0]["x_vector_only_mode"] is True


def test_an_asr_service_that_hears_no_speech_still_rejects_the_recording(lib, monkeypatch):
    m, model, voices = lib
    Asr(monkeypatch, m, text="  ")

    r = client(m).post("/voices/save", data={"name": "Silent"}, files=_audio())

    assert r.status_code == 400 and "transcribe" in r.json()["detail"]
    assert model.calls == [] and not voices.exists()


def test_a_provided_transcript_means_no_asr_call_at_all(lib, monkeypatch):
    m, model, voices = lib
    asr = Asr(monkeypatch, m, error=AssertionError("ASR must not be called"))

    r = client(m).post("/voices/save", data={"name": "Anna", "ref_text": "  Hallo Welt.  "}, files=_audio())

    assert r.status_code == 200, r.text
    assert asr.calls == []
    assert r.json()["ref_text"] == "Hallo Welt." and r.json()["ref_text_source"] == "provided"
    (call,) = _calls(model, "create_voice_clone_prompt")
    assert call["ref_audio_bytes"] == UPLOAD, "a supplied transcript describes the whole clip: no trimming"
    assert _meta(voices, "anna")["ref_text"] == "Hallo Welt."


def test_the_asr_segment_is_used_to_trim_the_reference(lib, monkeypatch):
    m, model, voices = lib
    Asr(monkeypatch, m, text="Eins. Zwei drei vier fuenf.", segments=[
        {"start": 0.0, "end": 1.0, "text": "Eins."},
        {"start": 1.0, "end": 6.0, "text": "Zwei drei vier fuenf."},
    ])

    r = client(m).post("/voices/save", data={"name": "Anna"}, files=_audio())

    assert r.status_code == 200, r.text
    (call,) = _calls(model, "create_voice_clone_prompt")
    assert call["ref_audio"].endswith("_trimmed.wav") and call["ref_audio_bytes"] == b"TRIMMED"
    assert r.json()["ref_text"] == "Zwei drei vier fuenf."
    assert r.json()["ref_text_source"] == "asr"


def test_the_saved_metadata_keeps_the_transcript(lib, monkeypatch):
    """`_save_voice_prompt` overwrote it with the prompt item's own text, which is
    None in x-vector mode."""
    m, _, voices = lib
    Asr(monkeypatch, m, text="Das ist die Referenz.")

    assert client(m).post("/voices/save", data={"name": "Anna"}, files=_audio()).status_code == 200

    meta = _meta(voices, "anna")
    assert meta["ref_text"] == "Das ist die Referenz."
    assert meta["x_vector_only_mode"] is True and meta["icl_mode"] is False
    listed = client(m).get("/voices").json()["voices"]
    assert [(v["id"], v["ref_text"]) for v in listed] == [("anna", "Das ist die Referenz.")]


def test_model_used_is_the_model_this_request_actually_used(lib, monkeypatch):
    """Read from the global at the end of the request, it came out blank when
    another actor had cleared it in the meantime."""
    m, _, voices = lib

    async def asr_that_sees_the_globals_cleared(audio_path, filename="a.wav"):
        m.current_model_name = ""
        return {"text": "Hallo Welt.", "segments": []}

    monkeypatch.setattr(m, "_auto_transcribe", asr_that_sees_the_globals_cleared)

    assert client(m).post("/voices/save", data={"name": "Anna"}, files=_audio()).status_code == 200
    assert _meta(voices, "anna")["model_used"] == BASE_06


def test_a_save_waiting_on_the_asr_service_keeps_its_model_resident(monkeypatch, tmp_path):
    """The reaper race end to end: real lifespan, real reaper, a TTL far shorter
    than the ASR wait. The model must survive the request that is using it."""
    voices = tmp_path / "voices"
    m = load_app(str(voices), TTS_MODEL_TTL="0.3")
    loads = []
    install_qwen_tts(monkeypatch, from_pretrained=lambda name, **kw: (
        loads.append(name), FakeQwenModel("base"))[1])
    monkeypatch.setattr(m, "_reaper_tick_seconds", lambda: 0.02)
    seen = {}

    async def slow_asr(audio_path, filename="a.wav"):
        await asyncio.sleep(1.0)           # more than three TTLs
        seen["resident"] = m.tts_model is not None
        return {"text": "Hallo Welt.", "segments": []}

    monkeypatch.setattr(m, "_auto_transcribe", slow_asr)

    from fastapi.testclient import TestClient
    with TestClient(m.app, raise_server_exceptions=False) as c:
        for _ in range(200):
            if c.get("/health").json()["model_resident"]:
                break
            time.sleep(0.02)
        r = c.post("/voices/save", data={"name": "Wait"}, files=_audio())

    assert r.status_code == 200, r.text
    assert seen["resident"] is True, "the reaper unloaded the model while the request was using it"
    assert _meta(voices, "wait")["model_used"] == BASE_06


# --- in-context mode ---------------------------------------------------------------


def test_icl_mode_stores_the_reference_codes_and_the_transcript(lib, monkeypatch):
    m, model, voices = lib
    asr = Asr(monkeypatch, m, error=AssertionError("ASR must not be called"))

    r = client(m).post("/voices/save", data={
        "name": "Anna", "mode": "icl", "ref_text": "Hallo Welt."}, files=_audio())

    assert r.status_code == 200, r.text
    assert r.json()["mode"] == "icl" and asr.calls == []
    (call,) = _calls(model, "create_voice_clone_prompt")
    assert call["x_vector_only_mode"] is False and call["ref_text"] == "Hallo Welt."
    assert (voices / "anna" / "ref_code.pt").exists()
    meta = _meta(voices, "anna")
    assert meta["icl_mode"] is True and meta["x_vector_only_mode"] is False
    assert meta["ref_text"] == "Hallo Welt."


def test_an_icl_voice_is_played_back_in_context(lib, monkeypatch):
    m, model, _ = lib
    c = client(m)
    assert c.post("/voices/save", data={
        "name": "Anna", "mode": "icl", "ref_text": "Hallo Welt."}, files=_audio()).status_code == 200

    r = c.post("/voices/anna/tts", data={"text": "Guten Tag.", "lang": "de"})

    assert r.status_code == 200, r.text
    (call,) = _calls(model, "generate_voice_clone")
    (item,) = call["voice_clone_prompt"]
    assert item.icl_mode is True and item.x_vector_only_mode is False
    assert item.ref_text == "Hallo Welt." and item.ref_code is not None
    assert call["language"] == "German" and r.headers["x-voice-id"] == "anna"


def test_an_x_vector_voice_is_still_played_back_as_x_vector(lib, monkeypatch):
    m, model, _ = lib
    c = client(m)
    Asr(monkeypatch, m)
    assert c.post("/voices/save", data={"name": "Anna"}, files=_audio()).status_code == 200

    assert c.post("/voices/anna/tts", data={"text": "Guten Tag."}).status_code == 200

    (item,) = _calls(model, "generate_voice_clone")[0]["voice_clone_prompt"]
    assert item.x_vector_only_mode is True and item.icl_mode is False
    assert item.ref_code is None and item.ref_text is None


def test_a_voice_saved_before_icl_existed_loads_exactly_as_it_did(lib):
    m, model, voices = lib
    import torch
    legacy = voices / "old"
    legacy.mkdir(parents=True)
    torch.save(make_tensor([0.5, 0.25]), legacy / "ref_spk_embedding.pt")
    (legacy / "metadata.json").write_text(json.dumps({
        "name": "Old", "lang": "auto", "ref_text": None,
        "x_vector_only_mode": True, "icl_mode": False}))

    r = client(m).post("/voices/old/tts", data={"text": "Hallo."})

    assert r.status_code == 200, r.text
    (item,) = _calls(model, "generate_voice_clone")[0]["voice_clone_prompt"]
    assert item.x_vector_only_mode is True and item.ref_code is None and item.ref_text is None


def test_stale_reference_codes_do_not_turn_an_x_vector_voice_into_an_icl_one(lib):
    m, model, voices = lib
    import torch
    d = voices / "mixed"
    d.mkdir(parents=True)
    torch.save(make_tensor([0.5]), d / "ref_spk_embedding.pt")
    torch.save(make_tensor([[1, 2]]), d / "ref_code.pt")
    (d / "metadata.json").write_text(json.dumps({"ref_text": "Hallo.", "icl_mode": False}))

    assert client(m).post("/voices/mixed/tts", data={"text": "Hallo."}).status_code == 200
    (item,) = _calls(model, "generate_voice_clone")[0]["voice_clone_prompt"]
    assert item.icl_mode is False and item.ref_code is None


def test_resaving_an_icl_voice_as_x_vector_removes_its_reference_codes(lib):
    m, _, voices = lib
    c = client(m)
    c.post("/voices/save", data={"name": "Anna", "mode": "icl", "ref_text": "Hallo."}, files=_audio())
    assert (voices / "anna" / "ref_code.pt").exists()

    r = c.post("/voices/save", data={"name": "Anna", "ref_text": "Hallo."}, files=_audio())

    assert r.status_code == 200
    assert not (voices / "anna" / "ref_code.pt").exists()
    assert _meta(voices, "anna")["icl_mode"] is False


def test_icl_without_a_transcript_asks_the_asr_service(lib, monkeypatch):
    m, model, voices = lib
    Asr(monkeypatch, m, text="Von der ASR.", segments=[{"start": 0.0, "end": 5.0, "text": "Von der ASR."}])

    r = client(m).post("/voices/save", data={"name": "Anna", "mode": "ICL"}, files=_audio())

    assert r.status_code == 200, r.text
    (call,) = _calls(model, "create_voice_clone_prompt")
    assert call["ref_text"] == "Von der ASR." and call["x_vector_only_mode"] is False


def test_icl_needs_a_transcript_and_says_so_when_the_asr_service_is_down(lib, monkeypatch):
    m, model, voices = lib
    Asr(monkeypatch, m, error=m._AsrUnavailable("ConnectError: down"))

    r = client(m).post("/voices/save", data={"name": "Anna", "mode": "icl"}, files=_audio())

    assert r.status_code == 502
    # names the way out, not the connection error (which carries the service's address)
    assert "ref_text" in r.json()["detail"] and "ConnectError" not in r.json()["detail"]
    assert model.calls == [] and not voices.exists()


@pytest.mark.parametrize("mode,expected", [
    ("xvector", "xvector"), ("x-vector", "xvector"), ("X_VECTOR", "xvector"), ("", "xvector"),
    ("icl", "icl"), ("ICL", "icl"), ("in-context", "icl"),
])
def test_mode_names(lib, monkeypatch, mode, expected):
    m, _, _ = lib
    Asr(monkeypatch, m)
    r = client(m).post("/voices/save", data={"name": "Anna", "mode": mode, "ref_text": "Hallo."}, files=_audio())
    assert r.status_code == 200 and r.json()["mode"] == expected


def test_an_unknown_mode_is_a_400(lib):
    m, model, _ = lib
    r = client(m).post("/voices/save", data={"name": "Anna", "mode": "clone-hard"}, files=_audio())
    assert r.status_code == 400 and "mode" in r.json()["detail"]
    assert model.calls == []


# --- voice ids ---------------------------------------------------------------------


@pytest.mark.parametrize("voice_id", ["abc%0A", "abc%0D%0A", "..%2Fetc", "a%20b", "a.b", "%0A"])
def test_a_voice_id_must_match_completely(lib, voice_id):
    """`$` matches before a trailing newline, so "abc\\n" passed the old check."""
    m, model, _ = lib
    c = client(m)
    assert c.delete(f"/voices/{voice_id}").status_code in (400, 404)
    r = c.post(f"/voices/{voice_id}/tts", data={"text": "Hallo."})
    assert r.status_code in (400, 404)
    if "%0A" in voice_id or "%0D" in voice_id:
        assert c.delete(f"/voices/{voice_id}").status_code == 400
        assert r.status_code == 400
    assert model.calls == []


def test_a_very_long_voice_name_is_a_400_not_a_filesystem_error(lib):
    m, model, _ = lib
    r = client(m).post("/voices/save", data={"name": "x" * 300, "ref_text": "Hallo."}, files=_audio())
    assert r.status_code == 400 and "too long" in r.json()["detail"]
    assert model.calls == []


def test_display_names_become_safe_ids(lib, monkeypatch):
    m, _, voices = lib
    Asr(monkeypatch, m)
    r = client(m).post("/voices/save", data={"name": "  My Voice! (Anna) "}, files=_audio())
    assert r.status_code == 200 and r.json()["voice_id"] == "my_voice_anna"
    assert _meta(voices, "my_voice_anna")["name"] == "  My Voice! (Anna) "


# --- the library -------------------------------------------------------------------


def test_playback_of_a_missing_or_partial_voice_is_a_404(lib):
    m, model, voices = lib
    (voices / "half").mkdir(parents=True)          # crashed mid-save: no embedding
    (voices / "half" / "metadata.json").write_text("{}")
    c = client(m)
    assert c.post("/voices/nobody/tts", data={"text": "Hallo."}).status_code == 404
    assert c.post("/voices/half/tts", data={"text": "Hallo."}).status_code == 404
    assert model.calls == []


def test_one_corrupt_profile_does_not_take_the_listing_down(lib):
    m, _, voices = lib
    c = client(m)
    c.post("/voices/save", data={"name": "Good", "ref_text": "Hallo."}, files=_audio())
    (voices / "broken").mkdir()
    (voices / "broken" / "metadata.json").write_text("{not json")

    listed = c.get("/voices")

    assert listed.status_code == 200
    assert [v["id"] for v in listed.json()["voices"]] == ["good"]


def test_an_empty_library_lists_nothing(lib):
    m, _, _ = lib
    assert client(m).get("/voices").json() == {"voices": []}


def test_deleting_a_voice_removes_it(lib):
    m, _, voices = lib
    c = client(m)
    c.post("/voices/save", data={"name": "Anna", "ref_text": "Hallo."}, files=_audio())
    assert c.delete("/voices/anna").json() == {"status": "ok", "deleted": "anna"}
    assert not (voices / "anna").exists()
    assert c.delete("/voices/anna").status_code == 404


def test_saved_voice_playback_on_the_wrong_variant_is_refused(monkeypatch, tmp_path):
    m = load_app(str(tmp_path / "voices"))
    install_qwen_tts(monkeypatch)
    base = install_model(m, FakeQwenModel("base"), BASE_06)
    c = client(m)
    c.post("/voices/save", data={"name": "Anna", "ref_text": "Hallo."}, files=_audio())
    custom = install_model(m, FakeQwenModel("custom_voice"), CUSTOM_17)

    r = c.post("/voices/anna/tts", data={"text": "Hallo."})

    assert r.status_code == 400 and "1.7B CustomVoice" in r.json()["detail"]
    assert custom.calls == [] and base.calls[-1][0] != "generate_voice_clone"


def test_playback_chunks_long_texts_and_applies_the_default_language(lib, monkeypatch):
    m, model, _ = lib
    monkeypatch.setattr(m, "TTS_MAX_BATCH", 2)
    c = client(m)
    c.post("/voices/save", data={"name": "Anna", "ref_text": "Hallo."}, files=_audio())
    text = " ".join(f"Das ist der Satz Nummer {i}." for i in range(5))

    r = c.post("/voices/anna/tts", data={"text": text})

    assert r.status_code == 200
    batches = _calls(model, "generate_voice_clone")
    assert [len(b["text"]) for b in batches] == [2, 2, 1]
    assert all(b["language"] == ["German"] * len(b["text"]) for b in batches)
