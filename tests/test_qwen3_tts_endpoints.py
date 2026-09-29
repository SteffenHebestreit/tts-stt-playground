"""HTTP behaviour of qwen3-tts: languages, /tts, speakers, bounds, error mapping.

Every test drives the real FastAPI app through a TestClient with a fake model
that follows the qwen-tts contract (see `qwen3_tts_loader`), so what is asserted
is the answer a caller gets and what the model was asked to do.

Findings these pin down:

- ``auto`` / a missing language meant English, and ``nl`` / ``pl`` were mapped to
  English by the gateway with no signal to the caller;
- `/tts` answered 400 for every failure, and on the deployment default (a Base
  model, which has no preset speakers) it could only ever fail;
- the speaker list was a hard-coded constant, most of whose names do not exist;
- no limit on text or upload size, and uploads were read whole into memory.
"""

from __future__ import annotations

import tempfile

import pytest

pytest.importorskip("numpy", reason="qwen3-tts app.py requires numpy")

from qwen3_tts_loader import (  # noqa: E402
    BASE_06, CUSTOM_06, CUSTOM_17, DESIGN_17, FakeQwenModel, capture_wav, client,
    install_model, load_app,
)

WAV = b"RIFFfake-wav"


def _audio(name="ref.wav", size=64):
    return {"file": (name, b"RIFF" + b"\0" * size, "audio/wav")}


def _calls(model, method):
    return [kwargs for name, kwargs in model.calls if name == method]


@pytest.fixture
def base_app(monkeypatch):
    """App with a resident Base model, plus the model."""
    m = load_app()
    capture_wav(monkeypatch, m)
    return m, install_model(m, FakeQwenModel("base"), BASE_06)


@pytest.fixture
def custom_app(monkeypatch):
    m = load_app()
    capture_wav(monkeypatch, m)
    return m, install_model(m, FakeQwenModel("custom_voice"), CUSTOM_17)


# --- language -----------------------------------------------------------------


@pytest.mark.parametrize("sent,expected", [
    (None, "German"),        # omitted
    ("", "German"),
    ("auto", "German"),
    ("AUTO", "German"),
    ("  auto ", "German"),
    ("German", "German"),
    ("german", "German"),
    ("de", "German"),
    ("de-DE", "German"),
    ("en", "English"),
    ("English", "English"),
    ("zh_CN", "Chinese"),
    ("fr", "French"),
    ("Spanish", "Spanish"),
])
def test_language_names_and_codes_resolve_to_the_spelling_qwen_tts_expects(base_app, sent, expected):
    m, model = base_app
    data = {"text": "Hallo Welt."}
    if sent is not None:
        data["lang"] = sent

    r = client(m).post("/clone", data=data, files=_audio())

    assert r.status_code == 200, r.text
    assert _calls(model, "generate_voice_clone")[-1]["language"] == expected
    assert r.headers["x-language"] == expected


def test_a_missing_language_is_german_on_every_endpoint(monkeypatch):
    """Not English: this platform is German-first."""
    m = load_app()
    capture_wav(monkeypatch, m)
    c = client(m)

    custom = install_model(m, FakeQwenModel("custom_voice"), CUSTOM_17)
    assert c.post("/tts", json={"text": "Hallo."}).status_code == 200
    assert _calls(custom, "generate_custom_voice")[-1]["language"] == "German"

    design = install_model(m, FakeQwenModel("voice_design"), DESIGN_17)
    r = c.post("/voice_design", json={"text": "Hallo.", "voice_description": "calm"})
    assert r.status_code == 200
    assert _calls(design, "generate_voice_design")[-1]["language"] == "German"

    base = install_model(m, FakeQwenModel("base"), BASE_06)
    r = c.post("/clone-with-ref-text", data={"text": "Hallo.", "ref_text": "Hi."}, files=_audio())
    assert r.status_code == 200
    assert _calls(base, "generate_voice_clone")[-1]["language"] == "German"


@pytest.mark.parametrize("configured,expected", [
    ("French", "French"),
    ("es", "Spanish"),
    ("Auto", "Auto"),          # the model's own inference, opted into explicitly
    ("klingon", "German"),     # invalid config falls back rather than stopping the service
    ("", "German"),
])
def test_the_default_language_is_configurable(monkeypatch, configured, expected):
    m = load_app(QWEN3_DEFAULT_LANGUAGE=configured)
    capture_wav(monkeypatch, m)
    model = install_model(m, FakeQwenModel("base"), BASE_06)

    r = client(m).post("/clone", data={"text": "Hello.", "lang": "auto"}, files=_audio())

    assert r.status_code == 200, r.text
    assert _calls(model, "generate_voice_clone")[-1]["language"] == expected


@pytest.mark.parametrize("sent", ["nl", "pl", "Dutch", "xx", "German language", "e"])
def test_an_unsupported_language_is_a_400_that_lists_the_supported_ones(base_app, sent):
    m, model = base_app

    r = client(m).post("/clone", data={"text": "Hallo.", "lang": sent}, files=_audio())

    assert r.status_code == 400, "the language was silently mapped to something else"
    detail = r.json()["detail"]
    assert sent in detail and "German" in detail and "Japanese" in detail
    assert model.calls == [], "the model was called with a language it does not speak"


@pytest.mark.parametrize("endpoint", ["/tts", "/voice_design", "/clone-with-ref-text"])
def test_every_endpoint_rejects_an_unsupported_language(monkeypatch, endpoint):
    m = load_app()
    capture_wav(monkeypatch, m)
    kind = {"/tts": ("custom_voice", CUSTOM_17), "/voice_design": ("voice_design", DESIGN_17)}.get(
        endpoint, ("base", BASE_06))
    model = install_model(m, FakeQwenModel(kind[0]), kind[1])
    c = client(m)

    if endpoint == "/clone-with-ref-text":
        r = c.post(endpoint, data={"text": "x.", "ref_text": "y.", "lang": "nl"}, files=_audio())
    elif endpoint == "/voice_design":
        r = c.post(endpoint, json={"text": "x.", "voice_description": "d", "lang": "nl"})
    else:
        r = c.post(endpoint, json={"text": "x.", "lang": "nl"})

    assert r.status_code == 400
    assert model.calls == []


# --- /tts ---------------------------------------------------------------------


def test_tts_on_a_base_model_is_an_explicit_capability_error_not_a_bad_request(base_app):
    """The deployment default is a Base model. /tts used to try reference-free
    cloning (which the library cannot do) and turn its failure into a 400."""
    m, model = base_app

    r = client(m).post("/tts", json={"text": "Hallo Welt.", "lang": "German", "speaker": "Vivian"})

    assert r.status_code == 409
    detail = r.json()["detail"]
    assert "0.6B Base" in detail, "does not name the loaded model"
    assert "built-in speakers" in detail
    assert "CustomVoice" in detail and "/clone" in detail, "does not say what to do instead"
    assert model.calls == []


def test_tts_speaks_with_the_requested_speaker_on_a_customvoice_model(custom_app):
    m, model = custom_app

    r = client(m).post("/tts", json={
        "text": "Hallo Welt.", "lang": "de", "speaker": "ryan", "instruct": "ruhig"})

    assert r.status_code == 200, r.text
    assert r.content == WAV
    assert r.headers["content-type"] == "audio/wav"
    assert r.headers["content-length"] == str(len(WAV)), "the finished body is sent whole"
    assert r.headers["x-speaker"] == "ryan" and r.headers["x-language"] == "German"
    (call,) = _calls(model, "generate_custom_voice")
    assert call == {"text": "Hallo Welt.", "speaker": "ryan", "language": "German", "instruct": "ruhig"}


def test_tts_names_the_valid_speakers_when_the_one_asked_for_does_not_exist(custom_app):
    m, model = custom_app

    r = client(m).post("/tts", json={"text": "Hallo.", "speaker": "Olivia"})

    assert r.status_code == 400
    detail = r.json()["detail"]
    assert "Olivia" in detail and "Uncle_Fu" in detail and "Ono_Anna" in detail
    assert model.calls == []


def test_a_real_failure_in_tts_is_a_500_with_the_reason(custom_app):
    m, model = custom_app
    model.fail_with = RuntimeError("CUDA out of memory")

    r = client(m).post("/tts", json={"text": "Hallo.", "speaker": "Vivian"})

    assert r.status_code == 500, "an internal failure must not look like a bad request"
    assert "CUDA out of memory" in r.json()["detail"]


@pytest.mark.parametrize("failure", [ValueError("shape mismatch"), OSError("disk"), KeyError("k")])
def test_no_unexpected_exception_is_reported_as_a_client_error(custom_app, failure):
    m, model = custom_app
    model.fail_with = failure
    r = client(m).post("/tts", json={"text": "Hallo.", "speaker": "Vivian"})
    assert r.status_code == 500


@pytest.mark.parametrize("failure", [RuntimeError("boom"), ValueError("bad shape")])
def test_a_generation_failure_in_clone_is_a_500(base_app, failure):
    m, model = base_app
    model.fail_with = failure
    r = client(m).post("/clone", data={"text": "Hallo.", "lang": "German"}, files=_audio())
    assert r.status_code == 500 and str(failure.args[0]) in r.json()["detail"]


# --- speakers -----------------------------------------------------------------


def test_the_speaker_list_comes_from_the_loaded_model(custom_app):
    m, _ = custom_app

    body = client(m).get("/speakers").json()

    assert body["speakers_source"] == "model"
    assert body["speakers"] == [
        "Aiden", "Dylan", "Eric", "Ono_Anna", "Ryan", "Serena", "Sohee", "Uncle_Fu", "Vivian"]
    assert "German" in body["languages"]


def test_a_model_speaker_the_service_has_never_heard_of_is_passed_through():
    m = load_app()
    install_model(m, FakeQwenModel("custom_voice", speakers=["zeta", "vivian"]), CUSTOM_17)
    assert client(m).get("/speakers").json()["speakers"] == ["zeta", "Vivian"]


def test_a_base_model_has_no_speakers_and_says_so(base_app):
    m, _ = base_app
    body = client(m).get("/speakers").json()
    assert body["speakers"] == [] and body["speakers_source"] == "model"


def test_status_reports_the_same_speakers(custom_app):
    m, _ = custom_app
    status = client(m).get("/status").json()
    assert status["builtin_speakers"] == client(m).get("/speakers").json()["speakers"]
    assert status["speakers_source"] == "model"
    assert status["default_language"] == "German"


def test_with_nothing_resident_a_base_target_has_no_speakers_and_is_labelled():
    m = load_app()          # default model is a Base model
    body = client(m).get("/speakers").json()
    assert body["speakers"] == [] and body["speakers_source"] == "registry"


def test_with_nothing_resident_a_customvoice_target_shows_the_labelled_fallback(monkeypatch):
    monkeypatch.setenv("QWEN3_TTS_MODEL", CUSTOM_06)   # read per call, like in the container
    m = load_app()
    body = client(m).get("/speakers").json()
    assert body["speakers_source"] == "fallback"
    assert body["speakers"] == [
        "Vivian", "Serena", "Uncle_Fu", "Dylan", "Eric", "Ryan", "Aiden", "Ono_Anna", "Sohee"]
    for invented in ("Ethan", "Olivia", "Aria", "Liam", "Nova", "Atlas", "Aurora", "Kai"):
        assert invented not in body["speakers"]


def test_listing_speakers_never_loads_a_model(monkeypatch):
    monkeypatch.setenv("QWEN3_TTS_MODEL", CUSTOM_06)
    m = load_app()
    monkeypatch.setattr(m, "load_model", lambda *a, **k: pytest.fail("a status read loaded the model"))
    c = client(m)
    assert c.get("/speakers").status_code == 200
    assert c.get("/status").status_code == 200
    assert c.get("/health").status_code == 200


def test_a_model_that_cannot_list_speakers_falls_back_instead_of_failing():
    m = load_app()

    class Opaque(FakeQwenModel):
        def get_supported_speakers(self):
            raise RuntimeError("no config")

    install_model(m, Opaque("custom_voice"), CUSTOM_06)
    body = client(m).get("/speakers").json()
    assert body["speakers_source"] == "fallback" and "Vivian" in body["speakers"]


# --- registry -----------------------------------------------------------------


def test_the_registry_offers_the_0_6b_customvoice_model():
    m = load_app()
    models = client(m).get("/models").json()["models"]
    assert CUSTOM_06 in models
    assert models[CUSTOM_06]["capabilities"] == ["tts", "custom_voice"]
    assert set(models) == {BASE_06, "Qwen/Qwen3-TTS-12Hz-1.7B-Base", CUSTOM_06, CUSTOM_17, DESIGN_17}


# --- bounds -------------------------------------------------------------------


@pytest.fixture
def small_limits(monkeypatch):
    m = load_app(MAX_TEXT_CHARS=50)
    capture_wav(monkeypatch, m)
    return m


def test_text_is_bounded_on_every_endpoint(small_limits):
    m = small_limits
    long = "a" * 51
    fine = "a" * 50
    c = client(m)
    base = install_model(m, FakeQwenModel("base"), BASE_06)

    for r in (
        c.post("/clone", data={"text": long}, files=_audio()),
        c.post("/clone-with-ref-text", data={"text": "ok.", "ref_text": long}, files=_audio()),
        c.post("/clone-with-ref-text", data={"text": long, "ref_text": "ok."}, files=_audio()),
        c.post("/voices/save", data={"name": "v", "ref_text": long}, files=_audio()),
        c.post("/voices/anything/tts", data={"text": long}),
    ):
        assert r.status_code == 413, r.text
        assert "MAX_TEXT_CHARS" in r.json()["detail"]
    assert base.calls == []

    custom = install_model(m, FakeQwenModel("custom_voice"), CUSTOM_17)
    assert c.post("/tts", json={"text": long}).status_code == 413
    assert c.post("/tts", json={"text": "ok.", "instruct": long}).status_code == 413
    assert custom.calls == []
    design = install_model(m, FakeQwenModel("voice_design"), DESIGN_17)
    assert c.post("/voice_design", json={"text": long, "voice_description": "d"}).status_code == 413
    assert c.post("/voice_design", json={"text": "ok.", "voice_description": long}).status_code == 413
    assert design.calls == []

    # exactly at the limit is accepted
    install_model(m, FakeQwenModel("custom_voice"), CUSTOM_17)
    assert c.post("/tts", json={"text": fine}).status_code == 200


@pytest.mark.parametrize("text", ["", "   ", "\n\t"])
def test_blank_text_is_a_400(custom_app, text):
    m, model = custom_app
    assert client(m).post("/tts", json={"text": text}).status_code == 400
    assert model.calls == []


def test_the_default_text_limit_is_5000_characters(custom_app):
    m, _ = custom_app
    assert m.MAX_TEXT_CHARS == 5000
    c = client(m)
    assert c.post("/tts", json={"text": "a" * 5000}).status_code == 200
    assert c.post("/tts", json={"text": "a" * 5001}).status_code == 413


def test_a_malformed_limit_falls_back_to_the_default_instead_of_stopping_the_service():
    m = load_app(MAX_TEXT_CHARS="lots", MAX_UPLOAD_MB="-3", TTS_MAX_BATCH="0")
    assert m.MAX_TEXT_CHARS == 5000
    assert m.MAX_UPLOAD_BYTES == 20 * 1024 * 1024
    assert m.TTS_MAX_BATCH == 8


@pytest.fixture
def tmp_dir(monkeypatch, tmp_path):
    """Temp files land here, so a leak is visible. Voices live beside it, not in it."""
    spool = tmp_path / "spool"
    spool.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(spool))
    return spool


@pytest.fixture
def voices(tmp_path):
    return str(tmp_path / "voices")


def test_an_oversized_upload_is_a_413_and_leaves_nothing_behind(monkeypatch, tmp_dir, voices):
    m = load_app(voices, MAX_UPLOAD_MB="0.001")           # about 1 KB
    capture_wav(monkeypatch, m)
    model = install_model(m, FakeQwenModel("base"), BASE_06)
    c = client(m)

    for endpoint, data in (
        ("/clone", {"text": "Hallo."}),
        ("/clone-with-ref-text", {"text": "Hallo.", "ref_text": "Hi."}),
        ("/voices/save", {"name": "big", "ref_text": "Hi."}),
    ):
        r = c.post(endpoint, data=data, files=_audio(size=4096))
        assert r.status_code == 413, (endpoint, r.text)
        assert "MAX_UPLOAD_MB" in r.json()["detail"]

    assert model.calls == [], "an oversized upload reached the model"
    assert list(tmp_dir.iterdir()) == [], "temp files were left behind"


def test_an_upload_past_the_limit_is_cut_off_while_it_is_copied(monkeypatch, tmp_dir, voices):
    """A client that lies about (or omits) the size is still bounded per chunk."""
    m = load_app(voices, MAX_UPLOAD_MB="0.001")
    capture_wav(monkeypatch, m)
    monkeypatch.setattr(m, "_UPLOAD_CHUNK", 100)
    install_model(m, FakeQwenModel("base"), BASE_06)

    import asyncio
    import io
    from starlette.datastructures import UploadFile

    upload = UploadFile(file=io.BytesIO(b"\0" * 5000), filename="x.wav")
    upload.size = None            # size unknown up front
    with pytest.raises(m.HTTPException) as exc:
        asyncio.run(m._spool_upload(upload))
    assert exc.value.status_code == 413
    assert list(tmp_dir.iterdir()) == []


def test_the_declared_body_size_is_refused_before_anything_is_parsed(monkeypatch, tmp_dir, voices):
    m = load_app(voices, MAX_UPLOAD_MB="0.001")
    model = install_model(m, FakeQwenModel("base"), BASE_06)

    r = client(m).post("/clone", data={"text": "Hallo."}, files=_audio(size=3 * 1024 * 1024))

    assert r.status_code == 413
    assert "larger than" in r.json()["detail"]
    assert model.calls == []
    assert list(tmp_dir.iterdir()) == []


def test_an_oversized_json_body_is_refused_by_content_length(custom_app):
    m, model = custom_app
    r = client(m).post("/tts", content=b'{"text": "' + b"a" * 400_000 + b'"}',
                       headers={"content-type": "application/json"})
    assert r.status_code == 413
    assert model.calls == []


def test_the_default_upload_limit_is_20_mb(custom_app):
    assert custom_app[0].MAX_UPLOAD_BYTES == 20 * 1024 * 1024


def test_an_empty_upload_is_a_400_not_a_model_error(base_app, tmp_dir):
    m, model = base_app
    r = client(m).post("/clone", data={"text": "Hallo."}, files={"file": ("e.wav", b"", "audio/wav")})
    assert r.status_code == 400 and "empty" in r.json()["detail"].lower()
    assert model.calls == []
    assert list(tmp_dir.iterdir()) == []


def test_the_upload_reaches_the_model_intact_however_many_chunks_it_takes(base_app, monkeypatch, tmp_dir):
    m, model = base_app
    monkeypatch.setattr(m, "_UPLOAD_CHUNK", 7)
    payload = bytes(range(256)) * 3

    r = client(m).post("/clone", data={"text": "Hallo."},
                       files={"file": ("ref.wav", payload, "audio/wav")})

    assert r.status_code == 200, r.text
    (call,) = _calls(model, "create_voice_clone_prompt")
    assert call["ref_audio_bytes"] == payload
    assert list(tmp_dir.iterdir()) == [], "the temp file outlived the request"


@pytest.mark.parametrize("filename,suffix", [
    ("voice.M4A", ".M4A"),
    ("noext", ".wav"),
    ("../../etc/passwd", ".wav"),
    ("evil.w av;rm", ".wavrm"),
    ("x." + "a" * 100, "." + "a" * 11),
])
def test_the_temp_suffix_comes_from_the_client_name_but_is_sanitised(base_app, filename, suffix):
    m, model = base_app
    r = client(m).post("/clone", data={"text": "Hallo."}, files={"file": (filename, b"RIFF1234", "audio/wav")})
    assert r.status_code == 200, r.text
    (call,) = _calls(model, "create_voice_clone_prompt")
    assert call["ref_audio"].endswith(suffix)
    assert "/" not in call["ref_audio"].rsplit("/", 1)[-1]


# --- response shape -------------------------------------------------------------


def test_audio_responses_carry_a_content_length_and_the_headers_callers_read(base_app):
    m, _ = base_app
    r = client(m).post("/clone", data={"text": "Hallo.", "lang": "de"},
                       files=_audio("stimme.wav"))
    assert r.status_code == 200
    assert r.headers["content-length"] == str(len(WAV))
    assert r.headers["x-clone-source"] == "stimme.wav"
    assert float(r.headers["x-generation-time"]) >= 0
    assert float(r.headers["x-audio-duration"]) > 0
    assert r.headers["x-language"] == "German"


def test_a_filename_with_non_latin1_characters_does_not_break_a_finished_clone(base_app):
    m, _ = base_app
    r = client(m).post("/clone", data={"text": "Hallo."},
                       files={"file": ("голос.wav", b"RIFF1234", "audio/wav")})
    assert r.status_code == 200
    assert r.headers["x-clone-source"].endswith(".wav")


def test_header_values_cannot_carry_control_characters(app_mod_plain):
    safe = app_mod_plain._safe_header("evil\r\nX-Injected: 1\x00")
    assert "\r" not in safe and "\n" not in safe and "\x00" not in safe


@pytest.fixture
def app_mod_plain():
    return load_app()


# --- long texts are chunked on the in-context route too --------------------------


def test_clone_with_ref_text_chunks_long_texts_with_one_in_context_prompt(monkeypatch):
    m = load_app(TTS_MAX_BATCH="2")
    capture_wav(monkeypatch, m)
    model = install_model(m, FakeQwenModel("base"), BASE_06)
    text = " ".join(f"Das ist der Satz Nummer {i}." for i in range(5))

    r = client(m).post("/clone-with-ref-text",
                       data={"text": text, "ref_text": "Hallo Welt.", "lang": "German"},
                       files=_audio())

    assert r.status_code == 200, r.text
    (prompt_call,) = _calls(model, "create_voice_clone_prompt")
    assert prompt_call["x_vector_only_mode"] is False and prompt_call["ref_text"] == "Hallo Welt."
    batches = _calls(model, "generate_voice_clone")
    assert [len(b["text"]) for b in batches] == [2, 2, 1], "the text was not split into bounded batches"
    for b in batches:
        assert all(item.icl_mode and item.ref_text == "Hallo Welt." for item in b["voice_clone_prompt"])
        assert b["language"] == ["German"] * len(b["text"])


def test_a_single_sentence_is_generated_in_one_call(monkeypatch):
    m = load_app()
    capture_wav(monkeypatch, m)
    model = install_model(m, FakeQwenModel("base"), BASE_06)
    r = client(m).post("/clone-with-ref-text",
                       data={"text": "Nur ein Satz.", "ref_text": "Hallo."}, files=_audio())
    assert r.status_code == 200
    (call,) = _calls(model, "generate_voice_clone")
    assert call["text"] == "Nur ein Satz." and call["language"] == "German"
