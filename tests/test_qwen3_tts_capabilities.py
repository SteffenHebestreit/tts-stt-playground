"""Capability routing and batch bounding in qwen3-tts.

The five Qwen3-TTS variants share one class, so every generation method is
present on all of them and raises at call time on the ones that cannot do the
work. `/tts` documents this and routes on the declared `AVAILABLE_MODELS`
capabilities (answering 409, since the loaded model is what conflicts). Two
endpoints did not:

- `/voice_design` probed with ``hasattr(model, 'generate_voice_design')``, which
  is exactly the trap `/tts`'s own comment warns about;
- `/clone-with-ref-text` checked nothing at all, even though its sibling
  `/clone` does — and the gateway routes to it whenever the browser supplies a
  reference transcript.

Both turned "wrong model loaded" into a raw 500 instead of the actionable 400
telling the operator which model to switch to.

Separately, `_generate_chunks` batched *every* sentence of the request into one
forward pass, so peak VRAM scaled with the caller's text length on a card the
whole stack shares.

The model stack is stubbed (see `qwen3_tts_loader`); these run offline.
"""

from __future__ import annotations

import pytest

pytest.importorskip("numpy", reason="qwen3-tts app.py requires numpy")
import numpy as np  # noqa: E402

from fastapi import HTTPException  # noqa: E402

from qwen3_tts_loader import (  # noqa: E402
    BASE_06, CUSTOM_06, CUSTOM_17, DESIGN_17, FakeQwenModel, capture_wav, client,
    install_model, load_app,
)


@pytest.fixture(scope="module")
def app_mod():
    return load_app()


BASE = BASE_06
CUSTOM_VOICE = CUSTOM_17
VOICE_DESIGN = DESIGN_17


# --- _require_capability ------------------------------------------------------


def test_a_base_model_may_clone(app_mod):
    app_mod._require_capability(BASE, "voice_clone", "a Base model")


def test_a_custom_voice_model_may_not_clone(app_mod):
    with pytest.raises(HTTPException) as exc:
        app_mod._require_capability(CUSTOM_VOICE, "voice_clone", "a Base model")
    assert exc.value.status_code == 400


def test_the_refusal_names_the_loaded_model_and_the_fix(app_mod):
    """An operator has to learn which model is loaded and what to switch to."""
    with pytest.raises(HTTPException) as exc:
        app_mod._require_capability(CUSTOM_VOICE, "voice_clone", "a Base model (1.7B Base)")
    detail = str(exc.value.detail)
    assert "1.7B CustomVoice" in detail, "the refusal does not say which model is loaded"
    assert "1.7B Base" in detail, "the refusal does not say what to switch to"
    assert "voice clone" in detail


def test_only_the_voice_design_model_may_design(app_mod):
    app_mod._require_capability(VOICE_DESIGN, "voice_design", "the VoiceDesign model")
    for other in (BASE, CUSTOM_VOICE):
        with pytest.raises(HTTPException):
            app_mod._require_capability(other, "voice_design", "the VoiceDesign model")


def test_an_unknown_model_name_is_refused_rather_than_waved_through(app_mod):
    """`current_model_name` is blank while a load is in flight."""
    for name in ("", None, "some/model-we-never-heard-of"):
        with pytest.raises(HTTPException) as exc:
            app_mod._require_capability(name, "voice_clone", "a Base model")
        assert exc.value.status_code == 400


def test_every_declared_capability_is_reachable(app_mod):
    """A capability no variant declares would make its endpoint permanently 400."""
    declared = set()
    for info in app_mod.AVAILABLE_MODELS.values():
        declared.update(info.get("capabilities", []))
    for capability in ("tts", "voice_clone", "custom_voice", "voice_design"):
        assert capability in declared, (
            f"no model variant declares '{capability}', so the endpoint guarding "
            f"on it can never succeed"
        )


def test_the_refusal_status_and_wording_can_be_overridden(app_mod):
    with pytest.raises(HTTPException) as exc:
        app_mod._require_capability(
            BASE, "custom_voice", "a CustomVoice model", status_code=409,
            what="built-in speakers")
    assert exc.value.status_code == 409
    assert "does not support built-in speakers" in exc.value.detail
    assert "switch to a CustomVoice model" in exc.value.detail


def test_the_0_6b_customvoice_variant_is_registered_and_can_serve_speakers(app_mod):
    info = app_mod.AVAILABLE_MODELS[CUSTOM_06]
    assert "custom_voice" in info["capabilities"] and "voice_clone" not in info["capabilities"]
    app_mod._require_capability(CUSTOM_06, "custom_voice", "a CustomVoice model")


# --- every endpoint refuses the wrong variant, through the HTTP layer ----------
#
# These replaced source scans that asserted the handlers *mentioned*
# `_require_capability`. What matters is the answer a caller gets, and that the
# model is never asked to do work it cannot.


def _upload():
    return {"file": ("ref.wav", b"RIFF" + b"\0" * 64, "audio/wav")}


def _call(c, endpoint):
    """POST *endpoint* with a valid body; returns the response."""
    if endpoint == "/tts":
        return c.post("/tts", json={"text": "Hallo Welt.", "lang": "German"})
    if endpoint == "/voice_design":
        return c.post("/voice_design", json={
            "text": "Hallo Welt.", "voice_description": "deep calm voice", "lang": "German"})
    if endpoint == "/clone":
        return c.post("/clone", data={"text": "Hallo Welt.", "lang": "German"}, files=_upload())
    if endpoint == "/clone-with-ref-text":
        return c.post("/clone-with-ref-text", data={
            "text": "Hallo Welt.", "ref_text": "Hallo.", "lang": "German"}, files=_upload())
    if endpoint == "/voices/save":
        return c.post("/voices/save", data={
            "name": "wrong-model", "ref_text": "Hallo."}, files=_upload())
    raise AssertionError(endpoint)


WRONG_MODEL = [
    # (endpoint, loaded variant, model name the refusal must mention, status)
    ("/clone", "custom_voice", CUSTOM_17, "1.7B CustomVoice", 400),
    ("/clone", "voice_design", DESIGN_17, "1.7B VoiceDesign", 400),
    ("/clone-with-ref-text", "custom_voice", CUSTOM_17, "1.7B CustomVoice", 400),
    ("/clone-with-ref-text", "voice_design", DESIGN_17, "1.7B VoiceDesign", 400),
    ("/voices/save", "custom_voice", CUSTOM_17, "1.7B CustomVoice", 400),
    ("/voices/save", "voice_design", DESIGN_17, "1.7B VoiceDesign", 400),
    ("/voice_design", "base", BASE_06, "0.6B Base", 400),
    ("/voice_design", "custom_voice", CUSTOM_17, "1.7B CustomVoice", 400),
    ("/tts", "base", BASE_06, "0.6B Base", 409),
    ("/tts", "voice_design", DESIGN_17, "1.7B VoiceDesign", 409),
]


@pytest.mark.parametrize("endpoint,kind,model_name,label,status", WRONG_MODEL)
def test_the_wrong_variant_is_refused_with_the_fix_before_the_model_is_used(
        endpoint, kind, model_name, label, status):
    m = load_app()
    model = install_model(m, FakeQwenModel(kind), model_name)
    r = _call(client(m), endpoint)

    assert r.status_code == status, r.text
    detail = r.json()["detail"]
    assert label in detail, "the refusal does not say which model is loaded"
    assert "switch to" in detail.lower()
    assert model.calls == [], "the model was asked to do work its variant cannot"


@pytest.mark.parametrize("endpoint,kind,model_name", [
    ("/tts", "custom_voice", CUSTOM_17),
    ("/voice_design", "voice_design", DESIGN_17),
    ("/clone", "base", BASE_06),
    ("/clone-with-ref-text", "base", BASE_06),
])
def test_the_right_variant_is_served(monkeypatch, endpoint, kind, model_name):
    m = load_app()
    capture_wav(monkeypatch, m)
    model = install_model(m, FakeQwenModel(kind), model_name)
    r = _call(client(m), endpoint)

    assert r.status_code == 200, r.text
    assert r.headers["content-type"] == "audio/wav"
    assert model.calls, "the endpoint never reached the model"


def test_a_model_name_outside_the_registry_is_refused_not_waved_through():
    m = load_app()
    model = install_model(m, FakeQwenModel("base"), "someone/else-Base")
    r = _call(client(m), "/clone")
    assert r.status_code == 400 and "does not support" in r.json()["detail"]
    assert model.calls == []


# --- bounded generation batches ----------------------------------------------


class _FakeModel:
    """Records the batch sizes it was asked for."""

    def __init__(self, sample_rate=24000):
        self.batch_sizes: list[int] = []
        self.sample_rate = sample_rate

    def generate_voice_clone(self, text, language, voice_clone_prompt):
        assert len(language) == len(text), "language list must match the batch"
        assert len(voice_clone_prompt) == len(text), "prompt list must match the batch"
        self.batch_sizes.append(len(text))
        return [np.full(100, 0.5, dtype=np.float32) for _ in text], self.sample_rate


def test_long_texts_are_split_into_bounded_batches(app_mod, monkeypatch):
    """Peak VRAM must not scale with how much text the caller sent."""
    monkeypatch.setattr(app_mod, "TTS_MAX_BATCH", 4)
    model = _FakeModel()
    sentences = [f"Sentence number {i}." for i in range(10)]

    audio, sr = app_mod._generate_chunks(model, sentences, "German", [object()])

    assert model.batch_sizes == [4, 4, 2], (
        f"expected batches of at most 4, got {model.batch_sizes}"
    )
    assert sr == 24000
    # 10 chunks of 100 samples, with 9 gaps of 150 ms between them.
    assert len(audio) == 10 * 100 + 9 * int(24000 * 0.15)


def test_a_short_text_is_still_a_single_batch(app_mod, monkeypatch):
    monkeypatch.setattr(app_mod, "TTS_MAX_BATCH", 8)
    model = _FakeModel()
    app_mod._generate_chunks(model, ["One.", "Two.", "Three."], "English", [object()])
    assert model.batch_sizes == [3]


def test_gaps_are_sized_from_the_rate_the_model_returned(app_mod, monkeypatch):
    """A model at any other rate produced audibly wrong pauses."""
    monkeypatch.setattr(app_mod, "TTS_MAX_BATCH", 8)
    model = _FakeModel(sample_rate=16000)
    audio, sr = app_mod._generate_chunks(model, ["A.", "B."], "English", [object()])
    assert sr == 16000
    assert len(audio) == 2 * 100 + int(16000 * 0.15)


def test_gaps_span_batch_boundaries_too(app_mod, monkeypatch):
    """Concatenating per batch would drop the pause between the last sentence of
    one batch and the first of the next."""
    monkeypatch.setattr(app_mod, "TTS_MAX_BATCH", 2)
    model = _FakeModel()
    audio, sr = app_mod._generate_chunks(
        model, ["A.", "B.", "C.", "D."], "English", [object()])
    assert model.batch_sizes == [2, 2]
    assert len(audio) == 4 * 100 + 3 * int(24000 * 0.15), (
        "expected three gaps for four sentences, including one across the "
        "batch boundary"
    )


def test_batch_size_is_at_least_one(app_mod):
    """A zero or negative TTS_MAX_BATCH would make the range() loop generate
    nothing at all and return an empty array."""
    assert app_mod.TTS_MAX_BATCH >= 1
