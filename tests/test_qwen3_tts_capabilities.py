"""Capability routing and batch bounding in qwen3-tts.

The four Qwen3-TTS variants share one class, so every generation method is
present on all of them and raises at call time on the ones that cannot do the
work. `/tts` documents this and routes on the declared `AVAILABLE_MODELS`
capabilities. Two endpoints did not:

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

The model stack is stubbed; these run offline.
"""

from __future__ import annotations

import ast
import os
import sys
import tempfile
import types
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest

pytest.importorskip("numpy", reason="qwen3-tts app.py requires numpy")
import numpy as np  # noqa: E402

from fastapi import HTTPException  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
SERVICE_DIR = REPO_ROOT / "qwen3-tts-service"


def _install_stubs():
    if "torch" not in sys.modules:
        torch = types.ModuleType("torch")
        torch.__version__ = "0.0.0-stub"
        torch.bfloat16 = "bfloat16"
        torch.float32 = "float32"
        torch.cuda = types.SimpleNamespace(
            is_available=lambda: False,
            empty_cache=lambda: None,
            ipc_collect=lambda: None,
            get_device_name=lambda i: "stub",
            memory_allocated=lambda: 0,
            get_device_properties=lambda i: types.SimpleNamespace(total_memory=0),
            is_bf16_supported=lambda: False,
        )
        torch.is_tensor = lambda x: False
        sys.modules["torch"] = torch

    if "soundfile" not in sys.modules:
        sf = types.ModuleType("soundfile")
        sf.write = lambda *a, **k: None
        sf.info = lambda p: types.SimpleNamespace(
            format="WAV", samplerate=16000, channels=1, frames=0)
        sys.modules["soundfile"] = sf

    if "uvicorn" not in sys.modules:
        uv = types.ModuleType("uvicorn")
        uv.run = lambda *a, **k: None
        sys.modules["uvicorn"] = uv


@pytest.fixture(scope="module")
def app_mod():
    _install_stubs()
    previous_voices_dir = os.environ.get("VOICES_DIR")
    os.environ["VOICES_DIR"] = tempfile.mkdtemp(prefix="qwen3-tts-caps-")
    sys.path.insert(0, str(SERVICE_DIR))
    try:
        spec = spec_from_file_location("qwen3_tts_caps_under_test", SERVICE_DIR / "app.py")
        module = module_from_spec(spec)
        assert spec.loader is not None
        sys.modules["qwen3_tts_caps_under_test"] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(SERVICE_DIR))
        if previous_voices_dir is None:
            os.environ.pop("VOICES_DIR", None)
        else:
            os.environ["VOICES_DIR"] = previous_voices_dir


BASE = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
CUSTOM_VOICE = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
VOICE_DESIGN = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"


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


# --- endpoints route on declared capabilities, not hasattr --------------------


def _function_source(name: str) -> str:
    tree = ast.parse((SERVICE_DIR / "app.py").read_text(encoding="utf-8"))
    node = next(
        (n for n in ast.walk(tree)
         if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name),
        None,
    )
    assert node is not None, f"{name} not found in qwen3-tts app.py"
    return ast.unparse(node)


@pytest.mark.parametrize("handler,capability", [
    ("clone_voice", "voice_clone"),
    ("clone_voice_with_ref_text", "voice_clone"),
    ("save_voice", "voice_clone"),
    ("tts_with_saved_voice", "voice_clone"),
    ("voice_design", "voice_design"),
])
def test_handler_checks_the_declared_capability(handler: str, capability: str):
    source = _function_source(handler)
    assert "_require_capability" in source, (
        f"{handler} does not check the loaded model's declared capabilities, so "
        f"the wrong variant produces a 500 instead of an actionable 400"
    )
    assert f'"{capability}"' in source or f"'{capability}'" in source


@pytest.mark.parametrize("handler", [
    "clone_voice", "clone_voice_with_ref_text", "save_voice",
    "tts_with_saved_voice", "voice_design",
])
def test_no_handler_probes_generation_methods_with_hasattr(handler: str):
    """The trap /tts's own comment warns about: the variants share a class, so
    the attribute exists everywhere and only raises when called."""
    source = _function_source(handler)
    assert "hasattr" not in source, (
        f"{handler} uses hasattr to decide what the model can do. Every variant "
        f"has the method; only some of them work. Route on AVAILABLE_MODELS."
    )


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
