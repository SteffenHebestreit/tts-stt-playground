"""Tests for qwen3-tts's idle-unload logic.

This service cannot use the shared ModelSlot pattern, because its model identity
can change at runtime via /load_model. It therefore has a bespoke reaper, and
bespoke code with a two-line safety property is exactly what needs pinning down:

- it must never unload while a generation is in flight;
- after unloading, it must reload the model the operator SWITCHED to, not the
  environment default — otherwise an idle period silently reverts their choice.

The model stack is stubbed; these run offline.
"""

import asyncio
import sys
import types
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest

pytest.importorskip("numpy", reason="qwen3-tts app.py requires numpy")

SERVICE_DIR = Path(__file__).resolve().parents[1] / "qwen3-tts-service"


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
        sf.info = lambda p: types.SimpleNamespace(format="WAV", samplerate=16000, channels=1, frames=0)
        sys.modules["soundfile"] = sf

    if "uvicorn" not in sys.modules:
        uv = types.ModuleType("uvicorn")
        uv.run = lambda *a, **k: None
        sys.modules["uvicorn"] = uv


@pytest.fixture(scope="module")
def app_mod():
    _install_stubs()
    sys.path.insert(0, str(SERVICE_DIR))
    try:
        spec = spec_from_file_location("qwen3_tts_under_test", SERVICE_DIR / "app.py")
        module = module_from_spec(spec)
        assert spec.loader is not None
        sys.modules["qwen3_tts_under_test"] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(SERVICE_DIR))


@pytest.fixture
def clean(app_mod):
    """Reset the module-level residency state around each test."""
    saved = (app_mod.tts_model, app_mod.model_loaded, app_mod.current_model_name,
             app_mod._desired_model_name, app_mod._inflight, app_mod.MODEL_TTL,
             app_mod._last_used)
    yield app_mod
    (app_mod.tts_model, app_mod.model_loaded, app_mod.current_model_name,
     app_mod._desired_model_name, app_mod._inflight, app_mod.MODEL_TTL,
     app_mod._last_used) = saved


# --- the safety property ----------------------------------------------------


def test_never_unloads_while_a_generation_is_in_flight(clean):
    """The whole point of the in-flight counter. A timestamp check alone would
    free the weights out from under a running worker thread."""
    m = clean
    m.tts_model = object()
    m.MODEL_TTL = 1.0
    m._last_used = 0.0          # long past the TTL
    m._inflight = 1

    assert m._should_unload(now=10_000.0) is False, "must not unload during generation"

    m._inflight = 0
    assert m._should_unload(now=10_000.0) is True, "must unload once idle"


def test_does_not_unload_before_the_ttl_elapses(clean):
    m = clean
    m.tts_model = object()
    m.MODEL_TTL = 300.0
    m._inflight = 0
    m._last_used = 1000.0

    assert m._should_unload(now=1100.0) is False   # only 100s idle
    assert m._should_unload(now=1300.0) is True    # exactly at the TTL


def test_ttl_negative_disables_unloading(clean):
    m = clean
    m.tts_model = object()
    m._inflight = 0
    m._last_used = 0.0
    m.MODEL_TTL = -1

    assert m._should_unload(now=10_000.0) is False


def test_ttl_zero_unloads_as_soon_as_idle(clean):
    m = clean
    m.tts_model = object()
    m._inflight = 0
    m._last_used = 1000.0
    m.MODEL_TTL = 0

    assert m._should_unload(now=1000.0) is True


def test_nothing_to_unload_when_not_resident(clean):
    m = clean
    m.tts_model = None
    m.MODEL_TTL = 1.0
    m._inflight = 0
    m._last_used = 0.0

    assert m._should_unload(now=10_000.0) is False


def test_touch_resets_the_idle_countdown(clean):
    m = clean
    m.tts_model = object()
    m.MODEL_TTL = 300.0
    m._inflight = 0
    m._last_used = 0.0

    assert m._should_unload() is True
    m._touch_model()
    assert m._should_unload() is False, "a fresh request must restart the countdown"


# --- unload itself ----------------------------------------------------------


def test_unload_clears_state_and_is_idempotent(clean):
    m = clean
    m.tts_model = object()
    m.model_loaded = True
    m.current_model_name = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"

    m._unload_qwen3_tts()
    assert m.tts_model is None
    assert m.model_loaded is False
    assert m.current_model_name == ""

    m._unload_qwen3_tts()          # must not raise on a second call
    assert m.tts_model is None


# --- the switched-model trap ------------------------------------------------


def test_reload_restores_the_switched_model_not_the_env_default(clean, monkeypatch):
    """The trap this design exists to avoid.

    After /load_model switches to VoiceDesign, an idle unload followed by a
    reload must bring VoiceDesign back. Reloading the env default instead would
    silently revert the operator's choice with no error anywhere.
    """
    m = clean
    switched_to = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
    monkeypatch.setenv("QWEN3_TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-0.6B-Base")

    requested = []

    def fake_load(model_name=None):
        requested.append(model_name)
        m.tts_model = object()
        m.current_model_name = model_name or "env-default"
        return m.tts_model

    monkeypatch.setattr(m, "load_model", fake_load)

    # State after a switch, then an idle unload.
    m._desired_model_name = switched_to
    m.tts_model = None
    m.current_model_name = ""

    model, name = asyncio.run(m._acquire_model())

    assert requested == [switched_to], (
        f"reload asked for {requested}, expected the switched-to model"
    )
    assert model is not None
    assert name == switched_to


def test_reload_uses_the_env_default_when_nothing_was_switched(clean, monkeypatch):
    m = clean
    requested = []

    def fake_load(model_name=None):
        requested.append(model_name)
        m.tts_model = object()
        m.current_model_name = model_name or "env-default"
        return m.tts_model

    monkeypatch.setattr(m, "load_model", fake_load)

    m._desired_model_name = None
    m.tts_model = None
    m.current_model_name = ""

    asyncio.run(m._acquire_model())

    assert requested == [None], "with no explicit switch, load_model picks the env default"


# --- poll interval ----------------------------------------------------------


def test_reaper_tick_is_bounded(clean):
    m = clean
    for ttl, lo, hi in [(0, 1.0, 30.0), (4, 1.0, 30.0), (300, 1.0, 30.0), (100000, 1.0, 30.0)]:
        m.MODEL_TTL = ttl
        tick = m._reaper_tick_seconds()
        assert lo <= tick <= hi, f"tick {tick} out of bounds for ttl {ttl}"
