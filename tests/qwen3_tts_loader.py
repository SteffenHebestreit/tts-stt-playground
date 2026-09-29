"""Shared harness for the qwen3-tts-service tests.

`qwen3-tts-service/app.py` reads most of its configuration at import time
(text and upload limits, default language, idle TTL, batch size), so a test that
needs a different configuration has to import a fresh copy of the module. That,
stand-ins for torch / soundfile / qwen_tts, and a fake model that follows the
real ``Qwen3TTSModel`` contract, is all this module provides.

The fake model mirrors the parts of the real wrapper the service depends on,
read from the ``qwen-tts`` 0.1.1 wheel: every variant has every ``generate_*``
method and the ones a variant cannot serve raise ``ValueError`` at call time;
languages are validated case-insensitively against ``["auto"] + 10 names``;
speakers against the checkpoint's own list; in-context prompts need ``ref_text``.

``QWEN3_TTS_SERVICE_DIR`` points the loader at another checkout of the service,
which is how the tests are run against the pre-fix code to prove they fail
there.
"""

from __future__ import annotations

import os
import pickle
import sys
import tempfile
import types
from dataclasses import dataclass
from importlib.util import module_from_spec, spec_from_file_location
from itertools import count
from pathlib import Path
from typing import Optional

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SERVICE_DIR = Path(os.environ.get("QWEN3_TTS_SERVICE_DIR") or REPO / "qwen3-tts-service")

# Settings the service reads from the environment. Anything not passed to
# `load_app` is removed while importing, so a variable exported in the
# developer's shell cannot change what a test measures.
MANAGED_ENV = (
    "QWEN3_TTS_MODEL", "QWEN3_DEFAULT_LANGUAGE", "QWEN3_TTS_ATTN_IMPLEMENTATION",
    "MAX_TEXT_CHARS", "MAX_UPLOAD_MB", "TTS_MODEL_TTL", "MODEL_TTL",
    "TTS_MAX_CONCURRENCY", "TTS_MAX_BATCH", "VOICES_DIR", "QWEN3_ASR_SERVICE_URL",
    "ALLOWED_ORIGINS", "ALLOW_CREDENTIALS", "QWEN3_TTS_REF_MAX_SECONDS",
    "TTS_QUEUE_TIMEOUT_S", "TTS_MAX_QUEUE",
)

BASE_06 = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
BASE_17 = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
CUSTOM_06 = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
CUSTOM_17 = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
DESIGN_17 = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"

# The languages qwen-tts accepts, lower-cased as the wrapper compares them.
MODEL_LANGUAGES = {
    "auto", "chinese", "english", "japanese", "korean", "german",
    "french", "russian", "portuguese", "spanish", "italian",
}
# The checkpoint's own speaker ids are lower-case (`spk_id` keys).
CUSTOM_VOICE_SPEAKERS = [
    "aiden", "dylan", "eric", "ono_anna", "ryan", "serena", "sohee", "uncle_fu", "vivian",
]


def _is_stub(module) -> bool:
    return getattr(module, "__file__", None) is None


def _pickle_save(obj, path, *args, **kwargs):
    with open(path, "wb") as fh:
        pickle.dump(obj, fh)


def _pickle_load(path, *args, **kwargs):
    with open(path, "rb") as fh:
        return pickle.load(fh)


def install_stubs() -> None:
    """torch / soundfile / uvicorn stand-ins, only where the real package is absent.

    Other test modules install a torch stub of their own with fewer attributes,
    so the ones this service needs are filled in on whichever stub is present.
    """
    if "torch" not in sys.modules:
        sys.modules["torch"] = types.ModuleType("torch")
    torch = sys.modules["torch"]
    if _is_stub(torch):
        defaults = {
            "__version__": "0.0.0-stub",
            "bfloat16": "bfloat16",
            "float32": "float32",
            "is_tensor": lambda x: False,
            "tensor": lambda x: x,
            "save": _pickle_save,
            "load": _pickle_load,
        }
        for name, value in defaults.items():
            if not hasattr(torch, name):
                setattr(torch, name, value)
        if not hasattr(torch, "cuda"):
            torch.cuda = types.SimpleNamespace()
        cuda_defaults = {
            "is_available": lambda: False,
            "empty_cache": lambda: None,
            "ipc_collect": lambda: None,
            "get_device_name": lambda i: "stub",
            "memory_allocated": lambda: 0,
            "get_device_properties": lambda i: types.SimpleNamespace(total_memory=0),
            "is_bf16_supported": lambda: False,
        }
        for name, value in cuda_defaults.items():
            if not hasattr(torch.cuda, name):
                setattr(torch.cuda, name, value)

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


def make_tensor(values):
    """A real tensor when torch is real (so `weights_only` loading is exercised), else the values."""
    import torch

    return values if _is_stub(torch) else torch.tensor(values)


_counter = count()


def load_app(voices_dir: Optional[str] = None, **env):
    """Import a fresh copy of the service with *env* as its whole configuration."""
    install_stubs()
    previous = {key: os.environ.get(key) for key in MANAGED_ENV}
    for key in MANAGED_ENV:
        os.environ.pop(key, None)
    os.environ["VOICES_DIR"] = voices_dir or tempfile.mkdtemp(prefix="qwen3-tts-voices-")
    os.environ.update({key: str(value) for key, value in env.items()})
    name = f"qwen3_tts_under_test_{next(_counter)}"
    sys.path.insert(0, str(SERVICE_DIR))
    try:
        spec = spec_from_file_location(name, SERVICE_DIR / "app.py")
        module = module_from_spec(spec)
        assert spec.loader is not None
        sys.modules[name] = module
        spec.loader.exec_module(module)
        _stand_in_for_the_reference_decoder(module)
        return module
    finally:
        sys.path.remove(str(SERVICE_DIR))
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _stand_in_for_the_reference_decoder(module) -> None:
    """Reference clips in these tests are placeholder bytes, not audio.

    The service measures every reference clip (soundfile, else librosa) before it
    uses it. The default here says "one second long"; a test of that check puts the
    real function back with ``module._measure_reference = module._real_measure_reference``
    (the pre-fix service has no such function, and nothing happens).
    """
    real = getattr(module, "_measure_reference", None)
    if real is not None:
        module._real_measure_reference = real
        module._measure_reference = lambda path, cap: (1.0, True)


@dataclass
class VoiceClonePromptItem:
    """Same fields as `qwen_tts.inference.qwen3_tts_model.VoiceClonePromptItem`."""

    ref_code: object
    ref_spk_embedding: object
    x_vector_only_mode: bool
    icl_mode: bool
    ref_text: Optional[str] = None


def install_qwen_tts(monkeypatch, from_pretrained=None) -> types.ModuleType:
    """Register a fake `qwen_tts` package (removed again by monkeypatch)."""
    root = types.ModuleType("qwen_tts")
    inference = types.ModuleType("qwen_tts.inference")
    module = types.ModuleType("qwen_tts.inference.qwen3_tts_model")
    module.VoiceClonePromptItem = VoiceClonePromptItem

    class Qwen3TTSModel:
        @staticmethod
        def from_pretrained(name, **kwargs):
            return from_pretrained(name, **kwargs)

    root.Qwen3TTSModel = Qwen3TTSModel
    root.inference = inference
    inference.qwen3_tts_model = module
    monkeypatch.setitem(sys.modules, "qwen_tts", root)
    monkeypatch.setitem(sys.modules, "qwen_tts.inference", inference)
    monkeypatch.setitem(sys.modules, "qwen_tts.inference.qwen3_tts_model", module)
    return root


class FakeQwenModel:
    """Follows the `Qwen3TTSModel` contract the service relies on.

    *kind* is the checkpoint's ``tts_model_type``. Every ``generate_*`` method
    exists on every kind and raises ``ValueError`` at call time on the ones that
    cannot do the work, exactly like the real wrapper.
    """

    def __init__(self, kind="base", sample_rate=24000, speakers=None):
        self.kind = kind
        self.sample_rate = sample_rate
        if speakers is None:
            speakers = CUSTOM_VOICE_SPEAKERS if kind == "custom_voice" else []
        self._speakers = speakers
        self.calls: list[tuple] = []
        self.fail_with: Optional[Exception] = None

    # -- the real wrapper's helpers -------------------------------------------
    def _need(self, kind, what):
        if self.kind != kind:
            raise ValueError(
                f"model with tts_model_type: {self.kind} does not support {what}, "
                "Please check Model Card or Readme for more details.")

    def _languages(self, language, n):
        languages = language if isinstance(language, list) else [language] * n
        bad = [x for x in languages if x is None or str(x).lower() not in MODEL_LANGUAGES]
        if bad:
            raise ValueError(f"Unsupported languages: {bad}. Supported: {sorted(MODEL_LANGUAGES)}")
        return languages

    def _audio(self, n):
        return [np.full(240, 0.25, dtype=np.float32) for _ in range(n)], self.sample_rate

    def _maybe_fail(self):
        if self.fail_with is not None:
            raise self.fail_with

    # -- API ------------------------------------------------------------------
    def get_supported_speakers(self):
        return None if self._speakers is None else list(self._speakers)

    def generate_custom_voice(self, text, speaker, language=None, instruct=None, **kw):
        self._need("custom_voice", "generate_custom_voice")
        texts = text if isinstance(text, list) else [text]
        self._languages(language if language is not None else "Auto", len(texts))
        if str(speaker).lower() not in {s.lower() for s in self._speakers}:
            raise ValueError(f"Unsupported speakers: [{speaker!r}]")
        self._maybe_fail()
        self.calls.append(("generate_custom_voice", dict(
            text=text, speaker=speaker, language=language, instruct=instruct)))
        return self._audio(len(texts))

    def generate_voice_design(self, text, instruct, language=None, **kw):
        self._need("voice_design", "generate_voice_design")
        texts = text if isinstance(text, list) else [text]
        self._languages(language if language is not None else "Auto", len(texts))
        self._maybe_fail()
        self.calls.append(("generate_voice_design", dict(
            text=text, instruct=instruct, language=language)))
        return self._audio(len(texts))

    def create_voice_clone_prompt(self, ref_audio, ref_text=None, x_vector_only_mode=False):
        self._need("base", "create_voice_clone_prompt")
        if not x_vector_only_mode and not ref_text:
            raise ValueError("ref_text is required when x_vector_only_mode=False (ICL mode). Bad index=0")
        self._maybe_fail()
        self.calls.append(("create_voice_clone_prompt", dict(
            ref_audio=ref_audio, ref_text=ref_text, x_vector_only_mode=x_vector_only_mode,
            ref_audio_bytes=Path(ref_audio).read_bytes() if isinstance(ref_audio, str) else None)))
        return [VoiceClonePromptItem(
            ref_code=None if x_vector_only_mode else make_tensor([[1, 2], [3, 4]]),
            ref_spk_embedding=make_tensor([0.5, 0.25]),
            x_vector_only_mode=bool(x_vector_only_mode),
            icl_mode=not x_vector_only_mode,
            ref_text=ref_text,
        )]

    def generate_voice_clone(self, text, language=None, ref_audio=None, ref_text=None,
                             x_vector_only_mode=False, voice_clone_prompt=None, **kw):
        self._need("base", "generate_voice_clone")
        texts = text if isinstance(text, list) else [text]
        self._languages(language if language is not None else "Auto", len(texts))
        if voice_clone_prompt is None and ref_audio is None:
            raise ValueError("Either `voice_clone_prompt` or `ref_audio` must be provided.")
        self._maybe_fail()
        self.calls.append(("generate_voice_clone", dict(
            text=text, language=language, ref_audio=ref_audio, ref_text=ref_text,
            voice_clone_prompt=voice_clone_prompt)))
        return self._audio(len(texts))


def install_model(module, model, name):
    """Make *model* the resident model of a loaded app module."""
    module.tts_model = model
    module.model_loaded = True
    module.current_model_name = name
    module._touch_model()
    return model


def client(module):
    """A TestClient without lifespan, so nothing tries to load a real model."""
    from fastapi.testclient import TestClient

    return TestClient(module.app, raise_server_exceptions=False)


def capture_wav(monkeypatch, module):
    """Make the stubbed soundfile emit recognisable bytes; return the list of writes."""
    writes: list[tuple] = []

    def write(buffer, audio, sr, format=None):
        writes.append((len(audio), sr, format))
        buffer.write(b"RIFFfake-wav")

    monkeypatch.setattr(module.sf, "write", write)
    return writes
