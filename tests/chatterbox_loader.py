"""Shared harness for the chatterbox-tts-service tests (no tests of its own).

`chatterbox-tts-service/app.py` reads its configuration at import time (limits,
checkpoint, chunk gap, idle TTL), so a test that needs a different configuration
imports a fresh copy of the module. The heavy stack is replaced by stand-ins while
it loads: torch, and a *functional* soundfile whose "WAV" is an ``.npy`` blob, so a
test can read back exactly which samples the service produced.

`FakeChatterbox` follows the parts of the real ``ChatterboxMultilingualTTS`` (read
from the chatterbox-tts 0.1.7 wheel and GitHub master) that the service depends on:

* the voice lives in ``model.conds``; ``generate(audio_prompt_path=...)`` REPLACES
  it (prepare_conditionals) and a differing ``exaggeration`` REWRITES ``conds.t3``
  on the shared object;
* one ``generate()`` call silently drops everything past a fixed length (the real
  one stops at 1000 speech tokens, ~40 s), returning HTTP-200-looking audio;
* ``generate`` asserts a voice exists when none was given.

The audio a fake model returns is a constant equal to the voice that spoke, so a
test reads "who talked" straight off the samples.

``CHATTERBOX_SERVICE_DIR`` points the loader at another checkout of the service,
which is how the new tests are run against the pre-fix code to prove they fail
there.
"""

from __future__ import annotations

import os
import sys
import threading
import types
from contextlib import contextmanager
from dataclasses import dataclass, field
from importlib.util import module_from_spec, spec_from_file_location
from itertools import count
from pathlib import Path
from typing import Optional

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SERVICE_DIR = Path(os.environ.get("CHATTERBOX_SERVICE_DIR") or REPO / "chatterbox-tts-service")

# Settings the service reads from the environment. Anything not passed to
# `load_app` is removed while importing, so a variable exported in the developer's
# shell cannot change what a test measures.
MANAGED_ENV = (
    "CHATTERBOX_DEFAULT_LANGUAGE", "CHATTERBOX_T3_MODEL", "CHATTERBOX_HF_REVISION",
    "CHATTERBOX_REPETITION_PENALTY", "CHATTERBOX_CHUNK_GAP_MS", "CHATTERBOX_REF_MAX_SECONDS",
    "CHATTERBOX_REF", "MAX_TEXT_CHARS", "MAX_UPLOAD_MB", "TTS_MODEL_TTL", "MODEL_TTL",
    "TTS_MAX_CONCURRENCY", "TTS_QUEUE_TIMEOUT_S", "TTS_MAX_QUEUE", "ALLOWED_ORIGINS", "ALLOW_CREDENTIALS",
)

SAMPLE_RATE = 24000
DEFAULT_VOICE = 0.25
CLONE_VOICE = 0.7
# Samples the fake produces per character of input.
SAMPLES_PER_CHAR = 10
# The fake's stand-in for the library's 1000-token cap.
MODEL_MAX_CHARS = 200


def _is_stub(module) -> bool:
    return getattr(module, "__file__", None) is None


def _install_torch_stub() -> None:
    """A torch stand-in only where the real package is absent; fills in what we need."""
    if "torch" not in sys.modules:
        sys.modules["torch"] = types.ModuleType("torch")
    torch = sys.modules["torch"]
    if not _is_stub(torch):
        return
    if not hasattr(torch, "__version__"):
        torch.__version__ = "0.0.0-stub"
    if not hasattr(torch, "is_tensor"):
        torch.is_tensor = lambda x: False
    if not hasattr(torch, "cuda"):
        torch.cuda = types.SimpleNamespace()
    for name, value in {
        "is_available": lambda: False,
        "empty_cache": lambda: None,
        "ipc_collect": lambda: None,
        "get_device_name": lambda i: "stub",
        "memory_allocated": lambda: 0,
        "get_device_properties": lambda i: types.SimpleNamespace(total_memory=0),
    }.items():
        if not hasattr(torch.cuda, name):
            setattr(torch.cuda, name, value)


def _soundfile_stub() -> types.ModuleType:
    """`soundfile.write` that stores the array as .npy, so tests can read it back."""
    sf = types.ModuleType("soundfile")

    def write(file, data, samplerate, **kwargs):
        array = np.asarray(data)
        if hasattr(file, "write"):
            np.save(file, array)
        else:
            with open(file, "wb") as fh:
                np.save(fh, array)

    sf.write = write
    sf.info = lambda p: types.SimpleNamespace(format="WAV", samplerate=SAMPLE_RATE, channels=1, frames=0)
    return sf


def _uvicorn_stub() -> types.ModuleType:
    uv = types.ModuleType("uvicorn")
    uv.run = lambda *a, **k: None
    return uv


@contextmanager
def _stubbed_modules():
    """soundfile / uvicorn stand-ins for the duration of the import only."""
    names = ("soundfile", "uvicorn")
    saved = {name: sys.modules.get(name) for name in names}
    sys.modules["soundfile"] = _soundfile_stub()
    sys.modules["uvicorn"] = _uvicorn_stub()
    try:
        yield
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


_counter = count()


def load_app(**env):
    """Import a fresh copy of the service with *env* as its whole configuration.

    The idle TTL defaults to "never" so no test leaves a 5-minute timer thread
    behind; tests of the TTL pass their own.
    """
    _install_torch_stub()
    env = {"TTS_MODEL_TTL": "-1", **env}
    previous = {key: os.environ.get(key) for key in MANAGED_ENV}
    for key in MANAGED_ENV:
        os.environ.pop(key, None)
    os.environ.update({key: str(value) for key, value in env.items()})
    name = f"chatterbox_under_test_{next(_counter)}"
    # The service's sibling import must come from ITS directory, not from whichever
    # service imported a module of that name first; put the cached one back after.
    cached_lifecycle = sys.modules.pop("model_lifecycle", None)
    sys.path.insert(0, str(SERVICE_DIR))
    try:
        with _stubbed_modules():
            spec = spec_from_file_location(name, SERVICE_DIR / "app.py")
            module = module_from_spec(spec)
            assert spec.loader is not None
            sys.modules[name] = module
            spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(SERVICE_DIR))
        if cached_lifecycle is None:
            sys.modules.pop("model_lifecycle", None)
        else:
            sys.modules["model_lifecycle"] = cached_lifecycle
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


# --- The fake library --------------------------------------------------------

@dataclass
class T3Cond:
    speaker_emb: float
    emotion_adv: float = 0.5


@dataclass
class Conditionals:
    t3: T3Cond
    gen: dict = field(default_factory=dict)


SUPPORTED_LANGUAGES = {"de": "German", "en": "English", "fr": "French", "es": "Spanish"}


class FakeChatterbox:
    """Behaves like ChatterboxMultilingualTTS where the service depends on it."""

    sr = SAMPLE_RATE

    def __init__(self, conds: Optional[Conditionals]):
        self.conds = conds
        self.calls: list = []
        self.lock = threading.Lock()
        # Set `gate` to make generate() block after it has read the voice (the
        # window in which the real library is used by a worker thread).
        self.gate: Optional[threading.Event] = None
        self.entered = threading.Event()
        self.running = 0
        self.max_running = 0
        # Length (samples) of every reference clip prepare_conditionals was given.
        self.prepared_samples: list = []

    def prepare_conditionals(self, wav_fpath, exaggeration=0.5):
        with open(wav_fpath, "rb") as fh:
            clip = np.load(fh)
        self.prepared_samples.append(int(clip.size))
        self.conds = Conditionals(T3Cond(float(clip.mean()), float(exaggeration)), {"ref": str(wav_fpath)})

    def generate(self, text, language_id, audio_prompt_path=None, exaggeration=0.5, cfg_weight=0.5,
                 temperature=0.8, repetition_penalty=2.0, min_p=0.05, top_p=1.0):
        with self.lock:
            self.running += 1
            self.max_running = max(self.max_running, self.running)
        try:
            if language_id and language_id.lower() not in SUPPORTED_LANGUAGES:
                raise ValueError(f"Unsupported language_id '{language_id}'.")
            if audio_prompt_path:
                self.prepare_conditionals(audio_prompt_path, exaggeration=exaggeration)
            else:
                assert self.conds is not None, "Please `prepare_conditionals` first or specify `audio_prompt_path`"
            if float(exaggeration) != self.conds.t3.emotion_adv:
                # Rewritten on the object the model owns, exactly like the library.
                self.conds.t3 = T3Cond(self.conds.t3.speaker_emb, float(exaggeration))
            self.calls.append({
                "text": text, "language_id": language_id, "audio_prompt_path": audio_prompt_path,
                "exaggeration": exaggeration, "cfg_weight": cfg_weight,
                "repetition_penalty": repetition_penalty,
            })
            if text.startswith("BOOM"):
                # After the voice was already replaced: the failure a clone hits.
                raise RuntimeError("boom")
            self.entered.set()
            if self.gate is not None:
                assert self.gate.wait(10), "test never released the generation gate"
            voice = self.conds.t3.speaker_emb
            emotion = self.conds.t3.emotion_adv
            self.calls[-1]["voice"] = voice
            self.calls[-1]["emotion"] = emotion
            samples = min(len(text), MODEL_MAX_CHARS) * SAMPLES_PER_CHAR
            return np.full((1, samples), voice, dtype=np.float32)
        finally:
            with self.lock:
                self.running -= 1


def default_conds() -> Conditionals:
    return Conditionals(T3Cond(DEFAULT_VOICE, 0.5), {"prompt": [1, 2, 3]})


class FakePackage:
    """What `install_chatterbox` registered, for assertions."""

    def __init__(self):
        self.model: Optional[FakeChatterbox] = None
        self.from_pretrained_calls: list = []
        self.snapshot_calls: list = []
        self.loads = 0


def install_chatterbox(monkeypatch, *, accepts_t3_model=True, conds="default",
                       repetition_default=None) -> FakePackage:
    """Register a fake `chatterbox.mtl_tts` (removed again by monkeypatch).

    *accepts_t3_model* False reproduces PyPI chatterbox-tts 0.1.7, whose
    ``from_pretrained(cls, device)`` has no ``t3_model``. *conds* is "default"
    (the checkpoint ships a built-in voice), None (it does not) or a Conditionals.
    """
    package = FakePackage()
    root = types.ModuleType("chatterbox")
    module = types.ModuleType("chatterbox.mtl_tts")
    module.SUPPORTED_LANGUAGES = SUPPORTED_LANGUAGES

    def snapshot_download(**kwargs):
        package.snapshot_calls.append(kwargs)
        return "/fake/snapshot"

    module.snapshot_download = snapshot_download

    def build(cls, device, t3_model):
        package.loads += 1
        package.from_pretrained_calls.append({"device": device, "t3_model": t3_model})
        # Like the real one: looks the module-level name up when it runs.
        module.snapshot_download(repo_id="ResembleAI/chatterbox", revision="main")
        voice = default_conds() if conds == "default" else conds
        model = FakeChatterbox(voice)
        package.model = model
        return model

    if accepts_t3_model:
        class ChatterboxMultilingualTTS:
            @classmethod
            def from_pretrained(cls, device, t3_model=None):
                return build(cls, device, t3_model)
    else:
        class ChatterboxMultilingualTTS:
            @classmethod
            def from_pretrained(cls, device):
                return build(cls, device, None)

    ChatterboxMultilingualTTS.__module__ = "chatterbox.mtl_tts"
    module.ChatterboxMultilingualTTS = ChatterboxMultilingualTTS
    root.mtl_tts = module
    monkeypatch.setitem(sys.modules, "chatterbox", root)
    monkeypatch.setitem(sys.modules, "chatterbox.mtl_tts", module)
    return package


def decode_audio(content: bytes) -> np.ndarray:
    """The samples behind a stubbed-soundfile "WAV" response body."""
    import io
    return np.load(io.BytesIO(content))


def decode_pcm16(content: bytes) -> np.ndarray:
    """Samples of a /tts-stream body: 44-byte header, then PCM16 mono."""
    return np.frombuffer(content[44:], dtype="<i2").astype(np.float32) / 32767.0


def npy_bytes(array) -> bytes:
    """A stand-in "audio file": the fake model and the test decoder read it back with np.load."""
    import io
    buffer = io.BytesIO()
    np.save(buffer, np.asarray(array, dtype=np.float32))
    return buffer.getvalue()


def reference_clip(seconds: float = 2.0, voice: float = CLONE_VOICE) -> bytes:
    return npy_bytes(np.full(int(seconds * SAMPLE_RATE), voice, dtype=np.float32))


def make_upload(content: Optional[bytes] = None, filename: str = "ref.wav"):
    """A Starlette UploadFile as the handlers receive it."""
    import io
    from starlette.datastructures import UploadFile

    data = reference_clip() if content is None else content
    return UploadFile(file=io.BytesIO(data), size=len(data), filename=filename)


def voices_in(audio: np.ndarray) -> set:
    """The distinct voices (non-zero sample values) present in a response's audio."""
    return {round(float(v), 3) for v in np.unique(np.asarray(audio)) if abs(float(v)) > 1e-6}


def wait_until(predicate, timeout: float = 5.0, interval: float = 0.005) -> bool:
    """Poll *predicate* from a test thread; returns whether it became true in time."""
    import time

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return bool(predicate())


async def wait_until_async(predicate, timeout: float = 5.0, interval: float = 0.005) -> bool:
    """`wait_until` for coroutines: yields to the event loop between polls."""
    import asyncio
    import time

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        await asyncio.sleep(interval)
    return bool(predicate())


def patch_decoder(monkeypatch, app, record=None):
    """Replace the librosa-based reference decoder with one that reads the .npy stand-in.

    It honours ``max_seconds`` the way ``librosa.load(duration=...)`` does. On a copy
    of the service that has no such function (the pre-fix code) this is a no-op.
    """
    def decode(path, max_seconds):
        if record is not None:
            record.append(max_seconds)
        clip = np.load(path)
        return clip[: int(max_seconds * SAMPLE_RATE)]

    monkeypatch.setattr(app, "_decode_reference", decode, raising=False)
