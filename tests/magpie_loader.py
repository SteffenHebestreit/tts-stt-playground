"""Shared harness for the magpie-tts-service tests (no tests of its own).

`magpie-tts-service/app.py` reads its configuration at import time (languages, speaker,
limits, idle TTL), so a test that needs another configuration imports a fresh copy of the
module. torch and soundfile are replaced by stand-ins while it loads, and the soundfile
stand-in is a *functional* recording one whose "WAV" is an ``.npy`` blob, so a test can
read back exactly which samples the service produced.

NeMo is imported lazily inside the service's model loader, so ``install_nemo`` registers
a fake ``nemo.collections.tts`` package for the length of a test. ``FakeMagpie`` follows
the parts of the real ``MagpieTTSModel`` (nemo_toolkit 3.0.0, checkpoint
nvidia/magpie_tts_multilingual_357m) that the service depends on, each measured on the
real model:

* ``do_tts(transcript, language, apply_TN, use_cfg, speaker_index)`` returns a (1, T)
  tensor and its (1,) length, both on the model's device: the fake's are CUDA tensors
  whose ``numpy()`` refuses until ``.cpu()`` (``FakeDeviceTensor``). A speaker index
  outside the baked range is a ``ValueError``;
* with ``apply_TN=False`` the tokenizer DROPS digits, so a digit-only text yields no
  audio at all (the fake does the same);
* with ``apply_TN=True`` a language's text normalizer is built on its first use and
  cached on the instance in ``_text_normalizers``; a build that failed is cached as
  None, and the language's digits are dropped from then on, as without normalization
  (``package.normalizer_builds`` records every build, ``install_nemo(failing_normalizers=...)``
  makes a first build fail, ``normalizer_cache=False`` models a NeMo without the cache);
* the decoder stops after ``inference_parameters.max_decoder_steps`` frames of
  ``codec_model_samples_per_frame`` samples and returns what it has, with no error
  (``install_nemo(decoder_cap=...)``; without it the fake has no limit and no such
  attributes);
* the real ``LANGUAGE_TOKENIZER_MAP`` lists tokenizers by name, and the checkpoint holds
  Italian, Vietnamese and Hindi tokenizers under names that map does not list. The fake
  map reproduces that gap, so ``it``, ``vi`` and ``hi`` are not routable even though a
  naive "does the model know the language" check would say yes.

The audio a fake model returns is a constant equal to the speaker that spoke, so a test
reads "who talked" straight off the samples.

``MAGPIE_SERVICE_DIR`` points the loader at another checkout of the service.
"""

from __future__ import annotations

import asyncio
import io
import os
import sys
import threading
import time
import types
from importlib.util import module_from_spec, spec_from_file_location
from itertools import count
from pathlib import Path
from typing import Optional

import numpy as np

from stub_modules import stubbed_modules

REPO = Path(__file__).resolve().parents[1]
SERVICE_DIR = Path(os.environ.get("MAGPIE_SERVICE_DIR") or REPO / "magpie-tts-service")

# Settings the service reads from the environment. Anything not passed to `load_app` is
# removed while importing, so a variable exported in the developer's shell cannot change
# what a test measures.
MANAGED_ENV = (
    "MAGPIE_MODEL", "MAGPIE_DEFAULT_LANGUAGE", "MAGPIE_DEFAULT_SPEAKER", "MAGPIE_SPEAKERS",
    "MAGPIE_APPLY_TN", "MAGPIE_USE_CFG", "MAGPIE_WARM_LANGUAGES", "MAGPIE_MAX_GROUP_CHARS", "MAGPIE_GROUP_GAP_MS",
    "MAX_TEXT_CHARS", "TTS_MODEL_TTL", "MODEL_TTL", "TTS_MAX_CONCURRENCY", "TTS_QUEUE_TIMEOUT_S",
    "TTS_MAX_QUEUE", "ALLOWED_ORIGINS", "ALLOW_CREDENTIALS",
)

SAMPLE_RATE = 22050
# Samples the fake produces per character of input.
SAMPLES_PER_CHAR = 10

# Tokenizer names the fake checkpoint holds: a faithful subset of the real one, including
# the Italian/Vietnamese/Hindi tokenizers that NeMo 3.0.0's language map cannot reach.
CHECKPOINT_TOKENIZERS = (
    "english_phoneme", "text_ce_tokenizer", "spanish_phoneme", "german_phoneme", "mandarin_phoneme",
    "japanese_phoneme", "hindi_phoneme", "french_chartokenizer", "italian_chartokenizer",
    "vietnamese_chartokenizer",
)
# A reduced copy of tts_dataset_utils.LANGUAGE_TOKENIZER_MAP (code -> candidate names).
LANGUAGE_TOKENIZER_MAP = {
    "en": ["english_phoneme", "english"],
    "de": ["german_phoneme", "german"],
    "es": ["spanish_phoneme", "spanish"],
    "fr": ["french_chartokenizer", "french"],
    "it": ["italian_phoneme", "italian"],
    "vi": ["vietnamese_phoneme", "vietnamese"],
    "zh": ["mandarin_phoneme", "mandarin"],
    "hi": ["hindi_chartokenizer", "hindi"],
    "ja": ["japanese_phoneme", "japanese"],
}


def _torch_stub() -> types.ModuleType:
    """A torch stand-in with what the service touches at import and in /status."""
    torch = types.ModuleType("torch")
    torch.__version__ = "0.0.0-stub"
    torch.is_tensor = lambda x: isinstance(x, FakeDeviceTensor)
    torch.cuda = types.SimpleNamespace(
        is_available=lambda: False,
        empty_cache=lambda: None,
        get_device_name=lambda i: "stub",
        memory_allocated=lambda: 0,
        get_device_properties=lambda i: types.SimpleNamespace(total_memory=0),
    )
    return torch


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
    return sf


def _uvicorn_stub() -> types.ModuleType:
    uv = types.ModuleType("uvicorn")
    uv.run = lambda *a, **k: None
    return uv


def _stubbed():
    return stubbed_modules(
        {"torch": _torch_stub, "soundfile": _soundfile_stub, "uvicorn": _uvicorn_stub},
        force=("soundfile", "uvicorn"),
    )


_counter = count()


def load_app(**env):
    """Import a fresh copy of the service with *env* as its whole configuration.

    The idle TTL defaults to "never" so no test leaves a 5-minute timer thread behind;
    tests of the TTL pass their own.
    """
    env = {"TTS_MODEL_TTL": "-1", **env}
    previous = {key: os.environ.get(key) for key in MANAGED_ENV}
    for key in MANAGED_ENV:
        os.environ.pop(key, None)
    os.environ.update({key: str(value) for key, value in env.items()})
    name = f"magpie_under_test_{next(_counter)}"
    # The service's sibling imports must come from ITS directory, not from whichever
    # service imported a module of that name first; put the cached ones back after.
    shared = ("model_lifecycle", "body_limit", "origin_guard")
    cached = {m: sys.modules.pop(m, None) for m in shared}
    sys.path.insert(0, str(SERVICE_DIR))
    try:
        with _stubbed():
            spec = spec_from_file_location(name, SERVICE_DIR / "app.py")
            module = module_from_spec(spec)
            assert spec.loader is not None
            sys.modules[name] = module
            spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(SERVICE_DIR))
        for m, old in cached.items():
            if old is None:
                sys.modules.pop(m, None)
            else:
                sys.modules[m] = old
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def load_support():
    """Import magpie_support (and its text_splitting sibling) the way the service does."""
    cached = sys.modules.pop("text_splitting", None)
    sys.path.insert(0, str(SERVICE_DIR))
    try:
        spec = spec_from_file_location(f"magpie_support_under_test_{next(_counter)}", SERVICE_DIR / "magpie_support.py")
        module = module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(SERVICE_DIR))
        if cached is not None:
            sys.modules["text_splitting"] = cached


# --- The fake library ----------------------------------------------------------------

class FakeDeviceTensor:
    """Just enough of a torch tensor on the GPU: ``numpy()`` refuses until ``.cpu()`` was called.

    Always on "cuda:0", whatever the fake model's device: the stubbed torch has no CUDA,
    so following ``model.device`` would hand the service CPU tensors, and the conversion
    every real request goes through would go untested.
    """

    def __init__(self, array, device: str = "cuda:0"):
        self._array = np.asarray(array)
        self.device = device

    def detach(self) -> "FakeDeviceTensor":
        return self

    def float(self) -> "FakeDeviceTensor":
        return FakeDeviceTensor(self._array.astype(np.float32), self.device)

    def cpu(self) -> "FakeDeviceTensor":
        return FakeDeviceTensor(self._array, "cpu")

    def numpy(self) -> np.ndarray:
        if self.device != "cpu":
            raise TypeError(
                f"can't convert {self.device} device type tensor to numpy. "
                "Use Tensor.cpu() to copy the tensor to host memory first."
            )
        return self._array

    def __array__(self, dtype=None, copy=None):
        array = self.numpy()
        return array if dtype is None else array.astype(dtype)


def _without_digits(text: str) -> str:
    return "".join(c for c in text if not c.isdigit())


class FakeMagpie:
    """Behaves like MagpieTTSModel where the service depends on it."""

    output_sample_rate = SAMPLE_RATE
    has_baked_context_embedding = True

    def __init__(self, package: "FakeNemo", speakers: int, tokenizers):
        self.package = package
        self.num_baked_speakers = speakers
        self.tokenizer = types.SimpleNamespace(tokenizers={name: object() for name in tokenizers})
        self.device = "cpu"
        self.moved_to_cpu = False
        self.lock = threading.Lock()
        self.running = 0
        self.max_running = 0
        # Set `gate` to make do_tts() block after it started (the window in which the
        # real library is used by a worker thread).
        self.gate: Optional[threading.Event] = None
        self.entered = threading.Event()
        if package.normalizer_cache:
            # MagpieTTSModel.__init__ (NeMo 3.0.0): `self._text_normalizers: Dict[str, Any] = {}`.
            self._text_normalizers: dict = {}
        if package.decoder_cap is not None:
            # One frame is one character's worth of fake audio.
            self.inference_parameters = types.SimpleNamespace(max_decoder_steps=package.decoder_cap)
            self.codec_model_samples_per_frame = SAMPLES_PER_CHAR

    def _normalized(self, transcript: str, language: str) -> str:
        """``MagpieTTSModel._get_normalized_text``, minus the normalizing: the fake speaks digits as they are."""
        cache = getattr(self, "_text_normalizers", None)
        if cache is None:
            self.package.normalizer_builds.append(language)
            return transcript
        if language not in cache:
            self.package.normalizer_builds.append(language)
            failing = self.package.failing_normalizers
            cache[language] = None if language in failing else object()
            failing.discard(language)  # only the first build of it fails
        return transcript if cache[language] is not None else _without_digits(transcript)

    def eval(self):
        return self

    def to(self, device):
        self.device = device
        return self

    def cpu(self):
        self.moved_to_cpu = True
        return self

    def do_tts(self, transcript, language="en", apply_TN=False, use_cfg=True, speaker_index=None):
        with self.lock:
            self.running += 1
            self.max_running = max(self.max_running, self.running)
        try:
            self.package.calls.append({
                "text": transcript, "language": language, "apply_TN": apply_TN, "use_cfg": use_cfg,
                "speaker_index": speaker_index,
            })
            self.entered.set()
            error, first = self.package.generate_error, self.package.generate_error_calls
            if error is not None and (first is None or len(self.package.calls) <= first):
                raise error
            if self.gate is not None:
                assert self.gate.wait(10), "a test left a generation blocked"
            index = 0 if speaker_index is None else speaker_index
            if not 0 <= index < self.num_baked_speakers:
                raise ValueError(
                    f"speaker_indices values must be in range [0, {self.num_baked_speakers - 1}], "
                    f"got min={index}, max={index}"
                )
            spoken = self._normalized(transcript, language) if apply_TN else _without_digits(transcript)
            samples = len(spoken.strip()) * SAMPLES_PER_CHAR
            if self.package.decoder_cap is not None:
                samples = min(samples, self.package.decoder_cap * SAMPLES_PER_CHAR)
            audio = np.full((1, samples), 0.1 * (index + 1), dtype=np.float32)
            return FakeDeviceTensor(audio), FakeDeviceTensor(np.array([samples], dtype=np.int64))
        finally:
            with self.lock:
                self.running -= 1


class FakeNemo:
    """What `install_nemo` registered, for assertions."""

    def __init__(self, speakers: int, tokenizers, load_error: Optional[Exception], generate_error: Optional[Exception],
                 generate_error_calls: Optional[int], decoder_cap: Optional[int] = None,
                 failing_normalizers=(), normalizer_cache: bool = True):
        self.speakers = speakers
        self.tokenizers = tokenizers
        self.load_error = load_error
        self.generate_error = generate_error
        # None: every do_tts call raises generate_error; N: only the first N calls do.
        self.generate_error_calls = generate_error_calls
        self.decoder_cap = decoder_cap
        self.failing_normalizers = set(failing_normalizers)
        self.normalizer_cache = normalizer_cache
        self.loads = 0
        self.load_names: list = []
        self.models: list = []
        self.calls: list = []
        # The language of every text-normalizer build, in order, across all model instances.
        self.normalizer_builds: list = []

    @property
    def model(self) -> FakeMagpie:
        return self.models[-1]

    def _new(self, name: str) -> FakeMagpie:
        self.loads += 1
        self.load_names.append(name)
        if self.load_error is not None:
            raise self.load_error
        model = FakeMagpie(self, self.speakers, self.tokenizers)
        self.models.append(model)
        return model


def install_nemo(monkeypatch, *, speakers: int = 5, tokenizers=CHECKPOINT_TOKENIZERS,
                 load_error: Optional[Exception] = None, generate_error: Optional[Exception] = None,
                 generate_error_calls: Optional[int] = None, decoder_cap: Optional[int] = None,
                 failing_normalizers=(), normalizer_cache: bool = True) -> FakeNemo:
    """Register a fake ``nemo.collections.tts`` (removed again by monkeypatch).

    ``decoder_cap``: the decoder's frame limit (NeMo's max_decoder_steps), counted in
    characters' worth of fake audio; longer speech comes back cut to it.
    ``failing_normalizers``: languages whose first text-normalizer build fails.
    ``normalizer_cache=False``: a model without NeMo 3.0's ``_text_normalizers``.
    """
    package = FakeNemo(speakers, tokenizers, load_error, generate_error, generate_error_calls,
                       decoder_cap, failing_normalizers, normalizer_cache)

    class MagpieTTSModel:
        @classmethod
        def from_pretrained(cls, model_name, **kwargs):
            return package._new(model_name)

        @classmethod
        def restore_from(cls, restore_path, **kwargs):
            return package._new(restore_path)

    def module(name: str, **attrs) -> types.ModuleType:
        mod = types.ModuleType(name)
        mod.__dict__.update(attrs)
        mod.__path__ = []  # lets submodules resolve
        monkeypatch.setitem(sys.modules, name, mod)
        return mod

    root = module("nemo")
    collections = module("nemo.collections")
    tts = module("nemo.collections.tts")
    models = module("nemo.collections.tts.models", MagpieTTSModel=MagpieTTSModel)
    parts = module("nemo.collections.tts.parts")
    utils = module("nemo.collections.tts.parts.utils")
    dataset_utils = module(
        "nemo.collections.tts.parts.utils.tts_dataset_utils", LANGUAGE_TOKENIZER_MAP=LANGUAGE_TOKENIZER_MAP
    )
    root.collections, collections.tts = collections, tts
    tts.models, tts.parts, parts.utils, utils.tts_dataset_utils = models, parts, utils, dataset_utils
    return package


# --- Small helpers -------------------------------------------------------------------

def decode_audio(content: bytes) -> np.ndarray:
    """The samples inside a response made with the recording soundfile stand-in."""
    return np.load(io.BytesIO(content))


def speakers_in(audio: np.ndarray) -> set:
    """Speaker indices heard in *audio* (the fake speaks a constant 0.1 * (index + 1))."""
    return {int(round(float(v) * 10)) - 1 for v in np.unique(np.round(audio, 4)) if v != 0}


def wait_until(predicate, timeout: float = 5.0, interval: float = 0.005) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return bool(predicate())


async def wait_until_async(predicate, timeout: float = 5.0, interval: float = 0.005) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        await asyncio.sleep(interval)
    return bool(predicate())
