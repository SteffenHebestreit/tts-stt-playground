"""Load the transformers ASR services' modules, and their apps under stand-ins (no tests of its own).

``load_module(service, name)`` imports one light module (``transformers_asr``,
``cohere_support``, ``granite_support``) from a service directory, with that
directory's own siblings, without torch or transformers.

``load_app(service, **env)`` imports a fresh copy of ``cohere-asr-service/app.py`` or
``granite-asr-service/app.py`` with *env* as its whole configuration and fakes for
everything heavy:

* ``torch``: CPU only, ``from_numpy``/``zeros``/``inference_mode`` and dtypes.
* ``transformers``: processors and models that "hear" words the way
  test_qwen3_asr_harness's fake Qwen does (a word is a run of constant amplitude, its
  text ``w<level>``), generate token ids for them, and stop at ``max_new_tokens``.
  Cohere's prompt is ten tokens; Granite's speech LLM echoes its prompt and records
  it; Granite's CTC model returns one row per piece.
* ``huggingface_hub.snapshot_download``: a temporary model directory with a
  config.json (the ``model_type`` of the service's model) and a small
  model.safetensors; the arguments of every call are recorded.
* ``silero_vad``: speech wherever the audio is not silent, or a failure on demand.
* ``soundfile``/``librosa``/``uvicorn`` and the fake ffmpeg of the qwen3-asr harness.

The fakes hang off the module: ``module._fake`` (a ``Fakes``). The app imports
transformers, huggingface_hub and silero_vad only when it loads a model, i.e. during
a test, so ``load_app`` puts those three into ``sys.modules`` through the test's
``monkeypatch``, which takes them out again when the test ends.
"""

from __future__ import annotations

import os
import sys
import tempfile
import threading
import types
from importlib.util import module_from_spec, spec_from_file_location
from itertools import count
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from test_qwen3_asr_harness import FakeLibrosa, fake_soundfile, hear_words

REPO = Path(__file__).resolve().parents[1]
SERVICE_DIRS = {
    "cohere": Path(os.environ.get("COHERE_ASR_SERVICE_DIR") or REPO / "cohere-asr-service"),
    "granite": Path(os.environ.get("GRANITE_ASR_SERVICE_DIR") or REPO / "granite-asr-service"),
}
LOCAL_MODULES = ("audio_pipeline", "model_lifecycle", "body_limit", "origin_guard", "transformers_asr",
                 "cohere_support", "granite_support")
MANAGED_ENV = (
    "COHERE_ASR_MODEL", "COHERE_ASR_REVISION", "COHERE_ASR_SHA256", "COHERE_ASR_DEFAULT_LANGUAGE",
    "COHERE_ASR_CHUNK_S", "COHERE_ASR_BATCH_SIZE", "COHERE_ASR_MAX_NEW_TOKENS", "COHERE_ASR_VAD",
    "COHERE_ASR_VAD_THRESHOLD",
    "GRANITE_ASR_MODEL", "GRANITE_ASR_REVISION", "GRANITE_ASR_SHA256", "GRANITE_ASR_DEFAULT_LANGUAGE",
    "GRANITE_SUPPORTED_LANGUAGES", "GRANITE_ASR_CHUNK_S", "GRANITE_ASR_BATCH_SIZE",
    "GRANITE_ASR_MAX_NEW_TOKENS", "GRANITE_ASR_VAD", "GRANITE_ASR_VAD_THRESHOLD",
    "HF_TOKEN", "ASR_MODEL_TTL", "MODEL_TTL", "ASR_MAX_CONCURRENCY", "ASR_MAX_QUEUE", "ASR_QUEUE_TIMEOUT_S",
    "MAX_UPLOAD_MB", "MAX_AUDIO_SECONDS", "ALLOWED_ORIGINS", "ALLOW_CREDENTIALS",
    "ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES",
)
RATE = 16000
WAIT = 5.0

_counter = count()


# --- light modules ----------------------------------------------------------------------


def load_module(service: str, name: str):
    """``<service dir>/<name>.py`` with that directory's own siblings, under a fresh name."""
    directory = SERVICE_DIRS[service]
    saved = {mod: sys.modules.get(mod) for mod in LOCAL_MODULES}
    sys.path.insert(0, str(directory))
    try:
        for mod in LOCAL_MODULES:
            sys.modules.pop(mod, None)
        unique = f"{name}_{service}_under_test_{next(_counter)}"
        spec = spec_from_file_location(unique, directory / f"{name}.py")
        module = module_from_spec(spec)
        sys.modules[unique] = module  # dataclasses resolves annotations through sys.modules
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(directory))
        for mod, previous in saved.items():
            if previous is None:
                sys.modules.pop(mod, None)
            else:
                sys.modules[mod] = previous


# --- fake torch --------------------------------------------------------------------------


class FakeTensor:
    """Just enough of a 2-D integer tensor: shape, slicing by rows/columns, tolist()."""

    def __init__(self, rows):
        self.rows = [list(row) for row in rows]

    @property
    def shape(self):
        return (len(self.rows), max((len(r) for r in self.rows), default=0))

    def tolist(self):
        return [list(row) for row in self.rows]

    def __getitem__(self, key):
        if isinstance(key, tuple):
            rows, cols = key
            return FakeTensor([row[cols] for row in self.rows[rows]])
        return self.rows[key]

    def __iter__(self):
        return iter(self.rows)

    def __len__(self):
        return len(self.rows)

    def to(self, *args, **kwargs):
        return self


class FakeBatch(dict):
    """A processor's BatchFeature: a dict whose .to() records its arguments."""

    def to(self, *args, **kwargs):
        self.moved = (args, kwargs)
        return self


def fake_torch() -> types.ModuleType:
    torch = types.ModuleType("torch")
    torch.float32, torch.float16, torch.bfloat16 = "torch.float32", "torch.float16", "torch.bfloat16"
    torch.cuda = SimpleNamespace(
        is_available=lambda: False, is_bf16_supported=lambda: True, empty_cache=lambda: None,
        ipc_collect=lambda: None, get_device_name=lambda index=0: "Fake GPU", memory_allocated=lambda: 0,
        memory_reserved=lambda: 0, get_device_properties=lambda index=0: SimpleNamespace(total_memory=8 << 30),
    )
    torch.from_numpy = lambda array: np.asarray(array)
    torch.zeros = lambda n: SimpleNamespace(numpy=lambda: np.zeros(n, dtype=np.float32))

    class _NoGrad:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    torch.inference_mode = lambda: _NoGrad()
    return torch


# --- fake models ---------------------------------------------------------------------------

EOS, PAD = 3, 2
_WORD_BASE = 1000  # token id of word level n is _WORD_BASE + n


def _word_tokens(samples) -> list[int]:
    return [_WORD_BASE + int(word[1:]) for word in hear_words(np.asarray(samples, dtype=np.float32))]


class FakeTokenizer:
    pad_token_id = PAD
    eos_token_id = EOS

    def batch_decode(self, sequences, skip_special_tokens=True, **kwargs):
        rows = sequences.tolist() if hasattr(sequences, "tolist") else sequences
        return [" ".join(f"w{t - _WORD_BASE}" for t in row if t >= _WORD_BASE) for row in rows]

    def apply_chat_template(self, chat, tokenize=False, add_generation_prompt=True):
        return "USER: " + chat[0]["content"] + "\n ASSISTANT:"


class Fakes:
    """The stand-ins one loaded app uses, and what they saw."""

    def __init__(self, service: str, model_dir: Path, model_type: str):
        self.service = service
        self.model_dir = model_dir
        self.model_type = model_type
        self.downloads: list[dict] = []
        self.loads: list[dict] = []
        self.generate_calls: list[dict] = []
        self.processor_calls: list[dict] = []
        self.vad_calls = 0
        self.vad_error: Exception | None = None
        self.load_error: Exception | None = None
        self.generate_gate: threading.Event | None = None
        self.generate_entered = threading.Event()

    # huggingface_hub
    def snapshot_download(self, repo_id, revision=None, allow_patterns=None, token=None, **kwargs):
        self.downloads.append({"repo_id": repo_id, "revision": revision, "allow_patterns": allow_patterns,
                               "token": token})
        return str(self.model_dir)

    # silero_vad
    def get_speech_timestamps(self, audio, model, **kwargs):
        self.vad_calls += 1
        if self.vad_error is not None:
            raise self.vad_error
        samples = np.asarray(audio)
        loud = np.flatnonzero(np.abs(samples) > 1e-4)
        if loud.size == 0:
            return []
        # One stretch per run of sound separated by >= 0.3 s of silence.
        gaps = np.flatnonzero(np.diff(loud) > int(0.3 * RATE))
        starts = np.concatenate(([loud[0]], loud[gaps + 1]))
        ends = np.concatenate((loud[gaps] + 1, [loud[-1] + 1]))
        return [{"start": int(a), "end": int(b)} for a, b in zip(starts, ends)]

    def _enter_generate(self, **record):
        self.generate_calls.append(record)
        self.generate_entered.set()
        if self.generate_gate is not None:
            self.generate_gate.wait(WAIT)


class _FakeModelBase:
    device = "cpu"
    dtype = "torch.float32"

    def __init__(self, fakes: Fakes):
        self.fakes = fakes

    def to(self, *args, **kwargs):
        return self

    def eval(self):
        return self


def _class_factory(fakes: Fakes, cls, kind: str):
    """A class whose from_pretrained records the call and builds *cls*."""

    class Loader:
        @staticmethod
        def from_pretrained(path, **kwargs):
            fakes.loads.append({"kind": kind, "path": str(path), **kwargs})
            if fakes.load_error is not None and kind == "model":
                raise fakes.load_error
            return cls(fakes)

    Loader.__name__ = cls.__name__
    return Loader


class FakeCohereProcessor:
    PROMPT = [7] * 10

    def __init__(self, fakes):
        self.fakes = fakes
        self.tokenizer = FakeTokenizer()

    def __call__(self, audio, language, sampling_rate=None, return_tensors=None, punctuation=True, **kwargs):
        self.fakes.processor_calls.append({"language": language, "lengths": [len(a) for a in audio],
                                           "sampling_rate": sampling_rate, "punctuation": punctuation})
        return FakeBatch({"input_features": list(audio), "attention_mask": None,
                          "decoder_input_ids": FakeTensor([self.PROMPT] * len(audio)), "audio_chunk_index": None})


class FakeCohereModel(_FakeModelBase):
    def generate(self, input_features, decoder_input_ids, max_new_tokens, **kwargs):
        self.fakes._enter_generate(pieces=len(input_features), max_new_tokens=max_new_tokens)
        rows = []
        for audio, prompt in zip(input_features, decoder_input_ids.tolist()):
            words = _word_tokens(audio)
            new = words[:max_new_tokens] + ([EOS] if len(words) < max_new_tokens else [])
            rows.append(prompt + new)
        width = max(len(r) for r in rows)
        return FakeTensor([r + [PAD] * (width - len(r)) for r in rows])


class FakeGraniteProcessor:
    """The speech LLM's processor: text prompts and audio; the prompt becomes input_ids."""

    def __init__(self, fakes):
        self.fakes = fakes
        self.tokenizer = FakeTokenizer()

    def __call__(self, text, audio, device="cpu", return_tensors=None, **kwargs):
        self.fakes.processor_calls.append({"prompts": list(text), "lengths": [len(a) for a in audio]})
        prompt_ids = [[5] * 12 for _ in text]
        return FakeBatch({"input_ids": FakeTensor(prompt_ids), "input_features": FakeTensor([[0]] * len(text)),
                          "input_features_mask": None, "_audio": list(audio)})


class FakeGraniteModel(_FakeModelBase):
    def generate(self, input_ids, max_new_tokens, _audio=None, **kwargs):
        self.fakes._enter_generate(pieces=len(_audio), max_new_tokens=max_new_tokens)
        rows = []
        for audio, prompt in zip(_audio, input_ids.tolist()):
            words = _word_tokens(audio)
            new = words[:max_new_tokens] + ([EOS] if len(words) < max_new_tokens else [])
            rows.append(prompt + new)
        width = max(len(r) for r in rows)
        return FakeTensor([r + [PAD] * (width - len(r)) for r in rows])


class FakeCtcProcessor:
    def __init__(self, fakes):
        self.fakes = fakes
        self.tokenizer = FakeTokenizer()

    def __call__(self, audio, sampling_rate=None, return_tensors=None, **kwargs):
        self.fakes.processor_calls.append({"lengths": [len(a) for a in audio], "sampling_rate": sampling_rate})
        return FakeBatch({"input_features": list(audio), "attention_mask": None})

    def batch_decode(self, sequences, skip_special_tokens=True):
        return self.tokenizer.batch_decode(sequences, skip_special_tokens=skip_special_tokens)


class FakeCtcModel(_FakeModelBase):
    def generate(self, input_features, attention_mask=None, **kwargs):
        self.fakes._enter_generate(pieces=len(input_features), max_new_tokens=None)
        rows = [_word_tokens(audio) for audio in input_features]
        width = max((len(r) for r in rows), default=0)
        return FakeTensor([r + [0] * (width - len(r)) for r in rows])


def _fake_transformers(fakes: Fakes) -> types.ModuleType:
    module = types.ModuleType("transformers")
    module.CohereAsrProcessor = _class_factory(fakes, FakeCohereProcessor, "processor")
    module.CohereAsrForConditionalGeneration = _class_factory(fakes, FakeCohereModel, "model")
    module.GraniteSpeechProcessor = _class_factory(fakes, FakeGraniteProcessor, "processor")
    module.GraniteSpeechForConditionalGeneration = _class_factory(fakes, FakeGraniteModel, "model")
    module.GraniteSpeech5Processor = _class_factory(fakes, FakeCtcProcessor, "processor")
    module.GraniteSpeech5ForCTC = _class_factory(fakes, FakeCtcModel, "model")
    return module


def _fake_hub(fakes: Fakes) -> types.ModuleType:
    module = types.ModuleType("huggingface_hub")
    module.snapshot_download = fakes.snapshot_download
    return module


def _fake_silero(fakes: Fakes) -> types.ModuleType:
    module = types.ModuleType("silero_vad")
    module.load_silero_vad = lambda: "fake-silero-model"
    module.get_speech_timestamps = fakes.get_speech_timestamps
    return module


def _fake_uvicorn() -> types.ModuleType:
    module = types.ModuleType("uvicorn")
    module.run = lambda *args, **kwargs: None
    return module


WEIGHTS = b"fake safetensors bytes"


# Captured at import: the hardening tests point tempfile at a spool directory and count
# what is left in it, and the fake checkpoint is not something the service left behind.
_MODEL_ROOT = Path(tempfile.gettempdir())


def _model_dir(model_type: str) -> Path:
    directory = Path(tempfile.mkdtemp(prefix="fake-model-", dir=_MODEL_ROOT))
    (directory / "config.json").write_text('{"model_type": "%s"}' % model_type, encoding="utf-8")
    (directory / "model.safetensors").write_bytes(WEIGHTS)
    return directory


def load_app(service: str, monkeypatch, *, model_type: str | None = None, **env):
    """A fresh copy of the service's app with *env* as its whole configuration (see the module doc)."""
    directory = SERVICE_DIRS[service]
    if model_type is None:
        model_type = {"cohere": "cohere_asr", "granite": "granite_speech"}[service]
        if service == "granite" and "turboctc" in str(env.get("GRANITE_ASR_MODEL", "")):
            model_type = "granite_speech5_ctc"
    fakes = Fakes(service, _model_dir(model_type), model_type)
    monkeypatch.setitem(sys.modules, "transformers", _fake_transformers(fakes))
    monkeypatch.setitem(sys.modules, "huggingface_hub", _fake_hub(fakes))
    monkeypatch.setitem(sys.modules, "silero_vad", _fake_silero(fakes))
    stubs = {"torch": fake_torch(), "soundfile": fake_soundfile(), "librosa": FakeLibrosa(),
             "uvicorn": _fake_uvicorn()}
    saved_env = {key: os.environ.get(key) for key in MANAGED_ENV}
    saved_modules = {name: sys.modules.get(name) for name in (*stubs, *LOCAL_MODULES)}
    name = f"{service}_asr_under_test_{next(_counter)}"
    sys.path.insert(0, str(directory))
    try:
        for key in MANAGED_ENV:
            os.environ.pop(key, None)
        os.environ.update({key: str(value) for key, value in env.items()})
        for local in LOCAL_MODULES:
            sys.modules.pop(local, None)
        sys.modules.update(stubs)
        spec = spec_from_file_location(name, directory / "app.py")
        module = module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        module._transformers_asr = sys.modules.get("transformers_asr")
    finally:
        sys.path.remove(str(directory))
        for key, value in saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        for mod_name, previous in saved_modules.items():
            if previous is None:
                sys.modules.pop(mod_name, None)
            else:
                sys.modules[mod_name] = previous
    module._fake = fakes
    # The Silero wrapper imports torch lazily at first use; give it the fake for good.
    gate = module._speech_gate
    if getattr(gate, "_silero", None) is not None and hasattr(gate._silero, "_torch"):
        gate._silero._torch = stubs["torch"]
        gate._silero._model = "fake-silero-model"
        gate._silero._get_speech_timestamps = fakes.get_speech_timestamps
    return module
