"""Shared harness for the parakeet-asr-service / canary-asr-service tests (no tests of its own).

Both apps import torch, librosa, soundfile and (on model load) NeMo at module
scope, and read their limits from the environment at import. So a test that needs
a different configuration imports a fresh copy of the app under a chosen
environment, with stand-ins for the heavy packages:

* ``torch``: only the surface the services touch, recording what they call on it
  (matmul precision, empty_cache). ``cuda`` and the compute capability are chosen
  per load.
* ``soundfile``: ``info`` reads real WAV headers with the stdlib ``wave`` module
  and fails on anything else, the way libsndfile fails on an mp3 it cannot open.
* ``librosa``: ``get_duration`` answers for the fake mp3 container below.
* a fake ``ffmpeg`` executable (a Python script) that records its argument list,
  honours ``-t`` and can fail, hang, or wait for the test to let it finish.
* a fake NeMo ASR model whose ``transcribe`` signature mirrors the real ones
  (read from the nemo_toolkit 3.0.0 wheel: ``RNNTModel.transcribe`` and
  ``EncDecMultiTaskModel.transcribe``), recording every call.

``PARAKEET_ASR_SERVICE_DIR`` / ``CANARY_ASR_SERVICE_DIR`` point the loader at
another checkout of a service, which is how the tests are run against the
pre-fix code to prove they fail there.
"""

from __future__ import annotations

import asyncio
import io
import json
import os
import stat
import struct
import sys
import tempfile
import threading
import types
import wave
from importlib.util import module_from_spec, spec_from_file_location
from itertools import count
from pathlib import Path
from types import SimpleNamespace

import httpx

REPO = Path(__file__).resolve().parents[1]
SERVICE_DIRS = {
    "parakeet": Path(os.environ.get("PARAKEET_ASR_SERVICE_DIR") or REPO / "parakeet-asr-service"),
    "canary": Path(os.environ.get("CANARY_ASR_SERVICE_DIR") or REPO / "canary-asr-service"),
}

# Settings the services read from the environment. Anything not passed to
# `load_app` is removed while importing, so a variable exported in the developer's
# shell cannot change what a test measures.
MANAGED_ENV = (
    "PARAKEET_ASR_MODEL", "CANARY_ASR_MODEL", "CANARY_DEFAULT_LANGUAGE", "CANARY_SUPPORTED_LANGUAGES",
    "ASR_MODEL_TTL", "MODEL_TTL", "ASR_MAX_CONCURRENCY", "ASR_MAX_QUEUE", "ASR_QUEUE_TIMEOUT_S",
    "ASR_MAX_BATCH", "MAX_UPLOAD_MB",
    "NEMO_MAX_AUDIO_S", "NEMO_MATMUL_PRECISION", "NEMO_BF16", "ALLOWED_ORIGINS", "ALLOW_CREDENTIALS",
    "ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES",
)

# Modules the apps import by bare name; the two services each have their own copy.
LOCAL_MODULES = ("model_lifecycle", "transcription", "nemo_common")

WAIT = 5.0  # upper bound for "must happen", never a duration under test

_counter = count()


# --- audio fixtures --------------------------------------------------------------

def wav_bytes(seconds: float, rate: int = 16000, channels: int = 1) -> bytes:
    """A silent 8-bit PCM WAV; 8-bit keeps multi-second clips small."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as out:
        out.setnchannels(channels)
        out.setsampwidth(1)
        out.setframerate(rate)
        out.writeframes(b"\x80" * (int(round(seconds * rate)) * channels))
    return buffer.getvalue()


_FAKE_MP3 = b"FAKEMP3"


def fake_mp3(actual_seconds: float, declared_seconds: float | None = None) -> bytes:
    """A container libsndfile cannot open.

    *declared_seconds* is what its header says (what librosa reports); None means
    the header has no usable length. *actual_seconds* is what decoding yields.
    """
    declared = -1.0 if declared_seconds is None else declared_seconds
    return _FAKE_MP3 + struct.pack("<dd", declared, actual_seconds) + b"\x00" * 256


# --- stand-ins for the heavy packages ---------------------------------------------

def _wave_info(path):
    try:
        with wave.open(str(path), "rb") as fh:
            return SimpleNamespace(
                format="WAV", samplerate=fh.getframerate(), channels=fh.getnchannels(),
                frames=fh.getnframes(),
            )
    except Exception as e:  # wave.Error, EOFError: libsndfile's LibsndfileError
        raise RuntimeError(f"Error opening {path!r}: {e}")


class FakeLibrosa(types.ModuleType):
    """``get_duration`` for the fake mp3; can be made to block, to prove the loop is free."""

    def __init__(self):
        super().__init__("librosa")
        self.calls = 0
        self.entered = threading.Event()
        self.release = threading.Event()
        self.block = False
        self.timed_out = False

    def get_duration(self, path=None, **kwargs):
        self.calls += 1
        if self.block:
            self.entered.set()
            # If the event loop is what is blocked here, nothing can set this.
            self.timed_out = not self.release.wait(WAIT)
        with open(path, "rb") as fh:
            head = fh.read(len(_FAKE_MP3) + 16)
        if not head.startswith(_FAKE_MP3):
            raise RuntimeError("librosa cannot read this")
        declared = struct.unpack("<d", head[len(_FAKE_MP3):len(_FAKE_MP3) + 8])[0]
        if declared < 0:
            raise RuntimeError("no duration in the header")
        return declared


def fake_soundfile() -> types.ModuleType:
    module = types.ModuleType("soundfile")
    module.info = _wave_info
    return module


def fake_torch(cuda: bool, capability=(8, 9), hip=None) -> types.ModuleType:
    torch = types.ModuleType("torch")
    torch.calls = []  # ("matmul", value) / ("empty_cache",)
    torch.bfloat16 = "bfloat16"
    torch.version = SimpleNamespace(hip=hip)
    torch.set_float32_matmul_precision = lambda value: torch.calls.append(("matmul", value))
    torch.cuda = SimpleNamespace(
        is_available=lambda: cuda,
        get_device_capability=lambda index=0: capability,
        empty_cache=lambda: torch.calls.append(("empty_cache",)),
        get_device_name=lambda index=0: "Fake GPU",
        memory_allocated=lambda: 0,
        get_device_properties=lambda index=0: SimpleNamespace(total_memory=8 << 30),
    )
    return torch


def _fake_uvicorn() -> types.ModuleType:
    module = types.ModuleType("uvicorn")
    module.run = lambda *args, **kwargs: None
    return module


def load_app(service: str, *, cuda: bool = False, capability=(8, 9), hip=None, **env):
    """Import a fresh copy of *service* ("parakeet" or "canary") with *env* as its whole configuration.

    ``module.torch`` is the fake torch; the fake librosa is ``module._test_librosa``.
    """
    service_dir = SERVICE_DIRS[service]
    librosa = FakeLibrosa()
    stubs = {
        "torch": fake_torch(cuda, capability, hip),
        "soundfile": fake_soundfile(),
        "librosa": librosa,
        "uvicorn": _fake_uvicorn(),
    }
    saved_env = {key: os.environ.get(key) for key in MANAGED_ENV}
    saved_modules = {name: sys.modules.get(name) for name in (*stubs, *LOCAL_MODULES)}
    name = f"{service}_asr_under_test_{next(_counter)}"
    sys.path.insert(0, str(service_dir))
    try:
        for key in MANAGED_ENV:
            os.environ.pop(key, None)
        os.environ.update({key: str(value) for key, value in env.items()})
        for local in LOCAL_MODULES:
            sys.modules.pop(local, None)  # always the checkout under test, never a cached one
        sys.modules.update(stubs)
        spec = spec_from_file_location(name, service_dir / "app.py")
        module = module_from_spec(spec)
        assert spec.loader is not None
        sys.modules[name] = module
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(service_dir))
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
    module._test_librosa = librosa
    return module


def load_nemo_common(service: str):
    """The shared helper module on its own, with fake soundfile/librosa; returns ``(module, librosa)``."""
    service_dir = SERVICE_DIRS[service]
    name = f"nemo_common_{service}_{next(_counter)}"
    spec = spec_from_file_location(name, service_dir / "nemo_common.py")
    module = module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module  # dataclasses resolves string annotations through sys.modules
    spec.loader.exec_module(module)
    module.sf = fake_soundfile()
    module.librosa = FakeLibrosa()
    return module, module.librosa


# --- fake ffmpeg -------------------------------------------------------------------

FAKE_FFMPEG = f"""#!{sys.executable}
import json, os, struct, sys, time, wave

args = sys.argv[1:]
root = os.environ["FAKE_FFMPEG_DIR"]
mode = open(root + "/mode").read().strip()
src = args[args.index("-i") + 1]
out = args[-1]
limit = float(args[args.index("-t") + 1]) if "-t" in args else None
record = {{"args": args, "src": src}}

if mode == "fail":
    sys.stderr.write("boom: cannot decode")
    open(root + "/calls.jsonl", "a").write(json.dumps(record) + "\\n")
    sys.exit(1)
if mode == "hang":
    open(root + "/pid", "w").write(str(os.getpid()))
    open(root + "/calls.jsonl", "a").write(json.dumps(record) + "\\n")
    time.sleep(60)
if mode == "wait":
    open(root + "/running", "w").write("1")
    deadline = time.time() + 20
    while not os.path.exists(root + "/release") and time.time() < deadline:
        time.sleep(0.01)

data = open(src, "rb").read()
if data.startswith(b"FAKEMP3"):
    seconds = struct.unpack("<d", data[15:23])[0]
else:
    with wave.open(src, "rb") as fh:
        seconds = fh.getnframes() / fh.getframerate()
if limit is not None:
    seconds = min(seconds, limit)
frames = int(round(seconds * 16000))
with wave.open(out, "wb") as fh:
    fh.setnchannels(1); fh.setsampwidth(1); fh.setframerate(16000)
    fh.writeframes(b"\\x80" * frames)
record["wrote_seconds"] = frames / 16000.0
open(root + "/calls.jsonl", "a").write(json.dumps(record) + "\\n")
"""


class FakeFfmpeg:
    """An ``ffmpeg`` executable plus the files it reads and writes."""

    def __init__(self, root: Path):
        self.dir = root / "fake-ffmpeg"
        self.dir.mkdir()
        self.path = self.dir / "ffmpeg"
        self.path.write_text(FAKE_FFMPEG)
        self.path.chmod(self.path.stat().st_mode | stat.S_IEXEC)
        self.mode = "ok"

    def install(self, module, monkeypatch) -> None:
        monkeypatch.setenv("FAKE_FFMPEG_DIR", str(self.dir))
        monkeypatch.setattr(module, "_FFMPEG", str(self.path))

    @property
    def mode(self) -> str:
        return (self.dir / "mode").read_text()

    @mode.setter
    def mode(self, value: str) -> None:
        (self.dir / "mode").write_text(value)

    def calls(self) -> list[dict]:
        log = self.dir / "calls.jsonl"
        if not log.exists():
            return []
        return [json.loads(line) for line in log.read_text().splitlines()]

    def release(self) -> None:
        (self.dir / "release").write_text("1")

    @property
    def running(self) -> bool:
        return (self.dir / "running").exists()

    def hung_pid(self) -> int | None:
        pid = self.dir / "pid"
        return int(pid.read_text()) if pid.exists() else None


def process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    try:  # a zombie still answers signal 0
        return Path(f"/proc/{pid}/stat").read_text().split()[2] != "Z"
    except OSError:
        return False


# --- fake NeMo models --------------------------------------------------------------

def hypothesis(text="hallo welt", segments=None):
    """What NeMo's TDT/AED decoders return with ``timestamps=True``."""
    if segments is None:
        segments = [{"start": 0.0, "end": 1.0, "segment": text}]
    return SimpleNamespace(text=text, timestamp={"segment": segments})


class _FakeModelBase:
    def __init__(self, reply=None):
        self.calls: list[dict] = []
        self.moves: list = []
        self.reply = reply  # callable(paths, kwargs) -> list, or None for the default
        self.eval_called = False

    def eval(self):
        self.eval_called = True
        return self

    def to(self, target):
        self.moves.append(target)
        return self

    def cpu(self):
        self.moves.append("cpu")
        return self

    def _run(self, audio, kwargs):
        paths = list(audio)
        self.calls.append({"paths": paths, **kwargs})
        if self.reply is not None:
            return self.reply(paths, kwargs)
        return [hypothesis(f"text:{Path(p).name}") for p in paths]


class FakeParakeet(_FakeModelBase):
    """``RNNTModel.transcribe``'s signature."""

    def transcribe(self, audio, use_lhotse=True, batch_size=4, return_hypotheses=False,
                   partial_hypothesis=None, num_workers=0, channel_selector=None, augmentor=None,
                   verbose=True, timestamps=None, override_config=None):
        return self._run(audio, {"batch_size": batch_size, "timestamps": timestamps,
                                 "num_workers": num_workers, "verbose": verbose})


class FakeCanary(_FakeModelBase):
    """``EncDecMultiTaskModel.transcribe``'s signature (``**prompt`` takes the languages)."""

    prompt_format = "canary2"

    def transcribe(self, audio, batch_size=4, return_hypotheses=False, num_workers=0,
                   channel_selector=None, augmentor=None, verbose=True, timestamps=None,
                   override_config=None, **prompt):
        return self._run(audio, {"batch_size": batch_size, "timestamps": timestamps, **prompt})


class FakeCanaryWithoutTimestamps(_FakeModelBase):
    """A NeMo build whose ``transcribe`` has no ``timestamps`` parameter (it is all ``**prompt``)."""

    prompt_format = "canary2"

    def transcribe(self, audio, batch_size=4, return_hypotheses=False, num_workers=0,
                   channel_selector=None, augmentor=None, verbose=True, **prompt):
        return self._run(audio, {"batch_size": batch_size, **prompt})


class FakeCanaryV1(FakeCanary):
    """canary-1b: prompt format without a timestamp slot, which rejects the request."""

    prompt_format = "canary"

    def transcribe(self, audio, *args, timestamps=None, **prompt):
        if timestamps:
            raise ValueError("Timestamp feature is not supported in Canary prompt format.")
        return super().transcribe(audio, *args, **prompt)


class OutOfMemoryError(RuntimeError):
    """Same name as torch.cuda.OutOfMemoryError, which is what the services look for."""


def install_model(module, model, *, ttl: float = -1):
    """Make *model* what the app's ModelSlot loads; returns the slot."""
    hook = next(getattr(module, n) for n in dir(module) if n.startswith("_release_"))
    slot = module.ModelSlot(lambda: model, ttl_seconds=ttl, name="test", on_unload=hook)
    module._model_slot = slot
    return slot


def install_fake_nemo(monkeypatch, factory, restore=None):
    """Register ``nemo.collections.asr`` so ``ASRModel.from_pretrained`` calls *factory(model_name)*.

    ``ASRModel.restore_from`` calls *restore(restore_path)* when given, and does not
    exist otherwise, so a load that should not have used it fails loudly.
    """
    asr = types.ModuleType("nemo.collections.asr")
    asr_model = SimpleNamespace(from_pretrained=lambda model_name: factory(model_name))
    if restore is not None:
        asr_model.restore_from = lambda restore_path: restore(restore_path)
    asr.models = SimpleNamespace(ASRModel=asr_model)
    collections = types.ModuleType("nemo.collections")
    collections.asr = asr
    nemo = types.ModuleType("nemo")
    nemo.collections = collections
    monkeypatch.setitem(sys.modules, "nemo", nemo)
    monkeypatch.setitem(sys.modules, "nemo.collections", collections)
    monkeypatch.setitem(sys.modules, "nemo.collections.asr", asr)


# --- HTTP helpers --------------------------------------------------------------------

def make_client(module) -> httpx.AsyncClient:
    """An in-process client; lifespan is not run, so nothing tries to preload a model."""
    transport = httpx.ASGITransport(app=module.app, raise_app_exceptions=False)
    return httpx.AsyncClient(transport=transport, base_url="http://asr.test", timeout=30.0)


def run(coro):
    return asyncio.run(coro)


async def wait_for(predicate, what: str, timeout: float = WAIT) -> None:
    """Poll until *predicate()*; a bound on how long a must-happen may take."""
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError(f"timed out waiting for {what}")
        await asyncio.sleep(0.01)


def tmp_files(directory: Path) -> list[str]:
    return sorted(p.name for p in directory.iterdir())


def use_tmpdir(monkeypatch, directory: Path) -> Path:
    """Send every ``tempfile`` allocation into *directory* so leaks are visible."""
    monkeypatch.setattr(tempfile, "tempdir", str(directory))
    return directory
