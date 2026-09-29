"""Harness for the piper-training-service tests that run without torch or a GPU.

`app.py` imports torch (through the pipeline and the exporter), librosa,
aiohttp and friends at import time, and reads most of its settings from the
environment then, so a test that wants a given configuration has to import a
fresh copy under stand-in modules. This provides that, plus a programmable fake
pipeline and exporter, so the real endpoints, job registry and background
runners can be exercised end to end in a few milliseconds.

What is real: `app.py`, `job_control`, `phonemization`, `audio_sources`,
`data_processor`, `audio_segmenter`, `stt_processor`, `validation`. What is
faked: torch-bound `training_pipeline` / `model_exporter` / `training_utils`,
and the audio and network libraries (`librosa`, `soundfile`, `aiofiles`,
`aiohttp`, `phonemizer`), which are replaced by a few lines each.

`PIPER_TRAINING_SERVICE_DIR` points the loader at another checkout of the
service, which is how the tests were run against the pre-fix code to show they
fail there.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import sys
import threading
import time
import types
import wave
from importlib.util import module_from_spec, spec_from_file_location
from itertools import count
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Optional

import httpx
import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
SERVICE_DIR = Path(os.environ.get("PIPER_TRAINING_SERVICE_DIR") or REPO / "piper-training-service")

# Modules of the service that are imported by their plain name. Dropped from
# sys.modules before and after every test so no state leaks between them.
SERVICE_MODULES = (
    "app", "data_processor", "audio_segmenter", "stt_processor", "phonemization",
    "audio_sources", "job_control", "validation", "training_pipeline",
    "model_exporter", "training_utils",
)
STUB_MODULES = ("librosa", "soundfile", "aiofiles", "aiohttp", "phonemizer", "phonemizer.backend")

# Settings the service reads from the environment; anything not passed to
# `load_training_service` is removed, so the developer's shell cannot change
# what a test measures.
MANAGED_ENV = (
    "TRAINING_MAX_CONCURRENT", "TRAINING_DEFAULT_LANGUAGE", "MAX_TEXT_CHARS", "MAX_UPLOAD_MB",
    "TRAINING_ALLOWED_AUDIO_DIRS", "TRAINING_ALLOWED_URL_HOSTS", "STT_MIN_CONFIDENCE",
    "PHONEMIZER_MAX_FAILURE_FRACTION", "TRAINING_SHUTDOWN_GRACE_S", "TRAINING_DEVICE_TYPE",
    "DEFAULT_DEPLOYMENT_TARGET", "SHARED_MODELS_DIR", "STT_SERVICE_URL", "ALLOW_CLIENT_STT_URL",
    "DEPLOYMENT_TARGETS_JSON", "UVICORN_TIMEOUT_KEEP_ALIVE", "ALLOWED_ORIGINS", "ALLOW_CREDENTIALS",
)

_counter = count()


# --- stand-ins for the audio / network libraries --------------------------------

def _make_librosa() -> types.ModuleType:
    mod = types.ModuleType("librosa")

    def _read(path, sr, offset, duration):
        source = path if hasattr(path, "read") else str(path)
        with wave.open(source, "rb") as w:
            rate = w.getframerate()
            w.setpos(min(int(offset * rate), w.getnframes()))
            frames = w.getnframes() - w.tell() if duration is None else int(duration * rate)
            raw = w.readframes(frames)
        return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0, rate

    def load(path, sr=22050, offset=0.0, duration=None, **_):
        return _read(path, sr, offset, duration)

    def get_duration(path=None, **_):
        with wave.open(str(path), "rb") as w:
            return w.getnframes() / w.getframerate()

    mod.load = load
    mod.get_duration = get_duration
    mod.effects = SimpleNamespace(trim=lambda y, top_db=60: (y, (0, len(y))))
    return mod


def _make_soundfile() -> types.ModuleType:
    mod = types.ModuleType("soundfile")

    def write(path, data, samplerate, **_):
        pcm = (np.clip(np.asarray(data), -1.0, 1.0) * 32767).astype("<i2")
        with wave.open(str(path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(int(samplerate))
            w.writeframes(pcm.tobytes())

    mod.write = write
    return mod


def _make_aiofiles() -> types.ModuleType:
    mod = types.ModuleType("aiofiles")

    class _File:
        def __init__(self, path, mode):
            self._fh = open(path, mode)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            self._fh.close()

        async def read(self, n=-1):
            return self._fh.read(n)

        async def write(self, data):
            return self._fh.write(data)

    mod.open = lambda path, mode="r", **_: _File(path, mode)
    return mod


def _make_aiohttp() -> types.ModuleType:
    mod = types.ModuleType("aiohttp")

    class ClientError(Exception):
        pass

    class ClientConnectionError(ClientError):
        pass

    class ClientConnectorError(ClientConnectionError):
        pass

    class ClientResponseError(ClientError):
        pass

    class FormData:
        def __init__(self):
            self.fields = []

        def add_field(self, name, value, **kwargs):
            self.fields.append((name, value, kwargs))

    class ClientTimeout:
        def __init__(self, total=None, **_):
            self.total = total

    class TCPConnector:
        def __init__(self, **_):
            pass

    class ClientSession:
        """Refuses to connect: a test that expects a request installs its own."""

        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def close(self):
            pass

        def _refuse(self, *args, **kwargs):
            raise ClientConnectorError("the test environment has no network")

        post = get = delete = _refuse

    mod.ClientError = ClientError
    mod.ClientConnectionError = ClientConnectionError
    mod.ClientConnectorError = ClientConnectorError
    mod.ClientResponseError = ClientResponseError
    mod.FormData = FormData
    mod.ClientTimeout = ClientTimeout
    mod.TCPConnector = TCPConnector
    mod.ClientSession = ClientSession
    return mod


def _make_phonemizer():
    """A phonemizer that "works": prefixes the text, and records language and text.

    Tests that need it to fail replace `phonemizer.phonemize`.
    """
    package = types.ModuleType("phonemizer")
    backend = types.ModuleType("phonemizer.backend")
    package.calls = []

    def phonemize(text, language="en-us", backend="espeak", strip=True, **_):
        package.calls.append(SimpleNamespace(text=text, language=language))
        return "p:" + text.lower()

    class EspeakBackend:
        @staticmethod
        def set_library(path):
            pass

    package.phonemize = phonemize
    package.backend = backend
    backend.EspeakBackend = EspeakBackend
    return package, backend


# --- the fake trainer and exporter ------------------------------------------------

class FakePipeline:
    """Stands in for OptimizedTrainingPipeline. Blocks until released when asked to.

    While blocked it polls the job status the way the real loop does at each
    epoch boundary and raises TrainingCancelled, so cancel behaves like the real
    thing: it takes effect only once the "thread" gets to look.
    """

    def __init__(self, cancelled_exc):
        self.device = SimpleNamespace(type="cpu")
        self.cancelled_exc = cancelled_exc
        self.calls: list = []
        self.block = False
        self.error: Optional[Exception] = None
        self.started = threading.Event()
        self.release = threading.Event()
        self.observed_status: list = []

    def train_sync(self, job_id, request, callback=None, override_config=None, resume_from=None):
        self.calls.append(SimpleNamespace(job_id=job_id, request=request, resume_from=resume_from))
        self.started.set()
        if callback:
            callback({"current_epoch": 1, "total_epochs": request.epochs, "progress": 10.0, "loss": 1.5})
        if self.block:
            while not self.release.wait(0.005):
                status = callback({"check_status": True}) if callback else None
                self.observed_status.append(getattr(status, "status", None))
                if status is not None and getattr(status, "status", None) == "cancelled":
                    raise self.cancelled_exc("Training cancelled at epoch 2/10")
        if self.error is not None:
            raise self.error
        checkpoint_dir = Path("checkpoints") / job_id
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        (checkpoint_dir / "final_model.pt").write_bytes(b"weights")
        if callback:
            callback({"status": "completed", "message": "Training completed successfully"})


class FakeExporter:
    """Stands in for ModelExporter. `blocking_seconds` blocks inside the coroutine, as torch does."""

    def __init__(self):
        self.calls: list = []
        self.error: Optional[Exception] = None
        self.blocking_seconds = 0.0

    async def export_to_onnx(self, job_id):
        self.calls.append(job_id)
        if self.blocking_seconds:
            time.sleep(self.blocking_seconds)
        if self.error is not None:
            raise self.error
        export_dir = Path("models") / job_id
        export_dir.mkdir(parents=True, exist_ok=True)
        onnx_path = export_dir / f"{job_id}.onnx"
        onnx_path.write_bytes(b"onnx")
        (export_dir / f"{job_id}.json").write_text("{}")
        return onnx_path


# --- loading ------------------------------------------------------------------------

def load_training_service(
    stack: contextlib.ExitStack,
    monkeypatch,
    tmp_path: Path,
    env: Optional[dict] = None,
    trainer_kind: Optional[str] = "experimental-mel-flow",
):
    """Import a fresh copy of the service under `env`, with the working directory at `tmp_path`.

    Returns a namespace with the app module and the fake pipeline / exporter.
    """
    saved = {name: sys.modules.get(name) for name in SERVICE_MODULES + STUB_MODULES}

    def _restore():
        for name, module in saved.items():
            sys.modules.pop(name, None)
            if module is not None:
                sys.modules[name] = module

    stack.callback(_restore)

    for name in SERVICE_MODULES + STUB_MODULES:
        sys.modules.pop(name, None)

    monkeypatch.chdir(tmp_path)
    for key in MANAGED_ENV:
        monkeypatch.delenv(key, raising=False)
    merged = {"DEFAULT_DEPLOYMENT_TARGET": "none", **(env or {})}
    for key, value in merged.items():
        monkeypatch.setenv(key, str(value))
    monkeypatch.syspath_prepend(str(SERVICE_DIR))

    sys.modules["librosa"] = _make_librosa()
    sys.modules["soundfile"] = _make_soundfile()
    sys.modules["aiofiles"] = _make_aiofiles()
    sys.modules["aiohttp"] = _make_aiohttp()
    phonemizer, phonemizer_backend = _make_phonemizer()
    sys.modules["phonemizer"] = phonemizer
    sys.modules["phonemizer.backend"] = phonemizer_backend

    class TrainingCancelled(Exception):
        pass

    pipeline = FakePipeline(TrainingCancelled)
    exporter = FakeExporter()

    training_pipeline = types.ModuleType("training_pipeline")
    training_pipeline.OptimizedTrainingPipeline = lambda: pipeline
    training_pipeline.TrainingCancelled = TrainingCancelled
    model_exporter = types.ModuleType("model_exporter")
    model_exporter.ModelExporter = lambda: exporter
    training_utils = types.ModuleType("training_utils")
    if trainer_kind:
        training_utils.TRAINER_KIND = trainer_kind
        training_utils.TRAINER_CAVEAT = "Experimental trainer: the exported voice is not intelligible speech."
    sys.modules["training_pipeline"] = training_pipeline
    sys.modules["model_exporter"] = model_exporter
    sys.modules["training_utils"] = training_utils

    spec = spec_from_file_location(f"piper_training_app_{next(_counter)}", SERVICE_DIR / "app.py")
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    stack.callback(sys.modules.pop, spec.name, None)
    spec.loader.exec_module(module)

    # The mel maths is librosa's, not what these tests are about.
    module.data_processor._compute_mel_spectrogram = lambda audio: np.zeros((80, 4), dtype=np.float32)

    return SimpleNamespace(
        module=module, pipeline=pipeline, exporter=exporter, TrainingCancelled=TrainingCancelled,
        root=tmp_path, aiohttp=sys.modules["aiohttp"], librosa=sys.modules["librosa"],
        phonemizer=sys.modules["phonemizer"],
    )


@pytest.fixture
def training_service(monkeypatch, tmp_path):
    """Factory: `svc = training_service(env={...})`. Everything is undone after the test."""
    with contextlib.ExitStack() as stack:
        def factory(env: Optional[dict] = None, trainer_kind: Optional[str] = "experimental-mel-flow"):
            return load_training_service(stack, monkeypatch, tmp_path, env, trainer_kind)

        yield factory


# --- helpers -------------------------------------------------------------------------

def asgi_client(module) -> httpx.AsyncClient:
    """An in-process client for the app; use `async with`."""
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=module.app), base_url="http://training")


def write_wav(path: Path, seconds: float = 1.5, rate: int = 22050) -> Path:
    """A real mono 16-bit WAV of a sine tone."""
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.arange(int(seconds * rate)) / rate
    pcm = (0.4 * np.sin(2 * np.pi * 220 * t) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())
    return path


def make_dataset(root: Path, model_name: str) -> Path:
    """The two files /train-from-dataset requires."""
    import json

    data = root / "data" / model_name
    data.mkdir(parents=True, exist_ok=True)
    entry = [{"audio_path": "audio/a.wav", "mel_path": "mel/a.npy", "text": "hallo welt", "phonemes": "hal oː"}]
    (data / "train.json").write_text(json.dumps(entry * 3))
    (data / "val.json").write_text(json.dumps(entry))
    return data


def write_job_state(root: Path, job_id: str, model_name: str, *, status: str = "training",
                    epoch: int = 5, total_epochs: int = 10, with_checkpoint: bool = True, **extra) -> Path:
    """A job_state.json (and the checkpoint it points at) as the trainer leaves them."""
    import json

    directory = root / "checkpoints" / job_id
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint = directory / f"checkpoint_epoch_{epoch}.pt"
    if with_checkpoint:
        checkpoint.write_bytes(b"checkpoint")
    (directory / "job_state.json").write_text(json.dumps({
        "job_id": job_id, "model_name": model_name, "language": "de", "epoch": epoch,
        "total_epochs": total_epochs, "loss": 0.5, "latest_checkpoint": str(checkpoint),
        "status": status, "config": {"sample_rate": 22050, "batch_size": 4}, **extra,
    }))
    return directory


async def wait_until(predicate: Callable[[], bool], timeout: float = 5.0, interval: float = 0.005):
    """Poll until `predicate()` is true; fail the test on timeout."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        await asyncio.sleep(interval)
    pytest.fail(f"condition not reached within {timeout}s")


async def settle(svc, timeout: float = 5.0):
    """Wait for every background job runner to finish (releasing a blocked pipeline first)."""
    svc.pipeline.release.set()
    tasks = list(getattr(svc.module, "_runner_tasks", ()))
    if tasks:
        await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout)


def status_of(svc, job_id: str) -> str:
    return svc.module.training_jobs[job_id].status


class LoopLag:
    """Measures how long the event loop went without getting a turn.

    A handler that does blocking work inline holds the loop for its whole
    duration, so /health and /status cannot answer; a ticker that asks for a
    10 ms nap and records how late it woke up sees exactly that gap.
    """

    def __init__(self, interval: float = 0.01):
        self.interval = interval
        self.max_gap = 0.0
        self._task = None

    async def _tick(self):
        while True:
            before = time.monotonic()
            await asyncio.sleep(self.interval)
            self.max_gap = max(self.max_gap, time.monotonic() - before)

    async def __aenter__(self):
        self._task = asyncio.get_running_loop().create_task(self._tick())
        await asyncio.sleep(self.interval * 3)  # let it start ticking
        return self

    async def __aexit__(self, *exc):
        # After a blocked stretch the ticker is due but has not run yet; without
        # this turn it is cancelled before it can record how late it was.
        await asyncio.sleep(self.interval * 3)
        self._task.cancel()
        await asyncio.gather(self._task, return_exceptions=True)
        return False


def install_fake_stt(monkeypatch, responder, gate=None):
    """Replace the STT client with one that never touches the network.

    Subclasses the real `STTProcessor`, so its parsing, confidence mapping and
    quality filter still run; only `transcribe_audio_file` (the HTTP call) is
    replaced by `responder(processor, audio_path) -> dict`, optionally after
    `gate` (an asyncio.Event) is set. Returns the class; `.instances` lists
    the processors created.
    """
    import audio_segmenter  # noqa: F401  (imported so its reference can be replaced)

    real = sys.modules["stt_processor"].STTProcessor

    class FakeSTT(real):
        instances: list = []
        waiting = 0

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            FakeSTT.instances.append(self)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def transcribe_audio_file(self, audio_path, return_segments=True, max_retries=3):
            if gate is not None:
                FakeSTT.waiting += 1
                await gate.wait()
            return await responder(self, audio_path)

    for name in ("stt_processor", "audio_segmenter"):
        monkeypatch.setattr(sys.modules[name], "STTProcessor", FakeSTT)
    app_module = next((m for n, m in sys.modules.items() if n.startswith("piper_training_app_")), None)
    if app_module is not None and hasattr(app_module, "STTProcessor"):
        monkeypatch.setattr(app_module, "STTProcessor", FakeSTT)
    return FakeSTT


def stt_result(text="das ist ein längerer test satz", seconds=2.0, avg_logprob=-0.1, count=1):
    """An STT response in the shape stt-service returns: avg_logprob, never confidence."""
    step = seconds
    return {
        "text": text,
        "segments": [
            {"start": i * step, "end": (i + 1) * step, "text": text,
             "avg_logprob": avg_logprob, "no_speech_prob": 0.01}
            for i in range(count)
        ],
        "language": "de",
    }
