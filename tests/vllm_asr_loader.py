"""Load vllm-asr-service's modules, and its app under stand-ins (no tests of its own).

``load_module(name)`` imports one light module (``vllm_support``, ``vllm_engine``,
``vad_pool``, ...) from the service directory, with that directory's own siblings.

``load_app(monkeypatch, **env)`` imports a fresh copy of ``vllm-asr-service/app.py``
with *env* as its whole configuration and fakes for everything heavy:

* the engine: ``FakeEngine`` stands in for ``vllm serve``. Its processes are
  ``FakeProcess`` objects (``module._popen``), its process group is "signalled" by
  marking them dead (``module._signal_group``), and its HTTP API answers through
  ``httpx.MockTransport`` (``module._SYNC_TRANSPORT`` / ``_ASYNC_TRANSPORT``):
  ``/health``, ``/sleep``, ``/wake_up``, ``/is_sleeping``,
  ``/v1/audio/transcriptions`` (multipart, like the real one) and
  ``/v1/chat/completions`` (``input_audio``). Like the transformers fakes it "hears"
  words: a run of constant amplitude is ``w<level>``.
* ``huggingface_hub.snapshot_download``: a temporary model directory with a
  config.json (``model_type`` of the configured family) and a small model.safetensors.
* ``silero_vad`` and ``torch`` (the VAD's), ``soundfile``/``librosa``/``uvicorn``: the
  transformers loader's and the qwen3-asr harness's. ffmpeg is switched off, so
  uploads decode through the fake librosa on any OS.

The VAD runs in the test process (``VLLM_ASR_VAD_WORKERS=0`` unless a test says
otherwise). The fakes hang off the module: ``module._fake`` (a ``FakeEngine``).
"""

from __future__ import annotations

import base64
import io
import json
import os
import subprocess
import sys
import tempfile
import threading
import types
import wave
from importlib.util import module_from_spec, spec_from_file_location
from itertools import count
from pathlib import Path

import httpx
import numpy as np

from test_qwen3_asr_harness import FakeLibrosa, fake_soundfile, hear_words
from transformers_asr_loader import fake_torch

REPO = Path(__file__).resolve().parents[1]
SERVICE_DIR = Path(os.environ.get("VLLM_ASR_SERVICE_DIR") or REPO / "vllm-asr-service")
LOCAL_MODULES = ("audio_pipeline", "model_lifecycle", "body_limit", "origin_guard", "transformers_asr",
                 "cohere_support", "granite_support", "vllm_support", "vllm_engine", "vad_pool")
MANAGED_ENV = (
    "VLLM_ASR_MODEL", "VLLM_ASR_REVISION", "VLLM_ASR_SHA256", "VLLM_ASR_DEFAULT_LANGUAGE",
    "VLLM_ASR_SUPPORTED_LANGUAGES", "VLLM_ASR_GPU_MEMORY_UTILIZATION", "VLLM_ASR_MAX_MODEL_LEN",
    "VLLM_ASR_MAX_NUM_SEQS", "VLLM_ASR_MAX_NUM_BATCHED_TOKENS", "VLLM_ASR_EXTRA_ARGS", "VLLM_ASR_IDLE_MODE",
    "VLLM_ASR_ENGINE_PORT", "VLLM_ASR_START_TIMEOUT_S", "VLLM_ASR_REQUEST_TIMEOUT_S", "VLLM_ASR_MAX_PENDING_CHUNKS",
    "VLLM_ASR_CHUNK_S", "VLLM_ASR_MAX_NEW_TOKENS", "VLLM_ASR_VAD", "VLLM_ASR_VAD_THRESHOLD",
    "VLLM_ASR_VAD_WORKERS",
    "HF_TOKEN", "ASR_MODEL_TTL", "MODEL_TTL", "ASR_MAX_CONCURRENCY", "ASR_MAX_QUEUE", "ASR_QUEUE_TIMEOUT_S",
    "MAX_UPLOAD_MB", "MAX_AUDIO_SECONDS", "ALLOWED_ORIGINS", "ALLOW_CREDENTIALS", "CUDA_VISIBLE_DEVICES",
)
MODEL_TYPES = {
    "ibm-granite/granite-speech-5.0-2b": "granite_speech",
    "evewashere/cohere-transcribe-03-2026-ungated": "cohere_asr",
    "CohereLabs/cohere-transcribe-03-2026": "cohere_asr",
    "Qwen/Qwen3-ASR-1.7B": "qwen3_asr",
    "primeline/whisper-large-v3-turbo-german": "whisper",
}
RATE = 16000
WAIT = 5.0
WEIGHTS = b"fake safetensors bytes"
KV_LINE = "(EngineCore pid=7) INFO [kv_cache_utils.py] GPU KV cache size: 12,864 tokens, Maximum concurrency 6x"

_counter = count()
_pids = count(4000)


# --- light modules ---------------------------------------------------------------------------


def load_module(name: str):
    """``vllm-asr-service/<name>.py`` with that directory's own siblings, under a fresh name."""
    saved = {mod: sys.modules.get(mod) for mod in LOCAL_MODULES}
    sys.path.insert(0, str(SERVICE_DIR))
    try:
        for mod in LOCAL_MODULES:
            sys.modules.pop(mod, None)
        unique = f"{name}_vllm_under_test_{next(_counter)}"
        spec = spec_from_file_location(unique, SERVICE_DIR / f"{name}.py")
        module = module_from_spec(spec)
        sys.modules[unique] = module  # dataclasses resolves annotations through sys.modules
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(SERVICE_DIR))
        for mod, previous in saved.items():
            if previous is None:
                sys.modules.pop(mod, None)
            else:
                sys.modules[mod] = previous


# --- the engine --------------------------------------------------------------------------------


def wav_words(data: bytes) -> list[str]:
    """What the fake engine hears in a 16-bit WAV."""
    with wave.open(io.BytesIO(data), "rb") as wav:
        assert wav.getframerate() == RATE and wav.getnchannels() == 1 and wav.getsampwidth() == 2
        frames = wav.readframes(wav.getnframes())
    samples = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    return hear_words(samples)


def parse_multipart(request: httpx.Request) -> tuple[dict, dict]:
    """``(fields, files)`` of a multipart request, as the fake engine needs them."""
    boundary = request.headers["content-type"].split("boundary=", 1)[1].encode()
    fields, files = {}, {}
    for part in request.content.split(b"--" + boundary):
        part = part.strip(b"\r\n")
        if not part or part == b"--":
            continue
        head, _, body = part.partition(b"\r\n\r\n")
        header = head.decode()
        name = header.split('name="', 1)[1].split('"', 1)[0]
        if 'filename="' in header:
            files[name] = body
        else:
            fields[name] = body.decode()
    return fields, files


class FakeProcess:
    """A ``subprocess.Popen`` of ``vllm serve`` that lives until its group is signalled."""

    def __init__(self, engine: "FakeEngine", argv, env=None, **kwargs):
        self.engine = engine
        self.argv = list(argv)
        self.env = dict(env or {})
        self.kwargs = kwargs
        self.pid = next(_pids)
        self.returncode = None
        self.signals: list[int] = []
        lines = list(engine.start_log)
        if engine.exit_on_start is not None:
            self.returncode = engine.exit_on_start
        self.stdout = io.StringIO("".join(line + "\n" for line in lines))

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        if self.returncode is None:
            raise subprocess.TimeoutExpired(self.argv, timeout)
        return self.returncode


class FakeEngine:
    """``vllm serve`` as the adapter uses it, for every process the app starts."""

    def __init__(self, model_dir: Path, model_type: str):
        self.model_dir = model_dir
        self.model_type = model_type
        self.processes: list[FakeProcess] = []
        self.downloads: list[dict] = []
        self.requests: list[dict] = []          # every chunk request: path, fields or body
        self.control: list[str] = []            # /sleep, /wake_up, ... in order
        self.start_log = [KV_LINE]
        self.exit_on_start = None               # a returncode: the process dies while starting
        self.health_failures = 0                # /health polls that fail before it answers
        self.ignore_term = False                # the process survives SIGTERM (KILL ends it)
        self.sleeping = False
        self.chunk_status = 200
        self.in_flight = 0
        self.max_in_flight = 0
        self.hold: threading.Event | None = None  # chunk requests wait for it
        self.entered = threading.Event()
        self.die_on_chunk = False               # the process exits when a chunk arrives
        self._lock = threading.Lock()

    # huggingface_hub
    def snapshot_download(self, repo_id, revision=None, allow_patterns=None, token=None, **kwargs):
        self.downloads.append({"repo_id": repo_id, "revision": revision, "allow_patterns": allow_patterns,
                               "token": token})
        return str(self.model_dir)

    # subprocess and signals
    def popen(self, argv, **kwargs) -> FakeProcess:
        process = FakeProcess(self, argv, **kwargs)
        self.processes.append(process)
        return process

    def signal(self, process: FakeProcess, sig: int) -> None:
        process.signals.append(sig)
        if self.ignore_term and len(process.signals) == 1:
            return  # survives the first signal (TERM); KILL ends it
        if process.returncode is None:
            process.returncode = -int(sig)

    @property
    def current(self) -> FakeProcess | None:
        return self.processes[-1] if self.processes else None

    def alive(self) -> bool:
        return self.current is not None and self.current.returncode is None

    # the control API (sync client)
    def handle_sync(self, request: httpx.Request) -> httpx.Response:
        if not self.alive():
            raise httpx.ConnectError("connection refused", request=request)
        path = request.url.path
        if path == "/health":
            if self.health_failures > 0:
                self.health_failures -= 1
                raise httpx.ConnectError("not listening yet", request=request)
            return httpx.Response(200)
        self.control.append(path)
        if path == "/sleep":
            self.sleeping = True
            return httpx.Response(200)
        if path == "/wake_up":
            self.sleeping = False
            return httpx.Response(200)
        if path == "/is_sleeping":
            return httpx.Response(200, json={"is_sleeping": self.sleeping})
        return httpx.Response(404)

    # the chunk API (async client)
    async def handle_async(self, request: httpx.Request) -> httpx.Response:
        import asyncio

        if not self.alive():
            raise httpx.ConnectError("connection refused", request=request)
        if self.die_on_chunk:
            self.current.returncode = -9
            raise httpx.RemoteProtocolError("server disconnected", request=request)
        await request.aread()
        path = request.url.path
        if path == "/v1/audio/transcriptions":
            fields, files = parse_multipart(request)
            record = {"path": path, "fields": fields}
            words = wav_words(files["file"])
        elif path == "/v1/chat/completions":
            body = json.loads(request.content)
            content = body["messages"][0]["content"]
            audio = next(part for part in content if part["type"] == "input_audio")
            text = next(part for part in content if part["type"] == "text")
            record = {"path": path, "body": body, "prompt": text["text"], "order": [p["type"] for p in content]}
            words = wav_words(base64.b64decode(audio["input_audio"]["data"]))
        else:
            return httpx.Response(404)
        with self._lock:
            self.requests.append(record)
            self.in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self.in_flight)
        self.entered.set()
        try:
            if self.hold is not None:
                await asyncio.to_thread(self.hold.wait, WAIT)
            if self.chunk_status != 200:
                return httpx.Response(self.chunk_status, json={"error": {"message": "refused"}})
            if path == "/v1/chat/completions":
                cap = int(record["body"]["max_tokens"])
                capped = len(words) > cap
                return httpx.Response(200, json={"choices": [{
                    "message": {"role": "assistant", "content": " ".join(words[:cap])},
                    "finish_reason": "length" if capped else "stop"}]})
            # Whisper's convention: the text starts with a space.
            return httpx.Response(200, json={"text": "".join(" " + w for w in words),
                                             "usage": {"type": "duration", "seconds": 1}})
        finally:
            with self._lock:
                self.in_flight -= 1


# --- the app ---------------------------------------------------------------------------------------


def _fake_silero() -> types.ModuleType:
    module = types.ModuleType("silero_vad")
    module.load_silero_vad = lambda: "fake-silero-model"

    def get_speech_timestamps(audio, model, **kwargs):
        samples = np.asarray(audio)
        loud = np.flatnonzero(np.abs(samples) > 1e-4)
        if loud.size == 0:
            return []
        gaps = np.flatnonzero(np.diff(loud) > int(0.3 * RATE))
        starts = np.concatenate(([loud[0]], loud[gaps + 1]))
        ends = np.concatenate((loud[gaps] + 1, [loud[-1] + 1]))
        return [{"start": int(a), "end": int(b)} for a, b in zip(starts, ends)]

    module.get_speech_timestamps = get_speech_timestamps
    return module


def _fake_hub(engine: FakeEngine) -> types.ModuleType:
    module = types.ModuleType("huggingface_hub")
    module.snapshot_download = engine.snapshot_download
    return module


def _fake_uvicorn() -> types.ModuleType:
    module = types.ModuleType("uvicorn")
    module.run = lambda *args, **kwargs: None
    return module


# Captured at import: the hardening tests point tempfile at a spool directory and count what is
# left in it, and the fake checkpoint is not something the service left behind.
_MODEL_ROOT = Path(tempfile.gettempdir())


def _model_dir(model_type: str) -> Path:
    directory = Path(tempfile.mkdtemp(prefix="fake-vllm-model-", dir=_MODEL_ROOT))
    (directory / "config.json").write_text('{"model_type": "%s"}' % model_type, encoding="utf-8")
    (directory / "model.safetensors").write_bytes(WEIGHTS)
    return directory


def load_app(monkeypatch, *, model_type: str | None = None, **env):
    """A fresh copy of the service's app with *env* as its whole configuration (see the module doc)."""
    env = {"VLLM_ASR_VAD_WORKERS": "0", **env}
    model = str(env.get("VLLM_ASR_MODEL") or "ibm-granite/granite-speech-5.0-2b")
    engine = FakeEngine(_model_dir(model_type or MODEL_TYPES.get(model, "whisper")), model_type or "")
    monkeypatch.setitem(sys.modules, "huggingface_hub", _fake_hub(engine))
    monkeypatch.setitem(sys.modules, "silero_vad", _fake_silero())
    # The VAD imports torch when it first runs, i.e. during the test.
    monkeypatch.setitem(sys.modules, "torch", fake_torch())
    stubs = {"torch": fake_torch(), "soundfile": fake_soundfile(), "librosa": FakeLibrosa(),
             "uvicorn": _fake_uvicorn()}
    saved_env = {key: os.environ.get(key) for key in MANAGED_ENV}
    saved_modules = {name: sys.modules.get(name) for name in (*stubs, *LOCAL_MODULES)}
    name = f"vllm_asr_under_test_{next(_counter)}"
    sys.path.insert(0, str(SERVICE_DIR))
    try:
        for key in MANAGED_ENV:
            os.environ.pop(key, None)
        os.environ.update({key: str(value) for key, value in env.items()})
        for local in LOCAL_MODULES:
            sys.modules.pop(local, None)
        sys.modules.update(stubs)
        spec = spec_from_file_location(name, SERVICE_DIR / "app.py")
        module = module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        module._cuda_after_import = os.environ.get("CUDA_VISIBLE_DEVICES")
    finally:
        sys.path.remove(str(SERVICE_DIR))
        for key, value in saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        for mod, previous in saved_modules.items():
            if previous is None:
                sys.modules.pop(mod, None)
            else:
                sys.modules[mod] = previous
    module._fake = engine
    module._popen = engine.popen
    module._signal_group = engine.signal
    module._SYNC_TRANSPORT = httpx.MockTransport(engine.handle_sync)
    module._ASYNC_TRANSPORT = httpx.MockTransport(engine.handle_async)
    module._ENGINE_POLL_S = 0.01
    module._FFMPEG = None
    return module
