"""Shared harness for the piper-tts-service tests (no tests of its own).

`piper-tts-service/app.py` reads its configuration at import time, so a test that
needs different settings imports a fresh copy under a chosen environment. The
heavy audio libraries (librosa, soundfile) are replaced by small stand-ins while
the module loads: the service only needs them for signal analysis and WAV
encoding, and the tests must run where they are not installed. Tests that want the
real thing swap it in with `monkeypatch.setattr(module, "librosa", ...)`.

The `piper` binary is replaced by a shell script on PATH, so the tests exercise
the real subprocess handling (timeouts, kills, concurrency) end to end.

`PIPER_TTS_SERVICE_DIR` points the loader at another checkout of the service,
which is how the new tests are run against the pre-fix code to prove they fail
there.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import stat
import sys
import types
from importlib.util import module_from_spec, spec_from_file_location
from itertools import count
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

REPO = Path(__file__).resolve().parents[1]
SERVICE_DIR = Path(os.environ.get("PIPER_TTS_SERVICE_DIR") or REPO / "piper-tts-service")

# Settings the service reads from the environment. Anything not passed to
# `load_piper_app` is removed while importing, so a variable exported in the
# developer's shell cannot change what a test measures.
MANAGED_ENV = (
    "PIPER_DATA_DIR", "PIPER_OUTPUT_DIR", "PIPER_STRICT_LANGUAGE", "PIPER_DEFAULT_LANGUAGE",
    "PIPER_DEFAULT_VOICE", "PIPER_AUTO_DETECT", "PIPER_TIMEOUT_S", "PIPER_MAX_CONCURRENCY",
    "MAX_TEXT_CHARS", "MAX_UPLOAD_MB", "MAX_ANALYZE_UPLOAD_MB", "OUTPUT_RETENTION_HOURS",
    "OUTPUT_PRUNE_INTERVAL_S", "ONNX_NUM_THREADS", "ONNX_SESSION_CACHE_SIZE",
    "ALLOWED_ORIGINS", "ALLOW_CREDENTIALS", "PIPER_ANALYZE_MAX_SECONDS",
    "PIPER_MAX_CUSTOM_VOICES", "PIPER_MAX_CUSTOM_MB", "PIPER_ONNX_VALIDATE_TIMEOUT_S",
)

_counter = count()

# A valid, tiny WAV (44-byte header + 100 samples of silence).
WAV_BYTES = (
    b"RIFF" + (36 + 200).to_bytes(4, "little") + b"WAVEfmt " + (16).to_bytes(4, "little")
    + (1).to_bytes(2, "little") + (1).to_bytes(2, "little") + (22050).to_bytes(4, "little")
    + (44100).to_bytes(4, "little") + (2).to_bytes(2, "little") + (16).to_bytes(2, "little")
    + b"data" + (200).to_bytes(4, "little") + b"\x00" * 200
)


def _fake_librosa() -> types.ModuleType:
    import numpy as np

    module = types.ModuleType("librosa")
    # `duration` limits the decode, like the real one.
    module.load = lambda path, sr=None, duration=None, **_kw: (np.zeros(100, dtype="float32"), 22050)
    module.feature = types.SimpleNamespace(
        rms=lambda y: np.full((1, 4), 0.25),
        zero_crossing_rate=lambda y: np.full((1, 4), 0.1),
        spectral_centroid=lambda y, sr: np.full((1, 4), 440.0),
    )
    module.effects = types.SimpleNamespace(time_stretch=lambda y, rate: y)
    return module


def _fake_soundfile() -> types.ModuleType:
    module = types.ModuleType("soundfile")

    def write(buf, audio, sample_rate, format="WAV"):
        buf.write(WAV_BYTES)

    module.write = write
    return module


def load_piper_app(env: dict | None = None):
    """Import a fresh copy of the service's app.py under `env`."""
    env = env or {}
    spec = spec_from_file_location(f"piper_tts_harness_{next(_counter)}", SERVICE_DIR / "app.py")
    module = module_from_spec(spec)

    saved_env = {key: os.environ.get(key) for key in MANAGED_ENV}
    stubs = {"librosa": _fake_librosa(), "soundfile": _fake_soundfile()}
    saved_modules = {name: sys.modules.get(name) for name in (*stubs, "naming")}
    sys.path.insert(0, str(SERVICE_DIR))
    try:
        for key in MANAGED_ENV:
            os.environ.pop(key, None)
        os.environ.update(env)
        sys.modules.pop("naming", None)  # always the checkout under test, never a cached one
        sys.modules.update(stubs)
        assert spec.loader is not None
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(SERVICE_DIR))
        for key, value in saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        for name, previous in saved_modules.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous
    return module


def piper_config(language_code="de_DE", quality="medium", dataset="thorsten", sample_rate=22050) -> dict:
    """The shape of a rhasspy/piper-voices `<id>.onnx.json`."""
    return {
        "audio": {"sample_rate": sample_rate, "quality": quality},
        "espeak": {"voice": language_code.split("_")[0]},
        "language": {"code": language_code},
        "dataset": dataset,
        "num_speakers": 1,
    }


def install_voice(models_dir: Path, voice_id: str, config: dict | None = None, model_bytes=b"onnx") -> None:
    """Drop an `<id>.onnx` + `<id>.onnx.json` pair into <models_dir>/default."""
    default = models_dir / "default"
    default.mkdir(parents=True, exist_ok=True)
    if config is None:
        language, dataset, quality = voice_id.split("-")
        config = piper_config(language, quality, dataset)
    (default / f"{voice_id}.onnx").write_bytes(model_bytes)
    (default / f"{voice_id}.onnx.json").write_text(json.dumps(config))


FAKE_PIPER = """#!/bin/sh
# Stand-in for the piper binary. FAKE_PIPER_DIR holds its inputs and records.
model=""; out=""
while [ $# -gt 0 ]; do
  case "$1" in
    --model) model="$2"; shift ;;
    --output_file) out="$2"; shift ;;
  esac
  shift
done
text=$(cat)
printf '%s\\t%s\\n' "$model" "$text" >> "$FAKE_PIPER_DIR/calls.log"
mode=$(cat "$FAKE_PIPER_DIR/mode" 2>/dev/null || echo ok)
case "$mode" in
  fail)
    echo "boom: model exploded" >&2
    printf 'partial' > "$out"
    exit 3 ;;
  hang)
    echo $$ > "$FAKE_PIPER_DIR/pid"
    printf 'partial' > "$out"
    exec sleep 60 ;;
  slow)
    touch "$FAKE_PIPER_DIR/running.$$"
    ls "$FAKE_PIPER_DIR" | grep -c '^running\\.' >> "$FAKE_PIPER_DIR/concurrency.log"
    sleep 0.4
    rm -f "$FAKE_PIPER_DIR/running.$$"
    cat "$FAKE_PIPER_DIR/voice.wav" > "$out" ;;
  *)
    cat "$FAKE_PIPER_DIR/voice.wav" > "$out" ;;
esac
"""


class FakePiper:
    """A `piper` executable on PATH plus the files it reads and writes."""

    def __init__(self, root: Path):
        self.dir = root / "fake-piper"
        self.dir.mkdir()
        (self.dir / "voice.wav").write_bytes(WAV_BYTES)
        script = self.dir / "piper"
        script.write_text(FAKE_PIPER)
        script.chmod(script.stat().st_mode | stat.S_IEXEC)
        self.mode = "ok"

    def env(self, monkeypatch) -> None:
        monkeypatch.setenv("PATH", f"{self.dir}{os.pathsep}{os.environ.get('PATH', '')}")
        monkeypatch.setenv("FAKE_PIPER_DIR", str(self.dir))

    @property
    def mode(self) -> str:
        return (self.dir / "mode").read_text()

    @mode.setter
    def mode(self, value: str) -> None:
        (self.dir / "mode").write_text(value)

    def calls(self) -> list[tuple[str, str]]:
        log = self.dir / "calls.log"
        if not log.exists():
            return []
        return [tuple(line.split("\t", 1)) for line in log.read_text().splitlines()]

    def pid(self) -> int | None:
        pid_file = self.dir / "pid"
        return int(pid_file.read_text()) if pid_file.exists() else None

    def peak_concurrency(self) -> int:
        log = self.dir / "concurrency.log"
        return max((int(x) for x in log.read_text().split()), default=0) if log.exists() else 0


def process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


class Service:
    """A running instance of the app: module, HTTP client, stand-in piper and directories."""

    def __init__(self, module, client, fake, models, out):
        self.module, self.client, self.fake, self.models, self.out = module, client, fake, models, out

    def speak(self, **body):
        body.setdefault("text", "Guten Tag, das ist ein Test.")
        return self.client.post("/tts", json=body)

    def model_used(self):
        """Voice id of the model the last piper invocation was asked to load."""
        model, _ = self.fake.calls()[-1]
        return os.path.basename(model)[: -len(".onnx")]


@pytest.fixture
def start(tmp_path, monkeypatch):
    """start(voices=..., **env) -> Service: install voices, load the app, run its lifespan."""
    stack = contextlib.ExitStack()
    fake = FakePiper(tmp_path)
    fake.env(monkeypatch)

    def _start(voices=("de_DE-thorsten-medium", "en_US-lessac-medium"), validate_models=False, **env):
        """`validate_models=True` keeps the real upload validation (a child process running
        onnxruntime); by default it accepts everything, so the tests can upload placeholder bytes."""
        models, out = tmp_path / "models", tmp_path / "out"
        out.mkdir(exist_ok=True)
        (models / "default").mkdir(parents=True, exist_ok=True)
        for voice in voices:
            install_voice(models, voice)
        module = load_piper_app({"PIPER_DATA_DIR": str(models), "PIPER_OUTPUT_DIR": str(out), **env})
        if not validate_models and hasattr(module, "_validate_onnx"):
            async def accept_any_model(*_args):
                return None

            module._validate_onnx = accept_any_model
        client = stack.enter_context(TestClient(module.app))
        return Service(module, client, fake, models, out)

    yield _start
    stack.close()


async def asgi_post_json(app, path, payload, disconnect_when=None, poll=0.02, limit=15.0):
    """POST JSON straight to the ASGI app; the client hangs up once `disconnect_when()` is true.

    TestClient cannot model a client that goes away mid-request. Returns the
    ASGI messages the app sent.
    """
    body = json.dumps(payload).encode()
    scope = {
        "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": "POST",
        "path": path, "raw_path": path.encode(), "query_string": b"", "root_path": "",
        "scheme": "http", "client": ("127.0.0.1", 1), "server": ("test", 80),
        "headers": [(b"host", b"test"), (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode())],
    }
    hang_up = asyncio.Event()
    sent_body = False
    sent: list = []

    async def receive():
        nonlocal sent_body
        if not sent_body:
            sent_body = True
            return {"type": "http.request", "body": body, "more_body": False}
        await hang_up.wait()
        return {"type": "http.disconnect"}

    async def send(message):
        sent.append(message)

    task = asyncio.ensure_future(app(scope, receive, send))
    deadline = asyncio.get_running_loop().time() + limit
    while not task.done() and asyncio.get_running_loop().time() < deadline:
        if disconnect_when is not None and disconnect_when():
            hang_up.set()
        await asyncio.sleep(poll)
    if not task.done():
        task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    return sent
