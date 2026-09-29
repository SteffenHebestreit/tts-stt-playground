"""Shared harness for the qwen3-asr-service tests (no tests of its own).

The app imports torch, librosa and soundfile at module scope and reads its limits
from the environment at import, so a test that needs another configuration
imports a fresh copy of it under a chosen environment, with stand-ins for the
heavy packages:

* ``torch``: only the surface the service touches (CPU, no CUDA).
* ``soundfile.info`` and ``librosa.load`` / ``get_duration`` read real WAV files
  with the stdlib ``wave`` module and refuse anything else, the way libsndfile
  refuses an mp3 it cannot open.
* a fake ``ffmpeg`` executable (a Python script) that decodes WAV to raw float32
  on stdout, honours ``-t``, records its argument list, and can fail or hang.
* a fake ``qwen_asr.Qwen3ASRModel`` whose ``transcribe`` follows the real one
  (read from the qwen-asr 0.0.6 wheel): it accepts a path or ``(array, sr)``
  items, splits its own input at 1200 s, and generates at most ``max_new_tokens``
  tokens *per such piece*, joining the pieces with ``""``. That per-piece cap is
  the behaviour the service's chunking exists for.

The fake model "hears" words instead of recognising speech: a word is a run of
constant amplitude, its text is ``w<level>`` (see ``encoded_wav``), so a test can
compute the exact transcript a lossless pipeline must produce.

``QWEN3_ASR_SERVICE_DIR`` points the loader at another checkout of the service,
which is how the tests are run against the pre-fix code to prove they fail there.
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
import numpy as np

REPO = Path(__file__).resolve().parents[1]
SERVICE_DIR = Path(os.environ.get("QWEN3_ASR_SERVICE_DIR") or REPO / "qwen3-asr-service")

# Settings the service reads from the environment. Anything not passed to
# `load_app` is removed while importing, so a variable exported in the developer's
# shell cannot change what a test measures.
MANAGED_ENV = (
    "QWEN3_ASR_MODEL", "QWEN3_ASR_CHUNK_S", "QWEN3_ASR_BATCH_SIZE", "QWEN3_ASR_MAX_NEW_TOKENS",
    "QWEN3_ASR_ATTN_IMPLEMENTATION", "ASR_MODEL_TTL", "MODEL_TTL", "ASR_MAX_CONCURRENCY",
    "ASR_MAX_QUEUE", "ASR_QUEUE_TIMEOUT_S", "MAX_UPLOAD_MB", "MAX_AUDIO_SECONDS", "ALLOWED_ORIGINS", "ALLOW_CREDENTIALS",
    "ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES",
)

# Modules the app imports by bare name.
LOCAL_MODULES = ("audio_pipeline", "model_lifecycle")

WAIT = 5.0  # upper bound for "must happen", never a duration under test

RATE = 16000
LIBRARY_PIECE_S = 1200  # qwen-asr's own split, MAX_ASR_INPUT_SECONDS
WORD_AMPLITUDE = 50     # int16 amplitude per word level

_counter = count()


# --- audio fixtures --------------------------------------------------------------

def wav_from_int16(samples: np.ndarray, rate: int = RATE) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(rate)
        out.writeframes(np.asarray(samples, dtype="<i2").tobytes())
    return buffer.getvalue()


def speech_levels(words: int, *, pause_every: int = 10) -> list[int]:
    """Word levels 1..500 (cycling; neighbours always differ), 0 = a quarter second of silence.

    A pause after every *pause_every* words is where the service may cut without
    tearing a word.
    """
    levels = []
    for i in range(words):
        levels.append(i % 500 + 1)
        if pause_every and (i + 1) % pause_every == 0:
            levels.append(0)
    return levels


def encoded_wav(levels: list[int], *, block_s: float = 0.25, rate: int = RATE) -> bytes:
    """A WAV with one block of ``block_s`` seconds per level: constant amplitude, or silence for 0."""
    block = int(block_s * rate)
    samples = np.repeat(np.asarray(levels, dtype=np.int32) * WORD_AMPLITUDE, block)
    return wav_from_int16(samples, rate)


def expected_transcript(levels: list[int]) -> str:
    return " ".join(f"w{level}" for level in levels if level > 0)


def silent_wav(seconds: float, rate: int = RATE) -> bytes:
    return wav_from_int16(np.zeros(int(seconds * rate), dtype=np.int16), rate)


_FAKE_MP3 = b"FAKEMP3"


def fake_mp3(actual_seconds: float, declared_seconds: float | None = None) -> bytes:
    """A container libsndfile cannot open.

    *declared_seconds* is what its header says (what librosa reports); None means
    the header has no usable length. *actual_seconds* is what decoding yields.
    """
    declared = -1.0 if declared_seconds is None else declared_seconds
    return _FAKE_MP3 + struct.pack("<dd", declared, actual_seconds) + b"\x00" * 256


def _read_wav(path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as fh:
        rate, width, channels = fh.getframerate(), fh.getsampwidth(), fh.getnchannels()
        raw = fh.readframes(fh.getnframes())
    if width != 2:
        raise RuntimeError("the fake reads 16-bit WAV only")
    data = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    return data, rate


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
    """``get_duration`` and ``load`` for WAV files and the fake mp3 header."""

    def __init__(self):
        super().__init__("librosa")
        self.load_calls: list[dict] = []
        self.duration_calls = 0
        self.fail = False

    def get_duration(self, path=None, **kwargs):
        self.duration_calls += 1
        with open(path, "rb") as fh:
            head = fh.read(len(_FAKE_MP3) + 16)
        if head.startswith(_FAKE_MP3):
            declared = struct.unpack("<d", head[len(_FAKE_MP3):len(_FAKE_MP3) + 8])[0]
            if declared < 0:
                raise RuntimeError("no duration in the header")
            return declared
        return _wave_info(path).frames / _wave_info(path).samplerate

    def load(self, path, sr=None, mono=True, duration=None, **kwargs):
        self.load_calls.append({"path": str(path), "sr": sr, "mono": mono, "duration": duration})
        if self.fail:
            raise RuntimeError("librosa cannot read this")
        try:
            data, rate = _read_wav(path)
        except Exception as e:
            raise RuntimeError(f"librosa cannot read {path!r}: {e}")
        if sr and sr != rate:
            raise RuntimeError("the fake does not resample")
        if duration is not None:
            data = data[: int(duration * rate)]
        return data, rate


def fake_soundfile() -> types.ModuleType:
    module = types.ModuleType("soundfile")
    module.info = _wave_info
    return module


def fake_torch() -> types.ModuleType:
    torch = types.ModuleType("torch")
    torch.float32 = "float32"
    torch.float16 = "float16"
    torch.bfloat16 = "bfloat16"
    torch.cuda = SimpleNamespace(
        is_available=lambda: False,
        is_bf16_supported=lambda: True,
        empty_cache=lambda: None,
        ipc_collect=lambda: None,
        get_device_name=lambda index=0: "Fake GPU",
        memory_allocated=lambda: 0,
        get_device_properties=lambda index=0: SimpleNamespace(total_memory=8 << 30),
    )
    return torch


def _fake_uvicorn() -> types.ModuleType:
    module = types.ModuleType("uvicorn")
    module.run = lambda *args, **kwargs: None
    return module


def load_app(**env):
    """Import a fresh copy of the service with *env* as its whole configuration.

    The fake librosa is ``module._test_librosa``; ``module._audio_pipeline`` is the
    copy of ``audio_pipeline`` the app imported.
    """
    librosa = FakeLibrosa()
    stubs = {
        "torch": fake_torch(),
        "soundfile": fake_soundfile(),
        "librosa": librosa,
        "uvicorn": _fake_uvicorn(),
    }
    saved_env = {key: os.environ.get(key) for key in MANAGED_ENV}
    saved_modules = {name: sys.modules.get(name) for name in (*stubs, *LOCAL_MODULES)}
    name = f"qwen3_asr_under_test_{next(_counter)}"
    sys.path.insert(0, str(SERVICE_DIR))
    try:
        for key in MANAGED_ENV:
            os.environ.pop(key, None)
        os.environ.update({key: str(value) for key, value in env.items()})
        for local in LOCAL_MODULES:
            sys.modules.pop(local, None)  # always the checkout under test, never a cached one
        sys.modules.update(stubs)
        spec = spec_from_file_location(name, SERVICE_DIR / "app.py")
        module = module_from_spec(spec)
        assert spec.loader is not None
        sys.modules[name] = module
        spec.loader.exec_module(module)
        module._audio_pipeline = sys.modules.get("audio_pipeline")
    finally:
        sys.path.remove(str(SERVICE_DIR))
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


def load_audio_pipeline():
    """``audio_pipeline`` on its own, with the fake soundfile/librosa; returns ``(module, librosa)``."""
    name = f"audio_pipeline_under_test_{next(_counter)}"
    spec = spec_from_file_location(name, SERVICE_DIR / "audio_pipeline.py")
    module = module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module  # dataclasses resolves string annotations through sys.modules
    spec.loader.exec_module(module)
    module.sf = fake_soundfile()
    module.librosa = FakeLibrosa()
    return module, module.librosa


# --- fake ffmpeg -------------------------------------------------------------------

FAKE_FFMPEG = f"""#!{sys.executable}
import os, sys, time

args = sys.argv[1:]
root = os.environ["FAKE_FFMPEG_DIR"]
mode = open(root + "/mode").read().strip()
if mode == "hang":
    # Published before the slow imports below (numpy alone takes longer than a short
    # test timeout on a loaded machine), and as temp file + rename so that a poller
    # never sees `pid` created but still empty.
    with open(root + "/pid.tmp", "w") as fh:
        fh.write(str(os.getpid()))
    os.replace(root + "/pid.tmp", root + "/pid")

import json, struct, wave
import numpy as np

src = args[args.index("-i") + 1]
limit = float(args[args.index("-t") + 1]) if "-t" in args else None
record = {{"args": args, "src": src}}

def log():
    open(root + "/calls.jsonl", "a").write(json.dumps(record) + "\\n")

if mode == "fail":
    sys.stderr.write("boom: cannot decode")
    log()
    sys.exit(1)
if mode == "hang":
    log()
    time.sleep(60)

data = open(src, "rb").read()
if data.startswith(b"FAKEMP3"):
    seconds = struct.unpack("<d", data[15:23])[0]
    samples = np.zeros(int(round(seconds * 16000)), dtype="<f4")
else:
    with wave.open(src, "rb") as fh:
        channels = fh.getnchannels()
        samples = np.frombuffer(fh.readframes(fh.getnframes()), dtype="<i2").astype("<f4") / 32768.0
        if channels > 1:
            samples = samples.reshape(-1, channels).mean(axis=1).astype("<f4")
if limit is not None:
    samples = samples[: int(round(limit * 16000))]
record["wrote_seconds"] = len(samples) / 16000.0
log()
sys.stdout.buffer.write(samples.astype("<f4").tobytes())
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
        # raising=False: the pre-fix app has no _FFMPEG (it never shelled out)
        monkeypatch.setattr(module, "_FFMPEG", str(self.path), raising=False)

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

    def hung_pid(self) -> int | None:
        """The hung fake's pid; None until it has published it (a missing or empty file is "not yet")."""
        try:
            text = (self.dir / "pid").read_text().strip()
        except OSError:
            return None
        return int(text) if text.isdigit() else None


def process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    try:  # a zombie still answers signal 0
        return Path(f"/proc/{pid}/stat").read_text().split()[2] != "Z"
    except OSError:
        return False


# --- fake qwen_asr -----------------------------------------------------------------

def hear_words(samples: np.ndarray) -> list[str]:
    """The words in *samples*: every constant-amplitude run of at least 25 ms."""
    levels = np.rint(np.abs(samples) * 32768.0 / WORD_AMPLITUDE).astype(np.int64)
    if levels.size == 0:
        return []
    edges = np.flatnonzero(np.diff(levels)) + 1
    starts = np.concatenate(([0], edges))
    ends = np.concatenate((edges, [levels.size]))
    return [
        f"w{levels[a]}" for a, b in zip(starts, ends)
        if levels[a] > 0 and b - a >= RATE // 40
    ]


class FakeQwen:
    """``Qwen3ASRModel`` as the service uses it.

    Like the real one it cuts its input at ``LIBRARY_PIECE_S``, generates at most
    ``max_new_tokens`` tokens per piece (the ``language X<asr_text>`` prefix counts
    unless the language was forced) and joins the pieces with ``""``.
    """

    def __init__(self, *, max_new_tokens=512, language="German", piece_languages=None, error=None,
                 gate: threading.Event | None = None):
        self.max_new_tokens = max_new_tokens
        self.language = language
        self.piece_languages = piece_languages  # optional list, one per item of a call
        self.error = error
        self.gate = gate  # transcribe waits for it, to hold a request in flight
        self.entered = threading.Event()
        self.calls: list[dict] = []
        self.processor = SimpleNamespace(
            tokenizer=SimpleNamespace(encode=lambda text, add_special_tokens=True: text.split())
        )

    @staticmethod
    def _samples(item) -> np.ndarray:
        if isinstance(item, str):
            return _read_wav(item)[0]
        return np.asarray(item[0], dtype=np.float32)

    def transcribe(self, audio, context="", language=None, return_time_stamps=False):
        items = audio if isinstance(audio, list) else [audio]
        self.calls.append({
            "items": len(items),
            "language": language,
            "as_paths": [isinstance(item, str) for item in items],
            "lengths_s": [len(self._samples(item)) / RATE for item in items],
            "rates": [item[1] if isinstance(item, tuple) else None for item in items],
        })
        self.entered.set()
        if self.gate is not None:
            self.gate.wait(WAIT)
        if self.error is not None:
            raise self.error
        results = []
        for index, item in enumerate(items):
            samples = self._samples(item)
            budget = self.max_new_tokens - (0 if language else 3)
            texts = []
            piece = LIBRARY_PIECE_S * RATE
            for start in range(0, max(len(samples), 1), piece):
                words = hear_words(samples[start:start + piece])
                texts.append(" ".join(words[:budget]))
            text = "".join(texts)
            if self.piece_languages is not None:
                detected = self.piece_languages[index % len(self.piece_languages)]
            else:
                detected = self.language
            results.append(SimpleNamespace(
                language=language or (detected if text else ""), text=text, time_stamps=None,
            ))
        return results


class FakeQwenFactory:
    """Stands in for ``qwen_asr.Qwen3ASRModel``; builds a ``FakeQwen`` from the loader's kwargs."""

    def __init__(self, **model_options):
        self.model_options = model_options
        self.loads: list[dict] = []
        self.models: list[FakeQwen] = []
        self.block: threading.Event | None = None  # from_pretrained waits for it
        self.fail: Exception | None = None
        self.entered = threading.Event()

    def from_pretrained(self, name, **kwargs):
        self.loads.append({"name": name, **kwargs})
        self.entered.set()
        if self.block is not None:
            self.block.wait(WAIT)
        if self.fail is not None:
            raise self.fail
        model = FakeQwen(max_new_tokens=kwargs.get("max_new_tokens", 512), **self.model_options)
        self.models.append(model)
        return model

    @property
    def model(self) -> FakeQwen:
        return self.models[-1]


def install_fake_qwen_asr(monkeypatch, **model_options) -> FakeQwenFactory:
    """Register ``qwen_asr`` so the app's own loader builds a ``FakeQwen``."""
    factory = FakeQwenFactory(**model_options)
    module = types.ModuleType("qwen_asr")
    module.Qwen3ASRModel = factory
    monkeypatch.setitem(sys.modules, "qwen_asr", module)
    return factory


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
