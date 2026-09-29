"""Shared harness for the stt-service tests (contains no tests of its own).

`app.py` imports torch and faster-whisper, which the unit-test environment does
not install, and reads most of its tuning from the environment at import time. So
each test module loads its own copy of the app under stubs with exactly the
environment it needs, and every knob the app reads is cleared first: an earlier
test module leaves values behind in ``os.environ`` (the WebSocket tests set a
1-second buffer cap), and inheriting them silently changes what a later test
measures.

Set ``STT_TEST_APP_DIR`` to a directory holding another copy of the service
(``git show <rev>:stt-service/app.py`` and its siblings) to run the same tests
against it, e.g. to confirm that a regression test fails on the old code.
"""

from __future__ import annotations

import os
import sys
import time
import types
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np

STT_DIR = Path(os.getenv("STT_TEST_APP_DIR") or Path(__file__).resolve().parents[1] / "stt-service")

# Everything app.py reads from the environment.
_ENV_KEYS = (
    "WS_MIN_NEW_AUDIO_S", "WS_WINDOW_S", "WS_MAX_BUFFER_S", "WS_MAX_SESSIONS",
    "WS_NO_SPEECH_THRESHOLD", "WS_MIN_AVG_LOGPROB", "WS_MAX_INTERIM_ERRORS",
    "WS_IDLE_TIMEOUT_S", "WS_LANGUAGE_LOCK_PROB", "USE_CUDA", "FORCE_ACCELERATION",
    "WHISPER_MODEL_SIZE", "WHISPER_COMPUTE_TYPE", "WHISPER_OOM_FALLBACK",
    "WHISPER_NUM_WORKERS", "WHISPER_MODEL_DIR", "STT_MODEL_TTL", "MODEL_TTL",
    "STT_DEFAULT_LANGUAGE", "STT_BATCH_WORKERS", "STT_BATCHED_INFERENCE",
    "STT_BATCH_SIZE", "MAX_UPLOAD_MB", "MAX_AUDIO_SECONDS", "HF_HUB_OFFLINE",
    "ALLOWED_ORIGINS", "ALLOW_CREDENTIALS",
)

_BASE_ENV = {"USE_CUDA": "false", "WHISPER_MODEL_SIZE": "tiny"}


class Word:
    """faster-whisper Word stand-in."""

    def __init__(self, start, end, word, probability=0.9):
        self.start, self.end, self.word, self.probability = start, end, word, probability


class Segment:
    """Minimal stand-in for a faster-whisper segment (no word timings by default)."""

    def __init__(self, text, no_speech_prob=0.0, avg_logprob=-0.2, start=0.0, end=1.0, words=None):
        self.text = text
        self.no_speech_prob = no_speech_prob
        self.avg_logprob = avg_logprob
        self.start = start
        self.end = end
        self.words = words


class Info:
    def __init__(self, language="en", language_probability=0.99, duration=1.0):
        self.language = language
        self.language_probability = language_probability
        self.duration = duration


class FakeWhisper:
    """Returns scripted hypotheses and records what it was asked.

    `default` is returned once `script` is exhausted: decodes are single-flight
    and skip-ahead by design, so a test can never assume one decode per audio
    frame. `fail_with` makes every transcribe() raise it (an exception instance)
    until cleared, for the failure-path tests.
    """

    instances: list["FakeWhisper"] = []

    def __init__(self, *args, **kwargs):
        self.init_args = args
        self.init_kwargs = kwargs
        self.script: list = []
        self.default: list = []
        self.calls: list[dict] = []
        self.audio_lengths: list[int] = []
        self.info = Info()
        self.fail_with = None
        FakeWhisper.instances.append(self)

    def transcribe(self, audio, **kwargs):
        self.calls.append(kwargs)
        try:
            self.audio_lengths.append(len(audio))
        except TypeError:  # a file path
            self.audio_lengths.append(-1)
        if self.fail_with is not None:
            raise self.fail_with
        segments = self.script.pop(0) if self.script else self.default
        return iter(segments), self.info


def _torch_stub():
    torch = types.ModuleType("torch")
    torch.__version__ = "0.0.0-stub"
    torch.cuda = types.SimpleNamespace(
        is_available=lambda: False,
        device_count=lambda: 0,
        get_device_name=lambda i: "stub",
        get_device_capability=lambda i: (8, 9),
    )
    torch.backends = types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: False))
    return torch


class FakeBatchedPipeline:
    """BatchedInferencePipeline stand-in that records how it was used."""

    instances: list["FakeBatchedPipeline"] = []

    def __init__(self, model):
        self.model = model
        self.calls: list[dict] = []
        FakeBatchedPipeline.instances.append(self)

    def transcribe(self, audio, **kwargs):
        self.calls.append(kwargs)
        return iter([Segment("batched text", start=0.0, end=1.0)]), Info(duration=1.0)


def load_stt_app(env: dict | None = None, name: str = "stt_app_under_test", model_cls=None):
    """Import stt-service/app.py under stubs with exactly `env` (plus a CPU base)."""
    saved_env = {key: os.environ.get(key) for key in _ENV_KEYS}
    for key in _ENV_KEYS:
        os.environ.pop(key, None)
    os.environ.update({**_BASE_ENV, **(env or {})})

    fw = types.ModuleType("faster_whisper")
    fw.WhisperModel = model_cls or FakeWhisper
    fw.BatchedInferencePipeline = FakeBatchedPipeline
    saved_modules = {key: sys.modules.get(key) for key in ("torch", "faster_whisper")}
    sys.modules["torch"] = _torch_stub()
    sys.modules["faster_whisper"] = fw

    sys.path.insert(0, str(STT_DIR))
    try:
        spec = spec_from_file_location(name, STT_DIR / "app.py")
        module = module_from_spec(spec)
        assert spec.loader is not None
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(STT_DIR))
        # An older copy of the service imports torch inside functions, at call
        # time, so when running against one the stubs must stay installed.
        for key, value in ({} if os.getenv("STT_TEST_APP_DIR") else saved_modules).items():
            if value is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value
        for key, value in saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def wait_for_preload(app_module, timeout: float = 10.0) -> None:
    """Block until the startup preload finished (a no-op on builds without one)."""
    event = getattr(app_module, "_preload_done", None)
    if event is not None:
        assert event.wait(timeout), "startup preload never finished"


def reset_model(app_module, model=None):
    """Put a fresh scripted model in place, as a resident, healthy one."""
    model = model or FakeWhisper()
    app_module.whisper_model = model
    app_module.model_loaded = True
    app_module.startup_error = None
    app_module._live_sessions = 0
    return model


def wait_until(predicate, timeout: float = 5.0, interval: float = 0.01) -> bool:
    """Poll `predicate` until it holds; True if it did within `timeout`."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return bool(predicate())


def pcm(seconds: float, rate: int = 16000, amplitude: int = 1000) -> bytes:
    """`seconds` of quiet noise as PCM16 little-endian mono."""
    rng = np.random.default_rng(0)
    samples = (rng.standard_normal(int(seconds * rate)) * amplitude).astype(np.int16)
    return samples.tobytes()


def tick_pcm(start_s: float, seconds: float, rate: int = 16000) -> bytes:
    """PCM whose sample values encode absolute time (one step per 10 ms).

    Lets a fake model recover where in the session a window starts from the audio
    alone, so a test can play a "real" model against a sliding window.
    """
    first = int(round(start_s * rate))
    idx = np.arange(first, first + int(round(seconds * rate)))
    return (idx // (rate // 100)).astype(np.int16).tobytes()


class TimelineWhisper(FakeWhisper):
    """A model that "hears" a ground-truth transcript laid out on a timeline.

    Audio must come from ``tick_pcm``. Given a window it returns the words that lie
    completely inside it, with times relative to the window, and garbles the word
    still being spoken at the window's end (as a real model does with a cut-off
    word) so that hypothesis only settles once more audio has arrived.
    """

    def __init__(self, truth, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.truth = truth          # [(start_s, end_s, text)]
        self.windows: list[tuple[float, float]] = []

    def transcribe(self, audio, **kwargs):
        self.calls.append(kwargs)
        if self.fail_with is not None:
            raise self.fail_with
        rate = 16000
        if len(audio) == 0:
            return iter([]), self.info
        first_tick = int(round(float(audio[0]) * 32768.0))
        start = first_tick * 0.01
        end = start + len(audio) / rate
        self.windows.append((start, end))
        words = []
        for w_start, w_end, text in self.truth:
            if w_start < start - 1e-6:
                continue
            if w_end <= end - 0.3:
                words.append(Word(w_start - start, w_end - start, " " + text))
            elif w_start < end:
                # A different mangling every call: a cut-off word is never heard
                # the same wrong way twice, or "agreement" would confirm it.
                words.append(Word(w_start - start, end - start, " " + text[:-1] + str(int(end * 10) % 7)))
        if not words:
            return iter([]), self.info
        segment = Segment(
            "".join(w.word for w in words), start=words[0].start, end=words[-1].end, words=words
        )
        return iter([segment]), self.info


class LiveSession:
    """A `/ws/transcribe` client with a reader thread, so a test never blocks on a
    message that the single-flight decoder happens not to send.

    The server sends when it sends: decodes coalesce and skip ahead, so how many
    partials a stream of frames produces is not something a test can count on.
    Sending from the test thread and collecting on another lets a test wait for
    "a partial that says X" instead.
    """

    def __init__(self, client, config: dict | None = None, path: str = "/ws/transcribe"):
        import threading

        self.messages: list[dict] = []
        self.closed: Exception | None = None
        self._cond = threading.Condition()
        self._cm = client.websocket_connect(path)
        self.ws = self._cm.__enter__()
        if config is not None:
            self.ws.send_json(config)
        self._reader = threading.Thread(target=self._read, daemon=True)
        self._reader.start()

    def _read(self):
        while True:
            try:
                message = self.ws.receive_json()
            except Exception as exc:  # WebSocketDisconnect once the server closes
                with self._cond:
                    self.closed = exc
                    self._cond.notify_all()
                return
            with self._cond:
                self.messages.append(message)
                self._cond.notify_all()

    def send_json(self, payload: dict):
        self.ws.send_json(payload)

    def send_audio(self, seconds: float, start_s: float | None = None):
        """Send `seconds` of audio; timeline-encoded when `start_s` is given."""
        self.ws.send_bytes(tick_pcm(start_s, seconds) if start_s is not None else pcm(seconds))

    def stop(self):
        self.ws.send_json({"event": "stop"})

    def wait_for(self, predicate, timeout: float = 5.0):
        """The first message satisfying `predicate` (already received or not)."""
        deadline = time.monotonic() + timeout
        seen = 0
        with self._cond:
            while True:
                for message in self.messages[seen:]:
                    if predicate(message):
                        return message
                seen = len(self.messages)
                remaining = deadline - time.monotonic()
                if remaining <= 0 or self.closed is not None:
                    raise AssertionError(
                        f"no matching message (closed={self.closed!r}); received: {self.messages}")
                self._cond.wait(min(remaining, 0.05))

    def wait_type(self, msg_type: str, timeout: float = 5.0):
        return self.wait_for(lambda m: m.get("type") == msg_type, timeout)

    def wait_closed(self, timeout: float = 5.0):
        deadline = time.monotonic() + timeout
        with self._cond:
            while self.closed is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise AssertionError(f"socket still open; received: {self.messages}")
                self._cond.wait(min(remaining, 0.05))
        return self.closed

    def of_type(self, msg_type: str) -> list[dict]:
        with self._cond:
            return [m for m in self.messages if m.get("type") == msg_type]

    def close(self):
        try:
            self._cm.__exit__(None, None, None)
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
