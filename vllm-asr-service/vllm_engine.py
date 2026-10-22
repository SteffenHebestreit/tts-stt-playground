"""The ``vllm serve`` subprocess: start it, wait for it, put it to sleep, wake it, stop it.

Blocking calls, made from the model slot's worker threads (a start takes 30-90 s).
Nothing here imports torch: the adapter process never touches the GPU, only the
engine does (app.py hides the GPU from the adapter itself).

Why each part exists:

* **A process group of its own.** ``vllm serve`` runs its engine core in a child
  process, and that child holds the GPU memory. Stopping only the API server process
  left the child, and its VRAM, behind; the service signals the whole group.
* **A failed start gives the GPU back.** Whatever happened (out of memory while
  sizing the KV cache, a crash, a start that never answers), the group is killed and
  reaped before the error is raised: granite-asr-service kept 3.3 GB on the NAS after
  a failed load until #93, and a subprocess that lingers would do the same.
* **The last lines of its log.** The engine's output goes to the container log as it
  is, and its tail is kept so a failed start can say why (``out of memory`` makes the
  /ready category ``out_of_memory``) and /status can report the KV-cache size vLLM
  chose.
* **Sleep instead of exit.** ``POST /sleep?level=1`` (vLLM's sleep mode, which needs
  ``--enable-sleep-mode`` and ``VLLM_SERVER_DEV_MODE=1``) moves the weights to host RAM
  and drops the KV cache: measured on an RTX 4080 it frees all but 0.4-0.9 GB of the
  engine's VRAM in 1-4 s, and ``POST /wake_up`` brings it back in under a second,
  against 30-70 s for a fresh start. Level 2 (weights discarded) is not used: the
  engine answered garbage after waking from it, with or without ``reload_weights``.
"""

from __future__ import annotations

import collections
import logging
import os
import re
import signal
import subprocess
import threading
import time
from typing import Callable, Mapping, Optional, Sequence

import httpx

logger = logging.getLogger(__name__)

STATE_STOPPED = "stopped"
STATE_STARTING = "starting"
STATE_RUNNING = "running"
STATE_SLEEPING = "sleeping"
STATE_FAILED = "failed"

_LOG_TAIL_LINES = 200
# What the engine prints when the GPU memory it was given is not enough. vLLM sizes
# its KV cache at start-up, so these end a start, not a request.
_OOM_MARKERS = (
    "out of memory", "outofmemoryerror", "larger than the available kv cache memory",
    "no available memory for the cache blocks", "is less than desired gpu memory utilization",
)
_KV_TOKENS = re.compile(r"GPU KV cache size: ([\d,]+) tokens")


class EngineError(RuntimeError):
    """The engine could not be started, woken or asked; ``str()`` is for the log."""


class EngineOutOfMemory(EngineError):
    """The engine did not fit in the GPU memory it was given ("out of memory" in the text, on purpose)."""


def signal_process_group(process: subprocess.Popen, sig: int) -> None:
    """Send *sig* to the engine's whole process group (it was started as a session leader)."""
    if hasattr(os, "killpg"):
        os.killpg(process.pid, sig)
    elif sig == getattr(signal, "SIGKILL", None):  # pragma: no cover - Windows has no groups here
        process.kill()
    else:  # pragma: no cover
        process.terminate()


def _looks_out_of_memory(lines: Sequence[str]) -> bool:
    text = "\n".join(lines).lower()
    return any(marker in text for marker in _OOM_MARKERS)


class Engine:
    """One ``vllm serve`` process and its state.

    *popen*, *signal_group* and *http* are injectable so the tests can stand in for
    the process, for signalling its group and for its HTTP API (an ``httpx.Client``
    factory taking ``base_url`` and ``timeout``).
    """

    def __init__(
        self,
        argv: Sequence[str],
        base_url: str,
        *,
        env: Optional[Mapping[str, str]] = None,
        start_timeout_s: float = 900.0,
        stop_timeout_s: float = 30.0,
        poll_s: float = 1.0,
        popen: Callable[..., subprocess.Popen] = subprocess.Popen,
        signal_group: Callable[[subprocess.Popen, int], None] = signal_process_group,
        http: Callable[..., httpx.Client] = httpx.Client,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.argv = list(argv)
        self.base_url = base_url.rstrip("/")
        self.env = dict(env) if env is not None else None
        self.start_timeout_s = start_timeout_s
        self.stop_timeout_s = stop_timeout_s
        self.poll_s = poll_s
        self._popen = popen
        self._signal = signal_group
        self._http = http
        self._clock = clock
        self._sleep = sleep
        self.process: Optional[subprocess.Popen] = None
        self.state = STATE_STOPPED
        self.log_tail: collections.deque = collections.deque(maxlen=_LOG_TAIL_LINES)
        self.start_seconds: Optional[float] = None
        self.wake_seconds: Optional[float] = None
        self.sleep_seconds: Optional[float] = None
        self.kv_cache_tokens: Optional[int] = None
        self._reader: Optional[threading.Thread] = None

    # --- status ------------------------------------------------------------------------

    @property
    def pid(self) -> Optional[int]:
        return self.process.pid if self.process is not None else None

    def alive(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def snapshot(self) -> dict:
        return {
            "state": self.state if (self.alive() or self.state in (STATE_STOPPED, STATE_FAILED)) else STATE_FAILED,
            "pid": self.pid,
            "start_seconds": self.start_seconds,
            "wake_seconds": self.wake_seconds,
            "sleep_seconds": self.sleep_seconds,
            "kv_cache_tokens": self.kv_cache_tokens,
        }

    # --- the log -----------------------------------------------------------------------

    def _pump(self, stream) -> None:
        """Copy the engine's output to the container log, keeping the tail."""
        for raw in iter(stream.readline, ""):
            line = raw.rstrip("\n")
            self.log_tail.append(line)
            match = _KV_TOKENS.search(line)
            if match:
                self.kv_cache_tokens = int(match.group(1).replace(",", ""))
            print(line, flush=True)
        stream.close()

    # --- start / stop --------------------------------------------------------------------

    def start(self) -> float:
        """Launch the engine and wait until it answers /health; the seconds it took.

        Raises EngineError (EngineOutOfMemory when the log says so) after killing what
        was started, so nothing keeps GPU memory after a failed start.
        """
        if self.alive():
            raise EngineError("the engine is already running")
        started = self._clock()
        self.state = STATE_STARTING
        self.log_tail.clear()
        self.kv_cache_tokens = None
        logger.info("Starting the engine: %s", " ".join(self.argv))
        try:
            self.process = self._popen(
                self.argv, env=self.env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL, text=True, bufsize=1, start_new_session=True)
        except OSError as exc:
            self.state = STATE_FAILED
            raise EngineError(f"the engine could not be launched: {exc}") from exc
        if self.process.stdout is not None:
            self._reader = threading.Thread(target=self._pump, args=(self.process.stdout,),
                                            name="vllm-log", daemon=True)
            self._reader.start()
        try:
            self._wait_healthy(started)
        except BaseException:
            self.stop()
            self.state = STATE_FAILED
            raise
        self.state = STATE_RUNNING
        self.start_seconds = round(self._clock() - started, 1)
        return self.start_seconds

    def _wait_healthy(self, started: float) -> None:
        deadline = started + self.start_timeout_s
        with self._http(base_url=self.base_url, timeout=5.0) as client:
            while True:
                code = self.process.poll()
                if code is not None:
                    self._drain_log()
                    tail = list(self.log_tail)[-30:]
                    last = "\n".join(tail)
                    if _looks_out_of_memory(tail):
                        raise EngineOutOfMemory(
                            f"out of memory: the engine does not fit in the GPU memory it was given "
                            f"(VLLM_ASR_GPU_MEMORY_UTILIZATION; exit code {code}); last lines:\n{last}")
                    raise EngineError(f"the engine exited with code {code} while starting; last lines:\n{last}")
                try:
                    if client.get("/health").status_code == 200:
                        return
                except httpx.HTTPError:
                    pass
                if self._clock() >= deadline:
                    raise EngineError(
                        f"the engine did not answer /health within {self.start_timeout_s:g} s "
                        f"(VLLM_ASR_START_TIMEOUT_S)")
                self._sleep(self.poll_s)

    def _drain_log(self, timeout: float = 2.0) -> None:
        if self._reader is not None:
            self._reader.join(timeout)

    def _signal_group(self, sig: int) -> None:
        process = self.process
        if process is None:
            return
        try:
            self._signal(process, sig)
        except (ProcessLookupError, PermissionError, OSError):
            pass

    def stop(self) -> None:
        """End the engine and everything it started (TERM, then KILL); safe to call twice."""
        process = self.process
        if process is None:
            self.state = STATE_STOPPED
            return
        if process.poll() is None:
            self._signal_group(signal.SIGTERM)
            try:
                process.wait(timeout=self.stop_timeout_s)
            except subprocess.TimeoutExpired:
                logger.warning("The engine did not stop within %g s; killing it", self.stop_timeout_s)
        # The API server can be gone while its engine core still runs: KILL the whole
        # group either way, so no child keeps the GPU memory.
        self._signal_group(getattr(signal, "SIGKILL", signal.SIGTERM))
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:  # pragma: no cover - a process KILL cannot stop
            logger.error("The engine (pid %s) survived SIGKILL", process.pid)
        self._drain_log()
        self.process = None
        self.state = STATE_STOPPED
        logger.info("The engine stopped")

    # --- sleep / wake ------------------------------------------------------------------------

    def _control(self, method: str, path: str, *, timeout: float = 120.0, **kwargs) -> httpx.Response:
        if not self.alive():
            raise EngineError("the engine is not running")
        with self._http(base_url=self.base_url, timeout=timeout) as client:
            response = client.request(method, path, **kwargs)
        if response.status_code != 200:
            raise EngineError(f"{method} {path} answered {response.status_code}: {response.text[:200]}")
        return response

    def is_sleeping(self) -> bool:
        payload = self._control("GET", "/is_sleeping", timeout=10.0).json()
        return bool(payload.get("is_sleeping"))

    def sleep_engine(self, level: int = 1) -> float:
        """Offload the weights to host RAM and drop the KV cache; the seconds it took."""
        started = self._clock()
        self._control("POST", "/sleep", params={"level": str(int(level))})
        if not self.is_sleeping():
            raise EngineError("the engine did not report sleeping after /sleep")
        self.state = STATE_SLEEPING
        self.sleep_seconds = round(self._clock() - started, 2)
        return self.sleep_seconds

    def wake(self) -> float:
        """Bring a sleeping engine back; the seconds it took."""
        started = self._clock()
        self._control("POST", "/wake_up")
        if self.is_sleeping():
            raise EngineError("the engine still reports sleeping after /wake_up")
        self.state = STATE_RUNNING
        self.wake_seconds = round(self._clock() - started, 2)
        return self.wake_seconds
