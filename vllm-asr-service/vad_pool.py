"""The speech gate in worker processes, so a busy service does not queue on one CPU core.

Silero VAD walks a recording in 32 ms windows from a Python loop, about 60 ms of CPU
per 15 s of audio, and holds the GIL between windows: threads do not run it in
parallel. With 32 requests at once, one in-process gate capped the whole service at
~200x real time (four gates in threads: ~250x) while the engine alone did ~450x
(Cohere Transcribe, RTX 4080). Each worker process here has its own Silero model.

``VadPool.regions()`` is a coroutine; it falls back to the in-process gate when the
pool is off (``workers=0``, the tests), for the cheap ``energy`` and ``off`` modes,
and if the pool breaks (a worker died), logging why once. Workers are started with
``spawn``: the adapter has threads by then, and a forked copy of a threaded process
is not safe. They inherit the adapter's hidden GPU (CUDA_VISIBLE_DEVICES=""), so they
never open a CUDA context either.
"""

from __future__ import annotations

import asyncio
import logging
import multiprocessing
from concurrent.futures import Executor, ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from typing import Callable, Optional

import numpy as np

from transformers_asr import SpeechGate

logger = logging.getLogger(__name__)

# One gate per worker process, made by the initializer.
_worker_gate: Optional[SpeechGate] = None


def _init_worker(mode: str, threshold: float) -> None:
    global _worker_gate
    _worker_gate = SpeechGate(mode, threshold=threshold)


def _worker_regions(samples: np.ndarray) -> tuple[list[tuple[int, int]], str]:
    """Speech ranges in a worker, and what gated them there (``describe()``)."""
    return _worker_gate.regions(samples), _worker_gate.describe()


class VadPool:
    """``workers`` processes running the gate, or the gate in this process when that is 0."""

    def __init__(self, mode: str, *, threshold: float = 0.5, workers: int = 4,
                 executor_factory: Optional[Callable[[], Executor]] = None):
        self.mode = mode
        self.threshold = threshold
        self.workers = max(0, int(workers)) if mode == "silero" else 0
        self.local = SpeechGate(mode, threshold=threshold)
        self.described = self.local.describe()
        self._factory = executor_factory or self._process_pool
        self._executor: Optional[Executor] = None
        self._broken = False

    def _process_pool(self) -> Executor:
        return ProcessPoolExecutor(
            max_workers=self.workers, mp_context=multiprocessing.get_context("spawn"),
            initializer=_init_worker, initargs=(self.mode, self.threshold))

    @property
    def in_process(self) -> bool:
        return self.workers == 0 or self._broken

    def describe(self) -> str:
        return self.local.describe() if self.in_process else self.described

    def _ensure(self) -> Executor:
        if self._executor is None:
            self._executor = self._factory()
        return self._executor

    def warm(self) -> None:
        """Start the workers and load each one's model (blocking; the engine start pays for it)."""
        silence = np.zeros(16000, dtype=np.float32)
        if self.in_process:
            self.local.regions(silence)
            return
        try:
            executor = self._ensure()
            futures = [executor.submit(_worker_regions, silence) for _ in range(self.workers)]
            for future in futures:
                _, self.described = future.result()
        except BrokenProcessPool as e:
            self._give_up(e)
            self.local.regions(silence)

    def _give_up(self, error: BaseException) -> None:
        if not self._broken:
            logger.error("The VAD worker pool broke (%s); gating in the adapter process from now on", error)
        self._broken = True

    async def regions(self, samples: np.ndarray) -> list[tuple[int, int]]:
        """Speech ranges of *samples* (16 kHz mono float32), off the event loop."""
        if not self.in_process:
            try:
                found, self.described = await asyncio.get_running_loop().run_in_executor(
                    self._ensure(), _worker_regions, samples)
                return found
            except BrokenProcessPool as e:
                self._give_up(e)
        return await asyncio.to_thread(self.local.regions, samples)

    def shutdown(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._executor = None
