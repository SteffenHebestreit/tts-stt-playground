"""vad_pool.VadPool: the speech gate in worker processes, and its fall-backs.

One real pool is started (``spawn``, two workers): it proves the worker function and
its module can be found by a fresh interpreter, which is how the image runs it. The
workers have no torch here, so their Silero falls back to the energy gate, and that
is reported, the same as in the adapter process.
"""

from __future__ import annotations

import asyncio
import importlib
import sys
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures.process import BrokenProcessPool

import numpy as np
import pytest

from vllm_asr_loader import LOCAL_MODULES, SERVICE_DIR, load_module

RATE = 16000


def speech() -> np.ndarray:
    audio = np.zeros(4 * RATE, dtype=np.float32)
    audio[RATE:2 * RATE] = 0.2
    audio[int(2.5 * RATE):3 * RATE] = 0.3
    return audio


def test_energy_and_off_are_gated_in_process_without_any_worker():
    pool_module = load_module("vad_pool")
    for mode in ("energy", "off"):
        pool = pool_module.VadPool(mode, workers=4, executor_factory=lambda: pytest.fail("no workers"))
        assert pool.in_process and pool.describe() == mode
        assert asyncio.run(pool.regions(speech()))


def test_workers_zero_gates_in_this_process():
    pool_module = load_module("vad_pool")
    pool = pool_module.VadPool("silero", workers=0, executor_factory=lambda: pytest.fail("no workers"))
    assert pool.in_process
    found = asyncio.run(pool.regions(speech()))
    assert found and all(b > a for a, b in found)


class _Broken(ThreadPoolExecutor):
    def submit(self, fn, *args, **kwargs):
        future: Future = Future()
        future.set_exception(BrokenProcessPool("a worker died"))
        return future


def test_a_broken_pool_falls_back_to_the_adapter_process_for_good(caplog):
    pool_module = load_module("vad_pool")
    pool = pool_module.VadPool("silero", workers=2, executor_factory=lambda: _Broken(max_workers=1))
    first = asyncio.run(pool.regions(speech()))
    second = asyncio.run(pool.regions(speech()))
    assert first == second and first
    assert pool.in_process
    assert sum("VAD worker pool broke" in r.getMessage() for r in caplog.records) == 1


@pytest.fixture
def real_vad_pool(monkeypatch):
    """vad_pool under its own name with the service directory on sys.path, as in the image."""
    monkeypatch.syspath_prepend(str(SERVICE_DIR))
    saved = {name: sys.modules.pop(name, None) for name in LOCAL_MODULES}
    try:
        yield importlib.import_module("vad_pool")
    finally:
        for name in LOCAL_MODULES:
            sys.modules.pop(name, None)
        for name, module in saved.items():
            if module is not None:
                sys.modules[name] = module


def test_spawned_workers_gate_like_the_adapter_process_would(real_vad_pool):
    pool = real_vad_pool.VadPool("silero", workers=2)
    try:
        pool.warm()
        assert not pool.in_process
        found = asyncio.run(pool.regions(speech()))
        local = real_vad_pool.SpeechGate("energy").regions(speech())
        # No torch in the test environment: each worker's Silero fell back to the energy gate.
        assert pool.describe() == "energy (silero unavailable)"
        assert found == local
    finally:
        pool.shutdown()
