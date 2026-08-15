"""Reference-counted model residency with an idle TTL.

Why this exists: several GPU services share one card. Holding every model
resident forever means peak VRAM is ``sum(models)``; with an idle TTL it becomes
``max(models) + one CUDA context per process``. On a 12 GB card that is the
difference between running three services and running four.

Two details are load-bearing and easy to get wrong:

1. **Reference counting, not a last-request timestamp.** Inference runs in worker
   threads (``asyncio.to_thread``), and cancelling the awaiting coroutine does
   not stop the thread. A timestamp-based reaper would free weights out from
   under a decode that is still running. Callers hold a reference for the whole
   duration of the call instead.

2. **``empty_cache()`` is required on unload.** ``del`` plus ``gc.collect()``
   returns memory only to PyTorch's caching allocator — ``nvidia-smi`` shows no
   change and other processes still cannot allocate. Only ``empty_cache()``
   releases it to the driver, which is the entire point here. (This is the
   opposite of the rule on the *request* path, where calling ``empty_cache()``
   per response destroys the allocator for no benefit.)

The CUDA context itself is never released — that is ~0.5 GB per process which
only exiting reclaims. Budget for it.

TTL contract, matching the convention used by speaches and Ollama:
    ``> 0``  seconds idle before unloading
    ``0``    unload as soon as the last caller releases
    ``-1``   never unload
"""

from __future__ import annotations

import asyncio
import gc
import logging
import threading
from contextlib import asynccontextmanager, contextmanager
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class ModelSlot:
    """Holds at most one loaded model, released after ``ttl`` seconds idle."""

    def __init__(
        self,
        loader: Callable[[], object],
        ttl_seconds: float = 300.0,
        name: str = "model",
        on_unload: Optional[Callable[[object], None]] = None,
    ):
        self._loader = loader
        self._on_unload = on_unload
        self._ttl = ttl_seconds
        self._name = name

        # RLock, not an asyncio lock: release() is called from executor threads.
        self._lock = threading.RLock()
        self._model: Optional[object] = None
        self._refs = 0
        self._timer: Optional[threading.Timer] = None

    # --- state ---------------------------------------------------------------

    # Both read WITHOUT the lock, deliberately.
    #
    # `_acquire` holds `self._lock` for the entire duration of `self._loader()`,
    # which for a NeMo checkpoint is tens of seconds to minutes. These two are
    # read by /health, which Docker polls with `timeout: 10s, retries: 3` — so
    # taking the lock here means three consecutive probe timeouts during any
    # reload and a container marked unhealthy, quite possibly restarted, in the
    # middle of loading its model. Idle unloading is what makes a reload happen
    # outside `start_period` at all, so this is not hypothetical.
    #
    # A single attribute read is atomic under the GIL. The answer can be one
    # instant stale, which is the right trade for a status field and no trade at
    # all for the safety property: nothing decides whether to free memory from
    # these — `try_unload` and `_expire` re-check `_refs` under the lock.

    @property
    def resident(self) -> bool:
        return self._model is not None

    @property
    def refs(self) -> int:
        return self._refs

    # --- use -----------------------------------------------------------------

    @contextmanager
    def acquire(self):
        """Yield the model, loading it if needed, pinned for the whole block."""
        model = self._acquire()
        try:
            yield model
        finally:
            self._release()

    @asynccontextmanager
    async def acquire_async(self):
        """Async form, for request handlers.

        Two reasons this is not just ``acquire()``:

        - Loading can take seconds to minutes, so it runs in a thread rather than
          blocking the event loop (which would stall every other request and the
          health endpoint).
        - The reference must be held across the ``await`` of the inference call.
          Acquiring and releasing before the await would let the idle timer fire
          and free the weights while a worker thread is still using them.
        """
        model = await asyncio.to_thread(self._acquire)
        try:
            yield model
        finally:
            self._release()

    def _acquire(self) -> object:
        with self._lock:
            self._cancel_timer()
            if self._model is None:
                logger.info("Loading %s...", self._name)
                self._model = self._loader()
                logger.info("%s loaded", self._name)
            self._refs += 1
            return self._model

    def _release(self) -> None:
        with self._lock:
            self._refs = max(0, self._refs - 1)
            if self._refs > 0:
                return
            if self._ttl < 0:
                return
            if self._ttl == 0:
                self._unload_locked()
                return
            self._cancel_timer()
            self._timer = threading.Timer(self._ttl, self._expire)
            # Daemon so a pending timer cannot keep the process alive on exit.
            self._timer.daemon = True
            self._timer.start()

    async def acquire_ref(self) -> object:
        """Take a reference that the caller MUST later hand back via release_ref.

        For streaming responses only. A StreamingResponse's generator runs after
        the handler has returned, so no ``with`` block can span the lifetime of
        the stream — the reference has to outlive the function that took it.
        Release it in the generator's ``finally``, or the model is pinned forever.
        """
        return await asyncio.to_thread(self._acquire)

    def release_ref(self) -> None:
        """Hand back a reference taken with ``acquire_ref``."""
        self._release()

    # --- unload --------------------------------------------------------------

    def _expire(self) -> None:
        with self._lock:
            if self._refs == 0:
                logger.info("%s idle for %.0fs, unloading", self._name, self._ttl)
                self._unload_locked()

    def unload(self) -> bool:
        """Unload now. Returns False and does nothing if the model is in use."""
        return self.try_unload()["unloaded"]

    def try_unload(self) -> dict:
        """Unload now, reporting *why* if it did not happen.

        ``unload()`` returns False both for "in use" and for "already gone",
        which an HTTP caller has to tell apart: the first is a 409 worth
        retrying, the second is a success. Decided under the lock, so the answer
        cannot be stale by the time it is returned.

        Returns ``{"unloaded": bool, "reason": "ok"|"busy"|"not_resident",
        "refs": int}``.
        """
        with self._lock:
            if self._refs > 0:
                logger.info("%s still in use (%d refs), not unloading", self._name, self._refs)
                return {"unloaded": False, "reason": "busy", "refs": self._refs}
            if self._model is None:
                return {"unloaded": False, "reason": "not_resident", "refs": 0}
            self._cancel_timer()
            return {"unloaded": self._unload_locked(), "reason": "ok", "refs": 0}

    def _unload_locked(self) -> bool:
        if self._model is None:
            return False
        model, self._model = self._model, None
        if self._on_unload is not None:
            try:
                self._on_unload(model)
            except Exception as e:
                logger.warning("%s unload hook failed: %s", self._name, e)
        del model
        gc.collect()
        _release_gpu_cache()
        logger.info("%s unloaded", self._name)
        return True

    def _cancel_timer(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None


def _release_gpu_cache() -> None:
    """Return cached allocator blocks to the driver, if there is a GPU."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            # ipc_collect additionally reclaims blocks shared with dead peers.
            torch.cuda.ipc_collect()
    except Exception as e:  # pragma: no cover - defensive
        logger.debug("Could not release GPU cache: %s", e)


def ttl_from_env(getenv: Callable[[str, str], str], *names: str, default: float = 300.0) -> float:
    """Read the first present TTL env var. Accepts seconds, or -1 / 0 sentinels."""
    for name in names:
        raw = (getenv(name, "") or "").strip()
        if not raw:
            continue
        try:
            return float(raw)
        except ValueError:
            logger.warning("Ignoring non-numeric %s=%r", name, raw)
    return default
