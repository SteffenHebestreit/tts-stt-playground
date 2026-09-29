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

API map (everything below the first block is additive; the first block is the
original surface and its behaviour is unchanged):

    ModelSlot(loader, ttl_seconds, name, on_unload, timer_factory=None)
    slot.resident / .refs                    lock-free status reads
    with slot.acquire() as model             blocking, for worker threads
    async with slot.acquire_async() as model request handlers
    await slot.acquire_ref() / slot.release_ref()   streaming, manual pairing
    slot.unload() / slot.try_unload()        deliberate unload, 409-aware

    Event-loop safe variants. ``_release`` / ``try_unload`` take the slot lock,
    and the lock is held for the whole of a model load (seconds to minutes), so
    calling them directly from a coroutine freezes the loop, /health included,
    until the load finishes:
    await slot.release_async()               pairs with acquire_ref()
    await slot.try_unload_async()             same dict as try_unload()
    await slot.unload_async()                 same bool as unload()

    Cancellation-safe acquisition. ``acquire_async`` / ``acquire_ref`` /
    ``acquire_lease`` shield the worker thread that takes the reference: if the
    awaiting task is cancelled (client disconnect) while a load is running, the
    thread still finishes and takes its reference, and a done-callback hands it
    straight back. Without that the reference was orphaned, the idle TTL never
    armed, and /unload answered 409 until the process was restarted.

    Leases, for streaming responses whose generator outlives the handler:
    lease = await slot.acquire_lease()       (or slot.lease() in a thread)
    lease.model                              the pinned model
    lease.release() / await lease.release_async() / lease.release_soon()
                                             idempotent: only the first counts
    lease.release_on_gc(obj)                 weakref.finalize(obj, ...) hook
    A lease also releases itself when it is garbage collected, so a generator
    that was never started (client gone before the first byte; its ``finally``
    never runs) cannot leak the reference. ``release_soon`` never blocks and is
    the one to use from done-callbacks and finalizers, which may run on the
    event loop.

    Readiness introspection (lock-free, safe to call from /health or /ready):
    slot.ever_loaded    True once any load has succeeded (sticky)
    slot.loading        a load is running right now
    slot.last_error     "Type: message" of the most recent failed load, else None
    slot.readiness()    {"ready", "reason", "detail", "resident", "ever_loaded",
                         "loading", "last_error"}

Intended /ready contract (kept separate from /health, which must stay 200 while
a model is merely idle or loading so Docker never restarts a service that is
doing its job):
    200  ever_loaded is True: the weights have been proven loadable, and a
         later TTL unload or one transient reload failure must not flip the
         service to unready (only a request can clear it, and none will be
         routed to an unready service);
    200  not loading and no last_error: never loaded yet but nothing is known to
         be wrong (lazy start, or TTL unloaded before any request);
    503  ``reason="loading"``: the FIRST load is still in progress;
    503  ``reason="load_failed"``: the last load failed and no load ever
         succeeded; ``detail`` carries ``last_error``.
"""

from __future__ import annotations

import asyncio
import gc
import logging
import threading
import weakref
from contextlib import asynccontextmanager, contextmanager
from typing import Callable, Optional

logger = logging.getLogger(__name__)

# Long CUDA / OOM messages would bloat a /ready body and Docker's health log.
_MAX_ERROR_CHARS = 500


def _describe_error(exc: BaseException) -> str:
    text = f"{type(exc).__name__}: {exc}"
    return text if len(text) <= _MAX_ERROR_CHARS else text[: _MAX_ERROR_CHARS - 3] + "..."


class ModelSlot:
    """Holds at most one loaded model, released after ``ttl`` seconds idle."""

    def __init__(
        self,
        loader: Callable[[], object],
        ttl_seconds: float = 300.0,
        name: str = "model",
        on_unload: Optional[Callable[[object], None]] = None,
        timer_factory: Optional[Callable[[float, Callable[[], None]], object]] = None,
    ):
        self._loader = loader
        self._on_unload = on_unload
        self._ttl = ttl_seconds
        self._name = name
        # Injectable so tests can fire the idle timer by hand instead of
        # sleeping against a tiny TTL. Must return an object with start(),
        # cancel() and a settable ``daemon``, like threading.Timer.
        self._timer_factory = timer_factory or threading.Timer

        # RLock, not an asyncio lock: release() is called from executor threads.
        self._lock = threading.RLock()
        self._model: Optional[object] = None
        self._refs = 0
        self._timer: Optional[object] = None
        # Bumped whenever the pending timer is cancelled or replaced. A timer
        # that already fired and is blocked on the lock in _expire() cannot be
        # cancelled; without this it would unload a model that was acquired and
        # released again in the meantime, up to one TTL early.
        self._timer_gen = 0

        self._loading = False
        self._ever_loaded = False
        self._last_error: Optional[str] = None

    # --- state ---------------------------------------------------------------

    # All of these are read WITHOUT the lock, deliberately.
    #
    # `_acquire` holds `self._lock` for the entire duration of `self._loader()`,
    # which for a NeMo checkpoint is tens of seconds to minutes. These are
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

    @property
    def ever_loaded(self) -> bool:
        """True once any load has succeeded. Never reset by a later unload."""
        return self._ever_loaded

    @property
    def loading(self) -> bool:
        """True while ``loader()`` is running."""
        return self._loading

    @property
    def last_error(self) -> Optional[str]:
        """``"Type: message"`` of the most recent failed load; cleared by a success."""
        return self._last_error

    def readiness(self) -> dict:
        """Snapshot for a /ready endpoint; see the contract in the module docstring.

        ``ready`` maps to HTTP 200 / 503. ``reason`` is ``"ok"``, ``"loading"``
        or ``"load_failed"``; ``detail`` is the load error for the latter.
        """
        # Read once each: the flags change from worker threads.
        ever_loaded, loading, error = self._ever_loaded, self._loading, self._last_error
        if ever_loaded:
            ready, reason, detail = True, "ok", None
        elif loading:
            ready, reason, detail = False, "loading", None
        elif error:
            ready, reason, detail = False, "load_failed", error
        else:
            ready, reason, detail = True, "ok", None
        return {
            "ready": ready,
            "reason": reason,
            "detail": detail,
            "resident": self.resident,
            "ever_loaded": ever_loaded,
            "loading": loading,
            "last_error": error,
        }

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

        Both ends are cancellation-safe: see ``_acquire_in_thread`` for the
        acquire, and ``release_async`` for the release.
        """
        model = await self._acquire_in_thread()
        try:
            yield model
        finally:
            await self.release_async()

    def _acquire(self) -> object:
        with self._lock:
            self._cancel_timer()
            if self._model is None:
                logger.info("Loading %s...", self._name)
                self._loading = True
                try:
                    model = self._loader()
                except BaseException as e:
                    # Order matters for the lock-free readers: publish the error
                    # before dropping `loading`, or /ready could observe "not
                    # loading, no error" and report a broken model as loadable.
                    self._last_error = _describe_error(e)
                    self._loading = False
                    raise
                self._model = model
                self._last_error = None
                self._ever_loaded = True
                self._loading = False
                logger.info("%s loaded", self._name)
            self._refs += 1
            return self._model

    async def _acquire_in_thread(self) -> object:
        """``_acquire`` on a worker thread, without orphaning the reference.

        ``asyncio.to_thread`` cannot help a cancelled caller: the thread keeps
        running, completes the acquire and bumps ``_refs``, but the coroutine
        that would have released it is gone. The reference then pins the model
        forever (idle TTL never arms, /unload answers 409). So the thread's
        future is shielded from the cancellation, and when the caller is cancelled
        a done-callback hands the reference back once the thread finishes.
        """
        fut = asyncio.get_running_loop().run_in_executor(None, self._acquire)
        try:
            return await asyncio.shield(fut)
        except asyncio.CancelledError:
            # Covers both "still running" and "finished but the caller was
            # cancelled before it could take the result".
            fut.add_done_callback(self._hand_back_abandoned)
            raise

    def _hand_back_abandoned(self, fut: "asyncio.Future") -> None:
        if fut.cancelled() or fut.exception() is not None:
            # No reference was taken (loader failed, or never ran).
            return
        logger.info("%s: caller went away during acquire, releasing its reference", self._name)
        self._release_detached()

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
            generation = self._timer_gen
            self._timer = self._timer_factory(self._ttl, lambda: self._expire(generation))
            # Daemon so a pending timer cannot keep the process alive on exit.
            self._timer.daemon = True
            self._timer.start()

    def _release_detached(self) -> None:
        """``_release`` on its own thread, for callers that must not block.

        Done-callbacks, finalizers and ``__del__`` can run on the event loop or
        inside the garbage collector; waiting there for a lock that a model load
        holds would freeze the loop, or worse. A dedicated thread rather than the
        default executor, so a saturated pool cannot delay it.
        """
        try:
            threading.Thread(
                target=self._release, name=f"{self._name}-release", daemon=True
            ).start()
        except RuntimeError:  # pragma: no cover - interpreter shutting down / no threads left
            self._release()

    async def release_async(self) -> None:
        """Event-loop-safe ``_release``; pairs with ``acquire_ref``.

        Shielded, so a task cancelled while waiting here cannot abandon the
        release: the thread still runs it. (Plain ``to_thread`` would drop a
        release that had not started yet.)
        """
        fut = asyncio.get_running_loop().run_in_executor(None, self._release)
        await asyncio.shield(fut)

    async def acquire_ref(self) -> object:
        """Take a reference that the caller MUST later hand back via release_ref.

        For streaming responses only. A StreamingResponse's generator runs after
        the handler has returned, so no ``with`` block can span the lifetime of
        the stream — the reference has to outlive the function that took it.
        Release it in the generator's ``finally``, or the model is pinned forever.

        Prefer ``acquire_lease``: ``release_ref`` is not idempotent, so a double
        release silently drops another caller's reference.
        """
        return await self._acquire_in_thread()

    def release_ref(self) -> None:
        """Hand back a reference taken with ``acquire_ref``."""
        self._release()

    async def acquire_lease(self) -> "ModelLease":
        """Take a reference wrapped in an idempotent, GC-safe ``ModelLease``."""
        return ModelLease(self, await self._acquire_in_thread())

    def lease(self) -> "ModelLease":
        """Blocking ``acquire_lease`` for worker threads."""
        return ModelLease(self, self._acquire())

    # --- unload --------------------------------------------------------------

    def _expire(self, generation: Optional[int] = None) -> None:
        with self._lock:
            if generation is not None and generation != self._timer_gen:
                return  # superseded: the model was used again after this armed
            if self._refs == 0:
                logger.info("%s idle for %.0fs, unloading", self._name, self._ttl)
                self._unload_locked()
            self._timer = None

    def unload(self) -> bool:
        """Unload now. Returns False and does nothing if the model is in use."""
        return self.try_unload()["unloaded"]

    def try_unload(self) -> dict:
        """Unload now, reporting *why* if it did not happen.

        ``unload()`` returns False both for "in use" and for "already gone",
        which an HTTP caller has to tell apart: the first is a 409 worth
        retrying, the second is a success. Decided under the lock, so the answer
        cannot be stale by the time it is returned.

        A load in progress counts as "busy": someone is waiting for the model, and
        answering at once beats blocking behind a load that can take minutes.

        Returns ``{"unloaded": bool, "reason": "ok"|"busy"|"not_resident",
        "refs": int}``.
        """
        # Refusal only. Anything that FREES memory is decided under the lock below.
        if self._loading:
            return {"unloaded": False, "reason": "busy", "refs": self._refs}
        with self._lock:
            if self._refs > 0:
                logger.info("%s still in use (%d refs), not unloading", self._name, self._refs)
                return {"unloaded": False, "reason": "busy", "refs": self._refs}
            if self._model is None:
                return {"unloaded": False, "reason": "not_resident", "refs": 0}
            self._cancel_timer()
            return {"unloaded": self._unload_locked(), "reason": "ok", "refs": 0}

    async def try_unload_async(self) -> dict:
        """Event-loop-safe ``try_unload``: the unload itself can take seconds."""
        return await asyncio.to_thread(self.try_unload)

    async def unload_async(self) -> bool:
        """Event-loop-safe ``unload``."""
        return (await self.try_unload_async())["unloaded"]

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
        self._timer_gen += 1
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None


class ModelLease:
    """One reference on a ``ModelSlot`` that can be released any number of times.

    Streaming responses are where a plain acquire/release pair fails: the
    generator's ``finally`` is the only release, and a generator that is never
    started (the client vanished before the first byte, or the response object
    was dropped) never runs its ``finally``. Every release path here is safe to
    call more than once — only the first one counts — so it can be attached to
    all of them: the generator's ``finally``, the handler's error path, a
    Starlette ``BackgroundTask``, ``add_done_callback`` and
    ``weakref.finalize``. As a last resort a lease that is garbage collected
    unreleased releases itself.

    Streaming pattern::

        lease = await slot.acquire_lease()
        try:
            first = await run_first_chunk(lease.model)
        except BaseException:
            await lease.release_async()   # nothing will consume the stream
            raise

        async def chunks():
            try:
                yield first
                ...
            finally:
                lease.release_soon()      # never blocks, safe in any context

        stream = chunks()
        lease.release_on_gc(stream)       # the unstarted-generator case
        return StreamingResponse(stream, background=BackgroundTask(lease.release_async))
    """

    def __init__(self, slot: ModelSlot, model: object):
        self._slot = slot
        self._model = model
        self._guard = threading.Lock()
        self._released = False
        self._finalizers: list = []

    @property
    def model(self) -> object:
        """The pinned model. Reading it after release is a use-after-unload bug."""
        if self._released:
            raise RuntimeError("model lease already released; the model may be unloaded")
        return self._model

    @property
    def released(self) -> bool:
        return self._released

    def _claim(self) -> bool:
        """True for exactly one caller, however many race for it."""
        with self._guard:
            if self._released:
                return False
            self._released = True
            finalizers, self._finalizers = self._finalizers, []
        for finalizer in finalizers:
            finalizer.detach()
        return True

    def release(self) -> bool:
        """Blocking release. Returns True only for the call that released."""
        if not self._claim():
            return False
        self._slot._release()
        return True

    async def release_async(self) -> bool:
        """Release without blocking the event loop (waits for the slot lock off-loop)."""
        if not self._claim():
            return False
        await self._slot.release_async()
        return True

    def release_soon(self) -> bool:
        """Release without ever blocking the caller; the slot is updated a moment later.

        For done-callbacks and finalizers, which can run on the event loop or in
        the garbage collector.
        """
        if not self._claim():
            return False
        self._slot._release_detached()
        return True

    def release_on_gc(self, obj: object) -> None:
        """Release when ``obj`` is garbage collected, if nobody released before.

        Attach to the streaming generator (or response): if it is dropped without
        ever running, this is the only thing that hands the reference back.
        ``obj`` must be weak-referenceable and must not be referenced BY this
        lease. Detached automatically on release.
        """
        finalizer = weakref.finalize(obj, self.release_soon)
        finalizer.atexit = False
        with self._guard:
            if not self._released:
                self._finalizers.append(finalizer)
                return
        finalizer.detach()

    def __enter__(self) -> "ModelLease":
        return self

    def __exit__(self, *exc) -> None:
        self.release()

    async def __aenter__(self) -> "ModelLease":
        return self

    async def __aexit__(self, *exc) -> None:
        await self.release_async()

    def __del__(self):
        # Safety net only: reaching here unreleased means every owner dropped the
        # lease, so nothing else can ever hand the reference back.
        try:
            if not self._released:
                logger.warning("model lease for %s dropped without release; releasing", self._slot._name)
                self.release_soon()
        except Exception:  # pragma: no cover - interpreter teardown
            pass


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
