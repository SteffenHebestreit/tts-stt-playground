"""Idle-unload timing for the shared Whisper model.

`.env.example` documents `MODEL_TTL` as "the single knob that lets several
multi-GB models share one card", applied to any service that does not set its
own. Every other GPU service honours it through `model_lifecycle.ModelSlot`;
stt-service could not, because its load path is a fallback ladder that sets six
module globals (device, compute_type, model_size_loaded, ...) rather than
returning an object a slot can hold. So it kept its own reference counter and
never grew the timer half — the largest always-on consumer in the stack was the
one that never gave its VRAM back.

This is that timer half, and nothing else. Reference counting stays in `app.py`
where the globals live; this module only answers "when should the unload run?".

Locking contract, because getting it wrong deadlocks the service: `_fire()`
calls `on_expire` with **no** lock of this module held. `app.py` acquires its
reference lock and then calls in here (`cancel()` from `acquire_model`), so the
reverse order must never happen or the two locks invert.

TTL contract, matching `model_lifecycle.ttl_from_env`:
    ``> 0``  seconds idle before unloading
    ``0``    unload as soon as the last caller releases
    ``-1``   never unload
"""

from __future__ import annotations

import logging
import threading
from typing import Callable

logger = logging.getLogger(__name__)


class IdleUnloader:
    """Runs ``on_expire`` once nothing has held the model for ``ttl`` seconds.

    Armed when the reference count reaches zero and cancelled when it leaves
    zero, so a burst of requests keeps resetting the clock instead of racing it.
    """

    def __init__(
        self,
        ttl: float,
        on_expire: Callable[[], object],
        name: str = "model",
        timer_factory: Callable[..., threading.Timer] = threading.Timer,
    ):
        self._ttl = float(ttl)
        self._on_expire = on_expire
        self._name = name
        # Injectable so tests can fire a timer by hand instead of racing a
        # wall-clock delay against a loaded CI machine.
        self._timer_factory = timer_factory
        self._lock = threading.Lock()
        self._timer: threading.Timer | None = None
        self._generation = 0

    @property
    def ttl(self) -> float:
        """Configured idle TTL in seconds (0 = immediate, negative = never)."""
        return self._ttl

    @property
    def enabled(self) -> bool:
        """False when the model is pinned resident for the process lifetime."""
        return self._ttl >= 0

    @property
    def armed(self) -> bool:
        """True while an unload is scheduled but has not fired yet."""
        with self._lock:
            return self._timer is not None

    def cancel(self) -> None:
        """Stop any pending unload. Safe to call when nothing is armed."""
        with self._lock:
            timer, self._timer = self._timer, None
            # Invalidates a timer whose thread already left cancel()'s reach and
            # is about to call _fire().
            self._generation += 1
        if timer is not None:
            timer.cancel()

    def arm(self) -> None:
        """Schedule the unload. Call this when the last reference is released."""
        if self._ttl < 0:
            return
        if self._ttl == 0:
            # Unload as soon as the model falls idle. Synchronous on purpose:
            # the caller asked for the memory back with no grace period.
            self.cancel()
            self._fire()
            return

        with self._lock:
            self._generation += 1
            timer = self._timer_factory(self._ttl, self._fire, args=(self._generation,))
            # Daemon, or a pending timer keeps the interpreter alive past shutdown.
            timer.daemon = True
            previous, self._timer = self._timer, timer
        if previous is not None:
            previous.cancel()
        timer.start()

    def _fire(self, generation: int | None = None) -> None:
        """Timer callback. Deliberately holds no lock while calling on_expire.

        ``generation`` identifies the arm() that scheduled this call. A timer that
        was superseded (re-armed or cancelled) after it had already started
        running must neither unload nor clear the handle of its successor, or
        cancel() could no longer reach the live timer.
        """
        with self._lock:
            if generation is not None and generation != self._generation:
                return
            self._timer = None
        try:
            self._on_expire()
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("%s idle unload failed: %s", self._name, exc)


def ttl_from_env(getenv: Callable[[str, str], str], *names: str, default: float = 300.0) -> float:
    """Read the first present TTL env var. Accepts seconds, or -1 / 0 sentinels.

    Deliberately identical in behaviour to `model_lifecycle.ttl_from_env`: the
    two live in different images and cannot share a file, but a caller reading
    `.env.example` must not have to know which service parses which way.
    """
    for name in names:
        raw = (getenv(name, "") or "").strip()
        if not raw:
            continue
        try:
            return float(raw)
        except ValueError:
            logger.warning("Ignoring non-numeric %s=%r", name, raw)
    return default


# Names of the huggingface_hub failures that are not OSErrors (HFValidationError is
# a ValueError: a repo id that is not one, or a path that does not exist).
_MODEL_FILE_ERROR_NAMES = frozenset({
    "HFValidationError", "RepositoryNotFoundError", "GatedRepoError", "EntryNotFoundError",
    "RevisionNotFoundError", "LocalEntryNotFoundError", "OfflineModeIsEnabled", "HfHubHTTPError",
})
# Native libraries (CTranslate2 among them) report a missing file as a plain RuntimeError.
_MODEL_FILE_MESSAGES = ("unable to open file", "no such file", "does not appear to have a file named")


def error_category(exc: BaseException) -> str:
    """A short, stable name for a failed model load; safe in an unauthenticated response.

    ``str(exc)`` is written for the operator: it carries file paths under the
    model cache, repo ids, URLs. This keeps only what a client or a probe may see,
    and the full text stays in the log. One of ``out_of_memory``,
    ``missing_dependency``, ``model_files_unavailable`` (the checkpoint could not
    be downloaded, found or read) or ``load_error``.

    Deliberately identical in behaviour to `model_lifecycle.error_category`, which
    lives in the other images and cannot be shared with this one (its own test
    holds the two to the same answers).
    """
    try:
        text = str(exc).lower()
    except Exception:  # a broken __str__ must not turn a failed load into a crash
        text = ""
    names = {cls.__name__ for cls in type(exc).__mro__}
    if isinstance(exc, MemoryError) or "OutOfMemoryError" in names or "out of memory" in text:
        return "out_of_memory"
    if isinstance(exc, ImportError):
        return "missing_dependency"
    if (
        isinstance(exc, OSError)
        or names & _MODEL_FILE_ERROR_NAMES
        or any(marker in text for marker in _MODEL_FILE_MESSAGES)
    ):
        return "model_files_unavailable"
    return "load_error"
