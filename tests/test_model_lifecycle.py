"""Tests for the reference-counted model TTL slot.

These matter more than most: the failure mode of getting this wrong is freeing
model weights out from under a running inference thread, which surfaces as a
segfault or garbage output rather than a clean error.

The module is dependency-light by design (no torch import at module scope), so
these run offline.
"""

import threading
import time
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest

SERVICE_DIR = Path(__file__).resolve().parents[1] / "chatterbox-tts-service"


def _load():
    spec = spec_from_file_location("model_lifecycle_under_test", SERVICE_DIR / "model_lifecycle.py")
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


ml = _load()


class _FakeModel:
    def __init__(self, tag=0):
        self.tag = tag


def _counting_loader():
    """Returns (loader, calls) where calls[0] counts loads."""
    calls = [0]

    def loader():
        calls[0] += 1
        return _FakeModel(calls[0])

    return loader, calls


# --- basic residency --------------------------------------------------------


def test_not_resident_until_first_acquire():
    loader, calls = _counting_loader()
    slot = ml.ModelSlot(loader, ttl_seconds=-1, name="test")
    assert not slot.resident
    assert calls[0] == 0

    with slot.acquire() as model:
        assert model.tag == 1
        assert slot.resident
    assert calls[0] == 1


def test_second_acquire_reuses_the_loaded_model():
    loader, calls = _counting_loader()
    slot = ml.ModelSlot(loader, ttl_seconds=-1, name="test")
    with slot.acquire() as a:
        pass
    with slot.acquire() as b:
        pass
    assert calls[0] == 1, "model should not be reloaded while resident"
    assert a is b


def test_ttl_negative_never_unloads():
    loader, _ = _counting_loader()
    slot = ml.ModelSlot(loader, ttl_seconds=-1, name="test")
    with slot.acquire():
        pass
    time.sleep(0.15)
    assert slot.resident, "ttl=-1 must keep the model resident"


def test_ttl_zero_unloads_on_release():
    loader, calls = _counting_loader()
    slot = ml.ModelSlot(loader, ttl_seconds=0, name="test")
    with slot.acquire():
        assert slot.resident
    assert not slot.resident, "ttl=0 must unload as soon as the last caller releases"

    with slot.acquire():
        pass
    assert calls[0] == 2, "a second use must reload"


def test_ttl_positive_unloads_after_idle():
    loader, _ = _counting_loader()
    slot = ml.ModelSlot(loader, ttl_seconds=0.05, name="test")
    with slot.acquire():
        pass
    assert slot.resident, "must stay resident immediately after release"
    time.sleep(0.35)
    assert not slot.resident, "must unload once the idle TTL elapses"


def test_new_acquire_cancels_a_pending_unload():
    """A request arriving inside the idle window must keep the model, not race
    the timer into unloading it mid-use."""
    loader, calls = _counting_loader()
    slot = ml.ModelSlot(loader, ttl_seconds=0.15, name="test")
    with slot.acquire():
        pass
    time.sleep(0.05)          # inside the window
    with slot.acquire():
        time.sleep(0.25)      # would have expired if the timer had not been cancelled
        assert slot.resident, "model was unloaded while still in use"
    assert calls[0] == 1


# --- the dangerous case: never unload something in use ----------------------


def test_unload_refuses_while_in_use():
    loader, _ = _counting_loader()
    slot = ml.ModelSlot(loader, ttl_seconds=-1, name="test")
    with slot.acquire():
        assert slot.refs == 1
        assert slot.unload() is False, "must not unload a model that is in use"
        assert slot.resident
    assert slot.unload() is True


def test_concurrent_holders_are_ref_counted():
    loader, calls = _counting_loader()
    slot = ml.ModelSlot(loader, ttl_seconds=0, name="test")
    release = threading.Event()
    seen = []

    def worker():
        with slot.acquire() as m:
            seen.append(m)
            release.wait(timeout=5)

    threads = [threading.Thread(target=worker) for _ in range(3)]
    for t in threads:
        t.start()

    # Wait for all three to be holding a reference. Polling rather than a
    # Barrier: the main thread is not one of the parties, and a mismatched
    # party count breaks the barrier instead of failing the assertion.
    deadline = time.monotonic() + 5
    while slot.refs < 3 and time.monotonic() < deadline:
        time.sleep(0.01)

    try:
        assert slot.refs == 3, f"expected 3 refs, got {slot.refs}"
        assert slot.resident
        assert calls[0] == 1, "three concurrent users must share one load"
    finally:
        release.set()
        for t in threads:
            t.join(timeout=5)

    # Only the LAST release may trigger the unload.
    assert slot.refs == 0
    assert not slot.resident
    assert len(seen) == 3
    assert calls[0] == 1


def test_exception_in_the_block_still_releases():
    loader, _ = _counting_loader()
    slot = ml.ModelSlot(loader, ttl_seconds=-1, name="test")
    with pytest.raises(ValueError):
        with slot.acquire():
            raise ValueError("boom")
    assert slot.refs == 0, "a failed request must not leak a reference"
    assert slot.unload() is True


def test_on_unload_hook_runs_and_a_failing_hook_does_not_block_unload():
    calls = []

    def loader():
        return _FakeModel()

    slot = ml.ModelSlot(loader, ttl_seconds=0, name="test",
                        on_unload=lambda m: calls.append(m))
    with slot.acquire():
        pass
    assert len(calls) == 1
    assert not slot.resident

    def bad_hook(_m):
        raise RuntimeError("hook failed")

    slot2 = ml.ModelSlot(loader, ttl_seconds=0, name="test2", on_unload=bad_hook)
    with slot2.acquire():
        pass
    assert not slot2.resident, "a failing unload hook must not leave the model resident"


# --- env parsing ------------------------------------------------------------


def test_ttl_from_env_reads_first_present_name():
    env = {"TTS_MODEL_TTL": "120"}
    assert ml.ttl_from_env(lambda k, d="": env.get(k, d), "TTS_MODEL_TTL", "MODEL_TTL") == 120.0


def test_ttl_from_env_falls_through_to_default():
    assert ml.ttl_from_env(lambda k, d="": "", "NOPE", default=300.0) == 300.0


def test_ttl_from_env_accepts_sentinels():
    for raw, expected in (("-1", -1.0), ("0", 0.0)):
        env = {"MODEL_TTL": raw}
        assert ml.ttl_from_env(lambda k, d="": env.get(k, d), "MODEL_TTL") == expected


def test_ttl_from_env_ignores_garbage():
    env = {"MODEL_TTL": "soon"}
    assert ml.ttl_from_env(lambda k, d="": env.get(k, d), "MODEL_TTL", default=42.0) == 42.0


# --- /health must never block on a model load --------------------------------
#
# _acquire holds the lock for the whole of loader(), which for a NeMo checkpoint
# is tens of seconds to minutes. Docker polls /health with timeout 10s and
# retries 3, and /health reads `resident` and `refs` — so if those took the lock,
# any reload would produce three consecutive probe timeouts and a container
# marked unhealthy mid-load. Idle unloading is what makes a reload happen
# outside start_period at all, so this is reachable in normal operation.


def test_status_properties_do_not_block_while_the_model_is_loading():
    """The property that keeps the healthcheck honest during a slow load."""
    release = threading.Event()
    loading = threading.Event()

    def slow_loader():
        loading.set()
        release.wait(timeout=5)
        return _FakeModel()

    slot = ml.ModelSlot(slow_loader, ttl_seconds=-1, name="slow")

    worker = threading.Thread(target=lambda: slot.acquire().__enter__(), daemon=True)
    worker.start()
    assert loading.wait(timeout=5), "loader never started"

    # The lock is held by the in-flight load right now. Both reads must answer
    # anyway, and quickly.
    done = threading.Event()
    seen = {}

    def probe():
        seen["resident"] = slot.resident
        seen["refs"] = slot.refs
        done.set()

    threading.Thread(target=probe, daemon=True).start()
    assert done.wait(timeout=2), (
        "reading slot.resident/.refs blocked while a load held the lock — "
        "/health would time out and the container would be marked unhealthy"
    )
    assert seen["resident"] is False  # not yet assigned
    assert seen["refs"] == 0

    release.set()
    worker.join(timeout=5)


def test_the_unload_decision_still_takes_the_lock():
    """Lock-free reads are for reporting only. Anything that FREES memory must
    still decide under the lock, or it can race a request that just arrived."""
    import inspect

    for name in ("try_unload", "_expire"):
        source = inspect.getsource(getattr(ml.ModelSlot, name))
        assert "self._lock" in source, (
            f"ModelSlot.{name} no longer decides under the lock; a status read "
            f"being lock-free must not spread to the safety path"
        )
