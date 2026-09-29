"""Tests for the reference-counted model TTL slot.

These matter more than most: the failure mode of getting this wrong is freeing
model weights out from under a running inference thread, which surfaces as a
segfault or garbage output rather than a clean error.

The module is dependency-light by design (no torch import at module scope), so
these run offline.

Determinism: the idle timer is injected (`_FakeTimers`) and fired by hand, and
every cross-thread hand-off is a `threading.Event`. No test asserts on how long
something took, so none can flake on a loaded CI box. The only clock in here is
the generous deadline in `_wait_until`, which bounds a wait for something that
must happen and never decides whether an assertion passes.
"""

import asyncio
import gc
import threading
import time
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest

SERVICE_DIR = Path(__file__).resolve().parents[1] / "chatterbox-tts-service"

WAIT = 5.0  # upper bound for "must happen", never a duration under test


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


class _FakeTimer:
    """Stands in for threading.Timer; `fire()` is the interval elapsing."""

    def __init__(self, interval, function):
        self.interval = interval
        self.function = function
        self.daemon = False
        self.started = False
        self.cancelled = False

    def start(self):
        self.started = True

    def cancel(self):
        self.cancelled = True

    def fire(self):
        """Run the callback even if cancelled: models a timer thread that had
        already fired and was waiting on the slot lock when cancel() landed."""
        self.function()


class _FakeTimers:
    def __init__(self):
        self.created = []

    def __call__(self, interval, function):
        timer = _FakeTimer(interval, function)
        self.created.append(timer)
        return timer

    @property
    def last(self):
        return self.created[-1]

    @property
    def armed(self):
        """Timers that are still pending (started and not cancelled)."""
        return [t for t in self.created if t.started and not t.cancelled]


def _slot(ttl, loader=None, **kw):
    timers = _FakeTimers()
    loader = loader or _counting_loader()[0]
    return ml.ModelSlot(loader, ttl_seconds=ttl, name="test", timer_factory=timers, **kw), timers


def _wait_until(predicate, timeout=WAIT):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            return False
        time.sleep(0.005)
    return True


async def _await_until(predicate, timeout=WAIT):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            return False
        await asyncio.sleep(0.005)
    return True


class _BlockingLoader:
    """A loader that parks until told to finish, and can then succeed or fail."""

    def __init__(self, error=None):
        self.started = threading.Event()
        self.proceed = threading.Event()
        self.error = error
        self.calls = 0

    def __call__(self):
        self.calls += 1
        self.started.set()
        assert self.proceed.wait(WAIT), "test never released the loader"
        if self.error is not None:
            raise self.error
        return _FakeModel(self.calls)


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
    slot, timers = _slot(-1)
    with slot.acquire():
        pass
    assert timers.created == [], "ttl=-1 must not even arm an idle timer"
    assert slot.resident, "ttl=-1 must keep the model resident"


def test_ttl_zero_unloads_on_release():
    loader, calls = _counting_loader()
    slot, timers = _slot(0, loader)
    with slot.acquire():
        assert slot.resident
    assert not slot.resident, "ttl=0 must unload as soon as the last caller releases"
    assert timers.created == [], "ttl=0 unloads inline, without a timer"

    with slot.acquire():
        pass
    assert calls[0] == 2, "a second use must reload"


def test_ttl_positive_arms_a_daemon_timer_and_unloads_when_it_fires():
    slot, timers = _slot(42.0)
    with slot.acquire():
        pass
    assert slot.resident, "must stay resident immediately after release"
    assert len(timers.armed) == 1
    timer = timers.last
    assert timer.interval == 42.0
    assert timer.daemon is True, "a pending timer must not keep the process alive on exit"

    timer.fire()
    assert not slot.resident, "must unload once the idle TTL elapses"


def test_default_timer_really_unloads_after_idle():
    """One end-to-end run with the real threading.Timer, waited on with an Event
    rather than a sleep."""
    unloaded = threading.Event()
    slot = ml.ModelSlot(lambda: _FakeModel(), ttl_seconds=0.01, name="test",
                        on_unload=lambda m: unloaded.set())
    with slot.acquire():
        pass
    assert unloaded.wait(WAIT), "the idle timer never unloaded the model"
    assert _wait_until(lambda: not slot.resident)


def test_new_acquire_cancels_a_pending_unload():
    """A request arriving inside the idle window must keep the model, not race
    the timer into unloading it mid-use."""
    loader, calls = _counting_loader()
    slot, timers = _slot(60.0, loader)
    with slot.acquire():
        pass
    first = timers.last
    with slot.acquire():
        assert first.cancelled, "the acquire must cancel the pending idle timer"
        # Even a timer that had already fired (cancel() cannot stop that thread)
        # must not free a model that is in use.
        first.fire()
        assert slot.resident, "model was unloaded while still in use"
    assert calls[0] == 1


def test_a_stale_timer_cannot_unload_a_model_that_was_used_since():
    """The timer thread had already fired and was waiting on the lock while a
    request came and went. The model was released again a moment ago, so the
    new idle window has barely started; the old timer must not end it."""
    slot, timers = _slot(60.0)
    with slot.acquire():
        pass
    stale = timers.last
    with slot.acquire():
        pass
    fresh = timers.last
    assert fresh is not stale

    stale.fire()
    assert slot.resident, "a superseded timer unloaded the model early"

    fresh.fire()
    assert not slot.resident, "the current timer must still unload"


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
    entered = [threading.Event() for _ in range(3)]
    seen = []

    def worker(flag):
        with slot.acquire() as m:
            seen.append(m)
            flag.set()
            release.wait(timeout=WAIT)

    threads = [threading.Thread(target=worker, args=(e,)) for e in entered]
    for t in threads:
        t.start()

    try:
        for e in entered:
            assert e.wait(WAIT), "a worker never acquired the model"
        assert slot.refs == 3, f"expected 3 refs, got {slot.refs}"
        assert slot.resident
        assert calls[0] == 1, "three concurrent users must share one load"
    finally:
        release.set()
        for t in threads:
            t.join(timeout=WAIT)

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
    loader = _BlockingLoader()
    slot = ml.ModelSlot(loader, ttl_seconds=-1, name="slow")

    worker = threading.Thread(target=slot.lease, daemon=True)
    worker.start()
    assert loader.started.wait(WAIT), "loader never started"

    # The lock is held by the in-flight load right now. Every read must answer
    # anyway, and quickly.
    done = threading.Event()
    seen = {}

    def probe():
        seen["resident"] = slot.resident
        seen["refs"] = slot.refs
        seen["ever_loaded"] = slot.ever_loaded
        seen["loading"] = slot.loading
        seen["last_error"] = slot.last_error
        seen["readiness"] = slot.readiness()
        done.set()

    threading.Thread(target=probe, daemon=True).start()
    assert done.wait(timeout=2), (
        "reading the slot status blocked while a load held the lock — "
        "/health would time out and the container would be marked unhealthy"
    )
    assert seen["resident"] is False  # not yet assigned
    assert seen["refs"] == 0
    assert seen["loading"] is True
    assert seen["ever_loaded"] is False
    assert seen["last_error"] is None
    assert seen["readiness"]["ready"] is False

    loader.proceed.set()
    worker.join(timeout=WAIT)


def test_unload_decisions_are_made_under_the_lock():
    """Lock-free reads are for reporting only. Anything that FREES memory must
    still decide under the lock, or it can race a request that just arrived.

    White-box: the test holds the slot lock the way an in-flight acquire would.
    A timer callback fired now has to wait for it, so that by the time it looks
    at the reference count it sees the acquire's reference.
    """
    slot, timers = _slot(60.0)
    with slot.acquire():
        pass
    expire = timers.last
    assert slot.refs == 0 and slot.resident

    finished = threading.Event()
    with slot._lock:
        threading.Thread(target=lambda: (expire.fire(), finished.set()), daemon=True).start()
        assert not finished.wait(0.2), (
            "the expiry callback decided to unload without taking the lock, so it "
            "can free the model under an acquire that is mid-flight"
        )
        slot._refs += 1  # the acquire completing
    assert finished.wait(WAIT)
    assert slot.resident, "expiry unloaded a model that had just been acquired"
    slot._release()  # balance

    # try_unload has the same obligation.
    slot2, _ = _slot(-1)
    with slot2.acquire():
        pass
    result = {}
    finished = threading.Event()
    with slot2._lock:
        threading.Thread(
            target=lambda: (result.update(slot2.try_unload()), finished.set()), daemon=True
        ).start()
        assert not finished.wait(0.2), "try_unload decided without taking the lock"
        slot2._refs += 1
    assert finished.wait(WAIT)
    assert result["reason"] == "busy" and slot2.resident
    slot2._release()


# --- L1: cancelling the awaiting task must not orphan the reference ----------
#
# asyncio.to_thread cannot be cancelled once its thread is running. A client
# that disconnects while the model is loading cancels the handler, but the
# thread still completes _acquire() and bumps the reference count; nothing is
# left to release it. Idle TTL never arms and /unload answers 409 forever.


class _AcquireProbe:
    """Wraps slot._acquire to say when worker threads enter and leave it, so a
    test can wait for "the thread that took the reference has finished" instead
    of guessing from `refs`, which is transiently 0 between load and increment."""

    def __init__(self, slot):
        self.entered = []
        self.finished = []
        real = slot._acquire

        def wrapped():
            self.entered.append(1)
            try:
                return real()
            finally:
                self.finished.append(1)

        slot._acquire = wrapped  # run_in_executor looks it up per call


class _SubmitProbe:
    """Counts the jobs a slot hands to its acquire worker, so a test can wait for
    "this waiter has queued its acquire" instead of sleeping."""

    def __init__(self, slot):
        self.count = 0
        real = slot._acquire_pool.submit

        def counting(fn, *args, **kwargs):
            self.count += 1
            return real(fn, *args, **kwargs)

        slot._acquire_pool.submit = counting


def _cancel_while_loading(slot, loader, acquire):
    """Start `acquire(slot)` as a task, cancel it mid-load, let the load finish.

    Returns once the worker thread has completed its acquire; the hand-back that
    follows is asynchronous, so callers wait for its effect themselves.
    """
    probe = _AcquireProbe(slot)

    async def main():
        task = asyncio.create_task(acquire(slot))
        assert await asyncio.to_thread(loader.started.wait, WAIT), "load never started"
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        loader.proceed.set()
        assert await _await_until(lambda: len(probe.finished) == 1), "acquire thread never finished"

    asyncio.run(main())


async def _use_async(slot):
    async with slot.acquire_async():
        await asyncio.sleep(3600)


async def _use_ref(slot):
    await slot.acquire_ref()
    await asyncio.sleep(3600)


async def _use_lease(slot):
    lease = await slot.acquire_lease()
    await asyncio.sleep(3600)
    lease.release()


@pytest.mark.parametrize("acquire", [_use_async, _use_ref, _use_lease],
                         ids=["acquire_async", "acquire_ref", "acquire_lease"])
def test_cancelled_acquire_hands_its_reference_back(acquire):
    loader = _BlockingLoader()
    slot, timers = _slot(60.0, loader)

    _cancel_while_loading(slot, loader, acquire)

    assert _wait_until(lambda: len(timers.armed) == 1), (
        f"no idle timer armed, so the model is pinned forever (refs={slot.refs})"
    )
    assert slot.refs == 0
    # The user-visible symptom: /unload answering 409 until the process restarts.
    assert slot.try_unload() == {"unloaded": True, "reason": "ok", "refs": 0}


def test_cancelled_acquire_with_ttl_zero_unloads_once_the_load_lands():
    loader = _BlockingLoader()
    unloaded = threading.Event()
    slot = ml.ModelSlot(loader, ttl_seconds=0, name="test", on_unload=lambda m: unloaded.set())

    async def main():
        task = asyncio.create_task(_use_async(slot))
        assert await asyncio.to_thread(loader.started.wait, WAIT)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        loader.proceed.set()
        assert await asyncio.to_thread(unloaded.wait, WAIT), "orphaned reference kept the model"

    asyncio.run(main())
    assert _wait_until(lambda: slot.refs == 0 and not slot.resident)


def test_a_cancelled_acquire_does_not_disturb_other_holders():
    """Cancelling one waiter must give back exactly its own reference."""
    loader = _BlockingLoader()
    slot, _ = _slot(-1, loader)
    probe = _AcquireProbe(slot)
    submitted = _SubmitProbe(slot)

    async def main():
        doomed = asyncio.create_task(_use_async(slot))
        assert await asyncio.to_thread(loader.started.wait, WAIT)
        keeper = asyncio.create_task(slot.acquire_lease())
        # The doomed acquire is inside the load; the keeper's is queued behind it.
        assert await _await_until(lambda: submitted.count == 2)
        doomed.cancel()
        with pytest.raises(asyncio.CancelledError):
            await doomed
        loader.proceed.set()
        lease = await keeper
        assert await _await_until(lambda: len(probe.finished) == 2)
        assert await _await_until(lambda: slot.refs == 1), f"refs={slot.refs}"
        assert lease.model is not None
        assert slot.try_unload()["reason"] == "busy"
        await lease.release_async()

    asyncio.run(main())
    assert slot.refs == 0


def test_a_failed_load_awaited_normally_raises_and_takes_no_reference():
    loader = _BlockingLoader(error=RuntimeError("no VRAM"))
    slot, _ = _slot(-1, loader)
    loader.proceed.set()

    async def main():
        with pytest.raises(RuntimeError, match="no VRAM"):
            async with slot.acquire_async():
                raise AssertionError("body must not run")

    asyncio.run(main())
    assert slot.refs == 0 and not slot.resident


def test_a_failed_load_after_the_caller_left_is_harmless():
    loader = _BlockingLoader(error=RuntimeError("no VRAM"))
    slot, timers = _slot(60.0, loader)

    async def main():
        task = asyncio.create_task(_use_async(slot))
        assert await asyncio.to_thread(loader.started.wait, WAIT)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        loader.proceed.set()
        assert await _await_until(lambda: not slot.loading and slot.last_error is not None)
        await asyncio.sleep(0.05)  # let the done-callback run

    asyncio.run(main())
    # Nothing was acquired, so nothing may be "handed back" (which would
    # underflow into a spurious timer / unload).
    assert slot.refs == 0
    assert timers.created == []


def test_cancelling_the_body_releases_the_reference():
    slot, _ = _slot(-1)

    async def main():
        entered = asyncio.Event()

        async def body():
            async with slot.acquire_async():
                entered.set()
                await asyncio.sleep(3600)

        task = asyncio.create_task(body())
        await entered.wait()
        assert slot.refs == 1
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        # Awaited release: the count is already down when the task is done.
        assert slot.refs == 0

    asyncio.run(main())


def test_release_async_survives_its_own_cancellation():
    slot, _ = _slot(-1)
    with slot.acquire():
        pass
    model = slot._acquire()  # take one reference by hand
    assert slot.refs == 1

    async def main():
        with slot._lock:  # the release thread has to wait for this
            task = asyncio.create_task(slot.release_async())
            await asyncio.sleep(0.05)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        assert await _await_until(lambda: slot.refs == 0), "cancelled release was dropped"

    asyncio.run(main())
    assert model is not None


# --- L2: nothing on the event loop may wait for the slot lock -----------------


def test_release_async_keeps_the_event_loop_free_while_the_unload_hook_runs():
    """ttl=0 unloads inside the release, and the unload hook (.cpu() on a
    multi-GB module) can take seconds. Run inline on the loop, that freezes
    every other request and /health for the duration."""
    hook_started = threading.Event()
    hook_may_finish = threading.Event()

    def hook(_model):
        hook_started.set()
        hook_may_finish.wait(WAIT)

    slot = ml.ModelSlot(lambda: _FakeModel(), ttl_seconds=0, name="test", on_unload=hook)

    async def main():
        await slot.acquire_ref()
        releasing = asyncio.create_task(slot.release_async())
        assert await asyncio.to_thread(hook_started.wait, WAIT)
        ticks = 0
        for _ in range(3):
            await asyncio.sleep(0)  # only returns if the loop is not blocked
            ticks += 1
        assert ticks == 3 and not releasing.done()
        hook_may_finish.set()
        await releasing

    asyncio.run(main())
    assert not slot.resident and slot.refs == 0


def test_try_unload_async_keeps_the_event_loop_free_and_reports_like_the_sync_form():
    hook_started = threading.Event()
    hook_may_finish = threading.Event()

    def hook(_model):
        hook_started.set()
        hook_may_finish.wait(WAIT)

    slot = ml.ModelSlot(lambda: _FakeModel(), ttl_seconds=-1, name="test", on_unload=hook)

    async def main():
        assert await slot.try_unload_async() == {
            "unloaded": False, "reason": "not_resident", "refs": 0}
        with slot.acquire():
            pass
        unloading = asyncio.create_task(slot.try_unload_async())
        assert await asyncio.to_thread(hook_started.wait, WAIT)
        await asyncio.sleep(0)
        assert not unloading.done()
        hook_may_finish.set()
        assert await unloading == {"unloaded": True, "reason": "ok", "refs": 0}
        assert await slot.unload_async() is False

    asyncio.run(main())


def test_try_unload_during_a_load_answers_busy_without_waiting_for_it():
    """/unload used to block until the load finished (minutes for NeMo), and on
    the event loop that froze the whole service."""
    loader = _BlockingLoader()
    slot, _ = _slot(-1, loader)
    worker = threading.Thread(target=slot.lease, daemon=True)
    worker.start()
    assert loader.started.wait(WAIT)

    answered = threading.Event()
    result = {}

    def call():
        result.update(slot.try_unload())
        answered.set()

    threading.Thread(target=call, daemon=True).start()
    assert answered.wait(2), "try_unload waited for the model load"
    assert result["unloaded"] is False and result["reason"] == "busy"

    loader.proceed.set()
    worker.join(timeout=WAIT)


# --- L3: readiness introspection ---------------------------------------------


def test_a_fresh_slot_is_loadable_but_has_never_loaded():
    slot, _ = _slot(-1)
    assert slot.ever_loaded is False
    assert slot.loading is False
    assert slot.last_error is None
    ready = slot.readiness()
    assert ready["ready"] is True and ready["reason"] == "ok"
    assert ready["resident"] is False


def test_readiness_is_503_while_the_first_load_runs_then_200():
    loader = _BlockingLoader()
    slot, _ = _slot(-1, loader)
    worker = threading.Thread(target=slot.lease, daemon=True)
    worker.start()
    assert loader.started.wait(WAIT)

    assert slot.loading is True and slot.ever_loaded is False
    during = slot.readiness()
    assert during["ready"] is False
    assert during["reason"] == "loading"

    loader.proceed.set()
    worker.join(timeout=WAIT)

    assert slot.loading is False and slot.ever_loaded is True
    after = slot.readiness()
    assert after["ready"] is True and after["reason"] == "ok"
    assert after["resident"] is True


def test_readiness_is_503_when_the_load_failed_and_recovers_on_success():
    attempts = []

    def flaky():
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("CUDA out of memory")
        return _FakeModel()

    slot, _ = _slot(-1, flaky)
    with pytest.raises(RuntimeError):
        slot.lease()

    assert slot.ever_loaded is False and slot.loading is False
    assert slot.last_error == "RuntimeError: CUDA out of memory"
    failed = slot.readiness()
    assert failed["ready"] is False
    assert failed["reason"] == "load_failed"
    assert failed["detail"] == "RuntimeError: CUDA out of memory"
    assert slot.refs == 0, "a failed load must not take a reference"

    with slot.acquire():
        pass
    assert slot.last_error is None, "a successful load clears the error"
    assert slot.readiness()["ready"] is True


def test_ever_loaded_is_sticky_across_ttl_unload_and_a_failed_reload():
    """Once the weights have loaded, the service stays ready: a TTL unload is
    normal, and one transient reload failure (another service grabbed the VRAM)
    must not turn the container unready, since nothing but a request can clear it."""
    fail = [False]

    def loader():
        if fail[0]:
            raise RuntimeError("out of memory")
        return _FakeModel()

    slot, timers = _slot(60.0, loader)
    with slot.acquire():
        pass
    timers.last.fire()
    assert not slot.resident
    assert slot.ever_loaded is True
    assert slot.readiness()["ready"] is True, "TTL unload must not make the service unready"

    fail[0] = True
    with pytest.raises(RuntimeError):
        with slot.acquire():
            pass
    assert slot.ever_loaded is True
    assert slot.last_error == "RuntimeError: out of memory"
    snapshot = slot.readiness()
    assert snapshot["ready"] is True
    assert snapshot["last_error"] == "RuntimeError: out of memory", "still visible for diagnostics"


def test_a_long_load_error_is_truncated():
    def loader():
        raise RuntimeError("x" * 5000)

    slot, _ = _slot(-1, loader)
    with pytest.raises(RuntimeError):
        slot.lease()
    assert len(slot.last_error) <= 500
    assert slot.last_error.startswith("RuntimeError: xxx")


# --- L4: leases ---------------------------------------------------------------


def test_lease_release_is_idempotent_and_only_the_first_counts():
    slot, _ = _slot(-1)
    other = slot.lease()
    lease = slot.lease()
    assert slot.refs == 2

    assert lease.release() is True
    assert lease.release() is False
    assert lease.release() is False
    assert lease.released is True
    assert slot.refs == 1, "a repeated release stole another holder's reference"
    assert slot.try_unload()["reason"] == "busy"

    other.release()
    assert slot.refs == 0


def test_lease_exposes_the_model_only_while_held():
    slot, _ = _slot(-1)
    lease = slot.lease()
    assert lease.model.tag == 1
    lease.release()
    with pytest.raises(RuntimeError, match="released"):
        lease.model


def test_racing_releases_release_exactly_once():
    slot, _ = _slot(-1)
    keeper = slot.lease()
    lease = slot.lease()
    start = threading.Barrier(8)
    outcomes = []

    def race():
        start.wait(WAIT)
        outcomes.append(lease.release())

    threads = [threading.Thread(target=race) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(WAIT)

    assert sorted(outcomes) == [False] * 7 + [True]
    assert slot.refs == 1
    keeper.release()


def test_lease_as_context_manager():
    slot, _ = _slot(-1)
    with slot.lease() as lease:
        assert slot.refs == 1
    assert lease.released and slot.refs == 0

    async def main():
        async with await slot.acquire_lease() as alease:
            assert slot.refs == 1
        assert alease.released

    asyncio.run(main())
    assert slot.refs == 0


def test_concurrent_async_releases_release_once():
    slot, _ = _slot(-1)

    async def main():
        keeper = await slot.acquire_lease()
        lease = await slot.acquire_lease()
        results = await asyncio.gather(*(lease.release_async() for _ in range(5)))
        assert sorted(results) == [False] * 4 + [True]
        assert slot.refs == 1
        await keeper.release_async()

    asyncio.run(main())
    assert slot.refs == 0


def test_release_soon_never_blocks_on_the_slot_lock():
    """It is the one that runs from done-callbacks and finalizers, i.e. possibly
    on the event loop, so it must return even when the lock is unavailable."""
    slot, _ = _slot(-1)
    lease = slot.lease()

    with slot._lock:
        returned = threading.Event()
        threading.Thread(target=lambda: (lease.release_soon(), returned.set()), daemon=True).start()
        assert returned.wait(2), "release_soon blocked on the slot lock"
        assert lease.released
        assert slot.refs == 1, "the slot cannot have been updated while the lock was held"
    assert _wait_until(lambda: slot.refs == 0), "the deferred release never landed"


async def _never_started_stream(lease):
    # Mirrors chatterbox's generate_chunks: the reference is released in a
    # `finally`, which only exists once the generator has been started.
    try:
        yield b"header"
    finally:
        lease.release()


def test_a_generator_that_never_started_releases_via_release_on_gc():
    """The streaming leak: the client is gone before Starlette iterates the body,
    so the generator's `finally` never runs. Closing/collecting an unstarted
    generator executes none of its code."""
    slot, _ = _slot(-1)
    lease = slot.lease()
    stream = _never_started_stream(lease)
    lease.release_on_gc(stream)
    assert slot.refs == 1

    del stream
    gc.collect()

    assert _wait_until(lambda: slot.refs == 0), "unstarted generator leaked its reference"
    assert lease.released


def test_without_release_on_gc_an_unstarted_generator_would_hold_the_reference():
    """Guards the premise of the test above, so it cannot pass vacuously."""
    slot, _ = _slot(-1)
    lease = slot.lease()
    stream = _never_started_stream(lease)
    del stream
    gc.collect()
    assert slot.refs == 1 and not lease.released
    lease.release()


def test_an_explicit_release_and_a_later_gc_release_only_count_once():
    slot, _ = _slot(-1)
    keeper = slot.lease()
    lease = slot.lease()
    stream = _never_started_stream(lease)
    lease.release_on_gc(stream)

    assert lease.release() is True
    del stream
    gc.collect()
    assert slot.refs == 1, "the finalizer released a second time"
    keeper.release()


def test_release_on_gc_after_release_registers_nothing():
    slot, _ = _slot(-1)
    keeper = slot.lease()
    lease = slot.lease()
    lease.release()
    stream = _never_started_stream(lease)
    lease.release_on_gc(stream)
    del stream
    gc.collect()
    assert slot.refs == 1
    keeper.release()


def test_a_dropped_lease_releases_itself():
    """Last-resort net: every owner dropped the lease, so nobody can release it."""
    slot, timers = _slot(60.0)
    lease = slot.lease()
    assert slot.refs == 1

    del lease
    gc.collect()

    assert _wait_until(lambda: slot.refs == 0), "a lease dropped unreleased pinned the model"
    assert _wait_until(lambda: len(timers.armed) == 1), "idle timer never armed"


def test_leases_from_the_async_and_sync_paths_share_one_load():
    loader, calls = _counting_loader()
    slot = ml.ModelSlot(loader, ttl_seconds=-1, name="test")

    async def main():
        lease = await slot.acquire_lease()
        return lease

    lease = asyncio.run(main())
    other = slot.lease()
    assert lease.model is other.model
    assert calls[0] == 1
    assert slot.refs == 2
    lease.release()
    other.release()


# --- A2: waiters behind a load must not empty the default executor ------------
#
# Every queued acquire used to park one default-executor thread on the slot lock.
# With the 4-CPU default pool of 8 and 60 waiters during a 4 s load, an unrelated
# asyncio.to_thread waited 3.7 s: uploads, /unload and every other user of the
# default executor stalled for the length of the load.


def _threads_named(prefix):
    return [t for t in threading.enumerate() if t.name.startswith(prefix)]


def test_sixty_waiters_during_a_load_leave_the_default_executor_free():
    loader = _BlockingLoader()
    slot = ml.ModelSlot(loader, ttl_seconds=-1, name="sixty waiters")
    submitted = _SubmitProbe(slot)

    async def main():
        waiters = [asyncio.create_task(slot.acquire_lease()) for _ in range(60)]
        try:
            assert await asyncio.to_thread(loader.started.wait, WAIT), "load never started"
            assert await _await_until(lambda: submitted.count == 60), "waiters never queued"

            # The load is still stuck. Unrelated work on the default executor, which
            # the parked waiters used to occupy completely, has to run now.
            assert await asyncio.wait_for(asyncio.to_thread(lambda: "free"), 3) == "free", (
                "an unrelated asyncio.to_thread waited behind the model load"
            )
            # So must /unload: it answers "busy" at once instead of queueing.
            assert await asyncio.wait_for(slot.try_unload_async(), 3) == {
                "unloaded": False, "reason": "busy", "refs": 0}
            # One thread waits for the slot, not sixty.
            assert len(_threads_named("sixty-waiters-acquire")) == 1
        finally:
            loader.proceed.set()  # a failure above must not leave the waiters stuck on the load

        leases = await asyncio.gather(*waiters)
        assert loader.calls == 1, "the waiters must share one load"
        assert slot.refs == 60
        await asyncio.gather(*(lease.release_async() for lease in leases))

    asyncio.run(main())
    assert slot.refs == 0


def test_a_waiter_cancelled_before_its_acquire_started_takes_nothing():
    """A queued acquire is withdrawn, not run for nobody.

    Run for nobody it would retry a failing load (minutes of NeMo) or take a
    reference only to hand it straight back.
    """
    loader = _BlockingLoader(error=RuntimeError("no VRAM"))
    slot, timers = _slot(60.0, loader)
    probe = _AcquireProbe(slot)
    submitted = _SubmitProbe(slot)

    async def main():
        first = asyncio.create_task(slot.acquire_lease())
        assert await asyncio.to_thread(loader.started.wait, WAIT)
        second = asyncio.create_task(slot.acquire_lease())
        assert await _await_until(lambda: submitted.count == 2)
        second.cancel()
        with pytest.raises(asyncio.CancelledError):
            await second
        loader.proceed.set()
        with pytest.raises(RuntimeError, match="no VRAM"):
            await first
        # One worker, first in first out: once this returns, the second acquire
        # would already have run had it not been withdrawn.
        await asyncio.wrap_future(slot._acquire_pool.submit(lambda: None))

    asyncio.run(main())
    assert loader.calls == 1, "the withdrawn acquire retried the load"
    assert len(probe.entered) == 1
    assert slot.refs == 0 and timers.created == []


def test_a_burst_of_cancelled_waiters_leaves_the_slot_usable():
    loader = _BlockingLoader()
    slot, timers = _slot(60.0, loader)
    submitted = _SubmitProbe(slot)

    async def main():
        waiters = [asyncio.create_task(_use_async(slot)) for _ in range(30)]
        assert await asyncio.to_thread(loader.started.wait, WAIT)
        assert await _await_until(lambda: submitted.count == 30)
        for waiter in waiters:
            waiter.cancel()
        await asyncio.gather(*waiters, return_exceptions=True)
        loader.proceed.set()
        # The slot still serves the next caller, which shares the one load.
        async with asyncio.timeout(WAIT):
            async with slot.acquire_async() as model:
                assert model is not None
        assert await _await_until(lambda: slot.refs == 0)

    asyncio.run(main())
    assert loader.calls == 1
    assert _wait_until(lambda: len(timers.armed) == 1), "the reference of the started acquire leaked"


# --- A3: what a failed load may say to an anonymous caller ---------------------


@pytest.mark.parametrize("error, category", [
    (MemoryError(), "out_of_memory"),
    (RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB"), "out_of_memory"),
    (type("OutOfMemoryError", (RuntimeError,), {})("nope"), "out_of_memory"),
    (ModuleNotFoundError("No module named 'nemo'"), "missing_dependency"),
    (FileNotFoundError("/root/.cache/huggingface/hub/models--x/snapshots/abc/model.bin"), "model_files_unavailable"),
    (ConnectionError("HTTPSConnectionPool(host='huggingface.co', port=443)"), "model_files_unavailable"),
    (PermissionError(13, "Permission denied", "/models/x.nemo"), "model_files_unavailable"),
    (type("HFValidationError", (ValueError,), {})("Repo id must be in the form 'repo_name'"), "model_files_unavailable"),
    (RuntimeError("weights are corrupt"), "load_error"),
    (KeyError("layers.3"), "load_error"),
])
def test_error_category_names_the_failure_without_repeating_it(error, category):
    assert ml.error_category(error) == category


def test_error_category_survives_an_exception_whose_str_raises():
    class Hostile(Exception):
        def __str__(self):
            raise RuntimeError("no str for you")

    assert ml.error_category(Hostile()) == "load_error"


def test_a_failed_load_publishes_a_path_free_category_next_to_the_raw_error():
    leak = "/root/.cache/huggingface/hub/models--nvidia--parakeet/snapshots/abc123/model.nemo"

    def loader():
        raise FileNotFoundError(f"No such file: {leak}")

    slot, _ = _slot(-1, loader)
    with pytest.raises(FileNotFoundError):
        slot.lease()

    assert slot.last_error_category == "model_files_unavailable"
    snapshot = slot.readiness()
    assert snapshot["reason"] == "load_failed"
    assert snapshot["error_category"] == "model_files_unavailable"
    assert leak not in snapshot["error_category"]
    # The raw text is still there for the log and for callers that authenticate.
    assert leak in snapshot["detail"] and leak in snapshot["last_error"]


def test_a_successful_load_clears_the_category_and_a_fresh_slot_has_none():
    attempts = []

    def flaky():
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("boom")
        return _FakeModel()

    slot, _ = _slot(-1, flaky)
    assert slot.last_error_category is None and slot.readiness()["error_category"] is None
    with pytest.raises(RuntimeError):
        slot.lease()
    assert slot.last_error_category == "load_error"
    with slot.acquire():
        pass
    assert slot.last_error_category is None and slot.readiness()["error_category"] is None


# --- A1: RequestGate ------------------------------------------------------------


class _Refused(Exception):
    def __init__(self, reason):
        super().__init__(reason)
        self.reason = reason


def _gate(max_active=1, max_queue=1, timeout=0.05):
    return ml.RequestGate(max_active, max_queue, timeout, busy=_Refused)


def test_admit_refuses_beyond_active_plus_queue_at_once_and_frees_the_place_on_exit():
    gate = _gate(max_active=2, max_queue=3)
    with gate.admit(), gate.admit(), gate.admit(), gate.admit(), gate.admit():
        assert gate.in_flight == 5
        with pytest.raises(_Refused) as refused:
            with gate.admit():
                raise AssertionError("a sixth request must not get in")
        assert refused.value.reason == "queue_full"
        assert gate.in_flight == 5, "a refused request must not count"
    assert gate.in_flight == 0
    with gate.admit():
        pass


def test_admit_gives_the_place_back_when_the_body_raises():
    gate = _gate(max_active=1, max_queue=0)
    with pytest.raises(ValueError):
        with gate.admit():
            raise ValueError("boom")
    assert gate.in_flight == 0
    with gate.admit():
        pass


def test_max_queue_zero_means_nobody_waits():
    gate = _gate(max_active=1, max_queue=0)
    with gate.admit():
        with pytest.raises(_Refused):
            with gate.admit():
                pass


def test_the_default_refusal_is_a_gate_busy_carrying_the_reason():
    gate = ml.RequestGate(1, 0, 0.05)
    with gate.admit():
        with pytest.raises(ml.GateBusy) as refused:
            with gate.admit():
                pass
    assert refused.value.reason == "queue_full" and ml.GateBusy.retry_after_s > 0


def test_a_waiter_that_outlasts_the_queue_timeout_is_refused_and_leaks_no_turn():
    gate = _gate(max_active=1, max_queue=2, timeout=0.05)

    async def main():
        holding = asyncio.Event()
        finish = asyncio.Event()

        async def holder():
            async with gate.turn():
                holding.set()
                await finish.wait()

        first = asyncio.create_task(holder())
        await holding.wait()
        with pytest.raises(_Refused) as refused:
            async with gate.turn():
                raise AssertionError("the turn was taken while another request held it")
        assert refused.value.reason == "queue_timeout"
        assert gate.running == 1

        finish.set()
        await first
        assert gate.running == 0
        # The timed-out waiter must not have swallowed the turn on its way out.
        async with asyncio.timeout(WAIT):
            async with gate.turn():
                assert gate.running == 1

    asyncio.run(main())


def test_turns_run_at_most_max_active_at_a_time_and_every_waiter_gets_one():
    gate = _gate(max_active=2, max_queue=8, timeout=WAIT)
    seen = {"now": 0, "peak": 0, "done": 0}

    async def request():
        with gate.admit():
            async with gate.turn():
                seen["now"] += 1
                seen["peak"] = max(seen["peak"], seen["now"])
                await asyncio.sleep(0.01)
                seen["now"] -= 1
                seen["done"] += 1

    async def main():
        await asyncio.gather(*(request() for _ in range(10)))

    asyncio.run(main())
    assert seen["peak"] == 2 and seen["done"] == 10
    assert gate.in_flight == 0 and gate.running == 0


def test_a_cancelled_waiter_is_dropped_from_the_queue_without_costing_a_turn():
    gate = _gate(max_active=1, max_queue=5, timeout=WAIT)

    async def main():
        holding = asyncio.Event()
        finish = asyncio.Event()

        async def holder():
            async with gate.turn():
                holding.set()
                await finish.wait()

        async def waiter():
            with gate.admit():
                async with gate.turn():
                    raise AssertionError("cancelled waiters must never run")

        first = asyncio.create_task(holder())
        await holding.wait()
        waiters = [asyncio.create_task(waiter()) for _ in range(4)]
        await _await_until(lambda: gate.in_flight == 4)
        for task in waiters:
            task.cancel()
        await asyncio.gather(*waiters, return_exceptions=True)
        assert gate.in_flight == 0 and gate.running == 1

        finish.set()
        await first
        async with asyncio.timeout(WAIT):
            async with gate.turn():
                pass

    asyncio.run(main())


def test_a_timeout_of_zero_takes_a_free_turn_and_refuses_a_busy_gate_without_waiting():
    gate = _gate(max_active=1, max_queue=1, timeout=0)

    async def main():
        async with gate.turn():
            with pytest.raises(_Refused) as refused:
                async with gate.turn():
                    pass
            assert refused.value.reason == "queue_timeout"
        async with gate.turn():  # free again
            pass

    asyncio.run(main())


def test_snapshot_reports_the_bounds_and_where_the_queue_stands():
    gate = _gate(max_active=1, max_queue=4, timeout=60)
    with gate.admit():
        assert gate.snapshot() == {
            "max_concurrency": 1, "max_queue": 4, "queue_timeout_seconds": 60,
            "in_flight": 1, "running": 0, "waiting": 1,
        }
