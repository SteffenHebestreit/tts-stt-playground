"""Tests for qwen3-tts's idle-unload logic.

This service cannot use the shared ModelSlot pattern, because its model identity
can change at runtime via /load_model. It therefore has a bespoke reaper, and
bespoke code with a two-line safety property is exactly what needs pinning down:

- it must never unload while a generation is in flight;
- after unloading, it must reload the model the operator SWITCHED to, not the
  environment default — otherwise an idle period silently reverts their choice.

The model stack is stubbed; these run offline.
"""

import asyncio
import threading
import time

import pytest

pytest.importorskip("numpy", reason="qwen3-tts app.py requires numpy")

from qwen3_tts_loader import BASE_06, DESIGN_17, load_app  # noqa: E402


@pytest.fixture(scope="module")
def app_mod():
    # Own copy of the module, with a writable voice library: the default,
    # /app/voices, only exists inside the container.
    return load_app()


@pytest.fixture
def clean(app_mod):
    """Reset the module-level residency state around each test."""
    saved = (app_mod.tts_model, app_mod.model_loaded, app_mod.current_model_name,
             app_mod._desired_model_name, app_mod._inflight, app_mod.MODEL_TTL,
             app_mod._last_used)
    # The primitives bind to the first loop that has to WAIT on them, and every
    # test here runs its own loop.
    app_mod._GEN_SEM = asyncio.Semaphore(1)
    app_mod._LOAD_LOCK = asyncio.Lock()
    app_mod._inflight = 0
    app_mod._active_requests = 0
    yield app_mod
    (app_mod.tts_model, app_mod.model_loaded, app_mod.current_model_name,
     app_mod._desired_model_name, app_mod._inflight, app_mod.MODEL_TTL,
     app_mod._last_used) = saved


# --- the safety property ----------------------------------------------------


def test_never_unloads_while_a_generation_is_in_flight(clean):
    """The whole point of the in-flight counter. A timestamp check alone would
    free the weights out from under a running worker thread."""
    m = clean
    m.tts_model = object()
    m.MODEL_TTL = 1.0
    m._last_used = 0.0          # long past the TTL
    m._inflight = 1

    assert m._should_unload(now=10_000.0) is False, "must not unload during generation"

    m._inflight = 0
    assert m._should_unload(now=10_000.0) is True, "must unload once idle"


def test_does_not_unload_before_the_ttl_elapses(clean):
    m = clean
    m.tts_model = object()
    m.MODEL_TTL = 300.0
    m._inflight = 0
    m._last_used = 1000.0

    assert m._should_unload(now=1100.0) is False   # only 100s idle
    assert m._should_unload(now=1300.0) is True    # exactly at the TTL


def test_ttl_negative_disables_unloading(clean):
    m = clean
    m.tts_model = object()
    m._inflight = 0
    m._last_used = 0.0
    m.MODEL_TTL = -1

    assert m._should_unload(now=10_000.0) is False


def test_ttl_zero_unloads_as_soon_as_idle(clean):
    m = clean
    m.tts_model = object()
    m._inflight = 0
    m._last_used = 1000.0
    m.MODEL_TTL = 0

    assert m._should_unload(now=1000.0) is True


def test_nothing_to_unload_when_not_resident(clean):
    m = clean
    m.tts_model = None
    m.MODEL_TTL = 1.0
    m._inflight = 0
    m._last_used = 0.0

    assert m._should_unload(now=10_000.0) is False


def test_touch_resets_the_idle_countdown(clean):
    """_touch_model must restart the countdown against the real clock.

    This is the one test that cannot inject `now`: the whole point is that
    _touch_model() stamps time.monotonic() itself, so the assertion has to use
    the same clock.

    _last_used is therefore anchored RELATIVE to that clock. It used to be set
    to 0.0 and the test assumed `time.monotonic() - 0.0 >= 300`, which is
    seconds since boot — true on any long-running machine, false on a freshly
    booted CI runner. It passed locally for months and failed on GitHub.
    """
    m = clean
    m.tts_model = object()
    m.MODEL_TTL = 300.0
    m._inflight = 0
    m._last_used = time.monotonic() - (m.MODEL_TTL + 1.0)

    assert m._should_unload() is True
    m._touch_model()
    assert m._should_unload() is False, "a fresh request must restart the countdown"


# --- unload itself ----------------------------------------------------------


def test_unload_clears_state_and_is_idempotent(clean):
    m = clean
    m.tts_model = object()
    m.model_loaded = True
    m.current_model_name = BASE_06

    m._unload_qwen3_tts()
    assert m.tts_model is None
    assert m.model_loaded is False
    assert m.current_model_name == ""

    m._unload_qwen3_tts()          # must not raise on a second call
    assert m.tts_model is None


# --- the switched-model trap ------------------------------------------------


def test_reload_restores_the_switched_model_not_the_env_default(clean, monkeypatch):
    """The trap this design exists to avoid.

    After /load_model switches to VoiceDesign, an idle unload followed by a
    reload must bring VoiceDesign back. Reloading the env default instead would
    silently revert the operator's choice with no error anywhere.
    """
    m = clean
    switched_to = DESIGN_17
    monkeypatch.setenv("QWEN3_TTS_MODEL", BASE_06)

    requested = []

    def fake_load(model_name=None):
        requested.append(model_name)
        m.tts_model = object()
        m.current_model_name = model_name or "env-default"
        return m.tts_model

    monkeypatch.setattr(m, "load_model", fake_load)

    # State after a switch, then an idle unload.
    m._desired_model_name = switched_to
    m.tts_model = None
    m.current_model_name = ""

    async def acquire():
        async with m._acquire_model() as (model, name):
            return model, name

    model, name = asyncio.run(acquire())

    assert requested == [switched_to], (
        f"reload asked for {requested}, expected the switched-to model"
    )
    assert model is not None
    assert name == switched_to


def test_reload_uses_the_env_default_when_nothing_was_switched(clean, monkeypatch):
    m = clean
    requested = []

    def fake_load(model_name=None):
        requested.append(model_name)
        m.tts_model = object()
        m.current_model_name = model_name or "env-default"
        return m.tts_model

    monkeypatch.setattr(m, "load_model", fake_load)

    m._desired_model_name = None
    m.tts_model = None
    m.current_model_name = ""

    async def acquire():
        async with m._acquire_model():
            pass

    asyncio.run(acquire())

    assert requested == [None], "with no explicit switch, load_model picks the env default"


# --- poll interval ----------------------------------------------------------


def test_reaper_tick_is_bounded(clean):
    m = clean
    for ttl, lo, hi in [(0, 1.0, 30.0), (4, 1.0, 30.0), (300, 1.0, 30.0), (100000, 1.0, 30.0)]:
        m.MODEL_TTL = ttl
        tick = m._reaper_tick_seconds()
        assert lo <= tick <= hi, f"tick {tick} out of bounds for ttl {ttl}"


# --- a request holds the model for its whole life -----------------------------
#
# `_acquire_model` used to hand back (model, name) and count nothing, so between
# obtaining the model and the first `_gen` the request was invisible to the
# reaper. /voices/save waits on the ASR service for up to a minute in exactly
# that gap. The reaper unloaded the model under it, the saved voice's
# `model_used` came out blank, and the request's reference kept the old weights
# alive next to whatever the next request reloaded.


def _resident(m, name=BASE_06):
    model = object()
    m.tts_model = model
    m.model_loaded = True
    m.current_model_name = name
    return model


def test_a_request_counts_as_in_flight_from_acquire_to_the_end_of_the_block(clean):
    m = clean
    _resident(m)

    async def scenario():
        seen = []
        async with m._acquire_model():
            seen.append(m._inflight)
        seen.append(m._inflight)
        with pytest.raises(RuntimeError):
            async with m._acquire_model():
                seen.append(m._inflight)
                raise RuntimeError("handler failed")
        seen.append(m._inflight)
        return seen

    assert asyncio.run(scenario()) == [1, 0, 1, 0]


def test_acquire_and_release_both_restart_the_idle_clock(clean):
    m = clean
    _resident(m)
    m._last_used = time.monotonic() - 10_000

    async def scenario():
        async with m._acquire_model():
            inside = time.monotonic() - m._last_used
            m._last_used = time.monotonic() - 10_000   # a long ASR wait
        return inside, time.monotonic() - m._last_used

    inside, after = asyncio.run(scenario())
    assert inside < 5, "acquiring the model did not count as use"
    assert after < 5, "the countdown must restart when the request ends, not when it began"


def test_the_reaper_leaves_the_model_alone_while_a_request_is_not_generating(clean, monkeypatch):
    """The Q1 race, end to end: real reaper loop, a request idling on a slow
    dependency for six times the TTL, then the request ends."""
    m = clean
    model = _resident(m)
    m.MODEL_TTL = 0.05
    monkeypatch.setattr(m, "_reaper_tick_seconds", lambda: 0.01)

    async def scenario():
        reaper = asyncio.create_task(m._idle_reaper())
        try:
            async with m._acquire_model() as (got, name):
                await asyncio.sleep(0.3)
                during = m.tts_model
            for _ in range(200):
                if m.tts_model is None:
                    break
                await asyncio.sleep(0.01)
            return got, name, during, m.tts_model
        finally:
            reaper.cancel()
            try:
                await reaper
            except asyncio.CancelledError:
                pass

    got, name, during, after = asyncio.run(scenario())
    assert got is model and name == BASE_06
    assert during is model, "the reaper unloaded the model under a request that held it"
    assert after is None, "once the request ended and the TTL passed, the model must go"


def test_a_cancelled_request_releases_its_hold(clean):
    m = clean
    _resident(m)

    async def scenario():
        entered = asyncio.Event()

        async def request():
            async with m._acquire_model():
                entered.set()
                await asyncio.sleep(30)

        task = asyncio.create_task(request())
        await entered.wait()
        held = m._inflight
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        return held, m._inflight

    assert asyncio.run(scenario()) == (1, 0)


# --- a generation thread keeps its hold until it actually ends -----------------


def test_a_cancelled_request_does_not_free_the_hold_of_its_running_generation(clean):
    """If the client goes away mid-generation the worker thread keeps reading the
    weights. Releasing the in-flight count (or the concurrency permit) when the
    awaiting coroutine is cancelled, as `_gen` used to, let the reaper or a model
    switch pull them out from under it."""
    m = clean
    started = threading.Event()
    release = threading.Event()

    def slow():
        started.set()
        release.wait(10)
        return "done"

    async def scenario():
        try:
            task = asyncio.create_task(m._gen(slow))
            while not started.is_set():
                await asyncio.sleep(0.005)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            held = m._inflight

            # The thread still owns the only permit: a second generation waits.
            second = asyncio.create_task(m._gen(lambda: "second"))
            await asyncio.sleep(0.05)
            second_finished_early = second.done()

            release.set()
            result = await asyncio.wait_for(second, 5)
            for _ in range(200):
                if m._inflight == 0:
                    break
                await asyncio.sleep(0.01)
            return held, second_finished_early, result, m._inflight
        finally:
            release.set()

    held, early, result, after = asyncio.run(scenario())
    assert held == 1, "the hold was released while the worker thread was still running"
    assert early is False, "the concurrency permit was released while the thread was still running"
    assert result == "second"
    assert after == 0


def test_gen_releases_its_hold_and_permit_when_the_call_raises(clean):
    m = clean

    def boom():
        raise ValueError("model exploded")

    async def scenario():
        with pytest.raises(ValueError, match="model exploded"):
            await m._gen(boom)
        inflight = m._inflight
        # The permit came back: this would hang forever otherwise.
        again = await asyncio.wait_for(m._gen(lambda: "ok"), 5)
        return inflight, again

    assert asyncio.run(scenario()) == (0, "ok")


def test_health_counts_a_request_once_even_while_its_generation_runs(clean):
    """Safety decisions go by the number of holds (a request plus its running
    generation), but the figure operators read is requests."""
    m = clean
    _resident(m)
    started, release = threading.Event(), threading.Event()

    def work():
        started.set()
        release.wait(5)

    async def scenario():
        try:
            async with m._acquire_model():
                gen = asyncio.create_task(m._gen(work))
                while not started.is_set():
                    await asyncio.sleep(0.005)
                report = await m.health()
                holds = m._inflight
                release.set()
                await gen
            return report, holds, (await m.health())["active_requests"]
        finally:
            release.set()

    report, holds, after = asyncio.run(scenario())
    assert report["active_requests"] == 1 and holds == 2
    assert after == 0


# --- POST /unload -------------------------------------------------------------


def test_unload_refuses_while_a_request_holds_the_model(clean):
    m = clean
    model = _resident(m)

    async def scenario():
        async with m._acquire_model():
            busy = await m.unload()
            still = m.tts_model
        idle = await m.unload()
        return busy, still, idle

    busy, still, idle = asyncio.run(scenario())
    import json
    assert busy.status_code == 409
    body = json.loads(busy.body)
    assert body["reason"] == "busy" and body["unloaded"] is False and body["inflight"] == 1
    assert still is model, "/unload freed a model a request was still using"
    assert idle["unloaded"] is True and m.tts_model is None


def test_unload_answers_409_at_once_while_a_load_is_running(clean):
    """A load can take minutes; /unload must not queue behind it."""
    m = clean
    model = _resident(m)

    async def scenario():
        async with m._LOAD_LOCK:
            return await asyncio.wait_for(m.unload(), 2)

    busy = asyncio.run(scenario())
    assert busy.status_code == 409
    assert m.tts_model is model


def test_unload_frees_the_weights_off_the_event_loop(clean, monkeypatch):
    """gc.collect() + empty_cache() synchronise the device; on the loop they
    would stall /health for as long as they take."""
    m = clean
    _resident(m)
    ran_on = []
    monkeypatch.setattr(m, "_unload_qwen3_tts", lambda: ran_on.append(threading.get_ident()))

    async def scenario():
        return threading.get_ident(), await m.unload()

    loop_thread, result = asyncio.run(scenario())
    assert ran_on and ran_on[0] != loop_thread
    assert result["reason"] == "ok"


def test_unload_reports_not_resident_distinctly(clean):
    m = clean
    m.tts_model = None
    result = asyncio.run(m.unload())
    assert result == {"unloaded": False, "reason": "not_resident",
                      "inflight": 0, "model_resident": False}


# --- POST /load_model ----------------------------------------------------------


def _fake_load(m, loads):
    def fake(model_name=None):
        loads.append(model_name)
        m.tts_model = object()
        m.model_loaded = True
        m.current_model_name = model_name
        return m.tts_model
    return fake


def test_switching_waits_for_requests_that_hold_the_current_model(clean, monkeypatch):
    """A holder that outlives the swap keeps the old weights resident next to the
    new ones, and would go on to use a model the service no longer reports."""
    m = clean
    loads = []
    monkeypatch.setattr(m, "load_model", _fake_load(m, loads))
    _resident(m)

    async def scenario():
        async with m._acquire_model() as (_, held_name):
            switch = asyncio.create_task(m.switch_model(m.LoadModelRequest(model=DESIGN_17)))
            await asyncio.sleep(0.2)
            loads_while_held = list(loads)
            # A request arriving mid-switch queues behind it and gets the NEW model.
            late = asyncio.create_task(_name_of(m))
            await asyncio.sleep(0.05)
            late_done_early = late.done()
        result = await asyncio.wait_for(switch, 5)
        return held_name, loads_while_held, late_done_early, await asyncio.wait_for(late, 5), result

    held_name, during, late_early, late_name, result = asyncio.run(scenario())
    assert held_name == BASE_06
    assert during == [], "the model was swapped while a request still held the old one"
    assert late_early is False
    assert late_name == DESIGN_17
    assert loads == [DESIGN_17]
    assert result["model"] == DESIGN_17 and m._desired_model_name == DESIGN_17


async def _name_of(m):
    async with m._acquire_model() as (_, name):
        return name


def test_switching_gives_up_with_409_when_the_holder_never_finishes(clean, monkeypatch):
    m = clean
    loads = []
    monkeypatch.setattr(m, "load_model", _fake_load(m, loads))
    monkeypatch.setattr(m, "_SWITCH_DRAIN_TIMEOUT", 0.1)
    model = _resident(m)

    async def scenario():
        async with m._acquire_model():
            with pytest.raises(m.HTTPException) as exc:
                await m.switch_model(m.LoadModelRequest(model=DESIGN_17))
            return exc.value

    error = asyncio.run(scenario())
    assert error.status_code == 409, "a busy model is a retryable 409, not a 500"
    assert loads == [] and m.tts_model is model


def test_a_failed_switch_is_a_500_and_leaves_the_desired_model_alone(clean, monkeypatch):
    m = clean
    m._desired_model_name = BASE_06
    _resident(m)

    def broken(model_name=None):
        raise OSError("download failed")

    monkeypatch.setattr(m, "load_model", broken)
    with pytest.raises(m.HTTPException) as exc:
        asyncio.run(m.switch_model(m.LoadModelRequest(model=DESIGN_17)))
    assert exc.value.status_code == 500 and "download failed" in exc.value.detail
    assert m._desired_model_name == BASE_06
