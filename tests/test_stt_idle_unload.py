"""Tests for stt-service's idle-unload timer.

`.env.example` calls `MODEL_TTL` "the single knob that lets several multi-GB
models share one card", and every other GPU service honoured it. stt-service did
not: it had reference counting and a manual POST /unload, but no clock. So the
knob freed qwen3-asr's ~4 GB and chatterbox's ~4 GB and then stalled against
1.6-3 GB of CTranslate2 weights that nothing ever released — on a 12 GB card,
the difference between the fourth service fitting and not.

`residency.py` is the timer half, kept free of torch and faster_whisper so it can
be tested for real. Its timers are injected, so nothing here waits on a wall
clock: a test fires a timer by hand, which cannot flake on a loaded machine. The
reference counting it drives lives in `app.py`, next to the globals `load_model()`
writes, and is exercised through the real app under stubs.
"""

from __future__ import annotations

import ast
import sys
import threading
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from test_stt_support import (
    FakeWhisper, Segment, load_stt_app, reset_model, wait_for_preload, wait_until,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SERVICE_DIR = REPO_ROOT / "stt-service"


class ManualTimer:
    """A threading.Timer that only fires when the test says so."""

    created: list["ManualTimer"] = []

    def __init__(self, interval, function, args=None, kwargs=None):
        self.interval = interval
        self.function = function
        self.args = tuple(args or ())
        self.kwargs = dict(kwargs or {})
        self.daemon = False
        self.started = False
        self.cancelled = False
        ManualTimer.created.append(self)

    def start(self):
        self.started = True

    def cancel(self):
        self.cancelled = True

    def fire(self):
        """The delay elapsed. A cancelled timer never fires, as with the real one."""
        if not self.cancelled:
            self.function(*self.args, **self.kwargs)

    def fire_despite_cancel(self):
        """The timer thread was already past cancel()'s reach when it was called."""
        self.function(*self.args, **self.kwargs)


@pytest.fixture(autouse=True)
def _fresh_timers():
    ManualTimer.created = []


@pytest.fixture(scope="module")
def residency():
    sys.path.insert(0, str(SERVICE_DIR))
    try:
        spec = spec_from_file_location(
            "stt_residency_under_test", SERVICE_DIR / "residency.py"
        )
        module = module_from_spec(spec)
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.path.remove(str(SERVICE_DIR))


# --- TTL parsing -------------------------------------------------------------


def test_ttl_prefers_the_service_specific_name(residency):
    env = {"STT_MODEL_TTL": "60", "MODEL_TTL": "300"}
    assert residency.ttl_from_env(lambda k, d="": env.get(k, d),
                                  "STT_MODEL_TTL", "MODEL_TTL") == 60.0


def test_ttl_falls_back_to_the_global_knob(residency):
    """The documented contract: MODEL_TTL applies where nothing more specific is set."""
    env = {"MODEL_TTL": "120"}
    assert residency.ttl_from_env(lambda k, d="": env.get(k, d),
                                  "STT_MODEL_TTL", "MODEL_TTL") == 120.0


def test_ttl_ignores_an_unparseable_value_rather_than_crashing(residency):
    env = {"STT_MODEL_TTL": "five minutes"}
    assert residency.ttl_from_env(lambda k, d="": env.get(k, d),
                                  "STT_MODEL_TTL", "MODEL_TTL", default=7.0) == 7.0


def test_ttl_sentinels_survive_parsing(residency):
    """-1 (never) and 0 (immediately) must not be mistaken for "unset"."""
    for raw, expected in (("-1", -1.0), ("0", 0.0)):
        env = {"STT_MODEL_TTL": raw}
        assert residency.ttl_from_env(lambda k, d="": env.get(k, d),
                                      "STT_MODEL_TTL", default=300.0) == expected


# --- IdleUnloader ------------------------------------------------------------


def _unloader(residency, ttl, on_expire):
    return residency.IdleUnloader(ttl, on_expire, timer_factory=ManualTimer)


def test_negative_ttl_never_arms(residency):
    """-1 pins the model resident; arming would defeat that."""
    calls = []
    unloader = _unloader(residency, -1, lambda: calls.append(1))
    unloader.arm()
    assert not unloader.armed
    assert ManualTimer.created == [], "a timer was created for a TTL that means 'never unload'"
    assert unloader.enabled is False


def test_zero_ttl_unloads_synchronously(residency):
    """0 means "give the memory back the moment nothing is using it"."""
    calls = []
    unloader = _unloader(residency, 0, lambda: calls.append(1))
    unloader.arm()
    assert calls == [1], "TTL=0 must not wait for a timer thread"
    assert not unloader.armed
    assert ManualTimer.created == []


def test_positive_ttl_fires_after_the_delay(residency):
    calls = []
    unloader = _unloader(residency, 300, lambda: calls.append(1))
    unloader.arm()
    timer = ManualTimer.created[-1]
    assert unloader.armed and timer.started and timer.daemon
    assert timer.interval == 300
    assert calls == []

    timer.fire()
    assert calls == [1]
    assert not unloader.armed, "a fired timer must not stay armed"


def test_the_default_timer_is_a_real_daemon_timer(residency):
    """One smoke test with a real threading.Timer, waiting on an Event rather than a clock."""
    fired = threading.Event()
    unloader = residency.IdleUnloader(0.02, fired.set)
    unloader.arm()
    assert fired.wait(timeout=10), "the idle timer never fired"
    assert not unloader.armed


def test_cancel_prevents_the_unload(residency):
    """The whole point of cancelling on acquire: an in-flight decode is not idle."""
    calls = []
    unloader = _unloader(residency, 300, lambda: calls.append(1))
    unloader.arm()
    timer = ManualTimer.created[-1]
    unloader.cancel()
    assert timer.cancelled and not unloader.armed
    timer.fire()
    assert calls == []


def test_a_timer_already_running_when_cancelled_does_not_unload(residency):
    """cancel() cannot stop a Timer thread that has left its wait: it goes on to
    call the callback. That callback must find out it was superseded."""
    calls = []
    unloader = _unloader(residency, 300, lambda: calls.append(1))
    unloader.arm()
    timer = ManualTimer.created[-1]
    unloader.cancel()
    timer.fire_despite_cancel()
    assert calls == [], "a cancelled timer unloaded the model of a request that had just arrived"


def test_rearming_resets_the_clock_instead_of_stacking_timers(residency):
    """A burst of short requests must keep pushing the deadline out, not queue
    up N unloads that all fire once the burst ends."""
    calls = []
    unloader = _unloader(residency, 300, lambda: calls.append(1))
    for _ in range(5):
        unloader.arm()
    timers = list(ManualTimer.created)
    assert len(timers) == 5
    assert [t.cancelled for t in timers] == [True, True, True, True, False], (
        "only the newest timer may stay live"
    )

    for stale in timers[:-1]:                 # even ones that were past cancel()
        stale.fire_despite_cancel()
    assert calls == [], "a superseded timer unloaded the model"
    assert unloader.armed, "a superseded timer dropped the handle of the live one"

    timers[-1].fire()
    assert calls == [1], f"expected exactly one unload, got {len(calls)}"


def test_a_superseded_timer_cannot_hide_its_successor_from_cancel(residency):
    """It used to clear the shared handle, after which cancel() could no longer
    reach the live timer and it fired in the middle of the next request."""
    calls = []
    unloader = _unloader(residency, 300, lambda: calls.append(1))
    unloader.arm()
    first = ManualTimer.created[-1]
    unloader.arm()
    second = ManualTimer.created[-1]

    first.fire_despite_cancel()
    unloader.cancel()

    assert second.cancelled, "cancel() no longer reaches the live timer"
    assert not unloader.armed


def test_a_failing_unload_does_not_kill_the_timer_thread(residency):
    """The reaper runs on a daemon thread with nobody to catch its exceptions."""
    raised = []

    def boom():
        raised.append(1)
        raise RuntimeError("CUDA is on fire")

    unloader = _unloader(residency, 300, boom)
    unloader.arm()
    ManualTimer.created[-1].fire()           # must swallow the error, not propagate it
    assert raised == [1]
    assert not unloader.armed

    # Still usable afterwards.
    ok_calls = []
    ok = _unloader(residency, 300, lambda: ok_calls.append(1))
    ok.arm()
    ManualTimer.created[-1].fire()
    assert ok_calls == [1]


# --- wiring in app.py --------------------------------------------------------
#
# The real app, loaded under stubs (test_stt_support.py), with a fake model and a
# recording stand-in for the unloader.


class RecordingUnloader:
    def __init__(self):
        self.events: list[str] = []

    def arm(self):
        self.events.append("arm")

    def cancel(self):
        self.events.append("cancel")


@pytest.fixture(scope="module")
def stt_app():
    return load_stt_app({"STT_MODEL_TTL": "300"}, name="stt_app_idle_unload")


@pytest.fixture
def app(stt_app, monkeypatch):
    stt_app._idle_unloader.cancel()
    reset_model(stt_app)
    stt_app._model_refs = 0
    recorder = RecordingUnloader()
    monkeypatch.setattr(stt_app, "_idle_unloader", recorder)
    stt_app.recorder = recorder
    return stt_app


def test_release_arms_the_timer_at_zero_references_and_acquire_cancels_it(app):
    """Without both halves the TTL either never fires or fires mid-decode."""
    first = app.acquire_model()
    second = app.acquire_model()
    assert app.recorder.events == ["cancel", "cancel"], (
        "acquire must cancel a pending unload, or it can fire while the decode is running"
    )
    assert first is second is app.whisper_model

    app.release_model()
    assert "arm" not in app.recorder.events, "the timer was armed while a decode still held the model"
    app.release_model()
    assert app.recorder.events[-1] == "arm", (
        "the last release did not arm the idle timer, so the model is never released"
    )


def test_a_surplus_release_cannot_drive_the_count_negative(app):
    app.release_model()
    assert app._model_refs == 0


def _function(name: str) -> ast.FunctionDef:
    tree = ast.parse((SERVICE_DIR / "app.py").read_text(encoding="utf-8"))
    return next(
        n for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name
    )


def _unloader_calls_under_the_reference_lock(func: ast.AST) -> list[str]:
    """Names of `_idle_unloader.*` calls nested inside `with _model_ref_lock:`."""
    found: list[str] = []
    for node in ast.walk(func):
        if not isinstance(node, ast.With):
            continue
        if not any("_model_ref_lock" in ast.unparse(item.context_expr)
                   for item in node.items):
            continue
        for sub in ast.walk(node):
            if (isinstance(sub, ast.Call)
                    and isinstance(sub.func, ast.Attribute)
                    and isinstance(sub.func.value, ast.Name)
                    and sub.func.value.id == "_idle_unloader"):
                found.append(f"_idle_unloader.{sub.func.attr}()")
    return found


@pytest.mark.parametrize("func_name", ["release_model", "unload_model", "acquire_model"])
def test_the_two_locks_are_never_held_at_once(func_name: str):
    """The lock order that keeps the reference lock and the unloader's lock from
    inverting against the timer thread.

    Structural on purpose: the inversion only bites when the timer thread and a
    request collide, which no test can provoke on demand. `_model_ref_lock` is
    not reentrant, and `acquire_model()` reaches into the unloader's lock, so
    holding the reference lock across any `_idle_unloader` call can deadlock
    against `_fire()` -> `unload_model()`.
    """
    offenders = _unloader_calls_under_the_reference_lock(_function(func_name))
    assert not offenders, (
        f"{func_name}() calls {offenders} while holding `_model_ref_lock`."
    )


def test_ttl_zero_does_not_self_deadlock_on_release():
    """With STT_MODEL_TTL=0 the last release unloads synchronously, through the
    reference lock the release has just let go of. Arming inside that lock would
    hang here forever."""
    app = load_stt_app({"STT_MODEL_TTL": "0"}, name="stt_app_idle_unload_ttl0")
    assert app.MODEL_TTL == 0
    reset_model(app)

    done = threading.Event()

    def use_and_release():
        app.acquire_model()
        app.release_model()
        done.set()

    worker = threading.Thread(target=use_and_release, daemon=True)
    worker.start()
    assert done.wait(5), "release_model() deadlocked with STT_MODEL_TTL=0"
    assert app.whisper_model is None, "TTL=0 did not release the model as soon as it fell idle"


def test_ttl_zero_skips_the_startup_preload():
    """Loading ~2 GB and warming it, only to drop it on the first arm(), is pure
    waste: TTL=0 means "do not hold this when nothing is using it"."""
    constructed = []

    class Counting(FakeWhisper):
        def __init__(self, *args, **kwargs):
            constructed.append(1)
            super().__init__(*args, **kwargs)

    app = load_stt_app({"STT_MODEL_TTL": "0"}, name="stt_app_idle_unload_preload", model_cls=Counting)
    with TestClient(app.app):
        wait_for_preload(app)
        assert constructed == [], "the model was loaded at boot only to be thrown away"
        assert app.whisper_model is None


def test_a_positive_ttl_preloads_and_starts_the_idle_clock():
    constructed = []

    class Counting(FakeWhisper):
        def __init__(self, *args, **kwargs):
            constructed.append(1)
            super().__init__(*args, **kwargs)

    app = load_stt_app({"STT_MODEL_TTL": "120"}, name="stt_app_idle_unload_clock", model_cls=Counting)
    app._idle_unloader = app.IdleUnloader(120, lambda: app.unload_model(), timer_factory=ManualTimer)
    with TestClient(app.app):
        wait_for_preload(app)
        assert constructed == [1]
        assert wait_until(lambda: app._idle_unloader.armed), (
            "a container that is booted and never used must still start its idle clock"
        )
        ManualTimer.created[-1].fire()
        assert app.whisper_model is None, "the idle timer did not release the model"


def test_health_reports_the_configured_ttl():
    """Operators need to see which residency policy a container is running."""
    app = load_stt_app({"STT_MODEL_TTL": "42"}, name="stt_app_idle_unload_health")
    with TestClient(app.app) as client:
        wait_for_preload(app)
        assert client.get("/health").json()["model_ttl_seconds"] == 42


def test_ttl_is_read_from_both_documented_names():
    """.env.example documents MODEL_TTL as the fallback for services without their
    own knob; stt-service must honour both names, the specific one first."""
    assert load_stt_app({"MODEL_TTL": "120"}, name="stt_app_ttl_global").MODEL_TTL == 120
    assert load_stt_app({"STT_MODEL_TTL": "60", "MODEL_TTL": "120"}, name="stt_app_ttl_both").MODEL_TTL == 60
    assert load_stt_app({}, name="stt_app_ttl_default").MODEL_TTL == 300


def test_no_decode_route_refuses_a_model_that_was_merely_unloaded():
    """`model_loaded` is False after any unload, for a model that reloads on
    demand. Gating a route on it answers 503 for a service that is fine —
    /transcribe-stream did exactly that and was the only path that refused."""
    app = load_stt_app({}, name="stt_app_idle_unload_routes")
    wav = b"RIFF" + b"\x00" * 60
    with TestClient(app.app) as client:
        wait_for_preload(app)
        reset_model(app).default = [Segment("hallo", start=0.0, end=1.0)]
        assert client.post("/unload").json()["unloaded"] is True
        assert app.whisper_model is None and app.model_loaded is False

        assert client.post("/transcribe", files={"audio": ("a.wav", wav, "audio/wav")}).status_code == 200
        assert app.whisper_model is not None                # reloaded on demand
        app.unload_model()
        assert client.post(
            "/transcribe-stream", files={"audio": ("a.wav", wav, "audio/wav")}).status_code == 200
        app.unload_model()
        assert client.post(
            "/detect_language", files={"file": ("a.wav", wav, "audio/wav")}).status_code == 200


def test_unload_refuses_while_a_decode_holds_the_model():
    app = load_stt_app({}, name="stt_app_idle_unload_busy")
    with TestClient(app.app) as client:
        wait_for_preload(app)
        reset_model(app)
        app.acquire_model()
        try:
            response = client.post("/unload")
            assert response.status_code == 409
            assert response.json()["model_refs"] == 1
            assert app.whisper_model is not None, "the model was pulled out from under a running decode"
        finally:
            app.release_model()
