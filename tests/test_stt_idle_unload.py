"""Tests for stt-service's idle-unload timer.

`.env.example` calls `MODEL_TTL` "the single knob that lets several multi-GB
models share one card", and every other GPU service honoured it. stt-service did
not: it had reference counting and a manual POST /unload, but no clock. So the
knob freed qwen3-asr's ~4 GB and chatterbox's ~4 GB and then stalled against
1.6-3 GB of CTranslate2 weights that nothing ever released — on a 12 GB card,
the difference between the fourth service fitting and not.

`residency.py` is the timer half, kept free of torch and faster_whisper so it can
be tested for real rather than by grepping the source. The reference counting it
drives still lives in `app.py`, next to the globals `load_model()` writes, and is
asserted statically below.
"""

from __future__ import annotations

import ast
import sys
import threading
import time
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SERVICE_DIR = REPO_ROOT / "stt-service"


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


def test_negative_ttl_never_arms(residency):
    """-1 pins the model resident; arming would defeat that."""
    fired = threading.Event()
    unloader = residency.IdleUnloader(-1, fired.set)
    unloader.arm()
    assert not unloader.armed
    assert not fired.wait(timeout=0.2)
    assert unloader.enabled is False


def test_zero_ttl_unloads_synchronously(residency):
    """0 means "give the memory back the moment nothing is using it"."""
    calls = []
    unloader = residency.IdleUnloader(0, lambda: calls.append(1))
    unloader.arm()
    assert calls == [1], "TTL=0 must not wait for a timer thread"
    assert not unloader.armed


def test_positive_ttl_fires_after_the_delay(residency):
    fired = threading.Event()
    unloader = residency.IdleUnloader(0.05, fired.set)
    unloader.arm()
    assert unloader.armed
    assert fired.wait(timeout=3), "idle timer never fired"
    assert not unloader.armed, "a fired timer must not stay armed"


def test_cancel_prevents_the_unload(residency):
    """The whole point of cancelling on acquire: an in-flight decode is not idle."""
    fired = threading.Event()
    unloader = residency.IdleUnloader(0.1, fired.set)
    unloader.arm()
    unloader.cancel()
    assert not fired.wait(timeout=0.5)
    assert not unloader.armed


def test_rearming_resets_the_clock_instead_of_stacking_timers(residency):
    """A burst of short requests must keep pushing the deadline out, not queue
    up N unloads that all fire once the burst ends."""
    calls: list[float] = []
    unloader = residency.IdleUnloader(0.15, lambda: calls.append(time.monotonic()))
    for _ in range(5):
        unloader.arm()
        time.sleep(0.02)
    assert not calls, "an unload fired while the clock was still being reset"
    time.sleep(0.5)
    assert len(calls) == 1, f"expected exactly one unload, got {len(calls)}"


def test_a_failing_unload_does_not_kill_the_timer_thread(residency):
    """The reaper runs on a daemon thread with nobody to catch its exceptions."""
    def boom():
        raise RuntimeError("CUDA is on fire")

    unloader = residency.IdleUnloader(0.01, boom)
    unloader.arm()
    time.sleep(0.3)
    assert not unloader.armed
    # Still usable afterwards.
    fired = threading.Event()
    ok = residency.IdleUnloader(0.01, fired.set)
    ok.arm()
    assert fired.wait(timeout=3)


# --- wiring in app.py --------------------------------------------------------
#
# Checked statically: app.py imports faster_whisper and torch, which the test
# environment deliberately does not install (see tests/README and conftest).


def _app_source() -> str:
    return (SERVICE_DIR / "app.py").read_text(encoding="utf-8")


def _function(name: str) -> ast.FunctionDef:
    tree = ast.parse(_app_source())
    return next(
        n for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name
    )


def test_release_arms_the_timer_and_acquire_cancels_it():
    """Without both halves the TTL either never fires or fires mid-decode."""
    assert "_idle_unloader.arm()" in ast.unparse(_function("release_model")), (
        "release_model() does not arm the idle timer, so the model is never "
        "released after the last caller lets go"
    )
    assert "_idle_unloader.cancel()" in ast.unparse(_function("acquire_model")), (
        "acquire_model() does not cancel a pending unload; a timer armed before "
        "the request could fire while the decode is running"
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
    """The deadlock this ordering exists to prevent.

    `_model_ref_lock` is not reentrant, and with STT_MODEL_TTL=0 `arm()` calls
    `unload_model()` synchronously — which takes that same lock. Separately,
    `acquire_model()` reaches into the unloader's lock, so holding the reference
    lock across any `_idle_unloader` call also inverts the pair against the
    timer thread. Neither is reachable as long as the calls stay outside.
    """
    offenders = _unloader_calls_under_the_reference_lock(_function(func_name))
    assert not offenders, (
        f"{func_name}() calls {offenders} while holding `_model_ref_lock`. "
        f"With STT_MODEL_TTL=0 that self-deadlocks on the first release; with "
        f"any TTL it can invert against the timer thread."
    )


def test_ttl_zero_skips_the_startup_preload():
    """Loading ~2 GB and warming it, only to drop it on the first arm(), is pure
    waste — TTL=0 means "do not hold this when nothing is using it"."""
    lifespan = ast.unparse(_function("_lifespan"))
    assert "MODEL_TTL == 0" in lifespan, (
        "_lifespan preloads unconditionally; with STT_MODEL_TTL=0 that loads and "
        "warms the model at boot purely to discard it"
    )


def test_health_reports_the_configured_ttl():
    """Operators need to see which residency policy a container is running."""
    assert '"model_ttl_seconds": MODEL_TTL' in _app_source()


def test_ttl_is_read_from_both_documented_names():
    source = _app_source()
    assert 'ttl_from_env(os.getenv, "STT_MODEL_TTL", "MODEL_TTL"' in source, (
        ".env.example documents MODEL_TTL as the fallback for services without "
        "their own knob; stt-service must honour both names"
    )


def test_no_decode_route_gates_on_model_loaded():
    """`model_loaded` is False after any unload, for a model that reloads on
    demand. Gating a route on it answers 503 for a service that is fine —
    /transcribe-stream did exactly that and was the only path that refused."""
    tree = ast.parse(_app_source())
    offenders: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        routed = any(
            isinstance(d, ast.Call)
            and any(isinstance(a, ast.Constant) and a.value in
                    ("/transcribe", "/transcribe-stream", "/detect_language")
                    for a in d.args)
            for d in node.decorator_list
        )
        if not routed:
            continue
        for sub in ast.walk(node):
            if isinstance(sub, ast.If) and "model_loaded" in ast.unparse(sub.test):
                offenders.append(f"{node.name}: if {ast.unparse(sub.test)}")

    assert not offenders, (
        "these decode routes still branch on the `model_loaded` global instead "
        "of letting acquire_model()/model_in_use() reload on demand: "
        f"{offenders}"
    )
