"""POST /unload: deliberate VRAM reclaim, and the registry claim about it.

The idle TTL already frees memory on its own schedule. /unload exists for the
case the TTL cannot serve: you are about to run something else on the same GPU
and need the memory back *now*.

The property that matters is that it never frees memory a running decode is
still reading. Every service therefore refuses with 409 while a reference is
outstanding, rather than unloading and hoping.

The registry claim is tested too, because a capability that lies is worse than
one that is missing — `detect_language` was advertised by two providers whose
route always returned null.
"""

from __future__ import annotations

import ast
import importlib.util
import os
import sys
import threading
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

# Services that implement POST /unload, and the capability they must declare.
UNLOAD_SERVICES = {
    "whisper": "stt-service",
    "qwen3-asr": "qwen3-asr-service",
    "qwen3": "qwen3-tts-service",
    "chatterbox": "chatterbox-tts-service",
}
UNLOAD_CAPABILITY = "model_unload"


def _routes(service_dir: str) -> set[str]:
    """Route paths declared with @app.<method>("...") in a service."""
    source = (REPO_ROOT / service_dir / "app.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    found: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for dec in node.decorator_list:
            if not isinstance(dec, ast.Call):
                continue
            func = dec.func
            if isinstance(func, ast.Attribute) and dec.args:
                arg = dec.args[0]
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    found.add(arg.value)
    return found


@pytest.mark.parametrize("provider,service_dir", sorted(UNLOAD_SERVICES.items()))
def test_service_actually_implements_unload(provider: str, service_dir: str):
    """The route must exist in the service that claims the capability."""
    assert "/unload" in _routes(service_dir), (
        f"{service_dir}/app.py declares no POST /unload, but the registry lists "
        f"{UNLOAD_CAPABILITY} for provider '{provider}'."
    )


@pytest.fixture(scope="module")
def declared_capabilities() -> dict[str, list[str]]:
    """{provider_id: capabilities} read from the registry the gateway builds.

    Built at runtime rather than parsed: PROVIDER_REGISTRY comes out of a
    function with env-driven branches, so the literal in the source is not the
    whole story. Grepping is also wrong here — the string "model_unload" appears
    in the gateway's own capability check, which would over-report.
    """
    app_path = REPO_ROOT / "frontend-service" / "app.py"
    spec = importlib.util.spec_from_file_location("frontend_app_unload_test", app_path)
    module = importlib.util.module_from_spec(spec)

    # Every optional provider must be visible, or the comparison below is
    # vacuous for whichever ones happen to be disabled — chatterbox is opt-in
    # and would silently drop out of the "declared" set while still shipping the
    # route it is being checked against.
    optional = {
        "ENABLE_WHISPER_CPP": "true",
        "ENABLE_PARAKEET_ASR": "true",
        "ENABLE_CANARY_ASR": "true",
        "ENABLE_CHATTERBOX_TTS": "true",
    }
    previous_env = {k: os.environ.get(k) for k in optional}
    previous_cwd = Path.cwd()
    sys.path.insert(0, str(app_path.parent))
    try:
        os.environ.update(optional)
        os.chdir(app_path.parent)
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(app_path.parent))
        for key, value in previous_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        os.chdir(previous_cwd)

    return {
        pid: provider.get("capabilities", [])
        for pid, provider in module.PROVIDER_REGISTRY["providers"].items()
    }


def test_registry_declares_unload_exactly_where_it_exists(declared_capabilities):
    """No provider may claim model_unload without implementing it, or vice versa."""
    declared = {
        pid for pid, caps in declared_capabilities.items()
        if UNLOAD_CAPABILITY in caps
    }
    implemented = {
        p for p, d in UNLOAD_SERVICES.items() if "/unload" in _routes(d)
    }

    assert declared == implemented, (
        f"registry declares {UNLOAD_CAPABILITY} for {sorted(declared)} but the "
        f"route exists in {sorted(implemented)}. A capability that lies is worse "
        f"than one that is absent."
    )


def test_stub_providers_do_not_claim_unload(declared_capabilities):
    """Providers with no residency management must not advertise unloading."""
    declared = declared_capabilities
    for pid in ("parakeet", "canary", "whisper-cpp", "piper"):
        if pid not in declared:
            continue
        assert UNLOAD_CAPABILITY not in declared[pid], (
            f"provider '{pid}' claims {UNLOAD_CAPABILITY} but implements no "
            f"POST /unload route"
        )


# --- ModelSlot.try_unload ----------------------------------------------------
#
# The shared primitive behind two of the four services. `unload()` returned
# False for both "in use" and "already gone"; an HTTP caller must tell those
# apart, since the first is a retryable 409 and the second is a success.


@pytest.fixture(scope="module")
def lifecycle():
    service_dir = REPO_ROOT / "chatterbox-tts-service"
    sys.path.insert(0, str(service_dir))
    try:
        spec = importlib.util.spec_from_file_location(
            "model_lifecycle_unload_test", service_dir / "model_lifecycle.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.path.remove(str(service_dir))


def test_try_unload_reports_not_resident_distinctly(lifecycle):
    slot = lifecycle.ModelSlot(lambda: object(), ttl_seconds=-1, name="test")
    result = slot.try_unload()
    assert result == {"unloaded": False, "reason": "not_resident", "refs": 0}


def test_try_unload_releases_a_resident_model(lifecycle):
    slot = lifecycle.ModelSlot(lambda: object(), ttl_seconds=-1, name="test")
    with slot.acquire():
        pass
    assert slot.resident

    result = slot.try_unload()
    assert result == {"unloaded": True, "reason": "ok", "refs": 0}
    assert not slot.resident


def test_try_unload_refuses_while_a_reference_is_held(lifecycle):
    """The safety property: never free memory a caller is still using."""
    slot = lifecycle.ModelSlot(lambda: object(), ttl_seconds=-1, name="test")

    entered = threading.Event()
    may_finish = threading.Event()
    seen: dict = {}

    def hold():
        with slot.acquire():
            entered.set()
            may_finish.wait(timeout=5)

    worker = threading.Thread(target=hold, daemon=True)
    worker.start()
    assert entered.wait(timeout=5), "worker never acquired the model"

    seen["busy"] = slot.try_unload()
    may_finish.set()
    worker.join(timeout=5)

    assert seen["busy"]["unloaded"] is False
    assert seen["busy"]["reason"] == "busy"
    assert seen["busy"]["refs"] >= 1
    assert slot.resident, "model was freed while a reference was outstanding"

    # Once the reference is gone it unloads normally.
    assert slot.try_unload()["reason"] == "ok"


def test_unload_bool_wrapper_still_matches_try_unload(lifecycle):
    """The old boolean API must keep behaving, since callers still use it."""
    slot = lifecycle.ModelSlot(lambda: object(), ttl_seconds=-1, name="test")
    assert slot.unload() is False  # not resident

    with slot.acquire():
        pass
    assert slot.unload() is True
    assert slot.unload() is False  # already gone


# --- stt-service reference counting -----------------------------------------
#
# stt-service does not use ModelSlot; it has its own counter because load_model()
# carries a fallback ladder that sets several module globals. Verified against
# the real source rather than a reimplementation.


def test_stt_transcribe_paths_never_read_the_global_model():
    """Every decode must go through the yielded reference, not the global.

    This is the actual bug /unload could have introduced: several decode paths
    read `whisper_model` at call time inside a worker thread, so an unload
    landing between the check and the read would hand them None.
    """
    source = (REPO_ROOT / "stt-service" / "app.py").read_text(encoding="utf-8")
    offenders = [
        line.strip()
        for line in source.splitlines()
        if "whisper_model.transcribe" in line and not line.strip().startswith("#")
    ]
    assert not offenders, (
        "these call sites still read the global model instead of the reference "
        f"yielded by model_in_use()/acquire_model(): {offenders}"
    )


def test_stt_unload_refuses_while_referenced():
    """unload_model() must report busy rather than dropping a referenced model."""
    source = (REPO_ROOT / "stt-service" / "app.py").read_text(encoding="utf-8")
    tree = ast.parse(source)

    func = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "unload_model"
    )
    body = ast.unparse(func)
    assert "_model_refs > 0" in body, "unload_model does not check the reference count"
    assert "'busy'" in body or '"busy"' in body, "unload_model does not report busy"


def test_stt_health_exposes_the_reference_count():
    """Callers need to know when a retry is worthwhile."""
    source = (REPO_ROOT / "stt-service" / "app.py").read_text(encoding="utf-8")
    assert '"model_refs": _model_refs' in source
