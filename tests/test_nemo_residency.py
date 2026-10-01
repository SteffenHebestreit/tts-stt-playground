"""Guards for residency in the two NeMo services (parakeet, canary).

These were the last GPU services holding their weights for the container's
lifetime — ~3 GB and ~2 GB, both opt-in and both bursty, which is the worst
possible profile for pinning a card. They now use the same `ModelSlot` as
qwen3-asr and chatterbox.

Static, because importing either app means importing `nemo` and `torch`. The
properties checked here are the ones whose failure is silent:

- a handler that calls `get_model()` on the request path takes a reference and
  hands it straight back, so the idle reaper is free to unload the weights the
  handler is about to run inference on;
- a transcription runner whose first parameter is not the model has silently
  gone back to reading the global, with the same consequence;

The unload hook's `.cpu()` and `model_loaded` reset, what `/health` reports about
residency, and skipping the preload at TTL 0 are exercised through the real
endpoints instead (test_nemo_services_uploads.py, test_nemo_services_runtime.py):
they used to be greps of the source here, which pass when the code is wrong.

`tests/test_model_lifecycle.py` covers ModelSlot's own behaviour, and
`tests/test_model_unload.py` covers the registry claim. This file covers the
wiring between them.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

NEMO_SERVICES = ["parakeet-asr-service", "canary-asr-service"]

# Runners that execute a forward pass and therefore must be handed the pinned
# model rather than fetching one themselves.
RUNNERS = ("_run_transcription", "_run_transcription_batch")


def _tree(service: str) -> ast.Module:
    return ast.parse((REPO_ROOT / service / "app.py").read_text(encoding="utf-8"))


def _functions(tree: ast.Module):
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield node


def _named(tree: ast.Module, name: str):
    return next((f for f in _functions(tree) if f.name == name), None)


def _is_route(node) -> bool:
    for deco in node.decorator_list:
        if isinstance(deco, ast.Call) and isinstance(deco.func, ast.Attribute):
            if isinstance(deco.func.value, ast.Name) and deco.func.value.id == "app":
                return True
    return False


def _called_names(node) -> set[str]:
    out = set()
    for sub in ast.walk(node):
        if not isinstance(sub, ast.Call):
            continue
        func = sub.func
        if isinstance(func, ast.Name):
            out.add(func.id)
        elif isinstance(func, ast.Attribute):
            out.add(func.attr)
    return out


@pytest.mark.parametrize("service", NEMO_SERVICES)
def test_asr_helper_pins_the_model_for_the_whole_call(service: str):
    """The reference must span the await, not just the call that takes it.

    `asyncio.to_thread` runs the forward pass on a worker thread; a reference
    released before that returns lets the reaper free weights mid-inference.
    """
    helper = _named(_tree(service), "_asr")
    assert helper is not None, f"{service}: no _asr helper — did the wrapper move?"
    body = ast.unparse(helper)
    # acquire_lease is the idempotent form of the same pin (ModelLease), for a
    # helper that has to hand the reference to something that outlives it.
    assert "acquire_async" in body or "acquire_lease" in body, (
        f"{service}: _asr does not acquire the model slot, so nothing stops the "
        f"idle reaper from unloading during a transcription"
    )
    assert "to_thread" in body, f"{service}: _asr no longer offloads the blocking call"


@pytest.mark.parametrize("service", NEMO_SERVICES)
def test_transcription_runners_take_the_pinned_model(service: str):
    """A runner that fetches its own model has drifted back to the global."""
    tree = _tree(service)
    for name in RUNNERS:
        runner = _named(tree, name)
        if runner is None:
            continue
        params = [a.arg for a in runner.args.args]
        assert params and params[0] == "model", (
            f"{service}: {name}{tuple(params)} does not take the pinned model as "
            f"its first argument, so it must be resolving one itself"
        )
        assert "get_model" not in _called_names(runner), (
            f"{service}: {name} calls get_model(), which returns a model whose "
            f"reference has already been released"
        )


@pytest.mark.parametrize("service", NEMO_SERVICES)
def test_no_route_handler_resolves_its_own_model(service: str):
    """`get_model()` releases before returning; it is for the preload only."""
    offenders = [
        node.name for node in _functions(_tree(service))
        if _is_route(node) and "get_model" in _called_names(node)
    ]
    assert not offenders, (
        f"{service}: route handlers {offenders} call get_model(). It hands the "
        f"reference back before returning, so the model can be unloaded between "
        f"that call and the inference that uses it. Go through _asr()."
    )


@pytest.mark.parametrize("service", NEMO_SERVICES)
def test_slot_is_wired_with_the_documented_ttl_names(service: str):
    source = (REPO_ROOT / service / "app.py").read_text(encoding="utf-8")
    assert 'ttl_from_env(os.getenv, "ASR_MODEL_TTL", "MODEL_TTL"' in source, (
        f"{service}: .env.example documents MODEL_TTL as the fallback for any "
        f"service without its own knob; both names must be honoured"
    )
    assert "on_unload=" in source, f"{service}: ModelSlot built without an unload hook"


# Every service that ships a copy of the ModelSlot module. The copies are found by glob,
# so a new one cannot escape the comparison and a missing one cannot pass it silently.
LIFECYCLE_SERVICES = {
    "parakeet-asr-service", "canary-asr-service", "qwen3-asr-service",
    "chatterbox-tts-service", "magpie-tts-service",
}


def test_the_lifecycle_module_is_not_forked_per_service():
    """Every copy of a concurrency primitive is one more place for it to drift."""
    copies = {
        path.parent.name: path.read_text(encoding="utf-8")
        for path in REPO_ROOT.glob("*-service/model_lifecycle.py")
    }
    assert set(copies) == LIFECYCLE_SERVICES, (
        f"model_lifecycle.py is copied into {sorted(copies)}, but this test expects exactly "
        f"{sorted(LIFECYCLE_SERVICES)}. Add a new copy to LIFECYCLE_SERVICES, or find out "
        f"where a listed one went."
    )
    # The copy tests/test_model_lifecycle.py exercises.
    reference = copies["chatterbox-tts-service"]
    diverged = sorted(name for name, text in copies.items() if text != reference)
    assert not diverged, (
        f"model_lifecycle.py differs in {diverged}. The copies exist because each "
        f"service builds its own image from its own context, not because they are "
        f"allowed to behave differently — a fix applied to one must be applied to "
        f"all."
    )
