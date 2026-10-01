"""Guard: no /health handler may load or pin a model.

This protects the idle-unload TTL. The frontend gateway polls every provider's
/health endpoint on a timer (frontend-service/app.py `/api/health`), so if a
health handler called get_model() or load_model(), that poll alone would keep
every model resident forever and the TTL would appear to do nothing — with no
error to explain why.

The services are correct today. This is a static check that keeps them that way,
because the failure is silent and would only show up as "TTL doesn't work".

Static source analysis rather than importing the apps: the point is to catch the
call being *written*, and this needs no torch/CUDA stubs for any of the services.
"""

import ast
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

SERVICES = [
    "stt-service",
    "qwen3-asr-service",
    "qwen3-tts-service",
    "chatterbox-tts-service",
    "magpie-tts-service",
    "piper-tts-service",
    "parakeet-asr-service",
    "canary-asr-service",
]

# Calls that either load weights or take a reference that blocks unloading.
# `acquire_lease` and `lease` are ModelSlot's lease API, the only one Magpie and
# Chatterbox use. Exact names only: a substring match would flag release().
FORBIDDEN = {"get_model", "load_model", "_acquire_model", "acquire", "acquire_async", "acquire_ref",
             "acquire_lease", "lease"}


def _health_functions(tree: ast.AST, path: str = "/health"):
    """Yield (name, node) for every function routed at a /health (or `path`) route."""
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for deco in node.decorator_list:
            if not isinstance(deco, ast.Call):
                continue
            for arg in deco.args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    if arg.value.rstrip("/").endswith(path) or arg.value == path:
                        yield node.name, node


def _called_names(node: ast.AST):
    for sub in ast.walk(node):
        if not isinstance(sub, ast.Call):
            continue
        func = sub.func
        if isinstance(func, ast.Name):
            yield func.id
        elif isinstance(func, ast.Attribute):
            yield func.attr


def _forbidden_uses(node: ast.AST) -> list:
    """The FORBIDDEN names *node* calls, or reads as an attribute: a bound method handed
    to a thread (`asyncio.to_thread(_model_slot.lease)`) is never called by the handler."""
    names = set(_called_names(node))
    names.update(sub.attr for sub in ast.walk(node) if isinstance(sub, ast.Attribute))
    return sorted(FORBIDDEN.intersection(names))


@pytest.mark.parametrize("body", [
    "lease = await _model_slot.acquire_lease()\n    await lease.release_async()",
    "async with await _model_slot.acquire_lease():\n        pass",
    "_model_slot.lease().release()",
    "await asyncio.to_thread(_model_slot.lease)",
])
def test_the_guard_catches_a_health_handler_that_takes_a_lease(body):
    """Magpie and Chatterbox pin their model only through ModelSlot.acquire_lease() and
    lease(); a guard that knew only acquire/acquire_async/acquire_ref let them through."""
    tree = ast.parse(f'@app.get("/health")\nasync def health():\n    {body}\n    return {{}}\n')
    [(_, node)] = _health_functions(tree)
    assert _forbidden_uses(node), f"the guard missed: {body!r}"


@pytest.mark.parametrize("service", SERVICES)
def test_health_handler_does_not_load_a_model(service):
    app_py = REPO / service / "app.py"
    if not app_py.exists():
        pytest.skip(f"{service} has no app.py")

    tree = ast.parse(app_py.read_text(encoding="utf-8"))
    handlers = list(_health_functions(tree))
    assert handlers, f"{service}: no /health handler found — did the route move?"

    for name, node in handlers:
        offenders = _forbidden_uses(node)
        assert not offenders, (
            f"{service}: /health handler '{name}' calls {offenders}. "
            "The gateway polls /health on a timer, so this would pin every model "
            "resident and silently defeat the idle-unload TTL."
        )


@pytest.mark.parametrize("service", SERVICES)
def test_health_handler_exists_and_is_cheap(service):
    """A health handler should not do I/O either — it runs every 30s per service."""
    app_py = REPO / service / "app.py"
    if not app_py.exists():
        pytest.skip(f"{service} has no app.py")

    tree = ast.parse(app_py.read_text(encoding="utf-8"))
    for name, node in _health_functions(tree):
        called = set(_called_names(node))
        expensive = called.intersection({"transcribe", "generate", "run", "read", "open"})
        assert not expensive, (
            f"{service}: /health handler '{name}' calls {sorted(expensive)} — "
            "health checks run every 30 seconds and must stay cheap."
        )


@pytest.mark.parametrize("service", SERVICES)
def test_ready_handler_does_not_load_a_model(service):
    """/ready may be polled on a timer like /health, so it has the same rule: a
    readiness probe that loaded the model would keep it resident for as long as
    anything polls it, and every service's /ready says it only reads state. (The
    behavioural twins are test_stt_http_endpoints.py and test_magpie_service.py;
    this catches the call being written.)"""
    app_py = REPO / service / "app.py"
    if not app_py.exists():
        pytest.skip(f"{service} has no app.py")
    tree = ast.parse(app_py.read_text(encoding="utf-8"))
    handlers = list(_health_functions(tree, "/ready"))
    assert handlers, f"{service}: no /ready handler found — did the route move?"
    for name, node in handlers:
        offenders = _forbidden_uses(node)
        assert not offenders, f"{service}: /ready handler '{name}' uses {offenders}."
