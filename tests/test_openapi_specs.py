"""Guards for the published OpenAPI specs.

Three files per service used to exist by hand: ``<service>/openapi.json``, the
copy under ``frontend-service/static/openapi/`` that the API documentation page
links to, and the code. They disagreed. The training spec documented
``/process_audio`` and ``/training_status/{session_id}``, which do not exist
(and carried version 2.0.0 in one copy and 2.1.0 in the other); the STT spec was
missing ``/unload`` and the whole readiness endpoint; the copies of the other two
were only identical by luck.

``scripts/sync_openapi.py`` now writes every file from the running app. These
tests hold the committed files to what the apps actually serve.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "sync_openapi.py"
STATIC = REPO_ROOT / "frontend-service" / "static" / "openapi"
HTTP_METHODS = {"get", "put", "post", "delete", "options", "head", "patch", "trace"}


def _load_script():
    spec = importlib.util.spec_from_file_location("sync_openapi_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve their annotations through it
    spec.loader.exec_module(module)
    return module


sync_openapi = _load_script()
SERVICES = sync_openapi.SERVICES
IDS = [s.key for s in SERVICES]

_APP_CACHE: dict = {}


def _app(service):
    if service.key not in _APP_CACHE:
        _APP_CACHE[service.key] = sync_openapi.load_app(service)
    return _APP_CACHE[service.key]


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _operations(spec: dict) -> set[tuple[str, str]]:
    return {(path, method.upper()) for path, item in spec["paths"].items()
            for method in item if method in HTTP_METHODS}


def _resolves(app, method: str, path: str) -> bool:
    """Would the app's router dispatch this request to a handler (path AND method)?

    Asks Starlette's own matcher instead of walking ``app.routes``: FastAPI keeps an
    included router as a single lazy entry, so a walk misses everything under
    ``include_router`` (all of ``/v1``).
    """
    from starlette.routing import Match

    concrete = re.sub(r"\{[^}]+\}", "x", path)
    scope = {"type": "http", "method": method, "path": concrete, "root_path": "",
             "headers": [], "query_string": b"", "scheme": "http", "server": ("test", 80)}
    return any(route.matches(scope)[0] == Match.FULL for route in app.router.routes)


# --- the copies -------------------------------------------------------------------------------


@pytest.mark.parametrize("service", [s for s in SERVICES if s.service_copy], ids=lambda s: s.key)
def test_the_gateway_copy_is_byte_identical_to_the_service_spec(service):
    """The docs page serves the static copy; it must be the file next to the code."""
    assert service.static_path.read_bytes() == service.spec_path.read_bytes(), (
        f"{service.static_path.relative_to(REPO_ROOT)} differs from "
        f"{service.spec_path.relative_to(REPO_ROOT)}. Run `python scripts/sync_openapi.py`."
    )


def test_every_spec_in_the_static_directory_is_generated():
    """A hand-edited file next to the generated ones would quietly drift again."""
    on_disk = {p.name for p in STATIC.glob("*.json")}
    generated = {s.static_name for s in SERVICES}
    assert on_disk == generated, (
        f"unmanaged: {sorted(on_disk - generated)}, missing: {sorted(generated - on_disk)}. "
        "Add a Service entry to scripts/sync_openapi.py or delete the file."
    )


def test_the_docs_page_only_links_specs_that_exist():
    html = (REPO_ROOT / "frontend-service" / "static" / "api_docs.html").read_text(encoding="utf-8")
    for name in re.findall(r'/static/openapi/([A-Za-z0-9_.-]+\.json)', html):
        assert (STATIC / name).is_file(), f"api_docs.html links /static/openapi/{name}, which does not exist"


def test_the_docs_page_links_every_published_spec():
    """The gateway's own spec (the /v1 and /api surface) was generated and served
    but never linked, so the one API a client outside the network uses had no
    entry on the page that documents the APIs."""
    html = (REPO_ROOT / "frontend-service" / "static" / "api_docs.html").read_text(encoding="utf-8")
    linked = set(re.findall(r'href="/static/openapi/([A-Za-z0-9_.-]+\.json)"', html))
    published = {p.name for p in STATIC.glob("*.json")}
    assert published - linked == set(), f"published but not linked from api_docs.html: {sorted(published - linked)}"


# --- the specs against the apps ------------------------------------------------------------------


@pytest.mark.parametrize("service", SERVICES, ids=IDS)
def test_documented_operations_are_real_routes(service):
    """Every path+method in the published spec is answered by the app."""
    app = _app(service)
    ghosts = sorted(op for op in _operations(_read(service.static_path)) if not _resolves(app, op[1], op[0]))
    assert not ghosts, (
        f"{service.static_path.name} documents operations the app does not serve: {ghosts}"
    )


def test_the_route_check_can_tell_a_missing_route_from_a_present_one():
    """Guards the guard: a matcher that says yes to everything would pass the test above."""
    app = _app(next(s for s in SERVICES if s.key == "training"))
    assert _resolves(app, "GET", "/status/{job_id}")
    assert not _resolves(app, "GET", "/training_status/{session_id}")
    assert not _resolves(app, "POST", "/status/{job_id}"), "wrong method must not match"
    gateway = _app(next(s for s in SERVICES if s.key == "gateway"))
    assert _resolves(gateway, "POST", "/v1/audio/speech")


@pytest.mark.parametrize("service", SERVICES, ids=IDS)
def test_committed_files_are_what_the_app_generates_today(service):
    """Catches a changed model, parameter, default or docstring, not just a new route."""
    fresh = sync_openapi.render(service, _app(service))
    for path in service.outputs:
        assert path.read_text(encoding="utf-8") == fresh, (
            f"{path.relative_to(REPO_ROOT)} is stale. Run `python scripts/sync_openapi.py`."
        )


def test_the_endpoints_that_were_wrong_are_right_now():
    training = _operations(_read(STATIC / "training.json"))
    assert ("/process_audio", "POST") not in training
    assert ("/training_status/{session_id}", "GET") not in training
    assert ("/status/{job_id}", "GET") in training and ("/ready", "GET") in training

    stt = _operations(_read(STATIC / "stt.json"))
    assert {("/unload", "POST"), ("/ready", "GET"), ("/transcribe", "POST")} <= stt

    gateway = _operations(_read(STATIC / "gateway.json"))
    assert {("/v1/audio/transcriptions", "POST"), ("/v1/audio/speech", "POST"),
            ("/api/stt", "POST"), ("/api/health", "GET")} <= gateway


@pytest.mark.parametrize("service", SERVICES, ids=IDS)
def test_specs_carry_the_published_port(service):
    spec = _read(service.static_path)
    assert spec["openapi"].startswith("3."), spec["openapi"]
    assert spec["servers"][0]["url"] == f"http://localhost:{service.port}"


def test_loading_an_app_leaves_the_interpreter_as_it_found_it():
    """The generator imports service modules under stand-ins; none may leak into the suite."""
    import os

    before_modules = set(sys.modules)
    before_env = dict(os.environ)
    before_cwd, before_path = os.getcwd(), list(sys.path)

    sync_openapi.load_app(SERVICES[0])

    assert os.getcwd() == before_cwd and sys.path == before_path
    assert dict(os.environ) == before_env
    leaked = {m for m in set(sys.modules) - before_modules
              if m in {"torch", "faster_whisper", "app", "residency", "json_utils", "local_agreement"}}
    assert not leaked, f"modules left behind: {sorted(leaked)}"
