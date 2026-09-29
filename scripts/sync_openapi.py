#!/usr/bin/env python3
"""Regenerate the services' OpenAPI specs from their FastAPI apps.

Three specs are published: ``<service>/openapi.json`` (next to the code) and a
copy under ``frontend-service/static/openapi/`` that the gateway serves from its
API documentation page. Both used to be written by hand. They drifted: the
training spec described ``/process_audio`` and ``/training_status/{session_id}``,
which do not exist, and the STT spec had no ``/unload``. The apps are the source of
truth, so this imports each one and writes what FastAPI derives from its routes.

The apps import torch, faster-whisper, librosa and friends at module level. None
of that is needed to *describe* the routes, so the heavy modules that are not
installed are replaced by inert stand-ins while the app is imported (the same
approach the unit tests use). A service whose import still fails is skipped with
the reason, not silently.

    python scripts/sync_openapi.py            # rewrite the spec files
    python scripts/sync_openapi.py --check    # exit 1 if any file is out of date
    python scripts/sync_openapi.py stt        # only one service

Run it whenever a route, a request or response model, or an endpoint's docstring
changes. tests/test_openapi_specs.py fails when the committed files are stale.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import json
import os
import sys
import tempfile
import types
from dataclasses import dataclass, field
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Callable, Iterator, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = REPO_ROOT / "frontend-service" / "static" / "openapi"


# --- stand-ins for modules the apps import but the schema does not need -------------------


class _Anything:
    """Accepts any construction, call, attribute or context use; does nothing."""

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return _Anything()

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return _Anything()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __iter__(self):
        return iter(())


class _Inert(types.ModuleType):
    """A module whose every attribute is a class that can be subclassed and called."""

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return _Anything


def _inert(name: str) -> types.ModuleType:
    module = _Inert(name)
    module.__path__ = []  # lets `import phonemizer.backend` resolve to a child
    return module


def _torch() -> types.ModuleType:
    torch = types.ModuleType("torch")
    torch.__version__ = "0.0.0-openapi"
    torch.cuda = types.SimpleNamespace(
        is_available=lambda: False, device_count=lambda: 0,
        get_device_name=lambda i: "none", get_device_capability=lambda i: (0, 0),
    )
    torch.backends = types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: False))
    return torch


def _faster_whisper() -> types.ModuleType:
    module = types.ModuleType("faster_whisper")
    module.WhisperModel = _Anything
    module.BatchedInferencePipeline = _Anything
    return module


def _training_trio() -> dict[str, types.ModuleType]:
    """The torch-bound modules of the training service, reduced to the names app.py uses."""

    class TrainingCancelled(Exception):
        pass

    pipeline = types.ModuleType("training_pipeline")
    pipeline.OptimizedTrainingPipeline = lambda: _Anything()
    pipeline.TrainingCancelled = TrainingCancelled
    exporter = types.ModuleType("model_exporter")
    exporter.ModelExporter = lambda: _Anything()
    return {"training_pipeline": pipeline, "model_exporter": exporter,
            "training_utils": types.ModuleType("training_utils")}


# --- what to publish ---------------------------------------------------------------------------


@dataclass
class Service:
    key: str                       # CLI name
    directory: str                 # service directory in the repo
    static_name: str               # file name under frontend-service/static/openapi/
    port: int                      # default host port, for the `servers` entry
    stubs: Callable[[], dict[str, types.ModuleType]] = lambda: {}
    env: dict[str, str] = field(default_factory=dict)
    # only stand in for these when the real package cannot be imported
    optional_stubs: tuple[str, ...] = ()
    # the gateway opens templates/ and static/ relative to its own directory
    chdir_to_service: bool = False
    # the gateway's spec is served from static/ only; a second copy would be shipped in the image
    service_copy: bool = True

    @property
    def spec_path(self) -> Path:
        return REPO_ROOT / self.directory / "openapi.json"

    @property
    def outputs(self) -> list[Path]:
        return ([self.spec_path] if self.service_copy else []) + [self.static_path]

    @property
    def static_path(self) -> Path:
        return STATIC_DIR / self.static_name


SERVICES = [
    Service(
        "stt", "stt-service", "stt.json", 5001,
        stubs=lambda: {"torch": _torch(), "faster_whisper": _faster_whisper()},
        env={"USE_CUDA": "false", "WHISPER_MODEL_SIZE": "tiny"},
    ),
    Service(
        "pipertts", "piper-tts-service", "pipertts.json", 5000,
        optional_stubs=("librosa", "soundfile"),
    ),
    Service(
        # The gateway's own surface: /v1 (OpenAI compatible) and the UI's /api. Its
        # dependencies (fastapi, httpx, jinja2) are the ones the tests install.
        "gateway", "frontend-service", "gateway.json", 3000,
        chdir_to_service=True, service_copy=False,
    ),
    Service(
        "training", "piper-training-service", "training.json", 8080,
        stubs=lambda: {
            **_training_trio(),
            **{name: _inert(name) for name in (
                "librosa", "soundfile", "aiofiles", "aiohttp", "phonemizer", "phonemizer.backend")},
        },
        env={"DEFAULT_DEPLOYMENT_TARGET": "none"},
    ),
]


@contextlib.contextmanager
def _isolated(service: Service) -> Iterator[None]:
    """Import state, environment and cwd restored afterwards, whatever the app did."""
    service_dir = REPO_ROOT / service.directory
    local = {p.stem for p in service_dir.glob("*.py")}
    stub_modules = service.stubs()
    for name in service.optional_stubs:
        try:
            importlib.import_module(name)
        except ImportError:
            stub_modules[name] = _inert(name)

    touched = local | set(stub_modules)
    saved_modules = {name: sys.modules.get(name) for name in touched}
    saved_env = {key: os.environ.get(key) for key in service.env}
    saved_path = list(sys.path)
    saved_cwd = os.getcwd()
    scratch = tempfile.TemporaryDirectory()
    try:
        for name in touched:
            sys.modules.pop(name, None)
        sys.modules.update(stub_modules)
        os.environ.update(service.env)
        sys.path.insert(0, str(service_dir))
        # nothing may be created next to the code, except where the app must find its files
        os.chdir(service_dir if service.chdir_to_service else scratch.name)
        yield
    finally:
        os.chdir(saved_cwd)
        sys.path[:] = saved_path
        for key, value in saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        for name in touched:
            sys.modules.pop(name, None)
            if saved_modules[name] is not None:
                sys.modules[name] = saved_modules[name]
        scratch.cleanup()


def load_app(service: Service):
    """Import the service's FastAPI app under stand-ins; returns the ``app`` object."""
    with _isolated(service):
        spec = spec_from_file_location(f"_openapi_{service.key}", REPO_ROOT / service.directory / "app.py")
        module = module_from_spec(spec)
        sys.modules[spec.name] = module
        try:
            spec.loader.exec_module(module)
        finally:
            sys.modules.pop(spec.name, None)
        return module.app


def render(service: Service, app=None) -> str:
    """The spec file text for *service*: deterministic, with a `servers` entry added."""
    app = app or load_app(service)
    schema = json.loads(json.dumps(app.openapi()))  # detach from the app's cached dict
    if service.key == "gateway":
        note = "The web UI and API gateway (FRONTEND_PORT); the only port meant to be reachable over the network."
    else:
        note = ("Published host port. Bound to 127.0.0.1 unless BACKEND_BIND_ADDR is changed; "
                "the gateway on :3000 proxies the routes the UI needs.")
    schema["servers"] = [{"url": f"http://localhost:{service.port}", "description": note}]
    return json.dumps(schema, indent=2, ensure_ascii=False) + "\n"


def sync(service: Service, check: bool = False, app=None) -> list[Path]:
    """Write (or, with *check*, only compare) both files; returns the ones that differ."""
    text = render(service, app)
    stale = []
    for path in service.outputs:
        current = path.read_text(encoding="utf-8") if path.exists() else None
        if current != text:
            stale.append(path)
            if not check:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text, encoding="utf-8")
    return stale


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("services", nargs="*", metavar="service",
                        help=f"services to sync (default: all of {', '.join(s.key for s in SERVICES)})")
    parser.add_argument("--check", action="store_true",
                        help="write nothing; exit 1 if a file is out of date")
    args = parser.parse_args(argv)
    unknown = sorted(set(args.services) - {s.key for s in SERVICES})
    if unknown:
        parser.error(f"unknown service(s): {', '.join(unknown)}")

    chosen = [s for s in SERVICES if not args.services or s.key in args.services]
    failures = 0
    stale_total = 0
    for service in chosen:
        try:
            stale = sync(service, check=args.check)
        except Exception as exc:  # the reason matters more than the type
            print(f"SKIPPED {service.key}: could not import {service.directory}/app.py "
                  f"({type(exc).__name__}: {exc})", file=sys.stderr)
            failures += 1
            continue
        verb = "out of date" if args.check else "updated"
        for path in stale:
            print(f"{verb}: {path.relative_to(REPO_ROOT)}")
        if not stale:
            print(f"ok: {service.key} (already current)")
        stale_total += len(stale)

    if failures:
        return 2
    return 1 if args.check and stale_total else 0


if __name__ == "__main__":
    sys.exit(main())
