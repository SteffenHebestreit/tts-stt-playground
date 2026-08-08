"""Guard: every Dockerfile must ship the local modules its app actually imports.

This exists because the same bug has now shipped three times, each time as a
container that dies at import with ``ModuleNotFoundError``:

    * ``stt-service`` split out ``json_utils.py``      -> not COPYd
    * ``qwen3-asr`` / ``chatterbox`` gained ``model_lifecycle.py`` -> not COPYd
    * ``frontend-service`` gained ``openai_router.py`` -> not COPYd

Nothing else catches it. The unit tests import modules straight off the
filesystem, where siblings are always present, so they pass regardless of what
the image contains. And CI cannot simply build the images to find out: the
``-rocm`` variants sit on ``rocm/dev-ubuntu-22.04:6.2-complete``, which is
roughly 30 GB unpacked against a GitHub runner's ~14 GB of free disk.

So this checks statically what a build would have proven: resolve the local
import closure from each image's entrypoint module, and assert the Dockerfile
copies all of it. It runs against *every* variant -- base, ``.rocm``,
``.vulkan`` -- which is precisely the coverage CI lacks.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

# `COPY --from=<stage>` moves build artefacts between stages rather than source
# from the build context, so it says nothing about which local modules ship.
_STAGE_COPY = "--from="


def _service_dirs() -> list[Path]:
    """Directories that build an image and contain Python source."""
    out = []
    for path in sorted(REPO_ROOT.iterdir()):
        if not path.is_dir() or path.name == "tests":
            continue
        if not any(path.glob("Dockerfile*")):
            continue
        if not (path / "app.py").exists():
            continue
        out.append(path)
    return out


def _dockerfiles() -> list[tuple[Path, Path]]:
    """(service_dir, dockerfile) for every variant of every service."""
    return [
        (svc, df)
        for svc in _service_dirs()
        for df in sorted(svc.glob("Dockerfile*"))
    ]


def _copy_targets(dockerfile: Path) -> list[str]:
    """Source paths named by COPY, with line continuations folded in."""
    text = dockerfile.read_text(encoding="utf-8")
    text = text.replace("\\\n", " ")

    sources: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line.upper().startswith("COPY "):
            continue
        parts = line.split()[1:]
        if any(p.startswith(_STAGE_COPY) for p in parts):
            continue
        parts = [p for p in parts if not p.startswith("--")]
        if len(parts) < 2:
            continue
        sources.extend(parts[:-1])  # last token is the destination
    return sources


def _ships(module: str, copy_sources: list[str]) -> bool:
    """Would ``module.py`` exist in the image, given these COPY sources?"""
    filename = f"{module}.py"
    for src in copy_sources:
        normalised = src.replace("\\", "/").lstrip("./").rstrip("/")
        if src in (".", "./") or normalised == "":
            return True  # COPY . .  ships the whole context
        if normalised == filename:
            return True
    return False


def _local_import_closure(service_dir: Path, entrypoint: str = "app") -> set[str]:
    """Local sibling modules reachable from ``entrypoint``, transitively.

    A module counts as local when a sibling ``<name>.py`` exists; anything else
    is a third-party or stdlib import and is installed by pip, not COPY.
    """
    seen: set[str] = set()
    queue = [entrypoint]

    while queue:
        current = queue.pop()
        source = service_dir / f"{current}.py"
        if not source.exists():
            continue

        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name.split(".")[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                # `from . import x` has no module; relative imports are not used
                # here (these are top-level scripts, not packages).
                if node.level == 0 and node.module:
                    names = [node.module.split(".")[0]]

            for name in names:
                if name == entrypoint or name in seen:
                    continue
                if (service_dir / f"{name}.py").exists():
                    seen.add(name)
                    queue.append(name)

    return seen


@pytest.mark.parametrize(
    "service_dir,dockerfile",
    _dockerfiles(),
    ids=[f"{s.name}/{d.name}" for s, d in _dockerfiles()],
)
def test_dockerfile_ships_every_local_import(service_dir: Path, dockerfile: Path):
    """Each variant must COPY every local module reachable from app.py."""
    required = _local_import_closure(service_dir)
    if not required:
        pytest.skip(f"{service_dir.name} has no local module imports")

    copied = _copy_targets(dockerfile)
    missing = sorted(m for m in required if not _ships(m, copied))

    assert not missing, (
        f"{dockerfile.relative_to(REPO_ROOT)} does not ship "
        f"{', '.join(m + '.py' for m in missing)}, but "
        f"{service_dir.name}/app.py imports it. The container will die at "
        f"startup with ModuleNotFoundError.\n"
        f"Fix: add `COPY {missing[0]}.py .` to {dockerfile.name}.\n"
        f"COPY sources found: {copied}"
    )


def test_every_service_with_python_is_covered():
    """The parametrisation must not silently go empty."""
    pairs = _dockerfiles()
    assert len(pairs) >= 9, f"expected to scan >=9 Dockerfiles, scanned {len(pairs)}"
    assert any(d.name == "Dockerfile.rocm" for _, d in pairs), "no .rocm variant scanned"


# --- ENV parity -------------------------------------------------------------
#
# The second failure mode: a variant that omits a runtime-critical ENV the base
# image sets. `PYTORCH_JIT=0` is the live example -- without it the NeMo
# services segfault on model load under torch 2.11 (see commit a2f43fe). That
# is device-independent, so a variant that drops it is always wrong.

PARITY_ENV_VARS = ("PYTORCH_JIT",)


def _env_vars(dockerfile: Path) -> dict[str, str]:
    text = dockerfile.read_text(encoding="utf-8").replace("\\\n", " ")
    found: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line.upper().startswith("ENV "):
            continue
        body = line[4:].strip()
        if "=" in body:
            for pair in body.split():
                if "=" in pair:
                    key, _, value = pair.partition("=")
                    found[key] = value.strip('"').strip("'")
        else:  # legacy `ENV KEY value`
            key, _, value = body.partition(" ")
            found[key] = value.strip().strip('"').strip("'")
    return found


@pytest.mark.parametrize("service_dir", _service_dirs(), ids=lambda p: p.name)
def test_variants_keep_runtime_critical_env(service_dir: Path):
    """If the base image sets a critical ENV, every variant must set it too."""
    base = service_dir / "Dockerfile"
    if not base.exists():
        pytest.skip("no base Dockerfile")

    base_env = _env_vars(base)
    variants = [d for d in sorted(service_dir.glob("Dockerfile.*"))]

    for var in PARITY_ENV_VARS:
        if var not in base_env:
            continue
        for variant in variants:
            variant_env = _env_vars(variant)
            assert variant_env.get(var) == base_env[var], (
                f"{variant.relative_to(REPO_ROOT)} sets {var}="
                f"{variant_env.get(var)!r} but the base Dockerfile sets "
                f"{var}={base_env[var]!r}. This value guards a segfault on "
                f"model load and is not device-specific."
            )


# --- test/runtime dependency parity ------------------------------------------
#
# The unit tests import frontend-service/app.py directly, so the FastAPI they
# run against must be the FastAPI the image ships. A floating `fastapi>=0.104`
# in tests/requirements.txt resolved to Starlette 1.x in CI, which removed the
# legacy TemplateResponse(name, context) signature — CI failed on a code path
# that worked fine in the container. Same name, two different frameworks.

SHARED_WITH_FRONTEND = ("fastapi", "jinja2", "python-multipart", "httpx")


def _pins(requirements: Path) -> dict[str, str]:
    """{package: exact version} for `pkg==version` lines only."""
    found: dict[str, str] = {}
    for raw in requirements.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if "==" not in line:
            continue
        name, _, version = line.partition("==")
        # strip extras such as uvicorn[standard]
        found[name.split("[", 1)[0].strip().lower()] = version.strip()
    return found


def test_test_deps_match_the_shipped_frontend_deps():
    """Packages shared with the gateway must be pinned to the same version."""
    service = _pins(REPO_ROOT / "frontend-service" / "requirements.txt")
    tests = _pins(REPO_ROOT / "tests" / "requirements.txt")

    mismatches = []
    for package in SHARED_WITH_FRONTEND:
        want = service.get(package)
        got = tests.get(package)
        if want is None:
            continue  # not pinned by the service; nothing to match
        if got != want:
            mismatches.append(f"{package}: tests={got!r} frontend={want!r}")

    assert not mismatches, (
        "tests/requirements.txt must pin these to the same versions as "
        "frontend-service/requirements.txt, or the suite tests a different "
        "framework than the image ships:\n  " + "\n  ".join(mismatches)
    )


# --- import-time side effects ------------------------------------------------
#
# `VOICES_DIR.mkdir(...)` at qwen3-tts module scope raised PermissionError on
# /app for any non-root caller. It passed every local run — in the container the
# path exists and the user is root — and only failed on CI. The same line would
# break a read-only rootfs.
#
# Importing a module must not require write access. Anything a service needs on
# disk should be created where it is used, which is also where the failure can
# be reported sensibly.

_WRITE_CALLS = {"mkdir", "makedirs", "mkdtemp", "touch", "write_text", "write_bytes"}


def _module_level_nodes(tree: ast.Module):
    """Walk statements that execute at import, skipping function/class bodies."""
    stack = list(tree.body)
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue  # only runs when called
        yield node
        for child in ast.iter_child_nodes(node):
            stack.append(child)


@pytest.mark.parametrize("service_dir", _service_dirs(), ids=lambda p: p.name)
def test_no_filesystem_writes_at_import(service_dir: Path):
    """No service may create files or directories just by being imported."""
    source = (service_dir / "app.py").read_text(encoding="utf-8")
    tree = ast.parse(source)

    offenders = []
    for node in _module_level_nodes(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
        if name in _WRITE_CALLS:
            offenders.append(f"line {node.lineno}: {ast.unparse(node)[:80]}")

    assert not offenders, (
        f"{service_dir.name}/app.py writes to the filesystem at import time:\n  "
        + "\n  ".join(offenders)
        + "\nThis fails for any caller without write access to the path (non-root "
          "CI, read-only rootfs). Create it where it is used instead."
    )
