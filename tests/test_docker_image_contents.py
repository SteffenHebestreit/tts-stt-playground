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
