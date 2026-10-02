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
import posixpath
import re
from pathlib import Path

import pytest

# Only the dev-overlay checks need a YAML parser, but a missing one must not turn
# the whole module into a silent skip in CI: compose_helpers fails instead when
# REQUIRE_DRIFT_SUITES=1 (set by the CI workflow) and skips only on a bare checkout.
from compose_helpers import load_compose

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


# --- the dev overlay ---------------------------------------------------------
#
# Same rule as the Dockerfile, different mechanism. docker-compose.dev.yml
# live-mounts source over the built image FILE BY FILE, naming each local module
# explicitly. A module that is imported but not listed silently keeps the copy
# baked into the image: edits to it appear to do nothing, and against an image
# built before that module existed the container dies at import with
# ModuleNotFoundError — the exact failure the COPY check above exists to stop,
# reached by the other route.


def _dev_overlay_mounts() -> dict[str, set[str]]:
    """{service: {basenames it mounts into the image}}."""
    document = load_compose(REPO_ROOT / "docker-compose.dev.yml")
    out: dict[str, set[str]] = {}
    for name, service in (document.get("services") or {}).items():
        names: set[str] = set()
        for volume in (service or {}).get("volumes") or []:
            if isinstance(volume, str) and ":" in volume:
                names.add(volume.split(":")[1].rstrip("/").rsplit("/", 1)[-1])
        out[name] = names
    return out


@pytest.mark.parametrize("service_dir", _service_dirs(), ids=lambda p: p.name)
def test_dev_overlay_mounts_every_local_import(service_dir: Path):
    """If the overlay mounts a service at all, it must mount its whole closure."""
    mounts = _dev_overlay_mounts().get(service_dir.name)
    if mounts is None:
        pytest.skip(f"{service_dir.name} is not in the dev overlay")

    required = _local_import_closure(service_dir)
    missing = sorted(f"{m}.py" for m in required if f"{m}.py" not in mounts)

    assert not missing, (
        f"docker-compose.dev.yml mounts {service_dir.name}/app.py but not "
        f"{', '.join(missing)}, which it imports. Edits to those files would be "
        f"ignored, and against an image built before they existed the container "
        f"dies at import.\nAdd: - \"./{service_dir.name}/{missing[0]}:/app/{missing[0]}\""
    )


def test_the_dev_overlay_scan_is_not_vacuous():
    """A rename in the overlay must not turn every case above into a skip."""
    covered = [d for d in _service_dirs() if d.name in _dev_overlay_mounts()]
    assert len(covered) >= 6, (
        f"only {len(covered)} services matched the dev overlay by name; the "
        f"parametrisation is mostly skipping"
    )


def test_every_service_with_python_is_covered():
    """The parametrisation must not silently go empty."""
    pairs = _dockerfiles()
    assert len(pairs) >= 9, f"expected to scan >=9 Dockerfiles, scanned {len(pairs)}"
    assert any(d.name == "Dockerfile.rocm" for _, d in pairs), "no .rocm variant scanned"


# --- files a module opens next to itself ---------------------------------------
#
# The import closure sees modules, not the files they read. public_suffix.py opens
# public_suffix_list.dat from its own directory, and an image without that file does
# not die at import: the Settings page quietly refuses every wildcard host name with
# "cannot be checked". So the data files a module names next to itself are held to
# the same rule as the modules, in the image and in the dev overlay.

_SIBLING_FILE = (
    # os.path.join(os.path.dirname(os.path.abspath(__file__)), "name")
    re.compile(r"os\.path\.join\(\s*os\.path\.dirname\(\s*(?:os\.path\.(?:abspath|realpath)\(\s*)?__file__\s*\)?\s*\)"
               r"\s*,\s*[\"']([^\"'/\\]+)[\"']"),
    # Path(__file__).parent / "name", Path(__file__).resolve().parent / "name"
    re.compile(r"Path\(\s*__file__\s*\)(?:\.resolve\(\))?\.parent\s*/\s*[\"']([^\"'/\\]+)[\"']"),
)


def _sibling_files(service_dir: Path) -> set[str]:
    """Files and folders that the app or a module it imports opens next to itself."""
    names: set[str] = set()
    for module in {"app", *_local_import_closure(service_dir)}:
        source = (service_dir / f"{module}.py").read_text(encoding="utf-8")
        for pattern in _SIBLING_FILE:
            names.update(pattern.findall(source))
    return names


def _ships_file(name: str, copy_sources: list[str]) -> bool:
    for src in copy_sources:
        normalised = src.replace("\\", "/").lstrip("./").rstrip("/")
        if src in (".", "./") or normalised in ("", name):
            return True
    return False


@pytest.mark.parametrize(
    "service_dir,dockerfile",
    _dockerfiles(),
    ids=[f"{s.name}/{d.name}" for s, d in _dockerfiles()],
)
def test_dockerfile_ships_every_file_a_module_opens_next_to_itself(service_dir: Path, dockerfile: Path):
    copied = _copy_targets(dockerfile)
    missing = sorted(name for name in _sibling_files(service_dir) if not _ships_file(name, copied))
    assert not missing, (
        f"{dockerfile.relative_to(REPO_ROOT)} does not ship {missing}, which {service_dir.name} opens next "
        f"to its own modules. Add it to a COPY line. COPY sources found: {copied}")


def test_the_dev_overlay_mounts_every_file_a_module_opens_next_to_itself():
    overlay = _dev_overlay_mounts()
    missing = [f"{service_dir.name}: {name}"
               for service_dir in _service_dirs() if service_dir.name in overlay
               for name in sorted(_sibling_files(service_dir)) if name not in overlay[service_dir.name]]
    assert not missing, f"docker-compose.dev.yml does not mount {missing}"


def test_the_gateway_ships_the_settings_page_modules_and_the_public_suffix_list():
    """Named, so the two scans above cannot pass by finding nothing."""
    service = REPO_ROOT / "frontend-service"
    assert {"settings_schema", "settings_store", "public_suffix"} <= _local_import_closure(service)
    assert {"public_suffix_list.dat", "static"} <= _sibling_files(service)
    shipped = {"settings_schema.py", "settings_store.py", "public_suffix.py", "public_suffix_list.dat"}
    assert all(_ships_file(name, _copy_targets(service / "Dockerfile")) for name in shipped)
    assert shipped <= _dev_overlay_mounts()["frontend-service"]


# --- the settings folders are bind mounts, never part of an image ----------------
#
# The Settings page saves into /app/settings (API key digests, the audit log), and
# /app/backend-settings is the folder for backend settings saved there. Both must be
# bind mounts from the app's dataset: a folder an image creates (RUN mkdir, a COPY into
# it, WORKDIR) or declares (VOLUME) puts the saves in the container layer or an
# anonymous volume, and the next recreate -- every TrueNAS Edit, Update and Stop/Start
# -- drops them, admin keys included. The gateway refuses to save unless
# /proc/self/mountinfo lists /app/settings as a mount; this keeps every image from
# creating the folder in the first place.

SETTINGS_FOLDERS = ("/app/settings", "/app/backend-settings")


def _instructions(text: str) -> list[tuple[str, str]]:
    """(INSTRUCTION, arguments) per Dockerfile instruction: continuations folded, comments dropped."""
    out = []
    for raw in text.replace("\\\r\n", " ").replace("\\\n", " ").splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            keyword, _, arguments = line.partition(" ")
            out.append((keyword.upper(), arguments.strip()))
    return out


def _paths(arguments: str, workdir: str) -> list[str]:
    """Every word of an instruction that could be a container path, made absolute against WORKDIR."""
    found = []
    for word in re.split(r"[\s,]+", arguments):
        word = word.strip("\"'[]();&|")
        if not word or word.startswith("-") or "$" in word or "=" in word:
            continue
        found.append(posixpath.normpath(posixpath.join(workdir, word)))
    return found


def _settings_folder_problems(text: str) -> list[str]:
    """Instructions that create, declare or write into a settings folder."""
    problems = []
    workdir = "/"
    for keyword, arguments in _instructions(text):
        if keyword in ("COPY", "ADD"):
            words = [w for w in arguments.split() if not w.startswith("--")]
            candidates = _paths(words[-1], workdir) if len(words) >= 2 else []
        elif keyword in ("RUN", "VOLUME", "WORKDIR"):
            candidates = _paths(arguments, workdir)
        else:
            candidates = []
        for path in candidates:
            if any(path == folder or path.startswith(folder + "/") for folder in SETTINGS_FOLDERS):
                problems.append(f"{keyword} {arguments}")
                break
        if keyword == "WORKDIR":
            workdir = posixpath.normpath(posixpath.join(workdir, arguments.strip("\"'")))
    return problems


def _all_dockerfiles() -> list[Path]:
    return sorted(p for p in REPO_ROOT.glob("**/Dockerfile*") if ".git" not in p.parts and p.is_file())


def test_no_image_creates_or_declares_a_settings_folder():
    scanned = _all_dockerfiles()
    assert len(scanned) >= 15, scanned
    problems = {str(p.relative_to(REPO_ROOT)): _settings_folder_problems(p.read_text(encoding="utf-8"))
                for p in scanned}
    problems = {name: found for name, found in problems.items() if found}
    assert not problems, (
        f"these images create or declare a settings folder, so saves would vanish with the container: {problems}")


_FOLDER_CHECK_CASES = {
    "relative-mkdir": ("WORKDIR /app\nRUN mkdir -p settings", True),
    "mkdir-among-others": ("WORKDIR /app\nRUN mkdir -p /app/static /app/backend-settings && chmod 700 /app/backend-settings",
                           True),
    "volume-json-form": ('VOLUME ["/app/settings"]', True),
    "volume-plain-form": ("VOLUME /app/backend-settings", True),
    "workdir": ("WORKDIR /app/settings", True),
    "copy-into-relative": ("WORKDIR /app\nCOPY defaults.json settings/", True),
    "copy-continued-line": ("COPY --chown=0:0 keys.json \\\n    /app/settings/keys.json", True),
    "run-touch": ("RUN touch /app/settings/gateway.json", True),
    "the-real-copy-line": ("WORKDIR /app\nCOPY settings_schema.py settings_store.py ./\n"
                           "RUN mkdir -p /app/static /app/templates", False),
    "a-similar-name": ("WORKDIR /app\nRUN mkdir -p /app/settings-old", False),
    "a-comment": ("# RUN mkdir -p /app/settings\nWORKDIR /app", False),
}


@pytest.mark.parametrize("dockerfile, caught", list(_FOLDER_CHECK_CASES.values()), ids=list(_FOLDER_CHECK_CASES))
def test_the_settings_folder_check_catches_what_it_is_for(dockerfile, caught):
    """Guards the guard: each way an image could create or declare the folder is noticed."""
    assert bool(_settings_folder_problems(dockerfile)) is caught


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

SHARED_WITH_FRONTEND = ("fastapi", "jinja2", "python-multipart", "httpx", "uvicorn")


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


def _local_class_inits(service_dir: Path) -> dict[str, ast.FunctionDef]:
    """{ClassName: its __init__} across the service's own modules."""
    found: dict[str, ast.FunctionDef] = {}
    for source_file in sorted(service_dir.glob("*.py")):
        tree = ast.parse(source_file.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                    found[node.name] = item
    return found


@pytest.mark.parametrize("service_dir", _service_dirs(), ids=lambda p: p.name)
def test_no_filesystem_writes_at_import(service_dir: Path):
    """No service may create files or directories just by being imported.

    Constructors count. `app.py` instantiates its helpers at module scope, so a
    `mkdir` inside one of their `__init__`s runs at import exactly as if it were
    written inline — but sits one frame away, which is how
    `ModelExporter.__init__` creating `models/` slipped past the first version of
    this check. That is the same shape as the qwen3-tts `VOICES_DIR.mkdir()` in
    the docstring above: fine as root in a container, PermissionError anywhere
    else.
    """
    source = (service_dir / "app.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    class_inits = _local_class_inits(service_dir)

    offenders = []
    for node in _module_level_nodes(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
        if name in _WRITE_CALLS:
            offenders.append(f"line {node.lineno}: {ast.unparse(node)[:80]}")
            continue

        # A module-level `Thing()` where Thing is defined in this service.
        init = class_inits.get(name) if name else None
        if init is None:
            continue
        for sub in ast.walk(init):
            if not isinstance(sub, ast.Call):
                continue
            inner = sub.func
            inner_name = (inner.attr if isinstance(inner, ast.Attribute)
                          else getattr(inner, "id", None))
            if inner_name in _WRITE_CALLS:
                offenders.append(
                    f"line {node.lineno}: {name}() -> {name}.__init__ "
                    f"line {sub.lineno}: {ast.unparse(sub)[:60]}"
                )

    assert not offenders, (
        f"{service_dir.name}/app.py writes to the filesystem at import time:\n  "
        + "\n  ".join(offenders)
        + "\nThis fails for any caller without write access to the path (non-root "
          "CI, read-only rootfs). Create it where it is used instead."
    )
