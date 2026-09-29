"""The CI workflow for the TrueNAS files must watch what it claims to, and guard what it runs.

.github/workflows/truenas.yml runs only when a TrueNAS file changes (a `paths` filter). A filter
that has gone stale (a renamed script, a new test module nobody added) means the workflow silently
stops running for exactly the files it exists for, and a test module that is missing from the
skip guard can go green while skipped. Both are checked here, from the parsed workflow.

The steps themselves are not re-run: ci.yml's actionlint job lints the file, and every command in
it is one this test suite runs in another form (docker compose config, the renderer, the scripts).
"""

from __future__ import annotations

import os
import re

import pytest

from compose_helpers import REPO_ROOT, require_yaml

yaml = require_yaml()

WORKFLOW = REPO_ROOT / ".github" / "workflows" / "truenas.yml"


@pytest.fixture(scope="module")
def workflow() -> dict:
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))


def repository_files() -> list[str]:
    files = []
    for root, dirs, names in os.walk(REPO_ROOT):
        dirs[:] = [d for d in dirs if d not in (".git", "__pycache__", "node_modules") and not d.startswith("venv")]
        files += [os.path.relpath(os.path.join(root, n), REPO_ROOT).replace(os.sep, "/") for n in names]
    return files


def triggers(workflow: dict) -> dict:
    return workflow.get("on", workflow.get(True))  # PyYAML reads a bare `on` as the boolean True


def glob_to_regex(pattern: str) -> re.Pattern:
    """GitHub's path filter glob: `*` stays inside a directory, `**` crosses them."""
    out, i = [], 0
    while i < len(pattern):
        if pattern.startswith("**", i):
            out.append(".*")
            i += 2
        elif pattern[i] == "*":
            out.append("[^/]*")
            i += 1
        elif pattern[i] == "?":
            out.append("[^/]")
            i += 1
        else:
            out.append(re.escape(pattern[i]))
            i += 1
    return re.compile("".join(out) + r"\Z")


def watched(workflow: dict, event: str) -> list[str]:
    return list(triggers(workflow)[event]["paths"])


def test_push_and_pull_request_watch_the_same_files(workflow):
    assert watched(workflow, "push") == watched(workflow, "pull_request")


def test_every_path_filter_still_matches_something(workflow):
    """A filter for a file that was renamed watches nothing, and nobody notices."""
    files = repository_files()
    for pattern in watched(workflow, "push"):
        regex = glob_to_regex(pattern)
        assert any(regex.match(f) for f in files), f"path filter {pattern!r} matches no file"


def test_every_file_that_makes_up_the_truenas_path_is_watched(workflow):
    regexes = [glob_to_regex(p) for p in watched(workflow, "push")]
    must_trigger = [
        "docker-compose.truenas-app.yml", "docker-compose.truenas.yml",
        ".env.truenas.example", ".env.truenas.production.example",
        "truenas/README.md", "truenas/custom-app-include.yml", "truenas/tts-stt/app.yaml",
        "truenas/tts-stt/questions.yaml", "truenas/tts-stt/templates/docker-compose.yaml",
        "docs/truenas-installation-guide.md", ".github/workflows/truenas.yml",
        *(p.relative_to(REPO_ROOT).as_posix() for p in (REPO_ROOT / "scripts" / "truenas").iterdir()
          if p.is_file() and p.suffix in (".sh", ".py")),
        *(p.relative_to(REPO_ROOT).as_posix() for p in (REPO_ROOT / "tests").glob("*truenas*.py")),
        *(p.relative_to(REPO_ROOT).as_posix() for p in (REPO_ROOT / "docs").glob("truenas-*.md")),
    ]
    unwatched = [f for f in must_trigger if not any(r.match(f) for r in regexes)]
    assert not unwatched, f"a change to these does not run the TrueNAS workflow: {unwatched}"


def pytest_step(workflow: dict) -> dict:
    steps = workflow["jobs"]["tests"]["steps"]
    return next(s for s in steps if "pytest" in str(s.get("run", "")))


def skip_guard_modules(workflow: dict) -> set[str]:
    steps = workflow["jobs"]["tests"]["steps"]
    guard = next(s for s in steps if "check_no_skips" in str(s.get("run", "")))
    return set(re.findall(r"--module\s+(\S+)", guard["run"]))


def test_every_truenas_test_module_is_run_and_guarded_against_skipping(workflow):
    modules = {p.relative_to(REPO_ROOT).as_posix() for p in (REPO_ROOT / "tests").glob("test_truenas_*.py")}
    assert "pytest tests/test_truenas_*.py" in pytest_step(workflow)["run"], "new modules must be picked up by the glob"
    missing = modules - skip_guard_modules(workflow)
    assert not missing, f"not in the skip guard of truenas.yml: {sorted(missing)}"
    stale = skip_guard_modules(workflow) - modules
    assert not stale, f"the skip guard names modules that do not exist: {sorted(stale)}"


def test_the_workflow_requires_the_drift_suites_instead_of_skipping_them(workflow):
    env = workflow["jobs"]["tests"].get("env", {})
    assert env.get("REQUIRE_DRIFT_SUITES") == "1"


def test_every_file_a_step_names_exists(workflow):
    """A `run:` that calls a script or reads a file that was renamed fails only when it runs."""
    named = set()
    for job in workflow["jobs"].values():
        for step in job.get("steps", []):
            text = str(step.get("run", ""))
            named |= set(re.findall(r"(?<![\w/.-])((?:scripts|truenas|tests|docs)/[\w./*-]+)", text))
            named |= set(re.findall(r"(?<![\w/.-])(docker-compose[\w.-]*\.yml|\.env\.truenas[\w.]*)", text))
    assert named, "the workflow names no repository files: this test would check nothing"
    for name in sorted(named):
        pattern = name.rstrip(".")
        assert list(REPO_ROOT.glob(pattern)), f"the workflow uses {name}, which does not exist"


def test_the_workflow_never_needs_more_than_read_access(workflow):
    assert workflow["permissions"] == {"contents": "read"}


def test_nothing_in_the_workflow_publishes_or_pulls_from_a_registry(workflow):
    """It validates files. Publishing belongs to publish-images.yml and needs a token this has not."""
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "docker push" not in text and "docker pull" not in text and "docker login" not in text
    assert "packages: write" not in text
