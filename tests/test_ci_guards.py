"""The CI guards that keep the drift suites from being skipped must themselves work.

``REQUIRE_DRIFT_SUITES=1`` turns a missing PyYAML into an error, and
``scripts/check_no_skips.py`` reads the JUnit report to catch a module that was
skipped anyway. Both are exercised here the way CI uses them: a real pytest run
in a subprocess whose environment lacks PyYAML.

The same goes for ``REQUIRE_TORCH_TESTS=1`` (tests/optional_deps.py), which makes the
torch-gated piper-training modules fail instead of skip on the CI job that installs
torch, and for the workflow rules that keep that job a gate.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
from compose_helpers import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CI = REPO_ROOT / ".github" / "workflows" / "ci.yml"
CHECK = REPO_ROOT / "scripts" / "check_no_skips.py"
TORCH_MODULES = ("tests/test_piper_training_core_export.py", "tests/test_piper_training_core_pipeline.py")
MODULE = "tests/test_env_wiring.py"


def _pytest(tmp_path: Path, *, without_yaml: bool, require: bool) -> Path:
    env = dict(os.environ)
    env.pop("REQUIRE_DRIFT_SUITES", None)
    if require:
        env["REQUIRE_DRIFT_SUITES"] = "1"
    if without_yaml:
        blocker = tmp_path / "blocker" / "yaml"
        blocker.mkdir(parents=True)
        (blocker / "__init__.py").write_text('raise ImportError("blocked for this test")\n')
        env["PYTHONPATH"] = os.pathsep.join([str(blocker.parent), env.get("PYTHONPATH", "")])
    report = tmp_path / "junit.xml"
    subprocess.run(
        [sys.executable, "-m", "pytest", MODULE, "-q", "-p", "no:cacheprovider",
         f"--junitxml={report}", "-k", "not truenas_app"],
        cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=180)
    return report


def _check(report: Path, *extra: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(CHECK), str(report), "--module", MODULE, *extra],
                          capture_output=True, text=True)


def test_a_module_that_ran_passes_the_check(tmp_path):
    report = _pytest(tmp_path, without_yaml=False, require=True)
    result = _check(report)
    assert result.returncode == 0, result.stderr


def test_a_module_skipped_for_lack_of_yaml_fails_the_check(tmp_path):
    """The failure mode this exists for: everything skipped, run reported green."""
    report = _pytest(tmp_path, without_yaml=True, require=False)
    result = _check(report)
    assert result.returncode == 1, "a fully skipped guard module must fail the CI step"
    assert "no test ran" in result.stderr or "was skipped" in result.stderr


def test_with_the_requirement_set_a_missing_parser_is_an_error_not_a_skip(tmp_path):
    report = _pytest(tmp_path, without_yaml=True, require=True)
    text = report.read_text(encoding="utf-8") if report.exists() else ""
    assert "<skipped" not in text, "REQUIRE_DRIFT_SUITES=1 must not skip, it must fail"
    assert "<error" in text or "<failure" in text


def test_an_allowed_skip_reason_is_tolerated(tmp_path):
    report = tmp_path / "junit.xml"
    report.write_text(textwrap.dedent("""\
        <testsuites><testsuite name="pytest">
          <testcase classname="tests.test_env_wiring" name="test_a"/>
          <testcase classname="tests.test_env_wiring" name="test_b[svc]">
            <skipped message="svc has no local module imports"/>
          </testcase>
        </testsuite></testsuites>
    """))
    assert _check(report).returncode == 1
    assert _check(report, "--allow", "has no local module imports").returncode == 0


@pytest.mark.parametrize("body", [
    "",   # not collected at all
    '<testcase classname="tests.test_env_wiring" name="test_a"><skipped message="x"/></testcase>',
])
def test_a_module_with_nothing_executed_fails_even_when_the_reason_is_allowed(tmp_path, body):
    report = tmp_path / "junit.xml"
    report.write_text(f'<testsuites><testsuite name="pytest">{body}</testsuite></testsuites>')
    result = _check(report, "--allow", "x")
    assert result.returncode == 1
    assert "no test ran" in result.stderr


# --- REQUIRE_TORCH_TESTS: the torch-gated training modules ------------------------------------------------


def _pytest_without_torch(tmp_path: Path, *, require: bool) -> tuple[subprocess.CompletedProcess, str]:
    """Run the two torch-gated modules in a process where `import torch` fails.

    `sys.modules["torch"] = None` makes the import raise ModuleNotFoundError whether or
    not torch is installed here, so this behaves the same on the unit job (no torch) and
    on a machine that has it.
    """
    env = dict(os.environ)
    env.pop("REQUIRE_TORCH_TESTS", None)
    if require:
        env["REQUIRE_TORCH_TESTS"] = "1"
    report = tmp_path / "junit.xml"
    code = (
        "import sys, pytest\n"
        "sys.modules['torch'] = None\n"
        "sys.exit(pytest.main(sys.argv[1:]))\n"
    )
    process = subprocess.run(
        [sys.executable, "-c", code, *TORCH_MODULES, "-q", "-p", "no:cacheprovider", f"--junitxml={report}"],
        cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=180)
    return process, report.read_text(encoding="utf-8") if report.exists() else ""


def test_without_the_switch_a_missing_torch_skips_the_training_modules(tmp_path):
    """The unit job installs no torch and must keep skipping them, not fail."""
    process, text = _pytest_without_torch(tmp_path, require=False)
    assert "<error" not in text and "<failure" not in text, process.stdout[-800:]
    assert text.count("<skipped") == len(TORCH_MODULES), text[:800]
    assert "torch" in text


def test_with_the_switch_a_missing_torch_fails_the_training_modules_instead_of_skipping(tmp_path):
    """The failure mode this exists for: the torch install broke, every test skipped, job green."""
    process, text = _pytest_without_torch(tmp_path, require=True)
    assert process.returncode != 0, "a missing torch must fail the run when REQUIRE_TORCH_TESTS=1"
    assert "<skipped" not in text, "REQUIRE_TORCH_TESTS=1 must not skip, it must fail"
    assert text.count("<error") == len(TORCH_MODULES), text[:800]
    assert "torch" in process.stdout


def test_the_switch_only_means_one_when_it_is_exactly_one(monkeypatch):
    """Same convention as REQUIRE_DRIFT_SUITES: "0", "" and "true" leave the skip in place."""
    import optional_deps

    for value, expected in (("1", True), ("0", False), ("", False), ("true", False)):
        monkeypatch.setenv("REQUIRE_TORCH_TESTS", value)
        assert optional_deps.torch_tests_required() is expected, value
    monkeypatch.delenv("REQUIRE_TORCH_TESTS")
    assert optional_deps.torch_tests_required() is False


# --- the workflow keeps the training job a gate ---------------------------------------------------------


def _ci() -> dict:
    return yaml.safe_load(CI.read_text(encoding="utf-8"))


def test_the_training_job_gates_the_run_and_the_publish_and_requires_torch():
    workflow = _ci()
    job = workflow["jobs"]["training"]
    assert "continue-on-error" not in job, "the training suites are the only tests of the piper-training fixes"
    # publish-images.yml calls this workflow with informational: false; a condition on
    # that input would drop the job from the publish gate.
    assert "if" not in job, "the training job must run whenever CI runs, including when publishing"
    assert (job.get("env") or {}).get("REQUIRE_TORCH_TESTS") == "1"
    steps = " ".join(str(step.get("run", "")) for step in job["steps"])
    for module in TORCH_MODULES:
        assert module in steps, f"the training job does not run {module}"
    assert "download.pytorch.org/whl/cpu" in steps, "the CPU wheels keep this gate fast (the PyPI one is ~4.5 GB)"
    assert "check_no_skips.py" in steps
    # The job that must keep skipping has to stay without the switch, at every level.
    assert "REQUIRE_TORCH_TESTS" not in (workflow.get("env") or {})
    assert "REQUIRE_TORCH_TESTS" not in (workflow["jobs"]["test"].get("env") or {})
    assert all("REQUIRE_TORCH_TESTS" not in (step.get("env") or {}) for step in workflow["jobs"]["test"]["steps"])


def test_the_informational_jobs_still_never_fail_the_run():
    """Promoting `training` must not have promoted the others with it."""
    jobs = _ci()["jobs"]
    for name in ("shellcheck", "dockerfile-lint", "audit", "browser"):
        assert jobs[name].get("continue-on-error") is True, name
        assert "informational" in str(jobs[name]["if"]), name
