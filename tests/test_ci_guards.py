"""The CI guards that keep the drift suites from being skipped must themselves work.

``REQUIRE_DRIFT_SUITES=1`` turns a missing PyYAML into an error, and
``scripts/check_no_skips.py`` reads the JUnit report to catch a module that was
skipped anyway. Both are exercised here the way CI uses them: a real pytest run
in a subprocess whose environment lacks PyYAML.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
CHECK = REPO_ROOT / "scripts" / "check_no_skips.py"
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
