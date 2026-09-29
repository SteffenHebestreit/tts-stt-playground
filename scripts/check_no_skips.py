#!/usr/bin/env python3
"""Fail a CI run in which a guard suite was skipped instead of run.

The config-drift suites (compose wiring, Dockerfile contents, deploy presets, OpenAPI
specs) compare files with each other and fail when they disagree. When a
dependency of the test environment is missing they used to *skip* -- which the
runner reports as success -- so a job that forgot to install PyYAML went green
while guarding nothing. That happened.

Two lines of defence: the suites raise instead of skipping when
``REQUIRE_DRIFT_SUITES=1`` (see tests/compose_helpers.py), and this script reads
the JUnit report of the run and fails when

* a named module ran no test at all (not collected, or skipped as a whole), or
* one of its tests was skipped for a reason that is not on the allow list.

    pytest tests/ --junitxml=junit.xml
    python scripts/check_no_skips.py junit.xml \\
        --module tests/test_env_wiring.py --module tests/test_docker_image_contents.py \\
        --allow "has no local module imports"
"""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable, Optional


def _belongs(case: ET.Element, module: str) -> bool:
    """Does this <testcase> come from *module* (a path like tests/test_x.py)?"""
    dotted = module.removesuffix(".py").replace("/", ".")
    stem = Path(module).stem
    haystack = " ".join(filter(None, (case.get("classname"), case.get("file"), case.get("name"))))
    return dotted in haystack or module in haystack or f"{stem}::" in haystack or (
        case.get("classname", "").endswith(stem))


def check(junit: Path, modules: Iterable[str], allow: Iterable[str] = ()) -> list[str]:
    """Problems found in the report; empty means every module really ran."""
    allowed = tuple(allow)
    root = ET.parse(junit).getroot()
    cases = list(root.iter("testcase"))
    problems: list[str] = []

    for module in modules:
        mine = [c for c in cases if _belongs(c, module)]
        executed = 0
        for case in mine:
            skipped = case.find("skipped")
            if skipped is None:
                executed += 1
                continue
            reason = " ".join(filter(None, (skipped.get("message"), skipped.text)))
            if not any(fragment in reason for fragment in allowed):
                problems.append(f"{module}: {case.get('name')} was skipped: {reason.strip()[:200]}")
        if executed == 0:
            problems.append(f"{module}: no test ran (collected {len(mine)}, all skipped or none found)")
    return problems


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("junit", type=Path)
    parser.add_argument("--module", action="append", required=True,
                        help="test module that must have run, e.g. tests/test_env_wiring.py")
    parser.add_argument("--allow", action="append", default=[],
                        help="a skip reason (substring) that is legitimate")
    args = parser.parse_args(argv)

    problems = check(args.junit, args.module, args.allow)
    for problem in problems:
        print(f"::error::{problem}" if _in_actions() else problem, file=sys.stderr)
    if problems:
        return 1
    print(f"ok: {len(args.module)} guard modules ran with no unexpected skips")
    return 0


def _in_actions() -> bool:
    import os
    return os.environ.get("GITHUB_ACTIONS") == "true"


if __name__ == "__main__":
    sys.exit(main())
