"""Optional heavy dependencies: skip where they are absent, unless CI says they must exist.

The piper-training fixes (a run that fails loudly instead of "completing" with the
initial random weights, an ONNX graph whose length follows the text) are tested only
in modules that need torch, onnx and onnxruntime, none of which the unit-test job
installs (tests/requirements.txt is deliberately light). A plain ``importorskip``
makes those modules skip there, which is right for a developer without torch, and
exactly wrong for the CI job that installs torch on purpose: if that install ever
broke, the job would report skips and stay green while guarding nothing.

``REQUIRE_TORCH_TESTS=1`` (set on that job in .github/workflows/ci.yml) turns the
skip into an import error at collection, which fails the run. The unit job leaves it
unset and keeps skipping. ``tests/test_ci_guards.py`` runs both cases for real.
"""

from __future__ import annotations

import importlib
import os

import pytest

REQUIRE_ENV = "REQUIRE_TORCH_TESTS"


def torch_tests_required() -> bool:
    return os.environ.get(REQUIRE_ENV) == "1"


def importorskip_unless_required(modname: str):
    """``pytest.importorskip``, but a failure to import when ``REQUIRE_TORCH_TESTS=1``.

    Use at module level in a test module whose whole subject needs the dependency.
    """
    if torch_tests_required():
        return importlib.import_module(modname)
    return pytest.importorskip(modname)
