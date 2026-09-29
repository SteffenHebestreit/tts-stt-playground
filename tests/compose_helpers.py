"""Shared plumbing for the suites that read the compose files and deploy profiles.

Three things live here because four test modules need the same behaviour and a
copy in each would drift:

* ``require_yaml`` -- PyYAML or a *decision*. The drift suites used to
  ``importorskip`` it, and a CI job that forgot to install it reported them as
  skips: green, and guarding nothing. With ``REQUIRE_DRIFT_SUITES=1`` (set by
  CI) a missing parser is an error instead.
* ``load_compose`` -- a YAML loader that accepts Compose's own tags
  (``!reset``, ``!override``), which SafeLoader rejects.
* ``interpolate`` -- Compose variable substitution, so a test can ask what a
  value becomes for a given environment without a Docker CLI.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Mapping, Optional

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def require_yaml():
    """Return the yaml module, or skip the module -- unless CI says it must exist."""
    try:
        import yaml
    except ImportError:
        if os.environ.get("REQUIRE_DRIFT_SUITES") == "1":
            raise
        pytest.skip("PyYAML needed to parse compose files", allow_module_level=True)
    return yaml


yaml = require_yaml()


class _ComposeLoader(yaml.SafeLoader):
    """SafeLoader that resolves Compose tags to their untagged value."""


def _untag(loader, _suffix, node):
    if isinstance(node, yaml.MappingNode):
        plain = yaml.MappingNode(loader.DEFAULT_MAPPING_TAG, node.value)
    elif isinstance(node, yaml.SequenceNode):
        plain = yaml.SequenceNode(loader.DEFAULT_SEQUENCE_TAG, node.value)
    else:
        plain = yaml.ScalarNode(loader.DEFAULT_SCALAR_TAG, node.value)
    return loader.construct_object(plain, deep=True)


_ComposeLoader.add_multi_constructor("", _untag)


def load_compose(path: Path) -> dict:
    """Parse a compose file; ``!reset`` / ``!override`` values keep their content."""
    return yaml.load(Path(path).read_text(encoding="utf-8"), Loader=_ComposeLoader) or {}


def load_compose_with_tags(path: Path) -> dict:
    """Like ``load_compose`` but ``!reset`` / ``!override`` nodes become ``Tagged`` markers."""

    class _Loader(yaml.SafeLoader):
        pass

    def tagged(loader, suffix, node):
        plain = _untag(loader, suffix, node)
        return Tagged(suffix, plain)

    _Loader.add_multi_constructor("!", tagged)
    return yaml.load(Path(path).read_text(encoding="utf-8"), Loader=_Loader) or {}


class Tagged:
    """A value that carried a Compose tag such as ``!reset`` or ``!override``."""

    def __init__(self, tag: str, value):
        self.tag = tag
        self.value = value

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Tagged({self.tag!r}, {self.value!r})"


_VAR = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?:(:?[-?+])([^{}]*))?\}")


def interpolate(value: str, env: Optional[Mapping[str, str]] = None) -> str:
    """Compose-style substitution of ``${VAR}``, ``${VAR:-d}``, ``${VAR-d}``.

    ``:-`` uses the default when the variable is unset OR empty, ``-`` only when
    it is unset (the ROCm overlay relies on the difference). Nested defaults such
    as ``${A:-${B:-x}/y}`` resolve innermost first. ``$$`` is a literal ``$``.
    """
    env = env or {}
    text = value.replace("$$", "\0")

    def substitute(match: re.Match) -> str:
        name, op, default = match.group(1), match.group(2), match.group(3) or ""
        present = name in env
        current = env.get(name, "")
        if op is None:
            return current
        if op == ":-":
            return current if present and current != "" else default
        if op == "-":
            return current if present else default
        raise ValueError(f"unsupported interpolation operator {op!r} in {value!r}")

    previous = None
    while previous != text:
        previous = text
        text = _VAR.sub(substitute, text)
    return text.replace("\0", "$")


def dotenv(path: Path) -> dict[str, str]:
    """The uncommented ``KEY=value`` assignments of an env file."""
    out: dict[str, str] = {}
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        out[key.strip()] = val.strip().strip('"').strip("'")
    return out


def env_mapping(service: dict) -> dict:
    """A service's ``environment`` as a dict, whichever form the file used."""
    env = service.get("environment") or {}
    if isinstance(env, dict):
        return dict(env)
    out = {}
    for entry in env:
        key, _, val = entry.partition("=")
        out[key] = val
    return out
