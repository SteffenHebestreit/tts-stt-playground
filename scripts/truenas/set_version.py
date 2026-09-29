#!/usr/bin/env python3
"""Move every place that names the TrueNAS release to one version, or check that they agree.

The single source of truth is ``app_version`` in truenas/tts-stt/app.yaml. Everything else that
defaults to a release (the two TrueNAS compose files, the two .env.truenas*.example files, the
catalog form, and the commands quoted in the docs) is derived from it:

    python scripts/truenas/set_version.py 0.3.0      # rewrite all of them
    python scripts/truenas/set_version.py --check    # exit 1 when they disagree (used by CI)

Only release-shaped tokens are touched, never a bare version number, so an unrelated 0.2.0 (a
package version, a date) stays put:

    IMAGE_TAG:-X.Y.Z   IMAGE_TAG=X.Y.Z   --tag X.Y.Z   checkout vX.Y.Z   /vX.Y.Z/

tests/test_truenas_release_consistency.py imports this module, so the rule the test enforces and
the rule this script applies are the same code.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Iterator

ROOT = Path(__file__).resolve().parents[2]

SEMVER = r"\d+\.\d+\.\d+"

# Files whose release tokens must follow app.yaml. Relative to the repository root.
TOKEN_FILES = (
    "docker-compose.truenas-app.yml",
    "docker-compose.truenas.yml",
    ".env.truenas.example",
    ".env.truenas.production.example",
    "truenas/README.md",
    "truenas/custom-app-include.yml",
    "truenas/tts-stt/app-readme.md",
    "docs/truenas-installation-guide.md",
    "docs/truenas-deployment.md",
    "docs/truenas-service-profiles.md",
    "docs/truenas-custom-app-checklist.md",
)

# (name, pattern). Group 1 is the text before the version, group 2 the version.
TOKEN_PATTERNS = (
    ("IMAGE_TAG default", re.compile(rf"(?<![A-Z_])(IMAGE_TAG(?::-|=))({SEMVER})(?![\d.])")),
    ("--tag", re.compile(rf"(--tag\s+)({SEMVER})(?![\d.])")),
    ("git checkout", re.compile(rf"(checkout\s+v)({SEMVER})(?![\d.])")),
    ("raw URL", re.compile(rf"(/v)({SEMVER})(?=/)")),
)

APP_YAML = "truenas/tts-stt/app.yaml"
QUESTIONS_YAML = "truenas/tts-stt/questions.yaml"

_APP_VERSION = re.compile(r'^(app_version:\s*)"?(\d+\.\d+\.\d+)"?\s*$', re.M)
_CHART_VERSION = re.compile(r"^(version:\s*)(\d+\.\d+\.\d+)\s*$", re.M)
# The `default:` line inside the `image_tag` question.
_QUESTION_DEFAULT = re.compile(
    r'(- variable: image_tag\b.*?\n\s+default:\s*)"?(\d+\.\d+\.\d+)"?', re.S
)


def source_version(root: Path = ROOT) -> str:
    """``app_version`` from truenas/tts-stt/app.yaml."""
    match = _APP_VERSION.search((root / APP_YAML).read_text(encoding="utf-8"))
    if not match:
        raise SystemExit(f"{APP_YAML}: no app_version line found")
    return match.group(2)


def chart_version(root: Path = ROOT) -> str:
    match = _CHART_VERSION.search((root / APP_YAML).read_text(encoding="utf-8"))
    if not match:
        raise SystemExit(f"{APP_YAML}: no version line found")
    return match.group(2)


def question_default(root: Path = ROOT) -> str:
    match = _QUESTION_DEFAULT.search((root / QUESTIONS_YAML).read_text(encoding="utf-8"))
    if not match:
        raise SystemExit(f"{QUESTIONS_YAML}: no image_tag default found")
    return match.group(2)


def release_tokens(root: Path = ROOT) -> Iterator[tuple[str, int, str, str]]:
    """Yield (file, line number, token name, version) for every release token."""
    for name in TOKEN_FILES:
        path = root / name
        if not path.is_file():
            continue
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            for token, pattern in TOKEN_PATTERNS:
                for match in pattern.finditer(line):
                    yield name, number, token, match.group(2)


def mismatches(root: Path = ROOT) -> list[str]:
    """Human-readable list of everything that disagrees with app_version."""
    expected = source_version(root)
    problems: list[str] = []
    if chart_version(root) != expected:
        problems.append(f"{APP_YAML}: version {chart_version(root)} != app_version {expected}")
    if question_default(root) != expected:
        problems.append(f"{QUESTIONS_YAML}: image_tag default {question_default(root)} != {expected}")
    for name, number, token, version in release_tokens(root):
        if version != expected:
            problems.append(f"{name}:{number}: {token} {version} != {expected}")
    return problems


def apply(new: str, root: Path = ROOT) -> list[str]:
    """Rewrite every release token to ``new``; returns the files that changed."""
    if not re.fullmatch(SEMVER, new):
        raise SystemExit(f"'{new}' is not MAJOR.MINOR.PATCH")
    changed: list[str] = []

    def rewrite(name: str, transform) -> None:
        path = root / name
        # Bytes in and out, so a checkout with CRLF line endings keeps them.
        old_text = path.read_bytes().decode("utf-8")
        new_text = transform(old_text)
        if new_text != old_text:
            path.write_bytes(new_text.encode("utf-8"))
            changed.append(name)

    def app_yaml(text: str) -> str:
        text = _APP_VERSION.sub(lambda m: f'{m.group(1)}"{new}"', text)
        return _CHART_VERSION.sub(lambda m: f"{m.group(1)}{new}", text)

    rewrite(APP_YAML, app_yaml)
    rewrite(QUESTIONS_YAML, lambda t: _QUESTION_DEFAULT.sub(lambda m: f'{m.group(1)}"{new}"', t))
    for name in TOKEN_FILES:
        if not (root / name).is_file():
            continue

        def tokens(text: str) -> str:
            for _token, pattern in TOKEN_PATTERNS:
                text = pattern.sub(lambda m: f"{m.group(1)}{new}", text)
            return text

        rewrite(name, tokens)
    return changed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("version", nargs="?", help="new release, MAJOR.MINOR.PATCH")
    parser.add_argument("--check", action="store_true", help="only verify; exit 1 on a mismatch")
    parser.add_argument("--root", type=Path, default=ROOT, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    if args.check or not args.version:
        problems = mismatches(args.root)
        if problems:
            print("\n".join(problems), file=sys.stderr)
            print(f"\n{len(problems)} place(s) disagree with app_version {source_version(args.root)}.",
                  file=sys.stderr)
            return 1
        print(f"All release tokens agree: {source_version(args.root)}")
        return 0

    changed = apply(args.version, args.root)
    print(f"Release set to {args.version}. Changed {len(changed)} file(s):")
    for name in changed:
        print(f"  {name}")
    remaining = mismatches(args.root)
    if remaining:
        print("\n".join(remaining), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
