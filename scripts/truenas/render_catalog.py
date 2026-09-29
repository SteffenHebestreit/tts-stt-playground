#!/usr/bin/env python3
"""Render the TrueNAS catalog template the way the Apps UI would, from the answers of questions.yaml.

    python scripts/truenas/render_catalog.py --set host_path=/mnt/tank/apps/tts-stt
    python scripts/truenas/render_catalog.py --set host_path=/mnt/tank/apps/tts-stt \
        --set enable_canary=true | docker compose -f - config -q

Used by CI (.github/workflows/truenas.yml) and by tests/test_truenas_catalog.py, so both render
exactly one way. It needs PyYAML and Jinja2 (both are in tests/requirements.txt).

What it reproduces from TrueNAS, and what it does not
  * Every question gets its default; ``--set name=value`` overrides one. A question without a
    default (the host path) has to be given.
  * A question whose ``show_if`` is false is left out of ``values``, and a template that reads a
    missing name fails (StrictUndefined) instead of rendering an empty string. TrueNAS does not
    promise to hand hidden answers to a template, so a template must not depend on them.
  * ``ix_volumes`` is injected only with ``--ix-volume-path``. The shape TrueNAS uses for it is an
    assumption (see the template header).
It does NOT use TrueNAS's ix-lib renderer: this is plain Jinja2.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
CATALOG = ROOT / "truenas" / "tts-stt"
QUESTIONS = CATALOG / "questions.yaml"
TEMPLATE = CATALOG / "templates" / "docker-compose.yaml"


def _yaml():
    try:
        import yaml
    except ImportError:  # pragma: no cover - the message is the point
        raise SystemExit("PyYAML is required: pip install pyyaml") from None
    return yaml


def _jinja():
    try:
        import jinja2
    except ImportError:  # pragma: no cover
        raise SystemExit("Jinja2 is required: pip install jinja2") from None
    return jinja2


def load_questions(path: Path = QUESTIONS) -> list[dict[str, Any]]:
    """The ``questions`` list of questions.yaml."""
    document = _yaml().safe_load(path.read_text(encoding="utf-8")) or {}
    return list(document.get("questions") or [])


def _coerce(question: Mapping[str, Any], raw: Any) -> Any:
    """Turn a command-line string into the type the question declares."""
    kind = (question.get("schema") or {}).get("type")
    if not isinstance(raw, str):
        return raw
    if kind == "boolean":
        if raw.lower() in ("true", "1", "yes", "on"):
            return True
        if raw.lower() in ("false", "0", "no", "off"):
            return False
        raise SystemExit(f"{question['variable']}: '{raw}' is not true or false")
    if kind == "int":
        try:
            return int(raw)
        except ValueError:
            raise SystemExit(f"{question['variable']}: '{raw}' is not an integer") from None
    return raw


def default_answers(
    questions: list[dict[str, Any]], overrides: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Default answer for every question, with ``overrides`` applied."""
    overrides = dict(overrides or {})
    known = {q["variable"] for q in questions}
    unknown = sorted(set(overrides) - known - {"ix_volumes"})
    if unknown:
        raise SystemExit(f"no such question: {', '.join(unknown)} (see questions.yaml)")
    answers: dict[str, Any] = {}
    for question in questions:
        name = question["variable"]
        schema = question.get("schema") or {}
        if name in overrides:
            answers[name] = _coerce(question, overrides.pop(name))
        elif "default" in schema:
            answers[name] = schema["default"]
        elif schema.get("required") and all(
            _condition_holds(answers, c) for c in schema.get("show_if") or []
        ):
            # A required question the form has no default for. The path of the data dataset is
            # the only one today: guessing it would install into the wrong place. One that the
            # form hides (the host path while ixVolume is chosen) is not asked, so not needed.
            raise SystemExit(f"question '{name}' has no default: pass --set {name}=<value>")
    answers.update(overrides)  # ix_volumes
    return answers


def _condition_holds(answers: Mapping[str, Any], condition: list[Any]) -> bool:
    name, operator, expected = condition
    actual = answers.get(name)
    if operator == "=":
        return actual == expected
    if operator == "!=":
        return actual != expected
    raise SystemExit(f"show_if operator '{operator}' is not supported here")


def visible_answers(
    questions: list[dict[str, Any]], answers: Mapping[str, Any]
) -> dict[str, Any]:
    """``answers`` without the questions the form would hide, evaluated top to bottom.

    A question hidden by an earlier hidden question is hidden as well: TrueNAS evaluates
    ``show_if`` against what is visible, not against defaults of fields the user never saw.
    """
    shown: dict[str, Any] = {}
    for question in questions:
        name = question["variable"]
        conditions = (question.get("schema") or {}).get("show_if") or []
        if all(_condition_holds(shown, c) for c in conditions):
            if name in answers:
                shown[name] = answers[name]
    for extra, value in answers.items():  # injected by TrueNAS, not a question
        if extra not in {q["variable"] for q in questions}:
            shown[extra] = value
    return shown


def render(
    answers: Mapping[str, Any],
    *,
    questions: list[dict[str, Any]] | None = None,
    strict: bool = True,
    template: Path = TEMPLATE,
) -> str:
    """Render the template. ``strict`` hides ``show_if`` questions and rejects unknown names."""
    jinja2 = _jinja()
    questions = load_questions() if questions is None else questions
    values = visible_answers(questions, answers) if strict else dict(answers)
    environment = jinja2.Environment(
        undefined=jinja2.StrictUndefined if strict else jinja2.Undefined,
        keep_trailing_newline=True,
    )
    return environment.from_string(template.read_text(encoding="utf-8")).render(values=values)


def _parse_assignment(text: str) -> tuple[str, str]:
    name, sep, value = text.partition("=")
    if not sep or not name:
        raise SystemExit(f"--set expects name=value, got '{text}'")
    return name, value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--set", action="append", default=[], metavar="NAME=VALUE",
                        help="answer one question (repeatable)")
    parser.add_argument("--answers", type=Path, help="JSON file of answers, applied before --set")
    parser.add_argument("--ix-volume-path", metavar="PATH",
                        help="inject ix_volumes for the ixVolume storage answer")
    parser.add_argument("--lenient", action="store_true",
                        help="render hidden questions too and print unknown names as empty")
    args = parser.parse_args(argv)

    questions = load_questions()
    overrides: dict[str, Any] = {}
    if args.answers:
        overrides.update(json.loads(args.answers.read_text(encoding="utf-8")))
    by_name = {q["variable"]: q for q in questions}
    for item in args.set:
        name, value = _parse_assignment(item)
        if name not in by_name:
            raise SystemExit(f"no such question: {name} (see questions.yaml)")
        overrides[name] = value
    answers = default_answers(questions, overrides)
    if args.ix_volume_path:
        dataset = answers.get("ix_volume_dataset", "data")
        answers["ix_volumes"] = {dataset: {"host_path": args.ix_volume_path}}
    sys.stdout.write(render(answers, questions=questions, strict=not args.lenient))
    return 0


if __name__ == "__main__":
    sys.exit(main())
