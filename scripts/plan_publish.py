#!/usr/bin/env python3
"""Decide what .github/workflows/publish-images.yml builds and which tags it may move.

The workflow used to hard-code this in YAML and got three things wrong at once:
a manual run from any branch published ``latest``, one failing CUDA build blocked
every other image, and nothing said which images a release consists of. The
decisions live here, where a test can call them, and the workflow only reads the
result.

Tag policy (the tags themselves are rendered by docker/metadata-action from the
flags below):

==================================  =====================================================
Trigger                             Tags
==================================  =====================================================
push of tag ``vX.Y.Z``              ``X.Y.Z``, ``X.Y``, ``latest``, ``sha-<short>``
push of tag ``vX.Y.Z-rc.1``         ``X.Y.Z-rc.1``, ``sha-<short>`` (never ``latest``)
push to ``master``                  ``latest``, ``master``, ``sha-<short>``
manual run, any ref                 ``sha-<short>``, ``<branch>`` and an optional extra
                                    tag; NEVER ``latest``, never a version-shaped tag
==================================  =====================================================

``latest`` therefore moves only when a release is cut or master advances, both of
which passed the CI checks first. Deployments that must be reproducible pin
``IMAGE_TAG=X.Y.Z`` (the TrueNAS files do).

Parakeet and Canary also get a rollback image built against NeMo 2.x, tagged like
the main one plus ``-nemo2`` (``0.2.0-nemo2``). It costs two more large builds, so
it is built for release tags and on request, not for every master push.

Inputs come from the environment, never from shell interpolation of workflow
expressions: GITHUB_EVENT_NAME, GITHUB_REF, GITHUB_REPOSITORY_OWNER,
PUBLISH_EXTRA_TAG, PUBLISH_NEMO2. Outputs go to $GITHUB_OUTPUT as ``build`` and
``merge`` (matrix JSON) and ``release``, ``latest``, ``nemo2``, ``extra_tag``,
``image_prefix``.

    python scripts/plan_publish.py --event push --ref refs/tags/v0.2.0 --owner SteffenHebestreit
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Optional

# (image suffix after "tts-stt-", build context, architectures). The three that
# can run on an ARM SBC are multi-arch; everything else sits on an nvidia/cuda
# base or needs CUDA and is amd64 only ON PURPOSE: an arm64 image of those would
# build or start wrongly and look supported.
SERVICES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("frontend", "./frontend-service", ("amd64", "arm64")),
    ("piper-tts", "./piper-tts-service", ("amd64", "arm64")),
    ("whisper-cpp", "./whisper-cpp-service", ("amd64", "arm64")),
    ("stt", "./stt-service", ("amd64",)),
    ("qwen3-asr", "./qwen3-asr-service", ("amd64",)),
    ("qwen3-tts", "./qwen3-tts-service", ("amd64",)),
    ("parakeet-asr", "./parakeet-asr-service", ("amd64",)),
    ("canary-asr", "./canary-asr-service", ("amd64",)),
    ("chatterbox-tts", "./chatterbox-tts-service", ("amd64",)),
    ("magpie-tts", "./magpie-tts-service", ("amd64",)),
    ("piper-training", "./piper-training-service", ("amd64",)),
)

RUNNERS = {"amd64": "ubuntu-latest", "arm64": "ubuntu-24.04-arm"}

# The rollback line for the two NeMo services. docker-compose.yml selects it with
# NEMO_IMAGE_SUFFIX=-nemo2; the specifier is the one the READMEs document.
NEMO2_SERVICES = ("parakeet-asr", "canary-asr")
NEMO2_SUFFIX = "-nemo2"
NEMO2_SPEC = ">=2.7.3,<3"

_RELEASE_TAG = re.compile(r"^v(\d+)\.(\d+)\.(\d+)(-[0-9A-Za-z][0-9A-Za-z.-]*)?$")
# What Docker accepts as a tag.
_DOCKER_TAG = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}$")
# Anything that reads like a version belongs to a release tag: a manual run must
# not be able to move the tag a TrueNAS install has pinned.
_VERSION_LIKE = re.compile(r"^v?\d+\.\d+(\.\d+)?([-+].*)?$")


class PlanError(ValueError):
    """The trigger or its inputs break the publishing policy; the run must stop."""


@dataclass
class Plan:
    build: list[dict] = field(default_factory=list)
    merge: list[dict] = field(default_factory=list)
    release: bool = False
    latest: bool = False
    nemo2: bool = False
    extra_tag: str = ""

    def outputs(self) -> dict[str, str]:
        return {
            "build": json.dumps({"include": self.build}, separators=(",", ":")),
            "merge": json.dumps({"include": self.merge}, separators=(",", ":")),
            "release": str(self.release).lower(),
            "latest": str(self.latest).lower(),
            "nemo2": str(self.nemo2).lower(),
            "extra_tag": self.extra_tag,
        }


def _truthy(value: Optional[str]) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _check_extra_tag(tag: str) -> str:
    tag = (tag or "").strip()
    if not tag:
        return ""
    if tag.lower() == "latest":
        raise PlanError(
            "a manual run never publishes 'latest': it would move the tag every "
            "default deployment follows from an arbitrary branch. Tag a release "
            "(vX.Y.Z) or merge to master.")
    if not _DOCKER_TAG.match(tag):
        raise PlanError(f"'{tag}' is not a valid Docker tag")
    if _VERSION_LIKE.match(tag):
        raise PlanError(
            f"'{tag}' looks like a release version. Version tags are written only by a "
            "release (a pushed vX.Y.Z tag), so that a pinned deployment never changes.")
    return tag


def make_plan(event: str, ref: str, extra_tag: str = "", nemo2: bool = False) -> Plan:
    """The build and merge matrices and the tag flags for one trigger, or PlanError."""
    plan = Plan()

    if event == "push":
        if ref.startswith("refs/tags/"):
            name = ref.removeprefix("refs/tags/")
            match = _RELEASE_TAG.match(name)
            if not match:
                raise PlanError(
                    f"tag '{name}' is not a release tag. Release tags look like v1.2.3 "
                    "(or v1.2.3-rc.1 for a prerelease).")
            plan.release = True
            plan.latest = match.group(4) is None
            plan.nemo2 = True
        elif ref == "refs/heads/master":
            plan.latest = True
        else:
            raise PlanError(f"a push to '{ref}' is not published: only master and release tags are")
        if extra_tag:
            raise PlanError("an extra tag is only meaningful for a manual run")
    elif event == "workflow_dispatch":
        plan.extra_tag = _check_extra_tag(extra_tag)
        plan.nemo2 = bool(nemo2)
    else:
        raise PlanError(f"the event '{event}' does not publish images")

    for name, context, archs in SERVICES:
        for arch in archs:
            plan.build.append({
                "name": name, "key": name, "suffix": "", "arch": arch,
                "runner": RUNNERS[arch], "context": context,
                "artifact": f"digests-{name}_{arch}", "build_args": "",
                "cache_from": f"type=gha,scope={name}-{arch}",
                "cache_to": f"type=gha,mode=max,scope={name}-{arch}",
            })
        plan.merge.append({"name": name, "key": name, "suffix": "", "archs": " ".join(archs)})

    if plan.nemo2:
        for name, context, archs in SERVICES:
            if name not in NEMO2_SERVICES:
                continue
            key = f"{name}{NEMO2_SUFFIX}"
            for arch in archs:
                plan.build.append({
                    "name": name, "key": key, "suffix": NEMO2_SUFFIX, "arch": arch,
                    "runner": RUNNERS[arch], "context": context,
                    "artifact": f"digests-{key}_{arch}",
                    "build_args": f"NEMO_TOOLKIT_SPEC={NEMO2_SPEC}",
                    # Reads the main image's layers (torch is the expensive one) and
                    # writes nothing: the runner cache is small and two variants
                    # exporting to one scope would evict each other.
                    "cache_from": f"type=gha,scope={name}-{arch}",
                    "cache_to": "",
                })
            plan.merge.append({"name": name, "key": key, "suffix": NEMO2_SUFFIX,
                               "archs": " ".join(archs)})
    return plan


def image_prefix(owner: str) -> str:
    """Registry path every image name is appended to.

    Lower-cased because GHCR and BuildKit reject a repository name with capitals,
    and ``github.repository_owner`` keeps the account's own capitalisation.
    """
    owner = owner.strip().lower()
    if not owner:
        raise PlanError("the repository owner is unknown (GITHUB_REPOSITORY_OWNER / --owner)")
    return f"ghcr.io/{owner}/tts-stt-"


def _write_outputs(outputs: dict[str, str]) -> None:
    path = os.environ.get("GITHUB_OUTPUT")
    lines = [f"{key}={value}" for key, value in outputs.items()]
    if path:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")
    else:
        print("\n".join(lines))


def _summary(plan: Plan, event: str, ref: str) -> str:
    services = sorted({leg["name"] for leg in plan.build})
    rows = [
        "## Publish plan", "",
        f"- trigger: `{event}` on `{ref}`",
        f"- version tags (`X.Y.Z`, `X.Y`): {'yes' if plan.release else 'no'}",
        f"- `latest` moves: {'yes' if plan.latest else 'no'}",
        f"- extra tag: {f'`{plan.extra_tag}`' if plan.extra_tag else 'none'}",
        f"- NeMo 2.x rollback images (`-nemo2`): {'yes' if plan.nemo2 else 'no'}",
        f"- images ({len(services)}): {', '.join(services)}",
        f"- build legs: {len(plan.build)}",
    ]
    return "\n".join(rows) + "\n"


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--event", default=os.environ.get("GITHUB_EVENT_NAME", ""))
    parser.add_argument("--ref", default=os.environ.get("GITHUB_REF", ""))
    parser.add_argument("--owner", default=os.environ.get("GITHUB_REPOSITORY_OWNER", ""))
    parser.add_argument("--extra-tag", default=os.environ.get("PUBLISH_EXTRA_TAG", ""))
    parser.add_argument("--nemo2", action="store_true", default=_truthy(os.environ.get("PUBLISH_NEMO2")))
    args = parser.parse_args(argv)

    try:
        plan = make_plan(args.event, args.ref, args.extra_tag, args.nemo2)
        prefix = image_prefix(args.owner)
    except PlanError as error:
        print(f"::error title=publish plan::{error}" if os.environ.get("GITHUB_ACTIONS") == "true"
              else f"error: {error}", file=sys.stderr)
        return 1

    _write_outputs({**plan.outputs(), "image_prefix": prefix})
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as handle:
            handle.write(_summary(plan, args.event, args.ref))
    return 0


if __name__ == "__main__":
    sys.exit(main())
