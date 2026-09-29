"""Drift guards for what the two NeMo ASR services share.

`nemo_common.py` exists twice because each image is built from its own
directory; nothing but this file keeps the copies the same. The Dockerfile
checks parse the instructions (CMD is JSON) rather than grepping the text.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
NEMO_SERVICES = ["parakeet-asr-service", "canary-asr-service"]


def _dockerfiles():
    return [
        (service, path)
        for service in NEMO_SERVICES
        for path in sorted((REPO_ROOT / service).glob("Dockerfile*"))
    ]


def _instructions(dockerfile: Path) -> list[tuple[str, str]]:
    """(INSTRUCTION, arguments) with continuations folded in and comments dropped."""
    text = dockerfile.read_text(encoding="utf-8").replace("\\\n", " ")
    out = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        keyword, _, rest = line.partition(" ")
        out.append((keyword.upper(), rest.strip()))
    return out


def test_nemo_common_is_the_same_file_in_both_services():
    copies = {s: (REPO_ROOT / s / "nemo_common.py").read_text(encoding="utf-8") for s in NEMO_SERVICES}
    first, second = copies.values()
    assert first == second, (
        "nemo_common.py differs between parakeet-asr-service and canary-asr-service. "
        "It is one module in two build contexts: a fix applied to one belongs in both."
    )


@pytest.mark.parametrize("service, dockerfile", _dockerfiles(), ids=lambda v: getattr(v, "name", v))
def test_every_image_variant_ships_the_shared_module(service, dockerfile):
    copied = [args.split()[0] for keyword, args in _instructions(dockerfile) if keyword == "COPY"]
    assert "nemo_common.py" in copied, f"{dockerfile.name} does not COPY nemo_common.py; the app cannot import"


@pytest.mark.parametrize("service, dockerfile", _dockerfiles(), ids=lambda v: getattr(v, "name", v))
def test_uvicorn_keeps_connections_alive_longer_than_the_gateways_pool(service, dockerfile):
    cmds = [args for keyword, args in _instructions(dockerfile) if keyword == "CMD"]
    assert len(cmds) == 1
    argv = json.loads(cmds[0])
    assert "--timeout-keep-alive" in argv, "uvicorn's 5 s default closes pooled gateway connections first"
    assert int(argv[argv.index("--timeout-keep-alive") + 1]) >= 60


@pytest.mark.parametrize("service, dockerfile", _dockerfiles(), ids=lambda v: getattr(v, "name", v))
def test_torchscript_stays_disabled(service, dockerfile):
    """torch 2.11's TorchScript compiler segfaults scripting NeMo's online_clustering.py at import."""
    envs = [args for keyword, args in _instructions(dockerfile) if keyword == "ENV"]
    assert any(env.replace("=", " ").split()[:2] == ["PYTORCH_JIT", "0"] for env in envs)
