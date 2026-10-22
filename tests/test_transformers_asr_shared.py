"""Drift guards for what the two transformers ASR services share.

cohere-asr-service and granite-asr-service build their images from their own
directories, so the modules they share exist twice (and ``audio_pipeline.py`` three
times: it is qwen3-asr-service's, copied unchanged). Nothing but this file keeps the
copies the same. The two images are also meant to be the same up to their own code
(one torch layer, one pip layer), which a host pulling both downloads once.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SERVICES = ["cohere-asr-service", "granite-asr-service"]
OWN_MODULE = {"cohere-asr-service": "cohere_support.py", "granite-asr-service": "granite_support.py"}


def _text(service: str, name: str) -> str:
    return (REPO_ROOT / service / name).read_text(encoding="utf-8").replace("\r\n", "\n")


def _instructions(dockerfile: Path) -> list[tuple[str, str]]:
    """(INSTRUCTION, arguments) with continuations folded in and comments dropped."""
    text = dockerfile.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\\\n", " ")
    out = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        keyword, _, rest = line.partition(" ")
        out.append((keyword.upper(), rest.strip()))
    return out


@pytest.mark.parametrize("name", ["transformers_asr.py", "requirements.txt"])
def test_the_shared_files_are_the_same_in_both_services(name):
    first, second = (_text(service, name) for service in SERVICES)
    assert first == second, (
        f"{name} differs between {SERVICES[0]} and {SERVICES[1]}. It is one file in two build "
        "contexts: a fix applied to one belongs in both.")


def test_audio_pipeline_is_qwen3_asrs_unchanged():
    reference = _text("qwen3-asr-service", "audio_pipeline.py")
    for service in SERVICES:
        assert _text(service, "audio_pipeline.py") == reference, (
            f"{service}/audio_pipeline.py has drifted from qwen3-asr-service's: the upload, decode and "
            "quiet-point chunking code is shared on purpose, so a fix belongs in all three copies.")


def test_both_images_are_one_recipe_up_to_their_own_code():
    """Same instructions up to the build-time import check, so every heavy layer is shared."""
    prefixes = []
    for service in SERVICES:
        steps = _instructions(REPO_ROOT / service / "Dockerfile")
        first_app_step = next(i for i, (kw, args) in enumerate(steps) if kw == "RUN" and "from transformers import" in args)
        prefixes.append(steps[:first_app_step])
    assert prefixes[0] == prefixes[1], "the two Dockerfiles diverge before their own code"


@pytest.mark.parametrize("service", SERVICES)
def test_each_image_ships_its_modules_and_checks_its_classes_at_build_time(service):
    steps = _instructions(REPO_ROOT / service / "Dockerfile")
    copied = {args.split()[0] for kw, args in steps if kw == "COPY"}
    for module in ("app.py", "audio_pipeline.py", "transformers_asr.py", OWN_MODULE[service],
                   "model_lifecycle.py", "body_limit.py", "origin_guard.py"):
        assert module in copied, f"{service}/Dockerfile does not COPY {module}"
    check = next(args for kw, args in steps if kw == "RUN" and "from transformers import" in args)
    if service == "cohere-asr-service":
        assert "CohereAsrForConditionalGeneration" in check and "CohereAsrProcessor" in check
    else:
        assert "GraniteSpeechForConditionalGeneration" in check and "GraniteSpeech5ForCTC" in check
    assert "load_silero_vad()" in check, "the VAD's bundled model must load at build time"


@pytest.mark.parametrize("service", SERVICES)
def test_uvicorn_keeps_connections_alive_longer_than_the_gateways_pool(service):
    cmds = [args for keyword, args in _instructions(REPO_ROOT / service / "Dockerfile") if keyword == "CMD"]
    assert len(cmds) == 1
    argv = json.loads(cmds[0])
    assert int(argv[argv.index("--timeout-keep-alive") + 1]) >= 60


def test_transformers_is_pinned_exactly_and_new_enough_for_both_models():
    """The model code lives in transformers: a floating version changes what is transcribed."""
    from packaging.version import Version

    lines = [line.strip() for line in _text(SERVICES[0], "requirements.txt").splitlines()
             if line.strip() and not line.startswith("#")]
    pin = next(line for line in lines if line.startswith("transformers"))
    assert pin.startswith("transformers=="), pin
    assert Version(pin.split("==", 1)[1]) >= Version("5.16.0"), "GraniteSpeech5ForCTC needs 5.16"
    assert not any(line.startswith(("torch", "torchaudio")) for line in lines), \
        "torch is installed from the cu128 index by the Dockerfile, not by pip from PyPI"
