"""Drift guards for vllm-asr-service: what it shares with the other ASR services, and its image.

The service builds from its own directory, so the modules it reuses are copies:
``transformers_asr.py`` (the speech gate, chunk planner, weight check) and
``cohere_support.py``/``granite_support.py`` (the Cohere mirror pin and digest, Granite's
prompt and keywords) are cohere/granite's, ``audio_pipeline.py`` is qwen3-asr's. Nothing
but this file keeps them the same. The image is not built by CI (its vLLM base is
32 GB unpacked); tests/test_docker_image_contents.py checks the COPY lines, this file
the rest of the recipe.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from compose_helpers import load_compose

REPO_ROOT = Path(__file__).resolve().parents[1]
SERVICE = REPO_ROOT / "vllm-asr-service"
VLLM_IMAGE = "vllm/vllm-openai:v0.31.0"


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8").replace("\r\n", "\n")


@pytest.mark.parametrize("name, origin", [
    ("transformers_asr.py", "granite-asr-service"),
    ("cohere_support.py", "cohere-asr-service"),
    ("granite_support.py", "granite-asr-service"),
    ("audio_pipeline.py", "qwen3-asr-service"),
])
def test_the_reused_modules_are_unchanged_copies(name, origin):
    assert _text(SERVICE / name) == _text(REPO_ROOT / origin / name), (
        f"vllm-asr-service/{name} has drifted from {origin}'s: it is one file in several build "
        "contexts, so a fix belongs in every copy.")


def test_the_vad_is_the_release_the_transformers_services_pin():
    def pin(path: Path) -> str:
        return re.search(r"^silero-vad==(\S+)$", _text(path), re.M).group(1)

    assert pin(SERVICE / "requirements.txt") == pin(REPO_ROOT / "granite-asr-service" / "requirements.txt")


def _dockerfile() -> str:
    return _text(SERVICE / "Dockerfile")


def test_the_engine_is_the_pinned_vllm_release():
    froms = re.findall(r"^FROM\s+(\S+)", _dockerfile(), re.M)
    assert froms == [VLLM_IMAGE]


def test_the_vad_is_installed_without_touching_the_engines_torch_and_checked_at_build_time():
    text = _dockerfile()
    assert "pip install --no-cache-dir --no-deps -r requirements.txt" in text
    assert "load_silero_vad()" in text and "torch.cuda.is_initialized()" in text


def test_the_container_runs_the_adapter_not_the_bases_vllm_serve():
    text = _dockerfile()
    assert re.search(r"^ENTRYPOINT \[\]$", text, re.M), "the base's entrypoint is `vllm serve`"
    cmd = re.search(r"^CMD (.+)$", text, re.M).group(1)
    assert '"app:app"' in cmd and '"5011"' in cmd
    # The gateway pools connections for 115 s; uvicorn's 5 s default closes them first.
    assert '"--timeout-keep-alive", "120"' in cmd
    assert "localhost:5011/health" in text


def test_compose_gives_it_its_own_concurrency_and_room_to_shut_the_engine_down():
    service = load_compose(REPO_ROOT / "docker-compose.yml")["services"]["vllm-asr-service"]
    env = service["environment"]
    # The other services' ASR_MAX_CONCURRENCY=1 would serialise what this one batches.
    assert env["ASR_MAX_CONCURRENCY"] == "${VLLM_ASR_MAX_CONCURRENCY:-32}"
    assert env["VLLM_ASR_GPU_MEMORY_UTILIZATION"] == "${VLLM_ASR_GPU_MEMORY_UTILIZATION:-0.45}"
    assert service["stop_grace_period"] == "60s"
    assert service["build"]["context"] == "./vllm-asr-service"
    assert "shm_size" in service
