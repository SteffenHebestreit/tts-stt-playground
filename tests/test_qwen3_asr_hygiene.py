"""What the Qwen3-ASR service tells an anonymous caller when something fails.

Finding covered: a 500 carried ``str(exc)`` (temp-file paths, model paths, library
internals), a batch item's error the same, /ready repeated the exception text of a
failed load as ``detail``, and a model configured as a checkpoint directory was
printed in /ready and /status. Now an error names a request id and the exception
text is in the log under it; /ready says a category; a path is shown as its last
component.
"""

from __future__ import annotations

import logging
import re

import pytest

from test_qwen3_asr_harness import (
    encoded_wav, install_fake_qwen_asr, load_app, make_client, run, speech_levels, tmp_files, use_tmpdir,
)

CLIP = encoded_wav(speech_levels(10))
LEAK = "/tmp/tmpab12cd.wav"


def build(tmp_path, monkeypatch, *, model_options=None, **env):
    tmp = tmp_path / "tmp"
    tmp.mkdir()
    use_tmpdir(monkeypatch, tmp)
    env.setdefault("ASR_MODEL_TTL", "-1")
    module = load_app(**env)
    factory = install_fake_qwen_asr(monkeypatch, **(model_options or {}))
    monkeypatch.setattr(module, "_FFMPEG", None, raising=False)
    return module, factory, tmp


def _request_id(text):
    match = re.search(r"Request id: ([0-9a-f]{12})\.", text)
    assert match, f"no request id in {text!r}"
    return match.group(1)


@pytest.mark.parametrize("path, field", [("/transcribe", "audio"), ("/detect_language", "file")])
def test_a_crash_is_answered_with_a_request_id_and_the_rest_is_logged(tmp_path, monkeypatch, caplog, path, field):
    module, *_ = build(tmp_path, monkeypatch, model_options={"error": RuntimeError(f"generate exploded on {LEAK}")})

    async def scenario():
        async with make_client(module) as client:
            return await client.post(path, files={field: ("clip.wav", CLIP, "audio/wav")})

    with caplog.at_level(logging.ERROR):
        response = run(scenario())

    assert response.status_code == 500
    detail = response.json()["detail"]
    assert "generate exploded" not in detail and "/tmp" not in detail
    assert _request_id(detail) in caplog.text and f"generate exploded on {LEAK}" in caplog.text


def test_a_batch_item_error_names_the_request_id_not_the_exception(tmp_path, monkeypatch, caplog):
    module, *_, tmp = build(tmp_path, monkeypatch, model_options={"error": RuntimeError(f"generate exploded on {LEAK}")})

    async def scenario():
        async with make_client(module) as client:
            files = [("audios", (f"c{i}.wav", CLIP, "audio/wav")) for i in range(2)]
            return await client.post("/transcribe-batch", files=files)

    with caplog.at_level(logging.ERROR):
        response = run(scenario())

    results = response.json()["results"]
    assert [r["status"] for r in results] == [500, 500]
    assert all("generate exploded" not in r["error"] and "/tmp" not in r["error"] for r in results)
    assert _request_id(results[0]["error"]) in caplog.text
    assert tmp_files(tmp) == []


def test_a_gpu_out_of_memory_keeps_its_advice(tmp_path, monkeypatch):
    module, *_ = build(tmp_path, monkeypatch,
                       model_options={"error": RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")})

    async def scenario():
        async with make_client(module) as client:
            return await client.post("/transcribe", files={"audio": ("clip.wav", CLIP, "audio/wav")})

    response = run(scenario())

    assert response.status_code == 503 and "QWEN3_ASR_BATCH_SIZE" in response.json()["detail"]


@pytest.mark.parametrize("error, category", [
    (MemoryError(), "out_of_memory"),
    (ModuleNotFoundError("No module named 'qwen_asr'"), "missing_dependency"),
    (FileNotFoundError("/models/qwen3-asr/config.json"), "model_files_unavailable"),
    (RuntimeError("weights are corrupt"), "load_error"),
])
def test_ready_says_a_category_for_a_failed_load_and_never_the_exception(tmp_path, monkeypatch, error, category):
    module, factory, _ = build(tmp_path, monkeypatch)
    factory.fail = error

    async def scenario():
        import asyncio
        async with make_client(module) as client:
            with pytest.raises(type(error)):
                await asyncio.to_thread(module.get_model)
            return await client.get("/ready")

    ready = run(scenario())

    assert ready.status_code == 503 and ready.json()["reason"] == "load_failed"
    assert ready.json()["detail"] == category
    assert "/models" not in ready.text and "corrupt" not in ready.text


def test_a_model_configured_as_a_path_is_shown_as_its_last_component(tmp_path, monkeypatch):
    module, *_ = build(tmp_path, monkeypatch, QWEN3_ASR_MODEL="/models/finetunes/qwen3-asr-de")

    async def scenario():
        async with make_client(module) as client:
            return await client.get("/ready"), await client.get("/status")

    ready, status = run(scenario())

    assert ready.json()["current_model"] == status.json()["current_model"] == "qwen3-asr-de"
    assert "/models/finetunes" not in ready.text + status.text
