"""What the two NeMo ASR services tell an anonymous caller when something fails.

Finding covered: a 500 carried ``str(exc)`` (temp-file paths, model paths, library
internals), a per-file batch error the same, /ready repeated the exception text of a
failed load as ``detail``, and a model configured as a path on a mounted volume was
printed in /ready, /status and every transcription. Now an error names a request id
and the exception text is in the log under it; /ready says a category
(``model_files_unavailable``, ``out_of_memory``, ``missing_dependency``,
``load_error``); a path is shown as its file name.

What stays: the 503 for a GPU out of memory keeps its advice, an error the service
wrote for the caller (413, 422, "no transcription") is passed on as written, and the
``/ready`` reasons stay ``loading`` / ``load_failed`` / ``ok``.
"""

from __future__ import annotations

import logging
import re

import pytest

from test_nemo_services_harness import (
    FakeCanary, FakeParakeet, OutOfMemoryError, hypothesis, install_model, load_app, make_client,
    run, wav_bytes,
)

SERVICES = ["parakeet", "canary"]
MODELS = {"parakeet": FakeParakeet, "canary": FakeCanary}
LEAK = "/tmp/tmpab12cd.wav"
MODEL_PATH = "/models/finetunes/parakeet-primeline.nemo"


@pytest.fixture(params=SERVICES)
def service(request):
    return request.param


def build(service, monkeypatch, **env):
    module = load_app(service, **env)
    monkeypatch.setattr(module, "_FFMPEG", None)
    model = MODELS[service]()
    install_model(module, model)
    return module, model


def _post(path, field):
    async def send(client):
        return await client.post(path, files={field: ("clip.wav", wav_bytes(1.0), "audio/wav")})
    return send


ROUTES = {
    "transcribe": _post("/transcribe", "audio"),
    "openai": _post("/v1/audio/transcriptions", "file"),
    "detect": _post("/detect_language", "file"),
}


def _request_id(text):
    match = re.search(r"Request id: ([0-9a-f]{12})\.", text)
    assert match, f"no request id in {text!r}"
    return match.group(1)


@pytest.mark.parametrize("route", sorted(ROUTES))
def test_every_route_answers_a_crash_with_a_request_id_and_logs_the_rest(service, monkeypatch, caplog, route):
    module, model = build(service, monkeypatch)

    def explode(paths, kwargs):
        raise RuntimeError(f"decoder exploded reading {LEAK}")

    model.reply = explode

    async def scenario():
        async with make_client(module) as client:
            return await ROUTES[route](client)

    with caplog.at_level(logging.ERROR):
        response = run(scenario())

    assert response.status_code == 500
    detail = response.json()["detail"]
    assert "decoder exploded" not in detail and "/tmp" not in detail
    assert _request_id(detail) in caplog.text and f"decoder exploded reading {LEAK}" in caplog.text


def test_a_batch_reports_every_failed_file_under_one_request_id(service, monkeypatch, caplog):
    module, model = build(service, monkeypatch)

    def explode(paths, kwargs):
        raise RuntimeError(f"decoder exploded reading {LEAK}")

    model.reply = explode

    async def scenario():
        async with make_client(module) as client:
            files = [("audios", (f"c{i}.wav", wav_bytes(1.0), "audio/wav")) for i in range(2)]
            return await client.post("/transcribe-batch", files=files)

    with caplog.at_level(logging.ERROR):
        response = run(scenario())

    assert response.status_code == 200
    errors = [r["error"] for r in response.json()["results"]]
    assert len(errors) == 2 and all(r["status"] == 500 for r in response.json()["results"])
    assert all("decoder exploded" not in e and "/tmp" not in e for e in errors)
    assert _request_id(errors[0]) in caplog.text
    assert "decoder exploded reading" in caplog.text


def test_a_file_that_cannot_be_prepared_does_not_leak_why(service, monkeypatch, caplog):
    module, model = build(service, monkeypatch)

    def broken_prepared_upload(upload):
        raise OSError(f"[Errno 28] No space left on device: {LEAK!r}")

    monkeypatch.setattr(module, "_prepared_upload", broken_prepared_upload)

    async def scenario():
        async with make_client(module) as client:
            files = [("audios", ("a.wav", wav_bytes(1.0), "audio/wav"))]
            return await client.post("/transcribe-batch", files=files)

    with caplog.at_level(logging.ERROR):
        response = run(scenario())

    (result,) = response.json()["results"]
    assert result["status"] == 500
    assert "No space left" not in result["error"] and "/tmp" not in result["error"]
    assert "No space left on device" in caplog.text


def test_a_gpu_out_of_memory_keeps_its_advice(service, monkeypatch):
    module, model = build(service, monkeypatch, cuda=True)

    def explode(paths, kwargs):
        raise OutOfMemoryError("CUDA out of memory. Tried to allocate 9.00 GiB")

    model.reply = explode

    async def scenario():
        async with make_client(module) as client:
            return await ROUTES["transcribe"](client)

    response = run(scenario())

    assert response.status_code == 503
    assert "out of memory" in response.json()["detail"].lower() and "NEMO_MAX_AUDIO_S" in response.json()["detail"]


def test_parakeet_says_what_it_wrote_for_the_client_when_the_model_returns_nothing(monkeypatch):
    module, model = build("parakeet", monkeypatch)
    model.reply = lambda paths, kwargs: [hypothesis("only one")]

    async def scenario():
        async with make_client(module) as client:
            files = [("audios", (f"c{i}.wav", wav_bytes(1.0), "audio/wav")) for i in range(2)]
            return await client.post("/transcribe-batch", files=files)

    results = run(scenario()).json()["results"]

    assert results[0]["text"] == "only one"
    assert results[1]["status"] == 500 and "no transcription" in results[1]["error"]
    assert "Request id:" in results[1]["error"]


@pytest.mark.parametrize("error, category", [
    (MemoryError(), "out_of_memory"),
    (ModuleNotFoundError("No module named 'nemo'"), "missing_dependency"),
    (FileNotFoundError(MODEL_PATH), "model_files_unavailable"),
    (ConnectionError("huggingface.co unreachable"), "model_files_unavailable"),
    (RuntimeError("weights are corrupt"), "load_error"),
])
def test_ready_says_a_category_for_a_failed_load_and_never_the_exception(service, monkeypatch, error, category):
    module, model = build(service, monkeypatch)

    def failing_loader():
        raise error

    module._model_slot = module.ModelSlot(failing_loader, ttl_seconds=-1, name="test")

    async def scenario():
        async with make_client(module) as client:
            with pytest.raises(type(error)):
                await run_in_thread(module.get_model)
            return await client.get("/ready")

    ready = run(scenario())

    assert ready.status_code == 503
    assert ready.json()["reason"] == "load_failed" and ready.json()["ready"] is False
    assert ready.json()["detail"] == category
    assert "/models" not in ready.text and "huggingface" not in ready.text and "corrupt" not in ready.text


async def run_in_thread(fn):
    import asyncio
    return await asyncio.to_thread(fn)


def test_a_model_configured_as_a_path_is_shown_as_its_file_name(monkeypatch):
    env = {"PARAKEET_ASR_MODEL": MODEL_PATH}
    module, model = build("parakeet", monkeypatch, **env)
    model.reply = lambda paths, kwargs: [hypothesis("hallo")]

    async def scenario():
        async with make_client(module) as client:
            return (await client.get("/ready"), await client.get("/status"),
                    await ROUTES["transcribe"](client))

    ready, status, transcribed = run(scenario())

    assert ready.json()["current_model"] == status.json()["current_model"] == "parakeet-primeline.nemo"
    assert transcribed.json()["model"] == "parakeet-primeline.nemo"
    assert "/models/finetunes" not in ready.text + status.text + transcribed.text


def test_canary_does_not_print_a_model_path_in_its_language_refusal(monkeypatch):
    module, model = build("canary", monkeypatch, CANARY_ASR_MODEL="/models/finetunes/canary-de.nemo")

    async def scenario():
        async with make_client(module) as client:
            return await client.post(
                "/transcribe", data={"language": "it"},
                files={"audio": ("clip.wav", wav_bytes(1.0), "audio/wav")})

    response = run(scenario())

    assert response.status_code == 422
    assert "canary-de.nemo" in response.json()["detail"] and "/models/finetunes" not in response.json()["detail"]


def test_a_hugging_face_id_is_shown_as_it_is(service, monkeypatch):
    module, model = build(service, monkeypatch)

    async def scenario():
        async with make_client(module) as client:
            return await client.get("/status")

    name = run(scenario()).json()["current_model"]

    assert name.startswith("nvidia/")
