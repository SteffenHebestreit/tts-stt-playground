"""Readiness, startup preload and inference tuning of the two NeMo ASR services.

Findings covered:

* /health was the only probe, so "healthy container, model not loaded (yet, or
  ever)" was indistinguishable from "ready"; /ready now reports it from the
  ``ModelSlot`` introspection, and /health stays 200 whatever the model is doing;
* the startup preload ran before the server accepted connections, leaving the
  port closed for the length of a first-time download; it now runs in the
  background;
* TF32 matmuls (``NEMO_MATMUL_PRECISION``) and opt-in bf16 (``NEMO_BF16``) are
  applied at load, bf16 only where the GPU can run it.
"""

from __future__ import annotations

import asyncio
import threading

import pytest

from test_nemo_services_harness import (
    WAIT, FakeCanary, FakeParakeet, install_fake_nemo, install_model, load_app, make_client, run,
    wait_for, wav_bytes,
)

SERVICES = ["parakeet", "canary"]
MODELS = {"parakeet": FakeParakeet, "canary": FakeCanary}
LOADERS = {"parakeet": "_load_parakeet", "canary": "_load_canary"}


@pytest.fixture(params=SERVICES)
def service(request):
    return request.param


class Loader:
    """A model loader the test controls: blocks until released, can fail."""

    def __init__(self, service):
        self.model = MODELS[service]()
        self.started = threading.Event()
        self.release = threading.Event()
        self.release.set()  # by default it does not block
        self.error: Exception | None = None
        self.calls = 0

    def __call__(self):
        self.calls += 1
        self.started.set()
        assert self.release.wait(WAIT), "the test never released the loader"
        if self.error is not None:
            raise self.error
        return self.model


def with_loader(service, monkeypatch, **env):
    module = load_app(service, **env)
    monkeypatch.setattr(module, "_FFMPEG", None)
    loader = Loader(service)
    hook = next(getattr(module, n) for n in dir(module) if n.startswith("_release_"))
    module._model_slot = module.ModelSlot(
        loader, ttl_seconds=module.MODEL_TTL, name="test", on_unload=hook)
    return module, loader


async def get(client, path):
    return await client.get(path)


# --- /ready ----------------------------------------------------------------------------

def test_a_service_that_has_not_been_asked_to_load_yet_is_ready(service, monkeypatch):
    module, loader = with_loader(service, monkeypatch)

    async def scenario():
        async with make_client(module) as client:
            return await get(client, "/ready")

    response = run(scenario())

    assert response.status_code == 200
    body = response.json()
    assert body["ready"] is True and body["reason"] == "ok" and body["model_resident"] is False
    assert loader.calls == 0, "polling /ready must never load the model"


def test_ready_is_503_loading_while_the_first_load_runs_and_health_stays_200(service, monkeypatch):
    module, loader = with_loader(service, monkeypatch)
    loader.release.clear()

    async def scenario():
        async with make_client(module) as client:
            request = asyncio.create_task(
                client.post("/transcribe", files={"audio": ("a.wav", wav_bytes(1.0), "audio/wav")}))
            assert await asyncio.to_thread(loader.started.wait, WAIT), "the load never started"
            during = await get(client, "/ready")
            health = await get(client, "/health")
            loader.release.set()
            done = await request
            after = await get(client, "/ready")
            return during, health, done, after

    during, health, done, after = run(scenario())

    assert during.status_code == 503
    assert during.json()["reason"] == "loading" and during.json()["ready"] is False
    assert during.headers["retry-after"] == "5"
    assert health.status_code == 200, "/health is liveness: a load in progress is not a failure"
    assert done.status_code == 200
    assert after.status_code == 200 and after.json()["model_resident"] is True


def test_ready_is_503_load_failed_after_a_failed_first_load_and_recovers(service, monkeypatch):
    module, loader = with_loader(service, monkeypatch)
    loader.error = RuntimeError("huggingface.co unreachable")

    async def scenario():
        async with make_client(module) as client:
            first = await client.post("/transcribe", files={"audio": ("a.wav", wav_bytes(1.0), "audio/wav")})
            failed = await get(client, "/ready")
            health = await get(client, "/health")
            loader.error = None
            second = await client.post("/transcribe", files={"audio": ("a.wav", wav_bytes(1.0), "audio/wav")})
            recovered = await get(client, "/ready")
            return first, failed, health, second, recovered

    first, failed, health, second, recovered = run(scenario())

    assert first.status_code == 500
    assert failed.status_code == 503
    assert failed.json()["reason"] == "load_failed"
    assert "huggingface.co unreachable" in failed.json()["detail"]
    assert health.status_code == 200
    assert second.status_code == 200 and recovered.status_code == 200


def test_an_idle_unload_does_not_make_a_proven_model_unready(service, monkeypatch):
    module, loader = with_loader(service, monkeypatch)

    async def scenario():
        async with make_client(module) as client:
            await client.post("/transcribe", files={"audio": ("a.wav", wav_bytes(1.0), "audio/wav")})
            unloaded = await client.post("/unload")
            ready = await get(client, "/ready")
            return unloaded, ready

    unloaded, ready = run(scenario())

    assert unloaded.json()["unloaded"] is True
    assert ready.status_code == 200
    assert ready.json()["model_resident"] is False and ready.json()["model_ever_loaded"] is True


def test_a_scheduled_but_not_yet_started_preload_counts_as_loading(service, monkeypatch):
    module, loader = with_loader(service, monkeypatch)
    module._preload_pending = True

    async def scenario():
        async with make_client(module) as client:
            return await get(client, "/ready")

    response = run(scenario())

    assert response.status_code == 503 and response.json()["reason"] == "loading"


# --- startup preload ---------------------------------------------------------------------

def test_the_preload_runs_in_the_background_so_the_port_answers_during_it(service, monkeypatch):
    module, loader = with_loader(service, monkeypatch, ASR_MODEL_TTL="300")
    loader.release.clear()

    async def scenario():
        async with module.app.router.lifespan_context(module.app):
            async with make_client(module) as client:
                assert await asyncio.to_thread(loader.started.wait, WAIT), "no preload was started"
                health = await get(client, "/health")
                loading = await get(client, "/ready")
                loader.release.set()
                await wait_for(lambda: not module._preload_pending, "the preload to finish")
                ready = await get(client, "/ready")
                return health, loading, ready

    health, loading, ready = run(scenario())

    assert health.status_code == 200
    assert loading.status_code == 503 and loading.json()["reason"] == "loading"
    assert ready.status_code == 200 and ready.json()["model_resident"] is True
    assert loader.calls == 1


def test_a_failed_preload_leaves_the_service_up_and_not_ready(service, monkeypatch):
    module, loader = with_loader(service, monkeypatch, ASR_MODEL_TTL="300")
    loader.error = RuntimeError("no network")

    async def scenario():
        async with module.app.router.lifespan_context(module.app):
            async with make_client(module) as client:
                await wait_for(lambda: not module._preload_pending, "the preload to give up")
                return await get(client, "/health"), await get(client, "/ready")

    health, ready = run(scenario())

    assert health.status_code == 200
    assert ready.status_code == 503 and ready.json()["reason"] == "load_failed"


def test_ttl_zero_skips_the_preload_entirely(service, monkeypatch):
    module, loader = with_loader(service, monkeypatch, ASR_MODEL_TTL="0")

    async def scenario():
        async with module.app.router.lifespan_context(module.app):
            async with make_client(module) as client:
                await asyncio.sleep(0.05)
                return await get(client, "/ready")

    response = run(scenario())

    assert loader.calls == 0
    assert response.status_code == 200


# --- inference tuning ----------------------------------------------------------------------

def load_model(service, monkeypatch, *, cuda=True, capability=(12, 0), hip=None, **env):
    """Run the real loader against a fake NeMo; returns (module, the model it loaded)."""
    module = load_app(service, cuda=cuda, capability=capability, hip=hip, **env)
    model = MODELS[service]()
    requested = []
    install_fake_nemo(monkeypatch, lambda name: requested.append(name) or model)
    loaded = getattr(module, LOADERS[service])()
    assert loaded is model
    module.requested_names = requested
    return module, model


def test_matmul_precision_defaults_to_high_on_cuda_and_the_model_is_moved_to_the_gpu(service, monkeypatch):
    module, model = load_model(service, monkeypatch)

    assert module.torch.calls == [("matmul", "high")]
    assert model.moves == ["cuda"], "bf16 must stay off unless NEMO_BF16 is set"
    assert model.eval_called
    assert module.compute_dtype == "float32"
    assert module.model_loaded is True


def test_the_configured_checkpoint_is_what_gets_loaded(service, monkeypatch):
    env = {"parakeet": {"PARAKEET_ASR_MODEL": "primeline/parakeet-primeline"},
           "canary": {"CANARY_ASR_MODEL": "nvidia/canary-1b-v2"}}[service]
    module, _ = load_model(service, monkeypatch, **env)

    assert module.requested_names == list(env.values())


def test_matmul_precision_can_be_set_and_junk_falls_back_to_high(service, monkeypatch):
    module, _ = load_model(service, monkeypatch, NEMO_MATMUL_PRECISION="medium")
    assert module.torch.calls == [("matmul", "medium")]

    module, _ = load_model(service, monkeypatch, NEMO_MATMUL_PRECISION="turbo")
    assert module.torch.calls == [("matmul", "high")]


def test_cpu_gets_no_gpu_tuning_at_all(service, monkeypatch):
    module, model = load_model(service, monkeypatch, cuda=False, NEMO_BF16="true")

    assert module.torch.calls == []
    assert model.moves == []
    assert module.compute_dtype == "float32"


def test_bf16_is_opt_in_and_needs_compute_capability_8(service, monkeypatch):
    module, model = load_model(service, monkeypatch, NEMO_BF16="true", capability=(12, 0))
    assert model.moves == ["cuda", "bfloat16"]
    assert module.compute_dtype == "bfloat16"

    module, model = load_model(service, monkeypatch, NEMO_BF16="true", capability=(8, 0))
    assert model.moves == ["cuda", "bfloat16"]

    module, model = load_model(service, monkeypatch, NEMO_BF16="true", capability=(7, 5))
    assert model.moves == ["cuda"], "Turing has no bf16 tensor cores"
    assert module.compute_dtype == "float32"


def test_bf16_is_not_used_on_rocm(service, monkeypatch):
    module, model = load_model(service, monkeypatch, NEMO_BF16="true", hip="6.2.41133")

    assert model.moves == ["cuda"]
    assert module.compute_dtype == "float32"


def test_status_reports_how_the_model_runs_and_what_the_limits_are(service, monkeypatch):
    module, model = load_model(
        service, monkeypatch, NEMO_BF16="true", MAX_UPLOAD_MB="50", NEMO_MAX_AUDIO_S="600")
    install_model(module, model)

    async def scenario():
        async with make_client(module) as client:
            return await get(client, "/status")

    body = run(scenario()).json()

    assert body["compute_dtype"] == "bfloat16"
    assert body["matmul_precision"] == "high"
    assert body["max_upload_mb"] == 50
    assert body["max_audio_seconds"] == 600


def test_invalid_limits_fall_back_to_the_defaults(service, monkeypatch):
    module = load_app(service, MAX_UPLOAD_MB="lots", NEMO_MAX_AUDIO_S="-5")

    assert module.MAX_UPLOAD_BYTES == 200 * 1024 * 1024
    assert module.MAX_AUDIO_SECONDS == 1500.0
