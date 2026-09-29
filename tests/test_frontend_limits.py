"""Bounds on what the gateway will accept, and the health-probe cache.

Before: an upload was spooled and then read whole into RAM by whichever handler
got it (the OpenAI route checked its 25 MB limit only AFTER `await file.read()`),
uvicorn has no body limit, and `/api/tts` accepted a 2 MB `text`. Each of these
is a way to make the gateway allocate memory or a GPU service run for as long as
the caller likes.

The size checks run against the ASGI app directly (httpx's ASGITransport) so a
chunked upload with no Content-Length, and one with a Content-Length that lies,
can be sent - a real client can do both.
"""

import asyncio

import httpx
import pytest
from fastapi.testclient import TestClient

from frontend_loader import asgi_client, install_stub, load_frontend_app, wav_bytes

MIB = 1024 * 1024

MULTIPART_HEAD = (
    b'--x\r\nContent-Disposition: form-data; name="provider"\r\n\r\nwhisper\r\n'
    b'--x\r\nContent-Disposition: form-data; name="audio"; filename="a.wav"\r\n'
    b"Content-Type: audio/wav\r\n\r\n"
)
MULTIPART_TYPE = {"content-type": "multipart/form-data; boundary=x"}


def _backend(method, url, kwargs):
    if url.endswith("/transcribe"):
        return {"text": "hallo", "segments": [], "language": "de"}
    if url.endswith("/tts"):
        return httpx.Response(200, content=wav_bytes(), headers={"content-type": "audio/wav"})
    if url.endswith("/train"):
        return {"job_id": "job-1"}
    return {"status": "ok"}


def _small_cap(monkeypatch, **env):
    module = load_frontend_app({"MAX_UPLOAD_MB": "1", **env})
    stub = install_stub(monkeypatch, module, _backend)
    return module, stub


# --- upload size --------------------------------------------------------------


def test_declared_oversize_upload_is_refused_without_reaching_a_backend(monkeypatch):
    module, stub = _small_cap(monkeypatch)
    client = TestClient(module.app)
    r = client.post(
        "/api/stt", data={"provider": "whisper"},
        files={"audio": ("a.wav", b"\0" * (2 * MIB), "audio/wav")})
    assert r.status_code == 413
    assert "Maximum is 1 MB" in r.json()["detail"]
    assert r.headers["connection"] == "close"
    assert stub.calls == []


def test_upload_under_the_cap_still_works(monkeypatch):
    module, stub = _small_cap(monkeypatch)
    r = TestClient(module.app).post(
        "/api/stt", data={"provider": "whisper"},
        files={"audio": ("a.wav", b"\0" * (MIB // 2), "audio/wav")})
    assert r.status_code == 200
    assert len(stub.calls) == 1


def test_chunked_upload_is_cut_off_at_the_cap_not_read_to_the_end(monkeypatch):
    """No Content-Length to check up front: the running count has to stop it,
    and it has to stop it EARLY rather than after buffering the lot."""
    module, stub = _small_cap(monkeypatch)
    pulled = 0

    async def body():
        nonlocal pulled
        yield MULTIPART_HEAD
        for _ in range(400):                       # 100 MiB on offer
            pulled += 1
            yield b"\0" * (256 * 1024)

    async def run():
        async with asgi_client(module) as client:
            return await client.post("/api/stt", content=body(), headers=MULTIPART_TYPE)

    r = asyncio.run(run())
    assert r.status_code == 413
    assert pulled < 10, f"the server kept reading after the cap: {pulled} chunks consumed"
    assert stub.calls == []


def test_understated_content_length_does_not_bypass_the_cap(monkeypatch):
    module, stub = _small_cap(monkeypatch)

    async def run():
        async with asgi_client(module) as client:
            return await client.post(
                "/api/stt", content=MULTIPART_HEAD + b"\0" * (3 * MIB),
                headers={**MULTIPART_TYPE, "content-length": "10"})

    assert asyncio.run(run()).status_code == 413
    assert stub.calls == []


def test_default_cap_admits_a_realistic_training_upload(monkeypatch):
    """The UI's training flow posts many WAVs at once (`audio_files`); the default
    has to be far above the '10+ minutes' it recommends."""
    module = load_frontend_app()
    stub = install_stub(monkeypatch, module, _backend)
    r = TestClient(module.app).post(
        "/api/training/train", data={"model_name": "demo"},
        files=[("audio_files", (f"{i}.wav", b"\0" * (30 * MIB), "audio/wav")) for i in range(2)])
    assert r.status_code == 200, r.text
    assert len(stub.calls) == 1


def test_json_bodies_have_their_own_small_cap(monkeypatch):
    """A 2 MB `text` was accepted; JSON never needs anywhere near the upload cap."""
    module = load_frontend_app({"MAX_UPLOAD_MB": "512"})
    stub = install_stub(monkeypatch, module, _backend)
    body = b'{"provider": "piper", "text": "' + b"a" * (2 * MIB) + b'"}'
    r = TestClient(module.app).post(
        "/api/tts", content=body, headers={"content-type": "application/json"})
    assert r.status_code == 413
    assert stub.calls == []


def test_oversize_body_on_v1_uses_the_openai_envelope(monkeypatch):
    module = load_frontend_app()
    install_stub(monkeypatch, module, _backend)
    body = b'{"input": "' + b"a" * (2 * MIB) + b'"}'
    r = TestClient(module.app).post(
        "/v1/audio/speech", content=body, headers={"content-type": "application/json"})
    assert r.status_code == 413
    error = r.json()["error"]
    assert error["type"] == "invalid_request_error"
    assert error["code"] == "request_too_large"


def test_a_handler_that_swallows_read_errors_still_yields_413(monkeypatch):
    """`create_speech` catches every exception from `request.json()` and would
    answer 400. The cap owns the response once it trips."""
    module = load_frontend_app()
    stub = install_stub(monkeypatch, module, _backend)

    async def body():
        yield b'{"input": "'
        for _ in range(6):
            yield b"a" * (256 * 1024)              # 1.5 MiB > the 1 MiB JSON cap
        yield b'"}'

    async def run():
        async with asgi_client(module) as client:
            return await client.post(
                "/v1/audio/speech", content=body(), headers={"content-type": "application/json"})

    r = asyncio.run(run())
    assert r.status_code == 413
    assert r.json()["error"]["code"] == "request_too_large"
    assert stub.calls == []


def test_get_requests_are_never_body_limited(monkeypatch):
    module, _ = _small_cap(monkeypatch)
    assert TestClient(module.app).get("/providers").status_code == 200


def test_invalid_size_setting_falls_back_to_the_default(monkeypatch):
    for bad in ("lots", "0", "-5", "nan", "inf"):
        module = load_frontend_app({"MAX_UPLOAD_MB": bad})
        assert module.MAX_UPLOAD_MB == 512.0, bad


# --- the OpenAI transcription route -------------------------------------------


def test_transcription_over_25mb_is_413_and_never_forwarded(monkeypatch):
    """Was 400, and only after the whole file had been read into memory."""
    module = load_frontend_app()
    stub = install_stub(monkeypatch, module, _backend)
    monkeypatch.setattr(module.openai_router, "MAX_UPLOAD_BYTES", 4096)
    r = TestClient(module.app).post(
        "/v1/audio/transcriptions", data={"model": "whisper-1"},
        files={"file": ("a.wav", b"\0" * 4097, "audio/wav")})
    assert r.status_code == 413
    error = r.json()["error"]
    assert error["code"] == "file_too_large" and error["param"] == "file"
    assert stub.calls == []


def test_transcription_exactly_at_the_limit_is_accepted(monkeypatch):
    module = load_frontend_app()
    install_stub(monkeypatch, module, _backend)
    monkeypatch.setattr(module.openai_router, "MAX_UPLOAD_BYTES", 4096)
    r = TestClient(module.app).post(
        "/v1/audio/transcriptions", data={"model": "whisper-1"},
        files={"file": ("a.wav", b"\0" * 4096, "audio/wav")})
    assert r.status_code == 200


def test_transcription_limit_is_read_in_slices_not_all_at_once(monkeypatch):
    """The point of the chunked read: an oversize file is abandoned part-way."""
    module = load_frontend_app()
    install_stub(monkeypatch, module, _backend)
    monkeypatch.setattr(module.openai_router, "MAX_UPLOAD_BYTES", 10)
    monkeypatch.setattr(module.openai_router, "_READ_CHUNK", 4)

    read_sizes = []

    class Upload:
        async def read(self, size=-1):
            read_sizes.append(size)
            return b"\0" * 4

    result = asyncio.run(module.openai_router._read_capped(Upload(), 10))
    assert result is None
    assert read_sizes == [4, 4, 4], "should stop at the first slice past the limit"


def test_v1_backstop_still_applies_to_a_grossly_oversize_request(monkeypatch):
    """The route allows 25 MB; the middleware refuses far more than that outright."""
    module = load_frontend_app()
    stub = install_stub(monkeypatch, module, _backend)

    async def run():
        async with asgi_client(module) as client:
            return await client.post(
                "/v1/audio/transcriptions",
                content=b"x", headers={**MULTIPART_TYPE, "content-length": str(60 * MIB)})

    r = asyncio.run(run())
    assert r.status_code == 413
    assert r.json()["error"]["code"] == "request_too_large"
    assert stub.calls == []


# --- free-text bounds ---------------------------------------------------------


def test_tts_text_over_the_limit_is_rejected(monkeypatch):
    module = load_frontend_app({"MAX_TTS_CHARS": "50"})
    stub = install_stub(monkeypatch, module, _backend)
    client = TestClient(module.app)
    assert client.post("/api/tts", json={"provider": "piper", "text": "a" * 51}).status_code == 422
    assert stub.calls == []
    assert client.post("/api/tts", json={"provider": "piper", "text": "a" * 50}).status_code == 200


def test_default_tts_limit_is_20000_characters(monkeypatch):
    module = load_frontend_app()
    install_stub(monkeypatch, module, _backend)
    client = TestClient(module.app)
    assert client.post("/api/tts", json={"provider": "piper", "text": "a" * 20001}).status_code == 422
    assert client.post("/api/tts", json={"provider": "piper", "text": "a" * 20000}).status_code == 200


@pytest.mark.parametrize("field,limit", [
    ("instructions", 4000), ("voice", 256), ("language", 64), ("quality", 32), ("gender", 32),
])
def test_other_free_text_fields_are_bounded(monkeypatch, field, limit):
    module = load_frontend_app()
    stub = install_stub(monkeypatch, module, _backend)
    client = TestClient(module.app)
    r = client.post("/api/tts", json={"provider": "qwen3", "text": "hi", field: "a" * (limit + 1)})
    assert r.status_code == 422
    assert stub.calls == []
    r = client.post("/api/tts", json={"provider": "qwen3", "text": "hi", field: "a" * limit})
    assert r.status_code == 200


def test_voice_design_and_model_selection_fields_are_bounded(monkeypatch):
    module = load_frontend_app({"MAX_TTS_CHARS": "50"})
    stub = install_stub(monkeypatch, module, _backend)
    client = TestClient(module.app)
    design = {"text": "hi", "voice_description": "warm", "lang": "English"}
    assert client.post("/api/providers/qwen3/voice-design", json={**design, "text": "a" * 51}).status_code == 422
    assert client.post("/api/providers/qwen3/voice-design",
                       json={**design, "voice_description": "a" * 4001}).status_code == 422
    assert client.post("/api/providers/qwen3/models/select", json={"model": "a" * 257}).status_code == 422
    assert stub.calls == []


# --- /api/health cache --------------------------------------------------------


class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def _health_module(monkeypatch, env=None, handler=None):
    module = load_frontend_app(env)
    clock = FakeClock()
    monkeypatch.setattr(module, "_health_clock", clock)
    stub = install_stub(monkeypatch, module, handler or (lambda m, u, k: {"status": "healthy"}))
    return module, stub, clock


def _probe_count(stub):
    return len([call for call in stub.calls if call[0] == "GET"])


def test_health_is_probed_once_per_ttl_window(monkeypatch):
    module, stub, clock = _health_module(monkeypatch)
    providers = len([p for p in module.PROVIDER_REGISTRY["providers"].values() if p.get("internal_url")])
    client = TestClient(module.app)

    first = client.get("/api/health").json()
    assert _probe_count(stub) == providers

    clock.advance(1.9)
    assert client.get("/api/health").json() == first
    assert _probe_count(stub) == providers, "a second call inside the TTL re-probed every backend"

    clock.advance(0.2)
    client.get("/api/health")
    assert _probe_count(stub) == 2 * providers, "the cache never expired"


def test_health_cache_can_be_disabled(monkeypatch):
    module, stub, _ = _health_module(monkeypatch, {"HEALTH_CACHE_TTL": "0"})
    providers = len([p for p in module.PROVIDER_REGISTRY["providers"].values() if p.get("internal_url")])
    client = TestClient(module.app)
    client.get("/api/health")
    client.get("/api/health")
    assert _probe_count(stub) == 2 * providers


def test_health_result_reflects_a_backend_that_recovers_after_the_ttl(monkeypatch):
    state = {"up": False}

    def handler(method, url, kwargs):
        if "piper-tts-service" in url and not state["up"]:
            raise httpx.ConnectError("refused", request=httpx.Request(method, url))
        return {"status": "healthy"}

    module, _, clock = _health_module(monkeypatch, handler=handler)
    client = TestClient(module.app)
    assert client.get("/api/health").json()["providers"]["piper"]["healthy"] is False
    state["up"] = True
    assert client.get("/api/health").json()["providers"]["piper"]["healthy"] is False   # cached
    clock.advance(2.5)
    assert client.get("/api/health").json()["providers"]["piper"]["healthy"] is True


def test_concurrent_health_requests_share_one_probe_round_without_blocking(monkeypatch):
    """Five simultaneous polls -> one fan-out, and the loop stays responsive while
    the (slow) backends answer."""
    async def slow(method, url, kwargs):
        await asyncio.sleep(0.15)
        return {"status": "healthy"}

    module, stub, _ = _health_module(monkeypatch, handler=slow)
    providers = len([p for p in module.PROVIDER_REGISTRY["providers"].values() if p.get("internal_url")])

    async def run():
        ticks = 0

        async def ticker():
            nonlocal ticks
            while True:
                await asyncio.sleep(0.01)
                ticks += 1

        tick_task = asyncio.create_task(ticker())
        async with asgi_client(module) as client:
            responses = await asyncio.gather(*(client.get("/api/health") for _ in range(5)))
        tick_task.cancel()
        return responses, ticks

    responses, ticks = asyncio.run(run())
    assert all(r.status_code == 200 for r in responses)
    assert _probe_count(stub) == providers, f"{_probe_count(stub)} probes for 5 concurrent polls"
    assert ticks >= 5, "the event loop was starved while probing"


def test_a_caller_that_gives_up_does_not_cancel_the_shared_probe(monkeypatch):
    async def slow(method, url, kwargs):
        await asyncio.sleep(0.1)
        return {"status": "healthy"}

    module, stub, _ = _health_module(monkeypatch, handler=slow)

    async def run():
        first = asyncio.create_task(module._cached_provider_health())
        second = asyncio.create_task(module._cached_provider_health())
        await asyncio.sleep(0.02)
        first.cancel()
        return await second

    result = asyncio.run(run())
    assert set(result["providers"]) == set(module.PROVIDER_REGISTRY["providers"])
    assert all(entry["healthy"] for entry in result["providers"].values())
