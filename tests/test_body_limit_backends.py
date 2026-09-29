"""Real chunked requests through every backend's own ASGI app.

The reviewer's repro: parakeet's body-limit middleware with a 2 MB limit answered a
*declared* 200 MB upload with 413 after reading 0 bytes, but a *chunked* 200 MB
upload (no Content-Length) was consumed to the end and written to a temp file
before the handler ran. The same held for every backend that had a middleware and
for stt-service, which had none. Starlette spools multipart to disk and JSON to RAM
before any handler can look, so an unauthenticated POST could fill either.

Each test here sends the body from a generator (so the request really is chunked),
counts how much of it the server took, and checks the three things that matter:
413, the handler never ran, and no spooled temp file is left behind.
"""

from __future__ import annotations

import asyncio
import contextlib

import pytest

from backend_apps import (
    BACKENDS, MIB, Offer, load_backend, multipart_content_type, multipart_head, spy_on_route,
)

# Every route of each backend that takes an upload (multipart) - the primary one
# from backend_apps.ROUTES comes first.
UPLOAD_ROUTES = {
    "stt-service": ["/transcribe-stream", "/transcribe", "/detect_language"],
    "piper-tts-service": ["/upload_model", "/analyze_audio"],
    "piper-training-service": ["/train", "/test-upload"],
    "chatterbox-tts-service": ["/clone", "/clone-with-ref-text"],
    "qwen3-tts-service": ["/clone", "/clone-with-ref-text", "/voices/save"],
    "qwen3-asr-service": ["/transcribe", "/detect_language", "/transcribe-batch"],
    "parakeet-asr-service": ["/transcribe", "/v1/audio/transcriptions", "/detect_language", "/transcribe-batch"],
    "canary-asr-service": ["/transcribe", "/v1/audio/transcriptions", "/detect_language", "/transcribe-batch"],
}

OFFERED = 64 * MIB
# The largest limit any route has at MAX_UPLOAD_MB=1 is the batch routes' 8 files + framing,
# about 9.5 MiB; anything that gets past this much has not been cut off at its limit.
CUT_OFF_BY = 12 * MIB


def _run(coro):
    return asyncio.run(coro)


def _small(name, monkeypatch, tmp_path, stack):
    """*name* with every upload limit at 1 MB (piper-tts has a second knob for analysis uploads)."""
    env = {"MAX_UPLOAD_MB": "1"}
    if name == "piper-tts-service":
        env["MAX_ANALYZE_UPLOAD_MB"] = "1"
    return load_backend(name, monkeypatch, tmp_path, stack, **env)


@pytest.fixture(params=BACKENDS)
def backend(request, monkeypatch, tmp_path):
    with contextlib.ExitStack() as stack:
        yield _small(request.param, monkeypatch, tmp_path, stack)


def _upload_cases():
    return [(name, path) for name in BACKENDS for path in UPLOAD_ROUTES[name]]


@pytest.fixture(params=_upload_cases(), ids=lambda case: f"{case[0]}{case[1]}")
def upload_case(request, monkeypatch, tmp_path):
    name, path = request.param
    with contextlib.ExitStack() as stack:
        yield _small(name, monkeypatch, tmp_path, stack), path


# --- chunked uploads --------------------------------------------------------------------

def test_a_chunked_upload_is_cut_off_at_the_limit_and_leaves_nothing_behind(upload_case):
    backend, path = upload_case
    handler = spy_on_route(backend.app, path)
    offer = Offer(multipart_head("file", extra_fields={"model_name": "voice", "voice_name": "v", "text": "hallo"}),
                  OFFERED)

    async def scenario():
        async with backend.client() as client:
            return await client.post(path, content=offer, headers=multipart_content_type())

    response = _run(scenario())

    assert response.status_code == 413, response.text
    assert "detail" in response.json()
    assert offer.pulled_bytes <= CUT_OFF_BY, (
        f"{backend.name} {path} read {offer.pulled_bytes / MIB:.1f} MiB of a body it should have refused "
        f"after a few MiB: chunked uploads bypass the limit")
    assert handler == [], "the handler ran for a request that was over its limit"
    assert backend.leftovers() == [], "a spooled upload was left behind"


def test_an_understated_content_length_does_not_bypass_the_limit(backend):
    path = backend.routes["upload"][1]
    handler = spy_on_route(backend.app, path)
    body = multipart_head("file") + b"\0" * (13 * MIB)

    async def scenario():
        async with backend.client() as client:
            return await client.post(
                path, content=body, headers={**multipart_content_type(), "content-length": "100"})

    response = _run(scenario())

    assert response.status_code == 413, response.text
    assert handler == [] and backend.leftovers() == []


def test_a_declared_oversize_upload_is_refused_before_any_of_it_is_read(upload_case):
    backend, path = upload_case
    handler = spy_on_route(backend.app, path)
    offer = Offer(multipart_head("file"), 4 * MIB)

    async def scenario():
        async with backend.client() as client:
            return await client.post(
                path, content=offer,
                headers={**multipart_content_type(), "content-length": str(500 * MIB)})

    response = _run(scenario())

    assert response.status_code == 413, response.text
    assert offer.pulled == 0, "read the body of a request whose declared length was already over the limit"
    assert handler == [] and backend.leftovers() == []


def test_the_refusal_is_a_small_json_413_that_closes_the_connection(backend):
    path = backend.routes["upload"][1]

    async def scenario():
        async with backend.client() as client:
            return await client.post(
                path, content=Offer(multipart_head("file"), OFFERED), headers=multipart_content_type())

    response = _run(scenario())

    assert response.status_code == 413
    assert response.headers["content-type"].startswith("application/json")
    assert response.headers["connection"] == "close"
    assert len(response.content) < 1024
    assert response.json()["detail"].startswith("Request body is larger than")


# --- bodies that are not uploads --------------------------------------------------------

def test_a_chunked_json_body_is_cut_off_before_it_is_held_in_memory(backend):
    path = backend.routes["json"]
    if path is None:
        pytest.skip(f"{backend.name} has no JSON body route")
    handler = spy_on_route(backend.app, path)
    offer = Offer(b'{"text": "', OFFERED)

    async def scenario():
        async with backend.client() as client:
            return await client.post(path, content=offer, headers={"content-type": "application/json"})

    response = _run(scenario())

    assert response.status_code == 413, response.text
    assert offer.pulled_bytes <= OFFERED // 2, (
        f"{backend.name} {path} took {offer.pulled_bytes / MIB:.0f} MiB of JSON into memory")
    assert handler == []


def test_a_route_that_takes_no_body_never_reads_one(backend):
    """POST /unload and friends: a hostile caller can still attach a body to them. The
    handler does not read it, so a chunked one costs nothing; a declared one is refused."""
    method, path = backend.routes["probe"]
    offer = Offer(b"", OFFERED)

    async def scenario():
        async with backend.client() as client:
            chunked = await client.request(method, path, content=offer)
            declared = await client.request(
                method, path, content=b"x", headers={"content-length": str(500 * MIB)})
            return chunked, declared

    chunked, declared = _run(scenario())

    assert offer.pulled == 0, "the body of a route that reads none was consumed"
    assert chunked.status_code != 413
    assert declared.status_code == 413, declared.text


# --- what must keep working --------------------------------------------------------------

def test_a_small_chunked_upload_is_not_mistaken_for_an_attack(backend):
    """No Content-Length is normal for some clients (streaming uploads); under the limit it must reach the app."""
    path = backend.routes["upload"][1]
    field = backend.routes["upload"][2]
    body = (multipart_head(field, extra_fields={"model_name": "voice"}) + b"\0" * 2048
            + b"\r\n--b--\r\n")

    async def chunks():
        for i in range(0, len(body), 500):
            yield body[i:i + 500]

    async def scenario():
        async with backend.client() as client:
            return await client.post(path, content=chunks(), headers=multipart_content_type())

    response = _run(scenario())

    assert response.status_code != 413, response.text
    assert response.status_code != 403, response.text


def test_a_get_is_never_body_limited(backend):
    """GET /health is what the orchestrator polls; it must not depend on this middleware."""
    async def scenario():
        async with backend.client() as client:
            return await client.get("/health", headers={"content-length": str(500 * MIB)})

    assert _run(scenario()).status_code != 413


# --- the per-route limits keep their old meaning ------------------------------------------

def _limit(module, path):
    fn = getattr(module, "_body_limit", None) or getattr(module, "_body_limit_for")
    return fn(path)


def test_piper_tts_model_uploads_and_analysis_uploads_have_their_own_limits(monkeypatch, tmp_path):
    with contextlib.ExitStack() as stack:
        m = load_backend("piper-tts-service", monkeypatch, tmp_path, stack,
                         MAX_UPLOAD_MB="100", MAX_ANALYZE_UPLOAD_MB="25", MAX_TEXT_CHARS="1000").module
        model_limit, analysis_limit, json_limit = (_limit(m, p) for p in ("/upload_model", "/analyze_audio", "/tts"))
    assert model_limit > 100 * MIB + 5 * MIB          # model + config + framing
    assert 25 * MIB < analysis_limit < 27 * MIB
    assert json_limit == 1000 * 4 + 64 * 1024


def test_piper_training_audio_uploads_get_max_upload_mb_and_the_rest_16_mb(monkeypatch, tmp_path):
    with contextlib.ExitStack() as stack:
        m = load_backend("piper-training-service", monkeypatch, tmp_path, stack, MAX_UPLOAD_MB="500").module
        assert _limit(m, "/train") == _limit(m, "/test-upload") == 501 * MIB
        assert _limit(m, "/resume-training") == _limit(m, "/train-from-dataset") == 16 * MIB


@pytest.mark.parametrize("name", ["chatterbox-tts-service", "qwen3-tts-service"])
def test_tts_reference_uploads_get_max_upload_mb_and_json_routes_a_text_sized_limit(name, monkeypatch, tmp_path):
    with contextlib.ExitStack() as stack:
        m = load_backend(name, monkeypatch, tmp_path, stack, MAX_UPLOAD_MB="20", MAX_TEXT_CHARS="5000").module
        upload, json_route = _limit(m, "/clone"), _limit(m, "/tts")
    assert 20 * MIB < upload < 22 * MIB
    assert json_route < MIB


@pytest.mark.parametrize("name", ["parakeet-asr-service", "canary-asr-service", "qwen3-asr-service"])
def test_asr_single_file_routes_get_one_file_and_the_batch_route_a_bounded_multiple(name, monkeypatch, tmp_path):
    with contextlib.ExitStack() as stack:
        m = load_backend(name, monkeypatch, tmp_path, stack, MAX_UPLOAD_MB="200").module
        single, batch, other = (_limit(m, p) for p in ("/transcribe", "/transcribe-batch", "/unload"))
    assert 200 * MIB < single < 202 * MIB
    assert 8 * 200 * MIB < batch < 8 * 202 * MIB, "a batch is bounded too, at a few files' worth"
    assert other <= MIB


def test_stt_batch_and_single_file_share_a_route_so_it_gets_the_batch_limit(monkeypatch, tmp_path):
    with contextlib.ExitStack() as stack:
        m = load_backend("stt-service", monkeypatch, tmp_path, stack, MAX_UPLOAD_MB="200").module
        transcribe, stream, detect, other = (
            _limit(m, p) for p in ("/transcribe", "/transcribe-stream", "/detect_language", "/unload"))
    assert 8 * 200 * MIB < transcribe < 8 * 202 * MIB
    assert 200 * MIB < stream == detect < 202 * MIB
    assert other <= MIB


def test_stt_keeps_its_switch_for_no_upload_limit(monkeypatch, tmp_path):
    """MAX_UPLOAD_MB <= 0 has always meant "no limit" there; it still does, for uploads only."""
    with contextlib.ExitStack() as stack:
        m = load_backend("stt-service", monkeypatch, tmp_path, stack, MAX_UPLOAD_MB="0").module
        assert m.MAX_UPLOAD_BYTES == 0
        assert _limit(m, "/transcribe") is None and _limit(m, "/transcribe-stream") is None
        assert _limit(m, "/unload") == MIB
