"""Upload size, audio length and event-loop behaviour of the two NeMo ASR services.

Findings covered (both services, same code path through ``nemo_common``):

* an upload used to be read into memory whole and kept next to the file written
  from it, with no cap; now it streams to disk in chunks and 413s past
  ``MAX_UPLOAD_MB``;
* NeMo encodes a file in one pass, so a very long recording ran the GPU out of
  memory and surfaced as a generic 500; now ``NEMO_MAX_AUDIO_S`` refuses it up
  front (413) and ffmpeg is told to stop at the limit (``-t``) so a file whose
  header lies cannot be decoded in full first;
* the duration probe (librosa fallback), the ffmpeg transcode and /unload ran on
  the event loop or in a thread nobody could cancel.

Every test drives the real app over ASGI with fakes only for torch, soundfile,
librosa, ffmpeg and the NeMo model (see ``test_nemo_services_harness``).
"""

from __future__ import annotations

import asyncio
import io
import threading

import pytest
from starlette.datastructures import UploadFile
from fastapi import HTTPException

from test_nemo_services_harness import (
    WAIT, FakeCanary, FakeFfmpeg, FakeParakeet, fake_mp3, install_model, load_app,
    load_nemo_common, make_client, process_alive, run, tmp_files, use_tmpdir, wait_for, wav_bytes,
)

SERVICES = ["parakeet", "canary"]
MODELS = {"parakeet": FakeParakeet, "canary": FakeCanary}


@pytest.fixture(params=SERVICES)
def service(request):
    return request.param


def build(service, tmp_path, monkeypatch, *, ffmpeg=True, **env):
    """(app module, fake model, fake ffmpeg or None, temp dir) with the model installed."""
    tmp = tmp_path / "tmp"
    tmp.mkdir()
    use_tmpdir(monkeypatch, tmp)
    module = load_app(service, **env)
    model = MODELS[service]()
    install_model(module, model)
    fake = None
    if ffmpeg:
        fake = FakeFfmpeg(tmp_path)
        fake.install(module, monkeypatch)
    else:
        monkeypatch.setattr(module, "_FFMPEG", None)
    return module, model, fake, tmp


async def post(client, data, *, name="clip.wav", path="/transcribe", field="audio", **form):
    return await client.post(path, files={field: (name, data, "application/octet-stream")}, data=form)


# --- upload size ---------------------------------------------------------------------

def test_an_oversized_upload_is_refused_with_413_and_leaves_nothing_behind(service, tmp_path, monkeypatch):
    module, model, _, tmp = build(service, tmp_path, monkeypatch, MAX_UPLOAD_MB="0.05")

    async def scenario():
        async with make_client(module) as client:
            return await post(client, wav_bytes(6.0))  # ~96 KB against a 52 KB limit

    response = run(scenario())

    assert response.status_code == 413
    assert "MAX_UPLOAD_MB" in response.json()["detail"]
    assert model.calls == []
    assert not module._model_slot.ever_loaded, "a refused upload must not cost a model load"
    assert tmp_files(tmp) == []


def test_a_declared_content_length_over_the_limit_is_refused_before_the_body_is_parsed(
    service, tmp_path, monkeypatch
):
    module, model, _, tmp = build(service, tmp_path, monkeypatch, MAX_UPLOAD_MB="0.05")

    async def scenario():
        async with make_client(module) as client:
            return await post(client, wav_bytes(80.0))  # ~1.3 MB, past limit + multipart slack

    response = run(scenario())

    assert response.status_code == 413
    assert response.json()["detail"].startswith("Request body is larger")
    assert model.calls == [] and tmp_files(tmp) == []


def test_the_early_413_still_carries_the_cors_headers(service, tmp_path, monkeypatch):
    """The body-limit layer sits inside CORS, or a browser reads the refusal as a network error.

    CORS is only there for an origin the operator listed: with ALLOWED_ORIGINS unset
    no CORS header is sent at all (tests/test_origin_guard.py covers that).
    """
    module, model, _, _ = build(
        service, tmp_path, monkeypatch, MAX_UPLOAD_MB="0.05", ALLOWED_ORIGINS="http://ui.example")

    async def scenario():
        async with make_client(module) as client:
            return await client.post(
                "/transcribe", headers={"Origin": "http://ui.example"},
                files={"audio": ("clip.wav", wav_bytes(80.0), "audio/wav")})

    response = run(scenario())

    assert response.status_code == 413
    assert response.headers["access-control-allow-origin"] == "http://ui.example"


def test_an_upload_under_the_limit_is_transcribed(service, tmp_path, monkeypatch):
    module, model, _, tmp = build(service, tmp_path, monkeypatch, MAX_UPLOAD_MB="0.05")

    async def scenario():
        async with make_client(module) as client:
            return await post(client, wav_bytes(2.0))  # 32 KB

    response = run(scenario())

    assert response.status_code == 200
    assert response.json()["text"].startswith("text:")
    assert len(model.calls) == 1
    assert tmp_files(tmp) == [], "temp files must be removed after the response"


def test_an_empty_upload_is_a_400(service, tmp_path, monkeypatch):
    module, model, _, tmp = build(service, tmp_path, monkeypatch)

    async def scenario():
        async with make_client(module) as client:
            return await post(client, b"")

    response = run(scenario())

    assert response.status_code == 400
    assert model.calls == [] and tmp_files(tmp) == []


class _SpyFile(io.BytesIO):
    def __init__(self, payload):
        super().__init__(payload)
        self.reads: list = []

    def read(self, size=-1):
        self.reads.append(size)
        return super().read(size)


class _NoWholeRead(UploadFile):
    async def read(self, size=-1):  # what the old handler called
        raise AssertionError("the whole upload was read into memory")


def test_uploads_are_copied_to_disk_in_bounded_chunks(service, tmp_path, monkeypatch):
    nc, _ = load_nemo_common(service)
    tmp = tmp_path / "tmp"
    tmp.mkdir()
    use_tmpdir(monkeypatch, tmp)
    payload = b"x" * (2 * 1024 * 1024 + 123)
    spy = _SpyFile(payload)

    async def scenario():
        path, size = await nc.spool_upload(_NoWholeRead(file=spy, filename="big.wav"), max_bytes=10**9)
        with open(path, "rb") as fh:
            assert fh.read() == payload
        return size

    assert run(scenario()) == len(payload)
    assert all(isinstance(n, int) and 0 < n <= nc.MIB for n in spy.reads), spy.reads
    assert len(spy.reads) >= 3


def test_a_chunked_upload_past_the_limit_stops_reading_and_deletes_its_file(service, tmp_path, monkeypatch):
    nc, _ = load_nemo_common(service)
    tmp = tmp_path / "tmp"
    tmp.mkdir()
    use_tmpdir(monkeypatch, tmp)
    spy = _SpyFile(b"x" * (8 * 1024 * 1024))

    async def scenario():
        await nc.spool_upload(UploadFile(file=spy, filename="big.wav"), max_bytes=nc.MIB)

    with pytest.raises(HTTPException) as caught:
        run(scenario())

    assert caught.value.status_code == 413
    assert len(spy.reads) <= 3, "kept reading an upload it had already decided to refuse"
    assert tmp_files(tmp) == []


# --- audio length --------------------------------------------------------------------

def test_audio_over_the_limit_is_refused_before_ffmpeg_or_the_model_runs(service, tmp_path, monkeypatch):
    module, model, fake, tmp = build(service, tmp_path, monkeypatch, NEMO_MAX_AUDIO_S="5")

    async def scenario():
        async with make_client(module) as client:
            return await post(client, wav_bytes(8.0, rate=8000))  # header says 8 s, needs conversion

    response = run(scenario())

    assert response.status_code == 413
    assert "NEMO_MAX_AUDIO_S" in response.json()["detail"]
    assert fake.calls() == [], "the header already proved it too long; no transcode needed"
    assert model.calls == []
    assert not module._model_slot.ever_loaded
    assert tmp_files(tmp) == []


def test_audio_exactly_at_the_limit_is_accepted(service, tmp_path, monkeypatch):
    module, model, _, _ = build(service, tmp_path, monkeypatch, NEMO_MAX_AUDIO_S="5")

    async def scenario():
        async with make_client(module) as client:
            return await post(client, wav_bytes(5.0))

    response = run(scenario())

    assert response.status_code == 200
    assert response.json()["duration"] == pytest.approx(5.0)


def test_a_header_that_lies_cannot_make_ffmpeg_decode_the_whole_file(service, tmp_path, monkeypatch):
    """No usable duration in the header: the transcode is the first time the length is known."""
    module, model, fake, tmp = build(service, tmp_path, monkeypatch, NEMO_MAX_AUDIO_S="5")

    async def scenario():
        async with make_client(module) as client:
            return await post(client, fake_mp3(actual_seconds=400.0, declared_seconds=None), name="clip.mp3")

    response = run(scenario())

    (call,) = fake.calls()
    assert call["args"][call["args"].index("-t") + 1] == "6", call["args"]
    assert call["wrote_seconds"] == pytest.approx(6.0), "ffmpeg was allowed past the -t guard"
    assert response.status_code == 413
    assert model.calls == []
    assert tmp_files(tmp) == []


def test_a_long_container_is_converted_and_the_model_gets_the_converted_file(service, tmp_path, monkeypatch):
    module, model, fake, tmp = build(service, tmp_path, monkeypatch, NEMO_MAX_AUDIO_S="5")

    async def scenario():
        async with make_client(module) as client:
            return await post(client, fake_mp3(actual_seconds=3.0, declared_seconds=3.0), name="clip.mp3")

    response = run(scenario())

    assert response.status_code == 200
    assert response.json()["duration"] == pytest.approx(3.0)
    (call,) = fake.calls()
    assert call["args"][call["args"].index("-ar") + 1] == "16000"
    assert call["args"][call["args"].index("-ac") + 1] == "1"
    assert model.calls[0]["paths"][0].endswith(".16k.wav")
    assert tmp_files(tmp) == []


def test_a_zero_limit_disables_the_length_guard(service, tmp_path, monkeypatch):
    module, model, fake, _ = build(service, tmp_path, monkeypatch, NEMO_MAX_AUDIO_S="0")

    async def scenario():
        async with make_client(module) as client:
            return await post(client, fake_mp3(actual_seconds=40.0, declared_seconds=40.0), name="clip.mp3")

    response = run(scenario())

    assert response.status_code == 200
    assert response.json()["duration"] == pytest.approx(40.0)
    (call,) = fake.calls()
    assert "-t" not in call["args"]


def test_a_native_16k_mono_wav_is_not_sent_through_ffmpeg(service, tmp_path, monkeypatch):
    module, model, fake, _ = build(service, tmp_path, monkeypatch)

    async def scenario():
        async with make_client(module) as client:
            return await post(client, wav_bytes(2.0))

    assert run(scenario()).status_code == 200
    assert fake.calls() == []
    assert not model.calls[0]["paths"][0].endswith(".16k.wav")


def test_a_failed_transcode_falls_back_to_the_original_file(service, tmp_path, monkeypatch):
    module, model, fake, tmp = build(service, tmp_path, monkeypatch)
    fake.mode = "fail"

    async def scenario():
        async with make_client(module) as client:
            return await post(client, wav_bytes(2.0, rate=8000))

    response = run(scenario())

    assert response.status_code == 200
    assert response.json()["duration"] == pytest.approx(2.0)
    assert len(fake.calls()) == 1
    assert model.calls[0]["paths"][0].endswith(".wav")
    assert not model.calls[0]["paths"][0].endswith(".16k.wav")
    assert tmp_files(tmp) == []


def test_without_ffmpeg_the_librosa_fallback_supplies_the_duration(service, tmp_path, monkeypatch):
    module, model, _, _ = build(service, tmp_path, monkeypatch, ffmpeg=False)

    async def scenario():
        async with make_client(module) as client:
            return await post(client, fake_mp3(actual_seconds=3.0, declared_seconds=3.0), name="clip.mp3")

    response = run(scenario())

    assert response.status_code == 200
    assert response.json()["duration"] == pytest.approx(3.0)
    assert module._test_librosa.calls == 1
    assert model.calls[0]["paths"][0].endswith(".mp3")


def test_audio_no_reader_can_open_is_a_422_not_a_500(service, tmp_path, monkeypatch):
    module, model, _, tmp = build(service, tmp_path, monkeypatch, ffmpeg=False)

    async def scenario():
        async with make_client(module) as client:
            return await post(client, fake_mp3(actual_seconds=3.0, declared_seconds=None), name="clip.mp3")

    response = run(scenario())

    assert response.status_code == 422
    assert model.calls == [] and tmp_files(tmp) == []


# --- the event loop --------------------------------------------------------------------

def test_the_librosa_fallback_does_not_block_the_event_loop(service, tmp_path, monkeypatch):
    """A long mp3 probed through librosa froze /health for the length of the probe."""
    module, model, _, _ = build(service, tmp_path, monkeypatch, ffmpeg=False)
    librosa = module._test_librosa
    librosa.block = True

    async def scenario():
        async with make_client(module) as client:
            request = asyncio.create_task(
                post(client, fake_mp3(actual_seconds=3.0, declared_seconds=3.0), name="clip.mp3"))
            assert await asyncio.to_thread(librosa.entered.wait, WAIT), "the probe never started"
            health = await client.get("/health")
            librosa.release.set()
            return health, await request

    health, response = run(scenario())

    assert not librosa.timed_out, "the probe was still blocking the event loop"
    assert health.status_code == 200
    assert response.status_code == 200


def test_the_transcode_does_not_block_the_event_loop(service, tmp_path, monkeypatch):
    module, model, fake, _ = build(service, tmp_path, monkeypatch)
    fake.mode = "wait"

    async def scenario():
        async with make_client(module) as client:
            request = asyncio.create_task(
                post(client, fake_mp3(actual_seconds=3.0, declared_seconds=3.0), name="clip.mp3"))
            await wait_for(lambda: fake.running, "ffmpeg to start")
            health = await client.get("/health")
            fake.release()
            return health, await request

    health, response = run(scenario())

    assert health.status_code == 200
    assert response.status_code == 200


def test_a_client_that_goes_away_kills_the_transcode(service, tmp_path, monkeypatch):
    module, model, fake, tmp = build(service, tmp_path, monkeypatch)
    fake.mode = "hang"

    async def scenario():
        async with make_client(module) as client:
            request = asyncio.create_task(
                post(client, fake_mp3(actual_seconds=3.0, declared_seconds=3.0), name="clip.mp3"))
            await wait_for(lambda: fake.hung_pid() is not None, "ffmpeg to start")
            pid = fake.hung_pid()
            request.cancel()
            with pytest.raises(asyncio.CancelledError):
                await request
            await wait_for(lambda: not process_alive(pid), "ffmpeg to be killed")

    run(scenario())

    assert model.calls == []
    assert tmp_files(tmp) == []


def test_a_transcode_that_outlives_its_timeout_is_killed_and_the_original_is_used(service, tmp_path, monkeypatch):
    nc, _ = load_nemo_common(service)
    tmp = tmp_path / "tmp"
    tmp.mkdir()
    use_tmpdir(monkeypatch, tmp)
    fake = FakeFfmpeg(tmp_path)
    monkeypatch.setenv("FAKE_FFMPEG_DIR", str(fake.dir))
    fake.mode = "hang"
    payload = wav_bytes(2.0, rate=8000)

    async def scenario():
        upload = UploadFile(file=io.BytesIO(payload), filename="clip.wav")
        async with nc.prepared_upload(
            # The timeout is what is under test, but it must not fire before the fake has
            # started and published its pid, or there is nothing to check afterwards: a
            # Python interpreter needs ~30 ms to get there, several times that on a loaded
            # runner. 0.3 s missed it (TypeError from process_alive(None)); 2 s is ~50x.
            upload, max_bytes=10**9, max_seconds=100.0, ffmpeg=str(fake.path), convert_timeout=2.0,
        ) as prepared:
            path = prepared.path
        pid = fake.hung_pid()
        assert pid is not None, "the timeout fired before the fake ffmpeg started: raise convert_timeout"
        await wait_for(lambda: not process_alive(pid), "the hung ffmpeg to be killed")
        return path

    path = run(scenario())

    assert not path.endswith(".16k.wav"), "a timed-out transcode must fall back to the original"
    assert tmp_files(tmp) == []


# --- /unload ----------------------------------------------------------------------------

def test_unload_does_not_block_the_event_loop_while_the_weights_move_off_the_gpu(service, tmp_path, monkeypatch):
    module, model, _, _ = build(service, tmp_path, monkeypatch)
    slot = module._model_slot
    with slot.acquire():
        pass  # resident, no references
    entered, release = threading.Event(), threading.Event()
    timed_out = []

    def slow_cpu():
        entered.set()
        timed_out.append(not release.wait(WAIT))  # only the test can set this
        return model

    model.cpu = slow_cpu

    async def scenario():
        async with make_client(module) as client:
            unloading = asyncio.create_task(client.post("/unload"))
            assert await asyncio.to_thread(entered.wait, WAIT), "the release hook never ran"
            health = await client.get("/health")
            release.set()
            return health, await unloading

    health, unloaded = run(scenario())

    assert timed_out == [False], "/unload held the event loop while the weights moved"
    assert health.status_code == 200
    assert unloaded.status_code == 200
    assert unloaded.json()["unloaded"] is True


def test_unload_moves_the_weights_off_the_gpu_and_clears_the_loaded_flag(service, tmp_path, monkeypatch):
    """`empty_cache()` frees only unreferenced blocks, and NeMo keeps its own reference to the
    module, so an unload that merely drops ours frees nothing: the hook has to `.cpu()` the
    weights. And `/health` must stop reporting a model that is gone."""
    module, model, _, _ = build(service, tmp_path, monkeypatch)
    with module._model_slot.acquire():
        pass  # resident, no references
    module.model_loaded = True
    moved = []
    model.cpu = lambda: moved.append(True) or model

    async def scenario():
        async with make_client(module) as client:
            return await client.post("/unload"), await client.get("/health")

    unloaded, health = run(scenario())

    assert unloaded.status_code == 200 and unloaded.json()["unloaded"] is True
    assert moved == [True], "the weights were dropped without being moved off the GPU"
    assert module.model_loaded is False
    body = health.json()
    assert body["model_resident"] is False, "/health keeps reporting a model that was unloaded"
    assert {"model_resident", "model_ttl_seconds", "active_requests"} <= set(body)
