"""Admission control in the two NeMo ASR services: how many requests wait, and for how long.

Finding covered: every request that reached the forward-pass semaphore had already
spooled its upload and converted the audio, then waited for the ones ahead of it
with no limit on the number or the time. A burst of uploads piled up temp files
and held every client until the slowest one finished.

Now ``ASR_MAX_CONCURRENCY + ASR_MAX_QUEUE`` requests are admitted (the next is
answered 503 + Retry-After at once, before its upload is touched), and a request
that waits longer than ``ASR_QUEUE_TIMEOUT_S`` for its turn is answered the same
way. Whichever way a request ends, its temp files go and its model reference is
handed back.

Every test drives the real app over ASGI; the fake model blocks on an event to
hold a request "in the forward pass" for exactly as long as the test wants.
"""

from __future__ import annotations

import asyncio
import threading

import pytest

from test_nemo_services_harness import (
    FakeCanary, FakeParakeet, WAIT, hypothesis, install_model, load_app, make_client, run, tmp_files,
    use_tmpdir, wait_for, wav_bytes,
)

SERVICES = ["parakeet", "canary"]
MODELS = {"parakeet": FakeParakeet, "canary": FakeCanary}


@pytest.fixture(params=SERVICES)
def service(request):
    return request.param


class Blocker:
    """A model reply that parks each call until the test opens that call's gate."""

    def __init__(self, gated_calls=(1,)):
        self.gates = {n: threading.Event() for n in gated_calls}
        self.calls = 0
        self.entered = threading.Event()
        self._lock = threading.Lock()

    def __call__(self, paths, kwargs):
        with self._lock:
            self.calls += 1
            number = self.calls
        self.entered.set()
        gate = self.gates.get(number)
        if gate is not None:
            assert gate.wait(WAIT), f"the test never opened the gate of call {number}"
        return [hypothesis(f"call {number}") for _ in paths]

    def open(self, number=1):
        self.gates[number].set()

    def open_all(self):
        for gate in self.gates.values():
            gate.set()


def build(service, monkeypatch, tmp_path, *, blocker=None, **env):
    """(module, model, blocker, temp dir): every clip is a native 16 kHz mono WAV, so one temp file per request."""
    tmp = tmp_path / "tmp"
    tmp.mkdir()
    use_tmpdir(monkeypatch, tmp)
    module = load_app(service, **env)
    monkeypatch.setattr(module, "_FFMPEG", None)
    model = MODELS[service]()
    blocker = blocker or Blocker()
    model.reply = blocker
    install_model(module, model)
    return module, model, blocker, tmp


async def post(client, name="clip.wav", seconds=1.0, path="/transcribe", field="audio"):
    return await client.post(path, files={field: (name, wav_bytes(seconds), "audio/wav")})


async def post_batch(client, names):
    files = [("audios", (name, wav_bytes(1.0), "audio/wav")) for name in names]
    return await client.post("/transcribe-batch", files=files)


async def queue(client) -> dict:
    return (await client.get("/status")).json()["queue"]


async def in_flight(client, count):
    async def reached():
        return (await queue(client))["in_flight"] == count

    deadline = asyncio.get_running_loop().time() + WAIT
    while not await reached():
        assert asyncio.get_running_loop().time() < deadline, f"in_flight never reached {count}"
        await asyncio.sleep(0.01)


def counting_prepared_upload(module, monkeypatch):
    """The filenames `_prepared_upload` was entered for: the work a refused request must never cause."""
    seen = []
    real = module._prepared_upload

    def counting(upload):
        seen.append(upload.filename)
        return real(upload)

    monkeypatch.setattr(module, "_prepared_upload", counting)
    return seen


def turn_requests(module, monkeypatch):
    """One entry per request that has asked the gate for its turn.

    "Admitted" is not "queued": an admitted request may still be copying its upload.
    A request that has asked is on the queue one event-loop step later, which a
    test that then polls on a timer always outlasts.
    """
    asked = []
    real = module._gate.turn

    def counting():
        asked.append(1)
        return real()

    monkeypatch.setattr(module._gate, "turn", counting)
    return asked


# --- the queue is bounded --------------------------------------------------------------------

def test_a_request_beyond_the_queue_is_refused_at_once_and_does_no_work(service, monkeypatch, tmp_path):
    module, model, blocker, tmp = build(
        service, monkeypatch, tmp_path, ASR_MAX_CONCURRENCY="1", ASR_MAX_QUEUE="2")
    prepared = counting_prepared_upload(module, monkeypatch)

    async def scenario():
        async with make_client(module) as client:
            running = asyncio.create_task(post(client, "run.wav"))
            await wait_for(blocker.entered.is_set, "the first request to reach the model")
            waiting = [asyncio.create_task(post(client, f"wait{i}.wav")) for i in range(2)]
            await in_flight(client, 3)

            # Everything is taken: one running, two waiting. The fourth is refused
            # while the others are still stuck, so this cannot be it merely waiting.
            shed = await asyncio.wait_for(post(client, "shed.wav"), WAIT)
            during = await queue(client)
            files_during = tmp_files(tmp)

            blocker.open_all()
            done = await asyncio.gather(running, *waiting)
            after = await queue(client)
            calls = blocker.calls
            retry = await post(client, "retry.wav")
            return shed, during, files_during, done, after, calls, retry

    shed, during, files_during, done, after, calls, retry = run(scenario())

    assert shed.status_code == 503
    assert shed.headers["retry-after"] == "5"
    assert "busy" in shed.json()["detail"].lower()
    assert sorted(prepared[:3]) == ["run.wav", "wait0.wav", "wait1.wav"] and "shed.wav" not in prepared, (
        "the refused request was already spooled or converted"
    )
    assert len(files_during) == 3, "only the three admitted requests may hold a temp file"
    assert during == {**during, "max_concurrency": 1, "max_queue": 2, "in_flight": 3, "running": 1, "waiting": 2}
    assert [r.status_code for r in done] == [200, 200, 200]
    assert calls == 3, "the refused request reached the model"
    assert after["in_flight"] == 0 and tmp_files(tmp) == []
    assert retry.status_code == 200, "the place was not given back"
    assert module._model_slot.refs == 0


def test_a_queue_of_zero_means_nobody_waits(service, monkeypatch, tmp_path):
    module, model, blocker, tmp = build(
        service, monkeypatch, tmp_path, ASR_MAX_CONCURRENCY="1", ASR_MAX_QUEUE="0")

    async def scenario():
        async with make_client(module) as client:
            running = asyncio.create_task(post(client, "run.wav"))
            await wait_for(blocker.entered.is_set, "the first request to reach the model")
            shed = await asyncio.wait_for(post(client, "shed.wav"), WAIT)
            blocker.open_all()
            return shed, await running

    shed, first = run(scenario())

    assert shed.status_code == 503 and first.status_code == 200


def test_every_route_that_takes_an_upload_is_behind_the_gate(service, monkeypatch, tmp_path):
    module, model, blocker, tmp = build(
        service, monkeypatch, tmp_path, ASR_MAX_CONCURRENCY="1", ASR_MAX_QUEUE="0")

    async def scenario():
        async with make_client(module) as client:
            running = asyncio.create_task(post(client, "run.wav"))
            await wait_for(blocker.entered.is_set, "the first request to reach the model")
            answers = {
                "transcribe": await post(client),
                "openai": await post(client, path="/v1/audio/transcriptions", field="file"),
                "detect": await post(client, path="/detect_language", field="file"),
                "batch": await post_batch(client, ["a.wav", "b.wav"]),
            }
            blocker.open_all()
            await running
            return answers

    answers = run(scenario())

    assert {name: r.status_code for name, r in answers.items()} == {
        "transcribe": 503, "openai": 503, "detect": 503, "batch": 503}
    assert all(r.headers["retry-after"] == "5" for r in answers.values())
    assert blocker.calls == 1


def test_a_batch_takes_one_place_however_many_files_it_carries(service, monkeypatch, tmp_path):
    """Three files with room for one request: the files are not requests."""
    module, model, blocker, tmp = build(
        service, monkeypatch, tmp_path, blocker=Blocker(gated_calls=()),
        ASR_MAX_CONCURRENCY="1", ASR_MAX_QUEUE="0")

    async def scenario():
        async with make_client(module) as client:
            return await post_batch(client, ["a.wav", "b.wav", "c.wav"])

    response = run(scenario())

    assert response.status_code == 200
    assert [r["filename"] for r in response.json()["results"]] == ["a.wav", "b.wav", "c.wav"]
    assert all("text" in r for r in response.json()["results"])
    assert tmp_files(tmp) == []


# --- the wait is bounded ---------------------------------------------------------------------

def test_a_waiter_that_outlasts_the_timeout_gets_503_and_leaves_nothing_behind(service, monkeypatch, tmp_path):
    module, model, blocker, tmp = build(
        service, monkeypatch, tmp_path, ASR_MAX_CONCURRENCY="1", ASR_MAX_QUEUE="2",
        ASR_QUEUE_TIMEOUT_S="0.3")

    async def scenario():
        async with make_client(module) as client:
            running = asyncio.create_task(post(client, "run.wav"))
            await wait_for(blocker.entered.is_set, "the first request to reach the model")
            timed_out = await asyncio.wait_for(post(client, "late.wav"), WAIT)
            # The running request is still stuck in the model: this is what is left.
            refs_after_timeout = module._model_slot.refs
            files_after_timeout = tmp_files(tmp)
            state = await queue(client)
            blocker.open_all()
            return timed_out, refs_after_timeout, files_after_timeout, state, await running

    timed_out, refs, files, state, first = run(scenario())

    assert timed_out.status_code == 503
    assert timed_out.headers["retry-after"] == "5"
    assert "ASR_QUEUE_TIMEOUT_S" in timed_out.json()["detail"]
    assert refs == 1, "the timed-out request kept its model reference"
    assert len(files) == 1, "the timed-out request left its upload behind"
    assert state["in_flight"] == 1 and state["waiting"] == 0
    assert first.status_code == 200
    assert blocker.calls == 1, "the timed-out request reached the model"
    assert module._model_slot.refs == 0 and tmp_files(tmp) == []


def test_a_timed_out_waiter_does_not_take_the_turn_from_the_next_one(service, monkeypatch, tmp_path):
    module, model, blocker, tmp = build(
        service, monkeypatch, tmp_path, ASR_MAX_CONCURRENCY="1", ASR_MAX_QUEUE="2",
        ASR_QUEUE_TIMEOUT_S="0.3")

    async def scenario():
        async with make_client(module) as client:
            running = asyncio.create_task(post(client, "run.wav"))
            await wait_for(blocker.entered.is_set, "the first request to reach the model")
            assert (await post(client, "late.wav")).status_code == 503
            blocker.open_all()
            assert (await running).status_code == 200
            return await asyncio.wait_for(post(client, "next.wav"), WAIT)

    assert run(scenario()).status_code == 200


def test_a_cancelled_waiter_gives_back_its_model_reference_and_its_files(service, monkeypatch, tmp_path):
    module, model, blocker, tmp = build(
        service, monkeypatch, tmp_path, ASR_MAX_CONCURRENCY="1", ASR_MAX_QUEUE="2")

    async def scenario():
        async with make_client(module) as client:
            running = asyncio.create_task(post(client, "run.wav"))
            await wait_for(blocker.entered.is_set, "the first request to reach the model")
            waiter = asyncio.create_task(post(client, "gone.wav"))
            await wait_for(lambda: module._model_slot.refs == 2, "the waiter to hold its reference")
            await in_flight(client, 2)

            waiter.cancel()
            with pytest.raises(asyncio.CancelledError):
                await waiter
            await wait_for(lambda: module._model_slot.refs == 1, "the cancelled waiter's reference to go back")
            files = tmp_files(tmp)
            state = await queue(client)

            blocker.open_all()
            return files, state, await running, await asyncio.wait_for(post(client, "next.wav"), WAIT)

    files, state, first, later = run(scenario())

    assert len(files) == 1, "the cancelled request left its upload behind"
    assert state["in_flight"] == 1 and state["waiting"] == 0
    assert first.status_code == 200 and later.status_code == 200
    assert blocker.calls == 2, "the cancelled request must never have run"
    assert module._model_slot.refs == 0 and tmp_files(tmp) == []


def test_a_batch_that_waits_too_long_is_a_503_and_its_files_are_removed(monkeypatch, tmp_path):
    """Parakeet decodes the whole batch in one pass, so a batch that gets no turn served nothing."""
    module, model, blocker, tmp = build(
        "parakeet", monkeypatch, tmp_path, ASR_MAX_CONCURRENCY="1", ASR_MAX_QUEUE="2",
        ASR_QUEUE_TIMEOUT_S="0.3")

    async def scenario():
        async with make_client(module) as client:
            running = asyncio.create_task(post(client, "run.wav"))
            await wait_for(blocker.entered.is_set, "the first request to reach the model")
            batch = await asyncio.wait_for(post_batch(client, ["a.wav", "b.wav", "c.wav"]), WAIT)
            files = tmp_files(tmp)
            refs = module._model_slot.refs
            blocker.open_all()
            return batch, files, refs, await running

    batch, files, refs, first = run(scenario())

    assert batch.status_code == 503 and batch.headers["retry-after"] == "5"
    assert len(files) == 1 and refs == 1
    assert first.status_code == 200 and blocker.calls == 1
    assert tmp_files(tmp) == []


def test_canary_batch_serves_what_it_can_and_reports_the_rest_as_503(monkeypatch, tmp_path):
    """Canary decodes file by file. The first is served; another request takes the
    turn; the second file then outwaits the queue and the third is not made to wait
    as long again."""
    module, model, blocker, tmp = build(
        "canary", monkeypatch, tmp_path, blocker=Blocker(gated_calls=(1, 2)),
        ASR_MAX_CONCURRENCY="1", ASR_MAX_QUEUE="2", ASR_QUEUE_TIMEOUT_S="0.3")
    asked = turn_requests(module, monkeypatch)

    async def scenario():
        async with make_client(module) as client:
            batch = asyncio.create_task(post_batch(client, ["a.wav", "b.wav", "c.wav"]))
            await wait_for(blocker.entered.is_set, "the batch's first file to reach the model")
            other = asyncio.create_task(post(client, "other.wav"))
            await wait_for(lambda: len(asked) == 2, "the other request to ask for its turn")
            blocker.open(1)  # the first file finishes; `other` was waiting first and takes the turn
            await wait_for(lambda: blocker.calls == 2, "the other request to take the turn")
            answer = await asyncio.wait_for(batch, WAIT)  # file b waits its 0.3 s, then is refused
            blocker.open(2)
            return answer, await other

    answer, other = run(scenario())

    assert answer.status_code == 200
    first, second, third = answer.json()["results"]
    assert first["filename"] == "a.wav" and first["text"] == "call 1"
    assert second["filename"] == "b.wav" and second["status"] == 503 and "busy" in second["error"]
    assert third["filename"] == "c.wav" and third["status"] == 503
    assert other.status_code == 200 and blocker.calls == 2, "a shed file must not reach the model"
    assert tmp_files(tmp) == []


# --- configuration ---------------------------------------------------------------------------

def test_the_defaults_scale_the_queue_with_the_concurrency(service, monkeypatch, tmp_path):
    module, *_ = build(service, monkeypatch, tmp_path, ASR_MAX_CONCURRENCY="2")

    async def scenario():
        async with make_client(module) as client:
            return await queue(client)

    state = run(scenario())

    assert state["max_concurrency"] == 2 and state["max_queue"] == 8
    assert state["queue_timeout_seconds"] == 60


def test_the_limits_can_be_set(service, monkeypatch, tmp_path):
    module, *_ = build(service, monkeypatch, tmp_path, ASR_MAX_QUEUE="7", ASR_QUEUE_TIMEOUT_S="12.5")

    async def scenario():
        async with make_client(module) as client:
            return await queue(client)

    state = run(scenario())

    assert state["max_queue"] == 7 and state["queue_timeout_seconds"] == 12.5


@pytest.mark.parametrize("env", [
    {"ASR_MAX_QUEUE": "many", "ASR_QUEUE_TIMEOUT_S": "soon", "ASR_MAX_CONCURRENCY": "lots"},
    {"ASR_MAX_QUEUE": "-3", "ASR_QUEUE_TIMEOUT_S": "0", "ASR_MAX_CONCURRENCY": "0"},
], ids=["not-numbers", "out-of-range"])
def test_a_bad_setting_costs_the_setting_not_the_service(service, monkeypatch, tmp_path, env):
    module, *_ = build(service, monkeypatch, tmp_path, **env)

    async def scenario():
        async with make_client(module) as client:
            return await queue(client)

    state = run(scenario())

    assert state["max_concurrency"] == 1 and state["max_queue"] == 4
    assert state["queue_timeout_seconds"] == 60
