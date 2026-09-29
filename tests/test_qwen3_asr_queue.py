"""Admission control in the Qwen3-ASR service: how many requests wait, and for how long.

Finding covered: every request that reached the forward-pass semaphore had already
spooled its upload and decoded the audio into memory (a 2 h recording is ~460 MB of
float32), then waited for the ones ahead of it with no limit on the number or the
time. A burst of uploads piled up temp files and RAM and held every client until
the slowest one finished.

Now ``ASR_MAX_CONCURRENCY + ASR_MAX_QUEUE`` requests are admitted (the next is
answered 503 + Retry-After at once, before its upload is touched), and a request
that waits longer than ``ASR_QUEUE_TIMEOUT_S`` for its turn is answered the same
way. Whichever way a request ends, its temp file goes and its model reference is
handed back.

Every test drives the real app over ASGI (fakes only for torch, soundfile, librosa,
ffmpeg and the qwen_asr model, see ``test_qwen3_asr_harness``). The model call is
held "in the forward pass" by wrapping ``_run_model`` with an event per call.
"""

from __future__ import annotations

import asyncio
import threading

import pytest

from test_qwen3_asr_harness import (
    WAIT, encoded_wav, install_fake_qwen_asr, load_app, make_client, run, speech_levels, tmp_files,
    use_tmpdir, wait_for,
)


class Gated:
    """``_run_model`` that parks each numbered call until the test opens its gate."""

    def __init__(self, real, gated_calls=(1,)):
        self.real = real
        self.gates = {n: threading.Event() for n in gated_calls}
        self.calls = 0
        self.entered = threading.Event()
        self._lock = threading.Lock()

    def __call__(self, model, pieces, language):
        with self._lock:
            self.calls += 1
            number = self.calls
        self.entered.set()
        gate = self.gates.get(number)
        if gate is not None:
            assert gate.wait(WAIT), f"the test never opened the gate of call {number}"
        return self.real(model, pieces, language)

    def open(self, number=1):
        self.gates[number].set()

    def open_all(self):
        for gate in self.gates.values():
            gate.set()


def build(tmp_path, monkeypatch, *, gated_calls=(1,), **env):
    """(module, gated model call, temp dir); no ffmpeg, so a native WAV is decoded by the fake librosa."""
    tmp = tmp_path / "tmp"
    tmp.mkdir()
    use_tmpdir(monkeypatch, tmp)
    env.setdefault("ASR_MODEL_TTL", "-1")
    module = load_app(**env)
    install_fake_qwen_asr(monkeypatch)
    monkeypatch.setattr(module, "_FFMPEG", None, raising=False)
    gated = Gated(module._run_model, gated_calls)
    monkeypatch.setattr(module, "_run_model", gated)
    return module, gated, tmp


CLIP = encoded_wav(speech_levels(10))


async def post(client, name="clip.wav", path="/transcribe", field="audio"):
    return await client.post(path, files={field: (name, CLIP, "audio/wav")})


async def post_batch(client, names):
    files = [("audios", (name, CLIP, "audio/wav")) for name in names]
    return await client.post("/transcribe-batch", files=files)


async def queue(client) -> dict:
    return (await client.get("/status")).json()["queue"]


async def in_flight(client, count):
    deadline = asyncio.get_running_loop().time() + WAIT
    while (await queue(client))["in_flight"] != count:
        assert asyncio.get_running_loop().time() < deadline, f"in_flight never reached {count}"
        await asyncio.sleep(0.01)


def counting_load_upload(module, monkeypatch):
    """The filenames `load_upload` was called for: the work a refused request must never cause."""
    seen = []
    real = module.load_upload

    async def counting(upload, **kwargs):
        seen.append(upload.filename)
        return await real(upload, **kwargs)

    monkeypatch.setattr(module, "load_upload", counting)
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

def test_a_request_beyond_the_queue_is_refused_at_once_and_does_no_work(tmp_path, monkeypatch):
    module, gated, tmp = build(tmp_path, monkeypatch, ASR_MAX_CONCURRENCY="1", ASR_MAX_QUEUE="2")
    loaded = counting_load_upload(module, monkeypatch)

    async def scenario():
        async with make_client(module) as client:
            running = asyncio.create_task(post(client, "run.wav"))
            await wait_for(gated.entered.is_set, "the first request to reach the model")
            waiting = [asyncio.create_task(post(client, f"wait{i}.wav")) for i in range(2)]
            await in_flight(client, 3)

            # One running, two waiting: the fourth is refused while the others are
            # still stuck, so this cannot be it merely waiting for its turn.
            shed = await asyncio.wait_for(post(client, "shed.wav"), WAIT)
            during = await queue(client)

            gated.open_all()
            done = await asyncio.gather(running, *waiting)
            calls = gated.calls
            return shed, during, done, calls, await post(client, "retry.wav")

    shed, during, done, calls, retry = run(scenario())

    assert shed.status_code == 503
    assert shed.headers["retry-after"] == "5"
    assert "busy" in shed.json()["detail"].lower()
    assert "shed.wav" not in loaded, "the refused request was already spooled and decoded"
    assert during["max_concurrency"] == 1 and during["max_queue"] == 2
    assert (during["in_flight"], during["running"], during["waiting"]) == (3, 1, 2)
    assert [r.status_code for r in done] == [200, 200, 200]
    assert calls == 3, "the refused request reached the model"
    assert retry.status_code == 200, "the place was not given back"
    assert module._model_slot.refs == 0 and tmp_files(tmp) == []


def test_every_route_that_takes_an_upload_is_behind_the_gate(tmp_path, monkeypatch):
    module, gated, tmp = build(tmp_path, monkeypatch, ASR_MAX_CONCURRENCY="1", ASR_MAX_QUEUE="0")

    async def scenario():
        async with make_client(module) as client:
            running = asyncio.create_task(post(client, "run.wav"))
            await wait_for(gated.entered.is_set, "the first request to reach the model")
            answers = {
                "transcribe": await post(client),
                "detect": await post(client, path="/detect_language", field="file"),
                "batch": await post_batch(client, ["a.wav", "b.wav"]),
            }
            gated.open_all()
            await running
            return answers

    answers = run(scenario())

    assert {name: r.status_code for name, r in answers.items()} == {
        "transcribe": 503, "detect": 503, "batch": 503}
    assert all(r.headers["retry-after"] == "5" for r in answers.values())
    assert gated.calls == 1


def test_a_batch_takes_one_place_however_many_files_it_carries(tmp_path, monkeypatch):
    module, gated, tmp = build(
        tmp_path, monkeypatch, gated_calls=(), ASR_MAX_CONCURRENCY="1", ASR_MAX_QUEUE="0")

    async def scenario():
        async with make_client(module) as client:
            return await post_batch(client, ["a.wav", "b.wav", "c.wav"])

    response = run(scenario())

    assert response.status_code == 200
    assert [r["filename"] for r in response.json()["results"]] == ["a.wav", "b.wav", "c.wav"]
    assert all("text" in r for r in response.json()["results"])


# --- the wait is bounded ---------------------------------------------------------------------

def test_a_waiter_that_outlasts_the_timeout_gets_503_and_gives_its_reference_back(tmp_path, monkeypatch):
    module, gated, tmp = build(
        tmp_path, monkeypatch, ASR_MAX_CONCURRENCY="1", ASR_MAX_QUEUE="2", ASR_QUEUE_TIMEOUT_S="0.3")

    async def scenario():
        async with make_client(module) as client:
            running = asyncio.create_task(post(client, "run.wav"))
            await wait_for(gated.entered.is_set, "the first request to reach the model")
            timed_out = await asyncio.wait_for(post(client, "late.wav"), WAIT)
            refs = module._model_slot.refs  # the running request is still in the model
            state = await queue(client)
            gated.open_all()
            return timed_out, refs, state, await running

    timed_out, refs, state, first = run(scenario())

    assert timed_out.status_code == 503
    assert timed_out.headers["retry-after"] == "5"
    assert "ASR_QUEUE_TIMEOUT_S" in timed_out.json()["detail"]
    assert refs == 1, "the timed-out request kept its model reference"
    assert state["in_flight"] == 1 and state["waiting"] == 0
    assert first.status_code == 200
    assert gated.calls == 1, "the timed-out request reached the model"
    assert module._model_slot.refs == 0 and tmp_files(tmp) == []


def test_a_timed_out_waiter_does_not_take_the_turn_from_the_next_one(tmp_path, monkeypatch):
    module, gated, tmp = build(
        tmp_path, monkeypatch, ASR_MAX_CONCURRENCY="1", ASR_MAX_QUEUE="2", ASR_QUEUE_TIMEOUT_S="0.3")

    async def scenario():
        async with make_client(module) as client:
            running = asyncio.create_task(post(client, "run.wav"))
            await wait_for(gated.entered.is_set, "the first request to reach the model")
            assert (await post(client, "late.wav")).status_code == 503
            gated.open_all()
            assert (await running).status_code == 200
            return await asyncio.wait_for(post(client, "next.wav"), WAIT)

    assert run(scenario()).status_code == 200


def test_a_cancelled_waiter_gives_back_its_model_reference(tmp_path, monkeypatch):
    module, gated, tmp = build(tmp_path, monkeypatch, ASR_MAX_CONCURRENCY="1", ASR_MAX_QUEUE="2")

    async def scenario():
        async with make_client(module) as client:
            running = asyncio.create_task(post(client, "run.wav"))
            await wait_for(gated.entered.is_set, "the first request to reach the model")
            waiter = asyncio.create_task(post(client, "gone.wav"))
            await wait_for(lambda: module._model_slot.refs == 2, "the waiter to hold its reference")
            await in_flight(client, 2)

            waiter.cancel()
            with pytest.raises(asyncio.CancelledError):
                await waiter
            await wait_for(lambda: module._model_slot.refs == 1, "the cancelled waiter's reference to go back")
            state = await queue(client)

            gated.open_all()
            return state, await running, await asyncio.wait_for(post(client, "next.wav"), WAIT)

    state, first, later = run(scenario())

    assert state["in_flight"] == 1 and state["waiting"] == 0
    assert first.status_code == 200 and later.status_code == 200
    assert gated.calls == 2, "the cancelled request must never have run"
    assert module._model_slot.refs == 0 and tmp_files(tmp) == []


def test_a_batch_serves_what_it_can_and_reports_the_rest_as_503(tmp_path, monkeypatch):
    """The first file is served; another request takes the turn; the second file
    then outwaits the queue and the third is not made to wait as long again."""
    module, gated, tmp = build(
        tmp_path, monkeypatch, gated_calls=(1, 2),
        ASR_MAX_CONCURRENCY="1", ASR_MAX_QUEUE="2", ASR_QUEUE_TIMEOUT_S="0.3")
    asked = turn_requests(module, monkeypatch)

    async def scenario():
        async with make_client(module) as client:
            batch = asyncio.create_task(post_batch(client, ["a.wav", "b.wav", "c.wav"]))
            await wait_for(gated.entered.is_set, "the batch's first file to reach the model")
            other = asyncio.create_task(post(client, "other.wav"))
            await wait_for(lambda: len(asked) == 2, "the other request to ask for its turn")
            gated.open(1)  # the first file finishes; `other` was waiting first and takes the turn
            await wait_for(lambda: gated.calls == 2, "the other request to take the turn")
            answer = await asyncio.wait_for(batch, WAIT)  # file b waits its 0.3 s, then is refused
            gated.open(2)
            return answer, await other

    answer, other = run(scenario())

    assert answer.status_code == 200
    first, second, third = answer.json()["results"]
    assert first["filename"] == "a.wav" and first["text"]
    assert second["filename"] == "b.wav" and second["status"] == 503 and "busy" in second["error"]
    assert third["filename"] == "c.wav" and third["status"] == 503
    assert other.status_code == 200 and gated.calls == 2, "a shed file must not reach the model"
    assert module._model_slot.refs == 0


def test_a_batch_that_gets_no_turn_at_all_is_a_plain_503(tmp_path, monkeypatch):
    module, gated, tmp = build(
        tmp_path, monkeypatch, ASR_MAX_CONCURRENCY="1", ASR_MAX_QUEUE="2", ASR_QUEUE_TIMEOUT_S="0.3")

    async def scenario():
        async with make_client(module) as client:
            running = asyncio.create_task(post(client, "run.wav"))
            await wait_for(gated.entered.is_set, "the first request to reach the model")
            batch = await asyncio.wait_for(post_batch(client, ["a.wav", "b.wav"]), WAIT)
            gated.open_all()
            return batch, await running

    batch, first = run(scenario())

    assert batch.status_code == 503 and batch.headers["retry-after"] == "5"
    assert first.status_code == 200 and gated.calls == 1


# --- configuration ---------------------------------------------------------------------------

def test_the_defaults_scale_the_queue_with_the_concurrency(tmp_path, monkeypatch):
    module, *_ = build(tmp_path, monkeypatch, ASR_MAX_CONCURRENCY="2")

    async def scenario():
        async with make_client(module) as client:
            return await queue(client)

    state = run(scenario())

    assert state["max_concurrency"] == 2 and state["max_queue"] == 8
    assert state["queue_timeout_seconds"] == 60


def test_the_limits_can_be_set(tmp_path, monkeypatch):
    module, *_ = build(tmp_path, monkeypatch, ASR_MAX_QUEUE="7", ASR_QUEUE_TIMEOUT_S="12.5")

    async def scenario():
        async with make_client(module) as client:
            return await queue(client)

    state = run(scenario())

    assert state["max_queue"] == 7 and state["queue_timeout_seconds"] == 12.5


@pytest.mark.parametrize("env", [
    {"ASR_MAX_QUEUE": "many", "ASR_QUEUE_TIMEOUT_S": "soon", "ASR_MAX_CONCURRENCY": "lots"},
    {"ASR_MAX_QUEUE": "-3", "ASR_QUEUE_TIMEOUT_S": "0", "ASR_MAX_CONCURRENCY": "0"},
], ids=["not-numbers", "out-of-range"])
def test_a_bad_setting_costs_the_setting_not_the_service(tmp_path, monkeypatch, env):
    module, *_ = build(tmp_path, monkeypatch, **env)

    async def scenario():
        async with make_client(module) as client:
            return await queue(client)

    state = run(scenario())

    assert state["max_concurrency"] == 1 and state["max_queue"] == 4
    assert state["queue_timeout_seconds"] == 60
