"""vllm_engine.Engine: the ``vllm serve`` subprocess's start, failed start, sleep, wake and stop.

The process and its HTTP API are vllm_asr_loader.FakeEngine's stand-ins, so this runs
anywhere. The lesson of #93 (a failed load kept 3.3 GB of the NAS card) is the test that
matters most: whatever goes wrong while the engine starts, its process group is killed
and reaped before the error is raised.
"""

from __future__ import annotations

import signal
from pathlib import Path

import httpx
import pytest

from vllm_asr_loader import FakeEngine, load_module

engine_module = load_module("vllm_engine")
SIGKILL = getattr(signal, "SIGKILL", signal.SIGTERM)


@pytest.fixture
def fake(tmp_path: Path) -> FakeEngine:
    return FakeEngine(tmp_path, "whisper")


def make(fake: FakeEngine, **kwargs):
    transport = httpx.MockTransport(fake.handle_sync)
    clock = {"t": 0.0}

    def advance(seconds):
        clock["t"] += seconds

    return engine_module.Engine(
        ["vllm", "serve", "/m"], "http://127.0.0.1:8100", env={"A": "1"},
        popen=fake.popen, signal_group=fake.signal,
        http=lambda **kw: httpx.Client(transport=transport, **kw),
        clock=lambda: clock["t"], sleep=advance, **kwargs)


def test_a_start_waits_for_health_and_reads_the_kv_cache_size_from_the_log(fake):
    fake.health_failures = 3
    engine = make(fake, poll_s=2.0)
    seconds = engine.start()
    assert engine.state == "running" and engine.alive()
    assert seconds == 6.0  # three failed probes, 2 s apart
    engine._drain_log()
    assert engine.kv_cache_tokens == 12864
    (process,) = fake.processes
    assert process.kwargs["start_new_session"] is True, "a group of its own, so its engine core dies with it"
    assert process.env == {"A": "1"}


@pytest.mark.parametrize("log, error", [
    ("ValueError: To serve at least one request ... larger than the available KV cache memory (1.28 GiB)",
     "EngineOutOfMemory"),
    ("torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB", "EngineOutOfMemory"),
    ("RuntimeError: Engine core initialization failed", "EngineError"),
])
def test_an_engine_that_dies_while_starting_says_why_and_is_reaped(fake, log, error):
    fake.start_log = ["starting", log]
    fake.exit_on_start = 1
    engine = make(fake)
    with pytest.raises(getattr(engine_module, error)) as caught:
        engine.start()
    assert log.split(":")[0] in str(caught.value)
    assert ("out of memory" in str(caught.value).lower()) == (error == "EngineOutOfMemory")
    assert engine.state == "failed" and engine.process is None


def test_a_start_that_never_answers_is_killed_with_its_whole_group(fake):
    """A hung start must give the GPU back too: TERM, then KILL, to the group, and reaped."""
    fake.health_failures = 10**6
    fake.ignore_term = True
    engine = make(fake, start_timeout_s=30.0, poll_s=1.0, stop_timeout_s=0.01)
    with pytest.raises(engine_module.EngineError, match="VLLM_ASR_START_TIMEOUT_S"):
        engine.start()
    (process,) = fake.processes
    assert process.signals == [signal.SIGTERM, SIGKILL]
    assert process.returncode is not None and engine.process is None and engine.state == "failed"


def test_a_process_that_cannot_be_launched_is_an_engine_error(fake):
    def refuse(*args, **kwargs):
        raise FileNotFoundError("vllm")

    engine = make(fake)
    engine._popen = refuse
    with pytest.raises(engine_module.EngineError, match="could not be launched"):
        engine.start()
    assert engine.state == "failed"


def test_stop_ends_the_group_and_is_safe_twice(fake):
    engine = make(fake)
    engine.start()
    engine.stop()
    (process,) = fake.processes
    assert process.signals[0] == signal.SIGTERM and SIGKILL in process.signals
    assert engine.state == "stopped" and engine.process is None and not engine.alive()
    engine.stop()
    assert engine.state == "stopped"


def test_sleep_and_wake_are_checked_against_what_the_engine_reports(fake):
    engine = make(fake)
    engine.start()
    engine.sleep_engine(1)
    assert engine.state == "sleeping" and fake.sleeping and engine.sleep_seconds is not None
    assert fake.control[:2] == ["/sleep", "/is_sleeping"]
    engine.wake()
    assert engine.state == "running" and not fake.sleeping and engine.wake_seconds is not None
    assert len(fake.processes) == 1, "waking is not a restart"


def test_an_engine_that_does_not_go_to_sleep_is_an_error(fake):
    engine = make(fake)
    engine.start()

    def ignore_sleep(request):
        if request.url.path == "/sleep":
            return httpx.Response(200)
        return fake.handle_sync(request)

    transport = httpx.MockTransport(ignore_sleep)
    engine._http = lambda **kw: httpx.Client(transport=transport, **kw)
    with pytest.raises(engine_module.EngineError, match="did not report sleeping"):
        engine.sleep_engine(1)


def test_control_calls_on_a_dead_engine_fail_at_once(fake):
    engine = make(fake)
    engine.start()
    fake.current.returncode = -9
    assert engine.snapshot()["state"] == "failed"
    with pytest.raises(engine_module.EngineError, match="not running"):
        engine.wake()
