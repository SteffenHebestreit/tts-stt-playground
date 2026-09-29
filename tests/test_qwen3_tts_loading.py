"""Loading, switching and readiness in qwen3-tts.

Findings these pin down:

- `load_model` called `torch.cuda.empty_cache()` before `gc.collect()`. The
  wrapper and its modules sit in reference cycles, so the cache had nothing to
  return yet and a model switch peaked at old + new VRAM;
- the default model in code (1.7B) disagreed with compose and `.env.example`
  (0.6B), the model a 12 GB shared card is sized for;
- the model loaded inside the lifespan, so the port stayed closed for the whole
  of a first-time download, and nothing could tell "loading" from "broken";
- `attn_implementation` chose FlashAttention-2 on any machine where the package
  imported, including CPU-only ones.
"""

from __future__ import annotations

import asyncio
import gc
import logging
import runpy
import sys
import threading
import time
import types
import weakref

import pytest

pytest.importorskip("numpy", reason="qwen3-tts app.py requires numpy")

from qwen3_tts_loader import (  # noqa: E402
    BASE_06, BASE_17, CUSTOM_06, SERVICE_DIR, FakeQwenModel, client,
    install_qwen_tts, load_app, stubbed_imports,
)


class Weights:
    """Stand-in for a loaded model that sits in a reference cycle, like the real one."""

    def __init__(self):
        self.me = self


@pytest.fixture
def cuda(monkeypatch):
    """Pretend to be on a CUDA device and record the cache calls."""
    m = load_app()
    monkeypatch.setattr(m, "device", "cuda")
    events = []
    monkeypatch.setattr(m.torch.cuda, "empty_cache", lambda: events.append("empty_cache"), raising=False)
    return m, events


# --- release order -------------------------------------------------------------------


def test_the_old_weights_are_collected_before_the_cache_is_emptied(cuda, monkeypatch):
    m, events = cuda
    old = Weights()
    alive_when = {}
    ref = weakref.ref(old)
    m.tts_model, m.model_loaded, m.current_model_name = old, True, BASE_06
    del old

    monkeypatch.setattr(m.torch.cuda, "empty_cache",
                        lambda: alive_when.setdefault("empty_cache", ref() is not None), raising=False)
    install_qwen_tts(monkeypatch, from_pretrained=lambda name, **kw: (
        alive_when.setdefault("from_pretrained", ref() is not None), FakeQwenModel())[1])

    gc.disable()      # only an explicit collect may free the cycle
    try:
        m.load_model(BASE_17)
    finally:
        gc.enable()

    assert alive_when["empty_cache"] is False, (
        "empty_cache() ran while the old weights were still allocated, so it returned nothing")
    assert alive_when["from_pretrained"] is False, (
        "the new weights started loading with the old ones still resident: VRAM peaks at old + new")


def test_unloading_frees_the_weights_before_emptying_the_cache(cuda, monkeypatch):
    m, _ = cuda
    old = Weights()
    ref = weakref.ref(old)
    m.tts_model, m.model_loaded, m.current_model_name = old, True, BASE_06
    del old
    seen = {}
    monkeypatch.setattr(m.torch.cuda, "empty_cache",
                        lambda: seen.setdefault("alive", ref() is not None), raising=False)

    gc.disable()
    try:
        m._unload_qwen3_tts()
    finally:
        gc.enable()

    assert seen == {"alive": False}
    assert m.tts_model is None and m.current_model_name == "" and m.model_loaded is False


def test_a_failed_load_gives_back_whatever_it_had_allocated(cuda, monkeypatch):
    m, events = cuda

    def broken(name, **kw):
        raise RuntimeError("CUDA out of memory")

    install_qwen_tts(monkeypatch, from_pretrained=broken)

    with pytest.raises(RuntimeError, match="out of memory"):
        m.load_model(BASE_06)

    assert events == ["empty_cache"], "a half-loaded model was left holding its VRAM"
    assert m.tts_model is None and m.current_model_name == "" and m._loading is False
    assert "CUDA out of memory" in m._last_load_error


# --- what gets loaded ------------------------------------------------------------------


def _record_loads(monkeypatch):
    loads = []
    install_qwen_tts(monkeypatch, from_pretrained=lambda name, **kw: (
        loads.append((name, kw)), FakeQwenModel())[1])
    return loads


def test_the_default_model_is_the_0_6b_base_that_compose_and_env_example_name(monkeypatch):
    m = load_app()
    loads = _record_loads(monkeypatch)
    monkeypatch.delenv("QWEN3_TTS_MODEL", raising=False)

    m.load_model()

    assert [name for name, _ in loads] == [BASE_06]
    assert m.current_model_name == BASE_06 and m.model_loaded is True


@pytest.mark.parametrize("configured,expected", [
    (CUSTOM_06, CUSTOM_06),
    ("  " + BASE_17 + "  ", BASE_17),
    ("", BASE_06),          # an empty variable is "unset", not the empty model name
    ("   ", BASE_06),
])
def test_the_environment_can_choose_the_initial_model(monkeypatch, configured, expected):
    m = load_app()
    loads = _record_loads(monkeypatch)
    monkeypatch.setenv("QWEN3_TTS_MODEL", configured)

    m.load_model()

    assert [name for name, _ in loads] == [expected]


def test_loading_an_already_loaded_model_is_a_no_op(monkeypatch):
    m = load_app()
    loads = _record_loads(monkeypatch)
    m.load_model(BASE_06)
    m.load_model(BASE_06)
    assert len(loads) == 1


def test_a_model_outside_the_registry_is_loaded_but_the_operator_is_told(monkeypatch, caplog):
    m = load_app()
    _record_loads(monkeypatch)
    with caplog.at_level(logging.WARNING):
        m.load_model("/models/my-finetune")
    assert "/models/my-finetune" in caplog.text and "registry" in caplog.text


def test_cpu_loading_uses_float32_and_sdpa(monkeypatch):
    m = load_app()
    loads = _record_loads(monkeypatch)
    m.load_model(BASE_06)
    (_, kwargs), = loads
    assert kwargs["device_map"] == "cpu"
    assert kwargs["dtype"] == m.torch.float32
    assert kwargs["attn_implementation"] == "sdpa"


def test_cuda_loading_uses_bfloat16_on_the_first_gpu(monkeypatch):
    m = load_app()
    monkeypatch.setattr(m, "device", "cuda")
    monkeypatch.setitem(sys.modules, "flash_attn", None)       # not installed
    loads = _record_loads(monkeypatch)
    m.load_model(BASE_06)
    (_, kwargs), = loads
    assert kwargs["device_map"] == "cuda:0"
    assert kwargs["dtype"] == m.torch.bfloat16
    assert kwargs["attn_implementation"] == "sdpa"


# --- attention implementation --------------------------------------------------------------


FLASH_ATTN = types.ModuleType("flash_attn")


@pytest.mark.parametrize("configured,device,flash_installed,expected", [
    # auto: the historic behaviour, minus FlashAttention-2 where it cannot run
    ("auto", "cuda", True, "flash_attention_2"),
    ("auto", "cuda", False, "sdpa"),
    ("auto", "cpu", True, "sdpa"),
    ("auto", "cpu", False, "sdpa"),
    (None, "cuda", False, "sdpa"),
    # explicit choices
    ("sdpa", "cuda", True, "sdpa"),
    ("eager", "cuda", True, "eager"),
    ("flash_attention_2", "cuda", True, "flash_attention_2"),
    ("flash_attention_2", "cuda", False, "sdpa"),      # asked for, missing: degrade, do not fail
    ("flash_attention_2", "cpu", True, "sdpa"),
    ("FLASH_ATTENTION_2", "cuda", True, "flash_attention_2"),
    ("turbo", "cuda", True, "flash_attention_2"),      # junk means auto
    ("turbo", "cuda", False, "sdpa"),
])
def test_attention_implementation(monkeypatch, configured, device, flash_installed, expected):
    env = {} if configured is None else {"QWEN3_TTS_ATTN_IMPLEMENTATION": configured}
    m = load_app(**env)
    monkeypatch.setattr(m, "device", device)
    monkeypatch.setitem(sys.modules, "flash_attn", FLASH_ATTN if flash_installed else None)

    assert m._attn_implementation() == expected


# --- readiness --------------------------------------------------------------------------------


def test_ready_says_ok_for_a_resident_model():
    m = load_app()
    m.tts_model, m.current_model_name = object(), BASE_06
    r = client(m).get("/ready")
    assert r.status_code == 200
    assert r.json()["ready"] is True and r.json()["reason"] == "resident"


def test_ready_stays_200_for_a_model_that_was_unloaded_on_purpose():
    m = load_app()
    r = client(m).get("/ready")
    assert r.status_code == 200
    assert r.json()["reason"] == "unloaded" and r.json()["model_resident"] is False


def test_ready_is_503_while_a_load_runs_and_asks_for_a_retry():
    m = load_app()
    m._loading = True
    r = client(m).get("/ready")
    assert r.status_code == 503 and r.json()["reason"] == "loading"
    assert r.headers["retry-after"] == "5"


def test_ready_is_503_until_the_startup_preload_has_finished():
    m = load_app()
    m._preload_pending = True
    r = client(m).get("/ready")
    assert r.status_code == 503 and r.json()["reason"] == "loading"


def test_ready_reports_a_failed_load_with_its_reason(monkeypatch):
    m = load_app()

    def broken(name, **kw):
        raise OSError("no route to huggingface.co")

    install_qwen_tts(monkeypatch, from_pretrained=broken)
    with pytest.raises(OSError):
        m.load_model(BASE_06)

    r = client(m).get("/ready")
    assert r.status_code == 503
    assert r.json()["reason"] == "load_failed"
    # unauthenticated: a category, not the exception text ("no route to huggingface.co")
    assert "huggingface" not in r.text and "OSError" not in r.text

    # the next attempt succeeding clears it
    install_qwen_tts(monkeypatch, from_pretrained=lambda name, **kw: FakeQwenModel())
    m.load_model(BASE_06)
    assert client(m).get("/ready").status_code == 200


def test_health_and_ready_answer_while_a_load_holds_the_lock():
    """An orchestrator polling them must not queue behind a multi-minute load."""
    import httpx
    m = load_app()

    async def scenario():
        transport = httpx.ASGITransport(app=m.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
            async with m._LOAD_LOCK:
                health = await asyncio.wait_for(c.get("/health"), 2)
                ready = await asyncio.wait_for(c.get("/ready"), 2)
                speakers = await asyncio.wait_for(c.get("/speakers"), 2)
                status = await asyncio.wait_for(c.get("/status"), 2)
        return health, ready, speakers, status

    health, ready, speakers, status = asyncio.run(scenario())
    assert [r.status_code for r in (health, ready, speakers, status)] == [200, 200, 200, 200]


def test_health_reports_a_load_in_progress():
    m = load_app()
    m._loading = True
    assert client(m).get("/health").json()["loading"] is True


# --- startup preload ---------------------------------------------------------------------------


def test_the_port_answers_while_the_model_is_still_loading(monkeypatch):
    """The lifespan used to load synchronously: nothing listened until the
    download finished, and a container could be marked unhealthy while it was
    doing exactly what it should."""
    m = load_app()
    started, gate = threading.Event(), threading.Event()

    def slow(name, **kw):
        started.set()
        assert gate.wait(10)
        return FakeQwenModel()

    install_qwen_tts(monkeypatch, from_pretrained=slow)

    from fastapi.testclient import TestClient
    with TestClient(m.app, raise_server_exceptions=False) as c:
        try:
            assert started.wait(5), "the preload never started"
            began = time.monotonic()
            health = c.get("/health")
            ready = c.get("/ready")
            assert time.monotonic() - began < 2, "a status read waited for the load"
            assert health.status_code == 200 and health.json()["loading"] is True
            assert ready.status_code == 503 and ready.json()["reason"] == "loading"
        finally:
            gate.set()
        for _ in range(200):
            if c.get("/ready").status_code == 200 and c.get("/health").json()["model_resident"]:
                break
            time.sleep(0.02)
        assert c.get("/ready").json()["reason"] == "resident"
        assert c.get("/health").json()["current_model"] == BASE_06


def test_a_failed_preload_leaves_the_service_up_and_reports_it(monkeypatch):
    m = load_app()

    def broken(name, **kw):
        raise OSError("disk full")

    install_qwen_tts(monkeypatch, from_pretrained=broken)

    from fastapi.testclient import TestClient
    with TestClient(m.app, raise_server_exceptions=False) as c:
        for _ in range(200):
            if c.get("/ready").json().get("reason") == "load_failed":
                break
            time.sleep(0.02)
        assert c.get("/health").status_code == 200
        ready = c.get("/ready")
        assert ready.status_code == 503 and ready.json()["reason"] == "load_failed"
        assert "disk full" not in ready.text


def test_ttl_zero_skips_the_preload_it_would_immediately_undo(monkeypatch):
    m = load_app(TTS_MODEL_TTL="0")
    loads = _record_loads(monkeypatch)

    from fastapi.testclient import TestClient
    with TestClient(m.app, raise_server_exceptions=False) as c:
        time.sleep(0.2)
        assert c.get("/ready").json()["reason"] == "unloaded"
    assert loads == []


def test_a_request_during_the_preload_waits_for_it_instead_of_loading_a_second_copy(monkeypatch):
    m = load_app()
    started, gate = threading.Event(), threading.Event()
    loads = []

    def slow(name, **kw):
        loads.append(name)
        started.set()
        assert gate.wait(10)
        return FakeQwenModel()

    install_qwen_tts(monkeypatch, from_pretrained=slow)

    async def scenario():
        m._preload_pending = True
        preload = asyncio.create_task(m._startup_preload())
        while not started.is_set():
            await asyncio.sleep(0.005)

        async def request():
            async with m._acquire_model() as (model, name):
                return model, name

        waiting = asyncio.create_task(request())
        await asyncio.sleep(0.1)
        still_waiting = not waiting.done()
        gate.set()
        model, name = await asyncio.wait_for(waiting, 5)
        await preload
        return still_waiting, model, name

    try:
        still_waiting, model, name = asyncio.run(scenario())
    finally:
        gate.set()
    assert still_waiting is True
    assert loads == [BASE_06], "the request started a second load of the same model"
    assert name == BASE_06 and isinstance(model, FakeQwenModel)
    assert m._preload_pending is False


# --- configuration -------------------------------------------------------------------------------


def test_the_legacy_model_ttl_variable_is_still_honoured():
    assert load_app(MODEL_TTL="42").MODEL_TTL == 42.0


def test_the_specific_ttl_wins_over_the_legacy_one():
    assert load_app(TTS_MODEL_TTL="7", MODEL_TTL="42").MODEL_TTL == 7.0


def test_the_default_ttl_is_five_minutes():
    assert load_app().MODEL_TTL == 300.0


def test_a_junk_ttl_falls_back_to_the_default():
    assert load_app(TTS_MODEL_TTL="soon").MODEL_TTL == 300.0


@pytest.mark.parametrize("limit,expect_parallel", [("1", False), ("2", True), (None, False)])
def test_generation_concurrency_is_configurable(limit, expect_parallel):
    env = {} if limit is None else {"TTS_MAX_CONCURRENCY": limit}
    m = load_app(**env)
    running, both_running, release = [], threading.Event(), threading.Event()

    def work():
        running.append(1)
        if len(running) == 2:
            both_running.set()
        release.wait(5)

    async def scenario():
        try:
            first = asyncio.create_task(m._gen(work))
            second = asyncio.create_task(m._gen(work))
            parallel = await asyncio.get_running_loop().run_in_executor(None, both_running.wait, 0.5)
            release.set()
            await asyncio.gather(first, second)
            return parallel
        finally:
            release.set()

    assert asyncio.run(scenario()) is expect_parallel


def test_running_the_module_directly_keeps_connections_alive_for_the_gateway(monkeypatch):
    """The gateway pools upstream sockets; uvicorn's 5 s default closes them first."""
    import uvicorn     # the real one: tests/requirements.txt installs it
    calls = []
    monkeypatch.setattr(uvicorn, "run", lambda *a, **k: calls.append((a, k)))

    # The module is executed again as __main__, so it needs its torch / soundfile
    # stand-ins again; they exist only for the duration of this call.
    with stubbed_imports():
        runpy.run_path(str(SERVICE_DIR / "app.py"), run_name="__main__")

    ((args, kwargs),) = calls
    assert kwargs["port"] == 5004 and kwargs["timeout_keep_alive"] == 120
