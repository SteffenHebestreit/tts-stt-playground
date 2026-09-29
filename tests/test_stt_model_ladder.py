"""How stt-service picks and loads its model: the fallback ladder, custom
(German fine-tune) model ids, the runtime it reports, and the startup preload.

Runs the real load path under stubs (see test_stt_support.py); WhisperModel is a
class the test controls, so an out-of-memory error is just an exception raised
from its constructor.
"""

import logging
import threading

import pytest
from fastapi.testclient import TestClient

from test_stt_support import FakeWhisper, load_stt_app, wait_for_preload, wait_until


@pytest.fixture(scope="module")
def stt_app():
    return load_stt_app({"WHISPER_MODEL_SIZE": "large-v3-turbo"}, name="stt_app_ladder")


@pytest.fixture
def app(stt_app, monkeypatch):
    """The app with nothing loaded and an NVIDIA-shaped preferred runtime."""
    stt_app._idle_unloader.cancel()
    monkeypatch.setenv("WHISPER_MODEL_SIZE", "large-v3-turbo")
    monkeypatch.setattr(stt_app, "whisper_model", None)
    monkeypatch.setattr(stt_app, "model_loaded", False)
    monkeypatch.setattr(stt_app, "model_size_loaded", None)
    monkeypatch.setattr(stt_app, "startup_error", None)
    monkeypatch.setattr(stt_app, "active_device", None)
    monkeypatch.setattr(stt_app, "active_compute_type", None)
    monkeypatch.setattr(stt_app, "device", "cuda")
    monkeypatch.setattr(stt_app, "compute_type", "int8_float16")
    monkeypatch.setattr(stt_app, "_model_refs", 0)
    return stt_app


class Recorder:
    """A WhisperModel whose construction can be made to fail like a CUDA OOM."""

    attempts: list = []
    fail_on_cuda = False
    fail_when = staticmethod(lambda size, device: False)

    def __init__(self, model_size, device=None, compute_type=None, **kwargs):
        Recorder.attempts.append((model_size, device, compute_type))
        self.kwargs = kwargs
        if (Recorder.fail_on_cuda and device == "cuda") or Recorder.fail_when(model_size, device):
            raise RuntimeError("CUDA failed with error out of memory")

    @classmethod
    def reset(cls, **settings):
        cls.attempts = []
        cls.fail_on_cuda = settings.get("fail_on_cuda", False)
        cls.fail_when = staticmethod(settings.get("fail_when", lambda size, device: False))


@pytest.fixture
def recorder(app, monkeypatch):
    Recorder.reset()
    monkeypatch.setattr(app, "WhisperModel", Recorder)
    return Recorder


# --- H1: a fallback is not permanent -----------------------------------------


def test_a_transient_oom_on_the_first_load_does_not_pin_the_service_to_cpu(app, recorder):
    """The ladder used to be rebuilt from globals it had just overwritten, so one
    OOM at boot meant every later load, including the one after an idle unload
    when the VRAM was free again, only ever tried the CPU."""
    recorder.reset(fail_on_cuda=True)
    app.load_model()
    assert app.active_device == "cpu", "the first load should have fallen back"
    assert app.model_size_loaded == "large-v3-turbo"

    recorder.fail_on_cuda = False              # the other service let go of the VRAM
    recorder.attempts.clear()
    assert app.unload_model()["unloaded"] is True
    app.load_model()

    assert recorder.attempts[0] == ("large-v3-turbo", "cuda", "int8_float16"), (
        f"the reload did not start from the preferred runtime: {recorder.attempts}"
    )
    assert app.active_device == "cuda" and app.active_compute_type == "int8_float16"
    assert (app.device, app.compute_type) == ("cuda", "int8_float16"), "the preferred runtime was overwritten"


def test_a_load_on_demand_after_an_unload_also_starts_from_the_top(app, recorder):
    recorder.reset(fail_on_cuda=True)
    app.load_model()
    app.unload_model()
    recorder.fail_on_cuda = False
    recorder.attempts.clear()

    model = app.acquire_model()              # what a request does after an idle unload
    app.release_model()

    assert model is app.whisper_model
    assert recorder.attempts[0][1] == "cuda"


def test_health_reports_the_runtime_actually_in_use(app, recorder):
    recorder.reset(fail_on_cuda=True)
    app.load_model()
    with TestClient(app.app) as client:
        wait_for_preload(app)
        # The preload found a model already resident and left it alone.
        body = client.get("/health").json()
        assert body["device"] == "cpu" and body["compute_type"] == "int8"
        assert body["preferred_device"] == "cuda"
        assert body["degraded"] is True

        recorder.fail_on_cuda = False
        app.unload_model()
        assert client.get("/health").json()["device"] == "cuda", (
            "nothing is resident, so health names what the next load will try"
        )
        app.load_model()
        body = client.get("/health").json()
        assert body["device"] == "cuda" and body["degraded"] is False


def test_the_ladder_is_rebuilt_from_the_preferred_runtime_each_time(app):
    first = app._load_attempts("large-v3-turbo", "cuda", "int8_float16")
    assert first == [
        ("large-v3-turbo", "cuda", "int8_float16"),
        ("small", "cuda", "int8_float16"),
        ("large-v3-turbo", "cpu", "int8"),
        ("small", "cpu", "int8"),
    ]
    assert app._load_attempts("large-v3-turbo", "cuda", "int8_float16") == first
    assert app._load_attempts("large-v3-turbo", "cpu", "int8") == [("large-v3-turbo", "cpu", "int8")]


# --- I1: custom models (German fine-tunes) -----------------------------------


@pytest.mark.parametrize("requested", [
    "primeline/whisper-large-v3-turbo-german",
    "someone/whisper-large-v3-turbo-german-ct2",
])
def test_a_hub_repo_id_is_valid_and_never_escalated_to_large_v3(app, requested):
    attempts = app._load_attempts(requested, "cuda", "int8_float16")

    assert attempts[0] == (requested, "cuda", "int8_float16")
    assert app.FALLBACK_MODEL_SIZE not in [size for size, _, _ in attempts], (
        "a working custom model was 'corrected' to a larger one on failure"
    )
    # It keeps the OOM semantics of a built-in: smaller multilingual model, then CPU.
    assert attempts == [
        (requested, "cuda", "int8_float16"),
        ("small", "cuda", "int8_float16"),
        (requested, "cpu", "int8"),
        ("small", "cpu", "int8"),
    ]


def test_a_local_ct2_directory_is_valid_and_is_measured_by_its_weights(app, tmp_path):
    export = tmp_path / "whisper-german-ct2"
    export.mkdir()
    (export / "model.bin").write_bytes(b"\0" * (3 * 1024 * 1024))
    path = str(export)

    assert app._is_valid_model_name(path)
    assert app._model_rank(path) == 3
    attempts = app._load_attempts(path, "cuda", "int8_float16")
    assert app.FALLBACK_MODEL_SIZE not in [size for size, _, _ in attempts]
    # `small` (490 MB) would be a step UP from a 3 MB export, so there is no such rung.
    assert attempts == [(path, "cuda", "int8_float16"), (path, "cpu", "int8")]


def test_a_custom_model_that_ooms_falls_to_small_then_cpu_never_to_large_v3(app, recorder, monkeypatch):
    requested = "primeline/whisper-large-v3-turbo-german"
    recorder.reset(fail_when=lambda size, device: size == requested and device == "cuda")
    monkeypatch.setenv("WHISPER_MODEL_SIZE", requested)
    app.load_model()

    assert [a[0] for a in recorder.attempts] == [requested, "small"]
    assert app.model_size_loaded == "small" and app.active_device == "cuda"


def test_english_only_builtins_are_not_treated_as_typos(app):
    """`distil-large-v3` is a real name; a failed load must not escalate it to large-v3."""
    attempts = app._load_attempts("distil-large-v3", "cuda", "int8_float16")
    assert "large-v3" not in [size for size, _, _ in attempts]


def test_a_mistyped_name_still_falls_back_to_a_known_one(app, recorder, monkeypatch):
    recorder.reset(fail_when=lambda size, device: size == "large-v4")
    monkeypatch.setenv("WHISPER_MODEL_SIZE", "large-v4")
    app.load_model()
    assert app.model_size_loaded == app.FALLBACK_MODEL_SIZE


def test_the_configured_model_dir_becomes_faster_whispers_download_root():
    constructed = []

    class Capturing(FakeWhisper):
        def __init__(self, *args, **kwargs):
            constructed.append(kwargs)
            super().__init__(*args, **kwargs)

    with_dir = load_stt_app({"WHISPER_MODEL_DIR": "/data/whisper"}, name="stt_app_ladder_dir", model_cls=Capturing)
    with_dir.load_model()
    assert constructed[-1]["download_root"] == "/data/whisper"

    without = load_stt_app({}, name="stt_app_ladder_nodir", model_cls=Capturing)
    without.load_model()
    assert "download_root" not in constructed[-1], (
        "an unset WHISPER_MODEL_DIR must keep the Hugging Face cache, where existing installs have their weights"
    )


def test_a_failed_load_in_offline_mode_says_the_cache_is_the_problem(app, recorder, monkeypatch, caplog):
    """The hint is for the operator, so it goes to the log; a probe gets a category."""
    recorder.reset(fail_when=lambda size, device: True)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    with caplog.at_level(logging.ERROR):
        app.load_model()
    assert app.startup_error
    assert "HF_HUB_OFFLINE" in caplog.text and "cache" in caplog.text
    assert "HF_HUB_OFFLINE" not in app.startup_error

    caplog.clear()
    monkeypatch.delenv("HF_HUB_OFFLINE")
    with caplog.at_level(logging.ERROR):
        app.load_model()
    assert "HF_HUB_OFFLINE" not in caplog.text


# --- startup preload ---------------------------------------------------------


def test_the_port_opens_while_the_first_load_is_still_running():
    """A first start downloads the model. Loading inside the lifespan meant the
    port stayed closed for the whole download and the container looked dead."""
    entered, gate = threading.Event(), threading.Event()

    class Slow(FakeWhisper):
        def __init__(self, *args, **kwargs):
            entered.set()
            gate.wait(15)
            super().__init__(*args, **kwargs)

    app = load_stt_app({}, name="stt_app_ladder_preload", model_cls=Slow)
    opened = threading.Event()
    result = {}

    def run():
        with TestClient(app.app) as client:
            opened.set()
            result["health"] = client.get("/health")
            result["ready"] = client.get("/ready")
            gate.set()
            wait_for_preload(app)
            result["ready_after"] = client.get("/ready")

    runner = threading.Thread(target=run, daemon=True)
    runner.start()
    try:
        assert opened.wait(5), "startup blocked on the model load"
        assert entered.wait(5)
        assert wait_until(lambda: "ready" in result)
        assert result["health"].status_code == 200
        assert result["ready"].status_code == 503 and result["ready"].json()["reason"] == "loading"
    finally:
        gate.set()
        runner.join(20)
    assert result["ready_after"].status_code == 200
    assert result["ready_after"].json()["reason"] == "resident"


def test_a_request_during_the_preload_waits_for_it_instead_of_loading_a_second_copy():
    entered, gate = threading.Event(), threading.Event()
    constructed = []

    class Slow(FakeWhisper):
        def __init__(self, *args, **kwargs):
            constructed.append(1)
            entered.set()
            gate.wait(15)
            super().__init__(*args, **kwargs)
            self.default = []

    app = load_stt_app({}, name="stt_app_ladder_race", model_cls=Slow)
    queued = threading.Event()

    class WatchedLock:
        """The reference lock, reporting the moment a caller has to wait for it.

        The preload holds it for the whole (gated) load, so the first caller that finds
        it taken is the request, parked behind that load. That is the state the test is
        about, and this is how it is known, instead of sleeping and hoping.
        """

        def __init__(self):
            self._lock = threading.Lock()

        def __enter__(self):
            if not self._lock.acquire(blocking=False):
                queued.set()
                self._lock.acquire()
            return self

        def __exit__(self, *exc):
            self._lock.release()

    app._model_ref_lock = WatchedLock()
    with TestClient(app.app) as client:
        assert entered.wait(5)
        answers = {}
        requester = threading.Thread(
            target=lambda: answers.setdefault("r", client.post(
                "/transcribe", files={"audio": ("a.wav", b"RIFF" + b"\0" * 40, "audio/wav")})),
            daemon=True,
        )
        requester.start()
        assert queued.wait(5), "the request never reached the reference lock the preload holds"
        assert constructed == [1], "a request that arrived during the load started a second one"
        gate.set()
        requester.join(10)
        assert not requester.is_alive(), "the request never finished"
        wait_for_preload(app)
    assert constructed == [1], f"the model was loaded {len(constructed)} times"
    # The audio is not valid, but the decode is a scripted stand-in: what matters is that
    # the request was served by the model the preload loaded, so the outcome is exact.
    assert answers["r"].status_code == 200, answers["r"].text
    assert answers["r"].json()["text"] == ""
