"""magpie-tts-service startup and shutdown: background preload, /ready meanwhile, unload at exit, idle TTL.

Only a ``with TestClient(...)`` block runs the app's lifespan; the other magpie tests use a
bare client or call the handlers. Without these, nothing would check that the preload is
scheduled at all, that it runs in the background (a first start downloads ~2.5 GB), that
it gives its reference back (or the idle TTL never arms and /unload answers 409 forever),
or that shutdown takes the weights off the GPU.
"""

import threading

from fastapi.testclient import TestClient

from magpie_loader import install_nemo, load_app, wait_until


def _hold_load(package):
    """Make the next model load block until the returned ``release`` event is set."""
    started, release = threading.Event(), threading.Event()
    real_new = package._new

    def slow_new(name):
        started.set()
        assert release.wait(10), "the test never released the load"
        return real_new(name)

    package._new = slow_new
    return started, release


def test_startup_does_not_wait_for_the_model_and_ready_says_loading_meanwhile(monkeypatch):
    package = install_nemo(monkeypatch)
    app = load_app()
    started, release = _hold_load(package)

    with TestClient(app.app) as client:
        assert started.wait(5), "no preload was started"
        health, loading = client.get("/health"), client.get("/ready")
        release.set()
        assert wait_until(lambda: app._model_slot.resident and not app._preload_pending)
        ready = client.get("/ready")

    assert health.status_code == 200, "/health is liveness: a load in progress is not a failure"
    assert loading.status_code == 503 and loading.json()["reason"] == "loading"
    assert loading.headers["Retry-After"] == "5"
    assert ready.status_code == 200 and ready.json()["model_ever_loaded"] is True


def test_the_preload_warms_the_model_gives_its_reference_back_and_shutdown_frees_it(monkeypatch):
    package = install_nemo(monkeypatch)
    app = load_app()

    with TestClient(app.app):
        assert wait_until(lambda: package.loads == 1 and app._model_slot.resident), "no preload happened"
        # The lease goes back from a thread a moment after the load: wait for it.
        assert wait_until(lambda: app._model_slot.refs == 0), "the preload kept its reference"
        assert [c["language"] for c in package.calls] == ["de"], "the preload is what warms the default language"
        model = package.model

    assert not app._model_slot.resident, "shutdown left the weights loaded"
    assert model.moved_to_cpu, "shutdown left the weights on the GPU"


def test_a_preload_failure_leaves_the_service_up_and_not_ready(monkeypatch):
    install_nemo(monkeypatch, load_error=OSError("no network for the download: /root/.cache/huggingface"))
    app = load_app()

    with TestClient(app.app) as client:
        assert wait_until(lambda: app._model_slot.last_error is not None and not app._preload_pending)
        health, ready = client.get("/health"), client.get("/ready")

    assert health.status_code == 200
    assert ready.status_code == 503 and ready.json()["reason"] == "load_failed"
    assert "/root/.cache" not in ready.text


def test_a_scheduled_but_not_yet_started_preload_counts_as_loading(monkeypatch):
    package = install_nemo(monkeypatch)
    app = load_app()
    app._preload_pending = True

    response = TestClient(app.app).get("/ready")

    assert response.status_code == 503 and response.json()["reason"] == "loading"
    assert package.loads == 0


def test_the_idle_ttl_is_five_minutes_unless_set():
    # load_app passes TTS_MODEL_TTL=-1 unless told otherwise; "" is "not set".
    assert load_app(TTS_MODEL_TTL="").MODEL_TTL == 300
    assert load_app(TTS_MODEL_TTL="", MODEL_TTL="60").MODEL_TTL == 60, "MODEL_TTL is the global fallback"
    assert load_app(TTS_MODEL_TTL="30", MODEL_TTL="60").MODEL_TTL == 30


def test_ttl_zero_frees_the_model_once_the_request_is_done(monkeypatch):
    package = install_nemo(monkeypatch)
    app = load_app(TTS_MODEL_TTL="0")
    client = TestClient(app.app)

    assert client.post("/tts", json={"text": "Hallo Welt.", "language": "de"}).status_code == 200

    assert wait_until(lambda: not app._model_slot.resident), "TTS_MODEL_TTL=0 kept the model"
    assert package.model.moved_to_cpu
