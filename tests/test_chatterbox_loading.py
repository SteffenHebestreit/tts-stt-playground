"""Model loading: T3 checkpoint selection, revision pin, repetition penalty, readiness.

chatterbox-tts 0.1.7 (PyPI, 2026-03-26) predates the v3 checkpoint: its
``ChatterboxMultilingualTTS.from_pretrained(cls, device)`` has no ``t3_model``, so
naively passing it raises TypeError. Only GitHub master accepts ``t3_model="v3"``.
The library is faked with both signatures (tests/chatterbox_loader.py).
"""

import logging
import sys
import threading

import pytest
from fastapi.testclient import TestClient

from chatterbox_loader import install_chatterbox, load_app, wait_until


def _warm(client):
    return client.post("/tts", json={"text": "Hallo.", "language": "de"})


def _use_loader(app, loader):
    """Swap the slot's loader (the real slot class and settings, a different load step)."""
    app._model_slot = app.ModelSlot(
        loader, ttl_seconds=app.MODEL_TTL, name="Chatterbox Multilingual", on_unload=app._forget_chatterbox
    )


# --- CHATTERBOX_T3_MODEL -----------------------------------------------------

def test_v3_is_the_default_and_is_passed_when_the_package_supports_it(monkeypatch):
    package = install_chatterbox(monkeypatch, accepts_t3_model=True)
    app = load_app()
    client = TestClient(app.app)

    assert _warm(client).status_code == 200

    assert package.from_pretrained_calls == [{"device": "cpu", "t3_model": "v3"}]
    health = client.get("/health").json()
    assert health["t3_model"] == "v3"
    assert health["t3_model_requested"] == "v3"
    assert client.get("/status").json()["t3_model"] == "v3"


def test_an_old_package_falls_back_to_the_default_checkpoint_with_a_warning(monkeypatch, caplog):
    """PyPI 0.1.7: from_pretrained(device) only. Must not crash, must say so."""
    package = install_chatterbox(monkeypatch, accepts_t3_model=False)
    app = load_app()
    client = TestClient(app.app)

    with caplog.at_level(logging.WARNING):
        response = _warm(client)

    assert response.status_code == 200
    assert package.from_pretrained_calls == [{"device": "cpu", "t3_model": None}]
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("CHATTERBOX_T3_MODEL=v3" in w and "does not accept t3_model" in w for w in warnings), warnings
    health = client.get("/health").json()
    assert health["t3_model"] == "default", "/health must report what actually loaded, not what was asked for"
    assert health["t3_model_requested"] == "v3"


def test_the_legacy_checkpoint_can_be_requested_explicitly(monkeypatch):
    package = install_chatterbox(monkeypatch, accepts_t3_model=True)
    client = TestClient(load_app(CHATTERBOX_T3_MODEL="v2").app)
    _warm(client)
    assert package.from_pretrained_calls[0]["t3_model"] == "v2"
    assert client.get("/health").json()["t3_model"] == "v2"


@pytest.mark.parametrize("value", ["default", "none"])
def test_default_means_the_library_default_and_omits_the_argument(monkeypatch, value):
    package = install_chatterbox(monkeypatch, accepts_t3_model=True)
    client = TestClient(load_app(CHATTERBOX_T3_MODEL=value).app)
    _warm(client)
    assert package.from_pretrained_calls[0]["t3_model"] is None
    assert client.get("/health").json()["t3_model"] == "default"


def test_v2_on_an_old_package_is_not_a_warning(monkeypatch, caplog):
    install_chatterbox(monkeypatch, accepts_t3_model=False)
    client = TestClient(load_app(CHATTERBOX_T3_MODEL="v2").app)
    with caplog.at_level(logging.WARNING):
        assert _warm(client).status_code == 200
    assert not [r for r in caplog.records if "t3_model" in r.getMessage()]


def test_health_reports_no_checkpoint_before_the_first_load_and_never_loads_one(monkeypatch):
    package = install_chatterbox(monkeypatch)
    client = TestClient(load_app().app)

    health = client.get("/health").json()

    assert health["t3_model"] is None
    assert health["t3_model_requested"] == "v3"
    assert package.loads == 0


def test_the_checkpoint_label_survives_an_idle_unload(monkeypatch):
    install_chatterbox(monkeypatch)
    app = load_app()
    client = TestClient(app.app)
    _warm(client)

    assert client.post("/unload").json()["unloaded"] is True

    health = client.get("/health").json()
    assert health["model_resident"] is False
    assert health["t3_model"] == "v3"


# --- CHATTERBOX_REPETITION_PENALTY -------------------------------------------

def test_the_library_default_repetition_penalty_is_used_when_unset(monkeypatch):
    package = install_chatterbox(monkeypatch)
    client = TestClient(load_app().app)
    _warm(client)
    # The fake's own default (2.0, like chatterbox-tts 0.1.7): nothing was passed.
    assert package.model.calls[-1]["repetition_penalty"] == 2.0
    assert client.get("/status").json()["repetition_penalty"] == 2.0


def test_an_explicit_repetition_penalty_is_applied_to_every_chunk(monkeypatch):
    package = install_chatterbox(monkeypatch)
    client = TestClient(load_app(CHATTERBOX_REPETITION_PENALTY="1.2").app)
    client.post("/tts", json={"text": "Erster Satz ist hier. " * 20, "language": "de"})

    assert len(package.model.calls) > 1
    assert {c["repetition_penalty"] for c in package.model.calls} == {1.2}
    assert client.get("/status").json()["repetition_penalty"] == 1.2


@pytest.mark.parametrize("value", ["abc", "0.5", "nan", "inf"])
def test_an_unusable_repetition_penalty_is_ignored_not_fatal(monkeypatch, value):
    package = install_chatterbox(monkeypatch)
    client = TestClient(load_app(CHATTERBOX_REPETITION_PENALTY=value).app)
    assert _warm(client).status_code == 200
    assert package.model.calls[-1]["repetition_penalty"] == 2.0


# --- CHATTERBOX_HF_REVISION --------------------------------------------------

def test_the_default_revision_leaves_the_library_alone(monkeypatch):
    package = install_chatterbox(monkeypatch)
    module = sys.modules["chatterbox.mtl_tts"]
    original = module.snapshot_download
    client = TestClient(load_app().app)

    _warm(client)

    assert package.snapshot_calls[-1]["revision"] == "main"
    assert client.get("/status").json()["hf_revision"] == "main"
    assert module.snapshot_download is original


def test_a_pinned_revision_reaches_the_download_and_is_undone_afterwards(monkeypatch):
    package = install_chatterbox(monkeypatch)
    module = sys.modules["chatterbox.mtl_tts"]
    original = module.snapshot_download
    client = TestClient(load_app(CHATTERBOX_HF_REVISION="0123abcd").app)

    assert _warm(client).status_code == 200

    assert package.snapshot_calls[-1]["revision"] == "0123abcd"
    assert client.get("/status").json()["hf_revision"] == "0123abcd"
    assert module.snapshot_download is original, "the library was left patched"


def test_a_pinned_revision_the_library_cannot_honour_fails_loudly(monkeypatch):
    install_chatterbox(monkeypatch)
    module = sys.modules["chatterbox.mtl_tts"]
    del module.snapshot_download
    app = load_app(CHATTERBOX_HF_REVISION="0123abcd")
    client = TestClient(app.app, raise_server_exceptions=False)

    response = _warm(client)

    assert response.status_code == 500
    assert "CHATTERBOX_HF_REVISION" in response.json()["detail"]


# --- /ready ------------------------------------------------------------------

def test_ready_is_200_before_any_load_and_does_not_start_one(monkeypatch):
    package = install_chatterbox(monkeypatch)
    client = TestClient(load_app().app)

    response = client.get("/ready")

    assert response.status_code == 200
    assert response.json()["ready"] is True
    assert package.loads == 0


def test_ready_is_503_while_the_first_load_runs_and_200_after(monkeypatch):
    package = install_chatterbox(monkeypatch)
    app = load_app()
    client = TestClient(app.app)
    release = threading.Event()
    real_load = app._load_chatterbox

    def slow_load():
        assert release.wait(10)
        return real_load()

    _use_loader(app, slow_load)
    loader = threading.Thread(target=lambda: _warm(client), daemon=True)
    loader.start()
    assert wait_until(lambda: app._model_slot.loading), "the load never started"

    during = client.get("/ready")
    health_during = client.get("/health")
    release.set()
    loader.join(10)
    after = client.get("/ready")

    assert during.status_code == 503
    assert during.json()["reason"] == "loading"
    assert health_during.status_code == 200, "/health must stay 200 while the model loads"
    assert after.status_code == 200
    assert after.json()["ever_loaded"] is True


def test_ready_reports_a_failed_first_load_and_recovers(monkeypatch):
    package = install_chatterbox(monkeypatch)
    app = load_app()
    client = TestClient(app.app, raise_server_exceptions=False)
    good_loader = app._load_chatterbox

    def failing_load():
        raise OSError("disk full while downloading the checkpoint")

    _use_loader(app, failing_load)
    assert _warm(client).status_code == 500
    failed = client.get("/ready")

    _use_loader(app, good_loader)
    assert _warm(client).status_code == 200
    recovered = client.get("/ready")

    assert failed.status_code == 503
    assert failed.json()["reason"] == "load_failed"
    assert "disk full" in failed.json()["detail"]
    assert recovered.status_code == 200


def test_an_idle_unload_does_not_make_the_service_unready(monkeypatch):
    install_chatterbox(monkeypatch)
    client = TestClient(load_app().app)
    _warm(client)
    client.post("/unload")

    ready = client.get("/ready")

    assert ready.status_code == 200
    assert ready.json()["resident"] is False


# --- Startup preload ---------------------------------------------------------

def test_the_model_is_preloaded_in_the_background_and_freed_on_shutdown(monkeypatch):
    package = install_chatterbox(monkeypatch)
    app = load_app()

    with TestClient(app.app) as client:
        assert wait_until(lambda: package.loads == 1 and app._model_slot.resident), "no preload happened"
        assert client.get("/ready").json()["ever_loaded"] is True
        assert app._model_slot.refs == 0, "the preload kept a reference"

    assert not app._model_slot.resident, "shutdown left the weights loaded"


def test_a_preload_failure_does_not_stop_the_service_from_starting(monkeypatch):
    install_chatterbox(monkeypatch)
    app = load_app()

    def failing_load():
        raise OSError("no network for the download")

    _use_loader(app, failing_load)
    with TestClient(app.app) as client:
        assert wait_until(lambda: app._model_slot.last_error is not None)
        assert client.get("/health").status_code == 200
        assert client.get("/ready").status_code == 503


def test_startup_does_not_wait_for_the_model(monkeypatch):
    """A first start downloads ~3 GB; the port must answer /health meanwhile."""
    install_chatterbox(monkeypatch)
    app = load_app()
    release = threading.Event()
    real_load = app._load_chatterbox

    def slow_load():
        assert release.wait(10)
        return real_load()

    _use_loader(app, slow_load)
    with TestClient(app.app) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/ready").json()["reason"] == "loading"
        release.set()
        assert wait_until(lambda: app._model_slot.resident)
