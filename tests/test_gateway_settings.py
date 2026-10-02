"""The Settings page's values in force in the gateway, live and in every worker.

Part 1: the values are saved the way the page saves them, through settings_store's
API, into a temporary folder named by TTS_STT_SETTINGS_DIR, or written by hand where
a test needs a damaged file. Requests then go through a TestClient, and the very next
one has to run under the change: in the module that saved it and in a second module
instance on the same folder (a second uvicorn worker), with no re-import and no wait.

Part 2: the Settings API itself (/settings, /api/settings*): its request rules, the
lockout self-check, commit-confirm, keys, the claim with a one-time code taken from
the container log (caplog), and the engine probe.

Numbers in the test names refer to the gateway tests of the phase-1 plan.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import importlib
import json
import logging
import os
import re
import secrets
import sys
import time

import httpx
import pytest
from fastapi.testclient import TestClient
from starlette.datastructures import Headers
from starlette.websockets import WebSocketDisconnect

from frontend_loader import REPO, SERVICE_DIR, asgi_call, install_stub, load_frontend_app, wav_bytes


def _import(name: str):
    sys.path.insert(0, str(SERVICE_DIR))
    try:
        return importlib.import_module(name)
    finally:
        sys.path.remove(str(SERVICE_DIR))


store_module = _import("settings_store")
router_module = _import("openai_router")

# Who saves: the admin, from the NAS's IP address (the way the page will).
ACTOR = store_module.Actor(ip="192.168.1.20", host="192.168.1.20:3000", credential=store_module.DEPLOYMENT_KEY)
DELETE_JOB = "/api/training/model/job-1"
YAML_KEY = "the-key-from-the-app-yaml"


def new_key() -> str:
    """A key as the Settings page generates it: tts_ and 256 bits of base64url."""
    return "tts_" + base64.urlsafe_b64encode(secrets.token_bytes(32)).decode("ascii").rstrip("=")


def bearer(key: str) -> dict:
    return {"Authorization": f"Bearer {key}"}


def backend(method, url, kwargs):
    if url.endswith("/tts"):
        return httpx.Response(200, content=wav_bytes(), headers={"content-type": "audio/wav"})
    if url.endswith(("/transcribe", "/inference")):
        return {"text": "hallo", "segments": [], "language": "de"}
    if url.endswith("/status") and "canary" in url:
        return {"supported_languages": ["de", "en"], "default_language": "de",
                "current_model": "nvidia/canary-1b-flash"}
    if url.endswith("/jobs"):
        return {"jobs": []}
    return {"status": "ok"}


@pytest.fixture(autouse=True)
def _own_ffmpeg_slots(monkeypatch):
    """openai_router is one module for every gateway a test loads: keep its slots per test."""
    monkeypatch.setattr(router_module, "_ffmpeg_slots", router_module.Slots(router_module.MAX_CONCURRENT_FFMPEG))


@pytest.fixture
def folder(tmp_path):
    path = tmp_path / "settings"
    path.mkdir()
    return path


@pytest.fixture
def writer(folder, tmp_path):
    """Saves like the Settings page (or another worker) does; a gateway only sees the files."""
    return store_module.SettingsStore(folder, verify_mount=False, lock_path=str(tmp_path / "settings.lock"))


def save(writer, **values):
    """Save these values on top of what is saved; None removes one (Reset to YAML)."""
    writer.refresh()
    return writer.update_preferences(
        {key: value for key, value in values.items() if value is not None},
        [key for key, value in values.items() if value is None],
        base_revision=writer.state.preferences.revision, actor=ACTOR)


def add_key(writer, name: str, role: str = "client") -> str:
    key = new_key()
    writer.add_key(name=name, role=role, key=key, actor=ACTOR)
    return key


def require_a_key(writer, *, deployment_key_present: bool = False) -> None:
    writer.refresh()
    writer.set_access(base_revision=writer.state.keys.revision, actor=ACTOR,
                      deployment_key_present=deployment_key_present, require_key=True)


def hand_edit(path, text: str) -> None:
    """Replace a file the way an editor does (a new file under the old name)."""
    temp = path.with_name(path.name + ".hand")
    temp.write_text(text, encoding="utf-8")
    os.replace(temp, path)


def gateway(monkeypatch, folder, env=None, handler=backend):
    """A gateway whose settings live in `folder`."""
    module = load_frontend_app({"TTS_STT_SETTINGS_DIR": str(folder), **(env or {})})
    stub = install_stub(monkeypatch, module, handler)
    return module, stub, TestClient(module.app)


def host_status(client, host: str) -> int:
    return client.get("/providers", headers={"Host": host}).status_code


# --- nothing saved: exactly the app YAML ----------------------------------------------------


YAML = {"API_KEY": YAML_KEY, "TRUSTED_HOSTS": "truenas.k2o", "ALLOWED_ORIGINS": "https://ui.example",
        "ENABLE_MAGPIE_TTS": "true", "MAX_TTS_CHARS": "100", "MAX_UPLOAD_MB": "64",
        "MAX_CONCURRENT_UPLOADS": "3", "DEFAULT_TTS_PROVIDER": "magpie"}


def test_without_saved_settings_the_gateway_runs_on_the_app_yaml(monkeypatch, folder):
    """An empty settings folder and no settings folder at all give the same gateway, and
    neither writes a thing."""
    unmounted = load_frontend_app(YAML)                     # /app/settings, not mounted
    empty, stub, client = gateway(monkeypatch, folder, YAML)

    for module in (unmounted, empty):
        assert module._policy == module._build_access_policy({**module._ENV_VALUES, "require_key": True})
        assert module._keyring.entries == ((hashlib.sha256(YAML_KEY.encode()).digest(),
                                            module._Credential(store_module.DEPLOYMENT_KEY, "admin")),)
        assert (module.MAX_UPLOAD_MB, module.MAX_REQUEST_BYTES, module.MAX_TTS_CHARS) == (64.0, 64 * 2**20, 100)
        assert module._upload_slots.limit == module.MAX_CONCURRENT_UPLOADS == 3
        assert module.ENABLE_MAGPIE_TTS is True and module.ENABLE_TRAINING is True
    assert unmounted.PROVIDER_REGISTRY == empty.PROVIDER_REGISTRY
    assert empty.PROVIDER_REGISTRY["ui"]["default_tts_provider"] == "magpie"
    # which values the YAML names, for the page's "from app YAML" / "default" badges
    assert {"TRUSTED_HOSTS", "ALLOWED_ORIGINS", "ENABLE_MAGPIE_TTS", "MAX_TTS_CHARS", "MAX_UPLOAD_MB",
            "MAX_CONCURRENT_UPLOADS", "DEFAULT_TTS_PROVIDER"} <= empty._ENV_SET
    assert not {"TRUST_PROXY_HEADERS", "TRUSTED_ORIGINS", "ENABLE_TRAINING"} & empty._ENV_SET
    assert router_module._ffmpeg_slots.limit == router_module.MAX_CONCURRENT_FFMPEG

    assert host_status(client, "truenas.k2o") == 200
    assert host_status(client, "speach.k2o") == 403
    assert client.get("/v1/models").status_code == 401
    assert client.get("/v1/models", headers=bearer(YAML_KEY)).status_code == 200
    r = client.get("/providers", headers={"Origin": "https://ui.example"})
    assert r.headers["access-control-allow-origin"] == "https://ui.example"
    assert client.post("/api/tts", json={"provider": "piper", "text": "a" * 101},
                       headers=bearer(YAML_KEY)).status_code == 422
    assert list(folder.iterdir()) == [], "the gateway wrote into the settings folder"


def test_a_request_only_stats_the_files_when_nothing_changed(monkeypatch, folder, writer):
    save(writer, TRUSTED_HOSTS=["speach.k2o"])
    add_key(writer, "Home Assistant")
    module, _, client = gateway(monkeypatch, folder)
    client.get("/providers")
    reads = module._settings_store.reads
    for _ in range(5):
        assert client.get("/providers", headers={"Host": "speach.k2o"}).status_code == 200
    assert module._settings_store.reads == reads, "an unchanged file was read again"


def test_saved_values_are_in_force_from_the_start_and_the_log_names_them(monkeypatch, folder, writer, caplog):
    save(writer, TRUSTED_HOSTS=["speach.k2o"], ENABLE_MAGPIE_TTS=True)
    with caplog.at_level(logging.INFO):
        module = load_frontend_app({"TTS_STT_SETTINGS_DIR": str(folder)})
    assert "speach.k2o" in module._policy.trusted_host_names
    assert module.ENABLE_MAGPIE_TTS is True and "magpie" in module.PROVIDER_REGISTRY["providers"]
    lines = [r.getMessage() for r in caplog.records if "gateway.json" in r.getMessage()]
    assert any("TRUSTED_HOSTS, ENABLE_MAGPIE_TTS" in line and "not from the app YAML" in line for line in lines), lines


# --- 13-15: a host name, saved once, works in every worker at once ------------------------------


def test_13_15_a_saved_host_name_works_from_the_next_request_in_every_worker(monkeypatch, folder, writer):
    first, _, one = gateway(monkeypatch, folder, {"TRUSTED_HOSTS": "truenas.k2o"})
    second, _, two = gateway(monkeypatch, folder, {"TRUSTED_HOSTS": "truenas.k2o"})
    for client in (one, two):
        assert host_status(client, "speach.k2o") == 403                      # 13
        assert host_status(client, "truenas.k2o") == 200

    # 14: saved from the NAS's IP address, which always passes the Host check
    save(writer, TRUSTED_HOSTS=["truenas.k2o", "speach.k2o"])

    for client in (one, two):                                                 # 15
        assert host_status(client, "speach.k2o:3000") == 200
        r = client.delete(DELETE_JOB, headers={"Host": "speach.k2o:3000", "Origin": "http://speach.k2o:3000"})
        assert r.status_code == 200
    assert first.PROVIDER_REGISTRY is not second.PROVIDER_REGISTRY           # two separate workers

    # The saved list replaces the YAML's: a name it leaves out is refused again.
    save(writer, TRUSTED_HOSTS=["speach.k2o"])
    for client in (one, two):
        assert host_status(client, "truenas.k2o") == 403
        assert host_status(client, "speach.k2o") == 200

    # Reset to YAML
    save(writer, TRUSTED_HOSTS=None)
    for client in (one, two):
        assert host_status(client, "truenas.k2o") == 200
        assert host_status(client, "speach.k2o") == 403


def test_deleting_gateway_json_returns_to_the_yaml_values_on_the_next_request(monkeypatch, folder, writer):
    save(writer, TRUSTED_HOSTS=["speach.k2o"], MAX_TTS_CHARS=10)
    module, _, client = gateway(monkeypatch, folder)
    assert host_status(client, "speach.k2o") == 200
    assert module.MAX_TTS_CHARS == 10

    (folder / "gateway.json").unlink()
    assert host_status(client, "speach.k2o") == 403
    assert module.MAX_TTS_CHARS == 5000


def test_a_wildcard_and_a_reverse_proxy_origin_are_trusted_live(monkeypatch, folder, writer):
    _, _, client = gateway(monkeypatch, folder)
    save(writer, TRUSTED_HOSTS=["*.k2o"], TRUSTED_ORIGINS=["https://tts.example.com"])
    assert host_status(client, "speach.k2o") == 200
    assert host_status(client, "k2o.evil.example") == 403
    # the origin's host is trusted, and its pages pass the Origin check behind a Host-rewriting proxy
    assert host_status(client, "tts.example.com") == 200
    r = client.delete(DELETE_JOB, headers={"Host": "frontend-service:3000", "Origin": "https://tts.example.com"})
    assert r.status_code == 200
    assert "access-control-allow-origin" not in r.headers


def test_turning_proxy_headers_on_makes_the_forwarded_host_the_one_checked(monkeypatch, folder, writer):
    _, _, client = gateway(monkeypatch, folder)
    headers = {"Host": "frontend-service:3000", "X-Forwarded-Host": "evil.example"}
    assert client.get("/providers", headers=headers).status_code == 200      # forged header ignored
    save(writer, TRUST_PROXY_HEADERS=True)
    assert client.get("/providers", headers=headers).status_code == 403
    save(writer, TRUST_PROXY_HEADERS=None)
    assert client.get("/providers", headers=headers).status_code == 200


def test_an_overdue_unconfirmed_proxy_change_is_undone_before_the_guard_runs(monkeypatch, folder, tmp_path):
    """The commit-confirm record, undone by whichever worker sees the deadline pass first."""
    now = {"t": 1_790_000_000.0}

    def clock():
        return now["t"]

    writer = store_module.SettingsStore(folder, verify_mount=False, lock_path=str(tmp_path / "lock"), clock=clock)
    first, _, one = gateway(monkeypatch, folder)
    second, _, two = gateway(monkeypatch, folder)
    for module in (first, second):
        monkeypatch.setattr(module, "_settings_store", store_module.SettingsStore(
            folder, verify_mount=False, lock_path=str(tmp_path / "lock"), clock=clock))

    result = save(writer, TRUSTED_ORIGINS=["https://tts.example.com"])
    assert result.pending is not None and result.pending.revision == result.revision
    for client in (one, two):
        assert host_status(client, "tts.example.com") == 200                  # live at once

    now["t"] += store_module.CONFIRM_WINDOW_S + 1
    assert host_status(two, "tts.example.com") == 403                         # undone, then checked
    assert host_status(one, "tts.example.com") == 403
    document = json.loads((folder / "gateway.json").read_text(encoding="utf-8"))
    assert document["values"] == {} and document["pending"] is None
    audit = (folder / "audit.jsonl").read_text(encoding="utf-8")
    assert '"event":"auto-reverted"' in audit


# --- 25: CORS origins, live --------------------------------------------------------------------


def test_25_an_explicit_cors_origin_is_added_and_removed_live(monkeypatch, folder, writer):
    origin = {"Origin": "https://ui.example"}
    workers = [gateway(monkeypatch, folder)[2], gateway(monkeypatch, folder)[2]]
    for client in workers:
        assert "access-control-allow-origin" not in client.get("/providers", headers=origin).headers
        assert client.delete(DELETE_JOB, headers=origin).status_code == 403

    save(writer, ALLOWED_ORIGINS=["https://ui.example"])
    for client in workers:
        assert client.get("/providers", headers=origin).headers["access-control-allow-origin"] == "https://ui.example"
        preflight = client.options(DELETE_JOB, headers={**origin, "Access-Control-Request-Method": "DELETE"})
        assert preflight.status_code == 200
        assert preflight.headers["access-control-allow-origin"] == "https://ui.example"
        assert client.delete(DELETE_JOB, headers=origin).status_code == 200
        assert client.delete(DELETE_JOB, headers={"Origin": "https://evil.example"}).status_code == 403
        # never on the Settings page and its API (no route yet: the 404 shows the path rule)
        assert "access-control-allow-origin" in client.get("/api/nothing-here", headers=origin).headers
        for path in ("/api/settings", "/api/settings/keys", "/settings"):
            assert "access-control-allow-origin" not in client.get(path, headers=origin).headers, path

    save(writer, ALLOWED_ORIGINS=None)
    for client in workers:
        assert "access-control-allow-origin" not in client.get("/providers", headers=origin).headers
        assert client.delete(DELETE_JOB, headers=origin).status_code == 403


def test_a_yaml_wildcard_origin_is_replaced_by_a_saved_list(monkeypatch, folder, writer):
    _, _, client = gateway(monkeypatch, folder, {"ALLOWED_ORIGINS": "*"})
    assert client.delete(DELETE_JOB, headers={"Origin": "https://evil.example"}).status_code == 200
    save(writer, ALLOWED_ORIGINS=["https://ui.example"])
    assert client.delete(DELETE_JOB, headers={"Origin": "https://evil.example"}).status_code == 403
    assert client.delete(DELETE_JOB, headers={"Origin": "https://ui.example"}).status_code == 200


# --- 45-47: limits ------------------------------------------------------------------------------


def _reference_422(limit: int, path: str, body: dict) -> bytes:
    """What FastAPI answers for pydantic's own Field(max_length=limit), as before the change."""
    from fastapi import FastAPI
    from pydantic import BaseModel, Field

    class Reference(BaseModel):
        text: str = Field(max_length=limit)

    async def endpoint(request):
        return {}

    # The class itself, not the string this module's `from __future__ import annotations` makes.
    endpoint.__annotations__ = {"request": Reference}
    reference = FastAPI()
    reference.post(path)(endpoint)
    response = TestClient(reference).post(path, json=body)
    assert response.status_code == 422
    return response.content


@pytest.mark.parametrize("limit", [50, 1])
def test_45_a_lower_saved_tts_limit_gives_the_byte_identical_422(monkeypatch, folder, writer, limit):
    from_yaml = TestClient(load_frontend_app({"MAX_TTS_CHARS": str(limit)}).app)
    _, stub, saved = gateway(monkeypatch, folder)
    save(writer, MAX_TTS_CHARS=limit)

    requests = [
        ("/api/tts", {"provider": "piper", "text": "a" * (limit + 1)}),
        ("/api/providers/qwen3/voice-design",
         {"text": "a" * (limit + 1), "voice_description": "warm", "lang": "German"}),
    ]
    for path, body in requests:
        expected = _reference_422(limit, path, body)
        for client in (saved, from_yaml):
            r = client.post(path, json=body)
            assert (r.status_code, r.content) == (422, expected), path
    noun = "character" if limit == 1 else "characters"
    assert f"at most {limit} {noun}".encode() in expected
    assert stub.calls == []
    assert saved.post("/api/tts", json={"provider": "piper", "text": "a" * limit}).status_code == 200


def test_45_a_raised_saved_tts_limit_lets_longer_texts_through(monkeypatch, folder, writer):
    _, _, client = gateway(monkeypatch, folder, {"MAX_TTS_CHARS": "10"})

    def max_lengths():
        schemas = client.get("/openapi.json").json()["components"]["schemas"]
        return [schemas[name]["properties"]["text"]["maxLength"]
                for name in ("FrontendTTSRequest", "ProviderVoiceDesignRequest")]

    assert max_lengths() == [10, 10]
    assert client.post("/api/tts", json={"provider": "piper", "text": "a" * 50}).status_code == 422
    save(writer, MAX_TTS_CHARS=100)
    assert client.post("/api/tts", json={"provider": "piper", "text": "a" * 50}).status_code == 200
    r = client.post("/api/tts", json={"provider": "piper", "text": "a" * 101})
    assert r.status_code == 422 and r.json()["detail"][0]["ctx"] == {"max_length": 100}
    assert max_lengths() == [100, 100], "the published schema kept the old limit"


def test_45_the_schema_keeps_its_field_order_and_the_yaml_limit(monkeypatch, folder):
    _, _, client = gateway(monkeypatch, folder)
    text = client.get("/openapi.json").json()["components"]["schemas"]["FrontendTTSRequest"]["properties"]["text"]
    assert list(text.items()) == [("type", "string"), ("maxLength", 5000), ("title", "Text")]


def test_46_the_upload_size_boundary_moves(monkeypatch, folder, writer):
    module, _, client = gateway(monkeypatch, folder)
    second, _, _ = gateway(monkeypatch, folder)

    def upload():
        return client.post("/api/stt", data={"provider": "whisper"},
                           files={"audio": ("a.wav", b"\0" * (2 * 2**20), "audio/wav")})

    assert upload().status_code == 200
    save(writer, MAX_UPLOAD_MB=1)
    r = upload()
    assert r.status_code == 413 and "Maximum is 1 MB" in r.json()["detail"]
    save(writer, MAX_UPLOAD_MB=2.5)
    assert upload().status_code == 200
    assert module.MAX_UPLOAD_MB == 2.5 and module.MAX_REQUEST_BYTES == int(2.5 * 2**20)
    TestClient(second.app).get("/providers")         # the other worker, on its next request
    assert second.MAX_REQUEST_BYTES == int(2.5 * 2**20)


def test_47_both_concurrency_limits_are_set_live(monkeypatch, folder, writer):
    module, _, client = gateway(monkeypatch, folder)
    save(writer, MAX_CONCURRENT_UPLOADS=2, MAX_CONCURRENT_FFMPEG=3)
    client.get("/providers")
    assert module._upload_slots.limit == module.MAX_CONCURRENT_UPLOADS == 2
    assert module.openai_router._ffmpeg_slots.limit == 3

    # Two uploads already running: the next one is refused with the new number...
    monkeypatch.setattr(module._upload_slots, "active", 2)
    r = client.post("/api/stt", data={"provider": "whisper"}, files={"audio": ("a.wav", b"RIFF", "audio/wav")})
    assert r.status_code == 503 and "2 uploads" in r.json()["detail"]
    # ...and admitted once the limit is raised.
    save(writer, MAX_CONCURRENT_UPLOADS=3)
    r = client.post("/api/stt", data={"provider": "whisper"}, files={"audio": ("a.wav", b"RIFF", "audio/wav")})
    assert r.status_code == 200

    # The same for ffmpeg: three conversions running, the fourth is busy until the limit rises.
    monkeypatch.setattr(module.openai_router._ffmpeg_slots, "active", 3)
    speech = {"model": "tts-1", "input": "Hallo", "voice": "alloy", "response_format": "mp3"}
    r = client.post("/v1/audio/speech", json=speech)
    assert r.status_code == 503 and r.json()["error"]["code"] == "server_busy"
    save(writer, MAX_CONCURRENT_FFMPEG=4)
    r = client.post("/v1/audio/speech", json=speech)
    assert r.status_code != 503       # 200 with ffmpeg installed, 501 without
    assert module.openai_router._ffmpeg_slots.active == 3

    save(writer, MAX_CONCURRENT_UPLOADS=None, MAX_CONCURRENT_FFMPEG=None)
    client.get("/providers")
    assert module._upload_slots.limit == 4
    assert module.openai_router._ffmpeg_slots.limit == router_module.MAX_CONCURRENT_FFMPEG


# --- 39, 40, 42, 43: engines ----------------------------------------------------------------------


def test_39_offering_magpie_makes_it_an_engine_everywhere(monkeypatch, folder, writer):
    module, stub, client = gateway(monkeypatch, folder)
    speech = {"model": "tts-1", "input": "Hallo", "voice": "alloy", "response_format": "wav"}
    assert "magpie" not in client.get("/providers").json()["providers"]

    save(writer, ENABLE_MAGPIE_TTS=True, DEFAULT_TTS_PROVIDER="magpie")
    registry = client.get("/providers").json()
    assert registry["providers"]["magpie"]["kind"] == "tts"
    assert registry["ui"]["enable_magpie_tts"] is True
    assert registry["ui"]["default_tts_provider"] == "magpie"
    page = client.get("/").text
    assert 'value="magpie" checked' in page
    assert 'id="service-status-magpie"' in page
    assert client.post("/api/tts", json={"provider": "magpie", "text": "Hallo"}).status_code == 200
    assert client.post("/v1/audio/speech", json=speech).status_code == 200
    urls = [url for method, url, _ in stub.calls if method == "POST"]
    assert urls == ["http://magpie-tts-service:5008/tts"] * 2, urls

    save(writer, ENABLE_MAGPIE_TTS=None, DEFAULT_TTS_PROVIDER=None)
    assert "magpie" not in client.get("/providers").json()["providers"]
    assert client.post("/api/tts", json={"provider": "magpie", "text": "Hallo"}).status_code == 404
    stub.calls.clear()
    assert client.post("/v1/audio/speech", json=speech).status_code == 200
    assert [url for _, url, _ in stub.calls] == ["http://piper-tts-service:5000/tts"]


def test_40_the_registry_object_stays_and_its_old_entries_are_not_changed(monkeypatch, folder, writer):
    import copy

    module, stub, client = gateway(monkeypatch, folder)
    registry = module.PROVIDER_REGISTRY
    piper = registry["providers"]["piper"]
    piper_before = copy.deepcopy(piper)

    save(writer, TRUSTED_HOSTS=["speach.k2o"])        # not an engine setting: no rebuild
    client.get("/providers")
    assert module.PROVIDER_REGISTRY["providers"]["piper"] is piper

    save(writer, ENABLE_WHISPER_CPP=True, DEFAULT_STT_PROVIDER="whisper-cpp")
    client.get("/providers")
    assert module.PROVIDER_REGISTRY is registry, "the /v1 router holds this object"
    assert "whisper-cpp" in registry["providers"]
    assert registry["providers"]["piper"] is not piper
    assert piper == piper_before, "an entry handed out earlier was changed"
    # ...and /v1 reads it per call: its default speech-to-text engine is the new one.
    stub.calls.clear()
    r = client.post("/v1/audio/transcriptions", data={"model": "whisper-1"},
                    files={"file": ("a.wav", wav_bytes(), "audio/wav")})
    assert r.status_code == 200, r.text
    assert [url for _, url, _ in stub.calls] == ["http://whisper-cpp:8080/inference"]


def test_42_without_training_there_is_no_route_and_no_status_row(monkeypatch, folder, writer):
    module, stub, client = gateway(monkeypatch, folder)
    assert client.get("/api/training/jobs").status_code == 200
    assert 'id="service-status-piper-training"' in client.get("/").text

    save(writer, ENABLE_TRAINING=False)
    stub.calls.clear()
    r = client.get("/api/training/jobs")
    assert r.status_code == 404 and "piper-training" in r.json()["detail"]
    assert client.delete(DELETE_JOB).status_code == 404
    assert stub.calls == []
    assert 'id="service-status-piper-training"' not in client.get("/").text
    registry = client.get("/providers").json()
    assert "piper-training" not in registry["providers"] and registry["ui"]["enable_training"] is False
    assert "piper-training" not in client.get("/health").json()["services"]
    assert module.ENABLE_TRAINING is False

    save(writer, ENABLE_TRAINING=None)
    assert client.get("/api/training/jobs").status_code == 200


def test_42_enable_training_false_in_the_yaml_hides_it_from_the_start(monkeypatch, folder):
    module, _, client = gateway(monkeypatch, folder, {"ENABLE_TRAINING": "false"})
    assert "piper-training" not in module.PROVIDER_REGISTRY["providers"]
    assert module.PROVIDER_REGISTRY["ui"]["enable_training"] is False
    assert client.get("/api/training/jobs").status_code == 404


def test_43_engines_from_provider_registry_json_cannot_be_toggled(monkeypatch, folder, writer):
    operator = {"providers": {"magpie": {
        "kind": "tts", "display_name": "My Magpie", "internal_url": "http://my-magpie:9000",
        "health_endpoint": "/health", "contracts": {"tts": "simple-json-tts-v1"},
        "ui": {"selectable_as_engine": True}}}}
    module, _, client = gateway(monkeypatch, folder, {"PROVIDER_REGISTRY_JSON": json.dumps(operator)})
    assert module._registry_override_providers() == {"magpie"}

    for offered in (False, True, None):
        save(writer, ENABLE_MAGPIE_TTS=offered)
        entry = client.get("/providers").json()["providers"]["magpie"]
        assert entry["internal_url"] == "http://my-magpie:9000", offered
        assert entry["display_name"] == "My Magpie"


def test_an_engine_change_resets_the_health_and_canary_caches(monkeypatch, folder, writer):
    module, stub, client = gateway(monkeypatch, folder, {"ENABLE_CANARY_ASR": "true"})

    def canary_asks():
        return sum(1 for _, url, _ in stub.calls if url.endswith("/status") and "canary" in url)

    client.get("/providers")
    client.get("/providers")
    assert canary_asks() == 1, "the canary language list is asked for once per interval"
    assert "magpie" not in client.get("/api/health").json()["providers"]

    save(writer, ENABLE_MAGPIE_TTS=True)
    assert "magpie" in client.get("/api/health").json()["providers"], "a stale health round was served"
    client.get("/providers")
    assert canary_asks() == 2, "the rebuilt canary entry kept the old interval"
    assert "de/en" in client.get("/providers").json()["providers"]["canary"]["display_name"]


# --- 37, 38: never fail open --------------------------------------------------------------------


def _saved_page_settings(writer):
    save(writer, TRUSTED_HOSTS=["speach.k2o"], MAX_TTS_CHARS=10)
    key = add_key(writer, "Home Assistant")
    require_a_key(writer)
    return key


def test_37_enable_settings_ui_false_ignores_gateway_json_and_keeps_the_keys(monkeypatch, folder, writer, caplog):
    key = _saved_page_settings(writer)
    with caplog.at_level(logging.WARNING):
        module, _, client = gateway(monkeypatch, folder, {"ENABLE_SETTINGS_UI": "false"})
    assert any("ENABLE_SETTINGS_UI=false" in r.getMessage() for r in caplog.records)
    assert host_status(client, "speach.k2o") == 403
    assert client.post("/api/tts", json={"provider": "piper", "text": "a" * 50}, headers=bearer(key)).status_code == 200
    assert client.get("/v1/models").status_code == 401
    assert client.get("/v1/models", headers=bearer(key)).status_code == 200


def test_37_a_safe_mode_file_ignores_gateway_json_and_keeps_the_keys(monkeypatch, folder, writer):
    key = _saved_page_settings(writer)
    _, _, client = gateway(monkeypatch, folder)
    tts = {"provider": "piper", "text": "a" * 50}
    assert host_status(client, "speach.k2o") == 200
    assert client.post("/api/tts", json=tts, headers=bearer(key)).status_code == 422

    (folder / "SAFE-MODE").write_text("", encoding="utf-8")
    assert host_status(client, "speach.k2o") == 403
    assert client.post("/api/tts", json=tts, headers=bearer(key)).status_code == 200
    assert client.get("/v1/models").status_code == 401
    assert client.get("/v1/models", headers=bearer(key)).status_code == 200

    (folder / "SAFE-MODE").unlink()
    assert host_status(client, "speach.k2o") == 200
    assert client.post("/api/tts", json=tts, headers=bearer(key)).status_code == 422


def test_38_a_damaged_keys_json_accepts_only_the_yaml_key(monkeypatch, folder, writer):
    page_key = add_key(writer, "Home Assistant", role="admin")
    module, stub, client = gateway(monkeypatch, folder, {"API_KEY": YAML_KEY})
    assert client.get("/v1/models", headers=bearer(page_key)).status_code == 200

    hand_edit(folder / "keys.json", '{"schema": 1, "revision": ')
    assert client.get("/v1/models", headers=bearer(page_key)).status_code == 401
    assert client.get("/v1/models").status_code == 401
    assert client.get("/v1/models", headers=bearer(YAML_KEY)).status_code == 200
    credential = module._bearer_credential(Headers({"authorization": f"Bearer {YAML_KEY}"}))
    assert (credential.name, credential.role) == (store_module.DEPLOYMENT_KEY, "client")

    # Deleting the file: the page keys are gone and the YAML key is admin again.
    (folder / "keys.json").unlink()
    assert client.get("/v1/models", headers=bearer(page_key)).status_code == 401
    assert client.get("/v1/models", headers=bearer(YAML_KEY)).status_code == 200
    assert module._bearer_credential(Headers({"authorization": f"Bearer {YAML_KEY}"})).role == "admin"


def test_38_a_damaged_keys_json_without_a_yaml_key_locks_the_api(monkeypatch, folder, writer):
    page_key = add_key(writer, "Home Assistant")
    _, stub, client = gateway(monkeypatch, folder)
    assert client.get("/v1/models").status_code == 200                       # open: no key required

    hand_edit(folder / "keys.json", "not json")
    for headers in ({}, bearer(page_key)):
        assert client.get("/v1/models", headers=headers).status_code == 401
        assert client.delete(DELETE_JOB, headers=headers).status_code == 401
    assert stub.calls == []
    assert client.get("/providers").status_code == 200                      # reads stay open


def test_locked_keys_keep_the_yaml_value_and_the_rest_applies(monkeypatch, folder, writer, caplog):
    save(writer, TRUSTED_HOSTS=["speach.k2o"], MAX_TTS_CHARS=10, MAX_UPLOAD_MB=1)
    with caplog.at_level(logging.WARNING):
        module, _, client = gateway(monkeypatch, folder, {
            "TRUSTED_HOSTS": "truenas.k2o", "SETTINGS_LOCKED_KEYS": "TRUSTED_HOSTS, max_tts_chars no_such_key"})
    assert any("no_such_key" in r.getMessage() for r in caplog.records)
    assert host_status(client, "truenas.k2o") == 200
    assert host_status(client, "speach.k2o") == 403
    assert client.post("/api/tts", json={"provider": "piper", "text": "a" * 50}).status_code == 200
    assert module.MAX_UPLOAD_MB == 1.0


def test_a_locked_require_key_follows_the_yaml(monkeypatch, folder, writer):
    key = add_key(writer, "Home Assistant")
    require_a_key(writer)
    _, _, client = gateway(monkeypatch, folder, {"SETTINGS_LOCKED_KEYS": "require_key"})
    assert client.get("/v1/models").status_code == 200       # no API_KEY in the YAML: open
    _, _, client = gateway(monkeypatch, folder)
    assert client.get("/v1/models").status_code == 401
    assert client.get("/v1/models", headers=bearer(key)).status_code == 200


def test_a_failure_while_applying_leaves_the_settings_in_force(monkeypatch, folder, writer, caplog):
    save(writer, TRUSTED_HOSTS=["speach.k2o"])
    module, _, client = gateway(monkeypatch, folder)

    def broken():
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(module._settings_store, "refresh", broken)
    with caplog.at_level(logging.ERROR):
        assert host_status(client, "speach.k2o") == 200
    assert any("could not apply the saved settings" in r.getMessage() for r in caplog.records)


# --- the keys made in the Settings page ------------------------------------------------------------


class _SilentUpstream:
    def __init__(self, seen, url):
        seen.append(url)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def send(self, message):
        pass

    def __aiter__(self):
        return self

    async def __anext__(self):
        raise StopAsyncIteration


def _ws_outcome(client, **kwargs):
    try:
        with client.websocket_connect("/ws/stt?provider=whisper", **kwargs) as ws:
            try:
                ws.receive_text()
            except WebSocketDisconnect as closed:
                if closed.code == 1008:
                    return ("closed", closed.code, closed.reason)
            return ("accepted",)
    except WebSocketDisconnect as closed:
        return ("closed", closed.code, closed.reason)


def test_page_keys_work_wherever_the_yaml_key_does_and_stop_when_revoked(monkeypatch, folder, writer, caplog):
    import websockets

    seen = []
    monkeypatch.setattr(websockets, "connect", lambda url, **kw: _SilentUpstream(seen, url))
    key = add_key(writer, "Home Assistant")
    caplog.set_level(logging.DEBUG)
    module, stub, client = gateway(monkeypatch, folder)
    assert client.get("/v1/models").status_code == 200                       # open

    require_a_key(writer)
    assert client.get("/v1/models").status_code == 401
    assert client.get("/v1/models").headers["www-authenticate"] == "Bearer"
    assert client.delete(DELETE_JOB).status_code == 401
    assert _ws_outcome(client) == ("closed", 1008, "A valid API key is required")
    assert seen == []

    assert client.get("/v1/models", headers=bearer(key)).status_code == 200
    assert client.delete(DELETE_JOB, headers=bearer(key)).status_code == 200
    protocol = "bearer." + base64.urlsafe_b64encode(key.encode()).decode().rstrip("=")
    assert _ws_outcome(client, subprotocols=["tts-stt.v1", protocol]) == ("accepted",)
    assert len(seen) == 1
    credential = module._bearer_credential(Headers({"authorization": f"Bearer {key}"}))
    assert (credential.name, credential.role) == ("Home Assistant", "client")

    record = writer.state.keys.keys[0]
    writer.revoke_key(record.id, actor=ACTOR, deployment_key_present=False)
    assert client.get("/v1/models", headers=bearer(key)).status_code == 401

    assert key not in caplog.text
    for path in folder.rglob("*"):
        if path.is_file():
            assert key not in path.read_text(encoding="utf-8", errors="replace"), path.name


def test_the_yaml_key_and_page_keys_are_told_apart(monkeypatch, folder, writer):
    admin = add_key(writer, "Laptop", role="admin")
    module, _, _ = gateway(monkeypatch, folder, {"API_KEY": YAML_KEY})
    credential = module._key_matches(admin)
    assert (credential.name, credential.role, credential.key_id) == ("Laptop", "admin", writer.state.keys.keys[0].id)
    assert module._key_matches(YAML_KEY).name == store_module.DEPLOYMENT_KEY
    assert module._key_matches("nope") is None
    assert module._key_matches(admin[:-1]) is None


def test_every_digest_is_compared_whichever_key_matches(monkeypatch, folder, writer):
    """No early exit: the time taken does not say which key, or whether one, matched."""
    keys = [add_key(writer, "Laptop", role="admin"), add_key(writer, "Home Assistant")]
    module, _, _ = gateway(monkeypatch, folder, {"API_KEY": YAML_KEY})
    calls = []
    real = module.hmac.compare_digest

    def spy(a, b):
        calls.append((len(a), len(b)))
        return real(a, b)

    monkeypatch.setattr(module.hmac, "compare_digest", spy)
    for presented in (YAML_KEY, *keys, "wrong"):
        calls.clear()
        module._key_matches(presented)
        assert calls == [(32, 32)] * 3, presented


# =================================================================================================
# Part 2: the Settings API
# =================================================================================================

NAS = "192.168.1.20:3000"          # the NAS by its IP address, which always passes the Host check
CODE = re.compile(r"\b[0-9A-Z]{5}(?:-[0-9A-Z]{5}){3}\b")


def page(host: str = NAS, key=None, **extra) -> dict:
    """The headers of a request the Settings page sends when it was opened as `host`."""
    headers = {"Host": host, "Origin": f"http://{host}"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    headers.update(extra)
    return headers


def api(monkeypatch, folder, env=None, handler=backend, missing=()):
    """A gateway for the API tests; the host names in `missing` do not resolve (not installed)."""
    module, stub, client = gateway(monkeypatch, folder, env, handler)

    async def resolver(host):
        return host not in missing

    monkeypatch.setattr(module, "_engine_resolver", resolver)
    return module, stub, client


def view_of(client, key=None, host=NAS) -> dict:
    r = client.get("/api/settings", headers=page(host, key))
    assert r.status_code == 200, r.text
    return r.json()


def field(view: dict, key: str) -> dict:
    return next(item for item in view["settings"] if item["key"] == key)


def put(client, key, changes=None, reset=None, *, host=NAS, base=None, **body):
    """PUT /api/settings the way the page does, based on the revision it last read."""
    payload = {"base_revision": view_of(client, key)["revision"] if base is None else base, **body}
    if changes is not None:
        payload["set"] = changes
    if reset is not None:
        payload["reset"] = reset
    return client.put("/api/settings", json=payload, headers=page(host, key))


def printed_code(caplog) -> str:
    codes = [code for record in caplog.records if record.name == "tts_stt.settings.claim"
             for code in CODE.findall(record.getMessage())]
    assert codes, "no one-time code in the container log"
    return codes[-1]


def claim(client, caplog, name="Laptop", **body) -> str:
    """Claim the server like its owner: a code printed to the log, then an admin key."""
    with caplog.at_level(logging.INFO):
        assert client.post("/api/settings/claim-code", json={}, headers=page()).status_code == 202
    key = new_key()
    r = client.post("/api/settings/claim", json={"code": printed_code(caplog), "name": name, "key": key, **body},
                    headers=page())
    assert r.status_code == 201, r.text
    return key


def create_key(client, admin, name, role="client") -> str:
    key = new_key()
    r = client.post("/api/settings/keys", json={"name": name, "role": role, "key": key}, headers=page(key=admin))
    assert r.status_code == 201, r.text
    return key


# --- the page and the view ---------------------------------------------------------------------


def test_the_settings_page_is_a_shell_under_a_strict_policy(monkeypatch, folder):
    _, stub, client = api(monkeypatch, folder, {"API_KEY": YAML_KEY})
    r = client.get("/settings", headers={"Accept": "text/html"})
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/html")
    assert r.headers["content-security-policy"] == (
        "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; "
        "frame-ancestors 'none'; base-uri 'none'; form-action 'none'")
    assert r.headers["x-frame-options"] == "DENY"
    assert r.headers["cache-control"] == "no-store"
    assert "/static/js/settings.js" in r.text and "<script>" not in r.text and "style=" not in r.text
    assert client.get("/static/js/settings.js").status_code == 200
    assert stub.calls == []


def test_the_view_names_where_every_value_comes_from(monkeypatch, folder, writer):
    save(writer, MAX_UPLOAD_MB=64, TRUSTED_HOSTS=["speach.k2o"], MAX_TTS_CHARS=10)
    env = {"API_KEY": YAML_KEY, "TRUSTED_HOSTS": "truenas.k2o", "MAX_CONCURRENT_UPLOADS": "3",
           "SETTINGS_LOCKED_KEYS": "MAX_TTS_CHARS"}
    _, _, client = api(monkeypatch, folder, env)
    view = view_of(client, YAML_KEY)
    assert (view["revision"], view["claimed"], view["can_write"]) == (1, True, True)
    assert view["credential"] == {"name": "deployment key", "role": "admin", "id": None, "deployment_key": True}
    hosts = field(view, "TRUSTED_HOSTS")
    assert (hosts["value"], hosts["source"], hosts["yaml_value"], hosts["yaml_set"]) == (
        ["speach.k2o"], "saved", ["truenas.k2o"], True)
    assert hosts["label"] == "Host names of this server" and hosts["max_items"] == 64
    assert (field(view, "MAX_CONCURRENT_UPLOADS")["source"], field(view, "MAX_CONCURRENT_UPLOADS")["value"]) == ("yaml", 3)
    assert (field(view, "MAX_CONCURRENT_FFMPEG")["source"], field(view, "MAX_CONCURRENT_FFMPEG")["value"]) == ("default", 4)
    chars = field(view, "MAX_TTS_CHARS")
    assert (chars["source"], chars["value"], chars["saved_value"], chars["ignored"], chars["locked"]) == (
        "locked", 5000, 10, True, True)
    assert chars["read_only"] == "Locked by SETTINGS_LOCKED_KEYS in the app YAML."
    require = field(view, "require_key")
    assert (require["value"], require["source"], require["read_only"]) == (
        True, "yaml", "The app YAML sets API_KEY, so a key is always required.")
    assert [group["id"] for group in view["groups"]] == ["access", "api_access", "engines", "limits"]
    assert [entry["revision"] for entry in view["history"]] == [1] and view["history"][0]["event"] == "save"
    assert view["saved_by"] == {"ip": "192.168.1.20", "host": "192.168.1.20:3000", "credential": "deployment key"}
    assert "stored" in [banner["id"] for banner in view["banners"]]
    assert view["mount"]["state"] == "override" and view["mount"]["writable"] is True
    assert view["deployment"]["locked_keys"] == ["MAX_TTS_CHARS"] and view["deployment"]["api_key_set"] is True


# --- 16-19: the lockout self-check ---------------------------------------------------------------


def test_16_17_removing_the_host_in_use_is_refused_and_works_from_the_ip_address(monkeypatch, folder):
    _, _, client = api(monkeypatch, folder, {"API_KEY": YAML_KEY, "TRUSTED_HOSTS": "truenas.k2o,speach.k2o"})
    r = put(client, YAML_KEY, {"TRUSTED_HOSTS": ["truenas.k2o"]}, host="speach.k2o:3000")
    assert r.status_code == 409 and r.json()["code"] == "would_lock_out"
    assert r.json()["detail"] == (
        "You are connected as 'speach.k2o'; this change would refuse that name. Keep it, or make the change "
        "from the server's IP address (http://<IP address>:3000/settings).")
    assert not (folder / "gateway.json").exists(), "a refused change was written"
    assert host_status(client, "speach.k2o") == 200
    assert '"result":"409 would_lock_out"' in (folder / "audit.jsonl").read_text(encoding="utf-8")

    r = put(client, YAML_KEY, {"TRUSTED_HOSTS": ["truenas.k2o"]}, host=NAS)                # 17
    assert r.status_code == 200 and r.json()["changed"] == ["TRUSTED_HOSTS"] and r.json()["written"] is True
    assert host_status(client, "speach.k2o") == 403
    assert host_status(client, "truenas.k2o") == 200


def test_18_turning_proxy_headers_off_from_behind_the_proxy_is_refused(monkeypatch, folder):
    env = {"API_KEY": YAML_KEY, "TRUST_PROXY_HEADERS": "true", "TRUSTED_HOSTS": "tts.example.com"}
    _, _, client = api(monkeypatch, folder, env)
    proxied = {"Host": "frontend-service:3000", "X-Forwarded-Host": "tts.example.com",
               "Origin": "https://tts.example.com", **bearer(YAML_KEY)}
    change = {"base_revision": 0, "set": {"TRUST_PROXY_HEADERS": False}}
    r = client.put("/api/settings", json=change, headers=proxied)
    assert r.status_code == 409 and r.json()["code"] == "would_lock_out"
    assert "X-Forwarded-Host" in r.json()["detail"]
    assert not (folder / "gateway.json").exists()

    # From the IP address it goes through, and the page has to confirm it.
    r = put(client, YAML_KEY, {"TRUST_PROXY_HEADERS": False})
    assert r.status_code == 200 and r.json()["confirm"]["keys"] == ["TRUST_PROXY_HEADERS"]


def test_19_discard_and_restore_are_self_checked_too(monkeypatch, folder):
    _, _, client = api(monkeypatch, folder, {"API_KEY": YAML_KEY})
    first = put(client, YAML_KEY, {"MAX_TTS_CHARS": 1000})                 # a version without speach.k2o
    assert first.status_code == 200
    assert put(client, YAML_KEY, {"TRUSTED_HOSTS": ["speach.k2o"]}).status_code == 200
    view = view_of(client, YAML_KEY, host="speach.k2o:3000")
    old = next(entry["id"] for entry in view["history"] if entry["revision"] == first.json()["revision"])

    for path, body in (("/api/settings/discard", {}), ("/api/settings/restore", {"history_id": old})):
        r = client.post(path, json={"base_revision": view["revision"], **body},
                        headers=page("speach.k2o:3000", YAML_KEY))
        assert r.status_code == 409 and r.json()["code"] == "would_lock_out", path
    assert field(view_of(client, YAML_KEY), "TRUSTED_HOSTS")["value"] == ["speach.k2o"]

    r = client.post("/api/settings/restore", json={"base_revision": view["revision"], "history_id": old},
                    headers=page(key=YAML_KEY))
    assert r.status_code == 200 and r.json()["changed"] == ["TRUSTED_HOSTS"]
    assert host_status(client, "speach.k2o") == 403
    r = client.post("/api/settings/discard", json={"base_revision": r.json()["revision"]}, headers=page(key=YAML_KEY))
    assert r.status_code == 200 and r.json()["changed"] == ["MAX_TTS_CHARS"]
    view = view_of(client, YAML_KEY)
    assert [entry["event"] for entry in view["history"][:3]] == ["discard", "restore", "save"]
    assert field(view, "MAX_TTS_CHARS")["source"] == "default"


# --- 20-22: commit-confirm ----------------------------------------------------------------------


def test_20_a_proxy_change_is_live_at_once_and_a_confirm_keeps_it(monkeypatch, folder):
    now = {"t": 1_790_000_000.0}
    module, _, client = api(monkeypatch, folder, {"API_KEY": YAML_KEY})
    monkeypatch.setattr(module, "_settings_clock", lambda: now["t"])
    r = put(client, YAML_KEY, {"TRUSTED_ORIGINS": ["https://tts.example.com/"]})
    assert r.status_code == 200
    confirm = r.json()["confirm"]
    assert confirm["keys"] == ["TRUSTED_ORIGINS"] and confirm["deadline"] == now["t"] + 60
    assert confirm["revision"] == r.json()["revision"] and confirm["window"] == 60
    assert host_status(client, "tts.example.com") == 200                               # live at once
    assert view_of(client, YAML_KEY)["pending"]["revision"] == confirm["revision"]

    # The page confirms through the new rules: here from behind the Host-rewriting proxy.
    proxied = {"Host": "frontend-service:3000", "Origin": "https://tts.example.com", **bearer(YAML_KEY)}
    r = client.post("/api/settings/confirm", json={"revision": confirm["revision"]}, headers=proxied)
    assert r.status_code == 200 and r.json() == {"confirmed": True, "revision": confirm["revision"]}
    assert view_of(client, YAML_KEY)["pending"] is None
    now["t"] += 3600
    assert host_status(client, "tts.example.com") == 200                               # kept
    r = client.post("/api/settings/confirm", json={"revision": confirm["revision"]}, headers=page(key=YAML_KEY))
    assert r.status_code == 409 and r.json()["code"] == "nothing_to_confirm"
    assert '"event":"confirmed"' in (folder / "audit.jsonl").read_text(encoding="utf-8")


def test_a_confirm_of_another_revision_keeps_the_change_pending(monkeypatch, folder):
    _, _, client = api(monkeypatch, folder, {"API_KEY": YAML_KEY})
    revision = put(client, YAML_KEY, {"TRUSTED_ORIGINS": ["https://tts.example.com"]}).json()["revision"]
    later = put(client, YAML_KEY, {"MAX_TTS_CHARS": 100})              # carries the pending record over
    assert later.json()["confirm"] is None
    r = client.post("/api/settings/confirm", json={"revision": revision + 1}, headers=page(key=YAML_KEY))
    assert r.status_code == 409 and r.json()["code"] == "nothing_to_confirm"
    assert view_of(client, YAML_KEY)["pending"]["revision"] == revision
    r = client.post("/api/settings/confirm", json={"revision": revision}, headers=page(key=YAML_KEY))
    assert r.status_code == 200 and view_of(client, YAML_KEY)["pending"] is None


def test_21_without_a_confirm_the_change_is_undone_in_every_worker(monkeypatch, folder):
    now = {"t": 1_790_000_000.0}
    first, _, one = api(monkeypatch, folder, {"API_KEY": YAML_KEY})
    second, _, two = api(monkeypatch, folder, {"API_KEY": YAML_KEY})
    for module in (first, second):
        monkeypatch.setattr(module, "_settings_clock", lambda: now["t"])

    r = put(one, YAML_KEY, {"TRUST_PROXY_HEADERS": True}, acknowledge=["TRUST_PROXY_HEADERS:proxy_headers_on"])
    assert r.status_code == 200 and r.json()["confirm"] is not None
    forged = {"Host": "frontend-service:3000", "X-Forwarded-Host": "evil.example"}
    for client in (one, two):
        assert client.get("/providers", headers=forged).status_code == 403             # in force in both

    now["t"] += 61
    assert two.get("/providers", headers=forged).status_code == 200                    # undone, then checked
    assert one.get("/providers", headers=forged).status_code == 200
    assert '"event":"auto-reverted"' in (folder / "audit.jsonl").read_text(encoding="utf-8")
    view = view_of(one, YAML_KEY)
    assert view["pending"] is None and field(view, "TRUST_PROXY_HEADERS")["value"] is False
    assert view["history"][0]["event"] == "auto-revert"
    r = one.post("/api/settings/confirm", json={"revision": 1}, headers=page(key=YAML_KEY))
    assert r.status_code == 409 and r.json()["code"] == "nothing_to_confirm"


def test_22_an_overdue_confirmation_found_at_start_is_undone_before_the_first_request(monkeypatch, folder, tmp_path):
    earlier = store_module.SettingsStore(folder, verify_mount=False, lock_path=str(tmp_path / "lock"),
                                         clock=lambda: time.time() - 3600)
    earlier.refresh()
    earlier.update_preferences({"TRUSTED_ORIGINS": ["https://tts.example.com"], "MAX_TTS_CHARS": 100},
                               base_revision=0, actor=ACTOR)
    module, _, _ = api(monkeypatch, folder)
    document = json.loads((folder / "gateway.json").read_text(encoding="utf-8"))
    assert document["pending"] is None and document["values"] == {"MAX_TTS_CHARS": 100}
    assert "tts.example.com" not in module._policy.trusted_host_names
    assert module.MAX_TTS_CHARS == 100
    assert '"event":"auto-reverted"' in (folder / "audit.jsonl").read_text(encoding="utf-8")


# --- 23, 24: validation ----------------------------------------------------------------------------


@pytest.mark.parametrize("key,entry", [
    ("TRUSTED_HOSTS", "*.de"), ("TRUSTED_HOSTS", "*.co.uk"), ("TRUSTED_HOSTS", "*.ts.net"),
    ("TRUSTED_HOSTS", "*"), ("TRUSTED_HOSTS", "not a host!"),
    ("TRUSTED_ORIGINS", "*"), ("ALLOWED_ORIGINS", "*"), ("ALLOWED_ORIGINS", "null"),
])
def test_23_an_unsafe_or_invalid_entry_is_refused_on_its_field(monkeypatch, folder, key, entry):
    _, _, client = api(monkeypatch, folder, {"API_KEY": YAML_KEY})
    r = put(client, YAML_KEY, {key: ["speach.k2o" if key == "TRUSTED_HOSTS" else "https://ui.example", entry]})
    assert r.status_code == 400 and r.json()["code"] == "invalid_input"
    assert list(r.json()["errors"]) == [key]
    assert not (folder / "gateway.json").exists()


def test_23_other_invalid_values_are_refused_per_field(monkeypatch, folder):
    _, _, client = api(monkeypatch, folder, {"API_KEY": YAML_KEY})
    r = put(client, YAML_KEY, {"MAX_TTS_CHARS": 0, "MAX_UPLOAD_MB": "big", "ENABLE_MAGPIE_TTS": "yes",
                               "require_key": True, "NO_SUCH": 1}, reset=["MAX_UPLOAD_MB", "API_KEY"])
    assert r.status_code == 400
    assert set(r.json()["errors"]) == {"MAX_TTS_CHARS", "MAX_UPLOAD_MB", "ENABLE_MAGPIE_TTS", "require_key",
                                       "NO_SUCH", "API_KEY"}


def test_24_a_single_label_wildcard_needs_an_acknowledgement(monkeypatch, folder):
    _, _, client = api(monkeypatch, folder, {"API_KEY": YAML_KEY})
    r = put(client, YAML_KEY, {"TRUSTED_HOSTS": ["*.k2o"]})
    assert r.status_code == 409 and r.json()["code"] == "needs_confirmation"
    assert [warning["token"] for warning in r.json()["warnings"]] == ["TRUSTED_HOSTS:single_label_wildcard"]
    assert host_status(client, "speach.k2o") == 403

    r = put(client, YAML_KEY, {"TRUSTED_HOSTS": ["*.k2o"]}, dry_run=True)       # the review dialog
    assert r.status_code == 200 and (r.json()["dry_run"], r.json()["written"]) == (True, False)
    assert r.json()["warnings"][0]["id"] == "single_label_wildcard" and r.json()["changed"] == ["TRUSTED_HOSTS"]
    assert not (folder / "gateway.json").exists()

    r = put(client, YAML_KEY, {"TRUSTED_HOSTS": ["*.k2o"]}, acknowledge=["TRUSTED_HOSTS:single_label_wildcard"])
    assert r.status_code == 200 and r.json()["written"] is True
    assert host_status(client, "speach.k2o") == 200
    # a bare warning id is no acknowledgement
    r = put(client, YAML_KEY, {"MAX_TTS_CHARS": 6000}, acknowledge=["tts_chars_above_backend_limit"])
    assert r.status_code == 409 and r.json()["warnings"][0]["token"] == "MAX_TTS_CHARS:tts_chars_above_backend_limit"


def test_a_name_that_always_works_is_saved_with_a_note(monkeypatch, folder):
    _, _, client = api(monkeypatch, folder, {"API_KEY": YAML_KEY})
    r = put(client, YAML_KEY, {"TRUSTED_HOSTS": ["truenas", "speach.k2o"]})
    assert r.status_code == 200 and r.json()["warnings"] == []
    assert "not needed" in r.json()["notes"]["TRUSTED_HOSTS"][0]


# --- 26, 27: never any CORS --------------------------------------------------------------------------


def test_26_27_the_settings_never_answer_with_cors_headers(monkeypatch, folder):
    _, _, client = api(monkeypatch, folder, {"API_KEY": YAML_KEY, "ALLOWED_ORIGINS": "https://ui.example"})
    origin = {"Origin": "https://ui.example"}
    assert client.get("/providers", headers=origin).headers["access-control-allow-origin"] == "https://ui.example"
    answers = [
        client.get("/api/settings", headers=origin),
        client.get("/api/settings", headers={**origin, **bearer(YAML_KEY)}),
        client.get("/settings", headers=origin),
        client.put("/api/settings", json={"base_revision": 0}, headers={**origin, **bearer(YAML_KEY)}),
        client.options("/api/settings", headers={**origin, "Access-Control-Request-Method": "PUT"}),       # 27
        client.options("/api/settings/keys", headers={
            **origin, "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "authorization,content-type"}),
    ]
    assert [r.status_code for r in answers] == [401, 200, 200, 403, 403, 403]
    for r in answers:
        assert not [name for name in r.headers if name.lower().startswith("access-control-")], r.request.url
    assert answers[4].json()["code"] == "cors_refused"


def test_26_a_cors_header_from_further_in_is_stripped_from_a_settings_answer(monkeypatch, folder):
    """No layer adds one today; the outermost middleware removes any that a future one would."""
    module, _, _ = api(monkeypatch, folder)
    sent = []

    async def send(message):
        sent.append(message)

    start = {"type": "http.response.start", "status": 200, "headers": [
        (b"content-type", b"application/json"), (b"access-control-allow-origin", b"https://ui.example"),
        (b"Access-Control-Allow-Credentials", b"true"), (b"cache-control", b"max-age=60")]}
    asyncio.run(module._settings_send(send)(start))
    assert sent[0]["headers"] == [(b"content-type", b"application/json"), (b"cache-control", b"no-store")]


# --- 28-36: who may change settings -------------------------------------------------------------------


def test_28_unclaimed_the_settings_can_be_read_but_not_changed(monkeypatch, folder):
    _, _, client = api(monkeypatch, folder)
    view = view_of(client)
    assert (view["claimed"], view["can_write"], view["credential"], view["can_claim"]) == (False, False, None, True)
    assert {"unclaimed", "open_api", "yaml"} <= {banner["id"] for banner in view["banners"]}
    writes = (
        ("PUT", "/api/settings", {"base_revision": 0, "set": {"TRUSTED_HOSTS": ["speach.k2o"]}}),
        ("POST", "/api/settings/discard", {"base_revision": 0}),
        ("POST", "/api/settings/keys", {"name": "Laptop", "role": "admin", "key": new_key()}),
        ("PUT", "/api/settings/access", {"base_revision": 0, "require_key": True}),
    )
    for method, path, body in writes:
        r = client.request(method, path, json=body, headers=page())
        assert r.status_code == 403 and r.json()["code"] == "not_claimed", path
    assert sorted(path.name for path in folder.iterdir()) == ["audit.jsonl"]
    assert client.get("/api/settings/engines", headers=page()).status_code == 200


def test_29_a_claim_with_the_code_from_the_log_creates_an_admin_key(monkeypatch, folder, caplog):
    _, _, client = api(monkeypatch, folder)
    with caplog.at_level(logging.INFO):
        r = client.post("/api/settings/claim-code", json={}, headers=page())
    assert r.status_code == 202 and "log" in r.json()["detail"] and r.json()["expires_in"] == 1800
    code = printed_code(caplog)
    assert code not in r.text

    key = new_key()
    wrong = {"code": "AAAAA-AAAAA-AAAAA-AAAAA", "name": "Laptop", "key": key}
    r = client.post("/api/settings/claim", json=wrong, headers=page())
    assert r.status_code == 403 and r.json()["code"] == "invalid_code"
    r = client.post("/api/settings/claim", json={**wrong, "code": code.lower()}, headers=page())
    assert r.status_code == 201
    record = r.json()
    assert (record["name"], record["role"], record["hint"]) == ("Laptop", "admin", key[:8])
    assert set(record) == {"id", "name", "role", "hint", "created_at", "keys_revision"}

    view = view_of(client, key)
    assert view["claimed"] and view["can_write"] and view["can_manage_keys"]
    assert view["credential"] == {"name": "Laptop", "role": "admin", "id": record["id"], "deployment_key": False}
    assert client.get("/api/settings", headers=page()).status_code == 401              # claimed now
    r = client.post("/api/settings/claim", json={**wrong, "code": code, "name": "Other", "key": new_key()},
                    headers=page())
    assert r.status_code == 403, "a code works once"

    stored = "".join(path.read_text(encoding="utf-8", errors="replace") for path in folder.rglob("*") if path.is_file())
    assert code not in stored and code.replace("-", "") not in stored
    audit = (folder / "audit.jsonl").read_text(encoding="utf-8")
    assert '"event":"claim"' in audit and '"credential":"one-time code"' in audit


def test_a_claim_can_require_a_key_and_recovers_a_damaged_keys_file(monkeypatch, folder, caplog):
    module, _, client = api(monkeypatch, folder)
    clock = {"t": time.time()}
    monkeypatch.setattr(module, "_settings_clock", lambda: clock["t"])
    first = claim(client, caplog, require_key=True)
    assert client.get("/v1/models").status_code == 401
    assert client.get("/v1/models", headers=bearer(first)).status_code == 200

    hand_edit(folder / "keys.json", "{damaged")
    view = view_of(client)                                   # fail closed: unclaimed, read-only, key required
    assert view["claimed"] is False and view["access"]["fail_closed"] is True
    assert "keys_damaged" in {banner["id"] for banner in view["banners"]}
    assert client.get("/v1/models", headers=bearer(first)).status_code == 401
    clock["t"] += 61                                         # one code per minute
    second = claim(client, caplog, name="Recovered")
    assert view_of(client, second)["credential"]["name"] == "Recovered"
    assert (folder / "keys.json.damaged").exists()


def test_30_a_client_key_may_use_the_api_but_not_the_settings(monkeypatch, folder, caplog):
    _, _, client = api(monkeypatch, folder)
    admin = claim(client, caplog, require_key=True)
    home_assistant = create_key(client, admin, "Home Assistant")
    r = client.get("/api/settings", headers=page(key=home_assistant))
    assert r.status_code == 403 and r.json()["code"] == "admin_key_required"
    assert client.get("/v1/models", headers=bearer(home_assistant)).status_code == 200
    assert client.get("/v1/models").status_code == 401
    keys = view_of(client, admin)["keys"]
    assert [(k["name"], k["role"]) for k in keys] == [("Laptop", "admin"), ("Home Assistant", "client")]
    assert [set(k) for k in keys] == [{"id", "name", "role", "hint", "created_at"}] * 2      # never a digest


def test_a_write_is_in_force_in_the_answering_worker_before_it_answers(monkeypatch, folder):
    module, _, client = api(monkeypatch, folder, {"API_KEY": YAML_KEY})
    revision = view_of(client, YAML_KEY)["revision"]
    r = client.put("/api/settings", json={"base_revision": revision, "set": {"MAX_TTS_CHARS": 100}},
                   headers=page(key=YAML_KEY))
    assert r.status_code == 200 and module.MAX_TTS_CHARS == 100            # no request in between
    key = new_key()
    r = client.post("/api/settings/keys", json={"name": "Phone", "role": "client", "key": key},
                    headers=page(key=YAML_KEY))
    assert r.status_code == 201 and module._key_matches(key).name == "Phone"


def test_a_damaged_keys_file_with_a_yaml_key_is_recovered_by_a_claim(monkeypatch, folder, caplog):
    _, _, client = api(monkeypatch, folder, {"API_KEY": YAML_KEY})
    create_key(client, YAML_KEY, "Laptop", role="admin")
    hand_edit(folder / "keys.json", "[]")
    view = view_of(client)                          # the YAML key is only a client now: unclaimed, readable
    assert view["claimed"] is False and view["access"]["deployment_key_role"] == "client"
    r = client.get("/api/settings", headers=page(key=YAML_KEY))
    assert r.status_code == 200 and r.json()["credential"]["role"] == "client"
    r = client.put("/api/settings", json={"base_revision": 0, "set": {"MAX_TTS_CHARS": 100}}, headers=page(key=YAML_KEY))
    assert r.status_code == 403 and r.json()["code"] == "not_claimed"
    recovered = claim(client, caplog, name="Recovered")
    assert view_of(client, recovered)["claimed"] is True
    assert client.get("/api/settings", headers=page(key=YAML_KEY)).json()["code"] == "admin_key_required"


def test_31_the_yaml_key_is_admin_until_a_page_key_makes_it_a_client(monkeypatch, folder):
    _, _, client = api(monkeypatch, folder, {"API_KEY": YAML_KEY})
    view = view_of(client, YAML_KEY)
    assert view["claimed"] and view["credential"]["deployment_key"] and view["can_write"]
    change = {"base_revision": view["keys_revision"], "deployment_key_role": "client"}
    r = client.put("/api/settings/access", json=change, headers=page(key=YAML_KEY))
    assert r.status_code == 403 and r.json()["code"] == "deployment_key_not_allowed"

    admin = create_key(client, YAML_KEY, "Laptop", role="admin")
    change["base_revision"] = view_of(client, admin)["keys_revision"]
    r = client.put("/api/settings/access", json=change, headers=page(key=admin))
    assert r.status_code == 200 and r.json()["access"] == {"require_key": True, "deployment_key_role": "client"}
    r = client.get("/api/settings", headers=page(key=YAML_KEY))
    assert r.status_code == 403 and r.json()["code"] == "admin_key_required"
    assert client.get("/v1/models", headers=bearer(YAML_KEY)).status_code == 200      # still an API key
    # the YAML key forces "require a key"
    r = client.put("/api/settings/access", json={"base_revision": view_of(client, admin)["keys_revision"],
                                                 "require_key": False}, headers=page(key=admin))
    assert r.status_code == 403 and r.json()["code"] == "locked_by_deployment"
    assert r.json()["errors"] == {"require_key": "The app YAML sets API_KEY, so a key is always required."}


def test_32_requiring_a_key_closes_v1_the_changing_api_and_the_live_microphone(monkeypatch, folder, caplog):
    import websockets

    seen = []
    monkeypatch.setattr(websockets, "connect", lambda url, **kw: _SilentUpstream(seen, url))
    _, _, client = api(monkeypatch, folder)
    admin = claim(client, caplog)
    assert client.get("/v1/models").status_code == 200
    r = client.put("/api/settings/access", json={"base_revision": view_of(client, admin)["keys_revision"],
                                                 "require_key": True}, headers=page(key=admin))
    assert r.status_code == 200 and r.json()["access"]["require_key"] is True
    assert client.get("/v1/models").status_code == 401
    assert client.delete(DELETE_JOB).status_code == 401
    assert _ws_outcome(client) == ("closed", 1008, "A valid API key is required")
    assert seen == []
    protocol = "bearer." + base64.urlsafe_b64encode(admin.encode()).decode().rstrip("=")
    assert _ws_outcome(client, subprotocols=["tts-stt.v1", protocol]) == ("accepted",)
    assert client.get("/v1/models", headers=bearer(admin)).status_code == 200
    assert "open_api" not in {banner["id"] for banner in view_of(client, admin)["banners"]}


def test_33_the_last_admin_credential_cannot_be_revoked(monkeypatch, folder, caplog):
    _, _, client = api(monkeypatch, folder)
    admin = claim(client, caplog)
    own = view_of(client, admin)["credential"]["id"]
    r = client.delete(f"/api/settings/keys/{own}", headers=page(key=admin))
    assert r.status_code == 409 and r.json()["code"] == "last_admin_credential"

    second = create_key(client, admin, "Phone", role="admin")
    r = client.delete(f"/api/settings/keys/{own}", headers=page(key=second))
    assert r.status_code == 200 and r.json()["revoked"]["name"] == "Laptop"
    assert client.get("/api/settings", headers=page(key=admin)).status_code == 401      # at once
    r = client.delete("/api/settings/keys/00000000", headers=page(key=second))
    assert r.status_code == 404 and r.json()["code"] == "key_not_found"


def test_34_no_answer_log_or_file_holds_a_key(monkeypatch, folder, caplog):
    caplog.set_level(logging.DEBUG)
    _, _, client = api(monkeypatch, folder, {"API_KEY": YAML_KEY})
    laptop, home_assistant = new_key(), new_key()
    wrong = laptop[:-2] + ("AA" if not laptop.endswith("AA") else "BB")
    answers = []

    def call(method, path, key=YAML_KEY, **kwargs):
        r = client.request(method, path, headers=page(key=key), **kwargs)
        answers.append(r.text + json.dumps(dict(r.headers)))
        return r.status_code

    assert call("POST", "/api/settings/keys", json={"name": "Laptop", "role": "admin", "key": laptop}) == 201
    assert call("POST", "/api/settings/keys", json={"name": "Home Assistant", "role": "client",
                                                    "key": home_assistant}) == 201
    assert call("POST", "/api/settings/keys", json={"name": "Again", "role": "client", "key": laptop}) == 400
    assert call("GET", "/api/settings", key=laptop) == 200
    assert call("GET", "/api/settings", key=home_assistant) == 403
    assert call("GET", "/api/settings", key=wrong) == 401
    assert call("PUT", "/api/settings", key=laptop, json={"base_revision": 0, "set": {"MAX_TTS_CHARS": 100}}) == 200
    ha_id = next(k["id"] for k in view_of(client, laptop)["keys"] if k["name"] == "Home Assistant")
    assert call("DELETE", f"/api/settings/keys/{ha_id}", key=laptop) == 200
    assert call("GET", "/api/settings", key=laptop) == 200

    files = {path.name: path.read_text(encoding="utf-8", errors="replace") for path in folder.rglob("*") if path.is_file()}
    assert "keys.json" in files and "audit.jsonl" in files
    for secret in (YAML_KEY, laptop, home_assistant, wrong):
        for text in answers:
            assert secret not in text
        assert secret not in caplog.text
        for name, text in files.items():
            assert secret not in text, name
    assert laptop[:8] in files["keys.json"] and laptop[8:12] not in files["keys.json"]


def test_35_a_missing_or_wrong_key_gets_a_bearer_challenge(monkeypatch, folder):
    _, _, client = api(monkeypatch, folder, {"API_KEY": YAML_KEY})
    for headers in (page(), page(key="wrong")):
        r = client.get("/api/settings", headers=headers)
        assert r.status_code == 401 and r.headers["www-authenticate"] == "Bearer"
        assert r.json()["code"] == "invalid_api_key"
    assert client.get("/settings").status_code == 200                # the empty page needs no key


def test_36_ten_wrong_keys_or_codes_from_one_address_get_429(monkeypatch, folder):
    module, _, client = api(monkeypatch, folder, {"API_KEY": YAML_KEY})
    now = {"t": 1000.0}
    monkeypatch.setattr(module._settings_limiter, "clock", lambda: now["t"])
    for _ in range(12):                                  # no key at all is not a guess
        assert client.get("/api/settings", headers=page()).status_code == 401
    for attempt in range(10):
        assert client.get("/api/settings", headers=page(key=f"guess-{attempt}")).status_code == 401
    r = client.get("/api/settings", headers=page(key=YAML_KEY))          # even the right key, now
    assert r.status_code == 429 and r.json()["code"] == "too_many_attempts"
    assert 0 < int(r.headers["retry-after"]) <= 600
    assert client.get("/v1/models", headers=bearer(YAML_KEY)).status_code == 200     # only the Settings API
    assert '"result":"401 invalid_api_key, address blocked"' in (folder / "audit.jsonl").read_text(encoding="utf-8")
    now["t"] += 601
    assert client.get("/api/settings", headers=page(key=YAML_KEY)).status_code == 200

    # A key that works clears the count: nine typos, the right key, nine more typos.
    for round_ in range(2):
        for attempt in range(9):
            assert client.get("/api/settings", headers=page(key=f"typo-{round_}-{attempt}")).status_code == 401
        assert client.get("/api/settings", headers=page(key=YAML_KEY)).status_code == 200

    wrong = {"code": "AAAAA-AAAAA-AAAAA-AAAAA", "name": "Laptop", "key": new_key()}
    for _ in range(10):
        assert client.post("/api/settings/claim", json=wrong, headers=page()).status_code == 403
    assert client.post("/api/settings/claim", json=wrong, headers=page()).status_code == 429


# --- 41, 44: engines -------------------------------------------------------------------------------


def test_41_un_offering_the_default_engine_is_refused(monkeypatch, folder):
    env = {"API_KEY": YAML_KEY, "ENABLE_MAGPIE_TTS": "true", "DEFAULT_TTS_PROVIDER": "magpie"}
    _, _, client = api(monkeypatch, folder, env)
    r = put(client, YAML_KEY, {"ENABLE_MAGPIE_TTS": False})
    assert r.status_code == 400 and list(r.json()["errors"]) == ["DEFAULT_TTS_PROVIDER"]
    assert "'magpie' is not offered" in r.json()["errors"]["DEFAULT_TTS_PROVIDER"]
    r = put(client, YAML_KEY, {"DEFAULT_STT_PROVIDER": "piper"})
    assert r.status_code == 400 and "not a speech-to-text engine" in r.json()["errors"]["DEFAULT_STT_PROVIDER"]
    assert not (folder / "gateway.json").exists()

    r = put(client, YAML_KEY, {"ENABLE_MAGPIE_TTS": False, "DEFAULT_TTS_PROVIDER": "piper"})
    assert r.status_code == 200 and r.json()["reload_main_ui"] is True
    assert "magpie" not in client.get("/providers").json()["providers"]
    # a change that touches no engine is not held up by them
    assert put(client, YAML_KEY, {"MAX_TTS_CHARS": 100}).json()["reload_main_ui"] is False


def test_back_to_the_app_yaml_is_allowed_even_where_its_default_engine_is_not_offered(monkeypatch, folder):
    env = {"API_KEY": YAML_KEY, "DEFAULT_TTS_PROVIDER": "magpie"}       # the YAML forgot ENABLE_MAGPIE_TTS
    _, _, client = api(monkeypatch, folder, env)
    r = put(client, YAML_KEY, {"ENABLE_MAGPIE_TTS": True}, acknowledge=["ENABLE_MAGPIE_TTS:engine_not_reachable"])
    assert r.status_code == 200
    r = client.post("/api/settings/discard", json={"base_revision": r.json()["revision"]}, headers=page(key=YAML_KEY))
    assert r.status_code == 200 and r.json()["changed"] == ["ENABLE_MAGPIE_TTS"]


def test_engines_from_provider_registry_json_and_locked_keys_are_refused_as_fixed(monkeypatch, folder):
    operator = {"providers": {"magpie": {"kind": "tts", "display_name": "My Magpie",
                                         "internal_url": "http://my-magpie:9000", "health_endpoint": "/health"}}}
    env = {"API_KEY": YAML_KEY, "PROVIDER_REGISTRY_JSON": json.dumps(operator), "SETTINGS_LOCKED_KEYS": "MAX_UPLOAD_MB"}
    _, stub, client = api(monkeypatch, folder, env)
    r = put(client, YAML_KEY, {"ENABLE_MAGPIE_TTS": True, "MAX_UPLOAD_MB": 64}, reset=["MAX_TTS_CHARS"])
    assert r.status_code == 403 and r.json()["code"] == "locked_by_deployment"
    assert set(r.json()["errors"]) == {"ENABLE_MAGPIE_TTS", "MAX_UPLOAD_MB"}
    view = view_of(client, YAML_KEY)
    assert "PROVIDER_REGISTRY_JSON" in field(view, "ENABLE_MAGPIE_TTS")["read_only"]
    assert view["engines"]["registry_json"] == ["magpie"]
    # the operator's entry may be the default even with its flag off
    assert put(client, YAML_KEY, {"DEFAULT_TTS_PROVIDER": "magpie"}).status_code == 200
    engines = client.get("/api/settings/engines", headers=page(key=YAML_KEY)).json()["engines"]
    magpie = next(engine for engine in engines if engine["provider"] == "magpie")
    assert magpie["registry_json"] is True and magpie["offered"] is True
    assert "http://my-magpie:9000/health" in [url for _, url, _ in stub.calls]


def test_44_the_engine_probe_tells_running_from_not_reachable_and_not_installed(monkeypatch, folder):
    def handler(method, url, kwargs):
        if "canary-asr-service" in url:
            raise httpx.ConnectError("connection refused")
        if "whisper-cpp" in url:
            return httpx.Response(503)
        return {"status": "healthy"}

    module, stub, client = api(monkeypatch, folder, {"API_KEY": YAML_KEY, "ENABLE_MAGPIE_TTS": "true"},
                               handler=handler, missing={"parakeet-asr-service", "piper-training-service"})
    now = {"t": 100.0}
    monkeypatch.setattr(module, "_engine_probe_clock", lambda: now["t"])
    r = client.get("/api/settings/engines", headers=page(key=YAML_KEY))
    assert r.status_code == 200 and r.json()["max_age"] == 30
    engines = {engine["provider"]: engine for engine in r.json()["engines"]}
    assert {provider: engine["state"] for provider, engine in engines.items()} == {
        "canary": "not_reachable", "parakeet": "not_installed", "chatterbox": "running",
        "magpie": "running", "whisper-cpp": "not_reachable", "piper-training": "not_installed"}
    assert (engines["magpie"]["offered"], engines["canary"]["offered"], engines["piper-training"]["offered"]) == (
        True, False, True)
    assert engines["parakeet"]["key"] == "ENABLE_PARAKEET_ASR" and engines["parakeet"]["kind"] == "stt"
    assert engines["parakeet"]["install"] == {
        "profile": "parakeet-asr",
        "truenas": 'Apps -> tts-stt -> Edit -> delete the line profiles: ["parakeet-asr"] -> Save (about 5 GB download)',
        "compose": "docker compose --profile parakeet-asr up -d"}
    probed = [url for _, url, _ in stub.calls]
    assert "http://whisper-cpp:8080/" in probed and "http://magpie-tts-service:5008/health" in probed
    assert not [url for url in probed if "parakeet" in url or "piper-training" in url], "asked a name that does not resolve"

    calls = len(stub.calls)
    client.get("/api/settings/engines", headers=page(key=YAML_KEY))
    assert len(stub.calls) == calls, "probed again within 30 s"
    now["t"] += 31
    client.get("/api/settings/engines", headers=page(key=YAML_KEY))
    assert len(stub.calls) > calls

    # Offering an engine that does not run needs an acknowledgement that says why.
    r = put(client, YAML_KEY, {"ENABLE_CANARY_ASR": True})
    assert r.status_code == 409 and r.json()["warnings"][0]["token"] == "ENABLE_CANARY_ASR:engine_not_reachable"
    r = put(client, YAML_KEY, {"ENABLE_PARAKEET_ASR": True})
    assert r.status_code == 409 and "not installed" in r.json()["warnings"][0]["message"]
    assert put(client, YAML_KEY, {"ENABLE_CHATTERBOX_TTS": True}).status_code == 200      # running: no warning


def test_a_slow_name_lookup_counts_as_not_installed_within_the_budget(monkeypatch, folder):
    module, stub, client = api(monkeypatch, folder, {"API_KEY": YAML_KEY})

    async def slow(host):
        if "chatterbox" in host:
            raise RuntimeError("resolver broke")
        await asyncio.sleep(5 if "magpie" in host else 0)
        return True

    monkeypatch.setattr(module, "_engine_resolver", slow)
    monkeypatch.setattr(module, "_ENGINE_PROBE_BUDGET_S", 0.2)
    started = time.monotonic()
    r = client.get("/api/settings/engines", headers=page(key=YAML_KEY))
    assert r.status_code == 200 and time.monotonic() - started < 3
    states = {engine["provider"]: engine["state"] for engine in r.json()["engines"]}
    assert (states["magpie"], states["chatterbox"], states["canary"]) == ("not_installed", "not_reachable", "running")


# --- 48-53: the request rules ------------------------------------------------------------------------


def test_48_only_small_json_without_duplicate_names_is_taken(monkeypatch, folder):
    _, stub, client = api(monkeypatch, folder, {"API_KEY": YAML_KEY})
    headers = page(key=YAML_KEY)
    body = json.dumps({"base_revision": 0, "set": {"MAX_TTS_CHARS": 100}})
    for content_type in ("text/plain", "application/x-www-form-urlencoded", "multipart/form-data; boundary=x"):
        r = client.put("/api/settings", content=body, headers={**headers, "Content-Type": content_type})
        assert r.status_code == 415 and r.json()["code"] == "unsupported_media_type", content_type
    r = client.post("/api/settings/claim-code", content=b"", headers=page())
    assert r.status_code == 415

    hosts = [f"host-{index:04d}-{'a' * 50}.example.com" for index in range(1100)]
    big = json.dumps({"base_revision": 0, "set": {"TRUSTED_HOSTS": hosts}})
    assert len(big) > 64 * 1024
    r = client.put("/api/settings", content=big, headers={**headers, "Content-Type": "application/json"})
    assert r.status_code == 413 and "64 KiB" in r.json()["detail"] and r.json()["code"] == "request_too_large"

    for duplicate in ('{"base_revision": 0, "set": {"MAX_TTS_CHARS": 100, "MAX_TTS_CHARS": 99999}}',
                      '{"base_revision": 0, "base_revision": 1}'):
        r = client.put("/api/settings", content=duplicate, headers={**headers, "Content-Type": "application/json"})
        assert r.status_code == 400 and r.json()["code"] == "duplicate_key", duplicate
    for broken in ("[1]", "{", '{"base_revision": NaN}'):
        r = client.put("/api/settings", content=broken, headers={**headers, "Content-Type": "application/json"})
        assert r.status_code == 400 and r.json()["code"] == "invalid_json", broken
    r = client.put("/api/settings", json={"base_revision": "0", "set": [], "dry_run": 1, "extra": True}, headers=headers)
    assert r.status_code == 400
    assert r.json()["errors"] == {"base_revision": "Expected the revision the change is based on.",
                                  "set": "Expected an object of settings.", "dry_run": "Expected true or false.",
                                  "extra": "Unknown field."}
    assert not (folder / "gateway.json").exists() and stub.calls == []

    r = client.put("/api/settings", content=body, headers={**headers, "Content-Type": "application/json; charset=utf-8"})
    assert r.status_code == 200


def test_49_a_foreign_page_cannot_change_settings_even_when_cors_lets_it_call_the_api(monkeypatch, folder):
    env = {"API_KEY": YAML_KEY, "ALLOWED_ORIGINS": "https://ui.example", "TRUSTED_ORIGINS": "https://tts.example.com"}
    _, _, client = api(monkeypatch, folder, env)
    change = {"base_revision": 0, "set": {"MAX_TTS_CHARS": 100}}
    refusals = ({"Origin": "https://ui.example"}, {"Origin": "https://evil.example"}, {"Origin": "null"},
                {"Sec-Fetch-Site": "cross-site"}, {"Sec-Fetch-Site": "same-site"})
    for extra in refusals:
        r = client.put("/api/settings", json=change, headers={**page(key=YAML_KEY), **extra})
        assert r.status_code == 403, extra
        assert r.json()["code"] in ("cross_origin_blocked", "cross_site_blocked"), extra
    assert not (folder / "gateway.json").exists()
    assert client.delete(DELETE_JOB, headers={"Origin": "https://ui.example", **bearer(YAML_KEY)}).status_code == 200
    # This UI behind a Host-rewriting proxy (TRUSTED_ORIGINS) is this UI.
    proxied = {"Host": "frontend-service:3000", "Origin": "https://tts.example.com", "Sec-Fetch-Site": "same-origin",
               **bearer(YAML_KEY)}
    assert client.put("/api/settings", json=change, headers=proxied).status_code == 200


def test_50_allowed_hosts_star_does_not_switch_the_check_off_for_the_settings(monkeypatch, folder):
    env = {"API_KEY": YAML_KEY, "ALLOWED_HOSTS": "*", "TRUSTED_HOSTS": "speach.k2o"}
    _, _, client = api(monkeypatch, folder, env)
    assert host_status(client, "evil.example") == 200                                  # everything else: off
    for path in ("/settings", "/api/settings"):
        r = client.get(path, headers={"Host": "evil.example", **bearer(YAML_KEY)})
        assert r.status_code == 403 and r.json()["code"] == "host_not_allowed", path
    assert client.get("/api/settings", headers=page("speach.k2o", YAML_KEY)).status_code == 200
    # ...and the self-check judges a change by that rule too
    r = put(client, YAML_KEY, {"TRUSTED_HOSTS": []}, host="speach.k2o")
    assert r.status_code == 409 and r.json()["code"] == "would_lock_out"


def test_51_no_settings_answer_is_cached_and_none_is_in_the_openapi(monkeypatch, folder):
    module, _, client = api(monkeypatch, folder, {"API_KEY": YAML_KEY})
    answers = [
        client.get("/settings"),
        client.get("/settings/nothing"),
        client.get("/api/settings"),
        client.get("/api/settings", headers=page(key=YAML_KEY)),
        client.put("/api/settings", content="x", headers={**page(key=YAML_KEY), "Content-Type": "text/plain"}),
        client.put("/api/settings", json={"base_revision": 0}, headers=page(key=YAML_KEY)),
        client.post("/api/settings/nothing", json={}, headers=page(key=YAML_KEY)),
        client.delete("/api/settings/keys/no such key", headers=page(key=YAML_KEY)),
        client.patch("/api/settings", json={}, headers=page(key=YAML_KEY)),
        client.options("/api/settings"),
        client.get("/api/settings", headers={"Host": "evil.example"}),
        client.get("/settings", headers={"Host": "evil.example", "Accept": "text/html"}),
    ]
    assert len({r.status_code for r in answers}) >= 7
    for r in answers:
        assert r.headers.get("cache-control") == "no-store", (r.request.method, r.request.url, r.status_code)
    assert "cache-control" not in client.get("/providers").headers
    paths = module.app.openapi()["paths"]
    assert [path for path in paths if "settings" in path] == []
    committed = json.loads((REPO / "frontend-service" / "static" / "openapi" / "gateway.json").read_text(encoding="utf-8"))
    assert [path for path in committed["paths"] if "settings" in path] == []


def test_52_a_browser_gets_an_escaped_html_refusal_and_the_api_keeps_json(monkeypatch, folder):
    module, stub, _ = api(monkeypatch, folder)
    host = 'evil.example"><script>alert(1)</script>'
    navigation = [("Host", host), ("Accept", "text/html,application/xhtml+xml,*/*;q=0.8")]
    for path in ("/", "/settings", "/api-docs"):
        status, headers, body = asyncio.run(asgi_call(module.app, "GET", path, navigation))
        text = body.decode("utf-8")
        assert status == 403 and headers["content-type"].startswith("text/html"), path
        assert "<script" not in text.lower() and "&lt;script&gt;" in text, path
        assert "<button" not in text and "<form" not in text and "<a " not in text
        assert "Settings" in text and "TRUSTED_HOSTS" in text and "IP address" in text
        assert headers["content-security-policy"].startswith("default-src 'none'")
        assert headers["cache-control"] == "no-store"
    # Over https a browser says how it fetches: a script's fetch() keeps JSON whatever it accepts.
    status, headers, body = asyncio.run(asgi_call(module.app, "GET", "/settings", [
        ("Host", "evil.example"), ("Accept", "text/html"), ("Sec-Fetch-Mode", "cors")]))
    assert status == 403 and headers["content-type"] == "application/json"
    status, headers, _ = asyncio.run(asgi_call(module.app, "GET", "/", [
        ("Host", "evil.example"), ("Accept", "*/*"), ("Sec-Fetch-Mode", "navigate")]))
    assert headers["content-type"].startswith("text/html")
    # API paths keep JSON, with the name of the setting to change, and so does a script anywhere.
    for path, request in (("/api/health", navigation), ("/api/settings", navigation),
                          ("/providers", [("Host", host), ("Accept", "*/*")])):
        status, headers, body = asyncio.run(asgi_call(module.app, "GET", path, request))
        detail = json.loads(body)["detail"]
        assert status == 403 and "TRUSTED_HOSTS" in detail and "Settings -> Access" in detail, path
        assert headers["content-type"] == "application/json"
    assert stub.calls == []

    # Where the page cannot add names, the refusal points at the app YAML only.
    locked = load_frontend_app({"TTS_STT_SETTINGS_DIR": str(folder), "SETTINGS_LOCKED_KEYS": "TRUSTED_HOSTS"})
    status, headers, body = asyncio.run(asgi_call(locked.app, "GET", "/providers", [("Host", "evil.example")]))
    detail = json.loads(body)["detail"]
    assert "TRUSTED_HOSTS in the app YAML" in detail and "Settings" not in detail
    status, headers, body = asyncio.run(asgi_call(locked.app, "GET", "/", navigation))
    assert "Settings" not in body.decode("utf-8") and "TRUSTED_HOSTS" in body.decode("utf-8")


def test_53_without_the_mount_nothing_can_be_saved(monkeypatch, caplog):
    module = load_frontend_app({"API_KEY": YAML_KEY})            # /app/settings, which is not mounted here
    install_stub(monkeypatch, module, backend)
    client = TestClient(module.app)
    view = view_of(client, YAML_KEY)
    assert view["mount"]["writable"] is False and view["mount"]["state"] in ("not_mounted", "missing")
    assert (view["can_write"], view["can_manage_keys"], view["can_claim"]) == (False, False, False)
    assert "not_mounted" in {banner["id"] for banner in view["banners"]}
    assert view["mount"]["lines"]["truenas"][0] == '- "${APP_DATA_DIR:?set dataset path}/settings:/app/settings"'
    writes = (
        ("PUT", "/api/settings", {"base_revision": 0, "set": {"MAX_TTS_CHARS": 100}}),
        ("POST", "/api/settings/discard", {"base_revision": 0}),
        ("POST", "/api/settings/restore", {"base_revision": 0, "history_id": "20261002T091203Z-r1"}),
        ("POST", "/api/settings/confirm", {"revision": 1}),
        ("POST", "/api/settings/keys", {"name": "Laptop", "role": "admin", "key": new_key()}),
        ("PUT", "/api/settings/access", {"base_revision": 0, "require_key": True}),
        ("POST", "/api/settings/claim-code", {}),
        ("POST", "/api/settings/claim", {"code": "AAAAA-AAAAA-AAAAA-AAAAA", "name": "x", "key": new_key()}),
    )
    with caplog.at_level(logging.INFO):
        for method, path, body in writes:
            r = client.request(method, path, json=body, headers=page(key=YAML_KEY))
            assert r.status_code == 409 and r.json()["code"] == "settings_not_mounted", path
    assert not [record for record in caplog.records if record.name == "tts_stt.settings.claim"]


# --- what else the API refuses or keeps -------------------------------------------------------------


def test_a_stale_revision_is_refused_and_names_the_current_one(monkeypatch, folder):
    _, _, client = api(monkeypatch, folder, {"API_KEY": YAML_KEY})
    assert put(client, YAML_KEY, {"MAX_TTS_CHARS": 100}).status_code == 200
    r = put(client, YAML_KEY, {"MAX_TTS_CHARS": 200}, base=0)
    assert r.status_code == 409 and r.json()["code"] == "revision_conflict" and r.json()["revision"] == 1
    assert put(client, YAML_KEY, {"MAX_TTS_CHARS": 100}).json()["written"] is False      # nothing to do


def test_disabled_or_safe_mode_settings_cannot_be_changed_but_keys_can_in_safe_mode(monkeypatch, folder, writer):
    _, _, disabled = api(monkeypatch, folder, {"API_KEY": YAML_KEY, "ENABLE_SETTINGS_UI": "false"})
    view = view_of(disabled, YAML_KEY)
    assert view["enabled"] is False and view["can_write"] is False and "disabled" in {b["id"] for b in view["banners"]}
    for method, path, body in (("PUT", "/api/settings", {"base_revision": 0}),
                               ("POST", "/api/settings/keys", {"name": "x", "role": "client", "key": new_key()}),
                               ("POST", "/api/settings/claim-code", {})):
        r = disabled.request(method, path, json=body, headers=page(key=YAML_KEY))
        assert r.status_code == 403 and r.json()["code"] == "settings_disabled", path

    (folder / "SAFE-MODE").write_text("", encoding="utf-8")
    _, _, client = api(monkeypatch, folder, {"API_KEY": YAML_KEY})
    r = put(client, YAML_KEY, {"MAX_TTS_CHARS": 100})
    assert r.status_code == 403 and r.json()["code"] == "settings_disabled" and "SAFE-MODE" in r.json()["detail"]
    assert create_key(client, YAML_KEY, "Recovery", role="admin")
    assert "safe_mode" in {banner["id"] for banner in view_of(client, YAML_KEY)["banners"]}


def test_a_new_code_is_refused_within_a_minute_and_codes_never_reach_the_audit(monkeypatch, folder, caplog):
    _, _, client = api(monkeypatch, folder)
    with caplog.at_level(logging.INFO):
        assert client.post("/api/settings/claim-code", json={}, headers=page()).status_code == 202
        r = client.post("/api/settings/claim-code", json={}, headers=page())
    assert r.status_code == 429 and r.json()["code"] == "rate_limited" and int(r.headers["retry-after"]) > 0
    code = printed_code(caplog)
    audit = [record.getMessage() for record in caplog.records if record.name == "tts_stt.settings.audit"]
    assert audit and not [line for line in audit if code in line]


def test_keys_are_validated_and_unique(monkeypatch, folder):
    _, _, client = api(monkeypatch, folder, {"API_KEY": YAML_KEY})
    key = create_key(client, YAML_KEY, "Laptop")
    attempts = (
        ({"name": "laptop", "role": "client", "key": new_key()}, "name"),
        ({"name": "Other", "role": "client", "key": key}, "key"),
        ({"name": "Other", "role": "root", "key": new_key()}, "role"),
        ({"name": "Other", "role": "client", "key": "tts_short"}, "key"),
        ({"name": "", "role": "client", "key": new_key()}, "name"),
    )
    for body, field_name in attempts:
        r = client.post("/api/settings/keys", json=body, headers=page(key=YAML_KEY))
        assert r.status_code == 400 and field_name in r.json()["errors"], body
        assert key not in r.text
    r = client.delete("/api/settings/keys/not a valid id", headers=page(key=YAML_KEY))
    assert r.status_code == 422


def test_restore_asks_for_the_acknowledgements_of_what_it_brings_back(monkeypatch, folder):
    _, _, client = api(monkeypatch, folder, {"API_KEY": YAML_KEY})
    r = put(client, YAML_KEY, {"MAX_TTS_CHARS": 6000}, acknowledge=["MAX_TTS_CHARS:tts_chars_above_backend_limit"])
    assert r.status_code == 200
    saved = view_of(client, YAML_KEY)["history"][0]["id"]
    r = client.post("/api/settings/discard", json={"base_revision": r.json()["revision"]}, headers=page(key=YAML_KEY))
    assert r.status_code == 200
    body = {"history_id": saved, "base_revision": r.json()["revision"]}
    r = client.post("/api/settings/restore", json=body, headers=page(key=YAML_KEY))
    assert r.status_code == 409 and r.json()["warnings"][0]["key"] == "MAX_TTS_CHARS"
    r = client.post("/api/settings/restore", json={**body, "dry_run": True}, headers=page(key=YAML_KEY))
    assert r.status_code == 200 and r.json()["written"] is False and r.json()["changed"] == ["MAX_TTS_CHARS"]
    r = client.post("/api/settings/restore", json={**body, "acknowledge": ["MAX_TTS_CHARS:tts_chars_above_backend_limit"]},
                    headers=page(key=YAML_KEY))
    assert r.status_code == 200 and r.json()["written"] is True
    assert field(view_of(client, YAML_KEY), "MAX_TTS_CHARS")["value"] == 6000
    gone = {"history_id": "20200101T000000Z-r99", "base_revision": r.json()["revision"]}
    r = client.post("/api/settings/restore", json=gone, headers=page(key=YAML_KEY))
    assert r.status_code == 404 and r.json()["code"] == "history_not_found"
