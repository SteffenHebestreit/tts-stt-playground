"""The Settings page's values in force in the gateway, live and in every worker.

Part 1 (no settings HTTP routes yet): the values are saved the way the page will save
them, through settings_store's API, into a temporary folder named by
TTS_STT_SETTINGS_DIR, or written by hand where a test needs a damaged file. Requests
then go through a TestClient, and the very next one has to run under the change: in
the module that saved it and in a second module instance on the same folder (a second
uvicorn worker), with no re-import and no wait.

Numbers in the test names refer to the gateway tests of the phase-1 plan.
"""

from __future__ import annotations

import base64
import hashlib
import importlib
import json
import logging
import os
import secrets
import sys

import httpx
import pytest
from fastapi.testclient import TestClient
from starlette.datastructures import Headers
from starlette.websockets import WebSocketDisconnect

from frontend_loader import SERVICE_DIR, install_stub, load_frontend_app, wav_bytes


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
