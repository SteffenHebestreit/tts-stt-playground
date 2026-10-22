"""The gateway's side of vllm-asr.

* The provider exists only behind ENABLE_VLLM_ASR, as an stt-form-v1 engine with German
  preselected, no "auto" entry, and the default model's (Granite's) six languages.
* Which model the service runs (VLLM_ASR_MODEL) decides its languages: the service's
  /status replaces the built-in list, as for Canary and Granite, so Qwen3-ASR's 28 or
  the German Whisper's one show up in the selector with the model's name.
* /api/stt reaches /transcribe with the language and a non-empty prompt as
  `initial_prompt`; the Settings page offers and names the engine.
"""

from __future__ import annotations

import asyncio
import json
import re
import sys

import httpx
from fastapi.testclient import TestClient

from frontend_loader import REPO, install_stub, load_frontend_app, wav_bytes

ON = {"ENABLE_VLLM_ASR": "true"}
URL = "http://vllm-asr-service:5011"


def _gateway(monkeypatch, handler, env=None):
    module = load_frontend_app(env)
    stub = install_stub(monkeypatch, module, handler)
    return module, stub, TestClient(module.app)


def _registry(client, path="/providers"):
    if path == "/providers":
        return client.get("/providers").json()
    page = client.get("/").text
    blob = re.search(r'<script[^>]*id="provider-registry-data"[^>]*>(.*?)</script>', page, re.S).group(1)
    return json.loads(blob)


def test_the_provider_is_offered_only_behind_its_flag(monkeypatch):
    off = load_frontend_app().PROVIDER_REGISTRY
    assert "vllm-asr" not in off["providers"] and off["ui"]["enable_vllm_asr"] is False

    on = load_frontend_app(ON).PROVIDER_REGISTRY
    entry = on["providers"]["vllm-asr"]
    assert entry["kind"] == "stt" and entry["internal_url"] == URL
    assert entry["contracts"] == {"transcribe": "stt-form-v1"}
    assert entry["capabilities"] == ["transcribe", "segments", "model_unload"]
    assert entry["language_detect"] is False
    languages = [item["value"] for item in entry["settings"]["languages"]]
    assert languages[0] == "de" and sorted(languages) == ["de", "en", "es", "fr", "ja", "pt"]
    assert "auto" not in languages and entry["ui"]["selectable_as_stt"] is True
    assert on["ui"]["enable_vllm_asr"] is True


def test_the_service_url_can_be_moved(monkeypatch):
    module = load_frontend_app({**ON, "VLLM_ASR_SERVICE_URL": "http://gpu-box:5011"})
    assert module.PROVIDER_REGISTRY["providers"]["vllm-asr"]["internal_url"] == "http://gpu-box:5011"


def test_it_can_be_the_default_engine_and_serve_v1(monkeypatch):
    seen = []

    def handler(method, url, kwargs):
        seen.append(url)
        return {"text": "hallo welt", "segments": [{"start": 0.5, "end": 1.5, "text": "hallo welt"}],
                "language": "de", "duration": 2.0}

    _, _, client = _gateway(monkeypatch, handler, {**ON, "DEFAULT_STT_PROVIDER": "vllm-asr"})
    r = client.post("/v1/audio/transcriptions", data={"model": "whisper-1", "response_format": "srt"},
                    files={"file": ("a.wav", wav_bytes(), "audio/wav")})
    assert r.status_code == 200
    assert "00:00:00,500 --> 00:00:01,500" in r.text and "hallo welt" in r.text
    assert seen[-1] == f"{URL}/transcribe"


# --- the languages follow VLLM_ASR_MODEL -------------------------------------------------------


class _Vllm:
    def __init__(self, monkeypatch, status):
        self.status = status
        self.asked = 0

        def handler(method, url, kwargs):
            if url == f"{URL}/status":
                self.asked += 1
                return self.status
            if url.endswith("/status"):
                raise httpx.ConnectError("not this one")
            return {"status": "ok"}

        self.module, self.stub, self.client = _gateway(monkeypatch, handler, ON)
        monkeypatch.setattr(self.module, "_language_clock", lambda: 1000.0)

    def entry(self, path="/providers"):
        return _registry(self.client, path)["providers"]["vllm-asr"]

    def languages(self, path="/providers"):
        return [item["value"] for item in self.entry(path)["settings"]["languages"]]


def _service_status(monkeypatch, **env):
    """vllm-asr-service's own /status, under the test loader's fakes."""
    from vllm_asr_loader import load_app

    service = load_app(monkeypatch, **env)

    async def fetch():
        transport = httpx.ASGITransport(app=service.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://vllm.test") as c:
            return (await c.get("/status")).json()

    return asyncio.run(fetch())


def test_qwen3_lists_its_28_languages_and_is_named_in_the_selector(monkeypatch):
    status = _service_status(monkeypatch, VLLM_ASR_MODEL="Qwen/Qwen3-ASR-1.7B")
    vllm = _Vllm(monkeypatch, status)
    assert len(vllm.languages()) == 28 and vllm.languages()[0] == "de"
    assert vllm.entry()["display_name"] == "vLLM (Qwen3-ASR-1.7B, 28 languages)"
    assert "vLLM (Qwen3-ASR-1.7B, 28 languages)" in vllm.client.get("/").text


def test_the_german_whisper_offers_german_only(monkeypatch):
    status = _service_status(monkeypatch, VLLM_ASR_MODEL="primeline/whisper-large-v3-turbo-german")
    vllm = _Vllm(monkeypatch, status)
    assert vllm.languages() == ["de"]
    assert vllm.entry()["display_name"] == "vLLM (whisper-large-v3-turbo-german, de)"


def test_cohere_lists_its_14_and_granite_its_6(monkeypatch):
    # Both statuses first: the gateway's stub replaces httpx.AsyncClient for the rest of the test.
    cohere_status = _service_status(monkeypatch, VLLM_ASR_MODEL="evewashere/cohere-transcribe-03-2026-ungated")
    granite_status = _service_status(monkeypatch)
    cohere = _Vllm(monkeypatch, cohere_status)
    assert len(cohere.languages()) == 14 and cohere.languages()[0] == "de"
    granite = _Vllm(monkeypatch, granite_status)
    assert sorted(granite.languages()) == ["de", "en", "es", "fr", "ja", "pt"]
    assert granite.entry()["display_name"] == "vLLM (granite-speech-5.0-2b, de/en/fr/ja/pt/es)"


# --- /api/stt ----------------------------------------------------------------------------------------


def test_api_stt_reaches_the_service_with_the_language_and_the_prompt(monkeypatch):
    def handler(method, url, kwargs):
        return {"text": "Guten Tag.", "segments": [{"start": 0.3, "end": 1.2, "text": "Guten Tag."}],
                "language": "de", "duration": 1.5, "warnings": [], "engine": "vllm"}

    _, stub, client = _gateway(monkeypatch, handler, ON)
    r = client.post("/api/stt", data={"provider": "vllm-asr", "language": "de", "prompt": " LUNA, Bautzen "},
                    files={"audio": ("a.wav", wav_bytes(), "audio/wav")})
    assert r.status_code == 200 and r.headers["x-provider"] == "vllm-asr"
    assert r.json() == {"text": "Guten Tag.", "segments": [{"start": 0.3, "end": 1.2, "text": "Guten Tag."}],
                        "language": "de", "duration": 1.5}
    method, url, kwargs = stub.calls[-1]
    assert url == f"{URL}/transcribe"
    assert kwargs["data"] == {"language": "de", "initial_prompt": "LUNA, Bautzen"}


def test_auto_reaches_the_service_as_auto(monkeypatch):
    """Qwen3-ASR identifies the language on an explicit auto; the service decides, not the gateway."""
    _, stub, client = _gateway(monkeypatch, lambda m, u, k: {"text": "x"}, ON)
    client.post("/api/stt", data={"provider": "vllm-asr", "language": "auto"},
                files={"audio": ("a.wav", wav_bytes(), "audio/wav")})
    assert stub.calls[-1][2]["data"] == {"language": "auto"}


# --- the Settings page ----------------------------------------------------------------------------------


def _schema():
    import importlib

    from frontend_loader import SERVICE_DIR

    sys.path.insert(0, str(SERVICE_DIR))
    try:
        return importlib.import_module("settings_schema")
    finally:
        sys.path.remove(str(SERVICE_DIR))


def test_the_settings_page_offers_and_names_the_engine():
    schema = _schema()
    assert schema.ENGINE_FLAGS["ENABLE_VLLM_ASR"] == ("vllm-asr", "stt")
    assert "ENABLE_VLLM_ASR" in schema.OPTIONAL_ENGINE_FLAGS
    offered = schema.offered_engines({"ENABLE_VLLM_ASR": True})
    assert offered.get("vllm-asr") == "stt"
    assert schema.check_engine_defaults(
        {"DEFAULT_STT_PROVIDER": "vllm-asr", "DEFAULT_TTS_PROVIDER": "piper", "ENABLE_VLLM_ASR": True},
        offered) == {}
    js = (REPO / "frontend-service" / "static" / "js" / "settings.js").read_text(encoding="utf-8")
    assert "'vllm-asr': 'vLLM-ASR'" in js


def test_the_settings_page_says_the_image_is_built_on_the_nas_first():
    """Deleting the profile line alone would pull an image that is not published."""
    module = load_frontend_app(ON)
    hint = module._install_hint(*module._ENGINE_PROFILES["vllm-asr"])
    assert hint["truenas"].startswith("Build its image on the NAS first (vllm-asr-service/README.md)")
    assert 'profiles: ["vllm-asr"]' in hint["truenas"]
    assert hint["compose"] == "docker compose --profile vllm-asr up -d"
    assert not module._install_hint(*module._ENGINE_PROFILES["granite-asr"])["truenas"].startswith("Build")
