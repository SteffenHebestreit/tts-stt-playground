"""The gateway's side of cohere-asr and granite-asr.

* Each provider exists only behind its flag, as an stt-form-v1 engine without
  language detection, with German preselected and no "auto" entry.
* Granite's language list follows the checkpoint its service runs (the 2b has six
  languages, the 470m TurboCTC seven): the service's /status replaces the built-in
  list, exactly as for Canary.
* /api/stt passes `prompt` on as `initial_prompt`, the name the stt-form-v1 services
  read (Granite's keyword list), and only when it says something.
* The Settings page offers both engines.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from frontend_loader import install_stub, load_frontend_app, wav_bytes

REPO = Path(__file__).resolve().parents[1]
BOTH = {"ENABLE_COHERE_ASR": "true", "ENABLE_GRANITE_ASR": "true"}


def _gateway(monkeypatch, handler, env=None):
    module = load_frontend_app(env)
    stub = install_stub(monkeypatch, module, handler)
    return module, stub, TestClient(module.app)


def _status(languages, default="de", model="ibm-granite/granite-speech-5.0-2b"):
    return {"status": "ok", "supported_languages": languages, "default_language": default, "current_model": model}


def _registry(client, path="/providers"):
    if path == "/providers":
        return client.get("/providers").json()
    page = client.get("/").text
    blob = re.search(r'<script[^>]*id="provider-registry-data"[^>]*>(.*?)</script>', page, re.S).group(1)
    return json.loads(blob)


# --- the registry ------------------------------------------------------------------------


def test_both_providers_are_offered_only_behind_their_flags(monkeypatch):
    off = load_frontend_app().PROVIDER_REGISTRY
    assert "cohere-asr" not in off["providers"] and "granite-asr" not in off["providers"]
    assert off["ui"]["enable_cohere_asr"] is False and off["ui"]["enable_granite_asr"] is False

    on = load_frontend_app(BOTH).PROVIDER_REGISTRY
    for provider_id, url in (("cohere-asr", "http://cohere-asr-service:5009"),
                             ("granite-asr", "http://granite-asr-service:5010")):
        entry = on["providers"][provider_id]
        assert entry["kind"] == "stt" and entry["internal_url"] == url
        assert entry["contracts"] == {"transcribe": "stt-form-v1"}
        assert entry["language_detect"] is False
        assert entry["capabilities"] == ["transcribe", "segments", "model_unload"]
        languages = [item["value"] for item in entry["settings"]["languages"]]
        assert languages[0] == "de" and entry["settings"]["defaults"]["language"] == "de"
        assert "auto" not in languages, "neither model detects the language"
        assert entry["ui"]["selectable_as_stt"] is True
    assert len(on["providers"]["cohere-asr"]["settings"]["languages"]) == 14
    assert on["ui"]["enable_cohere_asr"] is True and on["ui"]["enable_granite_asr"] is True


def test_the_service_urls_can_be_moved(monkeypatch):
    module = load_frontend_app({**BOTH, "COHERE_ASR_SERVICE_URL": "http://gpu-box:5009",
                                "GRANITE_ASR_SERVICE_URL": "http://gpu-box:5010"})
    providers = module.PROVIDER_REGISTRY["providers"]
    assert providers["cohere-asr"]["internal_url"] == "http://gpu-box:5009"
    assert providers["granite-asr"]["internal_url"] == "http://gpu-box:5010"


def test_either_can_be_the_default_engine_and_serve_v1(monkeypatch):
    seen = []

    def handler(method, url, kwargs):
        seen.append(url)
        return {"text": "hallo welt", "segments": [{"start": 0.5, "end": 1.5, "text": "hallo welt"}],
                "language": "de", "duration": 2.0}

    _, _, client = _gateway(monkeypatch, handler, {**BOTH, "DEFAULT_STT_PROVIDER": "granite-asr"})
    r = client.post("/v1/audio/transcriptions", data={"model": "whisper-1", "response_format": "srt"},
                    files={"file": ("a.wav", wav_bytes(), "audio/wav")})
    assert r.status_code == 200
    assert "00:00:00,500 --> 00:00:01,500" in r.text and "hallo welt" in r.text
    assert seen[-1] == "http://granite-asr-service:5010/transcribe"


# --- Granite's languages follow its checkpoint ------------------------------------------


class _Granite:
    def __init__(self, monkeypatch, status, env=None):
        self.status = status
        self.asked = 0
        self.now = 1000.0

        def handler(method, url, kwargs):
            if url == "http://granite-asr-service:5010/status":
                self.asked += 1
                if isinstance(self.status, Exception):
                    raise self.status
                return self.status
            if url.endswith("/status"):
                raise httpx.ConnectError("not this one")
            return {"status": "ok"}

        self.module, self.stub, self.client = _gateway(monkeypatch, handler, {**BOTH, **(env or {})})
        monkeypatch.setattr(self.module, "_language_clock", lambda: self.now)

    def entry(self, path="/providers"):
        return _registry(self.client, path)["providers"]["granite-asr"]

    def languages(self, path="/providers"):
        return [item["value"] for item in self.entry(path)["settings"]["languages"]]


def test_the_turboctc_adds_italian_and_is_named_in_the_selector(monkeypatch):
    granite = _Granite(monkeypatch, _status(["en", "fr", "de", "es", "pt", "it", "ja"],
                                            model="ibm-granite/granite-speech-5.0-470m-turboctc"))
    assert granite.languages()[0] == "de" and "it" in granite.languages() and len(granite.languages()) == 7
    assert granite.entry()["display_name"] == "Granite Speech (granite-speech-5.0-470m-turboctc, 7 languages)"
    page = granite.client.get("/").text
    assert "Granite Speech (granite-speech-5.0-470m-turboctc, 7 languages)" in page


def test_the_2b_lists_its_six_and_the_services_default_leads(monkeypatch):
    granite = _Granite(monkeypatch, _status(["en", "fr", "de", "es", "pt", "ja"], default="fr"))
    assert granite.languages()[0] == "fr" and len(granite.languages()) == 6
    assert granite.entry()["settings"]["defaults"]["language"] == "fr"
    assert granite.entry()["display_name"] == "Granite Speech (granite-speech-5.0-2b, fr/en/de/ja/pt/es)"


def test_an_unreachable_granite_keeps_the_built_in_list_and_is_not_asked_on_every_page(monkeypatch):
    granite = _Granite(monkeypatch, httpx.ConnectError("refused"))
    for _ in range(3):
        assert granite.languages() == ["de", "en", "fr", "ja", "pt", "es"]
    assert granite.asked == 1
    assert granite.entry()["display_name"] == "Granite Speech 5.0 (IBM, de/en/fr/ja/pt/es)"
    granite.now += 21
    granite.status = _status(["de", "en"])
    assert granite.languages() == ["de", "en"] and granite.asked == 2


def test_cohere_is_not_asked_for_its_languages(monkeypatch):
    calls = []

    def handler(method, url, kwargs):
        calls.append(url)
        return {"status": "ok"}

    _, _, client = _gateway(monkeypatch, handler, {"ENABLE_COHERE_ASR": "true"})
    client.get("/")
    client.get("/providers")
    assert not [url for url in calls if url.endswith("/status")], "its 14 languages do not depend on a setting"


def test_canary_and_granite_are_discovered_side_by_side(monkeypatch):
    def handler(method, url, kwargs):
        if url == "http://canary-asr-service:5006/status":
            return {"supported_languages": ["de", "en"], "default_language": "de",
                    "current_model": "nvidia/canary-180m-flash"}
        if url == "http://granite-asr-service:5010/status":
            return _status(["de", "it"], model="ibm-granite/granite-speech-5.0-470m-turboctc")
        return {"status": "ok"}

    _, _, client = _gateway(monkeypatch, handler, {**BOTH, "ENABLE_CANARY_ASR": "true"})
    providers = _registry(client)["providers"]
    assert providers["canary"]["display_name"] == "Canary (canary-180m-flash, de/en)"
    assert providers["granite-asr"]["display_name"] == "Granite Speech (granite-speech-5.0-470m-turboctc, de/it)"


def test_the_real_granite_status_is_what_the_gateway_understands(monkeypatch, tmp_path):
    """The service's own /status, both checkpoints, read by the gateway's discovery."""
    import asyncio

    from transformers_asr_loader import load_app

    statuses = []
    for model, expected in (("ibm-granite/granite-speech-5.0-2b", 6),
                            ("ibm-granite/granite-speech-5.0-470m-turboctc", 7)):
        service = load_app("granite", monkeypatch, GRANITE_ASR_MODEL=model)

        async def fetch():
            transport = httpx.ASGITransport(app=service.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://granite.test") as c:
                return (await c.get("/status")).json()

        statuses.append((asyncio.run(fetch()), expected))

    for reported, expected in statuses:
        granite = _Granite(monkeypatch, reported)
        assert sorted(granite.languages()) == sorted(reported["supported_languages"])
        assert len(granite.languages()) == expected and granite.languages()[0] == "de"


# --- /api/stt ----------------------------------------------------------------------------


@pytest.mark.parametrize("provider", ["cohere-asr", "granite-asr"])
def test_api_stt_reaches_the_service_with_the_language_and_normalises_the_answer(monkeypatch, provider):
    def handler(method, url, kwargs):
        return {"text": "Guten Tag.", "segments": [{"start": 0.3, "end": 1.2, "text": "Guten Tag.", "truncated": False}],
                "language": "de", "duration": 1.5, "warnings": [], "speech_seconds": 0.9}

    _, stub, client = _gateway(monkeypatch, handler, BOTH)
    r = client.post("/api/stt", data={"provider": provider, "language": "de"},
                    files={"audio": ("a.wav", wav_bytes(), "audio/wav")})
    assert r.status_code == 200 and r.headers["x-provider"] == provider
    assert r.json() == {"text": "Guten Tag.", "segments": [{"start": 0.3, "end": 1.2, "text": "Guten Tag."}],
                        "language": "de", "duration": 1.5}
    method, url, kwargs = stub.calls[-1]
    assert url.endswith("/transcribe") and kwargs["data"] == {"language": "de"}


def test_api_stt_passes_a_prompt_on_as_the_initial_prompt(monkeypatch):
    _, stub, client = _gateway(monkeypatch, lambda m, u, k: {"text": "x"}, BOTH)
    client.post("/api/stt", data={"provider": "granite-asr", "language": "de", "prompt": " ACME, Leipzig "},
                files={"audio": ("a.wav", wav_bytes(), "audio/wav")})
    assert stub.calls[-1][2]["data"] == {"language": "de", "initial_prompt": "ACME, Leipzig"}


@pytest.mark.parametrize("prompt", ["", "   "])
def test_an_empty_prompt_is_not_sent(monkeypatch, prompt):
    """Whisper reads any string as a prompt, an empty one included (#64)."""
    _, stub, client = _gateway(monkeypatch, lambda m, u, k: {"text": "x"})
    client.post("/api/stt", data={"provider": "whisper", "language": "de", "prompt": prompt},
                files={"audio": ("a.wav", wav_bytes(), "audio/wav")})
    assert "initial_prompt" not in stub.calls[-1][2]["data"]


def test_a_language_the_service_refuses_reaches_the_caller_in_its_words(monkeypatch):
    refusal = ("Language 'sv' is not supported by Cohere Transcribe. Supported: ar, de, el, en, es, fr, it, ja, ko, "
               "nl, pl, pt, vi, zh. Send 'auto' (or no language) to use the default ('de').")
    _, _, client = _gateway(monkeypatch, lambda m, u, k: httpx.Response(422, json={"detail": refusal}), BOTH)
    r = client.post("/api/stt", data={"provider": "cohere-asr", "language": "sv"},
                    files={"audio": ("a.wav", wav_bytes(), "audio/wav")})
    assert r.status_code == 422 and r.json()["detail"] == refusal


# --- the Settings page ---------------------------------------------------------------------


def _schema():
    """settings_schema the way the gateway imports it (it imports public_suffix by bare name)."""
    import importlib

    from frontend_loader import SERVICE_DIR

    sys.path.insert(0, str(SERVICE_DIR))
    try:
        return importlib.import_module("settings_schema")
    finally:
        sys.path.remove(str(SERVICE_DIR))


def test_the_settings_page_offers_both_engines():
    schema = _schema()
    assert schema.ENGINE_FLAGS["ENABLE_COHERE_ASR"] == ("cohere-asr", "stt")
    assert schema.ENGINE_FLAGS["ENABLE_GRANITE_ASR"] == ("granite-asr", "stt")
    assert {"ENABLE_COHERE_ASR", "ENABLE_GRANITE_ASR"} <= schema.OPTIONAL_ENGINE_FLAGS
    offered = schema.offered_engines({"ENABLE_COHERE_ASR": True, "ENABLE_GRANITE_ASR": False})
    assert offered.get("cohere-asr") == "stt" and "granite-asr" not in offered
    assert schema.check_engine_defaults(
        {"DEFAULT_STT_PROVIDER": "cohere-asr", "DEFAULT_TTS_PROVIDER": "piper", "ENABLE_COHERE_ASR": True},
        offered) == {}


def test_the_settings_page_names_both_engines():
    js = (REPO / "frontend-service" / "static" / "js" / "settings.js").read_text(encoding="utf-8")
    assert "'cohere-asr': 'Cohere-ASR'" in js and "'granite-asr': 'Granite-ASR'" in js


def test_a_file_part_called_prompt_is_not_turned_into_one(monkeypatch):
    _, stub, client = _gateway(monkeypatch, lambda m, u, k: {"text": "x"}, BOTH)
    client.post("/api/stt", data={"provider": "granite-asr", "language": "de"},
                files={"audio": ("a.wav", wav_bytes(), "audio/wav"), "prompt": ("p.txt", b"ACME", "text/plain")})
    assert "initial_prompt" not in stub.calls[-1][2]["data"]
