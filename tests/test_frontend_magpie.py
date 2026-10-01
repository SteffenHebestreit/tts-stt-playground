"""The gateway's side of the Magpie TTS provider.

* the provider exists only when ENABLE_MAGPIE_TTS is set, like the other optional backends;
* the request body is Magpie's (`text`, `language`, `speaker`), with "auto" left for the
  service to resolve: a gateway that filled in a language or a speaker itself would
  override MAGPIE_DEFAULT_LANGUAGE / MAGPIE_DEFAULT_SPEAKER on every request;
* the five speakers reach the voice selector through the existing `speaker-catalog-v1`
  contract, so the browser needs no code of its own for them.
"""

import json

import httpx
import pytest
from fastapi.testclient import TestClient

from frontend_loader import install_stub, load_frontend_app, wav_bytes
import magpie_loader

ENABLED = {"ENABLE_MAGPIE_TTS": "true"}


def _gateway(monkeypatch, handler, env=None):
    module = load_frontend_app(env)
    stub = install_stub(monkeypatch, module, handler)
    return module, stub, TestClient(module.app)


def _wav(method, url, kwargs):
    return httpx.Response(200, content=wav_bytes(), headers={"content-type": "audio/wav"})


def _json(payload, status=200):
    return lambda method, url, kwargs: httpx.Response(status, json=payload)


def _last_json(stub):
    return stub.calls[-1][2]["json"]


# --- registration ------------------------------------------------------------------------

def test_magpie_is_absent_unless_it_is_enabled(monkeypatch):
    module, _, client = _gateway(monkeypatch, _wav)

    assert "magpie" not in module.PROVIDER_REGISTRY["providers"]
    assert client.post("/api/tts", json={"provider": "magpie", "text": "hi"}).status_code == 404
    assert module.PROVIDER_REGISTRY["ui"]["enable_magpie_tts"] is False


def test_an_enabled_magpie_is_a_tts_provider_with_the_generic_panel(monkeypatch):
    module, _, _ = _gateway(monkeypatch, _wav, ENABLED)

    provider = module.PROVIDER_REGISTRY["providers"]["magpie"]

    assert provider["kind"] == "tts"
    assert provider["internal_url"] == "http://magpie-tts-service:5008"
    assert provider["contracts"] == {"tts": "simple-json-tts-v1", "voice_catalog": "speaker-catalog-v1"}
    assert set(provider["capabilities"]) == {"tts", "voice_catalog", "model_unload"}
    assert provider["ui"]["family"] == "piper" and provider["ui"]["selectable_as_engine"] is True
    assert module.PROVIDER_REGISTRY["ui"]["enable_magpie_tts"] is True


def test_the_language_options_are_only_ones_the_service_can_speak(monkeypatch):
    """NeMo 3.0.x reads any other language with English rules; offering them would invite a wrong 200."""
    module, _, _ = _gateway(monkeypatch, _wav, ENABLED)

    options = [item["value"] for item in module.PROVIDER_REGISTRY["providers"]["magpie"]["settings"]["languages"]]

    assert options == ["auto", "de", "en", "es", "fr", "ja", "zh"]
    assert module.PROVIDER_REGISTRY["providers"]["magpie"]["settings"]["defaults"]["language"] == "auto"


def test_the_service_url_is_configurable(monkeypatch):
    monkeypatch.setenv("MAGPIE_TTS_SERVICE_URL", "http://nas.local:5008")
    module, _, _ = _gateway(monkeypatch, _wav, ENABLED)

    assert module.PROVIDER_REGISTRY["providers"]["magpie"]["internal_url"] == "http://nas.local:5008"


def test_the_health_probe_reports_magpie_when_enabled(monkeypatch):
    def handler(method, url, kwargs):
        if "magpie" in url:
            return httpx.Response(200, json={"status": "ok", "model_loaded": False, "model_resident": False})
        return httpx.Response(200, json={"status": "healthy"})

    _, stub, client = _gateway(monkeypatch, handler, ENABLED)

    providers = client.get("/api/health").json()["providers"]

    assert providers["magpie"]["healthy"] is True
    assert any("magpie-tts-service:5008/health" in call[1] for call in stub.calls)


# --- the request body --------------------------------------------------------------------

def test_api_tts_sends_magpies_own_fields(monkeypatch):
    _, stub, client = _gateway(monkeypatch, _wav, ENABLED)

    r = client.post("/api/tts", json={
        "provider": "magpie", "text": "Guten Tag", "language": "de", "voice": "Leo", "speed": 1.4, "quality": "high"})

    assert r.status_code == 200
    assert _last_json(stub) == {"text": "Guten Tag", "language": "de", "speaker": "Leo"}
    assert stub.calls[-1][1] == "http://magpie-tts-service:5008/tts"


@pytest.mark.parametrize("language", ["auto", "", None])
def test_an_automatic_language_and_a_missing_voice_are_left_for_the_service_to_resolve(monkeypatch, language):
    _, stub, client = _gateway(monkeypatch, _wav, ENABLED)
    body = {"provider": "magpie", "text": "Guten Tag"}
    if language is not None:
        body["language"] = language

    client.post("/api/tts", json=body)

    assert _last_json(stub) == {"text": "Guten Tag", "language": "auto", "speaker": "auto"}


def test_what_the_gateway_sends_is_what_the_service_resolves_to_its_own_defaults(monkeypatch):
    """The two halves live in different services; feed the gateway's request to the real resolvers."""
    _, stub, client = _gateway(monkeypatch, _wav, ENABLED)
    support = magpie_loader.load_support()
    client.post("/api/tts", json={"provider": "magpie", "text": "Guten Tag"})
    sent = _last_json(stub)

    language = support.resolve_language(sent["language"], "de", ["de", "en"])
    speaker = support.resolve_speaker(sent["speaker"], list(support.DEFAULT_SPEAKERS), default_index=4)

    assert language == "de" and speaker == 4, "an 'auto' from the gateway must mean the service's configured defaults"


def test_a_speaker_the_gateway_forwards_resolves_in_the_service(monkeypatch):
    _, stub, client = _gateway(monkeypatch, _wav, ENABLED)
    support = magpie_loader.load_support()
    client.post("/api/tts", json={"provider": "magpie", "text": "hi", "voice": "Jason"})

    assert support.resolve_speaker(_last_json(stub)["speaker"], list(support.DEFAULT_SPEAKERS), 4) == 1


def test_the_gateway_text_limit_applies_to_magpie_too(monkeypatch):
    _, stub, client = _gateway(monkeypatch, _wav, {**ENABLED, "MAX_TTS_CHARS": "50"})

    assert client.post("/api/tts", json={"provider": "magpie", "text": "a" * 51}).status_code == 422
    assert stub.calls == []


# --- the voice list ----------------------------------------------------------------------

def test_the_five_speakers_reach_the_voice_selector_through_the_speaker_catalog(monkeypatch):
    payload = {
        "speakers": ["Aria", "Jason", "John", "Leo", "Sofia"],
        "languages": ["de", "en", "es", "fr", "ja", "zh"],
        "default_language": "de",
    }
    _, stub, client = _gateway(monkeypatch, _json(payload), ENABLED)

    body = client.get("/api/providers/magpie/voices").json()

    assert stub.calls[-1][1] == "http://magpie-tts-service:5008/speakers"
    assert [v["id"] for v in body["voices"]] == payload["speakers"]
    assert all(v["kind"] == "builtin" for v in body["voices"])
    assert body["default_language"] == "de", "the UI names the service's default in the Automatic option"


def test_the_speaker_catalog_is_exactly_what_the_real_service_answers(monkeypatch):
    """Feed the real /speakers body through the gateway rather than a hand-written stand-in."""
    from magpie_loader import install_nemo, load_app

    install_nemo(monkeypatch)
    service = TestClient(load_app().app)
    real = service.get("/speakers").json()
    _, _, client = _gateway(monkeypatch, _json(real), ENABLED)

    voices = client.get("/api/providers/magpie/voices").json()["voices"]

    assert [v["id"] for v in voices] == ["Aria", "Jason", "John", "Leo", "Sofia"]


# --- failures ----------------------------------------------------------------------------

def test_a_backend_traceback_is_not_forwarded_to_the_caller(monkeypatch):
    traceback = 'Traceback (most recent call last):\n  File "/app/app.py", line 88\nRuntimeError: /root/.cache/x'
    _, _, client = _gateway(monkeypatch, _json({"detail": traceback}, status=500), ENABLED)

    r = client.post("/api/tts", json={"provider": "magpie", "text": "hi"})

    assert r.status_code >= 500
    assert "Traceback" not in r.text and "/root/.cache" not in r.text


def test_the_services_400_for_an_unsupported_language_reaches_the_caller_with_its_list(monkeypatch):
    detail = "Language 'nl' is not supported by this Magpie deployment. Supported: de, en, es, fr, ja, zh."
    _, _, client = _gateway(monkeypatch, _json({"detail": detail}, status=400), ENABLED)

    r = client.post("/api/tts", json={"provider": "magpie", "text": "Hallo", "language": "nl"})

    assert r.status_code == 400
    assert "nl" in r.text and "de, en, es, fr, ja, zh" in r.text


def test_unload_is_proxied(monkeypatch):
    _, stub, client = _gateway(monkeypatch, _json({"model_resident": False, "reason": "unloaded"}), ENABLED)

    r = client.post("/api/providers/magpie/unload")

    assert r.status_code == 200 and r.json()["provider"] == "magpie"
    assert stub.calls[-1][1] == "http://magpie-tts-service:5008/unload"
