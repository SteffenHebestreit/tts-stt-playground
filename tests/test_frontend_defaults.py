"""Language defaults across the gateway and its backends.

* qwen3-tts resolves "auto" and an empty language to QWEN3_DEFAULT_LANGUAGE, but the
  gateway turned both into "English" before the service ever saw them;
* the STT services read an explicit "auto" as "detect" and a missing field as
  STT_DEFAULT_LANGUAGE, but the gateway dropped "auto", so the UI's Auto-Detect upload
  was transcribed in the operator's default language (German in the TrueNAS profile)
  while the live microphone, which sends "auto", detected;
* the UI reads `default_language` from /api/health and /api/providers/{id}/voices to
  say which language "Automatic" means, and the gateway dropped the field;
* Canary's language list was a hardcoded copy of what one checkpoint supports.
"""

import importlib.util
import json
import re
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from frontend_loader import install_stub, load_frontend_app, wav_bytes

REPO = Path(__file__).resolve().parents[1]


def _gateway(monkeypatch, handler, env=None):
    module = load_frontend_app(env)
    stub = install_stub(monkeypatch, module, handler)
    return module, stub, TestClient(module.app)


def _wav_response(method, url, kwargs):
    return httpx.Response(200, content=wav_bytes(), headers={"content-type": "audio/wav"})


def _last_json(stub):
    return stub.calls[-1][2]["json"]


# --- qwen3-tts: "auto" reaches the service ----------------------------------------------


@pytest.mark.parametrize("language", ["auto", "AUTO", "", "  "])
def test_api_tts_sends_auto_to_qwen3_for_an_automatic_language(monkeypatch, language):
    _, stub, client = _gateway(monkeypatch, _wav_response)
    r = client.post("/api/tts", json={"provider": "qwen3", "text": "Guten Tag", "language": language})
    assert r.status_code == 200
    assert _last_json(stub)["lang"] == "auto"


def test_api_tts_omitted_language_is_auto_for_qwen3(monkeypatch):
    _, stub, client = _gateway(monkeypatch, _wav_response)
    client.post("/api/tts", json={"provider": "qwen3", "text": "Guten Tag"})
    assert _last_json(stub)["lang"] == "auto"


def test_v1_speech_sends_auto_to_qwen3_by_default(monkeypatch):
    _, stub, client = _gateway(monkeypatch, _wav_response, {"DEFAULT_TTS_PROVIDER": "qwen3"})
    r = client.post("/v1/audio/speech", json={"input": "Guten Tag", "response_format": "wav"})
    assert r.status_code == 200
    assert _last_json(stub)["lang"] == "auto"
    client.post("/v1/audio/speech", json={"input": "Guten Tag", "response_format": "wav", "language": "de"})
    assert _last_json(stub)["lang"] == "German"


@pytest.mark.parametrize("language,expected", [
    ("de", "German"), ("en", "English"), ("fr", "French"), ("English", "English"), ("Japanese", "Japanese")])
def test_an_explicit_language_is_still_mapped_for_qwen3(monkeypatch, language, expected):
    _, stub, client = _gateway(monkeypatch, _wav_response)
    client.post("/api/tts", json={"provider": "qwen3", "text": "hi", "language": language})
    assert _last_json(stub)["lang"] == expected


@pytest.mark.parametrize("default,resolved", [(None, "German"), ("French", "French"), ("Auto", "Auto")])
def test_what_the_gateway_sends_is_what_qwen3_tts_resolves_to_its_own_default(monkeypatch, default, resolved):
    """The two halves live in different services; feed the gateway's request to the real one."""
    import qwen3_tts_loader

    env = {} if default is None else {"QWEN3_DEFAULT_LANGUAGE": default}
    service = qwen3_tts_loader.load_app(**env)
    _, stub, client = _gateway(monkeypatch, _wav_response)
    client.post("/api/tts", json={"provider": "qwen3", "text": "Guten Tag", "language": "auto"})
    assert service._resolve_language(_last_json(stub)["lang"]) == resolved


# --- STT: "auto" means detect, absence means the service default --------------------------


def _stt_client(monkeypatch, env=None):
    def handler(method, url, kwargs):
        return {"text": "hallo", "segments": [], "language": "de"}

    return _gateway(monkeypatch, handler, env)


def _transcribe(client, provider="whisper", **form):
    return client.post("/api/stt", data={"provider": provider, **form},
                       files={"audio": ("a.wav", wav_bytes(), "audio/wav")})


@pytest.mark.parametrize("provider,env", [
    ("whisper", None), ("qwen3-asr", None), ("parakeet", {"ENABLE_PARAKEET_ASR": "true"}),
    ("canary", {"ENABLE_CANARY_ASR": "true"}),
])
def test_api_stt_forwards_auto_explicitly_to_every_stt_form_backend(monkeypatch, provider, env):
    _, stub, client = _stt_client(monkeypatch, env)
    stub.calls.clear()
    assert _transcribe(client, provider, language="auto").status_code == 200
    posts = [c for c in stub.calls if c[0] == "POST"]
    assert posts[-1][2]["data"] == {"language": "auto"}


def test_api_stt_keeps_an_omitted_language_omitted(monkeypatch):
    _, stub, client = _stt_client(monkeypatch)
    _transcribe(client)
    assert stub.calls[-1][2]["data"] == {}
    _transcribe(client, language="")
    assert stub.calls[-1][2]["data"] == {}
    _transcribe(client, language="de")
    assert stub.calls[-1][2]["data"] == {"language": "de"}


def test_api_stt_sends_auto_to_whisper_cpp_whatever_the_spelling(monkeypatch):
    _, stub, client = _stt_client(monkeypatch, {"ENABLE_WHISPER_CPP": "true"})
    for form in ({"language": "auto"}, {"language": "AUTO"}, {}):
        _transcribe(client, "whisper-cpp", **form)
        assert stub.calls[-1][2]["data"]["language"] == "auto", form


def test_the_real_whisper_service_reads_that_auto_as_detect():
    """stt-service/json_utils.py is the code that interprets what the gateway sends."""
    spec = importlib.util.spec_from_file_location("stt_json_utils", REPO / "stt-service" / "json_utils.py")
    utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(utils)
    assert utils.normalize_language("auto") is None          # detect
    assert utils.normalize_language("de") == "de"


# --- default_language: the UI names what "Automatic" means ------------------------------


def _node_runner():
    """`run_js` from the Node-driven UI tests, or a skip where Node is not installed."""
    from test_frontend_ui_logic import NODE, run_js

    if NODE is None:
        pytest.skip("node is not installed")
    return run_js


def _piper_voices(payload):
    def handler(method, url, kwargs):
        return payload

    return handler


PIPER_VOICES = {"voices": {"de_DE-thorsten-medium": {"name": "thorsten", "language": "de_DE", "quality": "medium"}}}


def test_voice_catalog_forwards_the_default_language(monkeypatch):
    _, _, client = _gateway(monkeypatch, _piper_voices({**PIPER_VOICES, "default_language": "de"}))
    body = client.get("/api/providers/piper/voices").json()
    assert body["default_language"] == "de"
    assert body["voices"][0]["id"] == "de_DE-thorsten-medium"


@pytest.mark.parametrize("reported", [None, "", "   ", 5, ["de"], {"de": 1}, "x" * 65])
def test_voice_catalog_leaves_the_field_out_when_the_backend_reports_nothing_usable(monkeypatch, reported):
    payload = dict(PIPER_VOICES) if reported is None else {**PIPER_VOICES, "default_language": reported}
    _, _, client = _gateway(monkeypatch, _piper_voices(payload))
    assert "default_language" not in client.get("/api/providers/piper/voices").json()


def test_speaker_catalog_forwards_it_when_present(monkeypatch):
    _, _, client = _gateway(monkeypatch, _piper_voices({"speakers": ["Vivian"], "languages": ["German"],
                                                        "default_language": "German"}))
    assert client.get("/api/providers/qwen3/voices").json()["default_language"] == "German"


def test_provider_health_forwards_it_when_the_health_body_has_it(monkeypatch):
    def handler(method, url, kwargs):
        if "chatterbox" in url:
            return {"status": "ok", "default_language": "de", "model_loaded": True}
        return {"status": "healthy"}

    _, _, client = _gateway(monkeypatch, handler, {"ENABLE_CHATTERBOX_TTS": "true"})
    providers = client.get("/api/health").json()["providers"]
    assert providers["chatterbox"]["default_language"] == "de"
    assert "default_language" not in providers["piper"]


def test_the_ui_names_the_reported_default_in_the_automatic_option(monkeypatch, tmp_path):
    """End to end on the UI half: what the gateway answers goes through app.js's own loader,
    which is what turns "Auto-Detect" into "Automatic - server decides (default: German)"."""
    run_js = _node_runner()

    _, _, client = _gateway(monkeypatch, _piper_voices({**PIPER_VOICES, "default_language": "de"}))
    gateway_answer = client.get("/api/providers/piper/voices").json()

    result = run_js(tmp_path, f"""
        harness.fetchQueue.push(harness.json({json.dumps(gateway_answer)}));
        await refreshTTSVoices();
        await harness.flush();
        return {{
            remembered: serverDefaultLanguage.piper,
            label: autoLanguageLabel('piper', 'Auto-Detect'),
        }};
    """)
    assert result["remembered"] == "de"
    assert result["label"] == "Automatic - server decides (default: German)"


def test_without_the_field_the_ui_cannot_name_a_default(monkeypatch, tmp_path):
    """What the gateway used to answer: the label stays the generic one."""
    run_js = _node_runner()

    _, _, client = _gateway(monkeypatch, _piper_voices(dict(PIPER_VOICES)))
    gateway_answer = client.get("/api/providers/piper/voices").json()
    result = run_js(tmp_path, f"""
        harness.fetchQueue.push(harness.json({json.dumps(gateway_answer)}));
        await refreshTTSVoices();
        await harness.flush();
        return autoLanguageLabel('piper', 'Auto-Detect');
    """)
    assert result == "Automatic - server decides"


# --- Canary: the language list comes from the service ---------------------------------------

CANARY_25 = ["bg", "cs", "da", "de", "el", "en", "es", "et", "fi", "fr", "hr", "hu", "it", "lt", "lv",
             "mt", "nl", "pl", "pt", "ro", "ru", "sk", "sl", "sv", "uk"]
FALLBACK = ["de", "en", "fr", "es"]


class _Canary:
    """A canary service whose /status the test controls, with a clock the test moves."""

    def __init__(self, monkeypatch, env=None, status=None):
        self.status = status
        self.now = 1000.0
        self.asked = 0

        def handler(method, url, kwargs):
            if url.startswith("http://canary-asr-service:5006") and url.endswith("/status"):
                self.asked += 1
                outcome = self.status
                if isinstance(outcome, Exception):
                    raise outcome
                if isinstance(outcome, httpx.Response):
                    return outcome
                return outcome
            return {"status": "healthy"}

        self.module = load_frontend_app({"ENABLE_CANARY_ASR": "true", **(env or {})})
        self.stub = install_stub(monkeypatch, self.module, handler)
        # raising=False: the clock is only read by the discovery code, which is what these tests
        # are about; against a gateway without it they must fail on the languages, not here.
        monkeypatch.setattr(self.module, "_language_clock", lambda: self.now, raising=False)
        self.client = TestClient(self.module.app)

    def languages(self, path="/providers"):
        return [item["value"] for item in self.registry(path)["providers"]["canary"]["settings"]["languages"]]

    def registry(self, path="/providers"):
        if path == "/providers":
            return self.client.get("/providers").json()
        page = self.client.get("/").text
        blob = re.search(r'<script[^>]*id="provider-registry-data"[^>]*>(.*?)</script>', page, re.S).group(1)
        return json.loads(blob)


def _status(languages, default="de", model="nvidia/canary-1b-v2"):
    return {"status": "ok", "supported_languages": languages, "default_language": default, "current_model": model}


def test_canary_languages_come_from_what_the_service_reports(monkeypatch):
    canary = _Canary(monkeypatch, status=_status(CANARY_25))
    provider = canary.registry()["providers"]["canary"]
    values = [item["value"] for item in provider["settings"]["languages"]]
    assert sorted(values) == sorted(CANARY_25)
    assert values[0] == "de", "the service's default leads the list, as the fallback list always did"
    labels = {item["value"]: item["label"] for item in provider["settings"]["languages"]}
    assert labels["de"] == "German" and labels["uk"] == "Ukrainian" and labels["pl"] == "Polish"
    assert provider["settings"]["defaults"]["language"] == "de"
    assert provider["display_name"] == "Canary (canary-1b-v2, 25 languages)"


def test_the_page_carries_the_discovered_languages_and_the_status_is_read_once(monkeypatch):
    canary = _Canary(monkeypatch, status=_status(CANARY_25))
    embedded = canary.registry("/")
    assert len(embedded["providers"]["canary"]["settings"]["languages"]) == 25
    canary.registry("/")
    canary.registry("/providers")
    assert canary.asked == 1, "the list is cached, not re-read on every page load"


def test_a_flash_model_reports_four_languages_and_is_named_in_the_selector(monkeypatch):
    canary = _Canary(monkeypatch, status=_status(["es", "fr", "en", "de"], model="nvidia/canary-180m-flash"))
    assert canary.languages() == ["de", "en", "fr", "es"]        # the default, then by name
    assert canary.registry()["providers"]["canary"]["display_name"] == "Canary (canary-180m-flash, de/en/fr/es)"
    page = canary.client.get("/").text
    assert "Canary (canary-180m-flash, de/en/fr/es)" in page, "the STT engine dropdown is rendered server side"


def test_the_services_default_language_is_the_preselected_one(monkeypatch):
    canary = _Canary(monkeypatch, status=_status(["de", "en", "es", "fr"], default="fr"))
    provider = canary.registry()["providers"]["canary"]
    assert provider["settings"]["defaults"]["language"] == "fr"
    assert provider["settings"]["languages"][0]["value"] == "fr"


def test_a_default_the_service_does_not_list_falls_back_to_a_listed_language(monkeypatch):
    canary = _Canary(monkeypatch, status=_status(["en", "es"], default="de"))
    provider = canary.registry()["providers"]["canary"]
    default = provider["settings"]["defaults"]["language"]
    assert default in ("en", "es")
    assert provider["settings"]["languages"][0]["value"] == default


def test_the_hardcoded_list_is_only_a_fallback_for_an_unreachable_service(monkeypatch):
    canary = _Canary(monkeypatch, status=httpx.ConnectError("refused"))
    assert canary.languages() == FALLBACK
    assert canary.registry()["providers"]["canary"]["display_name"] == "Canary-180M (realtime, en/de/es/fr)"


@pytest.mark.parametrize("outcome", [
    httpx.Response(500, json={"detail": "boom"}),
    httpx.Response(200, content=b"<html>not json</html>"),
    {"supported_languages": []},
    {"supported_languages": "de,en"},
    {"supported_languages": [None, 5, "", "german", "d3"]},
    {"no": "languages"},
    ["de", "en"],
], ids=["500", "html", "empty", "string", "junk-codes", "missing", "list"])
def test_an_unusable_answer_keeps_the_fallback(monkeypatch, outcome):
    canary = _Canary(monkeypatch, status=outcome)
    assert canary.languages() == FALLBACK


def test_a_dead_canary_is_not_asked_again_until_the_retry_interval_has_passed(monkeypatch):
    canary = _Canary(monkeypatch, status=httpx.ConnectError("refused"))
    for _ in range(4):
        canary.languages()
    assert canary.asked == 1, "every page load would pay a connect attempt"

    canary.now += 21
    canary.status = _status(CANARY_25)
    assert len(canary.languages()) == 25
    assert canary.asked == 2


def test_a_discovered_list_is_refreshed_after_five_minutes_and_survives_a_failed_refresh(monkeypatch):
    canary = _Canary(monkeypatch, status=_status(["de", "en"], model="nvidia/canary-180m-flash"))
    assert canary.languages() == ["de", "en"]
    canary.now += 299
    canary.status = _status(CANARY_25)
    assert canary.languages() == ["de", "en"] and canary.asked == 1

    canary.now += 2                       # 301 s
    canary.status = httpx.ConnectError("restarting")
    assert canary.languages() == ["de", "en"], "the last good answer beats the built-in list"
    assert canary.asked == 2

    canary.now += 21
    canary.status = _status(CANARY_25)
    assert len(canary.languages()) == 25


def test_an_operator_supplied_canary_entry_is_left_alone(monkeypatch):
    override = {"providers": {"canary": {
        "kind": "stt", "display_name": "My Canary", "internal_url": "http://canary-asr-service:5006",
        "settings": {"languages": [{"value": "de", "label": "Deutsch"}], "defaults": {"language": "de"}}}}}
    canary = _Canary(monkeypatch, env={"PROVIDER_REGISTRY_JSON": json.dumps(override)}, status=_status(CANARY_25))
    provider = canary.registry()["providers"]["canary"]
    assert provider["settings"]["languages"] == [{"value": "de", "label": "Deutsch"}]
    assert provider["display_name"] == "My Canary"
    assert canary.asked == 0


def test_nothing_is_asked_when_canary_is_not_enabled(monkeypatch):
    module = load_frontend_app()
    stub = install_stub(monkeypatch, module, lambda m, u, k: {"status": "healthy"})
    client = TestClient(module.app)
    client.get("/")
    client.get("/providers")
    assert stub.calls == []


def test_the_real_canary_service_status_is_what_the_gateway_understands(monkeypatch):
    """Two services, one shape: run the gateway against the real /status of each supported model."""
    from test_nemo_services_harness import load_app, make_client, run

    # Asked before the gateway is loaded: its stub replaces httpx.AsyncClient for the whole process.
    statuses = []
    for model, expected in (("nvidia/canary-180m-flash", 4), ("nvidia/canary-1b-v2", 25)):
        service = load_app("canary", CANARY_ASR_MODEL=model)

        async def fetch():
            async with make_client(service) as client:
                return (await client.get("/status")).json()

        statuses.append((run(fetch()), expected))

    for reported, expected in statuses:
        canary = _Canary(monkeypatch, status=reported)
        assert sorted(canary.languages()) == sorted(reported["supported_languages"])
        assert len(canary.languages()) == expected
        assert canary.languages()[0] == reported["default_language"]
