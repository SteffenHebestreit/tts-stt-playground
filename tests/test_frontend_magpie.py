"""The gateway's side of the Magpie TTS provider.

* the provider exists only when ENABLE_MAGPIE_TTS is set, like the other optional backends;
* the request body is Magpie's (`text`, `language`, `speaker`), with "auto" left for the
  service to resolve: a gateway that filled in a language or a speaker itself would
  override MAGPIE_DEFAULT_LANGUAGE / MAGPIE_DEFAULT_SPEAKER on every request;
* the five speakers reach the voice selector through the existing `speaker-catalog-v1`
  contract; the panel lists them by name and shows Magpie none of Piper's other controls
  (quality, gender, speed, custom voices), which app.js reads from this registry
  (test_frontend_ui_logic.py runs that half);
* /tts answers only when the whole text is done, so the gateway's read budget covers the
  whole job, and a caller who hangs up has the backend call cancelled.
"""

import asyncio
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


def _language_options(module):
    return [item["value"] for item in module.PROVIDER_REGISTRY["providers"]["magpie"]["settings"]["languages"]]


def test_the_language_options_are_only_ones_the_service_can_speak(monkeypatch):
    """The service refuses any other language with a 400 (NeMo 3.0.x would read it with
    English rules), so offering one would only lead the user into that error. Compared
    with the service's own list rather than a literal, so the two cannot drift apart in
    either direction."""
    module, _, _ = _gateway(monkeypatch, _wav, ENABLED)

    options = _language_options(module)

    assert options[0] == "auto"
    assert options[1:] == list(magpie_loader.load_support().DOCUMENTED_LANGUAGES)
    assert module.PROVIDER_REGISTRY["providers"]["magpie"]["settings"]["defaults"]["language"] == "auto"


def test_the_language_options_are_what_the_service_lists_before_a_load(monkeypatch):
    magpie_loader.install_nemo(monkeypatch)
    listed = TestClient(magpie_loader.load_app().app).get("/languages").json()["languages"]
    module, _, _ = _gateway(monkeypatch, _wav, ENABLED)

    assert _language_options(module)[1:] == listed


def test_the_automatic_language_has_no_label_of_its_own(monkeypatch):
    """app.js keeps a label the registry chose; without one it names the default the
    service reports ("Automatic - server decides (default: German)")."""
    module, _, _ = _gateway(monkeypatch, _wav, ENABLED)

    assert module.PROVIDER_REGISTRY["providers"]["magpie"]["settings"]["languages"][0] == {"value": "auto"}


def test_the_automatic_voice_is_called_what_it_is(monkeypatch):
    """Magpie's "auto" is MAGPIE_DEFAULT_SPEAKER, one fixed voice; "Auto-Select Best Voice"
    promised a choice made for the text, which is what Piper's "auto" is."""
    module, _, _ = _gateway(monkeypatch, _wav, ENABLED)
    providers = module.PROVIDER_REGISTRY["providers"]

    magpie = providers["magpie"]["ui"]["messages"]["tts_generation"]
    piper = providers["piper"]["ui"]["messages"]["tts_generation"]

    assert magpie["voice_auto_option"] == "Service default voice"
    assert piper["voice_auto_option"] == "Auto-Select Best Voice"
    assert {key: value for key, value in magpie.items() if key != "voice_auto_option"} == {
        key: value for key, value in piper.items() if key != "voice_auto_option"}


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
    assert body["default_language"] == "de", (
        "the gateway must pass on the language 'auto' means; app.js names it in the Automatic option")


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


# --- how long the gateway waits for Magpie ----------------------------------------------
#
# Magpie has no streaming route: /tts answers once every group of sentences is done, so the
# gateway's read timeout is the budget for the whole job. It was a flat 300 s, which a long
# German text overran on the GPU the gateway then reported as "unavailable". Pinned here so
# nobody aligns it back with Chatterbox's 300 s (the gap between two streamed chunks).

GERMAN = ("Das ist ein ganz gewöhnlicher Satz. " * 200)[:5000]
CHINESE = ("这是一个很普通的句子。" * 600)[:5000]


def _read_timeout(stub):
    return stub.calls[-1][2]["timeout"].read


@pytest.mark.parametrize("text,language,expected", [
    ("Guten Tag.", "de", 600.0),          # never below the 600 s /v1 always gave it
    (GERMAN, "de", 680.0),                # 180 s + 0.1 s per character
    (CHINESE, "zh", 1430.0),              # 180 s + 0.25 s per character
    (CHINESE, "zh-CN", 1430.0),
    (CHINESE, "Chinese", 1430.0),
], ids=["short", "de-5000", "zh-5000", "zh-CN-5000", "Chinese-5000"])
def test_the_read_budget_covers_the_whole_text(monkeypatch, text, language, expected):
    _, stub, client = _gateway(monkeypatch, _wav, ENABLED)

    r = client.post("/api/tts", json={"provider": "magpie", "text": text, "language": language})

    assert r.status_code == 200
    assert len(text) in (len("Guten Tag."), 5000)
    assert _read_timeout(stub) == pytest.approx(expected)


def test_v1_waits_exactly_as_long_as_api_tts(monkeypatch):
    text = CHINESE[:4000]                 # /v1 takes at most 4096 characters
    _, stub, client = _gateway(monkeypatch, _wav, {**ENABLED, "DEFAULT_TTS_PROVIDER": "magpie"})

    assert client.post("/v1/audio/speech", json={
        "input": text, "language": "zh", "response_format": "wav"}).status_code == 200
    via_v1 = _read_timeout(stub)
    client.post("/api/tts", json={"provider": "magpie", "text": text, "language": "zh"})

    assert via_v1 == _read_timeout(stub) == pytest.approx(180.0 + 0.25 * 4000)


def test_a_read_timeout_says_magpie_did_not_finish_not_that_it_is_down(monkeypatch):
    def slow(method, url, kwargs):
        request = httpx.Request(method, url, extensions={"timeout": kwargs["timeout"].as_dict()})
        raise httpx.ReadTimeout("timed out", request=request)

    _, _, client = _gateway(monkeypatch, slow, ENABLED)

    r = client.post("/api/tts", json={"provider": "magpie", "text": "Guten Tag.", "language": "de"})

    assert r.status_code == 503
    assert "did not finish in time (read timeout 600 s)" in r.json()["detail"]
    assert "unavailable" not in r.json()["detail"]


# --- a caller who hangs up ------------------------------------------------------------------
#
# uvicorn never cancels a handler whose client went away; it only answers the next receive()
# with http.disconnect. So these drive the ASGI app the way a server does: the body, then a
# hang-up while the backend call is in flight. (TestClient sends its disconnect only after
# the response, and frontend_loader.asgi_call sends it at once.)


def _scope(path: str, body: bytes) -> dict:
    return {
        "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": "POST",
        "scheme": "http", "path": path, "raw_path": path.encode(), "query_string": b"", "root_path": "",
        "headers": [(b"host", b"testserver"), (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode())],
        "client": ("127.0.0.1", 50000), "server": ("testserver", 80),
    }


def _hang_up_while_magpie_works(monkeypatch, path, payload, env):
    """POST *payload*, hang up once the gateway waits for Magpie; returns what happened."""
    started, cancelled = [], []

    async def magpie_still_generating(method, url, kwargs):
        started.append(url)
        try:
            await asyncio.Event().wait()           # never answers on its own
        except asyncio.CancelledError:
            cancelled.append(url)
            raise

    module, _, _ = _gateway(monkeypatch, magpie_still_generating, env)
    body = json.dumps(payload).encode()

    async def main():
        hang_up = asyncio.Event()
        delivered = False
        sent = []

        async def receive():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": body, "more_body": False}
            await hang_up.wait()
            return {"type": "http.disconnect"}

        async def send(message):
            sent.append(message)

        handler = asyncio.create_task(module.app(_scope(path, body), receive, send))
        for _ in range(500):
            if started:
                break
            await asyncio.sleep(0.01)
        answered_before_the_hang_up = list(sent)
        hang_up.set()
        await asyncio.wait_for(handler, 5)
        return answered_before_the_hang_up, sent

    before, sent = asyncio.run(main())
    status = next(m["status"] for m in sent if m["type"] == "http.response.start")
    return started, cancelled, before, status


@pytest.mark.parametrize("path,payload,env", [
    ("/api/tts", {"provider": "magpie", "text": "Ein langer Text.", "language": "de"}, ENABLED),
    ("/v1/audio/speech", {"input": "Ein langer Text.", "response_format": "wav"},
     {**ENABLED, "DEFAULT_TTS_PROVIDER": "magpie"}),
], ids=["api-tts", "v1-speech"])
def test_a_caller_who_hangs_up_has_the_magpie_call_cancelled(monkeypatch, path, payload, env):
    started, cancelled, before, status = _hang_up_while_magpie_works(monkeypatch, path, payload, env)

    assert started == ["http://magpie-tts-service:5008/tts"]
    assert before == [], "the gateway answered before Magpie did"
    assert cancelled == started, "the request to Magpie was left running for a caller who had gone"
    assert status == 499
