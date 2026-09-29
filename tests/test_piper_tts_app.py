"""Behaviour of piper-tts-service: language defaults, voice discovery, readiness.

Each test drives the real FastAPI app over HTTP with a stand-in `piper` binary
(see test_piper_tts_harness.py), so voice choice is observed the way a client
sees it: which voice's model the binary was asked to load and what the response
headers claim.
"""

from __future__ import annotations

import json
import os
import time

import pytest

from test_piper_tts_harness import install_voice, piper_config, start  # noqa: F401  (fixture)

THORSTEN = "de_DE-thorsten-medium"
EVA = "de_DE-eva_k-x_low"
LESSAC = "en_US-lessac-medium"
ALAN = "en_GB-alan-medium"

GERMAN_TEXT = "Guten Tag, das ist ein Test."
ENGLISH_TEXT = "Good morning, this is a test of the system."


# --- P1: a request that names no language must not be read by an English voice ---------

def test_request_without_language_or_voice_uses_the_german_default(start):
    svc = start()

    r = svc.speak(text="Hello world")

    assert r.status_code == 200
    assert r.headers["X-Voice-Used"] == THORSTEN
    assert r.headers["X-Language"] == "de_DE"
    assert r.headers["X-Language-Fallback"] == "false"
    assert svc.model_used() == THORSTEN


@pytest.mark.parametrize("language", ["auto", "AUTO", " auto ", ""])
def test_auto_language_means_the_service_decides(start, language):
    svc = start()

    r = svc.speak(text="Hello world", language=language)

    assert r.headers["X-Voice-Used"] == THORSTEN
    assert r.headers["X-Language-Requested"] == "auto"


def test_default_language_is_configurable(start):
    svc = start(PIPER_DEFAULT_LANGUAGE="en")

    assert svc.speak(text="Hello world").headers["X-Voice-Used"] == LESSAC


def test_auto_guesses_english_text_when_an_english_voice_is_installed(start):
    svc = start()

    assert svc.speak(text=ENGLISH_TEXT).headers["X-Voice-Used"] == LESSAC
    assert svc.speak(text=ENGLISH_TEXT, language="auto").headers["X-Voice-Used"] == LESSAC


def test_auto_guesses_german_even_when_english_is_the_default(start):
    svc = start(PIPER_DEFAULT_LANGUAGE="en")

    r = svc.speak(text="Schöne Grüße aus Köln")

    assert r.headers["X-Voice-Used"] == THORSTEN
    assert r.headers["X-Language-Fallback"] == "false"


def test_auto_detection_can_be_switched_off(start):
    svc = start(PIPER_AUTO_DETECT="false")

    assert svc.speak(text=ENGLISH_TEXT).headers["X-Voice-Used"] == THORSTEN


def test_a_guess_without_a_matching_voice_is_not_reported_as_a_fallback(start):
    """The client never asked for English; a German-only install just uses German."""
    svc = start(voices=(THORSTEN,))

    r = svc.speak(text=ENGLISH_TEXT)

    assert r.status_code == 200
    assert r.headers["X-Voice-Used"] == THORSTEN
    assert r.headers["X-Language-Fallback"] == "false"


def test_an_explicit_language_wins_over_the_text(start):
    svc = start()

    r = svc.speak(text=GERMAN_TEXT, language="en")

    assert r.headers["X-Voice-Used"] == LESSAC
    assert r.headers["X-Language-Fallback"] == "false"
    assert r.headers["X-Language-Requested"] == "en"


@pytest.mark.parametrize("tag", ["de-DE", "de_DE", "de", "DE"])
def test_language_tag_spellings_all_select_german(start, tag):
    """Browsers and OpenAI clients send 'de-DE'; it used to match no voice at all."""
    svc = start()

    r = svc.speak(text="Hello", language=tag)

    assert r.headers["X-Voice-Used"] == THORSTEN
    assert r.headers["X-Language-Fallback"] == "false"


def test_a_regional_tag_prefers_the_regional_voice(start):
    svc = start(voices=(LESSAC, ALAN, THORSTEN))

    assert svc.speak(text="Hello", language="en_GB").headers["X-Voice-Used"] == ALAN
    assert svc.speak(text="Hello", language="en-US").headers["X-Voice-Used"] == LESSAC


def test_missing_language_is_reported_as_a_fallback(start):
    svc = start(voices=(LESSAC,))

    r = svc.speak(text=GERMAN_TEXT, language="de")

    assert r.status_code == 200
    assert r.headers["X-Voice-Used"] == LESSAC
    assert r.headers["X-Language"] == "en_US"
    assert r.headers["X-Language-Fallback"] == "true"


def test_missing_default_language_is_reported_even_when_none_was_requested(start):
    """No language, no German voice installed: English audio for German text, and the header says so."""
    svc = start(voices=(LESSAC,))

    r = svc.speak(text="Hallo")

    assert r.status_code == 200
    assert r.headers["X-Language"] == "en_US"
    assert r.headers["X-Language-Fallback"] == "true"


def test_strict_mode_refuses_instead_of_substituting(start):
    svc = start(voices=(LESSAC,), PIPER_STRICT_LANGUAGE="true")

    for body in ({"language": "de"}, {}):
        r = svc.speak(text=GERMAN_TEXT, **body)
        assert r.status_code == 400
        assert "en_US" in r.json()["detail"]
    assert svc.fake.calls() == []


def test_a_pinned_voice_is_used_and_a_conflicting_language_is_flagged(start):
    svc = start()

    plain = svc.speak(voice=LESSAC)
    assert plain.headers["X-Voice-Used"] == LESSAC
    assert plain.headers["X-Language-Fallback"] == "false"

    conflicting = svc.speak(voice=LESSAC, language="de")
    assert conflicting.headers["X-Voice-Used"] == LESSAC
    assert conflicting.headers["X-Language-Fallback"] == "true"


def test_an_unknown_voice_falls_back_to_the_language_choice(start):
    svc = start()

    r = svc.speak(voice="de_DE-nobody-high")

    assert r.status_code == 200
    assert r.headers["X-Voice-Used"] == THORSTEN


def test_a_language_that_is_not_ascii_cannot_break_the_response_headers(start):
    svc = start()

    r = svc.speak(text="Hello", language="日本語")

    assert r.status_code == 200
    assert r.headers["X-Language-Requested"] == "invalid"


# --- PIPER_DEFAULT_VOICE ----------------------------------------------------------

def test_default_voice_is_used_when_nothing_more_specific_is_asked(start):
    svc = start(voices=(THORSTEN, EVA), PIPER_DEFAULT_VOICE=EVA)

    r = svc.speak(text="Hello world")

    assert r.headers["X-Voice-Used"] == EVA
    assert r.headers["X-Quality"] == "x_low"


def test_default_voice_beats_list_order(start):
    svc = start(voices=(LESSAC, "en_US-amy-medium"), PIPER_DEFAULT_VOICE=LESSAC)

    assert svc.speak(text="Hello", language="en").headers["X-Voice-Used"] == LESSAC
    assert svc.speak(text="Hello", language="en", quality="medium").headers["X-Voice-Used"] == LESSAC


def test_default_voice_applies_only_to_its_own_language(start):
    svc = start(voices=(THORSTEN, EVA, LESSAC), PIPER_DEFAULT_VOICE=EVA)

    assert svc.speak(text="Hello", language="en").headers["X-Voice-Used"] == LESSAC


def test_an_explicit_quality_outranks_the_default_voice(start):
    """The UI always sends quality; asking for medium must still find a medium voice."""
    svc = start(voices=(THORSTEN, EVA), PIPER_DEFAULT_VOICE=EVA)

    assert svc.speak(text="Hello", quality="medium").headers["X-Voice-Used"] == THORSTEN
    assert svc.speak(text="Hello", quality="x_low").headers["X-Voice-Used"] == EVA


def test_the_ui_sending_gender_any_does_not_defeat_the_default_voice(start):
    svc = start(voices=(THORSTEN, EVA), PIPER_DEFAULT_VOICE=EVA)

    r = svc.speak(text="Hello", quality="x_low", gender="any")

    assert r.headers["X-Voice-Used"] == EVA


def test_a_default_voice_that_is_not_installed_is_ignored_and_reported(start):
    svc = start(PIPER_DEFAULT_VOICE="de_DE-nobody-high")

    assert svc.speak(text="Hello").headers["X-Voice-Used"] == THORSTEN
    warnings = svc.client.get("/ready").json()["warnings"]
    assert any("PIPER_DEFAULT_VOICE" in w for w in warnings)


# --- P2: installed voices are what is on disk -------------------------------------------

def test_a_voice_pair_in_the_volume_is_registered_at_startup(start):
    """A voice that is in no Python table, dropped into the models volume."""
    svc = start(voices=(THORSTEN, "pt_BR-faber-medium"))

    voices = svc.client.get("/voices").json()
    assert "pt_BR-faber-medium" in voices["voices"]
    assert voices["voices"]["pt_BR-faber-medium"]["language"] == "pt_BR"
    assert voices["voices"]["pt_BR-faber-medium"]["speaker"] == "faber"

    r = svc.speak(text="Olá", language="pt")
    assert r.headers["X-Voice-Used"] == "pt_BR-faber-medium"
    assert r.headers["X-Language-Fallback"] == "false"


def test_a_voice_that_is_not_installed_is_not_listed(start):
    svc = start(voices=(THORSTEN,))

    listed = svc.client.get("/voices").json()

    assert set(listed["voices"]) == {THORSTEN}
    assert listed["catalog_only"] is False
    assert listed["default_count"] == 1


def test_a_voice_dropped_in_after_startup_registers_itself(start):
    svc = start(voices=(THORSTEN,))
    assert LESSAC not in svc.client.get("/voices").json()["voices"]

    install_voice(svc.models, LESSAC)
    later = time.time() + 5  # a directory mtime that is guaranteed to differ from the scan's
    os.utime(svc.models / "default", (later, later))

    assert LESSAC in svc.client.get("/voices").json()["voices"]
    assert svc.speak(text=ENGLISH_TEXT).headers["X-Voice-Used"] == LESSAC


def test_refresh_voices_rereads_a_changed_config(start):
    svc = start(voices=(THORSTEN,))
    config = svc.models / "default" / f"{THORSTEN}.onnx.json"
    config.write_text(json.dumps(piper_config("de_AT", "high", "thorsten")))  # edited in place

    r = svc.client.post("/refresh_voices")

    assert r.status_code == 200
    assert r.json()["default_voices"] == 1
    voice = svc.client.get(f"/voice/{THORSTEN}").json()
    assert (voice["language"], voice["quality"]) == ("de_AT", "high")


def test_half_copied_pairs_and_broken_configs_are_not_registered(start):
    svc = start(voices=(THORSTEN,))
    default = svc.models / "default"
    (default / "fr_FR-siwis-medium.onnx").write_bytes(b"x")                 # model without config
    (default / "it_IT-riccardo-x_low.onnx.json").write_text("{}")           # config without model
    (default / "es_ES-mls-low.onnx").write_bytes(b"x")
    (default / "es_ES-mls-low.onnx.json").write_text("{not json")           # unreadable config
    (default / "nl_NL-mls_5809-low.onnx").write_bytes(b"x")
    (default / "nl_NL-mls_5809-low.onnx.json").write_text("[1, 2]")         # not an object
    (default / "bad.name.onnx").write_bytes(b"x")
    (default / "bad.name.onnx.json").write_text("{}")                       # unsafe id

    assert svc.client.post("/refresh_voices").json()["default_voices"] == 1
    assert set(svc.client.get("/voices").json()["voices"]) == {THORSTEN}


def test_a_config_without_language_falls_back_to_espeak_then_the_file_name(start):
    svc = start(voices=())
    install_voice(svc.models, "sv_SE-nst-medium", {"espeak": {"voice": "sv"}, "audio": {"sample_rate": 16000}})
    install_voice(svc.models, "cs_CZ-jirka-low", {})

    svc.client.post("/refresh_voices")
    voices = svc.client.get("/voices").json()["voices"]

    assert voices["sv_SE-nst-medium"]["language"] == "sv"
    assert voices["sv_SE-nst-medium"]["sample_rate"] == 16000
    assert voices["cs_CZ-jirka-low"]["language"] == "cs_CZ"
    assert voices["cs_CZ-jirka-low"]["quality"] == "low"


def test_an_empty_models_volume_lists_the_catalog_but_is_not_ready(start):
    """A bind-mounted models volume hides what the image downloaded."""
    svc = start(voices=())

    listed = svc.client.get("/voices").json()
    assert listed["catalog_only"] is True
    assert THORSTEN in listed["voices"]

    ready = svc.client.get("/ready")
    assert ready.status_code == 503
    assert ready.json()["status"] == "not_ready"
    assert any("no voice models" in p for p in ready.json()["problems"])

    r = svc.speak()
    assert r.status_code == 404
    assert THORSTEN in r.json()["detail"]
    assert svc.client.get("/health").status_code == 200  # liveness is unaffected


# --- /ready -----------------------------------------------------------------------------

def test_ready_reports_the_installed_voices(start):
    svc = start()

    r = svc.client.get("/ready")

    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ready"
    assert body["default_voices"] == 2
    assert body["default_language"] == "de"
    assert body["problems"] == []


def test_ready_warns_when_the_default_language_has_no_voice(start):
    svc = start(voices=(LESSAC,))

    body = svc.client.get("/ready").json()

    assert body["status"] == "ready"
    assert any("PIPER_DEFAULT_LANGUAGE=de" in w for w in body["warnings"])


def test_ready_is_503_without_the_piper_binary(start, monkeypatch):
    svc = start()
    monkeypatch.setenv("PATH", "/nonexistent")

    r = svc.client.get("/ready")

    assert r.status_code == 503
    assert any("piper binary" in p for p in r.json()["problems"])


def test_voices_reports_the_configured_defaults(start):
    svc = start(PIPER_DEFAULT_LANGUAGE="en", PIPER_DEFAULT_VOICE=LESSAC)

    body = svc.client.get("/voices").json()

    assert body["default_language"] == "en"
    assert body["default_voice"] == LESSAC


# --- request validation -----------------------------------------------------------------

def test_text_over_the_limit_is_rejected(start):
    svc = start(MAX_TEXT_CHARS="50")

    assert svc.speak(text="x" * 50).status_code == 200
    assert svc.speak(text="x" * 51).status_code == 422


def test_a_json_body_far_over_the_limit_is_refused_before_it_is_read(start):
    svc = start(MAX_TEXT_CHARS="50")

    r = svc.client.post("/tts", content=b'{"text": "' + b"x" * 500_000 + b'"}',
                        headers={"content-type": "application/json"})

    assert r.status_code == 413


def test_multiline_text_reaches_piper_as_one_line(start):
    svc = start()

    svc.speak(text="Erste Zeile.\n\nZweite   Zeile.")

    assert svc.fake.calls()[-1][1] == "Erste Zeile. Zweite Zeile."


def test_an_invalid_voice_name_with_a_trailing_newline_is_rejected(start):
    svc = start()

    assert svc.client.delete("/voice/abc%0A").status_code == 400
    assert svc.client.post("/synthesize", json={"text": "Hi", "voice_name": "abc\n"}).status_code == 400
