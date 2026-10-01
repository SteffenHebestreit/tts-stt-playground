"""magpie-tts-service end to end: HTTP behaviour, validation, grouping, failure hygiene, admission, cancellation.

The model is a fake that follows the measured behaviour of the real one (see
magpie_loader.py); what is under test is the service around it. Every refusal must happen
BEFORE the model is touched (`package.loads == 0`), because loading costs seconds and
~4 GB of VRAM.
"""

import asyncio
import json
import socket
import threading
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from magpie_loader import (
    SAMPLE_RATE, SAMPLES_PER_CHAR, decode_audio, install_nemo, load_app, speakers_in, wait_until, wait_until_async,
)
from test_piper_tts_harness import asgi_post_json

TEXT = "Hallo Welt."

# One Chinese sentence of 173 characters whose clauses are joined by "，" alone: NeMo 3.0.0
# would not split it (no 。？！… inside), so one do_tts call would stop it at ~23 s.
LONG_COMMA_SENTENCE = "，".join([
    "随着城市化进程的不断加快", "越来越多的年轻人离开家乡来到大城市工作和生活", "他们在追求更好发展机会的同时",
    "也面临着住房成本高涨和通勤时间过长等诸多现实问题", "而这些问题如果长期得不到有效解决",
    "不仅会影响个人的身心健康和生活质量", "还可能对整个社会的稳定与可持续发展产生深远的负面影响",
    "因此政府和企业以及社会各界都应当共同努力", "积极探索切实可行的解决方案",
]) + "。"


def _tts(client, text=TEXT, **body):
    return client.post("/tts", json={"text": text, "language": "de", **body})


def _real(package):
    """The generations a request caused: the load's own warm-up generation is the first call."""
    return package.calls[1:]


def _warm(app):
    with app._model_slot.acquire():
        pass


# --- The happy path --------------------------------------------------------------------

def test_speech_comes_back_as_wav_from_the_requested_speaker_and_language(monkeypatch):
    package = install_nemo(monkeypatch)
    client = TestClient(load_app().app)

    response = _tts(client, speaker="Leo")

    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"
    assert response.headers["X-Sample-Rate"] == str(SAMPLE_RATE)
    assert response.headers["X-Language"] == "de" and response.headers["X-Speaker"] == "Leo"
    assert response.headers["X-Chunk-Count"] == "1"
    audio = decode_audio(response.content)
    assert audio.size == len(TEXT) * SAMPLES_PER_CHAR
    assert speakers_in(audio) == {3}
    [call] = _real(package)
    assert call["text"] == TEXT and call["language"] == "de" and call["speaker_index"] == 3


def test_the_model_is_asked_for_the_normalizer_and_guidance_by_default(monkeypatch):
    """Without text normalization the real tokenizer drops every digit; with it, they are spoken."""
    package = install_nemo(monkeypatch)
    client = TestClient(load_app().app)

    _tts(client, "Am 3. Mai 2026 kostet es 12,50 Euro.")

    [call] = _real(package)
    assert call["apply_TN"] is True and call["use_cfg"] is True


def test_normalization_and_guidance_can_be_switched_off(monkeypatch):
    package = install_nemo(monkeypatch)
    client = TestClient(load_app(MAGPIE_APPLY_TN="false", MAGPIE_USE_CFG="false").app)

    _tts(client)

    [call] = _real(package)
    assert call["apply_TN"] is False and call["use_cfg"] is False


def test_a_text_with_nothing_speakable_is_a_400_not_an_empty_wav(monkeypatch):
    """With normalization off the real tokenizer drops digits, so a digits-only text yields no audio at all."""
    install_nemo(monkeypatch)
    client = TestClient(load_app(MAGPIE_APPLY_TN="false").app)

    response = _tts(client, "2026")

    assert response.status_code == 400 and "no audio" in response.json()["detail"]


def test_the_same_digits_are_spoken_with_normalization_on(monkeypatch):
    install_nemo(monkeypatch)
    client = TestClient(load_app().app)

    response = _tts(client, "2026")

    assert response.status_code == 200 and decode_audio(response.content).size == 4 * SAMPLES_PER_CHAR


# --- Defaults --------------------------------------------------------------------------

def test_blank_language_and_speaker_use_the_configured_defaults(monkeypatch):
    package = install_nemo(monkeypatch)
    client = TestClient(load_app(MAGPIE_DEFAULT_LANGUAGE="en", MAGPIE_DEFAULT_SPEAKER="Jason").app)

    response = client.post("/tts", json={"text": TEXT})

    assert response.status_code == 200
    assert response.headers["X-Language"] == "en" and response.headers["X-Speaker"] == "Jason"
    [call] = _real(package)
    assert call["language"] == "en" and call["speaker_index"] == 1


def test_the_stock_defaults_are_german_and_sofia(monkeypatch):
    package = install_nemo(monkeypatch)
    client = TestClient(load_app().app)

    response = client.post("/tts", json={"text": TEXT, "language": "auto"})

    assert response.headers["X-Language"] == "de" and response.headers["X-Speaker"] == "Sofia"
    assert _real(package)[0]["speaker_index"] == 4


@pytest.mark.parametrize("speaker, index", [("sofia", 4), (0, 0), ("2", 2), (3, 3)])
def test_a_speaker_is_a_name_in_any_case_or_an_index(monkeypatch, speaker, index):
    package = install_nemo(monkeypatch)
    client = TestClient(load_app().app)

    assert _tts(client, speaker=speaker).status_code == 200
    assert _real(package)[0]["speaker_index"] == index


@pytest.mark.parametrize("default", ["Nobody", "²", pytest.param("9" * 5000, id="5000-digits")])
def test_a_default_speaker_that_names_nobody_falls_back_to_the_first_speaker(monkeypatch, default):
    """Resolved at import: "²" passed isdigit() and int() refused it, which stopped the service."""
    package = install_nemo(monkeypatch)
    client = TestClient(load_app(MAGPIE_DEFAULT_SPEAKER=default).app)

    response = client.post("/tts", json={"text": TEXT, "language": "de"})

    assert response.status_code == 200 and response.headers["X-Speaker"] == "Aria"
    assert _real(package)[0]["speaker_index"] == 0


# --- Refusals happen before the model is touched ---------------------------------------

@pytest.mark.parametrize("language", ["nl", "Dutch", "xx", "it", "pt-BR"])
def test_an_unsupported_language_is_a_400_and_never_reaches_the_model(monkeypatch, language):
    """NeMo would answer these with the English tokenizer and a normal 200."""
    package = install_nemo(monkeypatch)
    client = TestClient(load_app().app)

    response = _tts(client, language=language)

    assert response.status_code == 400
    detail = response.json()["detail"]
    assert language in detail and "de, en, es, fr, ja, zh" in detail
    assert package.loads == 0


def test_the_supported_languages_come_from_the_loaded_models_tokenizers(monkeypatch):
    package = install_nemo(monkeypatch, tokenizers=("english_phoneme", "german_phoneme"))
    client = TestClient(load_app().app)
    assert client.get("/languages").json()["languages"] == ["de", "en", "es", "fr", "ja", "zh"], "before load: documented"

    assert _tts(client).status_code == 200

    assert client.get("/languages").json()["languages"] == ["de", "en"]
    refused = _tts(client, language="fr")
    assert refused.status_code == 400 and "de, en" in refused.json()["detail"]
    assert len(_real(package)) == 1


def test_after_a_load_of_the_full_checkpoint_only_the_languages_nemo_routes_are_offered(monkeypatch):
    """The checkpoint holds Italian, Vietnamese and Hindi tokenizers NeMo 3.0.0's map cannot route to.

    A check that asked "does the model have a tokenizer for this language family" would offer
    them, and NeMo would read them with English rules and answer 200.
    """
    package = install_nemo(monkeypatch)  # CHECKPOINT_TOKENIZERS: the real tokenizer names
    app = load_app()
    client = TestClient(app.app)
    _warm(app)

    assert client.get("/languages").json()["languages"] == ["de", "en", "es", "fr", "ja", "zh"]
    for code in ("it", "vi", "hi"):
        response = _tts(client, "Ciao.", language=code)
        assert response.status_code == 400, code
    assert _real(package) == [], "an unroutable language reached do_tts"


def test_a_default_language_the_model_cannot_speak_refuses_requests_without_a_language(monkeypatch):
    """A misconfigured default must not silently become English."""
    package = install_nemo(monkeypatch)
    client = TestClient(load_app(MAGPIE_DEFAULT_LANGUAGE="nl").app)

    refused = client.post("/tts", json={"text": TEXT})
    explicit = _tts(client, language="de")

    assert refused.status_code == 400 and "nl" in refused.json()["detail"]
    assert explicit.status_code == 200
    assert package.loads == 1 and all(c["text"] != "Hallo." for c in package.calls), "no warm-up in a language it cannot speak"


@pytest.mark.parametrize("speaker", [
    "Bob", 5, "5", -1,
    # isdigit() is true for these but int() refuses them: a bare 500 before.
    "²", "⑤", "1²", pytest.param("9" * 5000, id="5000-digits"),
])
def test_an_unknown_speaker_is_a_400_listing_the_valid_ones(monkeypatch, speaker):
    package = install_nemo(monkeypatch)
    client = TestClient(load_app().app)

    response = _tts(client, speaker=speaker)

    assert response.status_code == 400
    assert "0 = Aria" in response.json()["detail"] and "4 = Sofia" in response.json()["detail"]
    assert package.loads == 0


def test_a_json_boolean_is_not_a_speaker(monkeypatch):
    """pydantic would coerce `true` to speaker 1 and answer 200 with the wrong voice."""
    package = install_nemo(monkeypatch)
    client = TestClient(load_app().app)

    response = _tts(client, speaker=True)

    assert response.status_code == 422
    assert package.loads == 0


def test_a_blank_text_is_a_400(monkeypatch):
    package = install_nemo(monkeypatch)
    client = TestClient(load_app().app)

    assert _tts(client, "   ").status_code == 400
    assert package.loads == 0


def test_a_text_over_the_limit_is_a_413_that_says_how_to_raise_it(monkeypatch):
    package = install_nemo(monkeypatch)
    client = TestClient(load_app(MAX_TEXT_CHARS="20").app)

    response = _tts(client, "x" * 21)

    assert response.status_code == 413
    assert "MAX_TEXT_CHARS" in response.json()["detail"] and "20" in response.json()["detail"]
    assert package.loads == 0


def test_a_huge_body_is_cut_off_before_it_is_parsed(monkeypatch):
    package = install_nemo(monkeypatch)
    client = TestClient(load_app(MAX_TEXT_CHARS="100").app)

    response = client.post("/tts", content=b'{"text": "' + b"x" * (2 * 1024 * 1024) + b'"}',
                           headers={"content-type": "application/json"})

    assert response.status_code == 413
    assert package.loads == 0


def test_a_cross_origin_request_is_refused_unless_the_origin_is_listed(monkeypatch):
    package = install_nemo(monkeypatch)
    client = TestClient(load_app().app)

    refused = client.post("/tts", json={"text": TEXT}, headers={"Origin": "https://evil.example"})
    unload = client.post("/unload", headers={"Origin": "https://evil.example"})

    assert refused.status_code == 403 and unload.status_code == 403
    assert package.loads == 0


# --- Long text -------------------------------------------------------------------------

def test_long_text_is_generated_in_whole_sentence_groups_and_joined_with_a_pause(monkeypatch):
    package = install_nemo(monkeypatch)
    client = TestClient(load_app(MAGPIE_MAX_GROUP_CHARS="60", MAGPIE_GROUP_GAP_MS="100").app)
    sentences = [f"Das ist der {n} Satz des Textes." for n in ("erste", "zweite", "dritte", "vierte")]

    response = _tts(client, " ".join(sentences))

    assert response.status_code == 200
    calls = _real(package)
    assert len(calls) == int(response.headers["X-Chunk-Count"]) > 1
    assert all(len(c["text"]) <= 60 and c["text"].endswith("Textes.") for c in calls)
    assert " ".join(c["text"] for c in calls) == " ".join(sentences)
    audio = decode_audio(response.content)
    gap = int(SAMPLE_RATE * 0.1)
    assert audio.size == sum(len(c["text"]) * SAMPLES_PER_CHAR for c in calls) + gap * (len(calls) - 1)
    assert np.count_nonzero(audio == 0) == gap * (len(calls) - 1), "the pauses are silence and nothing else is"


@pytest.mark.parametrize("language", ["zh", "ja"])
def test_chinese_and_japanese_text_is_generated_in_groups_too(monkeypatch, language):
    """NeMo splits them only at 。？！… above 100 characters / 80 words: a long sentence was one call, cut off at ~23 s."""
    package = install_nemo(monkeypatch)
    client = TestClient(load_app().app)
    text = "这是第一句话。这是第二句话。" * 10 + LONG_COMMA_SENTENCE

    response = _tts(client, text, language=language)

    assert response.status_code == 200
    calls = [c["text"] for c in _real(package)]
    assert int(response.headers["X-Chunk-Count"]) == len(calls) > 1
    assert all(len(c) <= 66 for c in calls), "a group longer than a third of MAGPIE_MAX_GROUP_CHARS"
    assert "".join(calls) == text


@pytest.mark.parametrize("language, first, second, joint, cap", [
    pytest.param("de", "Das ist der erste Satz.", "Das ist der zweite Satz.", " ", 40, id="de"),
    pytest.param("zh", "这是第一句话，这是第二句话。", "这是第三句话，这是第四句话。", "", 20, id="zh"),
])
def test_a_group_that_fills_the_decoders_limit_is_generated_again_in_halves(
        monkeypatch, caplog, language, first, second, joint, cap):
    """NeMo stops decoding after 500 frames (23.2 s) and returns what it has, with no error: a 200 with the end missing."""
    package = install_nemo(monkeypatch, decoder_cap=cap)
    client = TestClient(load_app(MAGPIE_GROUP_GAP_MS="100").app)
    text = first + joint + second  # one group, longer than the decoder can speak

    with caplog.at_level("WARNING"):
        response = _tts(client, text, language=language)

    assert response.status_code == 200
    assert [c["text"] for c in _real(package)] == [text, first, second]
    assert response.headers["X-Chunk-Count"] == "2"
    audio = decode_audio(response.content)
    gap = int(SAMPLE_RATE * 0.1)
    assert audio.size == (len(first) + len(second)) * SAMPLES_PER_CHAR + gap, "the cut-off attempt is in the answer"
    assert "generating it again as 2 pieces" in caplog.text


def test_a_half_that_still_fills_the_limit_is_kept_with_a_warning_not_a_500(monkeypatch, caplog):
    """A generation that never stops cannot be fixed by cutting the text further."""
    first, second = "Das ist der erste Satz.", "Das ist der zweite Satz."
    package = install_nemo(monkeypatch, decoder_cap=10)
    client = TestClient(load_app(MAGPIE_GROUP_GAP_MS="100").app)

    with caplog.at_level("WARNING"):
        response = _tts(client, f"{first} {second}")

    assert response.status_code == 200
    assert [c["text"] for c in _real(package)] == [f"{first} {second}", first, second], "a half was cut again"
    assert caplog.text.count("its end may be missing") == 2
    assert decode_audio(response.content).size == 2 * 10 * SAMPLES_PER_CHAR + int(SAMPLE_RATE * 0.1)


def test_a_group_too_short_to_halve_is_kept_with_a_warning(monkeypatch, caplog):
    package = install_nemo(monkeypatch, decoder_cap=5)
    client = TestClient(load_app().app)

    with caplog.at_level("WARNING"):
        response = _tts(client)

    assert response.status_code == 200 and response.headers["X-Chunk-Count"] == "1"
    assert [c["text"] for c in _real(package)] == [TEXT]
    assert "Group 1 of 1 (11 characters) filled the decoder's limit of 50 samples" in caplog.text


def test_audio_below_the_decoders_limit_is_not_generated_again(monkeypatch, caplog):
    package = install_nemo(monkeypatch, decoder_cap=len(TEXT) + 1)
    client = TestClient(load_app().app)

    with caplog.at_level("WARNING"):
        assert _tts(client).status_code == 200

    assert [c["text"] for c in _real(package)] == [TEXT]
    assert "decoder" not in caplog.text


@pytest.mark.parametrize("raw, expected", [("1000", 200), ("251", 200), ("250", 250), ("40", 40), ("39", 200)])
def test_the_group_ceiling_stays_where_nemo_does_not_chunk_a_group_again(monkeypatch, raw, expected):
    """Above about 250 characters an English group crosses NeMo's 45-word threshold."""
    install_nemo(monkeypatch)
    client = TestClient(load_app(MAGPIE_MAX_GROUP_CHARS=raw).app)

    assert client.get("/status").json()["max_group_chars"] == expected


# --- Loading, warm-up, release ---------------------------------------------------------

def test_the_load_warms_the_default_language_so_the_first_request_does_not_pay_for_it(monkeypatch):
    """The first German call builds the text normalizer (~12 s on the real model); later calls cost nothing extra."""
    package = install_nemo(monkeypatch)
    app = load_app()

    _warm(app)

    assert package.loads == 1
    [warmup] = package.calls
    assert warmup["language"] == "de" and warmup["apply_TN"] is True and warmup["text"] == "Hallo."


def test_more_languages_are_warmed_when_asked_and_only_the_ones_the_model_can_speak(monkeypatch, caplog):
    """The first request in a language builds its normalizer (12-48 s measured): warm them at load instead."""
    package = install_nemo(monkeypatch)
    app = load_app(MAGPIE_WARM_LANGUAGES="en, FR, de, nl")

    with caplog.at_level("WARNING"):
        _warm(app)

    assert [c["language"] for c in package.calls] == ["de", "en", "fr"], "default first, no duplicates, no nl"
    assert any("Not warming 'nl'" in r.getMessage() for r in caplog.records)


def test_a_failed_extra_warm_up_does_not_stop_the_others_or_the_load(monkeypatch):
    package = install_nemo(monkeypatch, generate_error=RuntimeError("normalizer exploded"), generate_error_calls=2)
    app = load_app(MAGPIE_WARM_LANGUAGES="en,fr")

    _warm(app)

    assert [c["language"] for c in package.calls] == ["de", "en", "fr"], "the failures of de and en did not skip fr"
    assert package.loads == 1


def test_only_the_default_language_is_warmed_unless_asked(monkeypatch):
    package = install_nemo(monkeypatch)

    _warm(load_app())

    assert [c["language"] for c in package.calls] == ["de"]


def test_a_failed_warm_up_does_not_fail_the_load(monkeypatch):
    package = install_nemo(monkeypatch, generate_error=RuntimeError("normalizer exploded"), generate_error_calls=1)
    client = TestClient(load_app().app)

    response = _tts(client)

    assert response.status_code == 200, "the model loaded; only the warm-up generation failed"
    assert package.loads == 1 and len(package.calls) == 2


def test_the_model_is_loaded_by_hub_id_or_by_path(monkeypatch, tmp_path):
    package = install_nemo(monkeypatch)
    _warm(load_app())
    assert package.load_names == ["nvidia/magpie_tts_multilingual_357m"]

    checkpoint = tmp_path / "magpie.nemo"
    checkpoint.write_bytes(b"x")
    _warm(load_app(MAGPIE_MODEL=str(checkpoint)))
    assert package.load_names[-1] == str(checkpoint)


def test_a_checkpoint_path_is_never_shown_to_callers(monkeypatch, tmp_path):
    install_nemo(monkeypatch)
    checkpoint = tmp_path / "private-dir" / "magpie.nemo"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"x")
    client = TestClient(load_app(MAGPIE_MODEL=str(checkpoint)).app)

    for path in ("/ready", "/status"):
        body = client.get(path).text
        assert "private-dir" not in body and "magpie.nemo" in body


def test_health_and_status_never_load_the_model(monkeypatch):
    package = install_nemo(monkeypatch)
    client = TestClient(load_app().app)

    health, ready, status = client.get("/health"), client.get("/ready"), client.get("/status")

    assert health.status_code == 200 and health.json()["model_resident"] is False
    assert ready.status_code == 200 and ready.json()["ready"] is True, "nothing is known to be wrong yet"
    assert status.json()["service"] == "Magpie-TTS" and status.json()["queue"]["max_concurrency"] == 1
    assert status.json()["apply_text_normalization"] is True
    assert package.loads == 0


def test_unload_frees_the_model_and_the_next_request_reloads_it(monkeypatch):
    package = install_nemo(monkeypatch)
    client = TestClient(load_app().app)
    assert _tts(client).status_code == 200
    first = package.model

    unloaded = client.post("/unload")

    assert unloaded.status_code == 200 and unloaded.json()["model_resident"] is False
    assert first.moved_to_cpu, "the weights must leave the GPU even if NeMo still references the module"
    assert _tts(client).status_code == 200
    assert package.loads == 2


# NeMo 3.0.0 keeps each language's text normalizer on the model instance and builds it on
# the language's first use (12-48 s). They are CPU-only, so they outlive an idle unload.

def test_a_reload_after_an_unload_neither_warms_up_nor_builds_a_normalizer_again(monkeypatch):
    package = install_nemo(monkeypatch)
    client = TestClient(load_app().app)

    assert _tts(client).status_code == 200
    assert client.post("/unload").status_code == 200
    assert _tts(client).status_code == 200

    assert package.loads == 2
    assert [c["text"] for c in package.calls] == ["Hallo.", TEXT, TEXT], "the reload warmed up again"
    assert package.normalizer_builds == ["de"]
    first, second = package.models
    assert first._text_normalizers is second._text_normalizers


def test_after_an_unload_none_of_the_warm_up_languages_is_warmed_again(monkeypatch):
    package = install_nemo(monkeypatch)
    app = load_app(MAGPIE_WARM_LANGUAGES="en,fr")
    client = TestClient(app.app)
    _warm(app)
    assert [c["language"] for c in package.calls] == ["de", "en", "fr"]

    assert client.post("/unload").status_code == 200
    assert _tts(client).status_code == 200

    assert package.loads == 2
    assert [c["text"] for c in package.calls[3:]] == [TEXT], "the reload warmed up again"
    assert package.normalizer_builds == ["de", "en", "fr"]


def test_a_normalizer_a_request_built_is_not_built_again_after_a_reload(monkeypatch):
    package = install_nemo(monkeypatch)
    client = TestClient(load_app().app)

    assert _tts(client, "Bonjour.", language="fr").status_code == 200
    assert client.post("/unload").status_code == 200
    assert _tts(client, "Bonjour.", language="fr").status_code == 200

    assert package.loads == 2 and package.normalizer_builds == ["de", "fr"]


def test_the_normalizers_also_outlive_an_unload_by_the_idle_timer(monkeypatch):
    package = install_nemo(monkeypatch)
    app = load_app(TTS_MODEL_TTL="0.2")
    client = TestClient(app.app)

    assert _tts(client, "Bonjour.", language="fr").status_code == 200
    assert wait_until(lambda: not app._model_slot.resident), "the idle timer never unloaded the model"
    assert _tts(client, "Bonjour.", language="fr").status_code == 200

    assert package.loads == 2
    assert package.normalizer_builds == ["de", "fr"]
    assert [c["text"] for c in package.calls].count("Hallo.") == 1


def test_a_normalizer_whose_build_failed_is_built_again_by_the_next_load(monkeypatch):
    """NeMo caches None for a failed build and from then on drops every digit; only a new build speaks them again."""
    package = install_nemo(monkeypatch, failing_normalizers=("de",))
    client = TestClient(load_app().app)
    text = "Zimmer 2026."

    broken = _tts(client, text)
    assert client.post("/unload").status_code == 200
    fixed = _tts(client, text)

    assert package.normalizer_builds == ["de", "de"]
    assert [c["text"] for c in package.calls] == ["Hallo.", text, "Hallo.", text], "the failed language was not warmed again"
    assert decode_audio(broken.content).size == len("Zimmer .") * SAMPLES_PER_CHAR, "the failed build kept the digits"
    assert decode_audio(fixed.content).size == len(text) * SAMPLES_PER_CHAR, "the digits are still missing"


def test_a_model_that_keeps_no_normalizers_where_the_service_sees_them_warms_up_on_every_load(monkeypatch):
    package = install_nemo(monkeypatch, normalizer_cache=False)
    client = TestClient(load_app().app)

    assert _tts(client).status_code == 200
    assert client.post("/unload").status_code == 200
    assert _tts(client).status_code == 200

    assert [c["text"] for c in package.calls] == ["Hallo.", TEXT, "Hallo.", TEXT]


def test_a_checkpoint_with_no_baked_speakers_is_a_load_failure_with_a_helpful_log(monkeypatch, caplog):
    install_nemo(monkeypatch, speakers=0)
    client = TestClient(load_app().app)

    with caplog.at_level("ERROR"):
        response = _tts(client)

    assert response.status_code == 500
    assert any("baked speakers" in r.getMessage() for r in caplog.records)
    assert client.get("/ready").json()["reason"] == "load_failed"


# --- Speakers --------------------------------------------------------------------------

def test_the_speaker_catalog_has_the_shape_the_gateway_normalizes(monkeypatch):
    install_nemo(monkeypatch)
    client = TestClient(load_app().app)

    body = client.get("/speakers").json()

    assert body["speakers"] == ["Aria", "Jason", "John", "Leo", "Sofia"]
    assert body["default_language"] == "de" and "de" in body["languages"]


def test_the_catalog_follows_the_models_speaker_count_once_it_is_loaded(monkeypatch):
    install_nemo(monkeypatch, speakers=7)
    app = load_app(MAGPIE_SPEAKERS="A,B")
    client = TestClient(app.app)
    assert client.get("/speakers").json()["speakers"] == ["A", "B"], "before load: exactly the configured names"
    assert _tts(client, speaker=6).status_code == 400, "before load the configured names are all that is known"

    _warm(app)

    assert _tts(client, speaker=6).status_code == 200

    assert client.get("/speakers").json()["speakers"] == ["A", "B"] + [f"Speaker {i}" for i in range(2, 7)]


def test_header_values_cannot_carry_control_characters():
    """h11 refuses such a header value, and with it the response whose audio was already generated."""
    safe_header = load_app()._safe_header

    safe = safe_header("evil\r\nX-Injected: 1\x00")

    assert "\r" not in safe and "\n" not in safe and "\x00" not in safe
    assert safe_header("\tSofia\x7f") == "Sofia", "h11 also refuses a space at either end"
    assert safe_header("Soфia") == "So?ia"


def test_a_speaker_name_with_a_control_character_answers_200_with_a_clean_header(monkeypatch):
    """The name is cleaned in place: its position in MAGPIE_SPEAKERS is its speaker index."""
    package = install_nemo(monkeypatch)
    client = TestClient(load_app(MAGPIE_SPEAKERS="Aria,Jason,John,Leo,So\x07fia").app)

    response = _tts(client, speaker=4)

    assert response.status_code == 200 and response.headers["X-Speaker"] == "Sofia"
    assert _real(package)[0]["speaker_index"] == 4
    assert client.get("/speakers").json()["speakers"] == ["Aria", "Jason", "John", "Leo", "Sofia"]


# --- Failure hygiene -------------------------------------------------------------------

def test_a_failed_load_says_nothing_internal_to_the_caller_and_everything_to_the_log(monkeypatch, caplog):
    install_nemo(monkeypatch, load_error=OSError("[Errno 28] No space left on device: '/root/.cache/huggingface/x'"))
    client = TestClient(load_app().app)

    with caplog.at_level("ERROR"):
        response = _tts(client)
        ready = client.get("/ready")

    assert response.status_code == 500
    for secret in ("/root/.cache", "No space left", "huggingface", "Errno"):
        assert secret not in response.text and secret not in ready.text, f"reveals {secret!r}"
    assert "request id" in response.json()["detail"]
    assert ready.status_code == 503 and ready.json()["reason"] == "load_failed"
    assert any("No space left" in r.getMessage() for r in caplog.records), "the operator lost the detail"
    assert client.get("/health").status_code == 200


def test_a_failed_generation_is_a_generic_500_with_a_request_id(monkeypatch, caplog):
    install_nemo(monkeypatch, generate_error=RuntimeError("tensor shape (1, 2) at /opt/nemo/magpietts.py:4521"),
                 generate_error_calls=None)
    client = TestClient(load_app().app)

    with caplog.at_level("ERROR"):
        response = _tts(client)

    assert response.status_code == 500
    assert "/opt/nemo" not in response.text and "tensor shape" not in response.text
    request_id = response.headers["X-Request-ID"]
    assert request_id in response.json()["detail"]
    assert any(request_id in r.getMessage() and "tensor shape" in r.getMessage() for r in caplog.records)


def test_cuda_out_of_memory_is_a_503_telling_the_caller_to_retry(monkeypatch):
    install_nemo(monkeypatch, generate_error=RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB"))
    client = TestClient(load_app().app)

    response = _tts(client)

    assert response.status_code == 503 and int(response.headers["Retry-After"]) >= 1
    assert "out of memory" in response.json()["detail"] and "Tried to allocate" not in response.json()["detail"]


# --- Admission and cancellation --------------------------------------------------------

@pytest.fixture
def busy(monkeypatch):
    """busy(**env) -> (app, package): a loaded model whose next generation blocks on `model.gate`."""
    def _busy(**env):
        package = install_nemo(monkeypatch)
        app = load_app(**env)
        _warm(app)
        model = package.model
        model.gate = threading.Event()
        model.entered.clear()
        package.calls.clear()
        return app, package

    return _busy


def _settled(app) -> bool:
    return app._model_slot.refs == 0 and app._gate.in_flight == 0


def _request(app, text, **fields):
    return app.TTSRequest(text=text, language="de", **fields)


def test_requests_beyond_the_queue_depth_are_turned_away_at_once(busy):
    app, package = busy(TTS_QUEUE_TIMEOUT_S="30", TTS_MAX_QUEUE="1")
    model = package.model

    async def main():
        running = asyncio.create_task(app.text_to_speech(_request(app, "Läuft.")))
        assert await asyncio.to_thread(model.entered.wait, 5), "no generation started"
        queued = asyncio.create_task(app.text_to_speech(_request(app, "Wartet.")))
        assert await wait_until_async(lambda: app._gate.in_flight == 2)
        began = time.monotonic()
        with pytest.raises(app.HTTPException) as caught:
            await app.text_to_speech(_request(app, "Zu viel."))
        turned_away_after = time.monotonic() - began
        model.gate.set()
        first, second = await running, await queued
        assert await wait_until_async(lambda: _settled(app)), "a turn or a model reference was leaked"
        return caught.value, turned_away_after, first, second

    error, turned_away_after, first, second = asyncio.run(main())

    assert error.status_code == 503 and "TTS_MAX_QUEUE" in error.detail
    assert int(error.headers["Retry-After"]) >= 1
    assert turned_away_after < 1.0, "the excess request waited instead of being shed"
    assert first.status_code == 200 and second.status_code == 200
    assert [c["text"] for c in package.calls] == ["Läuft.", "Wartet."]


def test_a_request_waits_at_most_the_queue_timeout_and_gives_everything_back(busy):
    app, package = busy(TTS_QUEUE_TIMEOUT_S="0.4", TTS_MAX_QUEUE="4")
    model = package.model

    async def main():
        first = asyncio.create_task(app.text_to_speech(_request(app, "Erster.")))
        assert await asyncio.to_thread(model.entered.wait, 5)
        began = time.monotonic()
        with pytest.raises(app.HTTPException) as caught:
            await app.text_to_speech(_request(app, "Zweiter."))
        waited = time.monotonic() - began
        model.gate.set()
        await first
        assert await wait_until_async(lambda: _settled(app))
        third = await app.text_to_speech(_request(app, "Dritter."))
        return caught.value, waited, third

    error, waited, third = asyncio.run(main())

    assert error.status_code == 503 and "TTS_QUEUE_TIMEOUT_S" in error.detail
    assert int(error.headers["Retry-After"]) >= 1
    assert 0.3 < waited < 5
    assert third.status_code == 200
    assert [c["text"] for c in package.calls] == ["Erster.", "Dritter."], "the timed-out request was generated"


def test_a_cancelled_request_leaves_the_model_pinned_until_its_thread_is_done(busy):
    """The client leaves mid-generation: /unload and the idle reaper must still not free weights in use."""
    app, package = busy()
    model = package.model

    async def main():
        task = asyncio.create_task(app.text_to_speech(_request(app, "Erster Satz. Zweiter Satz.")))
        assert await asyncio.to_thread(model.entered.wait, 5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        during = await app.unload()
        model.gate.set()
        assert await wait_until_async(lambda: _settled(app)), "the thread's reference was never given back"
        after = await app.unload()
        return during, after

    during, after = asyncio.run(main())

    assert during.status_code == 409, "the weights were released while a thread was still reading them"
    assert after["model_resident"] is False and model.moved_to_cpu


def test_a_cancelled_request_stops_after_its_current_group(busy):
    app, package = busy(MAGPIE_MAX_GROUP_CHARS="40")
    model = package.model
    text = " ".join(f"Das ist der {n} Satz hier." for n in ("erste", "zweite", "dritte", "vierte", "fünfte"))

    async def main():
        task = asyncio.create_task(app.text_to_speech(_request(app, text)))
        assert await asyncio.to_thread(model.entered.wait, 5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        model.gate.set()
        assert await wait_until_async(lambda: _settled(app))

    asyncio.run(main())

    assert len(package.calls) == 1, "the remaining groups were generated for a client that had gone"


def test_two_generations_never_run_in_the_model_together_even_after_a_cancellation(busy):
    """A cancelled request frees its turn at once; its thread is still inside the model."""
    app, package = busy()
    model = package.model

    async def main():
        first = asyncio.create_task(app.text_to_speech(_request(app, "Erster.")))
        assert await asyncio.to_thread(model.entered.wait, 5)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        second = asyncio.create_task(app.text_to_speech(_request(app, "Zweiter.")))
        await asyncio.sleep(0.2)
        model.gate.set()
        response = await second
        assert await wait_until_async(lambda: _settled(app))
        return response

    assert asyncio.run(main()).status_code == 200
    assert model.max_running == 1


def test_the_service_serves_the_next_request_after_a_queue_full_refusal(busy):
    app, package = busy(TTS_MAX_QUEUE="0", TTS_QUEUE_TIMEOUT_S="30")
    model = package.model

    async def main():
        running = asyncio.create_task(app.text_to_speech(_request(app, "Läuft.")))
        assert await asyncio.to_thread(model.entered.wait, 5)
        with pytest.raises(app.HTTPException) as caught:
            await app.text_to_speech(_request(app, "Zu viel."))
        model.gate.set()
        await running
        assert await wait_until_async(lambda: _settled(app))
        return caught.value, await app.text_to_speech(_request(app, "Danach."))

    error, after = asyncio.run(main())

    assert error.status_code == 503 and after.status_code == 200


# --- A client that hangs up ---------------------------------------------------------------
#
# uvicorn and Starlette never cancel a handler whose client went away; they only answer its
# next receive() with http.disconnect. asgi_post_json hangs up exactly that way.

FIVE_SENTENCES = " ".join(f"Das ist der {n} Satz hier." for n in ("erste", "zweite", "dritte", "vierte", "fünfte"))


def _status(messages) -> int:
    return next(m["status"] for m in messages if m["type"] == "http.response.start")


def test_a_client_that_hangs_up_stops_its_generation_after_the_current_group(busy, caplog):
    """A gateway that gave up (its read timeout) used to leave every remaining group generating for nobody."""
    app, package = busy(MAGPIE_MAX_GROUP_CHARS="40")
    model = package.model
    assert len(app.group_text(FIVE_SENTENCES, "de", app.MAX_GROUP_CHARS)) == 5

    async def main():
        sent = await asgi_post_json(app.app, "/tts", {"text": FIVE_SENTENCES, "language": "de"},
                                    disconnect_when=model.entered.is_set)
        model.gate.set()  # the first group, still running in its thread, may finish now
        assert await wait_until_async(lambda: _settled(app)), "the thread's lease or the queue place was kept"
        return sent

    with caplog.at_level("INFO"):
        sent = asyncio.run(main())

    assert _status(sent) == 499
    assert len(package.calls) == 1, f"{len(package.calls)} of 5 groups were generated for a client that had gone"
    assert "client went away" in caplog.text
    assert TestClient(app.app).post("/tts", json={"text": TEXT, "language": "de"}).status_code == 200


def test_a_queued_request_whose_client_hangs_up_leaves_the_queue_at_once(busy):
    """Not when its turn comes: until then it would hold a place a waiting client could use."""
    app, package = busy(TTS_QUEUE_TIMEOUT_S="30")
    model = package.model
    leave = threading.Event()

    async def main():
        running = asyncio.create_task(app.text_to_speech(_request(app, "Läuft.")))
        assert await asyncio.to_thread(model.entered.wait, 5)
        queued = asyncio.create_task(asgi_post_json(
            app.app, "/tts", {"text": "Wartet.", "language": "de"}, disconnect_when=leave.is_set))
        assert await wait_until_async(lambda: app._gate.in_flight == 2), "the second request never queued"
        leave.set()
        sent = await queued
        in_flight = app._gate.in_flight
        model.gate.set()
        first = await running
        assert await wait_until_async(lambda: _settled(app))
        return sent, in_flight, first

    sent, in_flight, first = asyncio.run(main())

    assert _status(sent) == 499
    assert in_flight == 1, "the departed client kept its place in the queue"
    assert first.status_code == 200
    assert [c["text"] for c in package.calls] == ["Läuft."], "the abandoned request was generated"


def test_a_client_that_stays_gets_its_audio_while_the_service_listens_for_a_hang_up(busy):
    app, package = busy()
    package.model.gate.set()

    sent = asyncio.run(asgi_post_json(app.app, "/tts", {"text": TEXT, "language": "de"}))

    assert _status(sent) == 200
    assert [c["text"] for c in package.calls] == [TEXT]


def test_under_uvicorn_a_closed_connection_stops_the_generation(busy):
    """The tests above hand the handler http.disconnect themselves; here uvicorn's h11 server does."""
    uvicorn = pytest.importorskip("uvicorn")
    app, package = busy(MAGPIE_MAX_GROUP_CHARS="40")
    model = package.model
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    port = listener.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(
        app.app, lifespan="off", http="h11", ws="none", log_config=None, log_level="warning"))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [listener]}, daemon=True)
    thread.start()
    try:
        assert wait_until(lambda: server.started, timeout=10), "uvicorn did not start"
        body = json.dumps({"text": FIVE_SENTENCES, "language": "de"}).encode()
        with socket.create_connection(("127.0.0.1", port), timeout=10) as client:
            client.sendall(
                b"POST /tts HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\n"
                + f"Content-Length: {len(body)}\r\n\r\n".encode() + body
            )
            assert model.entered.wait(10), "no generation started"
        # The connection is closed while the first group is still running.
        assert wait_until(lambda: app._gate.in_flight == 0, timeout=10), "the handler never learned the client left"
        model.gate.set()
        assert wait_until(lambda: _settled(app), timeout=10)
    finally:
        model.gate.set()
        server.should_exit = True
        thread.join(10)
        listener.close()

    assert len(package.calls) == 1, f"{len(package.calls)} of 5 groups were generated for a client that had gone"
