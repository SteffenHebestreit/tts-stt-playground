"""Unit tests for piper-tts-service/naming.py (dependency-light helpers)."""

import json
import os
import time
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException


def _load_naming():
    """Load the standalone naming module without importing onnxruntime/librosa."""
    path = Path(__file__).resolve().parents[1] / "piper-tts-service" / "naming.py"
    spec = spec_from_file_location("piper_tts_naming", path)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


naming = _load_naming()


def _voice(language, quality, gender=None):
    return SimpleNamespace(language=language, quality=quality, gender=gender)


# --- sanitize_voice_name ---

@pytest.mark.parametrize("value", ["luna", "Voice_1", "de-DE-x", "abc123"])
def test_sanitize_accepts_valid(value):
    assert naming.sanitize_voice_name(value) == value


@pytest.mark.parametrize("value", ["", "../x", "a/b", "a b", "naïve", "a.b", "a\\b", None, "luna\n", "luna\r\n", "\nluna", 42])
def test_sanitize_rejects_invalid(value):
    with pytest.raises(HTTPException) as exc:
        naming.sanitize_voice_name(value)
    assert exc.value.status_code == 400


# --- select_best_voice ---

def _catalog():
    return {
        "en_US-lessac-medium": _voice("en_US", "medium", "male"),
        "en_US-amy-medium": _voice("en_US", "medium", "female"),
        "de_DE-thorsten-medium": _voice("de_DE", "medium", "male"),
        "de_DE-eva_k-x_low": _voice("de_DE", "x_low", "female"),
    }


def test_select_matches_language_and_quality():
    assert naming.select_best_voice(_catalog(), "de", "x_low") == "de_DE-eva_k-x_low"


def test_select_prefers_gender_when_available():
    assert naming.select_best_voice(_catalog(), "de", "medium", "male") == "de_DE-thorsten-medium"


def test_select_falls_back_to_english_for_unknown_language():
    chosen = naming.select_best_voice(_catalog(), "xx", "medium")
    assert chosen.startswith("en_")


def test_select_quality_fallback_keeps_language():
    # No 'high' German voice -> falls back to any German voice, not English
    assert naming.select_best_voice(_catalog(), "de", "high").startswith("de_")


def test_select_ultimate_fallback_on_empty_catalog():
    assert naming.select_best_voice({}, "de", "medium") == "en_US-lessac-medium"


# --- normalize_phoneme_id_map ---

def test_normalize_keeps_plain_int_ids():
    assert naming.normalize_phoneme_id_map({"a": 1, "b": 2}) == {"a": 1, "b": 2}


def test_normalize_unwraps_piper_style_lists():
    assert naming.normalize_phoneme_id_map({"_": [0], "^": [1], "a": [5, 9]}) == {"_": 0, "^": 1, "a": 5}


def test_normalize_drops_unusable_entries():
    assert naming.normalize_phoneme_id_map({
        "empty": [],
        "string": "3",
        "none": None,
        "bool": True,
        "ok": 7,
    }) == {"ok": 7}


def test_normalize_handles_none_and_empty():
    assert naming.normalize_phoneme_id_map(None) == {}
    assert naming.normalize_phoneme_id_map({}) == {}


# --- prune_old_outputs ---

def test_prune_removes_only_old_files(tmp_path):
    old = tmp_path / "old.wav"
    new = tmp_path / "new.wav"
    old.write_bytes(b"x")
    new.write_bytes(b"y")
    now = time.time()
    os.utime(old, (now - 48 * 3600, now - 48 * 3600))  # 48h old
    os.utime(new, (now, now))

    removed = naming.prune_old_outputs(str(tmp_path), retention_hours=24, now=now)

    assert removed == 1
    assert not old.exists()
    assert new.exists()


def test_prune_disabled_when_retention_zero(tmp_path):
    f = tmp_path / "a.wav"
    f.write_bytes(b"x")
    os.utime(f, (0, 0))  # very old
    assert naming.prune_old_outputs(str(tmp_path), retention_hours=0) == 0
    assert f.exists()


def test_prune_missing_dir_is_safe():
    assert naming.prune_old_outputs("/no/such/dir", retention_hours=24) == 0


# --- language matching / fallback detection ---------------------------------
#
# select_best_voice() degrades to an English voice when nothing serves the
# requested language. That is deliberate (the service keeps working) but it means
# a German request can return English audio with HTTP 200. language_matches() is
# what lets the service detect and report that, so these lock the semantics down.


def test_language_matches_on_base_tag():
    assert naming.language_matches("de_DE", "de")
    assert naming.language_matches("de_DE", "de_DE")
    assert naming.language_matches("en_US", "en")
    assert naming.language_matches("en_GB", "en_US")   # same base language


def test_language_mismatch_is_detected():
    assert not naming.language_matches("en_US", "de")
    assert not naming.language_matches("de_DE", "fr")
    assert not naming.language_matches("nl_NL", "de")  # near neighbours must not match


def test_auto_or_empty_request_matches_anything():
    """No specific language was asked for, so nothing was substituted."""
    for req in ("auto", "AUTO", "", None):
        assert naming.language_matches("de_DE", req)


def test_language_matching_is_case_insensitive():
    assert naming.language_matches("DE_DE", "de")
    assert naming.language_matches("de_DE", "DE")


def test_english_fallback_is_reported_as_a_mismatch():
    """The exact silent-failure case: a German request served by the hardcoded
    en_US default must be detectable by the service."""
    voices = {
        "en_US-lessac-medium": _voice("en_US", "medium"),
    }
    chosen = naming.select_best_voice(voices, "de", "medium")
    assert chosen == "en_US-lessac-medium"          # the fallback happened
    assert not naming.language_matches(voices[chosen].language, "de")  # and is visible


def test_german_request_served_by_german_voice_is_not_a_mismatch():
    voices = {
        "en_US-lessac-medium": _voice("en_US", "medium"),
        "de_DE-thorsten-medium": _voice("de_DE", "medium"),
    }
    chosen = naming.select_best_voice(voices, "de", "medium")
    assert chosen == "de_DE-thorsten-medium"
    assert naming.language_matches(voices[chosen].language, "de")


# --- BCP-47 spellings and regions ----------------------------------------------------------
#
# Browsers and OpenAI clients send 'de-DE'. Splitting only on '_' turned that into an
# unknown language, so a German request was served by an English voice.


@pytest.mark.parametrize("tag, base", [
    ("de", "de"), ("de_DE", "de"), ("de-DE", "de"), ("DE-at", "de"), (" en_US ", "en"),
    ("auto", ""), ("AUTO", ""), ("", ""), (None, ""),
])
def test_base_language(tag, base):
    assert naming.base_language(tag) == base


@pytest.mark.parametrize("requested", ["de", "de_DE", "de-DE", "DE", "de-at"])
def test_hyphenated_and_regional_tags_match_a_german_voice(requested):
    assert naming.language_matches("de_DE", requested)
    assert naming.select_best_voice(_catalog(), requested, "medium").startswith("de_")


def test_hyphenated_language_is_not_reported_as_a_fallback():
    assert naming.language_matches("de_DE", "de-DE")
    assert not naming.language_matches("en_US", "de-DE")


def test_select_prefers_the_requested_region():
    voices = {**_catalog(), "en_GB-alan-medium": _voice("en_GB", "medium")}
    assert naming.select_best_voice(voices, "en_GB", "medium") == "en_GB-alan-medium"
    assert naming.select_best_voice(voices, "en-US", "medium") == "en_US-lessac-medium"
    assert naming.select_best_voice(voices, "en", "medium") == "en_US-lessac-medium"  # no region asked


def test_select_matches_the_base_language_exactly_not_by_prefix():
    voices = {"deu_XX-x-medium": _voice("deu_XX", "medium"), "en_US-lessac-medium": _voice("en_US", "medium")}
    assert naming.select_best_voice(voices, "de", "medium") == "en_US-lessac-medium"


def test_empty_or_auto_language_considers_every_voice():
    assert naming.select_best_voice(_catalog(), "auto", "x_low") == "de_DE-eva_k-x_low"
    assert naming.select_best_voice(_catalog(), "", "medium") in _catalog()


# --- preferred voice, fallback language, gender --------------------------------------------

def test_preferred_voice_goes_first_among_the_voices_of_the_language():
    assert naming.select_best_voice(_catalog(), "de", None, preferred="de_DE-eva_k-x_low") == "de_DE-eva_k-x_low"
    assert naming.select_best_voice(_catalog(), "de", None) == "de_DE-thorsten-medium"  # no preference: first listed
    assert naming.select_best_voice(_catalog(), "de", None, preferred="de_DE-thorsten-medium") == "de_DE-thorsten-medium"


def test_preferred_voice_yields_to_a_quality_it_does_not_have():
    assert naming.select_best_voice(_catalog(), "de", "x_low", preferred="de_DE-thorsten-medium") == "de_DE-eva_k-x_low"


def test_preferred_voice_of_another_language_is_ignored():
    assert naming.select_best_voice(_catalog(), "en", "medium", preferred="de_DE-thorsten-medium").startswith("en_")


def test_preferred_voice_yields_to_a_requested_gender():
    assert naming.select_best_voice(_catalog(), "en", "medium", "female", preferred="en_US-lessac-medium") == "en_US-amy-medium"


@pytest.mark.parametrize("gender", ["any", "ANY", "auto", " any ", ""])
def test_gender_no_preference_spellings_do_not_filter(gender):
    """The UI sends 'any'; it must not be compared with a voice's gender."""
    assert naming.select_best_voice(_catalog(), "en", "medium", gender, preferred="en_US-lessac-medium") == "en_US-lessac-medium"


def test_unserved_language_falls_back_to_english_then_the_fallback_language():
    german_only = {k: v for k, v in _catalog().items() if k.startswith("de_")}
    assert naming.select_best_voice(_catalog(), "fr", "medium", fallback_language="de").startswith("en_")
    assert naming.select_best_voice(german_only, "fr", "medium", fallback_language="de").startswith("de_")
    assert naming.select_best_voice(german_only, "fr", "medium") == "en_US-lessac-medium"  # nothing to offer


# --- detect_language ----------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "Guten Tag, das ist ein Test.",
    "Ich habe heute keine Zeit für dich.",
    "Schöne Grüße aus Köln",
    "Straße",                       # eszett alone is enough
    "Die Katze sitzt auf dem Tisch und schläft.",
    "Wie geht es dir? Mir geht es sehr gut.",
])
def test_detects_german(text):
    assert naming.detect_language(text) == "de"


@pytest.mark.parametrize("text", [
    "Good morning, this is a test of the system.",
    "I would like to have a coffee, please.",
    "They said that you can come with us.",
    "What is the weather like today?",
    "I went to Zürich with the team and it was great",   # one umlaut must not outvote the sentence
])
def test_detects_english(text):
    assert naming.detect_language(text) == "en"


@pytest.mark.parametrize("text", [
    "", "   ", None, "Hello", "Hello world", "Hallo", "OK", "12345", "!!!",
    "Bonjour tout le monde, comment allez-vous",
    "in an war will hat bin so was",            # words both languages share carry no evidence
    "Der Kunde said the",                        # tie: one clue each way is not evidence
])
def test_no_confident_guess_gives_none(text):
    assert naming.detect_language(text) is None


def test_detection_only_looks_at_the_start_of_long_text():
    text = ("Guten Tag, das ist ein Test. " * 100) + "the and is are " * 10000
    assert naming.detect_language(text) == "de"


# --- parse_voice_id / scan_voice_dir -------------------------------------------------------

@pytest.mark.parametrize("voice_id, expected", [
    ("de_DE-thorsten-medium", ("de_DE", "thorsten", "medium")),
    ("de_DE-eva_k-x_low", ("de_DE", "eva_k", "x_low")),
    ("en_GB-northern_english_male-medium", ("en_GB", "northern_english_male", "medium")),
    ("xx_XX-a-b-high", ("xx_XX", "a-b", "high")),
    ("my-voice", ("my", "voice", None)),
    ("solo", (None, None, None)),
])
def test_parse_voice_id(voice_id, expected):
    parsed = naming.parse_voice_id(voice_id)
    assert (parsed["language"], parsed["speaker"], parsed["quality"]) == expected


def _write_voice(directory, voice_id, config=None, model=True):
    directory.joinpath(f"{voice_id}.onnx.json").write_text(json.dumps(config if config is not None else {}))
    if model:
        directory.joinpath(f"{voice_id}.onnx").write_bytes(b"x")


def test_scan_reads_metadata_from_the_piper_config(tmp_path):
    _write_voice(tmp_path, "de_DE-thorsten-medium", {
        "audio": {"sample_rate": 22050, "quality": "medium"},
        "language": {"code": "de_DE"},
        "dataset": "thorsten",
    })

    assert naming.scan_voice_dir(tmp_path) == {
        "de_DE-thorsten-medium": {
            "name": "de_DE-thorsten-medium", "language": "de_DE", "speaker": "thorsten",
            "quality": "medium", "sample_rate": 22050,
        }
    }


def test_scan_prefers_the_config_over_the_file_name(tmp_path):
    _write_voice(tmp_path, "de_DE-thorsten-medium", {
        "audio": {"sample_rate": 16000, "quality": "high"}, "language": {"code": "de_AT"}, "dataset": "other",
    })

    voice = naming.scan_voice_dir(tmp_path)["de_DE-thorsten-medium"]

    assert (voice["language"], voice["speaker"], voice["quality"], voice["sample_rate"]) == ("de_AT", "other", "high", 16000)


def test_scan_falls_back_to_espeak_then_file_name_then_defaults(tmp_path):
    _write_voice(tmp_path, "sv_SE-nst-medium", {"espeak": {"voice": "sv"}})
    _write_voice(tmp_path, "cs_CZ-jirka-low", {})
    _write_voice(tmp_path, "custom", {"audio": {"sample_rate": "fast"}})

    found = naming.scan_voice_dir(tmp_path)

    assert found["sv_SE-nst-medium"]["language"] == "sv"
    assert found["cs_CZ-jirka-low"]["language"] == "cs_CZ" and found["cs_CZ-jirka-low"]["quality"] == "low"
    assert found["custom"] == {"name": "custom", "language": "en", "speaker": "custom", "quality": "medium", "sample_rate": 22050}


def test_scan_needs_both_files_and_skips_unusable_ones(tmp_path):
    _write_voice(tmp_path, "ok-voice-low", {})
    _write_voice(tmp_path, "no-model-low", {}, model=False)
    (tmp_path / "no-config-low.onnx").write_bytes(b"x")
    (tmp_path / "broken-json-low.onnx").write_bytes(b"x")
    (tmp_path / "broken-json-low.onnx.json").write_text("{nope")
    _write_voice(tmp_path, "not-an-object-low", [1, 2])
    _write_voice(tmp_path, "unsafe.name-low", {})
    (tmp_path / "notes.txt").write_text("hi")

    assert list(naming.scan_voice_dir(tmp_path)) == ["ok-voice-low"]


def test_scan_is_sorted_and_tolerates_a_missing_directory(tmp_path):
    for voice_id in ("z_Z-b-low", "a_A-b-low", "m_M-b-low"):
        _write_voice(tmp_path, voice_id, {})

    assert list(naming.scan_voice_dir(tmp_path)) == ["a_A-b-low", "m_M-b-low", "z_Z-b-low"]
    assert naming.scan_voice_dir(tmp_path / "missing") == {}
