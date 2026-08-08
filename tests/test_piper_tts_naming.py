"""Unit tests for piper-tts-service/naming.py (dependency-light helpers)."""

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


@pytest.mark.parametrize("value", ["", "../x", "a/b", "a b", "naïve", "a.b", "a\\b", None])
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
