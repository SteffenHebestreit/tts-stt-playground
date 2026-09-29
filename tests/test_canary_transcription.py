"""Unit tests for canary-asr-service/transcription.py (NeMo hypothesis parsing)."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace


def _load_transcription():
    """Load the standalone transcription module without importing torch/nemo."""
    path = Path(__file__).resolve().parents[1] / "canary-asr-service" / "transcription.py"
    spec = spec_from_file_location("canary_transcription", path)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


tr = _load_transcription()


def test_parse_plain_string():
    text, segments = tr.parse_hypothesis("  hallo welt  ")
    assert text == "hallo welt"
    assert segments == []


def test_parse_hypothesis_with_segments():
    hyp = SimpleNamespace(
        text=" guten tag ",
        timestamp={"segment": [
            {"start": 0.0, "end": 1.2, "segment": " guten "},
            {"start": 1.2, "end": 2.0, "segment": "tag"},
        ]},
    )
    text, segments = tr.parse_hypothesis(hyp)
    assert text == "guten tag"
    assert segments == [
        {"start": 0.0, "end": 1.2, "text": "guten"},
        {"start": 1.2, "end": 2.0, "text": "tag"},
    ]


def test_parse_hypothesis_no_timestamps():
    hyp = SimpleNamespace(text="x", timestamp=None)
    text, segments = tr.parse_hypothesis(hyp)
    assert text == "x"
    assert segments == []


def test_parse_hypothesis_uses_text_key_fallback():
    hyp = SimpleNamespace(text="full", timestamp={"segment": [{"start": 0, "end": 1, "text": "word"}]})
    _, segments = tr.parse_hypothesis(hyp)
    assert segments[0]["text"] == "word"


def test_parse_hypothesis_skips_malformed_segments():
    hyp = SimpleNamespace(text="t", timestamp={"segment": [
        {"start": "bad", "end": 1.0, "segment": "x"},  # non-numeric start -> skipped
        {"start": 0.0, "end": 2.0, "segment": "ok"},
    ]})
    _, segments = tr.parse_hypothesis(hyp)
    assert segments == [{"start": 0.0, "end": 2.0, "text": "ok"}]


def test_parse_hypothesis_missing_text_attr():
    hyp = SimpleNamespace(timestamp={})
    text, segments = tr.parse_hypothesis(hyp)
    assert text == ""
    assert segments == []


# --- language tables ---------------------------------------------------------------
#
# Sources: NeMo's checkpoint table (docs/source/asr/asr_checkpoints.rst) lists
# canary-180m-flash, canary-1b-flash and canary-1b as "EU4" and canary-1b-v2 as
# "EU25". The exact 25 codes were not verifiable against the tokenizer (the
# checkpoint is not downloadable here), see the comment in transcription.py.

import pytest


@pytest.mark.parametrize("model", [
    "nvidia/canary-180m-flash", "nvidia/canary-1b-flash", "nvidia/canary-1b", "/models/my-finetune.nemo",
    "", None,
])
def test_the_flash_models_and_unknown_names_decode_english_german_spanish_french(model):
    assert tr.canary_supported_languages(model) == {"en", "de", "es", "fr"}


@pytest.mark.parametrize("model", ["nvidia/canary-1b-v2", "NVIDIA/Canary-1B-V2", "/cache/canary-1b-v2.nemo"])
def test_canary_1b_v2_decodes_the_25_european_languages(model):
    languages = tr.canary_supported_languages(model)
    assert len(languages) == 25
    assert {"de", "en", "es", "fr", "it", "pl", "nl", "pt", "ru", "uk", "bg", "hr", "cs", "da", "et", "fi",
            "el", "hu", "lv", "lt", "mt", "ro", "sk", "sl", "sv"} == set(languages)


def test_the_override_wins_and_accepts_commas_spaces_and_locale_tags():
    assert tr.canary_supported_languages("nvidia/canary-180m-flash", "de, PL;it-IT  nl") == {"de", "pl", "it", "nl"}


@pytest.mark.parametrize("override", ["", "   ", ",,", "123", "de1"])
def test_an_unusable_override_is_ignored(override):
    assert tr.canary_supported_languages("nvidia/canary-1b-v2", override) == tr.CANARY_EU25_LANGUAGES


@pytest.mark.parametrize("raw, code", [
    ("de", "de"), ("DE", "de"), (" de-DE ", "de"), ("pt_BR", "pt"), ("zh-Hant-TW", "zh"), ("", ""), (None, ""),
])
def test_language_codes_reduce_to_their_primary_subtag(raw, code):
    assert tr.normalize_language_code(raw) == code


def test_known_models_are_recognised_by_name():
    assert tr.is_known_canary_model("nvidia/canary-180m-flash")
    assert tr.is_known_canary_model("nvidia/canary-1b-v2")
    assert not tr.is_known_canary_model("/models/whatever.nemo")
    assert not tr.is_known_canary_model("")
