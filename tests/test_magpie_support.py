"""Unit tests for magpie-tts-service/magpie_support.py and text_splitting.py (no torch, no NeMo).

Every rule here was measured on nvidia/magpie_tts_multilingual_357m with nemo_toolkit
3.0.0; the module docstring of magpie_support.py says what and why.
"""

import numpy as np
import pytest

from magpie_loader import CHECKPOINT_TOKENIZERS, LANGUAGE_TOKENIZER_MAP, load_support

support = load_support()


# --- languages ------------------------------------------------------------------------

def test_the_languages_nemo_can_route_are_those_whose_tokenizer_the_checkpoint_has():
    """NeMo's map names tokenizers; a code whose candidates the model lacks would be read with English rules."""
    assert support.supported_languages(LANGUAGE_TOKENIZER_MAP, CHECKPOINT_TOKENIZERS) == [
        "de", "en", "es", "fr", "ja", "zh"]


@pytest.mark.parametrize("code", ["it", "vi", "hi"])
def test_a_language_the_checkpoint_has_a_tokenizer_for_is_still_unsupported_when_nemos_map_cannot_reach_it(code):
    """The checkpoint holds Italian, Vietnamese and Hindi tokenizers under names the 3.0.0 map does not list."""
    assert code in LANGUAGE_TOKENIZER_MAP
    assert code not in support.supported_languages(LANGUAGE_TOKENIZER_MAP, CHECKPOINT_TOKENIZERS)


@pytest.mark.parametrize("value, expected", [
    ("de", "de"), ("DE", "de"), (" de ", "de"), ("de-DE", "de"), ("de_AT", "de"),
    ("German", "de"), ("deutsch", "de"), ("English", "en"), ("en-US", "en"), ("Español", "es"),
])
def test_a_language_is_accepted_as_a_code_a_locale_or_a_name(value, expected):
    assert support.resolve_language(value, "de", ["de", "en", "es"]) == expected


@pytest.mark.parametrize("value", [None, "", "  ", "auto", "AUTO"])
def test_blank_and_auto_mean_the_configured_default(value):
    assert support.resolve_language(value, "de", ["de", "en"]) == "de"


@pytest.mark.parametrize("value", ["nl", "Dutch", "xx", "it", "pt-BR"])
def test_an_unsupported_language_is_refused_and_the_message_lists_the_supported_ones(value):
    """NeMo would answer these with the English tokenizer and an ordinary success."""
    with pytest.raises(support.LanguageNotSupported) as exc:
        support.resolve_language(value, "de", ["de", "en", "fr"])
    assert value in str(exc.value)
    assert "de, en, fr" in str(exc.value)
    assert exc.value.supported == ["de", "en", "fr"]


def test_a_default_the_model_cannot_speak_fails_loudly_instead_of_speaking_english():
    with pytest.raises(support.LanguageNotSupported) as exc:
        support.resolve_language("auto", "nl", ["de", "en"])
    assert "nl" in str(exc.value)


# --- speakers -------------------------------------------------------------------------

LABELS = ["Aria", "Jason", "John", "Leo", "Sofia"]


@pytest.mark.parametrize("value, expected", [
    ("Sofia", 4), ("sofia", 4), ("SOFIA", 4), (" Leo ", 3), ("Aria", 0),
    (2, 2), ("2", 2), (" 4 ", 4), (0, 0),
])
def test_a_speaker_is_a_name_in_any_case_or_an_index(value, expected):
    assert support.resolve_speaker(value, LABELS, default_index=1) == expected


@pytest.mark.parametrize("value", [None, "", "  ", "auto", "Auto"])
def test_a_blank_speaker_is_the_default(value):
    assert support.resolve_speaker(value, LABELS, default_index=3) == 3


@pytest.mark.parametrize("value", ["Bob", 5, "5", -1, "-1", 99, True, "Sofia2"])
def test_an_unknown_speaker_is_refused_and_the_message_lists_the_valid_ones(value):
    """An out-of-range index is a ValueError inside the model; the caller should be told what is valid."""
    with pytest.raises(support.SpeakerNotFound) as exc:
        support.resolve_speaker(value, LABELS, default_index=0)
    assert "0 = Aria" in str(exc.value) and "4 = Sofia" in str(exc.value)


def test_speaker_names_are_parsed_in_order_without_duplicates():
    assert support.parse_speakers("Ann, Bob ,ann, Cy") == ("Ann", "Bob", "Cy")
    assert support.parse_speakers(None) == support.DEFAULT_SPEAKERS
    assert support.parse_speakers("  ") == support.DEFAULT_SPEAKERS
    assert support.parse_speakers(" , ,") == support.DEFAULT_SPEAKERS


def test_labels_follow_the_models_speaker_count():
    names = ("A", "B", "C")
    assert support.speaker_labels(names, 3) == ["A", "B", "C"]
    assert support.speaker_labels(names, 5) == ["A", "B", "C", "Speaker 3", "Speaker 4"]
    assert support.speaker_labels(names, 2) == ["A", "B"]
    assert support.speaker_labels(names, None) == ["A", "B", "C"]


def test_a_default_speaker_that_names_nobody_falls_back_to_the_first(caplog):
    with caplog.at_level("WARNING"):
        assert support.default_speaker_index("Nobody", LABELS) == 0
    assert "MAGPIE_DEFAULT_SPEAKER" in caplog.text
    assert support.default_speaker_index("Leo", LABELS) == 3


# --- text grouping --------------------------------------------------------------------

def test_german_ordinals_and_abbreviations_do_not_end_a_group():
    """"am 3. Mai", "ca. 5", "z. B." and "Dr. Müller" are not sentence ends: splitting there changes how it is read."""
    text = "Am 3. Mai kostet es ca. 12 Euro, z. B. für Dr. Müller. Danach gibt es Kaffee."
    groups = support.group_text(text, "de", 400)
    assert groups == [text]
    assert support.group_text(text, "de", 60) == [
        "Am 3. Mai kostet es ca. 12 Euro, z. B. für Dr. Müller.", "Danach gibt es Kaffee."]


def test_groups_never_exceed_the_ceiling_even_without_punctuation():
    text = " ".join(["wort"] * 200)
    groups = support.group_text(text, "de", 120)
    assert len(groups) > 1
    assert all(len(g) <= 120 for g in groups)
    assert " ".join(groups) == text


def test_groups_hold_whole_sentences_and_lose_nothing():
    sentences = [f"Das ist der {n} Satz dieses Textes." for n in ("erste", "zweite", "dritte", "vierte", "fünfte")]
    text = " ".join(sentences)
    groups = support.group_text(text, "de", 80)
    assert " ".join(groups) == text
    assert all(g.rstrip().endswith("Textes.") for g in groups)


@pytest.mark.parametrize("language", ["zh", "ja"])
def test_chinese_and_japanese_go_through_whole_because_nemo_has_sentence_rules_for_them(language):
    """They have no spaces and end sentences with marks the splitter does not know; cutting would cut mid-sentence."""
    text = "这是第一句话。这是第二句话。" * 40
    assert support.group_text(text, language, 60) == [text]


def test_blank_text_makes_no_groups_and_other_text_always_makes_one():
    assert support.group_text("   \n ", "de", 200) == []
    assert support.group_text("Hallo", "de", 200) == ["Hallo"]
    assert support.group_text("...", "de", 200) == ["..."]


# --- audio ----------------------------------------------------------------------------

def test_parts_are_joined_with_the_requested_silence_between_them_only():
    rate = 1000
    joined = support.join_audio([np.ones(100), np.ones(50) * 2, np.ones(10) * 3], rate, gap_ms=150)
    assert joined.dtype == np.float32
    assert joined.size == 100 + 150 + 50 + 150 + 10
    assert np.all(joined[100:250] == 0) and joined[0] == 1 and joined[-1] == 3


def test_empty_parts_are_skipped_and_no_parts_make_no_audio():
    assert support.join_audio([np.zeros(0), np.ones(5), np.zeros(0)], 1000, 150).size == 5
    assert support.join_audio([], 1000, 150).size == 0
    assert support.join_audio([np.zeros(0)], 1000, 150).size == 0


def test_a_zero_gap_joins_the_parts_directly():
    assert support.join_audio([np.ones(3), np.ones(4)], 22050, 0).size == 7


def test_two_dimensional_model_output_is_flattened():
    assert support.join_audio([np.ones((1, 8))], 1000, 0).shape == (8,)


# --- configuration --------------------------------------------------------------------

def test_a_junk_number_costs_the_knob_not_the_service(monkeypatch, caplog):
    monkeypatch.setenv("X_NUM", "banana")
    with caplog.at_level("WARNING"):
        assert support.env_number("X_NUM", 7, cast=int, minimum=1) == 7
    assert "X_NUM" in caplog.text
    monkeypatch.setenv("X_NUM", "0")
    assert support.env_number("X_NUM", 7, cast=int, minimum=1) == 7
    monkeypatch.setenv("X_NUM", "nan")
    assert support.env_number("X_NUM", 7.5, cast=float) == 7.5
    monkeypatch.setenv("X_NUM", "12")
    assert support.env_number("X_NUM", 7, cast=int, minimum=1) == 12
    monkeypatch.delenv("X_NUM")
    assert support.env_number("X_NUM", 7, cast=int) == 7


@pytest.mark.parametrize("raw, expected", [("1", True), ("true", True), (" YES ", True), ("on", True),
                                           ("0", False), ("false", False), ("nope", False)])
def test_flags(monkeypatch, raw, expected):
    monkeypatch.setenv("X_FLAG", raw)
    assert support.env_flag("X_FLAG", default=not expected) is expected


def test_an_unset_or_blank_flag_is_its_default(monkeypatch):
    monkeypatch.delenv("X_FLAG", raising=False)
    assert support.env_flag("X_FLAG", True) is True
    monkeypatch.setenv("X_FLAG", "  ")
    assert support.env_flag("X_FLAG", True) is True and support.env_flag("X_FLAG", False) is False


def test_a_model_path_is_shown_as_its_file_name_only():
    assert support.public_model_name("/data/models/magpie.nemo") == "magpie.nemo"
    assert support.public_model_name("nvidia/magpie_tts_multilingual_357m") == "nvidia/magpie_tts_multilingual_357m"
    assert support.public_model_name("") == ""


def test_cuda_out_of_memory_is_recognised_by_type_name_and_by_message():
    class OutOfMemoryError(RuntimeError):
        pass

    assert support.is_cuda_oom(OutOfMemoryError("x"))
    assert support.is_cuda_oom(RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB"))
    assert not support.is_cuda_oom(RuntimeError("shape mismatch"))
