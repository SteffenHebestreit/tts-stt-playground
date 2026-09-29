"""Unit tests for stt-service/json_utils.py (dependency-light helpers)."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


def _load_json_utils():
    """Load the standalone json_utils module without importing torch/faster-whisper."""
    path = Path(__file__).resolve().parents[1] / "stt-service" / "json_utils.py"
    spec = spec_from_file_location("stt_json_utils", path)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


ju = _load_json_utils()


def test_clean_replaces_nan_and_inf():
    data = {
        "a": float("nan"),
        "b": float("inf"),
        "c": float("-inf"),
        "d": 1.5,
        "e": "ok",
        "f": [float("nan"), 2, {"g": float("inf")}],
    }
    cleaned = ju.clean_json_inf_nan(data)
    assert cleaned["a"] is None
    assert cleaned["b"] is None
    assert cleaned["c"] is None
    assert cleaned["d"] == 1.5
    assert cleaned["e"] == "ok"
    assert cleaned["f"] == [None, 2, {"g": None}]


def test_clean_passthrough_finite_and_non_float():
    assert ju.clean_json_inf_nan(0.0) == 0.0
    assert ju.clean_json_inf_nan(5) == 5
    assert ju.clean_json_inf_nan("x") == "x"
    assert ju.clean_json_inf_nan(None) is None
    assert ju.clean_json_inf_nan([]) == []


def test_clean_result_is_json_serialisable():
    import json
    cleaned = ju.clean_json_inf_nan({"v": float("nan"), "list": [float("inf")]})
    # allow_nan=False would raise if any NaN/Inf survived
    json.dumps(cleaned, allow_nan=False)


def test_is_multilingual_flags():
    assert ju.is_multilingual("large-v3") is True
    assert ju.is_multilingual("medium") is True
    assert ju.is_multilingual("tiny.en") is False
    assert ju.is_multilingual("distil-large-v3") is False


def test_english_only_set_membership():
    assert "small.en" in ju.ENGLISH_ONLY_MODELS
    assert "large-v3" not in ju.ENGLISH_ONLY_MODELS


def test_is_multilingual_understands_repo_ids_and_paths():
    assert ju.is_multilingual("primeline/whisper-large-v3-turbo-german") is True
    assert ju.is_multilingual("/app/models/whisper-large-v3-ct2") is True
    assert ju.is_multilingual("someone/whisper-small.en") is False
    assert ju.is_multilingual("/app/models/distil-large-v3") is False
    assert ju.is_multilingual("distil-large-v3.5") is False


def test_translate_capability_is_judged_by_model_name():
    for name in ("large-v3", "large-v2", "medium", "small", "org/whisper-large-v3-german-ct2"):
        assert ju.model_can_translate(name) is True, name
    for name in (
        "large-v3-turbo", "turbo", "distil-large-v3", "tiny.en",
        "primeline/whisper-large-v3-turbo-german", "/app/models/my-turbo-ct2",
    ):
        assert ju.model_can_translate(name) is False, name


def test_custom_model_detection(tmp_path):
    assert ju.is_custom_model("primeline/whisper-large-v3-turbo-german") is True
    assert ju.is_custom_model("/app/models/whisper-ct2") is True
    assert ju.is_custom_model(str(tmp_path)) is True
    assert ju.is_custom_model("large-v3") is False
    assert ju.is_custom_model("large-v4") is False, "a mistyped size is not a custom model"


def test_normalize_language_reduces_locale_tags():
    assert ju.normalize_language("de") == "de"
    assert ju.normalize_language(" DE-de ") == "de"
    assert ju.normalize_language("de_AT") == "de"
    assert ju.normalize_language("pt-BR") == "pt"
    assert ju.normalize_language("zh-Hant") == "zh"
    assert ju.normalize_language("haw") == "haw"
    assert ju.normalize_language("yue") == "yue"


def test_normalize_language_maps_auto_and_blank_to_none():
    for value in (None, "", "  ", "auto", "AUTO"):
        assert ju.normalize_language(value) is None


def test_normalize_language_maps_iso_variants_whisper_spells_differently():
    assert ju.normalize_language("iw") == "he"
    assert ju.normalize_language("jv") == "jw"
    assert ju.normalize_language("nb") == "no"


def test_normalize_language_rejects_what_whisper_cannot_decode():
    import pytest

    for value in ("klingon", "xx", "123", "german", "-de"):
        with pytest.raises(ValueError):
            ju.normalize_language(value)


def test_the_language_list_is_whispers_hundred_codes():
    assert len(ju.WHISPER_LANGUAGES) == 100 == len(set(ju.WHISPER_LANGUAGES))
    assert {"de", "en", "yue"} <= set(ju.WHISPER_LANGUAGES)
