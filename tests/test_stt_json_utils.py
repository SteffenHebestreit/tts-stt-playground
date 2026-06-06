"""Unit tests for stt-service/json_utils.py (dependency-light helpers)."""

import math
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
