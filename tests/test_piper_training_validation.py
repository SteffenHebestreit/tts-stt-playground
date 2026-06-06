"""Unit tests for piper-training-service/validation.py (dependency-light helpers)."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest
from fastapi import HTTPException


def _load_validation():
    """Load the standalone validation module without importing the torch stack."""
    path = Path(__file__).resolve().parents[1] / "piper-training-service" / "validation.py"
    spec = spec_from_file_location("pt_validation", path)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


validation = _load_validation()


# --- safe_name (path-traversal guard) ---

@pytest.mark.parametrize("value", ["luna", "my_voice", "Voice-1", "a1b2c3", "550e8400-e29b-41d4-a716-446655440000"])
def test_safe_name_accepts_normal_names(value):
    assert validation.safe_name(value) == value


def test_safe_name_strips_whitespace():
    assert validation.safe_name("  luna  ") == "luna"


@pytest.mark.parametrize("value", [
    "../etc", "..", ".", "", "   ",
    "a/b", "a\\b", "/abs", "C:\\x",
    "foo/../bar", "name\x00", "sub/dir",
])
def test_safe_name_rejects_traversal(value):
    with pytest.raises(HTTPException) as exc:
        validation.safe_name(value, field="model_name")
    assert exc.value.status_code == 400
    assert "model_name" in exc.value.detail


# --- coerce_resume_int ---

@pytest.mark.parametrize("value,default,expected", [
    (5, 1, 5),
    ("12", 1, 12),
    (0, 7, 7),       # non-positive -> default
    (-3, 7, 7),
    (None, 9, 9),
    ("nope", 9, 9),
    (2.9, 1, 2),     # int() truncates
])
def test_coerce_resume_int(value, default, expected):
    assert validation.coerce_resume_int(value, default) == expected


# --- coerce_resume_path ---

def test_coerce_resume_path_valid():
    p = validation.coerce_resume_path("checkpoints/job/ck.pt")
    assert p is not None
    assert p.name == "ck.pt"


@pytest.mark.parametrize("value", [None, "", "   ", 123, [], {}])
def test_coerce_resume_path_invalid(value):
    assert validation.coerce_resume_path(value) is None


# --- phoneme_id_map_from_entries (must match TTSDataset ordering) ---

def test_phoneme_map_ordering_and_specials():
    entries = [{"phonemes": "ab"}, {"phonemes": "ba c"}]
    result = validation.phoneme_id_map_from_entries(entries)
    # sorted(set("abc ") ∪ {<pad>,<unk>,<start>,<end>,' '})
    expected_keys = sorted({"a", "b", "c", " ", "<pad>", "<unk>", "<start>", "<end>"})
    assert list(result.keys()) == expected_keys
    # ids are a contiguous 0..n-1 range in sorted order
    assert list(result.values()) == list(range(len(expected_keys)))


def test_phoneme_map_falls_back_to_text_field():
    result = validation.phoneme_id_map_from_entries([{"text": "xy"}])
    assert "x" in result and "y" in result


def test_phoneme_map_specials_only_when_empty():
    result = validation.phoneme_id_map_from_entries([])
    assert set(result.keys()) == {"<pad>", "<unk>", "<start>", "<end>", " "}


def test_phoneme_map_ignores_non_dict_entries():
    result = validation.phoneme_id_map_from_entries(["not-a-dict", None, {"phonemes": "z"}])
    assert "z" in result
