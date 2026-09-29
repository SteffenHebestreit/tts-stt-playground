"""Unit tests for parakeet-asr-service/transcription.py (NeMo hypothesis parsing)."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace


def _load_transcription():
    """Load the standalone transcription module without importing torch/nemo."""
    path = Path(__file__).resolve().parents[1] / "parakeet-asr-service" / "transcription.py"
    spec = spec_from_file_location("parakeet_transcription", path)
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


def test_the_two_nemo_services_parse_hypotheses_identically():
    """Both services decode the same Hypothesis shape; a fix to one parser belongs in both."""
    path = Path(__file__).resolve().parents[1] / "canary-asr-service" / "transcription.py"
    spec = spec_from_file_location("canary_transcription_for_comparison", path)
    canary = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(canary)

    samples = [
        "  hallo  ",
        SimpleNamespace(text=" x ", timestamp={"segment": [{"start": 0, "end": 1.5, "segment": " a "}]}),
        SimpleNamespace(text="t", timestamp={"segment": [{"start": "bad", "end": 1, "segment": "x"}]}),
        SimpleNamespace(timestamp=None),
    ]
    for sample in samples:
        assert tr.parse_hypothesis(sample) == canary.parse_hypothesis(sample)
