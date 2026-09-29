"""The language an exported voice is phonemised in must be the one it was trained in.

Training phonemises with `phonemization.espeak_voice()` (pt-BR -> `pt-br`, fr-BE ->
`fr-be`, en-GB -> `en-gb`, en -> `en-us`). The exporter kept a closed table of eight
primary codes with an `en-us` fallback for everything else, so a pt-BR voice was
trained on Brazilian Portuguese phonemes and exported with
`phonemizer_language: "en-us"`: piper-tts (which phonemises with exactly that field)
read Portuguese as English. fr-BE failed the same way and en-GB got the American
accent. The exporter now asks the same function training does.

The torch-free tests cover the function and, through the module source, that the
exporter has no table of its own. The end-to-end export (tiny model, ONNX, config
bundle) is in test_piper_training_core_export.py, which needs torch.
"""

from __future__ import annotations

import ast
import logging
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest

SERVICE_DIR = Path(__file__).resolve().parents[1] / "piper-training-service"


@pytest.fixture(scope="module")
def phonemization():
    """The real module, loaded under its own name so no other test's stand-ins are involved."""
    spec = spec_from_file_location("pt_phonemization_export", SERVICE_DIR / "phonemization.py")
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop(spec.name, None)


@pytest.mark.parametrize("language, expected", [
    ("pt-BR", "pt-br"),        # the reported case: trained as pt-br, exported as en-us
    ("pt_BR", "pt-br"),
    ("fr-BE", "fr-be"),
    ("en-GB", "en-gb"),
    ("en-US", "en-us"),
    ("en", "en-us"),
    ("de", "de"),
    ("de-AT", "de"),           # a regional tag espeak has no voice for reduces to its primary code
    ("fr", "fr-fr"),
    ("pt", "pt"),
    ("es", "es"), ("it", "it"), ("nl", "nl"), ("ru", "ru"),
])
def test_the_exported_voice_is_the_voice_training_used(phonemization, language, expected):
    assert phonemization.exported_voice(language) == expected
    assert phonemization.exported_voice(language) == phonemization.espeak_voice(language), (
        "the exporter and the dataset builder disagree about how this language is phonemised")


def test_every_language_training_accepts_exports_in_its_training_voice(phonemization):
    tags = set(phonemization.PHONEMIZER_LANGUAGES) | set(phonemization._REGIONAL_VOICES)
    assert {"pt-br", "fr-be", "en-gb"} <= tags
    for tag in sorted(tags):
        assert phonemization.exported_voice(tag) == phonemization.espeak_voice(tag), tag


def test_the_old_exporter_table_got_the_regional_voices_wrong(phonemization):
    """What the removed table answered, kept as the measure of what this fixes."""
    old_table = {'de': 'de', 'en': 'en-us', 'fr': 'fr-fr', 'es': 'es', 'it': 'it', 'nl': 'nl', 'pt': 'pt', 'ru': 'ru'}
    wrong = {tag: (old_table.get(tag, 'en-us'), phonemization.espeak_voice(tag))
             for tag in ("pt-br", "fr-be", "en-gb")}
    assert all(old != new for old, new in wrong.values()), wrong


def test_an_unsupported_tag_is_exported_as_en_us_and_says_so(phonemization, caplog):
    """Only a checkpoint trained before unsupported languages were refused can carry one. Those
    were phonemised as en-us, so that is what matches their symbols; the export is not blocked."""
    with caplog.at_level(logging.WARNING):
        assert phonemization.exported_voice("sv") == "en-us"
    assert any("'sv'" in r.getMessage() and "en-us" in r.getMessage() for r in caplog.records)


@pytest.mark.parametrize("value", [None, "", "  ", "klingon", "xx-YY"])
def test_a_missing_or_unknown_language_never_raises_during_export(phonemization, value):
    assert phonemization.exported_voice(value) == "en-us"


def test_the_exporter_has_no_language_table_of_its_own():
    """The single source of truth is phonemization.py: model_exporter must call it and
    must not carry a second mapping (the regression was exactly such a table)."""
    tree = ast.parse((SERVICE_DIR / "model_exporter.py").read_text(encoding="utf-8"))

    imported = {alias.name for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
                and node.module == "phonemization" for alias in node.names}
    assert "exported_voice" in imported

    tables = [node for node in ast.walk(tree) if isinstance(node, ast.Dict)
              and {"de", "fr", "pt"} <= {k.value for k in node.keys if isinstance(k, ast.Constant)}]
    assert not tables, "model_exporter.py defines its own language table again"

    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and getattr(node.func, "id", None) == "exported_voice"]
    assert calls, "the exporter no longer asks phonemization for the voice"
