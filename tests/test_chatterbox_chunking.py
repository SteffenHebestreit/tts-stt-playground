"""Offline tests for the chatterbox streaming text splitter.

`_split_sentences` decides time-to-first-audio for the streaming TTS endpoint,
and `/tts-stream` indexes `sentences[0]` directly — so a bad return value here is
either a 500 or a silent reversion to full-text latency.

The model stack is stubbed; these are pure-function tests.
"""

import sys
import types
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest

pytest.importorskip("numpy", reason="chatterbox app.py requires numpy")

SERVICE_DIR = Path(__file__).resolve().parents[1] / "chatterbox-tts-service"

MAX = 180
MIN = 60
FIRST = 40


def _install_stubs():
    """Stub the heavy audio/ML imports so app.py can be imported offline."""
    if "torch" not in sys.modules:
        torch = types.ModuleType("torch")
        torch.__version__ = "0.0.0-stub"
        torch.cuda = types.SimpleNamespace(
            is_available=lambda: False,
            get_device_name=lambda i: "stub",
            memory_allocated=lambda: 0,
            get_device_properties=lambda i: types.SimpleNamespace(total_memory=0),
            empty_cache=lambda: None,
        )
        torch.is_tensor = lambda x: False
        sys.modules["torch"] = torch

    if "soundfile" not in sys.modules:
        sf = types.ModuleType("soundfile")
        sf.write = lambda *a, **k: None
        sf.info = lambda p: types.SimpleNamespace(format="WAV", samplerate=16000, channels=1, frames=0)
        sys.modules["soundfile"] = sf

    if "uvicorn" not in sys.modules:
        uv = types.ModuleType("uvicorn")
        uv.run = lambda *a, **k: None
        sys.modules["uvicorn"] = uv


@pytest.fixture(scope="module")
def split():
    _install_stubs()
    # app.py does `from model_lifecycle import ...`, a sibling module, so the
    # service directory has to be importable — loading app.py by path alone is
    # not enough.
    sys.path.insert(0, str(SERVICE_DIR))
    try:
        spec = spec_from_file_location("chatterbox_app_under_test", SERVICE_DIR / "app.py")
        module = module_from_spec(spec)
        assert spec.loader is not None
        sys.modules["chatterbox_app_under_test"] = module
        spec.loader.exec_module(module)
        return module._split_sentences
    finally:
        sys.path.remove(str(SERVICE_DIR))


# --- Inputs that must never hang or crash ----------------------------------

@pytest.mark.parametrize("text", [
    "",
    "   ",
    "\n\n\n",
    ".",
    "a",
    "a" * 500,                       # one unbreakable token, far over the ceiling
    "wort " * 200,                   # no terminal punctuation at all
    "x," * 300,                      # commas only
    "Satz. " * 100,                  # many short sentences
    "\n".join(["Zeile"] * 100),
])
def test_terminates_and_returns_a_list(split, text):
    """The ceiling loop must always make progress — an infinite loop here hangs
    the whole service, since /tts-stream generates chunk 0 synchronously."""
    result = split(text, FIRST, MIN, MAX)
    assert isinstance(result, list)
    assert all(isinstance(chunk, str) for chunk in result)


def test_blank_input_returns_empty_list(split):
    """`/tts-stream` rejects blank text before calling this, but the contract
    must still be a list rather than [''], which would synthesize silence."""
    assert split("", FIRST, MIN, MAX) == []
    assert split("   \n  ", FIRST, MIN, MAX) == []


def test_non_blank_input_always_yields_a_first_chunk(split):
    """`/tts-stream` indexes sentences[0]; an empty list there is an IndexError."""
    for text in ("a", "Hallo.", "wort " * 200, "a" * 500):
        assert len(split(text, FIRST, MIN, MAX)) >= 1


# --- The ceiling is the whole point ----------------------------------------

def test_ceiling_is_respected_without_terminal_punctuation(split):
    """LLM output routinely lacks terminal punctuation. Before the ceiling
    existed this collapsed to one chunk and TTFA reverted to full-text latency."""
    text = "und dann sagte er dass es wirklich sehr viel besser waere wenn wir " * 6
    chunks = split(text, FIRST, MIN, MAX)
    assert len(chunks) > 1
    assert max(len(c) for c in chunks) <= MAX


def test_merge_step_does_not_reintroduce_oversized_chunks(split):
    """The merge loop used to be able to exceed the ceiling it had just
    enforced — and chunk 0 is exactly the one that sets TTFA."""
    text = "Kurz gesagt: " + ("sehr lange ausfuehrliche Erklaerung ohne Punkt " * 5)
    chunks = split(text, FIRST, MIN, MAX)
    assert max(len(c) for c in chunks) <= MAX


def test_no_chunk_starts_with_a_comma(split):
    """Cutting on a comma must keep the comma with the chunk it terminates.

    Otherwise the model speaks a leading ', ...' and the pause the comma encoded
    is missing from the end of the previous chunk.

    This text is chosen to actually exercise the comma branch: one sentence of
    202 chars (over the 180 ceiling) with its only ', ' at index 64 (past the
    60-char floor), so the splitter cuts there rather than on whitespace.
    """
    text = ("Wir haben das Thema gestern im Detail besprochen und ausgewertet, "
            "damit wirklich alle Beteiligten genau verstehen worum es hier eigentlich geht "
            "und was als naechstes zu tun ist ohne weitere Verzoegerung")
    assert len(text) > MAX and ", " in text[MIN:MAX]

    chunks = split(text, FIRST, MIN, MAX)
    assert len(chunks) > 1, "expected the ceiling to split this text"
    for chunk in chunks:
        assert not chunk.startswith(","), f"chunk starts with a comma: {chunk!r}"
        assert not chunk.startswith(" ")
    # The comma stays with the clause it closes.
    assert chunks[0].endswith(",")


def test_first_chunk_is_small_enough_to_help_ttfa(split):
    """Chunk 0 alone determines time-to-first-audio, so it gets a lower floor
    than the rest; a large first chunk defeats the endpoint's purpose."""
    text = "Guten Tag. " + ("Dies ist ein weiterer vollstaendiger Satz. " * 10)
    chunks = split(text, FIRST, MIN, MAX)
    assert len(chunks) > 1
    assert len(chunks[0]) <= MAX


def test_content_is_preserved(split):
    """Splitting must not lose or duplicate words."""
    text = ("Der erste Satz ist hier. Der zweite Satz folgt darauf. "
            "Und der dritte Satz beendet den Absatz.")
    chunks = split(text, FIRST, MIN, MAX)
    assert " ".join(chunks).split() == text.split()


def test_single_short_sentence_is_one_chunk(split):
    text = "Guten Tag."
    assert split(text, FIRST, MIN, MAX) == ["Guten Tag."]
