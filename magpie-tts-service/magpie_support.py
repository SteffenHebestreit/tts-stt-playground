"""Pure helpers for the Magpie TTS service: no torch, no NeMo, no FastAPI.

app.py keeps the model, the request gate and the HTTP surface; everything here is a
plain function so the tests can exercise it without any of them. What the functions
decide was measured on nvidia/magpie_tts_multilingual_357m with nemo_toolkit 3.0.0:

* ``MagpieTTSModel.do_tts`` picks its tokenizer with ``get_tokenizer_for_language``,
  which answers ``english_phoneme`` for any language it does not know. A request for
  Dutch, or for Italian (whose tokenizer the checkpoint has, under a name NeMo 3.0.0's
  language map does not list), is therefore read aloud with English rules and comes
  back as an ordinary 200. ``supported_languages`` lists only the languages whose
  tokenizer NeMo will really route to, and ``resolve_language`` refuses the rest.
* Without text normalization the phoneme tokenizer drops every digit ("3. Mai 2026",
  "12,50 Euro", "14:30 Uhr" are simply not spoken), so normalization is on by default.
"""

from __future__ import annotations

import logging
import math
import os
from typing import Iterable, Mapping, Optional, Sequence, Union

import numpy as np

import text_splitting

logger = logging.getLogger(__name__)

# The checkpoint's baked speaker embeddings, in order (index = position). The model
# knows only the count; the names are the ones on its Hugging Face card.
DEFAULT_SPEAKERS = ("Aria", "Jason", "John", "Leo", "Sofia")

# Languages the checkpoint speaks and NeMo 3.0.0 routes correctly, used before the
# model has been loaded (and therefore before its tokenizers can be read).
DOCUMENTED_LANGUAGES = ("de", "en", "es", "fr", "ja", "zh")

# Text in these languages has no spaces and ends sentences with marks the splitter
# does not know, so cutting it into groups would cut mid-sentence. NeMo has its own
# sentence rules for them (tts_dataset_utils._SENTENCE_ENDINGS); the text goes through whole.
SELF_CHUNKED_LANGUAGES = frozenset({"ja", "zh"})

# Written-out names an operator or a caller is likely to use for a language.
LANGUAGE_NAMES = {
    "english": "en", "german": "de", "deutsch": "de", "spanish": "es", "espanol": "es", "español": "es",
    "french": "fr", "francais": "fr", "français": "fr", "japanese": "ja", "chinese": "zh", "mandarin": "zh",
    "italian": "it", "portuguese": "pt", "korean": "ko", "hindi": "hi", "arabic": "ar", "vietnamese": "vi",
    "dutch": "nl",
}


class LanguageNotSupported(ValueError):
    """The language is not one this deployment can speak; the message lists the ones it can."""

    def __init__(self, requested: str, supported: Sequence[str]):
        self.requested = requested
        self.supported = list(supported)
        super().__init__(
            f"Language '{requested}' is not supported by this Magpie deployment. "
            f"Supported: {', '.join(self.supported) or 'none'}."
        )


class SpeakerNotFound(ValueError):
    """The speaker is neither a known name nor an index in range; the message lists the valid ones."""

    def __init__(self, requested: object, labels: Sequence[str]):
        self.requested = requested
        self.labels = list(labels)
        listing = ", ".join(f"{i} = {name}" for i, name in enumerate(self.labels))
        super().__init__(f"Speaker {requested!r} is not available. Use a name or an index: {listing}.")


# --- configuration -------------------------------------------------------------------

def env_number(name: str, default, *, cast=float, minimum=None):
    """Parse a numeric env var; junk or out-of-range input falls back to *default*.

    A typo in a tuning knob should cost the knob, not stop the service at import.
    """
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = cast(raw.strip())
    except ValueError:
        logger.warning("Ignoring invalid %s=%r; using %s", name, raw, default)
        return default
    if not math.isfinite(value) or (minimum is not None and value < minimum):
        logger.warning("Ignoring %s=%r (not usable); using %s", name, raw, default)
        return default
    return value


def env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def public_model_name(name: str) -> str:
    """A model name as an unauthenticated response may show it (a ``.nemo`` path becomes its file name)."""
    if not name:
        return name
    if os.path.isabs(name) or name.startswith(("./", "../", "~")):
        return os.path.basename(os.path.normpath(name)) or name
    return name


def is_cuda_oom(exc: BaseException) -> bool:
    """Whether *exc* is a CUDA out-of-memory error, by type name or message (no torch import needed)."""
    return type(exc).__name__ == "OutOfMemoryError" or "CUDA out of memory" in str(exc)


# --- languages -----------------------------------------------------------------------

def normalize_language(value: Optional[str]) -> str:
    """``"de-DE"``, ``"de_AT"``, ``"German"`` and ``" DE "`` all become ``"de"``; blank stays blank."""
    text = (value or "").strip().lower().replace("-", "_")
    if not text:
        return ""
    if text in LANGUAGE_NAMES:
        return LANGUAGE_NAMES[text]
    return text.split("_")[0]


def supported_languages(language_map: Mapping[str, Sequence[str]], tokenizer_names: Iterable[str]) -> list[str]:
    """Language codes NeMo will route to a tokenizer the loaded model really has.

    ``language_map`` is ``tts_dataset_utils.LANGUAGE_TOKENIZER_MAP`` (code -> candidate
    tokenizer names, best first). A code whose candidates the model lacks is left out:
    ``get_tokenizer_for_language`` would answer ``english_phoneme`` for it.
    """
    available = set(tokenizer_names)
    return sorted(code for code, candidates in language_map.items() if any(c in available for c in candidates))


def resolve_language(value: Optional[str], default: str, supported: Sequence[str]) -> str:
    """The language code to synthesize in; ``LanguageNotSupported`` for one the model cannot speak.

    Blank and ``"auto"`` mean the configured default, which is validated like any other
    value, so a misconfigured default fails loudly instead of speaking English.
    """
    requested = normalize_language(value)
    code = default if requested in ("", "auto") else requested
    if code not in supported:
        raise LanguageNotSupported(code if requested in ("", "auto") else (value or code).strip(), supported)
    return code


# --- speakers ------------------------------------------------------------------------

def parse_speakers(raw: Optional[str]) -> tuple[str, ...]:
    """Comma-separated speaker names, in baked-embedding order; blank means the checkpoint's own five."""
    if raw is None or not raw.strip():
        return DEFAULT_SPEAKERS
    names: list[str] = []
    for item in raw.split(","):
        name = item.strip()
        if name and name.lower() not in {n.lower() for n in names}:
            names.append(name)
    return tuple(names) or DEFAULT_SPEAKERS


def speaker_labels(names: Sequence[str], count: Optional[int]) -> list[str]:
    """One label per baked speaker: the configured names, padded with ``Speaker N`` or cut to *count*."""
    if not count or count < 1:
        return list(names)
    return [names[i] if i < len(names) else f"Speaker {i}" for i in range(count)]


def resolve_speaker(value: Union[str, int, None], labels: Sequence[str], default_index: int) -> int:
    """The baked-speaker index for a name (any case) or an index (an int or digits); ``SpeakerNotFound`` otherwise."""
    if value is None or (isinstance(value, str) and value.strip().lower() in ("", "auto")):
        return default_index
    if isinstance(value, bool):
        raise SpeakerNotFound(value, labels)
    if isinstance(value, int) or (isinstance(value, str) and value.strip().isdigit()):
        index = int(value)
        if 0 <= index < len(labels):
            return index
        raise SpeakerNotFound(value, labels)
    wanted = str(value).strip().lower()
    for index, label in enumerate(labels):
        if label.lower() == wanted:
            return index
    raise SpeakerNotFound(value, labels)


def default_speaker_index(name: str, labels: Sequence[str]) -> int:
    """Index of the configured default speaker; the first speaker (with a warning) if it names none."""
    try:
        return resolve_speaker(name, labels, 0)
    except SpeakerNotFound:
        logger.warning("MAGPIE_DEFAULT_SPEAKER=%r is not one of %s; using %r", name, list(labels), labels[0])
        return 0


# --- text and audio ------------------------------------------------------------------

def group_text(text: str, language: str, max_chars: int) -> list[str]:
    """Whole-sentence groups, each short enough for one single-chunk generation; never empty for non-blank text."""
    text = text.strip()
    if not text:
        return []
    if language in SELF_CHUNKED_LANGUAGES:
        return [text]
    return text_splitting.split_for_synthesis(text, max_chars) or [text]


def join_audio(parts: Sequence[np.ndarray], sample_rate: int, gap_ms: int) -> np.ndarray:
    """Mono float32 samples of *parts* in order, *gap_ms* of silence between them; empty parts are skipped."""
    pieces = [np.asarray(p, dtype=np.float32).reshape(-1) for p in parts]
    pieces = [p for p in pieces if p.size]
    if not pieces:
        return np.zeros(0, dtype=np.float32)
    gap = np.zeros(int(sample_rate * max(0, gap_ms) / 1000), dtype=np.float32)
    out: list[np.ndarray] = []
    for i, piece in enumerate(pieces):
        if i and gap.size:
            out.append(gap)
        out.append(piece)
    return np.concatenate(out)
