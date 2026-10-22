"""IBM Granite Speech specifics for granite-asr-service (no torch or transformers at import).

Two kinds of checkpoint are served, told apart by the ``model_type`` in their
config.json:

* ``speech-llm`` (``granite_speech``): ``ibm-granite/granite-speech-5.0-2b``, a
  conformer encoder and a projector in front of granite-4.0-1b. It follows a text
  prompt: the service names the language in it (the model card's language-ID
  prompting, which also keeps it from translating), asks for punctuation and
  capitalisation, and appends a keyword list for keyword biasing.
* ``ctc`` (``granite_speech5_ctc``): ``ibm-granite/granite-speech-5.0-470m-turboctc``,
  a 470M conformer with a CTC head, decoded greedily in one forward pass. It takes
  no prompt at all: the language is checked and reported but does not steer it, and
  keywords cannot be applied.

Neither produces timestamps.
"""

from __future__ import annotations

import re
from typing import Iterable, Optional

FAMILY_LLM = "speech-llm"
FAMILY_CTC = "ctc"
MODEL_TYPES = {"granite_speech": FAMILY_LLM, "granite_speech5_ctc": FAMILY_CTC}

DEFAULT_MODEL = "ibm-granite/granite-speech-5.0-2b"
TURBOCTC_MODEL = "ibm-granite/granite-speech-5.0-470m-turboctc"

# Both repositories were last changed on 2026-10-21 (the 470m became multilingual
# that day); a benchmark is only worth something if the weights it measured are the
# ones that run, so each known model is pinned to the commit that was measured.
KNOWN_MODELS = {
    DEFAULT_MODEL: {
        "revision": "f4e68169fa2a5751acc4b5da9957855e5e8fc3f1",
        "family": FAMILY_LLM,
        "languages": frozenset({"en", "fr", "de", "es", "pt", "ja"}),
    },
    TURBOCTC_MODEL: {
        "revision": "c85a3307b6abec4a169daaf11e8dfa80ac27ad81",
        "family": FAMILY_CTC,
        "languages": frozenset({"en", "fr", "de", "es", "pt", "it", "ja"}),
    },
}
# For a checkpoint not in the table (a fine-tune, a local directory): what its kind
# of model supports. GRANITE_SUPPORTED_LANGUAGES overrides either.
LANGUAGES_BY_FAMILY = {
    FAMILY_LLM: KNOWN_MODELS[DEFAULT_MODEL]["languages"],
    FAMILY_CTC: KNOWN_MODELS[TURBOCTC_MODEL]["languages"],
}

# How the speech LLM's prompt names a language (English names, as on the model card).
LANGUAGE_NAMES = {
    "en": "English", "fr": "French", "de": "German", "es": "Spanish",
    "pt": "Portuguese", "ja": "Japanese", "it": "Italian",
}

# The model card's prompts (its "Preferred prompt by task" table).
PROMPT_PUNCTUATED = "transcribe the speech with proper punctuation and capitalization."
PROMPT_LANGUAGE = "The speech is in {language}. Transcribe it exactly; do not translate. "
KEYWORDS_PREFIX = " Keywords: "

# The model card's limit for one keyword list.
MAX_KEYWORDS = 2000
MAX_KEYWORD_CHARS = 100
_KEYWORD_SPLIT = re.compile(r"[,;\n\r]+")


class KeywordError(ValueError):
    """A keyword list the service refuses; ``str()`` is a message for the client."""


def default_revision(model: str) -> str:
    known = KNOWN_MODELS.get(model)
    return known["revision"] if known else "main"


def family_for(model_type: Optional[str]) -> str:
    """The kind of checkpoint a config.json ``model_type`` names; ValueError for one this service cannot run."""
    family = MODEL_TYPES.get((model_type or "").strip())
    if family is None:
        raise ValueError(
            f"model_type {model_type!r} is not a Granite Speech checkpoint this service can run "
            f"(supported: {', '.join(sorted(MODEL_TYPES))})")
    return family


def supported_languages(model: str, family: str, override: Optional[str] = None) -> frozenset:
    """Languages to accept: *override* (``GRANITE_SUPPORTED_LANGUAGES``) when it names any,
    else the model's own list, else what its kind of model supports."""
    if override:
        codes = {part.strip().lower().replace("_", "-").split("-", 1)[0]
                 for part in re.split(r"[,\s;]+", override)}
        codes = {code for code in codes if code.isalpha()}
        if codes:
            return frozenset(codes)
    known = KNOWN_MODELS.get(model)
    if known:
        return known["languages"]
    return LANGUAGES_BY_FAMILY[family]


def parse_keywords(*values: Optional[str]) -> list[str]:
    """The keyword list of a request: every value split at commas, semicolons and line breaks.

    The ``keywords`` form field and ``initial_prompt`` (what the gateway's /v1 sends
    for OpenAI's ``prompt``) are both read this way, so ``"LUNA, Bautzen, Satellite"``
    biases towards those three words. Duplicates (ignoring case) are dropped, the
    first spelling kept. More than ``MAX_KEYWORDS`` keywords, or one longer than
    ``MAX_KEYWORD_CHARS`` characters, is refused: silently using part of a list
    would bias towards a list nobody asked for.
    """
    keywords: list[str] = []
    seen: set[str] = set()
    for value in values:
        for part in _KEYWORD_SPLIT.split(value or ""):
            word = " ".join(part.split())
            if not word:
                continue
            if len(word) > MAX_KEYWORD_CHARS:
                raise KeywordError(
                    f"Keyword '{word[:40]}...' is longer than {MAX_KEYWORD_CHARS} characters. "
                    f"Keywords are separated by commas, semicolons or line breaks.")
            folded = word.casefold()
            if folded in seen:
                continue
            seen.add(folded)
            keywords.append(word)
    if len(keywords) > MAX_KEYWORDS:
        raise KeywordError(f"{len(keywords)} keywords; the model takes at most {MAX_KEYWORDS}.")
    return keywords


def build_prompt(language: Optional[str], keywords: Iterable[str] = ()) -> str:
    """The user turn for the speech LLM: language, task, keywords (the audio token goes first).

    *language* is an ISO code from ``LANGUAGE_NAMES`` or None (no language named).
    """
    text = ""
    if language:
        text += PROMPT_LANGUAGE.format(language=LANGUAGE_NAMES.get(language, language))
    text += PROMPT_PUNCTUATED
    words = [w for w in keywords if w]
    if words:
        text += KEYWORDS_PREFIX + ", ".join(words)
    return "<|audio|>" + text
