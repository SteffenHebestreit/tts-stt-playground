"""Dependency-light helpers for the Canary ASR service.

Extracted from app.py so the NeMo-hypothesis parsing and the language tables can
be unit-tested without importing torch / nemo_toolkit / librosa.
"""

import re
from typing import Optional


def parse_hypothesis(hyp) -> tuple:
    """Convert one NeMo hypothesis (or plain string) into ``(text, segments)``.

    Handles both the plain-string and Hypothesis-object shapes NeMo may return,
    and defends against malformed timestamp entries.
    """
    if isinstance(hyp, str):
        return hyp.strip(), []

    text = (getattr(hyp, "text", "") or "").strip()
    timestamp = getattr(hyp, "timestamp", None) or {}
    raw_segments = timestamp.get("segment", []) if isinstance(timestamp, dict) else []

    segments = []
    for seg in raw_segments:
        try:
            segments.append({
                "start": float(seg.get("start", 0.0) or 0.0),
                "end": float(seg.get("end", 0.0) or 0.0),
                "text": (seg.get("segment") or seg.get("text") or "").strip(),
            })
        except (TypeError, ValueError, AttributeError):
            continue
    return text, segments


# The languages of NeMo's "EU4" checkpoints: canary-180m-flash, canary-1b-flash
# and canary-1b (NeMo docs/source/asr/asr_checkpoints.rst).
CANARY_EU4_LANGUAGES = frozenset({"en", "de", "es", "fr"})

# canary-1b-v2 ("EU25"). NeMo's checkpoint table lists Belarusian in its EU25
# glossary as well, which the canary-1b-v2 model card does not, so only the 25
# languages both agree on are listed; a deployment that has verified more can add
# them with CANARY_SUPPORTED_LANGUAGES.
CANARY_EU25_LANGUAGES = frozenset({
    "bg", "hr", "cs", "da", "nl", "en", "et", "fi", "fr", "de", "el", "hu", "it",
    "lv", "lt", "mt", "pl", "pt", "ro", "sk", "sl", "es", "sv", "ru", "uk",
})


def normalize_language_code(raw: Optional[str]) -> str:
    """``"de-DE"`` / ``"de_DE"`` / ``" DE "`` -> ``"de"``; empty stays empty."""
    return (raw or "").strip().lower().replace("_", "-").split("-", 1)[0]


def is_known_canary_model(model_name: str) -> bool:
    """True when *model_name* is a checkpoint whose language set is tabulated here."""
    name = (model_name or "").lower()
    # "canary-1b" also covers canary-1b-flash and canary-1b-v2
    return "canary-1b" in name or "canary-180m-flash" in name


def canary_supported_languages(model_name: str, override: Optional[str] = None) -> frozenset:
    """The decoder languages to accept for *model_name*.

    *override* (``CANARY_SUPPORTED_LANGUAGES``, comma or space separated) wins
    when it names at least one language. Otherwise the set follows the model: the
    25 European languages for canary-1b-v2, en/de/es/fr for the flash models and
    canary-1b. A name that is not recognised (a local .nemo path, a fine-tune)
    gets en/de/es/fr, the historic behaviour.
    """
    if override:
        codes = {normalize_language_code(part) for part in re.split(r"[,\s;]+", override)}
        codes = {code for code in codes if code.isalpha()}
        if codes:
            return frozenset(codes)
    if "canary-1b-v2" in (model_name or "").lower():
        return CANARY_EU25_LANGUAGES
    return CANARY_EU4_LANGUAGES
