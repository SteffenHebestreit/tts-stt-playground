"""Dependency-light helpers for the PiperTTS service.

Extracted from app.py so the name-sanitisation, voice-selection, and output
pruning logic can be unit-tested without importing onnxruntime / librosa / piper.
"""

import os
import re
import time
from typing import Optional

from fastapi import HTTPException

# Voice/model names may only contain alphanumerics, dash, and underscore.
SAFE_NAME_RE = re.compile(r"^[a-zA-Z0-9_\-]+$")


def sanitize_voice_name(name: str) -> str:
    """Return *name* if it contains only safe characters, otherwise raise 400."""
    if not name or not SAFE_NAME_RE.match(name):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid voice name '{name}': only alphanumeric, dash, and underscore allowed.",
        )
    return name


def select_best_voice(all_voices: dict, language: str, quality: str, gender: Optional[str] = None) -> str:
    """Pick the best voice from *all_voices* matching language/quality/optional gender.

    *all_voices* maps voice id -> object exposing ``.language``, ``.quality`` and
    optionally ``.gender``. Falls back to any matching-language voice, then any
    English voice, then the hard-coded default ``en_US-lessac-medium``.
    """
    base_language = (language or "").split("_")[0]

    matching = [(n, v) for n, v in all_voices.items() if v.language.startswith(base_language)]
    if not matching:
        matching = [(n, v) for n, v in all_voices.items() if v.language.startswith("en")]
    if not matching:
        return "en_US-lessac-medium"

    preferred = [(n, v) for n, v in matching if v.quality == quality]
    if not preferred:
        preferred = matching

    if gender:
        gendered = [(n, v) for n, v in preferred if getattr(v, "gender", None) == gender]
        if gendered:
            preferred = gendered

    return preferred[0][0]


def prune_old_outputs(output_dir: str, retention_hours: float, now: Optional[float] = None) -> int:
    """Best-effort removal of files in *output_dir* older than *retention_hours*.

    Returns the number of files removed. ``retention_hours <= 0`` disables pruning.
    """
    if retention_hours <= 0:
        return 0
    cutoff = (now if now is not None else time.time()) - retention_hours * 3600
    removed = 0
    try:
        for entry in os.scandir(output_dir):
            if not entry.is_file():
                continue
            try:
                if entry.stat().st_mtime < cutoff:
                    os.unlink(entry.path)
                    removed += 1
            except OSError:
                pass
    except FileNotFoundError:
        pass
    return removed
