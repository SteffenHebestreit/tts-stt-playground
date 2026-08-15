"""Dependency-light helpers for the Piper training service.

Extracted from app.py / model_exporter.py so the security- and
correctness-critical pure logic can be unit-tested without importing torch
or the rest of the training stack.
"""

import os
from pathlib import Path
from typing import Optional

from fastapi import HTTPException


def safe_name(value: str, field: str = "name") -> str:
    """Validate a model name or job id used to build filesystem paths.

    Rejects values that could escape the data/checkpoints/models roots
    (path separators, ``..``, NUL bytes, absolute paths). Normal voice
    names and UUID job ids pass through unchanged.
    """
    name = (value or "").strip()
    if (
        not name
        or name in {".", ".."}
        or "/" in name
        or "\\" in name
        or "\x00" in name
        or os.path.basename(name) != name
    ):
        raise HTTPException(status_code=400, detail=f"Invalid {field}: {value!r}")
    return name


# The UI offers 100-5000, but the API enforced nothing. `epochs=0` produces a
# job that reports 0/0 and divides by zero the moment anything computes a
# percentage; a very large value is not wrong so much as unbounded, and the
# operator has no way to tell it apart from a typo until hours later.
MIN_EPOCHS = 1
MAX_EPOCHS = 100_000


def validate_epochs(value: int, field: str = "epochs") -> int:
    """Bound a requested epoch count, rejecting values that cannot mean anything."""
    try:
        epochs = int(value)
    except (TypeError, ValueError, OverflowError):
        # OverflowError is the one that is easy to miss: int(float("inf"))
        # raises it rather than ValueError, so an infinite value escaped as a
        # 500 instead of the 400 every other bad input gets.
        raise HTTPException(status_code=400, detail=f"Invalid {field}: {value!r}")
    if not (MIN_EPOCHS <= epochs <= MAX_EPOCHS):
        raise HTTPException(
            status_code=400,
            detail=f"{field} must be between {MIN_EPOCHS} and {MAX_EPOCHS} (got {epochs})",
        )
    return epochs


def coerce_resume_int(value, default: int) -> int:
    """Coerce persisted checkpoint state values to positive integers."""
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def coerce_resume_path(value) -> Optional[Path]:
    """Coerce a persisted checkpoint path to a usable Path object."""
    if not isinstance(value, (str, os.PathLike)):
        return None
    text = str(value).strip()
    if not text:
        return None
    return Path(text)


def phoneme_id_map_from_entries(entries) -> dict:
    """Build a phoneme->id map from dataset metadata entries.

    Mirrors ``TTSDataset._create_phoneme_vocab()`` exactly:
    ``sorted(phonemes ∪ special_tokens)`` so the exported inference vocabulary
    matches the ids the model was trained with.
    """
    phonemes = set()
    for item in entries or []:
        if not isinstance(item, dict):
            continue
        phoneme_text = item.get("phonemes", item.get("text", ""))
        if phoneme_text:
            phonemes.update(list(phoneme_text))

    special_tokens = ["<pad>", "<unk>", "<start>", "<end>", " "]
    all_phonemes = sorted(phonemes.union(special_tokens))
    return {p: i for i, p in enumerate(all_phonemes)}
