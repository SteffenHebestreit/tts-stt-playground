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
