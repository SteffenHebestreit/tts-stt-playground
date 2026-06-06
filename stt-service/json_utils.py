"""Dependency-light JSON/model helpers for the STT service.

Extracted from app.py so they can be unit-tested without importing torch or
faster-whisper.
"""

import math

# faster-whisper model variants that only support English.
ENGLISH_ONLY_MODELS = {
    "tiny.en", "base.en", "small.en", "medium.en",
    "distil-large-v2", "distil-large-v3", "distil-medium.en", "distil-small.en",
}


def clean_json_inf_nan(data):
    """Recursively replace float inf/NaN with ``None`` so JSON serialisation succeeds."""
    if isinstance(data, dict):
        return {k: clean_json_inf_nan(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [clean_json_inf_nan(i) for i in data]
    elif isinstance(data, float):
        if math.isinf(data) or math.isnan(data):
            return None  # Replace with null in JSON
        return data
    return data


def is_multilingual(model_size: str) -> bool:
    """Return True if *model_size* is a multilingual (not English-only) Whisper model."""
    return model_size not in ENGLISH_ONLY_MODELS
