"""Dependency-light JSON/model helpers for the STT service.

Extracted from app.py so they can be unit-tested without importing torch or
faster-whisper.
"""

import math
import os

# faster-whisper model variants that only support English.
ENGLISH_ONLY_MODELS = {
    "tiny.en", "base.en", "small.en", "medium.en",
    "distil-large-v2", "distil-large-v3", "distil-large-v3.5",
    "distil-medium.en", "distil-small.en",
}

# Language codes Whisper's tokenizer accepts (faster_whisper.tokenizer
# _LANGUAGE_CODES, large-v3 incl. Cantonese). Kept here rather than imported so
# the live path can validate a client-supplied code without loading torch.
WHISPER_LANGUAGES = (
    "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr", "pl", "ca", "nl",
    "ar", "sv", "it", "id", "hi", "fi", "vi", "he", "uk", "el", "ms", "cs", "ro",
    "da", "hu", "ta", "no", "th", "ur", "hr", "bg", "lt", "la", "mi", "ml", "cy",
    "sk", "te", "fa", "lv", "bn", "sr", "az", "sl", "kn", "et", "mk", "br", "eu",
    "is", "hy", "ne", "mn", "bs", "kk", "sq", "sw", "gl", "mr", "pa", "si", "km",
    "sn", "yo", "so", "af", "oc", "ka", "be", "tg", "sd", "gu", "am", "yi", "lo",
    "uz", "fo", "ht", "ps", "tk", "nn", "mt", "sa", "lb", "my", "bo", "tl", "mg",
    "as", "tt", "haw", "ln", "ha", "ba", "jw", "su", "yue",
)
_LANGUAGE_SET = frozenset(WHISPER_LANGUAGES)
# ISO 639-1 codes whose Whisper spelling differs.
_LANGUAGE_ALIASES = {"iw": "he", "jv": "jw", "nb": "no"}


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


def normalize_language(value):
    """Return the Whisper code for a client-supplied language, or ``None`` for auto.

    Browsers and API clients send locale tags ("de-DE", "de_AT", " DE "), which
    faster-whisper rejects outright. Reduce them to the primary subtag, and raise
    ``ValueError`` for anything Whisper cannot decode so callers can answer with a
    clear message instead of failing inside a worker thread.
    """
    if value is None:
        return None
    text = str(value).strip().lower().replace("_", "-")
    if text in ("", "auto"):
        return None
    code = text.split("-", 1)[0]
    code = _LANGUAGE_ALIASES.get(code, code)
    if code not in _LANGUAGE_SET:
        raise ValueError(f"Unsupported language {value!r}")
    return code


def _model_basename(model: str) -> str:
    """Last path component of a model name, so 'org/whisper-small.en' -> 'whisper-small.en'."""
    return str(model).replace("\\", "/").rstrip("/").rsplit("/", 1)[-1].lower()


def is_multilingual(model_size: str) -> bool:
    """Return True if *model_size* is a multilingual (not English-only) Whisper model.

    Also understands HF repo ids and local paths: the ``.en`` suffix and the
    distil-* names mark an English-only model wherever they appear in the path.
    """
    if model_size in ENGLISH_ONLY_MODELS:
        return False
    base = _model_basename(model_size)
    return not (base in ENGLISH_ONLY_MODELS or base.endswith(".en"))


def model_can_translate(model: str) -> bool:
    """Whether *model* can serve ``task="translate"`` (speech -> English text).

    The turbo and distil checkpoints were distilled/fine-tuned on transcription
    only and answer a translate request with source-language text; English-only
    models have nothing to translate from. A custom id or path is judged by its
    name because the weights carry no capability flag: a German fine-tune of the
    turbo model ("...turbo-german") inherits the limitation.
    """
    if not is_multilingual(model):
        return False
    base = _model_basename(model)
    return "turbo" not in base and "distil" not in base


def is_custom_model(name: str) -> bool:
    """True for an HF repo id ("org/name") or a filesystem path, not a built-in size.

    faster-whisper resolves the built-in names through its own table and treats
    anything containing a slash as a repo id (or, if it is a directory, a local
    CTranslate2 export). Such a name is valid by construction and must not be
    "corrected" to another model when a load fails.
    """
    return "/" in name or os.sep in name or os.path.isdir(name)
