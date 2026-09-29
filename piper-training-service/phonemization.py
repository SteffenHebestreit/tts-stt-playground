"""Text to phonemes for dataset metadata, with failures that are never silent.

Two things used to go wrong here, both invisibly.

The language map lived in two files and both fell back to ``en-us`` for any
code they did not know, while ``/prepare-dataset`` never passed a language at
all. German recordings were therefore phonemised as English, and the voice was
then served (by the Piper runtime) with the German rules it was never trained on.

And a phonemiser failure fell back to the raw text. Raw text is not a phoneme
string, but it goes through the same vocabulary builder, so every letter of it
became a "phoneme" id: a few such samples silently widen the vocabulary and
teach the model a spelling-to-audio mapping that nothing at inference produces.

This module imports ``phonemizer`` lazily and depends on nothing else in the
service, so it can be tested without espeak or torch.
"""

import logging
import os
import threading
from typing import Callable, Optional, Sequence

logger = logging.getLogger(__name__)

# ISO code -> espeak-ng voice. A language absent from this table is refused
# rather than guessed at: silently phonemising it as English is what the old
# code did.
PHONEMIZER_LANGUAGES = {
    "de": "de", "en": "en-us", "fr": "fr-fr", "es": "es",
    "it": "it", "nl": "nl", "pt": "pt", "ru": "ru",
}

# Regional variants espeak has a distinct voice for; every other tag is reduced
# to its primary subtag ("de-AT" -> "de").
_REGIONAL_VOICES = {"en-gb": "en-gb", "en-us": "en-us", "pt-br": "pt-br", "fr-be": "fr-be"}

# espeak-ng keeps its voice and settings in process-global state. Dataset builds
# now run in worker threads, so two of them (a training job's metadata step and a
# /prepare-dataset for another model) must not interleave their calls.
_ESPEAK_LOCK = threading.Lock()

# Share of samples that may fail before the whole build is abandoned. A stray
# unpronounceable line is normal; a missing espeak library fails every sample.
DEFAULT_MAX_FAILURE_FRACTION = 0.05


class PhonemizationError(RuntimeError):
    """The phonemiser is unavailable, or too many samples could not be phonemised."""


def normalize_language(value: Optional[str]) -> str:
    """The primary language code for a request value ("de-DE" -> "de").

    Raises ``ValueError`` for a language this service cannot phonemise, with the
    supported list in the message so the caller can put it straight in a 400.
    """
    text = str(value or "").strip().lower().replace("_", "-")
    primary = text.split("-")[0]
    if text in _REGIONAL_VOICES:
        return text
    if primary in PHONEMIZER_LANGUAGES:
        return primary
    raise ValueError(
        f"Unsupported language {value!r}. Supported: {', '.join(sorted(PHONEMIZER_LANGUAGES))}"
    )


def espeak_voice(language: str) -> str:
    """The espeak-ng voice for a (possibly regional) language code."""
    code = normalize_language(language)
    return _REGIONAL_VOICES.get(code) or PHONEMIZER_LANGUAGES[code]


def exported_voice(language: Optional[str]) -> str:
    """The espeak voice written into an exported bundle (``phonemizer_language``).

    The runtime phonemises text with this, so it has to be the voice the dataset
    was phonemised with: ``espeak_voice`` is that single source of truth. The
    exporter used to keep its own closed table with an ``en-us`` fallback, which
    exported a pt-BR voice (trained with ``pt-br``) as English, and fr-BE and
    en-GB likewise.

    A tag this service cannot phonemise can only come from a checkpoint trained
    before unsupported languages were refused, when they were phonemised as
    ``en-us`` (the old table's fallback). Its symbols ARE English phonemes, so
    ``en-us`` is what matches them; the export goes ahead, loudly, rather than
    stranding a finished checkpoint.
    """
    try:
        return espeak_voice(language)
    except ValueError:
        logger.warning(
            "Checkpoint language %r is not supported for phonemisation; exporting it with "
            "phonemizer_language 'en-us', which is what older releases trained it with.",
            language,
        )
        return "en-us"


def stt_language(language: Optional[str]) -> str:
    """The language to ask the STT service for: the primary code, or "auto".

    The STT side transcribes with the language it is given; "auto" lets it
    detect per clip, which on a 1-15 s clip of a single-language dataset
    occasionally picks the wrong language and produces text in it.
    """
    if not language:
        return "auto"
    return normalize_language(language).split("-")[0]


def max_failure_fraction() -> float:
    """PHONEMIZER_MAX_FAILURE_FRACTION, read per call so a test can change it."""
    raw = os.getenv("PHONEMIZER_MAX_FAILURE_FRACTION", "").strip()
    if not raw:
        return DEFAULT_MAX_FAILURE_FRACTION
    try:
        value = float(raw)
    except ValueError:
        logger.warning("PHONEMIZER_MAX_FAILURE_FRACTION=%r is not a number; using %s",
                       raw, DEFAULT_MAX_FAILURE_FRACTION)
        return DEFAULT_MAX_FAILURE_FRACTION
    if not 0.0 <= value <= 1.0:
        logger.warning("PHONEMIZER_MAX_FAILURE_FRACTION=%r is outside 0..1; using %s",
                       raw, DEFAULT_MAX_FAILURE_FRACTION)
        return DEFAULT_MAX_FAILURE_FRACTION
    return value


def _load_espeak() -> Callable[[str, str], str]:
    """A ``(text, voice) -> phonemes`` callable backed by espeak-ng.

    Raises ``PhonemizationError`` when phonemizer is not installed. There is no
    degraded mode: without a phonemiser the honest outcome is no dataset.
    """
    try:
        from phonemizer import phonemize
        from phonemizer.backend import EspeakBackend
    except ImportError as exc:
        raise PhonemizationError(
            f"phonemizer is not installed ({exc}); cannot build phoneme metadata"
        ) from exc

    if not os.getenv("PHONEMIZER_ESPEAK_LIBRARY"):  # phonemizer's own override wins
        import ctypes.util
        # find_library gives a soname when ldconfig knows it; otherwise try the
        # Debian multiarch locations (the ARM64 image is not x86_64).
        candidates = [
            ctypes.util.find_library("espeak-ng"),
            "/usr/lib/x86_64-linux-gnu/libespeak-ng.so.1",
            "/usr/lib/aarch64-linux-gnu/libespeak-ng.so.1",
        ]
        library = next((c for c in candidates if c and (not os.path.isabs(c) or os.path.exists(c))), None)
        if library:
            try:
                EspeakBackend.set_library(library)
            except Exception as exc:  # a wrong path would surface at the first call anyway
                logger.warning("Could not point phonemizer at %s: %s", library, exc)

    def _phonemize(text: str, voice: str) -> str:
        return phonemize(text, language=voice, backend="espeak", strip=True)

    return _phonemize


def phonemize_texts(
    texts: Sequence[str],
    language: str,
    *,
    backend: Optional[Callable[[str, str], str]] = None,
    max_fraction: Optional[float] = None,
) -> list:
    """Phonemes for each text, or ``None`` where that sample failed.

    A failed sample is logged and returned as ``None`` for the caller to drop;
    it is never replaced by its raw text. If more than *max_fraction* of the
    samples fail (or all of them do), the build is abandoned with
    ``PhonemizationError`` naming the first cause.
    """
    voice = espeak_voice(language)
    run = backend or _load_espeak()
    limit = max_failure_fraction() if max_fraction is None else max_fraction

    results: list = []
    failures = 0
    first_error: Optional[str] = None
    with _ESPEAK_LOCK:
        for index, text in enumerate(texts):
            try:
                phonemes = run(text, voice)
                if not isinstance(phonemes, str) or not phonemes.strip():
                    raise ValueError("the phonemiser returned no phonemes")
                results.append(phonemes)
            except Exception as exc:
                failures += 1
                first_error = first_error or f"sample {index}: {exc}"
                if failures <= 5:  # a broken espeak fails every sample; do not log thousands
                    logger.error("Phonemization failed for sample %d (%s): %s", index, voice, exc)
                results.append(None)

    total = len(results)
    if failures and (failures == total or failures / total > limit):
        raise PhonemizationError(
            f"Phonemization failed for {failures} of {total} samples "
            f"(limit {limit:.0%}); first failure: {first_error}. "
            f"Check the espeak-ng installation and that language {language!r} is right."
        )
    if failures:
        logger.warning("Dropping %d of %d samples that could not be phonemised", failures, total)
    return results
