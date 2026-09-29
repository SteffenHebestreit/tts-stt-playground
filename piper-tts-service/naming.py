"""Dependency-light helpers for the PiperTTS service.

Extracted from app.py so the name-sanitisation, voice-selection, language
handling, voice-directory scanning and output pruning logic can be unit-tested
without importing onnxruntime / librosa / piper.
"""

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Dict, Optional

from fastapi import HTTPException

logger = logging.getLogger(__name__)

# Voice/model names may only contain alphanumerics, dash, and underscore. Used
# with fullmatch(): the old `^...$` + match() accepted a trailing newline
# ("$" matches before it), which then ended up in a path and a response header.
SAFE_NAME_RE = re.compile(r"[a-zA-Z0-9_\-]+")
MAX_VOICE_NAME_LEN = 100

# Piper's quality ladder, as it appears at the end of a voice id.
VOICE_QUALITIES = ("x_low", "low", "medium", "high")
_LANG_TAG_RE = re.compile(r"[a-z]{2,3}(_[A-Z]{2})?")


def sanitize_voice_name(name: str) -> str:
    """Return *name* if it contains only safe characters, otherwise raise 400."""
    if not name or not isinstance(name, str) or not SAFE_NAME_RE.fullmatch(name):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid voice name '{name}': only alphanumeric, dash, and underscore allowed.",
        )
    if len(name) > MAX_VOICE_NAME_LEN:
        # It becomes a directory and two file names; past the filesystem's limit that
        # was an OSError, i.e. a 500 whose message carried the resolved path.
        raise HTTPException(
            status_code=400,
            detail=f"Invalid voice name: at most {MAX_VOICE_NAME_LEN} characters allowed.",
        )
    return name


def _norm_tag(tag: Optional[str]) -> str:
    """Lower-case a language tag and unify the separator ('de-DE' -> 'de_de')."""
    return (tag or "").strip().lower().replace("-", "_")


def base_language(tag: Optional[str]) -> str:
    """Primary language subtag: 'de_DE', 'de-DE' and 'DE' all give 'de'.

    Empty and ``auto`` give '' (no specific language). Browsers and OpenAI
    clients send BCP-47 tags with a hyphen; splitting only on '_' used to turn
    'de-DE' into an unknown language and silently pick an English voice.
    """
    tag = _norm_tag(tag)
    if tag == "auto":
        return ""
    return tag.split("_")[0]


def select_best_voice(
    all_voices: dict,
    language: str,
    quality: Optional[str],
    gender: Optional[str] = None,
    preferred: Optional[str] = None,
    fallback_language: Optional[str] = None,
) -> str:
    """Pick the best voice from *all_voices* matching language/quality/optional gender.

    *all_voices* maps voice id -> object exposing ``.language``, ``.quality`` and
    optionally ``.gender``. An empty/``auto`` *language* considers every voice; an
    empty *quality* means no quality preference. When nothing serves the language
    it degrades to any English voice, then to *fallback_language* (the service's
    default language), then to the hard-coded ``en_US-lessac-medium`` - the
    caller checks that one actually exists.

    *preferred* (the operator's PIPER_DEFAULT_VOICE) goes first among the voices
    serving the language, so it wins unless the quality/gender asked for rules it
    out. Gender ``any``/``auto`` (what the UI sends for "no preference") is no
    preference.
    """
    base = base_language(language)

    def pool(lang_base: str):
        return [(n, v) for n, v in all_voices.items()
                if not lang_base or base_language(v.language) == lang_base]

    matching = pool(base)
    if not matching:
        matching = pool("en")
    if not matching and fallback_language:
        matching = pool(base_language(fallback_language))
    if not matching:
        return "en_US-lessac-medium"

    # 'en_GB' should get the British voice when there is one, not whichever
    # English voice happens to be listed first.
    tag = _norm_tag(language)
    if "_" in tag:
        regional = [(n, v) for n, v in matching if _norm_tag(v.language) == tag]
        if regional:
            matching = regional

    matching.sort(key=lambda item: item[0] != preferred)  # stable: preferred first

    chosen = [(n, v) for n, v in matching if quality and v.quality == quality] or matching

    if gender and gender.strip().lower() not in ("any", "auto"):
        gendered = [(n, v) for n, v in chosen if getattr(v, "gender", None) == gender]
        if gendered:
            chosen = gendered

    return chosen[0][0]


def language_matches(voice_language: str, requested_language: str) -> bool:
    """True if *voice_language* serves *requested_language* (base-tag comparison).

    ``de`` matches ``de_DE`` and ``de-DE``; ``en`` matches ``en_US`` and
    ``en_GB``. An empty or ``auto`` request matches anything, since no specific
    language was asked for.
    """
    requested = base_language(requested_language)
    if not requested:
        return True
    return base_language(voice_language) == requested


# --- language guess for "auto" -------------------------------------------------
#
# Not a language identifier: it only has to tell German from English, the two
# languages the platform is used with, and to say "don't know" otherwise so the
# operator's PIPER_DEFAULT_LANGUAGE decides. Words that are common in BOTH
# languages ("in", "an", "was", "war", "will", "hat", "bin", "die") are left out
# on purpose; each set holds only words that are strong evidence on their own.

_DE_WORDS = frozenset(
    "der die das den dem des ein eine einen einem und ist nicht ich du er sie es wir ihr "
    "mit von zu auf für bei aus nach auch aber oder wenn wie noch nur sich sind wird "
    "sehr habe haben kann dass mich mir uns heute guten hallo bitte danke".split()
)
_EN_WORDS = frozenset(
    "the and is are were be been to of that this with for you it not have has had would "
    "can could i he she we they my your his their what which who from at on but or "
    "if as by hello please thanks".split()
)
_WORD_RE = re.compile(r"[a-zäöüß]+")
_GERMAN_LETTERS = frozenset("äöüß")
# Long inputs are decided by their opening; the scan must stay O(1) per request.
_DETECT_WINDOW = 2000
# A guess needs at least this much evidence and must beat the other language;
# one shared token ("hello" in a German sentence) must not flip the voice.
_DETECT_MIN_SCORE = 2


def detect_language(text: str) -> Optional[str]:
    """Guess ``'de'`` or ``'en'`` from *text*, or ``None`` when the evidence is thin.

    Umlauts and eszett count as evidence for German (as one unit, so a single
    'Zürich' in an English sentence does not outvote the English words around it).
    """
    lowered = (text or "")[:_DETECT_WINDOW].lower()
    words = _WORD_RE.findall(lowered)
    de = sum(1 for w in words if w in _DE_WORDS)
    en = sum(1 for w in words if w in _EN_WORDS)
    if any(ch in _GERMAN_LETTERS for ch in lowered):
        de += _DETECT_MIN_SCORE
    if de >= _DETECT_MIN_SCORE and de > en:
        return "de"
    if en >= _DETECT_MIN_SCORE and en > de:
        return "en"
    return None


# --- voice directory scan ------------------------------------------------------

def parse_voice_id(voice_id: str) -> Dict[str, Optional[str]]:
    """Split a Piper voice id ``<lang>-<speaker>-<quality>`` into its parts.

    Tolerant: unknown shapes give ``None`` for the parts that cannot be read, so
    a hand-named file still registers (the config JSON fills in the rest).
    """
    parts = voice_id.split("-")
    language = parts[0] if len(parts) >= 2 else None
    quality = parts[-1] if len(parts) >= 2 and parts[-1] in VOICE_QUALITIES else None
    middle = parts[1:-1] if quality else parts[1:]
    speaker = "-".join(middle) or None
    return {"language": language, "speaker": speaker, "quality": quality}


def scan_voice_dir(directory) -> Dict[str, dict]:
    """Find installed Piper voices: every ``<id>.onnx.json`` with a sibling ``<id>.onnx``.

    Returns ``{voice_id: {name, language, speaker, quality, sample_rate}}`` sorted
    by id, so voice selection is deterministic whatever order the filesystem
    lists files in. A pair dropped into the directory registers itself on the
    next scan; a half-copied pair (config without model, or the reverse) does not.
    Unreadable or malformed configs are skipped with a warning rather than
    breaking the scan.
    """
    found: Dict[str, dict] = {}
    root = Path(directory)
    try:
        entries = sorted(root.glob("*.onnx.json"))
    except OSError as exc:
        logger.warning("Cannot scan voice directory %s: %s", root, exc)
        return found

    for config_path in entries:
        voice_id = config_path.name[: -len(".onnx.json")]
        if not SAFE_NAME_RE.fullmatch(voice_id):
            logger.warning("Ignoring voice file with an unsafe name: %s", config_path.name)
            continue
        if not (root / f"{voice_id}.onnx").is_file():
            continue
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            logger.warning("Skipping voice %s: unreadable config (%s)", voice_id, exc)
            continue
        if not isinstance(config, dict):
            logger.warning("Skipping voice %s: config is not a JSON object", voice_id)
            continue

        parsed = parse_voice_id(voice_id)
        language_block = config.get("language") if isinstance(config.get("language"), dict) else {}
        audio = config.get("audio") if isinstance(config.get("audio"), dict) else {}
        espeak = config.get("espeak") if isinstance(config.get("espeak"), dict) else {}
        sample_rate = audio.get("sample_rate")
        # The config is authoritative; the espeak voice ("de", "en-us") is the next
        # best evidence, and the file name only counts when it looks like a tag.
        name_language = parsed["language"] if _LANG_TAG_RE.fullmatch(parsed["language"] or "") else None
        espeak_language = str(espeak.get("voice") or "").split("-")[0] or None
        found[voice_id] = {
            "name": voice_id,
            "language": str(language_block.get("code") or espeak_language or name_language or "en"),
            "speaker": str(config.get("dataset") or parsed["speaker"] or voice_id),
            "quality": str(audio.get("quality") or parsed["quality"] or "medium"),
            "sample_rate": sample_rate if isinstance(sample_rate, int) and sample_rate > 0 else 22050,
        }
    return found


def normalize_phoneme_id_map(phoneme_to_id: dict) -> dict:
    """Normalize a phoneme->id map to plain integer ids.

    Piper-format configs map each phoneme to a *list* of ids (``"_": [0]``)
    while training vocabs map to a plain int. Entries that yield no usable
    integer id are dropped.
    """
    normalized = {}
    for phoneme, ids in (phoneme_to_id or {}).items():
        if isinstance(ids, (list, tuple)):
            ids = ids[0] if ids else None
        if isinstance(ids, bool) or not isinstance(ids, int):
            continue
        normalized[phoneme] = ids
    return normalized


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
