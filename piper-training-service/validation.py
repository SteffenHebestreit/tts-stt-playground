"""Dependency-light helpers for the Piper training service.

Extracted from app.py / model_exporter.py so the security- and
correctness-critical pure logic can be unit-tested without importing torch
or the rest of the training stack.
"""

import json
import logging
import os
import random
import re
from pathlib import Path
from typing import Optional

from fastapi import HTTPException

logger = logging.getLogger(__name__)


# One alphabet for every name that becomes a path component or a URL segment: the
# piper runtime (`SAFE_NAME_RE`), qwen3 and the gateway accept exactly these
# characters, so a name this service lets through but they refuse is a voice that
# trains fine and then cannot be deployed or listed. The old check only blocked
# path separators and let CR, LF, ESC, spaces and `?#%` through: harmless for the
# filesystem, but they landed in log lines, in `custom/<name>` and in upload URLs.
# `.` is not in the set, which rules out `.` and `..` by construction.
NAME_PATTERN = re.compile(r"[A-Za-z0-9_-]{1,64}")
NAME_RULE = "1-64 characters: letters, digits, '_' and '-'"


def is_safe_name(value) -> bool:
    """Is *value* exactly a valid model name / job id (no stripping, no coercion)?"""
    return isinstance(value, str) and NAME_PATTERN.fullmatch(value) is not None


def safe_name(value: str, field: str = "name") -> str:
    """Validate a model name or job id taken from a request.

    Surrounding whitespace is stripped (a pasted name with a trailing newline
    still works); everything else must match ``[A-Za-z0-9_-]{1,64}``. UUID job
    ids and every name the other services accept pass through unchanged.
    """
    name = value.strip() if isinstance(value, str) else ""
    if not is_safe_name(name):
        # !r escapes control characters, so the detail cannot carry a log-forging
        # payload back out either.
        raise HTTPException(
            status_code=400,
            detail=f"Invalid {field}: {value!r} (expected {NAME_RULE})",
        )
    return name


def stored_name(value) -> Optional[str]:
    """A name read back from ``job_state.json`` or from memory, or ``None``.

    Input is checked once, at the endpoint, but the same names come back from
    disk after a restart, and a volume can be restored from a backup or edited by
    hand. Nothing read from there is trusted to still be a name: a state file
    saying ``"model_name": "../checkpoints"`` must not become
    ``shutil.rmtree("data/../checkpoints")``. No stripping here, since our own
    writer only ever stored validated names.
    """
    return value if is_safe_name(value) else None


def confined_path(root, name: str) -> Path:
    """``root/name``, after proving it is a direct child of *root*; else ``ValueError``.

    The second guard in front of every ``rmtree`` and every path built from a
    stored name. A valid-looking name must still not lead elsewhere: a
    ``data/<name>`` that is a symlink to another directory resolves outside
    *root*, and is refused rather than followed. The path returned is the plain
    ``root/name`` (what the caller would have built anyway), not the resolved one.
    """
    if not is_safe_name(name):
        raise ValueError(f"not a valid name: {name!r}")
    base = Path(root).resolve()
    resolved = (base / name).resolve()
    if resolved.parent != base or not resolved.is_relative_to(base):
        raise ValueError(f"{name!r} resolves to {str(resolved)!r}, outside {str(base)!r}")
    return Path(root) / name


def is_within(path, directory) -> bool:
    """Does *path*, once symlinks and ``..`` are resolved, lie inside *directory*?

    For paths that are read back from a state file rather than built from a name
    (``latest_checkpoint``): what is loaded on resume must be one of the job's own
    checkpoints, not whatever file the state names.
    """
    try:
        return Path(path).resolve().is_relative_to(Path(directory).resolve())
    except (OSError, RuntimeError, ValueError):
        return False


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


VAL_FRACTION = 0.1
SPLIT_SEED = 1234


def split_train_val(entries: list, val_fraction: float = VAL_FRACTION,
                    seed: int = SPLIT_SEED) -> tuple[list, list]:
    """Split dataset entries into (train, val), reproducibly.

    Two things the previous inline version got wrong.

    ``np.random.permutation`` with no seed reshuffled on every call, so
    retraining the same voice compared a new model against a validation set it
    had partly trained on last time — and the loss curves of two runs were not
    comparable at all. Sorting first and seeding makes the split a pure function
    of the segment set, which also matters because ``_run_retrain_from_segments``
    collects its segments concurrently and hands them over in arrival order.

    ``max(1, int(n * 0.1))`` also guaranteed at least one validation entry even
    when there was only one entry in total, which left the training set empty
    and surfaced hours later as a bare "Dataset is empty".
    """
    total = len(entries)
    if total < 2:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Need at least 2 usable segments to build a train/validation "
                f"split, got {total}. Upload more audio, or lower the quality "
                f"filters if segments are being discarded."
            ),
        )

    # Sort by a stable key so concurrent collection order cannot change the split.
    def _key(entry):
        if isinstance(entry, dict):
            return str(entry.get("audio_path") or entry.get("text") or "")
        return str(entry)

    ordered = sorted(entries, key=_key)

    n_val = min(max(1, int(total * val_fraction)), total - 1)
    rng = random.Random(seed)
    indices = list(range(total))
    rng.shuffle(indices)

    val_idx = set(indices[:n_val])
    val = [ordered[i] for i in indices[:n_val]]
    train = [ordered[i] for i in range(total) if i not in val_idx]
    return train, val


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


# Every id the model can be given for a symbol it has no row for, plus padding.
SPECIAL_TOKENS = ["<pad>", "<unk>", "<start>", "<end>", " "]


def phoneme_id_map_from_entries(entries) -> dict:
    """Build a phoneme->id map from dataset metadata entries.

    ``sorted(phonemes ∪ special_tokens)``, contiguous from 0. This is how a NEW
    training run derives its vocabulary; once a run exists its vocabulary is the
    persisted ``vocab.json`` / the map inside the checkpoint, never a rebuild,
    because the ids are the row numbers of the trained embedding.
    """
    phonemes = set()
    for item in entries or []:
        if not isinstance(item, dict):
            continue
        phoneme_text = item.get("phonemes", item.get("text", ""))
        if phoneme_text:
            phonemes.update(list(phoneme_text))

    all_phonemes = sorted(phonemes.union(SPECIAL_TOKENS))
    return {p: i for i, p in enumerate(all_phonemes)}


# --- vocabulary persistence ---------------------------------------------------
#
# The vocabulary used to be rebuilt from whatever train.json held at the moment
# of use: by the dataset at the start of a run, and again by the exporter at the
# end. Any change of the dataset in between (a re-segmentation, another
# /prepare-dataset for the same voice, a resume after new data) silently
# re-numbered the symbols, so the exported model looked up rows of the embedding
# that belonged to different symbols. The map is therefore written once, next to
# the checkpoints, embedded in every checkpoint, and read back from there.

VOCAB_FILENAME = "vocab.json"
VOCAB_FORMAT = 1


def validate_phoneme_id_map(mapping, n_vocab: Optional[int] = None) -> dict:
    """Return *mapping* as ``{str: int}`` or raise ``ValueError`` saying what is wrong.

    Checks what an embedding lookup needs: non-empty string symbols, integer ids
    that are non-negative, unique and (when *n_vocab* is given) below the
    embedding size, and the ``<pad>`` / ``<unk>`` entries the dataset and the
    runtime fall back to.
    """
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError("phoneme vocabulary is empty or not a mapping")

    clean = {}
    seen = {}
    for symbol, idx in mapping.items():
        if not isinstance(symbol, str) or not symbol:
            raise ValueError(f"phoneme vocabulary has an invalid symbol: {symbol!r}")
        # bool is an int subclass; True would silently become id 1.
        if isinstance(idx, bool) or not isinstance(idx, int) or idx < 0:
            raise ValueError(f"phoneme vocabulary maps {symbol!r} to an invalid id: {idx!r}")
        if idx in seen:
            raise ValueError(
                f"phoneme vocabulary maps both {seen[idx]!r} and {symbol!r} to id {idx}"
            )
        seen[idx] = symbol
        clean[symbol] = idx

    for token in ("<pad>", "<unk>"):
        if token not in clean:
            raise ValueError(f"phoneme vocabulary has no {token} entry")

    if n_vocab is not None and max(clean.values()) >= n_vocab:
        raise ValueError(
            f"phoneme vocabulary uses id {max(clean.values())} but the model's "
            f"embedding has only {n_vocab} rows"
        )
    return clean


def save_vocab(path: Path, mapping: dict, *, n_vocab: Optional[int] = None,
               boundary_tokens: bool = True) -> None:
    """Write the vocabulary atomically (a half-written file must never be read back)."""
    clean = validate_phoneme_id_map(mapping, n_vocab)
    payload = {
        "format": VOCAB_FORMAT,
        "n_vocab": n_vocab,
        "boundary_tokens": bool(boundary_tokens),
        "phoneme_id_map": clean,
    }
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def load_vocab(path: Path, n_vocab: Optional[int] = None) -> dict:
    """Read and validate a vocabulary written by :func:`save_vocab`."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read vocabulary {path}: {exc}") from exc
    if not isinstance(payload, dict) or "phoneme_id_map" not in payload:
        raise ValueError(f"{path} is not a vocabulary file (no phoneme_id_map)")
    return validate_phoneme_id_map(payload["phoneme_id_map"], n_vocab)


def vocab_for_checkpoint(checkpoint: dict, checkpoint_dir: Path,
                         n_vocab: Optional[int] = None) -> tuple[Optional[dict], str]:
    """The vocabulary a checkpoint was trained with, and where it came from.

    Order: the map embedded in the checkpoint, then ``vocab.json`` beside it.
    ``(None, "none")`` for a checkpoint written before vocabularies were
    persisted; the caller decides how to degrade for those. A vocabulary that IS
    present but invalid raises, since guessing around it is what this exists to
    stop.
    """
    embedded = checkpoint.get("phoneme_to_id") if isinstance(checkpoint, dict) else None
    if embedded:
        return validate_phoneme_id_map(embedded, n_vocab), "checkpoint"
    vocab_path = Path(checkpoint_dir) / VOCAB_FILENAME
    if vocab_path.exists():
        return load_vocab(vocab_path, n_vocab), VOCAB_FILENAME
    return None, "none"


# --- environment knobs and checkpoint housekeeping ----------------------------

def env_int(name: str, default: int, minimum: int = 0) -> int:
    """Read an integer environment variable; fall back to *default* on a bad value.

    A typo in a knob (``KEEP_LAST_CHECKPOINTS=five``) must not stop the service
    from starting, but it must not pass unnoticed either, hence the warning.
    Values below *minimum* are clamped up to it.
    """
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw.strip())
    except ValueError:
        logger.warning("Ignoring %s=%r: not an integer, using %d", name, raw, default)
        return default
    if value < minimum:
        logger.warning("%s=%d is below the minimum %d, using %d", name, value, minimum, minimum)
        return minimum
    return value


_PERIODIC_CHECKPOINT = re.compile(r"^checkpoint_epoch_(\d+)\.pt$")


def prune_old_checkpoints(checkpoint_dir: Path, keep_last: int,
                          protect: tuple = ()) -> list[Path]:
    """Delete all but the newest *keep_last* ``checkpoint_epoch_N.pt`` files.

    "Newest" is by epoch number, not by name or mtime: ``checkpoint_epoch_100``
    sorts before ``checkpoint_epoch_25`` as text. Only files matching the
    periodic pattern are considered, so ``final_model.pt``, ``best_model.pt``,
    ``vocab.json`` and ``job_state.json`` are never touched, and anything in
    *protect* survives regardless (the checkpoint a resume started from is still
    referenced until the first new one is written). ``keep_last <= 0`` keeps
    everything. Returns the deleted paths.

    A 10000-epoch job at the default interval writes 2000 files of a few hundred
    MB each, which used to fill the volume.
    """
    if keep_last <= 0:
        return []
    directory = Path(checkpoint_dir)
    if not directory.is_dir():
        return []

    numbered = []
    for entry in directory.iterdir():
        match = _PERIODIC_CHECKPOINT.match(entry.name)
        if match and entry.is_file():
            numbered.append((int(match.group(1)), entry))
    numbered.sort()

    protected = {Path(p).resolve() for p in protect}
    removed = []
    for _, entry in numbered[:-keep_last]:
        if entry.resolve() in protected:
            continue
        try:
            entry.unlink()
            removed.append(entry)
        except OSError as exc:
            logger.warning("Could not prune checkpoint %s: %s", entry, exc)
    return removed
