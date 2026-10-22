"""What the two transformers ASR services share: weights, languages, the speech gate, chunks.

cohere-asr-service and granite-asr-service each build their image from their own
directory, so this file exists twice; tests/test_transformers_asr_shared.py fails if
the copies drift. It needs numpy at import and nothing else heavy: torch and
silero_vad are imported only when the Silero VAD is first used, and
huggingface_hub only when weights are fetched, so the unit tests run without them.
``audio_pipeline.py`` next to it is qwen3-asr-service's upload, decode and
quiet-point chunking code, copied unchanged.

Why each part exists:

* **Weights.** Every file is fetched for a pinned revision with an allow-list (no
  remote code, no demo audio, no second checkpoint), and the weights file can be
  checked against a known SHA-256 before anything reads it (``verify_weights``):
  Cohere's default source is a third-party mirror of a gated repository.
* **Languages.** Neither model detects the language reliably on its own (Cohere
  Transcribe has no language identification at all), so "auto" and an empty value
  mean the service's configured default, and a language the model does not know is
  refused (422) rather than transcribed as another one.
* **The speech gate.** Both models are eager to transcribe: silence and room noise
  come back as invented words (the Cohere model card says so and asks for a VAD in
  front). Only the stretches a voice activity detector calls speech reach the model;
  a file without any comes back empty without a model call.
* **Chunks.** A long recording is cut in its pauses (``plan_speech_chunks``), never
  through a word; a single stretch of speech longer than the model's window is cut
  at its quietest point. Neither model has timestamps, so each chunk becomes one
  segment whose start and end are the chunk's own (``build_segments``).
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
import re
import threading
from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Sequence

import numpy as np

from audio_pipeline import SAMPLE_RATE, Chunk, join_texts, plan_chunks

logger = logging.getLogger(__name__)


# --- configuration ----------------------------------------------------------------

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


def env_flag(name: str, default: bool) -> bool:
    """A boolean env var; anything unrecognised keeps *default* (with a warning)."""
    raw = (os.getenv(name) or "").strip().lower()
    if not raw:
        return default
    if raw in _TRUE:
        return True
    if raw in _FALSE:
        return False
    logger.warning("Ignoring %s=%r (not a boolean); using %s", name, raw, default)
    return default


def env_choice(name: str, default: str, choices: Iterable[str]) -> str:
    """A lower-cased env var from *choices*; anything else keeps *default* (with a warning)."""
    allowed = tuple(choices)
    raw = (os.getenv(name) or "").strip().lower()
    if not raw:
        return default
    if raw in allowed:
        return raw
    logger.warning("Ignoring %s=%r (one of %s); using %s", name, raw, ", ".join(allowed), default)
    return default


def public_model_name(name: str) -> str:
    """A model name as an unauthenticated response may show it.

    A model setting also takes the path of a checkpoint directory on a mounted
    volume; the directory layout is not for callers, so such a value is shown as its
    last component. Hugging Face ids (``org/name``) are returned unchanged.
    """
    if not name:
        return name
    if os.path.isabs(name) or name.startswith(("./", "../", "~")):
        return os.path.basename(os.path.normpath(name)) or name
    return name


def is_local_model(name: str) -> bool:
    """A path on a mounted volume rather than a Hugging Face id."""
    return bool(name) and (os.path.isabs(name) or name.startswith(("./", "../", "~")))


def is_cuda_oom(exc: BaseException) -> bool:
    """torch.cuda.OutOfMemoryError, or a RuntimeError that says the same."""
    return type(exc).__name__ == "OutOfMemoryError" or "out of memory" in str(exc).lower()


# --- languages ---------------------------------------------------------------------


class UnsupportedLanguage(ValueError):
    """A language the model cannot transcribe; ``str()`` is a message for the client."""


def normalize_language_code(raw: Optional[str]) -> str:
    """``"de-DE"`` / ``"de_DE"`` / ``" DE "`` -> ``"de"``; empty stays empty."""
    return (raw or "").strip().lower().replace("_", "-").split("-", 1)[0]


def is_auto(raw: Optional[str]) -> bool:
    value = (raw or "").strip().lower()
    return not value or value == "auto"


def resolve_language(raw: Optional[str], *, default: str, supported: Iterable[str], model: str) -> str:
    """The language a request is transcribed in.

    ``auto`` and an empty value take *default* (the service's ``*_DEFAULT_LANGUAGE``);
    a locale tag reduces to its language. Anything the model does not know raises
    ``UnsupportedLanguage`` naming the ones it does: transcribing it as the default
    instead would hand back confident text in the wrong language with a 200.
    """
    allowed = frozenset(supported)
    if is_auto(raw):
        return default
    code = normalize_language_code(raw)
    if code in allowed:
        return code
    shown = (raw or "").strip()[:32]
    raise UnsupportedLanguage(
        f"Language '{shown}' is not supported by {model}. Supported: {', '.join(sorted(allowed))}. "
        f"Send 'auto' (or no language) to use the default ('{default}')."
    )


def default_language_from_env(raw: Optional[str], *, supported: Iterable[str], fallback: str) -> str:
    """``*_DEFAULT_LANGUAGE``, normalised; a value the model does not know falls back with a warning."""
    allowed = frozenset(supported)
    code = normalize_language_code(raw) if not is_auto(raw) else ""
    if not code:
        return fallback
    if code in allowed:
        return code
    logger.warning("Default language %r is not one the model supports (%s); using %r",
                   raw, ", ".join(sorted(allowed)), fallback)
    return fallback


# --- weights -------------------------------------------------------------------------

# What a model directory needs, and nothing else: configs, tokenizer and chat-template
# files, the (possibly sharded) weights. Leaves out remote-code .py files (the services
# load transformers' own classes), demo audio, model cards' images, signatures and
# any second checkpoint such as granite-speech-5.0-2b's out_llm.safetensors.
DOWNLOAD_PATTERNS = (
    "*.json", "*.jinja", "*.txt", "*.model",
    "model.safetensors", "model-*-of-*.safetensors",
)
WEIGHTS_FILE = "model.safetensors"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SHA256_OFF = {"off", "none", "skip", "false", "0", "no"}


class WeightsMismatch(RuntimeError):
    """The weights file is not the one the configuration says it must be."""


def expected_sha256(setting: Optional[str], default: Optional[str]) -> Optional[str]:
    """The SHA-256 the weights must have: *setting* (``*_SHA256``) when given, else *default*.

    ``off`` (or ``none``/``skip``) disables the check. Anything that is not 64 hex
    digits is a configuration error, raised rather than ignored: a typo must not turn
    a safety check off without a word.
    """
    raw = (setting or "").strip().lower()
    if not raw:
        return default
    if raw in _SHA256_OFF:
        return None
    if _SHA256_RE.match(raw):
        return raw
    raise ValueError(f"not a SHA-256 hex digest (or 'off'): {raw[:80]!r}")


def file_sha256(path: str, *, block: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            data = fh.read(block)
            if not data:
                return digest.hexdigest()
            digest.update(data)


# (real path, size, mtime_ns) of files already hashed in this process, with the result:
# a reload after an idle unload does not read 4 GB again.
_verified: dict[tuple[str, int, int], str] = {}
_verified_lock = threading.Lock()


def verify_weights(model_dir: str, sha256: Optional[str], *, size: Optional[int] = None,
                   filename: str = WEIGHTS_FILE, hasher: Callable[[str], str] = file_sha256) -> Optional[str]:
    """Check ``model_dir/filename`` against *sha256* (and *size*); raise ``WeightsMismatch`` if not.

    Returns the digest, or None when no digest is expected (nothing is read then).
    The size is compared first because it costs nothing and catches a truncated
    download at once. The file is the resolved target of the snapshot's link, so the
    check reads exactly what ``from_pretrained`` will.
    """
    if not sha256:
        return None
    path = os.path.realpath(os.path.join(model_dir, filename))
    try:
        stat = os.stat(path)
    except OSError as exc:
        raise WeightsMismatch(f"{filename} is missing from the model directory") from exc
    if size is not None and stat.st_size != size:
        raise WeightsMismatch(
            f"{filename} is {stat.st_size} bytes, expected {size}: refusing to load it")
    key = (path, stat.st_size, stat.st_mtime_ns)
    with _verified_lock:
        known = _verified.get(key)
    actual = known or hasher(path)
    if actual != sha256:
        raise WeightsMismatch(
            f"{filename} has SHA-256 {actual}, expected {sha256}: refusing to load it")
    with _verified_lock:
        _verified[key] = actual
    return actual


# --- the speech gate ----------------------------------------------------------------------

# A pause longer than this ends a chunk even when the next speech would still fit: the
# silence itself is never shown to the model, which turns long silences into words.
MERGE_GAP_S = 2.0
# Speech the VAD finds this close to the start or the end of the file is extended to
# it. A recording that begins mid-word gives the VAD no run-up, and Silero then starts
# the first stretch late: on an MLS clip it began at 0.71 s and the first word was lost.
EDGE_SNAP_S = 1.0
# Silero's own parameters (silero_vad.get_speech_timestamps). 100 ms of silence
# already separates two stretches, so the chunk planner can cut in the gaps between
# words; 250 ms of padding keeps the onset and the last consonant of each stretch.
SILERO_MIN_SILENCE_MS = 100
SILERO_MIN_SPEECH_MS = 250
SILERO_SPEECH_PAD_MS = 250

# The energy gate (VAD mode "energy", the fallback when Silero cannot load): a frame is
# voiced when it is this much above the file's noise floor (its 10th-percentile frame)
# and above an absolute floor. Stationary noise at any level stays below its own floor
# plus the margin, so a file of pure room noise is gated out as a whole.
ENERGY_FRAME_S = 0.03
ENERGY_MARGIN_DB = 12.0
ENERGY_MIN_DB = -60.0
ENERGY_PAD_S = 0.25
ENERGY_MIN_RUN_S = 0.1

VAD_MODES = ("silero", "energy", "off")


def _frame_db(samples: np.ndarray, frame: int) -> np.ndarray:
    count = samples.shape[0] // frame
    if count == 0:
        return np.zeros(0, dtype=np.float64)
    frames = samples[: count * frame].astype(np.float64).reshape(count, frame)
    rms = np.sqrt(np.mean(frames * frames, axis=1))
    return 20.0 * np.log10(np.maximum(rms, 1e-9))


def energy_regions(samples: np.ndarray, *, margin_db: float = ENERGY_MARGIN_DB,
                   min_db: float = ENERGY_MIN_DB, frame_s: float = ENERGY_FRAME_S,
                   pad_s: float = ENERGY_PAD_S, min_run_s: float = ENERGY_MIN_RUN_S) -> list[tuple[int, int]]:
    """Sample ranges whose short-term level stands out from the file's own noise floor."""
    frame = max(1, int(frame_s * SAMPLE_RATE))
    db = _frame_db(samples, frame)
    if db.size == 0:
        return []
    floor = float(np.percentile(db, 10))
    voiced = db >= max(min_db, floor + margin_db)
    regions: list[tuple[int, int]] = []
    start = None
    for index, flag in enumerate(voiced):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            regions.append((start, index))
            start = None
    if start is not None:
        regions.append((start, len(voiced)))
    min_run = max(1, int(round(min_run_s / frame_s)))
    pad = int(pad_s * SAMPLE_RATE)
    total = samples.shape[0]
    return [
        (max(0, a * frame - pad), min(total, b * frame + pad))
        for a, b in regions if b - a >= min_run
    ]


class SileroVad:
    """Silero VAD (the model ships inside the silero-vad wheel), loaded on first use.

    One model object holds recurrent state, so calls are serialised; it runs on the
    CPU, outside the GPU turn, and costs a few seconds per hour of audio.
    """

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self._model = None
        self._get_speech_timestamps = None
        self._torch = None
        self._lock = threading.Lock()

    def load(self) -> None:
        with self._lock:
            if self._model is None:
                import torch
                from silero_vad import get_speech_timestamps, load_silero_vad
                self._torch = torch
                self._model = load_silero_vad()
                self._get_speech_timestamps = get_speech_timestamps

    def __call__(self, samples: np.ndarray) -> list[tuple[int, int]]:
        self.load()
        with self._lock:
            audio = self._torch.from_numpy(np.ascontiguousarray(samples, dtype=np.float32))
            found = self._get_speech_timestamps(
                audio, self._model, threshold=self.threshold, sampling_rate=SAMPLE_RATE,
                min_speech_duration_ms=SILERO_MIN_SPEECH_MS,
                min_silence_duration_ms=SILERO_MIN_SILENCE_MS,
                speech_pad_ms=SILERO_SPEECH_PAD_MS,
            )
        return [(int(item["start"]), int(item["end"])) for item in found]


class SpeechGate:
    """Which parts of a recording reach the model: ``silero`` (default), ``energy`` or ``off``.

    ``describe()`` says what is actually in force, for /status: a Silero that could
    not be loaded falls back to the energy gate (logged once), which still keeps
    silence and steady noise away from the model.
    """

    def __init__(self, mode: str, *, threshold: float = 0.5, silero: Optional[Callable] = None):
        self.mode = mode if mode in VAD_MODES else "silero"
        self._silero = silero if silero is not None else (SileroVad(threshold) if self.mode == "silero" else None)
        self.fallback_reason: Optional[str] = None

    def describe(self) -> str:
        if self.mode == "silero" and self.fallback_reason:
            return "energy (silero unavailable)"
        return self.mode

    def regions(self, samples: np.ndarray) -> list[tuple[int, int]]:
        """Speech ranges in samples. Blocking: run it in a thread."""
        total = int(samples.shape[0])
        if total == 0:
            return []
        if self.mode == "off":
            return [(0, total)]
        if self.mode == "silero" and not self.fallback_reason:
            try:
                return self._silero(samples)
            except Exception as exc:  # an import or model failure, never the audio's fault
                self.fallback_reason = type(exc).__name__
                logger.error("Silero VAD failed (%s: %s); using the energy gate from now on",
                             type(exc).__name__, exc)
        return energy_regions(samples)


# --- chunks and segments ------------------------------------------------------------------


def _normalise(regions: Iterable[tuple[int, int]], total: int) -> list[tuple[int, int]]:
    """Sorted, clipped to the audio, overlapping and touching ranges merged, empty ones dropped."""
    spans = sorted((max(0, int(a)), min(total, int(b))) for a, b in regions)
    merged: list[list[int]] = []
    for start, end in spans:
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(a, b) for a, b in merged]


def plan_speech_chunks(samples: np.ndarray, regions: Iterable[tuple[int, int]], *, max_chunk_s: float,
                       merge_gap_s: float = MERGE_GAP_S, edge_snap_s: float = EDGE_SNAP_S) -> list[Chunk]:
    """Group the speech *regions* into chunks of at most *max_chunk_s* seconds, cut in pauses.

    Neighbouring regions share a chunk while the pause between them is at most
    *merge_gap_s* and the chunk stays within the limit; the audio between them,
    pause included, is kept as it is. A longer pause, or the limit, starts the next
    chunk, so every cut falls between two stretches of speech. A single stretch
    longer than the limit is cut at its quietest points (``audio_pipeline.plan_chunks``).
    Speech within *edge_snap_s* of either end of the file is extended to it.
    Everything outside the chunks is never transcribed.
    """
    total = int(samples.shape[0])
    max_len = max(1, int(max_chunk_s * SAMPLE_RATE))
    gap = int(merge_gap_s * SAMPLE_RATE)
    snap = int(edge_snap_s * SAMPLE_RATE)
    spans = _normalise(regions, total)
    if spans and spans[0][0] <= snap:
        spans[0] = (0, spans[0][1])
    if spans and total - spans[-1][1] <= snap:
        spans[-1] = (spans[-1][0], total)
    groups: list[list[int]] = []
    for start, end in spans:
        if groups and start - groups[-1][1] <= gap and end - groups[-1][0] <= max_len:
            groups[-1][1] = end
        else:
            groups.append([start, end])
    spans: list[tuple[int, int]] = []
    for start, end in groups:
        if end - start <= max_len:
            spans.append((start, end))
            continue
        for piece in plan_chunks(samples[start:end], chunk_s=max_chunk_s):
            spans.append((start + piece.start, start + piece.end))
    return [Chunk(index, start, end) for index, (start, end) in enumerate(spans)]


def speech_seconds(chunks: Sequence[Chunk]) -> float:
    return sum(chunk.end - chunk.start for chunk in chunks) / SAMPLE_RATE


@dataclass(frozen=True)
class ChunkText:
    text: str
    capped: bool = False   # generation stopped on the token cap: probably cut off


def build_segments(chunks: Sequence[Chunk], outputs: Sequence[ChunkText]) -> tuple[list[dict], str, list[str]]:
    """``(segments, text, capped spans)`` for the chunks and what the model made of them.

    One segment per chunk that produced text, with the chunk's start and end: the
    models give no timestamps of their own, so a segment is as precise as its chunk.
    """
    if len(chunks) != len(outputs):
        raise ValueError(f"{len(outputs)} transcripts for {len(chunks)} chunks")
    segments = []
    capped = []
    for chunk, out in zip(chunks, outputs):
        text = (out.text or "").strip()
        if not text:
            continue
        segment = {"start": round(chunk.start_s, 3), "end": round(chunk.end_s, 3), "text": text}
        if out.capped:
            segment["truncated"] = True
            capped.append(f"{chunk.start_s:.0f}-{chunk.end_s:.0f} s")
        segments.append(segment)
    return segments, join_texts([s["text"] for s in segments]), capped


def batches(items: Sequence, size: int) -> list[Sequence]:
    size = max(1, int(size))
    return [items[i:i + size] for i in range(0, len(items), size)]


def max_new_tokens_for(chunk_s: float, *, per_second: float = 20.0, minimum: int = 256) -> int:
    """A token cap no real speech reaches in *chunk_s* seconds (a very fast speaker makes ~10/s),
    and one that still stops a decoder caught in a repetition loop within seconds."""
    return max(minimum, int(math.ceil(chunk_s * per_second)))
