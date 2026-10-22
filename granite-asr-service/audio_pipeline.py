"""Upload handling, decoding and chunking for the Qwen3-ASR service.

Why the service cuts the audio itself instead of handing qwen-asr the file: the
library splits only at 1200 s but caps generation at ``max_new_tokens`` *per
chunk*, so a recording longer than about two minutes came back as HTTP 200 with
the tail missing, and with one fabricated segment for the whole file. Cutting
the audio into short pieces before the model sees it bounds the tokens each
piece needs and gives every piece a real start and end.

Everything here is either a coroutine or meant for ``asyncio.to_thread``: a
decode of a long file on the event loop stalls /health for as long as it runs.
Only numpy is needed at import; soundfile and librosa are optional (the images
have both) so the unit tests do not need them.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import HTTPException, UploadFile

try:  # the images always have both; stripped test environments may not
    import soundfile as sf
except ImportError:  # pragma: no cover
    sf = None
try:
    import librosa
except ImportError:  # pragma: no cover
    librosa = None

logger = logging.getLogger(__name__)

MIB = 1024 * 1024
SAMPLE_RATE = 16000  # Qwen3-ASR's feature extractor works at 16 kHz mono

_COPY_CHUNK = MIB
# multipart boundaries and part headers on top of the file bytes
MULTIPART_SLACK = MIB
# Audio is refused only when it overshoots the limit by more than this, so a file
# of exactly the limit survives resampling rounding.
_OVER_LIMIT_TOLERANCE_S = 0.5
# ffmpeg is told to stop this far past the limit: enough to measure "too long",
# little enough that a file with a wrong header cannot be decoded to the end.
_DECODE_MARGIN_S = 1.0
_FFMPEG_TIMEOUT_S = 600.0

# qwen-asr zero-pads anything shorter than this before it reaches the encoder.
MIN_MODEL_INPUT_S = 0.5
# A last piece shorter than this is not worth a model call of its own (and the
# encoder handles a sub-second clip badly): the previous cut moves earlier instead.
MIN_TAIL_S = 1.0


# --- configuration -------------------------------------------------------------

def env_number(name: str, default, *, cast=float, minimum=None, maximum=None):
    """Parse a numeric env var; junk or out-of-range input falls back to *default*.

    A typo in a tuning knob should cost the knob, not stop the service at import.
    """
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = cast(raw.strip())
    except ValueError:
        logger.warning("Ignoring invalid %s=%r; using %s", name, raw, default)
        return default
    if minimum is not None and value < minimum:
        logger.warning("Ignoring %s=%r (below %s); using %s", name, raw, minimum, default)
        return default
    if maximum is not None and value > maximum:
        logger.warning("%s=%r is above %s; using %s", name, raw, maximum, maximum)
        return maximum
    return value


# --- uploads -------------------------------------------------------------------

def unlink_quiet(*paths: Optional[str]) -> None:
    for path in paths:
        if path:
            with contextlib.suppress(OSError):
                os.unlink(path)


def _safe_suffix(filename: Optional[str]) -> str:
    """A safe temp-file suffix from a client-supplied name (decoders sniff it)."""
    suffix = re.sub(r"[^A-Za-z0-9.]", "", Path(filename).suffix) if filename else ""
    return suffix[:12] or ".wav"


class _TooLarge(Exception):
    pass


async def spool_upload(upload: UploadFile, *, max_bytes: int) -> tuple[str, int]:
    """Copy *upload* to a temp file in chunks, off the event loop; return ``(path, size)``.

    413 (leaving no file behind) past *max_bytes*, 400 for an empty upload. The
    old ``await upload.read()`` held the whole file in memory, and the handler
    then kept those bytes alive next to the file it had just written.
    """
    detail = f"Upload is larger than {max_bytes / MIB:g} MB (MAX_UPLOAD_MB)."
    if upload.size is not None and upload.size > max_bytes:
        raise HTTPException(status_code=413, detail=detail)

    fd, path = tempfile.mkstemp(suffix=_safe_suffix(upload.filename))
    os.close(fd)

    def _copy() -> int:
        total = 0
        with open(path, "wb") as out:
            while True:
                chunk = upload.file.read(_COPY_CHUNK)
                if not chunk:
                    return total
                total += len(chunk)
                if total > max_bytes:
                    raise _TooLarge()
                out.write(chunk)

    try:
        total = await asyncio.to_thread(_copy)
    except _TooLarge:
        unlink_quiet(path)
        raise HTTPException(status_code=413, detail=detail)
    except BaseException:
        unlink_quiet(path)
        raise
    if total == 0:
        unlink_quiet(path)
        raise HTTPException(status_code=400, detail="Audio file is empty.")
    return path, total


# The request-body limit is body_limit.py (a copy in every service directory): it
# checks the declared Content-Length and counts the bytes that arrive, which the
# middleware that used to live here did not, so a chunked upload got past it.


# --- duration ------------------------------------------------------------------

def probe_duration(path: str) -> Optional[float]:
    """Duration in seconds from the header without decoding, or None if unreadable.

    Blocking: run in a thread. libsndfile answers for the common containers in
    microseconds; librosa is the fallback for the ones it cannot open, and it can
    shell out, so it is never something to call on the event loop.
    """
    if sf is not None:
        try:
            info = sf.info(path)
            if info.samplerate:
                return info.frames / float(info.samplerate)
        except Exception:
            pass
    if librosa is not None:
        try:
            return float(librosa.get_duration(path=path))
        except Exception:
            pass
    return None


def _describe(seconds: float) -> str:
    return f"{seconds / 60:.1f} min" if seconds >= 120 else f"{seconds:.0f} s"


def check_duration(duration: Optional[float], max_seconds: float) -> None:
    """413 when the audio is longer than *max_seconds* (0 disables the limit)."""
    if max_seconds > 0 and duration is not None and duration > max_seconds + _OVER_LIMIT_TOLERANCE_S:
        raise HTTPException(
            status_code=413,
            detail=(
                f"Audio is longer than the {_describe(max_seconds)} limit "
                f"(MAX_AUDIO_SECONDS={max_seconds:g}). Split it into shorter parts."
            ),
        )


# --- decoding ------------------------------------------------------------------

class AudioDecodeError(Exception):
    pass


def _decode_ffmpeg(path: str, ffmpeg: str, cap_seconds: Optional[float], timeout: float) -> Optional[np.ndarray]:
    """16 kHz mono float32 samples through ffmpeg; None (with a warning) if that did not work.

    ffmpeg rather than a Python decoder first: it opens every container a browser
    or phone produces (webm, m4a, opus), and streaming 32-bit floats through a
    pipe keeps a long recording at 64 KB per second instead of the several
    copies at the native rate and channel count a decode-then-resample makes.
    """
    cmd = [
        ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-i", path,
        "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE),
    ]
    if cap_seconds:
        cmd += ["-t", f"{cap_seconds:g}"]
    cmd += ["-f", "f32le", "pipe:1"]
    try:
        proc = subprocess.run(
            cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        logger.warning("ffmpeg decode timed out after %.0fs; trying the Python decoder", timeout)
        return None
    except OSError as e:
        logger.warning("ffmpeg could not be started: %s; trying the Python decoder", e)
        return None
    if proc.returncode != 0 or not proc.stdout:
        tail = (proc.stderr or b"").decode("utf-8", "replace").strip()[-300:]
        logger.warning("ffmpeg decode failed (rc=%s): %s; trying the Python decoder", proc.returncode, tail)
        return None
    usable = len(proc.stdout) - len(proc.stdout) % 4
    return np.frombuffer(proc.stdout, dtype="<f4", count=usable // 4)


def _decode_librosa(path: str, cap_seconds: Optional[float]) -> np.ndarray:
    if librosa is None:
        raise AudioDecodeError("no audio decoder is installed")
    try:
        samples, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True, duration=cap_seconds)
    except Exception as e:
        raise AudioDecodeError(str(e)) from e
    return np.asarray(samples, dtype=np.float32)


def decode_audio(
    path: str, *, cap_seconds: Optional[float] = None, ffmpeg: Optional[str] = None,
    timeout: float = _FFMPEG_TIMEOUT_S,
) -> np.ndarray:
    """The file as 16 kHz mono float32, at most *cap_seconds* long. Blocking: run in a thread."""
    samples = _decode_ffmpeg(path, ffmpeg, cap_seconds, timeout) if ffmpeg else None
    if samples is None:
        samples = _decode_librosa(path, cap_seconds)
    if samples.size and not np.isfinite(samples).all():
        samples = np.nan_to_num(samples, nan=0.0, posinf=0.0, neginf=0.0)
    return samples


@dataclass(frozen=True)
class DecodedAudio:
    samples: np.ndarray  # 16 kHz mono float32
    duration: float      # seconds of *samples* (a prefix of the file when decode_seconds was set)
    declared: Optional[float]  # duration from the file header, None if it has none
    size: int            # bytes uploaded


async def load_upload(
    upload: UploadFile,
    *,
    max_bytes: int,
    max_seconds: float,
    ffmpeg: Optional[str],
    decode_seconds: Optional[float] = None,
) -> DecodedAudio:
    """Spool an upload, refuse it if too large or too long, decode it; the temp file is gone on return.

    413 for a file over *max_bytes* or longer than *max_seconds*, 400 for an empty
    one, 422 for one no decoder can open. The length is checked from the header
    first, and again on the decoded samples, whose length is capped at decode
    time so the check holds even when the header is wrong.

    *decode_seconds* decodes only that much from the start (language detection
    needs one chunk, not the whole recording); the length limit is not applied
    then, since nothing beyond the prefix is processed.
    """
    path = None
    try:
        path, size = await spool_upload(upload, max_bytes=max_bytes)
        declared = await asyncio.to_thread(probe_duration, path)
        if decode_seconds is None:
            check_duration(declared, max_seconds)
            cap = max_seconds + _DECODE_MARGIN_S if max_seconds > 0 else None
        else:
            cap = decode_seconds
        try:
            samples = await asyncio.to_thread(decode_audio, path, cap_seconds=cap, ffmpeg=ffmpeg)
        except AudioDecodeError as e:
            logger.info("Could not decode %s: %s", upload.filename, e)
            raise HTTPException(status_code=422, detail="The audio could not be read (unsupported or corrupt file).")
        if samples.size == 0:
            raise HTTPException(status_code=422, detail="The audio file contains no samples.")
        duration = samples.size / float(SAMPLE_RATE)
        if decode_seconds is None:
            check_duration(duration, max_seconds)
        return DecodedAudio(samples=samples, duration=duration, declared=declared, size=size)
    finally:
        unlink_quiet(path)


# --- chunking ------------------------------------------------------------------

@dataclass(frozen=True)
class Chunk:
    index: int
    start: int  # first sample
    end: int    # one past the last sample

    @property
    def start_s(self) -> float:
        return self.start / SAMPLE_RATE

    @property
    def end_s(self) -> float:
        return self.end / SAMPLE_RATE


def plan_chunks(samples: np.ndarray, *, chunk_s: float, search_s: Optional[float] = None) -> list[Chunk]:
    """Cut *samples* into contiguous pieces of at most *chunk_s* seconds, at quiet points.

    Every piece is at most *chunk_s* long, the pieces tile the audio exactly (no
    gap, no overlap, so nothing needs de-duplicating afterwards), and a cut lands
    on the quietest 50 ms in the last *search_s* seconds before the limit so it
    falls between words rather than through one. The last piece is never shorter
    than ``MIN_TAIL_S``: the previous cut moves earlier instead.
    """
    total = int(samples.shape[0])
    if total == 0:
        return []
    max_len = max(1, int(chunk_s * SAMPLE_RATE))
    if total <= max_len:
        return [Chunk(0, 0, total)]

    search = int((search_s if search_s is not None else min(5.0, chunk_s * 0.25)) * SAMPLE_RATE)
    min_tail = min(int(MIN_TAIL_S * SAMPLE_RATE), max_len // 2)
    win = max(4, int(0.05 * SAMPLE_RATE))

    chunks: list[Chunk] = []
    start = 0
    while total - start > max_len:
        # The tail must stay >= min_tail, so the cut never goes past total - min_tail.
        cut = min(start + max_len, total - min_tail)
        # Never search back past half a chunk: a cut near `start` would make a sliver.
        left = max(start + max_len // 2, cut - search)
        if cut - left <= win:
            boundary = cut
        else:
            energy = np.abs(samples[left:cut], dtype=np.float32)
            running = np.concatenate(([0.0], np.cumsum(energy, dtype=np.float64)))
            windowed = running[win:] - running[:-win]
            # Latest of the quietest windows: ties (digital silence) then make the
            # longer piece, not the shorter one.
            quietest = len(windowed) - 1 - int(np.argmin(windowed[::-1]))
            boundary = left + quietest + win // 2
        chunks.append(Chunk(len(chunks), start, boundary))
        start = boundary
    chunks.append(Chunk(len(chunks), start, total))
    return chunks


def model_input(piece: np.ndarray) -> np.ndarray:
    """*piece* as the model should see it: zero-padded to the minimum length qwen-asr accepts."""
    minimum = int(MIN_MODEL_INPUT_S * SAMPLE_RATE)
    if piece.shape[0] >= minimum:
        return piece
    return np.pad(piece, (0, minimum - piece.shape[0]), mode="constant").astype(np.float32)


# --- text ----------------------------------------------------------------------

def _is_unspaced(ch: str) -> bool:
    """Scripts written without spaces between words: no space belongs at a seam next to them."""
    code = ord(ch)
    return (
        0x0E00 <= code <= 0x0E7F      # Thai
        or 0x3000 <= code <= 0x30FF   # CJK punctuation, hiragana, katakana
        or 0x3400 <= code <= 0x4DBF   # CJK extension A
        or 0x4E00 <= code <= 0x9FFF   # CJK unified ideographs
        or 0xFF00 <= code <= 0xFFEF   # fullwidth forms
    )


def join_texts(parts: list[str]) -> str:
    """Concatenate per-chunk transcripts.

    qwen-asr joins its own chunks with ``"".join`` (fine for its 20-minute cuts of
    Chinese, wrong for German: the last word of one piece fuses with the first of
    the next). A space goes between pieces unless either side of the seam is a
    script that does not use them.
    """
    out = ""
    for part in parts:
        part = (part or "").strip()
        if not part:
            continue
        if out and not (_is_unspaced(out[-1]) or _is_unspaced(part[0])):
            out += " "
        out += part
    return out
