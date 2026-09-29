"""Upload handling, audio preparation and inference tuning shared by the NeMo ASR services.

parakeet-asr-service and canary-asr-service each build their image from their
own directory, so this file exists twice; tests/test_nemo_services_shared.py
fails if the copies drift. It imports neither torch nor NeMo (the tuning helpers
take the ``torch`` module as an argument), so the tests exercise it without
either.

The last section holds the NeMo API compatibility helpers. The ASR inference
surface these services use is the same in nemo_toolkit 2.7.x and 3.x, so they
detect what a NeMo build accepts or returns instead of asking which version it is.

Everything that touches a file or a subprocess is either a coroutine or meant to
be called through ``asyncio.to_thread``: a probe of a long mp3 or an ffmpeg
transcode on the event loop stalls /health for as long as it runs.
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib.metadata
import inspect
import logging
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

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
TARGET_SAMPLE_RATE = 16000  # both model families expect 16 kHz mono input

_COPY_CHUNK = MIB
# ffmpeg is told to stop this far past the limit, and audio is refused only when
# it still overshoots the limit by more than the tolerance. Between them a file
# of exactly the limit survives resampling rounding, while a file of any longer
# length is cut off (bounding CPU and disk) yet still measures over the limit.
_TRIM_MARGIN_S = 1.0
_OVER_LIMIT_TOLERANCE_S = 0.5
# multipart boundaries and part headers on top of the file bytes
MULTIPART_SLACK = MIB


# --- configuration -------------------------------------------------------------

def env_number(name: str, default, *, cast=float, minimum=None):
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
    return value


def env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


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


# --- audio preparation ---------------------------------------------------------

@dataclass(frozen=True)
class AudioProbe:
    duration: Optional[float]  # seconds, None when no reader could open the file
    native: bool               # already a 16 kHz mono WAV: NeMo reads it as is


def probe_audio(path: str) -> AudioProbe:
    """Duration and format from the header, without decoding. Blocking: run in a thread.

    libsndfile answers for the common containers in microseconds; librosa is the
    fallback for the ones it cannot open (some mp3/opus builds), which can shell
    out and is therefore never something to call on the event loop.
    """
    info = None
    if sf is not None:
        try:
            info = sf.info(path)
        except Exception:
            info = None
    if info is not None and getattr(info, "samplerate", 0):
        native = info.format == "WAV" and info.samplerate == TARGET_SAMPLE_RATE and info.channels == 1
        return AudioProbe(info.frames / float(info.samplerate), native)
    if librosa is not None:
        try:
            return AudioProbe(float(librosa.get_duration(path=path)), False)
        except Exception:
            pass
    return AudioProbe(None, False)


def _describe(seconds: float) -> str:
    return f"{seconds / 60:.1f} min" if seconds >= 120 else f"{seconds:.0f} s"


def check_duration(duration: Optional[float], max_seconds: float) -> None:
    """413 when the audio is longer than *max_seconds* (0 disables the limit)."""
    if max_seconds > 0 and duration is not None and duration > max_seconds + _OVER_LIMIT_TOLERANCE_S:
        raise HTTPException(
            status_code=413,
            detail=(
                f"Audio is longer than the {_describe(max_seconds)} limit "
                f"(NEMO_MAX_AUDIO_S={max_seconds:g}); NeMo encodes a file in one pass and runs out of "
                f"GPU memory on very long recordings. Split it into shorter parts."
            ),
        )


def converted_path(src_path: str) -> str:
    return f"{src_path}.16k.wav"


async def _ffmpeg_to_16k_mono(
    src_path: str, out_path: str, *, ffmpeg: str, max_seconds: float, timeout: float
) -> bool:
    """Transcode to 16 kHz mono WAV; False (with a warning) when that did not work.

    The child is killed on timeout and on cancellation. It used to run under
    ``subprocess.run`` in a worker thread, which a disconnecting client could not
    interrupt: the transcode ran to completion for nobody.
    """
    cmd = [
        ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-i", src_path,
        "-ar", str(TARGET_SAMPLE_RATE), "-ac", "1",
    ]
    if max_seconds > 0:
        # A file whose header lies about its length (or has none) would otherwise
        # be decoded to the end and written out in full before any check runs.
        cmd += ["-t", f"{max_seconds + _TRIM_MARGIN_S:g}"]
    cmd += ["-f", "wav", out_path]
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
    except OSError as e:
        logger.warning("ffmpeg could not be started: %s; using original file", e)
        return False
    try:
        _, stderr = await asyncio.wait_for(proc.communicate(), timeout)
    except asyncio.TimeoutError:
        await _kill(proc)
        logger.warning("ffmpeg conversion timed out after %.0fs; using original file", timeout)
        return False
    except BaseException:
        await _kill(proc)
        raise
    if proc.returncode == 0 and os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return True
    tail = (stderr or b"").decode("utf-8", "replace").strip()[-300:]
    logger.warning("ffmpeg conversion failed (rc=%s): %s; using original file", proc.returncode, tail)
    return False


async def _kill(proc: asyncio.subprocess.Process) -> None:
    with contextlib.suppress(ProcessLookupError):
        proc.kill()
    with contextlib.suppress(Exception):
        await proc.wait()


@dataclass(frozen=True)
class PreparedAudio:
    path: str        # what NeMo should read
    duration: float  # seconds
    size: int        # bytes uploaded


@contextlib.asynccontextmanager
async def prepared_upload(
    upload: UploadFile,
    *,
    max_bytes: int,
    max_seconds: float,
    ffmpeg: Optional[str],
    convert_timeout: float = 120.0,
):
    """Spool an upload, refuse it if too large or too long, convert it; clean up on exit.

    413 for a file over *max_bytes* or longer than *max_seconds*, 400 for an
    empty one, 422 for one no reader can open. The duration is checked from the
    header before ffmpeg runs, and again on the converted file, whose length is
    capped by ``-t`` so the check holds even when the header is wrong.
    """
    src = None
    try:
        src, size = await spool_upload(upload, max_bytes=max_bytes)
        probe = await asyncio.to_thread(probe_audio, src)
        check_duration(probe.duration, max_seconds)
        path, duration = src, probe.duration
        if ffmpeg and not probe.native:
            out = converted_path(src)
            if await _ffmpeg_to_16k_mono(
                src, out, ffmpeg=ffmpeg, max_seconds=max_seconds, timeout=convert_timeout
            ):
                path = out
                duration = (await asyncio.to_thread(probe_audio, out)).duration
                check_duration(duration, max_seconds)
        if duration is None:
            raise HTTPException(status_code=422, detail="The audio could not be read (unsupported or corrupt file).")
        yield PreparedAudio(path=path, duration=duration, size=size)
    finally:
        if src:
            unlink_quiet(src, converted_path(src))


# --- inference tuning ----------------------------------------------------------

MATMUL_PRECISIONS = ("highest", "high", "medium")


def matmul_precision_from_env(default: str = "high") -> str:
    raw = (os.getenv("NEMO_MATMUL_PRECISION") or "").strip().lower()
    if not raw:
        return default
    if raw not in MATMUL_PRECISIONS:
        logger.warning("Ignoring invalid NEMO_MATMUL_PRECISION=%r (use one of %s); using %s",
                       raw, ", ".join(MATMUL_PRECISIONS), default)
        return default
    return raw


def bf16_unsupported_reason(torch) -> Optional[str]:
    """Why bf16 inference cannot be used here, or None when it can."""
    if getattr(getattr(torch, "version", None), "hip", None):
        return "ROCm is not covered"
    try:
        major, minor = torch.cuda.get_device_capability(0)
    except Exception as e:
        return f"the compute capability could not be read ({e})"
    if major < 8:
        return f"compute capability {major}.{minor} is below 8.0"
    return None


def tune_for_inference(model, torch, *, device: str, matmul_precision: str, bf16: bool):
    """Apply the inference defaults NVIDIA's own transcribe script uses; return ``(model, dtype_name)``.

    TF32 matmuls (``high``) are what the published RTFx figures were measured
    with. bf16 is opt-in because German WER has not been re-measured under it.
    Both are CUDA-only settings, so on CPU this changes nothing.
    """
    if device != "cuda":
        return model, "float32"
    try:
        torch.set_float32_matmul_precision(matmul_precision)
    except Exception as e:
        logger.warning("Could not set float32 matmul precision to %s: %s", matmul_precision, e)
    if not bf16:
        return model, "float32"
    reason = bf16_unsupported_reason(torch)
    if reason:
        logger.warning("NEMO_BF16 is set but bf16 inference is not used: %s; staying in float32", reason)
        return model, "float32"
    logger.info("Running inference in bfloat16 (NEMO_BF16); German WER has not been re-measured with it")
    return model.to(torch.bfloat16), "bfloat16"


def is_cuda_oom(exc: BaseException) -> bool:
    """CUDA/ROCm out-of-memory, whichever exception type this torch raises."""
    return type(exc).__name__ == "OutOfMemoryError" or "out of memory" in str(exc).lower()


# --- NeMo API compatibility ----------------------------------------------------

def public_model_name(name: str) -> str:
    """A model name as an unauthenticated response may show it.

    ``PARAKEET_ASR_MODEL`` / ``CANARY_ASR_MODEL`` also take the path of a ``.nemo``
    file on a mounted volume; the directory layout is not for callers, so such a
    value is shown as its file name. NGC names and Hugging Face ids (``org/name``)
    are returned unchanged.
    """
    if not name:
        return name
    if os.path.isabs(name) or name.startswith(("./", "../", "~")):
        return os.path.basename(os.path.normpath(name)) or name
    return name


def load_nemo_model(nemo_asr, name: str):
    """Load *name*: a path to a ``.nemo`` file, or a Hugging Face repo id / NGC name.

    ``ASRModel.from_pretrained`` treats every string containing "/" as a Hub repo
    id, so a checkpoint kept on a volume (a fine-tune such as parakeet-primeline
    downloaded ahead of time) fails with ``HFValidationError`` on 2.7.x and 3.x
    alike. ``restore_from`` is what ``from_pretrained`` calls once it has the file.
    """
    if os.path.isfile(name):
        return nemo_asr.models.ASRModel.restore_from(restore_path=name)
    return nemo_asr.models.ASRModel.from_pretrained(model_name=name)


_dropped_kwargs_warned: set = set()


def filter_transcribe_kwargs(model, kwargs: dict) -> dict:
    """*kwargs* reduced to what ``model.transcribe`` names, unless it takes ``**kwargs``.

    A NeMo build without one of the optional parameters (``timestamps`` and
    ``verbose`` are newer than the 2.0 line) then answers without it, instead of
    failing every request with a TypeError. The canary ``transcribe`` ends in
    ``**prompt``, which takes everything, so nothing is dropped there.
    """
    try:
        parameters = inspect.signature(model.transcribe).parameters
    except (AttributeError, TypeError, ValueError):
        return dict(kwargs)  # cannot tell; the call itself will say what is wrong
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
        return dict(kwargs)
    dropped = [name for name in kwargs if name not in parameters]
    for name in dropped:
        if name not in _dropped_kwargs_warned:
            _dropped_kwargs_warned.add(name)
            logger.warning("This NeMo build's transcribe() has no '%s' parameter; calling it without", name)
    return {name: value for name, value in kwargs.items() if name in parameters}


def transcribe_results(output) -> list:
    """One entry per input file, whatever container ``transcribe`` returned them in.

    2.7.x and 3.x return a list of ``Hypothesis``. The declared return type of
    both also allows a tuple of lists (what older hybrid RNNT/CTC models return:
    the main decoder's results first) and n-best lists; each is reduced to the best
    hypothesis of each file.
    """
    if output is None:
        return []
    if isinstance(output, tuple):
        output = output[0] if output else []
    if isinstance(output, str) or hasattr(output, "text"):
        return [output]
    results = []
    for item in output:
        nbest = getattr(item, "n_best_hypotheses", None)
        if isinstance(item, (list, tuple)):
            item = item[0] if item else ""
        elif nbest:
            item = nbest[0]
        results.append(item)
    return results


def move_to_cpu(model) -> None:
    """``model.cpu()``, and the same for the auxiliary model NeMo keeps outside the module tree.

    Canary checkpoints that carry a forced-alignment model for timestamps restore
    it with ``object.__setattr__`` (not registered as a submodule) and move it to
    the GPU on the first timestamped request. ``model.cpu()`` alone left it there,
    so /unload reported the memory freed while the aligner still held it.
    """
    failure = None
    for target in (model, getattr(model, "timestamps_asr_model", None)):
        if target is None:
            continue
        try:
            target.cpu()
        except Exception as e:
            failure = failure or e
    if failure is not None:
        raise failure


def runtime_versions(torch) -> dict:
    """The library versions that decide what this image can run, for /status. Never raises.

    ``cuda_arch_list`` is what the installed torch has kernels for: on a Blackwell
    card (RTX 50) ``sm_120`` must be in it, whatever the driver says. It is read
    from the compiled-in flags because ``torch.cuda.get_arch_list()`` answers ``[]``
    while no GPU is visible, which is exactly when this is worth looking at.
    """
    try:
        nemo_version = importlib.metadata.version("nemo-toolkit")
    except Exception:
        nemo_version = None
    info = {
        "nemo_toolkit": nemo_version,
        "torch": getattr(torch, "__version__", None),
        "torch_cuda": getattr(getattr(torch, "version", None), "cuda", None),
    }
    try:
        info["cuda_arch_list"] = torch._C._cuda_getArchFlags().split()
    except Exception:
        try:
            info["cuda_arch_list"] = list(torch.cuda.get_arch_list())
        except Exception:
            pass
    return info
