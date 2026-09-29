"""PiperTTS service for text-to-speech synthesis.

Supports 40+ pre-trained voices across multiple languages and custom
ONNX models trained via the Piper Training service. Default voices use
the Piper binary; custom VITS models use direct ONNX Runtime inference.
"""

import os
import uuid
import tempfile
import json
import asyncio
import contextlib
import logging
import threading
from collections import OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path
import shutil
from typing import Dict, Optional

logger = logging.getLogger(__name__)

import numpy as np
import soundfile as sf

from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.background import BackgroundTask
from pydantic import BaseModel, Field, ValidationError
import uvicorn
import librosa
import io

from naming import (
    sanitize_voice_name,
    base_language,
    language_matches,
    detect_language,
    scan_voice_dir,
    select_best_voice as _select_best_voice,
    prune_old_outputs,
    normalize_phoneme_id_map,
)


# The variables are read with a literal os.getenv("NAME") at each call site (the
# helpers only parse the value): tests/test_env_wiring.py finds what a service
# reads, and what compose must forward to it, by looking for exactly that.
def _number(raw, name: str, default, cast=float, minimum=None):
    """Parse a numeric env value; a typo falls back to the default instead of failing the import."""
    raw = (raw or "").strip()
    if not raw:
        return default
    try:
        value = cast(raw)
    except ValueError:
        logger.warning("Ignoring non-numeric %s=%r; using %s", name, raw, default)
        return default
    if minimum is not None and value < minimum:
        logger.warning("Ignoring %s=%r (below %s); using %s", name, raw, minimum, default)
        return default
    return value


def _flag(raw, default: bool = False) -> bool:
    raw = (raw or "").strip().lower()
    return default if not raw else raw in {"1", "true", "yes", "on"}


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    """Register the installed voices on startup and sweep old outputs on a timer."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    await asyncio.to_thread(_refresh_default_voices)
    await load_custom_voices()
    _log_voice_summary()
    pruner = asyncio.create_task(_prune_loop())
    try:
        yield
    finally:
        pruner.cancel()


async def _prune_loop():
    """Sweep expired outputs on a timer instead of on the request path.

    The sweep is O(files written in the retention window) and used to run
    synchronously inside /tts — so it got slower the more the service was used,
    and it ran on the event loop.
    """
    while True:
        try:
            await asyncio.to_thread(_prune_old_outputs)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning(f"Output prune failed: {e}")
        await asyncio.sleep(PRUNE_INTERVAL_S)


app = FastAPI(
    title="PiperTTS Service",
    description="Text-to-Speech using Piper with custom and default models",
    lifespan=_lifespan,
)

allowed_origins_str = os.getenv("ALLOWED_ORIGINS", "*")
allowed_origins = [origin.strip() for origin in allowed_origins_str.split(",")] if allowed_origins_str else ["*"]
allow_credentials = os.getenv("ALLOW_CREDENTIALS", "false").strip().lower() in {"1", "true", "yes", "on"}
if "*" in allowed_origins and allow_credentials:
    allow_credentials = False

# Where voice models live. compose has always set PIPER_DATA_DIR next to
# PIPER_OUTPUT_DIR, but only the output half was ever read — every model path
# below was the literal "/app/models", so pointing the data dir somewhere else
# silently changed nothing and the service kept reading the old location.
MODELS_DIR = Path(os.getenv("PIPER_DATA_DIR", "/app/models"))
DEFAULT_MODELS_DIR = MODELS_DIR / "default"
CUSTOM_MODELS_DIR = MODELS_DIR / "custom"

# Generated default-voice WAVs land in OUTPUT_DIR and would otherwise accumulate
# indefinitely. Files older than the retention window are pruned best-effort.
OUTPUT_DIR = os.getenv("PIPER_OUTPUT_DIR", "/app/output")
OUTPUT_RETENTION_HOURS = float(os.getenv("OUTPUT_RETENTION_HOURS", "24"))
PRUNE_INTERVAL_S = float(os.getenv("OUTPUT_PRUNE_INTERVAL_S", "3600"))
# When a requested language has no voice, select_best_voice() substitutes an
# English one. Default keeps that (non-breaking) but reports it; set true to
# return 400 instead, which is usually what an API consumer wants.
STRICT_LANGUAGE = _flag(os.getenv("PIPER_STRICT_LANGUAGE"))

# The UI and the gateway omit `language` for "Auto-Detect", and the OpenAI-style
# route has no such field at all. Without a service-side default that meant
# en_US: German text read by an English voice, with HTTP 200.
DEFAULT_LANGUAGE = os.getenv("PIPER_DEFAULT_LANGUAGE", "").strip() or "de"
if not base_language(DEFAULT_LANGUAGE):  # "auto" is not a language to default to
    DEFAULT_LANGUAGE = "de"
# Optional explicit voice id, used when the request names neither a voice nor a
# quality/gender. Deploy profiles have set it for a while; nothing read it.
DEFAULT_VOICE = os.getenv("PIPER_DEFAULT_VOICE", "").strip() or None
# For an omitted/"auto" language, look at the text (German vs English only) and
# fall back to DEFAULT_LANGUAGE when the evidence is thin. false = always default.
AUTO_DETECT = _flag(os.getenv("PIPER_AUTO_DETECT"), True)

# One `piper` process per request, each loading a model and running its own
# ONNX Runtime threads, with nothing bounding how many run at once or for how
# long. Half the cores by default leaves the rest to the event loop and the gateway.
SYNTH_TIMEOUT_S = _number(os.getenv("PIPER_TIMEOUT_S"), "PIPER_TIMEOUT_S", 60.0, float, minimum=1.0)
MAX_CONCURRENCY = _number(
    os.getenv("PIPER_MAX_CONCURRENCY"), "PIPER_MAX_CONCURRENCY", max(1, (os.cpu_count() or 2) // 2), int, minimum=1)

# Input bounds. The gateway caps text at 20000 characters (MAX_TTS_CHARS); the
# service applies the same bound so a direct caller cannot bypass it.
MAX_TEXT_CHARS = _number(os.getenv("MAX_TEXT_CHARS"), "MAX_TEXT_CHARS", 20000, int, minimum=1)
_MIB = 1024 * 1024
MAX_UPLOAD_BYTES = int(_number(os.getenv("MAX_UPLOAD_MB"), "MAX_UPLOAD_MB", 100.0, float, minimum=0.001) * _MIB)
MAX_ANALYZE_BYTES = int(
    _number(os.getenv("MAX_ANALYZE_UPLOAD_MB"), "MAX_ANALYZE_UPLOAD_MB", 25.0, float, minimum=0.001) * _MIB)
# Piper configs are a few tens of KB (the phoneme map); anything near this is not one.
MAX_CONFIG_BYTES = 5 * _MIB
_UPLOAD_CHUNK = 1024 * 1024
# multipart boundaries and part headers on top of the file bytes
_MULTIPART_SLACK = _MIB


def _body_limit(path: str) -> int:
    """Largest request body (bytes) accepted at *path*, by declared Content-Length."""
    if path == "/upload_model":
        return MAX_UPLOAD_BYTES + MAX_CONFIG_BYTES + _MULTIPART_SLACK
    if path == "/analyze_audio":
        return MAX_ANALYZE_BYTES + _MULTIPART_SLACK
    # JSON: 4 bytes per character is the UTF-8 worst case, plus the other fields
    return MAX_TEXT_CHARS * 4 + 64 * 1024


class _BodyLimitMiddleware:
    """Answer 413 from the Content-Length header, before anything is buffered.

    FastAPI parses the whole body (multipart parts are spooled to disk, JSON is
    held in memory) before a handler runs, so a size check inside the handler
    comes after the cost was paid. Clients that send no Content-Length (chunked)
    are still bounded per file by the handlers.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and scope["method"] in ("POST", "PUT"):
            declared = dict(scope["headers"]).get(b"content-length", b"")
            limit = _body_limit(scope["path"])
            if declared.isdigit() and int(declared) > limit:
                response = JSONResponse(
                    status_code=413,
                    content={"detail": f"Request body is larger than {limit // _MIB} MB."},
                )
                await response(scope, receive, send)
                return
        await self.app(scope, receive, send)


# Added before CORS so CORS is the outer layer and a 413 still carries its headers.
app.add_middleware(_BodyLimitMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _prune_old_outputs() -> None:
    """Best-effort removal of generated WAVs older than the retention window."""
    prune_old_outputs(OUTPUT_DIR, OUTPUT_RETENTION_HOURS)


def _sanitize_voice_name(name: str) -> str:
    """Return *name* if it contains only safe characters, otherwise raise."""
    return sanitize_voice_name(name)


def _unlink_quiet(path) -> None:
    with contextlib.suppress(OSError):
        os.unlink(path)


class TTSRequest(BaseModel):
    """Request body for standard Piper text-to-speech synthesis."""

    text: str = Field(max_length=MAX_TEXT_CHARS)
    voice: Optional[str] = Field(None, max_length=128)
    # None/"auto": the service decides (guessed from the text, else PIPER_DEFAULT_LANGUAGE).
    language: Optional[str] = Field(None, max_length=32)
    quality: str = "medium"  # x_low, low, medium, high
    gender: Optional[str] = None  # male, female
    # gt=0 guards the `1.0 / speed` length-scale conversion, which turned
    # speed=0 into a ZeroDivisionError -> 500, and passed negatives to the binary.
    speed: float = Field(1.0, gt=0.0, le=4.0)
    output_format: str = "wav"

class VoiceInfo(BaseModel):
    """Metadata describing a built-in or custom voice model."""

    name: str
    language: str
    speaker: str
    quality: str
    sample_rate: int
    gender: Optional[str] = None
    model_type: str = "default"  # "default" or "custom"

class VoiceCloneRequest(BaseModel):
    """Request body for synthesis with a named custom voice."""

    text: str = Field(max_length=MAX_TEXT_CHARS)
    voice_name: str
    reference_audio: Optional[str] = None
    # gt=0 guards the `1.0 / speed` length-scale conversion, which turned
    # speed=0 into a ZeroDivisionError -> 500, and passed negatives to the binary.
    speed: float = Field(1.0, gt=0.0, le=4.0)

# Available default voices by language
DEFAULT_VOICES = {
    # English voices
    "en_US-lessac-medium": VoiceInfo(
        name="en_US-lessac-medium",
        language="en_US",
        speaker="lessac",
        quality="medium",
        sample_rate=22050,
        model_type="default"
    ),
    "en_US-amy-medium": VoiceInfo(
        name="en_US-amy-medium",
        language="en_US",
        speaker="amy",
        quality="medium",
        sample_rate=22050,
        model_type="default"
    ),
    "en_GB-alan-medium": VoiceInfo(
        name="en_GB-alan-medium",
        language="en_GB",
        speaker="alan",
        quality="medium",
        sample_rate=22050,
        model_type="default"
    ),
    # German voices
    "de_DE-thorsten-medium": VoiceInfo(
        name="de_DE-thorsten-medium",
        language="de_DE",
        speaker="thorsten",
        quality="medium",
        sample_rate=22050,
        model_type="default"
    ),
    "de_DE-eva_k-x_low": VoiceInfo(
        name="de_DE-eva_k-x_low",
        language="de_DE",
        speaker="eva_k",
        quality="x_low",
        sample_rate=22050,
        model_type="default"
    ),
    # French voices
    "fr_FR-siwis-medium": VoiceInfo(
        name="fr_FR-siwis-medium",
        language="fr_FR",
        speaker="siwis",
        quality="medium",
        sample_rate=22050,
        model_type="default"
    ),
    # Spanish voices
    "es_ES-mls_9972-low": VoiceInfo(
        name="es_ES-mls_9972-low",
        language="es_ES",
        speaker="mls_9972",
        quality="low",
        sample_rate=22050,
        model_type="default"
    ),
    # Italian voices
    "it_IT-riccardo-x_low": VoiceInfo(
        name="it_IT-riccardo-x_low",
        language="it_IT",
        speaker="riccardo",
        quality="x_low",
        sample_rate=22050,
        model_type="default"
    ),
    # Dutch voices
    "nl_NL-mls_5809-low": VoiceInfo(
        name="nl_NL-mls_5809-low",
        language="nl_NL",
        speaker="mls_5809",
        quality="low",
        sample_rate=22050,
        model_type="default"
    )
}

# DEFAULT_VOICES is metadata, not the source of truth: the voices that are really
# installed are whatever pairs of <id>.onnx / <id>.onnx.json sit in
# DEFAULT_MODELS_DIR. The Dockerfile used to be the only place that decided that
# (and a bind-mounted models volume hides what the image downloaded), so a voice
# dropped into the volume was invisible and a listed one could have no file.
# Only when the directory holds no voice at all is this catalog advertised, so
# /tts can say which model file is missing instead of "no voice".
_DEFAULT_REGISTRY: Dict[str, VoiceInfo] = {}
_CATALOG_ONLY = False
_SCAN_STAMP: object = "unscanned"
_SCAN_LOCK = threading.Lock()


def _dir_stamp(path: Path):
    """Cheap change marker for a directory: its mtime (moves when a file is added or removed)."""
    try:
        return os.stat(path).st_mtime_ns
    except OSError:
        return None


def _refresh_default_voices() -> int:
    """Re-scan DEFAULT_MODELS_DIR and replace the default-voice registry; returns its size."""
    global _DEFAULT_REGISTRY, _CATALOG_ONLY, _SCAN_STAMP
    with _SCAN_LOCK:
        stamp = _dir_stamp(DEFAULT_MODELS_DIR)
        found = scan_voice_dir(DEFAULT_MODELS_DIR)
        if found:
            registry = {vid: VoiceInfo(model_type="default", **meta) for vid, meta in found.items()}
            _CATALOG_ONLY = False
        else:
            registry = dict(DEFAULT_VOICES)
            _CATALOG_ONLY = True
            logger.warning(
                "No voice models (<id>.onnx + <id>.onnx.json) found in %s; listing the built-in "
                "catalog, but synthesis with it fails until the model files are installed.",
                DEFAULT_MODELS_DIR,
            )
        _DEFAULT_REGISTRY = registry
        _SCAN_STAMP = stamp
        return len(registry)


def _default_voices() -> Dict[str, VoiceInfo]:
    """The default-voice registry, re-scanned first if the directory changed since the last scan."""
    if _dir_stamp(DEFAULT_MODELS_DIR) != _SCAN_STAMP:
        _refresh_default_voices()
    return _DEFAULT_REGISTRY


def _all_voices() -> Dict[str, VoiceInfo]:
    """Default and custom voices; a custom voice shadows a default one with the same id."""
    return {**_default_voices(), **CUSTOM_VOICES}


def _log_voice_summary() -> None:
    voices = _all_voices()
    logger.info(
        "Voices: %d default (%s), %d custom; default language %s, default voice %s",
        len(_DEFAULT_REGISTRY), "catalog only" if _CATALOG_ONLY else "installed",
        len(CUSTOM_VOICES), DEFAULT_LANGUAGE, DEFAULT_VOICE or "-",
    )
    for warning in _configuration_warnings(voices):
        logger.warning(warning)


def _configuration_warnings(voices: Dict[str, VoiceInfo]) -> list:
    """Settings that point at voices which are not installed (shown by /ready and logged)."""
    warnings = []
    if DEFAULT_VOICE and DEFAULT_VOICE not in voices:
        warnings.append(f"PIPER_DEFAULT_VOICE={DEFAULT_VOICE} is not an installed voice; it is ignored.")
    if not any(language_matches(v.language, DEFAULT_LANGUAGE) for v in voices.values()):
        warnings.append(
            f"No installed voice serves PIPER_DEFAULT_LANGUAGE={DEFAULT_LANGUAGE}; "
            "requests without a language fall back to another language."
        )
    return warnings


# Custom voices loaded at startup from CUSTOM_MODELS_DIR
CUSTOM_VOICES: Dict[str, VoiceInfo] = {}

# Cache of loaded ONNX inference sessions keyed by model path. Each entry stores
# the file mtime so a re-trained/re-uploaded model is reloaded automatically.
# Bounded and lock-guarded: sessions are multi-hundred-MB and requests build them
# from a thread pool, so an unbounded unsynchronised dict both grew without limit
# and let concurrent first-requests each construct their own session.
_ONNX_SESSION_CACHE: "OrderedDict[str, tuple]" = OrderedDict()
_ONNX_CACHE_SIZE = max(1, int(os.getenv("ONNX_SESSION_CACHE_SIZE", "4")))
_ONNX_CACHE_LOCK = threading.Lock()
# Fewer threads than cores on purpose: these are short sequences where ORT's
# thread ramp-up costs more than the parallelism returns.
_ONNX_THREADS = max(1, int(os.getenv("ONNX_NUM_THREADS", "2")))


def _get_onnx_session(model_path: str):
    """Return a cached ONNX Runtime session for *model_path*, loading it if needed."""
    import onnxruntime as ort

    mtime = os.path.getmtime(model_path)
    with _ONNX_CACHE_LOCK:
        cached = _ONNX_SESSION_CACHE.get(model_path)
        if cached and cached[0] == mtime:
            _ONNX_SESSION_CACHE.move_to_end(model_path)
            return cached[1]

        sess_options = ort.SessionOptions()
        sess_options.inter_op_num_threads = _ONNX_THREADS
        sess_options.intra_op_num_threads = _ONNX_THREADS
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        session = ort.InferenceSession(model_path, sess_options=sess_options)

        _ONNX_SESSION_CACHE[model_path] = (mtime, session)
        _ONNX_SESSION_CACHE.move_to_end(model_path)
        while len(_ONNX_SESSION_CACHE) > _ONNX_CACHE_SIZE:
            _ONNX_SESSION_CACHE.popitem(last=False)
        return session


@app.get("/")
async def root():
    """Return service identity and readiness status."""
    return {"service": "PiperTTS Service", "status": "ready", "version": "1.0.0"}


@app.get("/health")
async def health():
    """Health-check endpoint used by Docker and monitoring."""
    return {"status": "healthy"}


@app.get("/ready")
async def ready():
    """Readiness: can this instance synthesise anything right now?

    /health only proves the process is up. This answers 503 while no voice model
    is installed (an empty bind-mounted models volume is the usual cause), the
    piper binary is missing, or the output directory is not writable, and lists
    misconfigured defaults as warnings. Cheap: one directory stat, no model load.
    """
    default_voices = _default_voices()
    problems = []
    installed_default = 0 if _CATALOG_ONLY else len(default_voices)
    if installed_default == 0 and not CUSTOM_VOICES:
        problems.append(f"no voice models installed in {DEFAULT_MODELS_DIR} or {CUSTOM_MODELS_DIR}")
    if installed_default and shutil.which("piper") is None:
        problems.append("the piper binary is not on PATH")
    if not (os.path.isdir(OUTPUT_DIR) and os.access(OUTPUT_DIR, os.W_OK)):
        problems.append(f"output directory {OUTPUT_DIR} is not writable")
    body = {
        "status": "not_ready" if problems else "ready",
        "default_voices": installed_default,
        "custom_voices": len(CUSTOM_VOICES),
        "default_language": DEFAULT_LANGUAGE,
        "default_voice": DEFAULT_VOICE,
        "problems": problems,
        "warnings": _configuration_warnings(_all_voices()),
    }
    return JSONResponse(status_code=503 if problems else 200, content=body)


@app.get("/voices")
async def list_voices():
    """List all available voices grouped by language."""
    default_voices = _default_voices()
    all_voices = _all_voices()

    # Group by language
    voices_by_language = {}
    for voice_name, voice_info in all_voices.items():
        lang = voice_info.language
        if lang not in voices_by_language:
            voices_by_language[lang] = []
        voices_by_language[lang].append(voice_info.model_dump())

    return {
        "voices": all_voices,
        "voices_by_language": voices_by_language,
        "total": len(all_voices),
        "default_count": len(default_voices),
        "custom_count": len(CUSTOM_VOICES),
        "supported_languages": list(voices_by_language.keys()),
        "default_language": DEFAULT_LANGUAGE,
        "default_voice": DEFAULT_VOICE if DEFAULT_VOICE in all_voices else None,
        # True while no model file was found and the built-in catalog is shown instead
        "catalog_only": _CATALOG_ONLY,
    }

def select_best_voice(language: str, quality: Optional[str], gender: Optional[str] = None,
                      preferred: Optional[str] = None) -> str:
    """Pick the best voice across all registered voices (see naming.select_best_voice)."""
    return _select_best_voice(_all_voices(), language, quality, gender,
                              preferred=preferred, fallback_language=DEFAULT_LANGUAGE)


def _effective_language(language: Optional[str], text: str, voices: Dict[str, VoiceInfo]) -> str:
    """The language to pick a voice for.

    An explicit language is used as given. Omitted or "auto" means the service
    decides: a confident German/English guess from the text (only if a voice for
    it is installed, so a wrong guess cannot turn into a "fallback"), otherwise
    PIPER_DEFAULT_LANGUAGE.
    """
    requested = (language or "").strip()
    if base_language(requested):
        return requested
    if AUTO_DETECT:
        guess = detect_language(text)
        if guess and any(language_matches(v.language, guess) for v in voices.values()):
            return guess
    return DEFAULT_LANGUAGE


def _header_value(value: str) -> str:
    """A client-supplied string made safe for a response header (latin-1, no CR/LF)."""
    return value if value.isascii() and value.isprintable() else "invalid"

# ffprobe reads a header; it should answer immediately.
FFPROBE_TIMEOUT_S = 30


async def analyze_audio_with_ffmpeg(file_path: str) -> Dict:
    """Run ``ffprobe`` on *file_path* and return codec/duration/quality metadata."""
    try:
        # Get basic audio info
        cmd = [
            "ffprobe", "-v", "quiet", "-print_format", "json", 
            "-show_format", "-show_streams", file_path
        ]
        
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        # Bounded: ffprobe only reads a header, so it should answer at once. A
        # wedged one would otherwise keep the request and the child process
        # alive for as long as the client is willing to wait.
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=FFPROBE_TIMEOUT_S)
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            return {"error": f"ffprobe timed out after {FFPROBE_TIMEOUT_S}s"}

        if process.returncode != 0:
            return {"error": f"FFmpeg analysis failed: {stderr.decode()}"}
        
        ffprobe_data = json.loads(stdout.decode())
        
        # Extract audio stream info
        audio_stream = None
        for stream in ffprobe_data.get("streams", []):
            if stream.get("codec_type") == "audio":
                audio_stream = stream
                break
        
        if not audio_stream:
            return {"error": "No audio stream found"}
        
        analysis = {
            "duration": float(ffprobe_data.get("format", {}).get("duration", 0)),
            "sample_rate": int(audio_stream.get("sample_rate", 0)),
            "channels": int(audio_stream.get("channels", 0)),
            "codec": audio_stream.get("codec_name"),
            "bit_rate": int(audio_stream.get("bit_rate", 0)) if audio_stream.get("bit_rate") else None,
            "format": ffprobe_data.get("format", {}).get("format_name"),
            "size_bytes": int(ffprobe_data.get("format", {}).get("size", 0))
        }
        
        # Add quality assessment
        if analysis["sample_rate"] >= 22050 and analysis["channels"] >= 1:
            analysis["quality_assessment"] = "good"
        elif analysis["sample_rate"] >= 16000:
            analysis["quality_assessment"] = "acceptable"
        else:
            analysis["quality_assessment"] = "poor"
        
        return analysis
        
    except Exception as e:
        return {"error": f"Audio analysis failed: {str(e)}"}

# --- bounded work: concurrency slots, upload spooling -------------------------

# The semaphore belongs to the running event loop (asyncio primitives are bound to
# one), so it is created on first use and re-created if the loop changes.
_SLOTS: Optional[tuple] = None


def _slots() -> asyncio.Semaphore:
    global _SLOTS
    loop = asyncio.get_running_loop()
    if _SLOTS is None or _SLOTS[0] is not loop:
        _SLOTS = (loop, asyncio.Semaphore(MAX_CONCURRENCY))
    return _SLOTS[1]


async def _acquire_slot() -> asyncio.Semaphore:
    """Wait for one of PIPER_MAX_CONCURRENCY slots; 503 if none frees up within the timeout.

    Waiting is bounded by the same budget as the work itself, so a saturated
    instance sheds load with a Retry-After instead of queueing requests forever.
    """
    slots = _slots()
    try:
        await asyncio.wait_for(slots.acquire(), timeout=SYNTH_TIMEOUT_S)
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=503,
            detail=f"All {MAX_CONCURRENCY} synthesis slots are busy (PIPER_MAX_CONCURRENCY).",
            headers={"Retry-After": str(max(1, int(SYNTH_TIMEOUT_S // 4)))},
        )
    return slots


async def _run_in_slot(func, *args):
    """Run blocking *func* in a worker thread inside a slot, bounded by PIPER_TIMEOUT_S.

    A thread cannot be killed, so on timeout the request is answered 504 at once
    but the slot stays taken until the thread really finishes: the limit counts
    running work, not answered requests.
    """
    slots = await _acquire_slot()

    def _release(task: "asyncio.Future"):
        if not task.cancelled():
            task.exception()  # mark retrieved; the caller may have stopped listening
        slots.release()

    try:
        task = asyncio.ensure_future(asyncio.to_thread(func, *args))
    except BaseException:
        slots.release()
        raise
    task.add_done_callback(_release)
    try:
        return await asyncio.wait_for(asyncio.shield(task), timeout=SYNTH_TIMEOUT_S)
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=504,
            detail=f"Synthesis timed out after {SYNTH_TIMEOUT_S:g}s (PIPER_TIMEOUT_S).",
        )


class _TooLarge(Exception):
    pass


async def _spool_upload(upload: UploadFile, dest, limit: int, what: str) -> int:
    """Copy *upload* to *dest* in chunks off the event loop; 413 (and no file) past *limit*."""
    detail = f"{what} is larger than {limit / _MIB:g} MB."
    if upload.size is not None and upload.size > limit:
        raise HTTPException(status_code=413, detail=detail)

    def _copy() -> int:
        total = 0
        with open(dest, "wb") as out:
            while True:
                chunk = upload.file.read(_UPLOAD_CHUNK)
                if not chunk:
                    return total
                total += len(chunk)
                if total > limit:
                    raise _TooLarge()
                out.write(chunk)

    try:
        return await asyncio.to_thread(_copy)
    except _TooLarge:
        _unlink_quiet(dest)
        raise HTTPException(status_code=413, detail=detail)
    except BaseException:
        _unlink_quiet(dest)
        raise


def _librosa_summary(path: str) -> Dict:
    """Signal statistics for an audio file (CPU-bound: call from a worker thread)."""
    audio_data, sr = librosa.load(path, sr=None)
    return {
        "duration": len(audio_data) / sr,
        "sample_rate": sr,
        "rms_energy": float(librosa.feature.rms(y=audio_data)[0].mean()),
        "zero_crossing_rate": float(librosa.feature.zero_crossing_rate(audio_data)[0].mean()),
        "spectral_centroid": float(librosa.feature.spectral_centroid(y=audio_data, sr=sr)[0].mean()),
    }


@app.post("/analyze_audio")
async def analyze_audio(audio_file: UploadFile = File(...)):
    """Upload an audio file and receive codec, duration, and quality metadata."""
    fd, temp_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        await _spool_upload(audio_file, temp_path, MAX_ANALYZE_BYTES, "Audio file")

        # Analyze with ffmpeg
        analysis = await analyze_audio_with_ffmpeg(temp_path)

        # Add librosa analysis for more details. It decodes and runs FFTs over the
        # whole file, so it goes to a worker thread (the loop also serves /tts) and
        # takes a slot like any other CPU-heavy job.
        try:
            analysis["librosa"] = await _run_in_slot(_librosa_summary, temp_path)
        except HTTPException:
            raise
        except Exception as e:
            analysis["librosa_error"] = str(e)

        return analysis

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Audio analysis failed: {str(e)}")
    finally:
        # Always clean up the temp file, even on failure
        _unlink_quiet(temp_path)

def _custom_onnx_infer(model_path: str, text: str, voice_name: str, speed: float = 1.0) -> bytes:
    """Run direct ONNX inference for custom-trained VITS models.

    Custom models were trained with a character-level IPA phoneme vocab
    via espeak, not Piper's espeak phoneme_id_map format — so the Piper
    binary can't be used. We phonemize here using the same settings as
    training (phonemizer + espeak backend), then look up IDs.

    The exported graph has no duration/length-scale input, so *speed* is
    applied as a pitch-preserving time-stretch on the generated audio.
    """
    from phonemizer import phonemize

    # Load config (piper-tts scans for {voice}.json, not .onnx.json)
    config_path = Path(model_path).parent / f"{Path(model_path).stem}.json"
    if not config_path.exists():
        config_path = Path(model_path).with_suffix('.onnx.json')

    phoneme_to_id: dict = {}
    sample_rate = 22050
    phonemizer_lang = "de"  # sensible default; overridden from config

    if config_path.exists():
        cfg = json.loads(config_path.read_text())
        phoneme_to_id = cfg.get("phoneme_id_map", {})
        sample_rate = cfg.get("audio", {}).get("sample_rate", 22050)
        phonemizer_lang = cfg.get("phonemizer_language", phonemizer_lang)

    # Fallback: phoneme_vocab.json saved alongside the model
    if not phoneme_to_id:
        vocab_path = CUSTOM_MODELS_DIR / voice_name / "phoneme_vocab.json"
        if vocab_path.exists():
            phoneme_to_id = json.loads(vocab_path.read_text())

    if not phoneme_to_id:
        raise RuntimeError(
            f"No phoneme_id_map found for voice '{voice_name}'. "
            "Cannot perform inference without the phoneme vocabulary."
        )

    # Piper-format configs map each phoneme to a *list* of ids ("_": [0]);
    # training vocabs map to a plain int. Normalize to ints so ID lookup
    # below never feeds lists into the int64 tensor.
    phoneme_to_id = normalize_phoneme_id_map(phoneme_to_id)
    if not phoneme_to_id:
        raise RuntimeError(
            f"phoneme_id_map for voice '{voice_name}' contains no usable integer ids."
        )

    # Phonemize exactly as during training:
    #   phonemize(text, language=lang, backend='espeak', strip=True)
    ipa_text = phonemize(
        text,
        language=phonemizer_lang,
        backend='espeak',
        strip=True,
    )

    # Map each IPA character to its training ID
    pad_id   = phoneme_to_id.get("<pad>", 0)
    start_id = phoneme_to_id.get("<start>", pad_id)
    end_id   = phoneme_to_id.get("<end>", pad_id)
    unk_id   = phoneme_to_id.get("<unk>", pad_id)

    ids = [start_id]
    for ch in ipa_text:
        ids.append(phoneme_to_id.get(ch, unk_id))
    ids.append(end_id)

    text_tensor   = np.array([ids], dtype=np.int64)
    length_tensor = np.array([len(ids)], dtype=np.int64)

    # Run ONNX inference (session is cached across requests)
    session = _get_onnx_session(model_path)

    outputs = session.run(
        None,
        {"text": text_tensor, "text_lengths": length_tensor},
    )
    audio = outputs[0]  # shape: (batch, time) or (time,)
    if audio.ndim > 1:
        audio = audio[0]
    audio = audio.astype(np.float32)

    # Apply playback speed (clamped to the UI slider range). Best-effort:
    # fall back to the unstretched audio if the clip is too short to stretch.
    if speed and abs(speed - 1.0) > 1e-3 and audio.size > 0:
        try:
            rate = float(np.clip(speed, 0.5, 2.0))
            audio = librosa.effects.time_stretch(audio, rate=rate).astype(np.float32)
        except Exception as stretch_err:
            logger.warning(f"Speed adjustment failed for voice '{voice_name}': {stretch_err}")

    # Write to WAV buffer
    buf = io.BytesIO()
    sf.write(buf, audio, sample_rate, format="WAV")
    buf.seek(0)
    return buf.read()


async def _client_gone(http_request: Request) -> None:
    """Return once the client has closed the connection."""
    while True:
        message = await http_request.receive()
        if message["type"] == "http.disconnect":
            return


async def _run_piper(cmd: list, text: str, http_request: Request) -> None:
    """Run the piper binary on *text* inside a slot.

    The process is killed on timeout (504) and when the client disconnects (499)
    instead of running to completion for nobody: each one holds a loaded model
    and the cores it was given.
    """
    slots = await _acquire_slot()
    try:
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError:
            raise HTTPException(status_code=500, detail="TTS generation failed: the piper binary is not installed.")
        talk = asyncio.ensure_future(process.communicate(input=text.encode()))
        gone = asyncio.ensure_future(_client_gone(http_request))
        try:
            done, _ = await asyncio.wait(
                {talk, gone}, timeout=SYNTH_TIMEOUT_S, return_when=asyncio.FIRST_COMPLETED)
            if talk in done:
                _, stderr = talk.result()
                if process.returncode != 0:
                    message = stderr.decode(errors="replace").strip()[-2000:] or "Unknown error"
                    raise HTTPException(status_code=500, detail=f"TTS generation failed: {message}")
                return
            timed_out = gone not in done
        finally:
            gone.cancel()
            if not talk.done():
                with contextlib.suppress(ProcessLookupError):
                    process.kill()
                # The pipes close with the process, which lets communicate() return and reap it.
                await asyncio.gather(talk, return_exceptions=True)
        if timed_out:
            raise HTTPException(
                status_code=504,
                detail=f"TTS generation timed out after {SYNTH_TIMEOUT_S:g}s (PIPER_TIMEOUT_S).",
            )
        raise HTTPException(status_code=499, detail="Client closed the request.")
    finally:
        slots.release()


@app.post("/tts")
async def text_to_speech(request: TTSRequest, http_request: Request):
    """Generate speech audio from text.

    Selects the best matching voice if none is specified.  Custom VITS models
    are served via ONNX Runtime; default Piper voices use the Piper CLI.
    """
    output_path = None
    try:
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="Text must not be empty")

        # Piper (and the custom ONNX path) only produce WAV; reject other
        # formats instead of returning mislabeled audio.
        if request.output_format.lower() != "wav":
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported output_format '{request.output_format}': only 'wav' is supported",
            )

        all_voices = _all_voices()
        requested_language = (request.language or "").strip()
        pinned = bool(request.voice) and request.voice in all_voices
        if request.voice and not pinned:
            logger.warning("Voice '%s' is not installed; choosing one by language instead.", request.voice)

        if pinned:
            # The client chose the voice. A language given next to it is only
            # compared with the voice's, never used to change the choice.
            voice_id = request.voice
            wanted_language = requested_language
        else:
            wanted_language = _effective_language(requested_language, request.text, all_voices)
            # PIPER_DEFAULT_VOICE is the operator's pick when nothing more specific
            # was asked. `quality` has a schema default ("medium"), so a request that
            # never mentioned it must not outvote the operator's voice; one that did
            # (the UI always sends it) still selects by quality.
            preferred = DEFAULT_VOICE if DEFAULT_VOICE in all_voices else None
            quality = request.quality
            if preferred and "quality" not in request.model_fields_set:
                quality = None
            voice_id = select_best_voice(wanted_language, quality, request.gender, preferred=preferred)
            if voice_id not in all_voices:
                raise HTTPException(
                    status_code=404,
                    detail=f"No suitable voice found for language '{wanted_language}' and quality '{request.quality}'",
                )

        voice_info = all_voices[voice_id]

        # select_best_voice() degrades to English (then to the default language)
        # when nothing serves the requested language. That keeps the service
        # working, but on its own it means a client asking for German TTS on a
        # deployment with no German voice gets ENGLISH AUDIO for German text
        # — wrong output, HTTP 200, no signal. Make the substitution explicit.
        # `wanted_language` is what the service was trying to serve, so a request
        # that left the language open and hit a missing default is reported too.
        lang_fallback = not language_matches(voice_info.language, wanted_language)
        if lang_fallback:
            logger.warning(
                "No voice for language '%s'; falling back to '%s' (%s). "
                "Set PIPER_STRICT_LANGUAGE=true to fail instead.",
                wanted_language, voice_id, voice_info.language,
            )
            if STRICT_LANGUAGE:
                available = sorted({v.language for v in all_voices.values()})
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"No voice available for language '{wanted_language}'. "
                        f"Available languages: {', '.join(available)}"
                    ),
                )

        headers = {
            "X-Voice-Used": voice_id,
            "X-Language": _header_value(voice_info.language),
            "X-Language-Requested": _header_value(requested_language) if base_language(requested_language) else "auto",
            "X-Language-Fallback": "true" if lang_fallback else "false",
            "X-Quality": _header_value(voice_info.quality),
        }

        # Determine model path
        if voice_info.model_type == "default":
            model_path = str(DEFAULT_MODELS_DIR / f"{voice_id}.onnx")
        else:
            model_path = str(CUSTOM_MODELS_DIR / voice_id / f"{voice_id}.onnx")

        if not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail=f"Model file not found for voice '{voice_id}'")

        # Custom VITS models use a character-level IPA vocab — route to direct ONNX inference
        if voice_info.model_type == "custom":
            wav_bytes = await _run_in_slot(
                _custom_onnx_infer, model_path, request.text, voice_id, request.speed
            )
            return StreamingResponse(io.BytesIO(wav_bytes), media_type="audio/wav", headers=headers)

        # Standard Piper binary for default models. Output pruning runs on a
        # background timer (see _prune_loop), not here.
        output_filename = f"{uuid.uuid4()}.wav"
        output_path = os.path.join(OUTPUT_DIR, output_filename)

        cmd = [
            "piper",
            "--model", model_path,
            "--output_file", output_path
        ]

        if request.speed != 1.0:
            cmd.extend(["--length_scale", str(1.0 / request.speed)])

        # Collapse newlines: the piper CLI treats each input line as a separate
        # utterance and writes them over the same --output_file, so multi-line
        # text would return only the last line's audio.
        await _run_piper(cmd, " ".join(request.text.split()), http_request)

        if not os.path.exists(output_path):
            raise HTTPException(status_code=500, detail="TTS output file was not created")

        response = FileResponse(
            path=output_path,
            filename=output_filename,
            media_type="audio/wav",
            headers=headers,
            # Nothing else serves OUTPUT_DIR (there is no StaticFiles mount), so
            # the file is dead the moment it has been streamed.
            background=BackgroundTask(os.unlink, output_path),
        )
        output_path = None  # the response owns the file now
        return response

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # A killed, failed or timed-out piper leaves a partial WAV behind.
        if output_path:
            _unlink_quiet(output_path)

@app.post("/synthesize")
async def synthesize_with_custom_voice(request: VoiceCloneRequest):
    """Synthesize speech with a named custom voice (uses ONNX Runtime)."""
    voice_name = _sanitize_voice_name(request.voice_name)

    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text must not be empty")

    if voice_name not in CUSTOM_VOICES:
        raise HTTPException(status_code=404, detail=f"Custom voice '{voice_name}' not found")

    model_path = str(CUSTOM_MODELS_DIR / voice_name / f"{voice_name}.onnx")
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail=f"Custom model file not found for '{voice_name}'")

    try:
        wav_bytes = await _run_in_slot(
            _custom_onnx_infer, model_path, request.text, voice_name, request.speed
        )
        return StreamingResponse(
            io.BytesIO(wav_bytes),
            media_type="audio/wav",
            headers={"X-Custom-Voice": voice_name},
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _custom_voice_info(voice_name: str, config: dict) -> VoiceInfo:
    """Voice metadata from a custom model's config (its ``model_card`` and ``audio`` blocks)."""
    card = config.get("model_card") if isinstance(config.get("model_card"), dict) else {}
    audio = config.get("audio") if isinstance(config.get("audio"), dict) else {}
    return VoiceInfo(
        name=voice_name,
        language=card.get("language", "en"),
        speaker=card.get("speaker", voice_name),
        quality=audio.get("quality", "medium"),
        sample_rate=audio.get("sample_rate", 22050),
        model_type="custom",
    )


@app.post("/upload_model")
async def upload_custom_model(
    model_file: UploadFile = File(...),
    config_file: Optional[UploadFile] = File(None),
    voice_name: str = Form(...),
    model_name: str = Form(None),
):
    """Upload a custom-trained ONNX model and optional JSON config."""
    part_paths = []
    created_dir = False
    try:
        final_voice_name = _sanitize_voice_name(model_name or voice_name)

        voice_dir = CUSTOM_MODELS_DIR / final_voice_name
        created_dir = not voice_dir.exists()
        voice_dir.mkdir(parents=True, exist_ok=True)
        model_path = voice_dir / f"{final_voice_name}.onnx"
        config_path = voice_dir / f"{final_voice_name}.json"

        # Everything is written beside its final name and moved into place only
        # once the whole upload checked out. Writing straight to the final path
        # truncated a working voice at the first byte of a re-upload, and a bad
        # config left a model without one.
        model_part = voice_dir / f".{final_voice_name}.onnx.part"
        config_part = voice_dir / f".{final_voice_name}.json.part"
        part_paths = [model_part, config_part]

        await _spool_upload(model_file, model_part, MAX_UPLOAD_BYTES, "Model file")

        if config_file:
            await _spool_upload(config_file, config_part, MAX_CONFIG_BYTES, "Config file")
            try:
                config = json.loads(await asyncio.to_thread(config_part.read_text, "utf-8"))
            except ValueError as e:
                raise HTTPException(status_code=400, detail=f"Config file is not valid JSON: {e}")
            if not isinstance(config, dict):
                raise HTTPException(status_code=400, detail="Config file must contain a JSON object.")
        else:
            # Generate basic config if none provided
            config = {
                "audio": {
                    "sample_rate": 22050,
                    "quality": "medium"
                },
                "espeak": {
                    "voice": "en-us"
                },
                "inference": {
                    "noise_scale": 0.667,
                    "length_scale": 1,
                    "noise_w": 0.8
                },
                "phoneme_type": "espeak",
                "phoneme_map": {},
                "phoneme_id_map": {
                    "_": [0], "^": [1], "$": [2], " ": [3]
                },
                "model_card": {
                    "language": "en",
                    "speaker": final_voice_name,
                    "dataset": f"Custom training - {final_voice_name}",
                    "license": "Custom"
                }
            }
            await asyncio.to_thread(config_part.write_text, json.dumps(config, indent=2), "utf-8")

        try:
            voice_info = _custom_voice_info(final_voice_name, config)
        except ValidationError as e:
            raise HTTPException(status_code=400, detail=f"Config file has unusable voice metadata: {e.errors()[0]['msg']}")

        os.replace(model_part, model_path)
        os.replace(config_part, config_path)
        created_dir = False

        CUSTOM_VOICES[final_voice_name] = voice_info

        return {
            "status": "success",
            "message": f"Custom voice '{final_voice_name}' uploaded successfully",
            "voice_info": voice_info
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model upload failed: {str(e)}")
    finally:
        for part in part_paths:
            _unlink_quiet(part)
        if created_dir:
            with contextlib.suppress(OSError):
                voice_dir.rmdir()

@app.delete("/voice/{voice_name}")
async def delete_custom_voice(voice_name: str):
    """Delete a custom voice and its model files."""
    try:
        voice_name = _sanitize_voice_name(voice_name)

        if voice_name not in CUSTOM_VOICES:
            raise HTTPException(status_code=404, detail=f"Custom voice '{voice_name}' not found")

        voice_dir = CUSTOM_MODELS_DIR / voice_name
        if voice_dir.exists():
            shutil.rmtree(voice_dir)

        # Drop any cached ONNX session for this voice's model
        with _ONNX_CACHE_LOCK:
            _ONNX_SESSION_CACHE.pop(str(voice_dir / f"{voice_name}.onnx"), None)

        # Remove from custom voices
        del CUSTOM_VOICES[voice_name]
        
        return {"status": "success", "message": f"Custom voice '{voice_name}' deleted"}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/voice/{voice_name}")
async def get_voice_info(voice_name: str):
    """Return metadata for a single voice by name."""
    all_voices = _all_voices()
    if voice_name not in all_voices:
        raise HTTPException(status_code=404, detail=f"Voice '{voice_name}' not found")
    
    return all_voices[voice_name]

def _scan_custom_voices() -> Dict[str, VoiceInfo]:
    """Read every valid custom voice under ``CUSTOM_MODELS_DIR`` (blocking file IO)."""
    found: Dict[str, VoiceInfo] = {}
    if not CUSTOM_MODELS_DIR.exists():
        return found

    for voice_dir in sorted(CUSTOM_MODELS_DIR.iterdir()):
        if voice_dir.is_dir():
            voice_name = voice_dir.name
            config_path = voice_dir / f"{voice_name}.json"
            model_path = voice_dir / f"{voice_name}.onnx"

            if config_path.exists() and model_path.exists():
                try:
                    with open(config_path, 'r') as f:
                        config = json.load(f)

                    found[voice_name] = _custom_voice_info(voice_name, config)
                    logger.info(f"Loaded custom voice: {voice_name}")

                except Exception as e:
                    logger.warning(f"Failed to load custom voice {voice_name}: {e}")
    return found


async def load_custom_voices():
    """Scan ``CUSTOM_MODELS_DIR`` and register each valid voice into ``CUSTOM_VOICES``."""
    found = await asyncio.to_thread(_scan_custom_voices)
    # No await between clear and update: the registry is never seen half-empty.
    CUSTOM_VOICES.clear()
    CUSTOM_VOICES.update(found)

@app.post("/refresh_voices")
async def refresh_voices():
    """Re-scan the default and custom model directories and update the voice registry."""
    # Drop cached sessions too — a removed voice would otherwise keep serving
    # from a session whose model file no longer exists.
    with _ONNX_CACHE_LOCK:
        _ONNX_SESSION_CACHE.clear()
    default_count = await asyncio.to_thread(_refresh_default_voices)
    await load_custom_voices()
    
    return {
        "status": "success",
        "message": "Voice list refreshed",
        "default_voices": default_count,
        "custom_voices": len(CUSTOM_VOICES)
    }

if __name__ == "__main__":
    # timeout_keep_alive: the gateway pools connections to this service, and
    # uvicorn's 5 s default closes them between requests, racing the next one.
    uvicorn.run(app, host="0.0.0.0", port=5000, timeout_keep_alive=120)
