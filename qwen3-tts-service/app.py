"""Qwen3-TTS text-to-speech and voice-cloning service.

Supports multiple Qwen3-TTS model variants (Base, CustomVoice, VoiceDesign)
with a persistent voice library for saved speaker embeddings.
"""

import os
import io
import gc
import json
import re
import shutil
import time
import asyncio
import subprocess
import threading
import tempfile
import logging
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path

import torch
import numpy as np
import soundfile as sf
import httpx
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import Response, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _number(raw, name, default, cast=int, minimum=None):
    """Parse a numeric env var; junk or out-of-range input falls back to *default*.

    A typo in a tuning knob should cost the knob, not stop the service at import.
    """
    if raw is None or str(raw).strip() == "":
        return default
    try:
        value = cast(str(raw).strip())
    except ValueError:
        logger.warning("Ignoring invalid %s=%r; using %s", name, raw, default)
        return default
    if minimum is not None and value < minimum:
        logger.warning("Ignoring %s=%r (below %s); using %s", name, raw, minimum, default)
        return default
    return value


# --- Model registry ----------------------------------------------------------

# 0.6B Base is the deployment default (compose, .env.example): it leaves room for
# the ASR stack on a 12 GB shared card. The 1.7B variants are one /load_model or
# QWEN3_TTS_MODEL away.
DEFAULT_MODEL = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"

# Available Qwen3-TTS model variants. The single source of truth for what a
# loaded model can do: endpoints route on "capabilities", never on hasattr.
AVAILABLE_MODELS = {
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base": {
        "name": "1.7B Base",
        "description": "General-purpose TTS and voice cloning (4.5GB)",
        "size": "1.7B",
        "capabilities": ["tts", "voice_clone"],
    },
    "Qwen/Qwen3-TTS-12Hz-0.6B-Base": {
        "name": "0.6B Base",
        "description": "Smaller, faster model for TTS and voice cloning (2.5GB)",
        "size": "0.6B",
        "capabilities": ["tts", "voice_clone"],
    },
    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice": {
        "name": "1.7B CustomVoice",
        "description": "Optimized for custom voice synthesis with built-in speakers (4.5GB)",
        "size": "1.7B",
        "capabilities": ["tts", "custom_voice"],
    },
    "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice": {
        "name": "0.6B CustomVoice",
        "description": "Smaller built-in-speaker model; no instruction control (2.5GB)",
        "size": "0.6B",
        "capabilities": ["tts", "custom_voice"],
    },
    "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign": {
        "name": "1.7B VoiceDesign",
        "description": "Design voices via text description (4.5GB)",
        "size": "1.7B",
        "capabilities": ["tts", "voice_design"],
    },
}

_BASE_MODELS = "a Base model (1.7B Base or 0.6B Base)"
_CUSTOM_VOICE_MODELS = (
    "a CustomVoice model (1.7B or 0.6B CustomVoice) via POST /load_model, "
    "or use /clone or a saved voice instead"
)


def _configured_model() -> str:
    """The model the environment asks for when nothing was switched at runtime."""
    return (os.getenv("QWEN3_TTS_MODEL") or "").strip() or DEFAULT_MODEL


# --- Languages ---------------------------------------------------------------

# Spelling the qwen-tts API expects (README examples: "Chinese", "English", ...;
# it validates case-insensitively). "Auto" is accepted by the model itself and
# means "infer from the text".
SUPPORTED_LANGUAGES = [
    "Chinese", "English", "Japanese", "Korean", "German",
    "French", "Russian", "Portuguese", "Spanish", "Italian"
]
_LANGUAGE_BY_KEY = {name.lower(): name for name in SUPPORTED_LANGUAGES}
_LANGUAGE_BY_CODE = {
    "zh": "Chinese", "en": "English", "ja": "Japanese", "ko": "Korean",
    "de": "German", "fr": "French", "ru": "Russian", "pt": "Portuguese",
    "es": "Spanish", "it": "Italian",
}


def _lookup_language(value: Optional[str]) -> Optional[str]:
    """Canonical spelling for a language name or ISO code; None when unsupported.

    ``"Auto"`` maps to itself. Codes with a region ("de-DE", "zh_CN") resolve by
    their primary subtag.
    """
    key = (value or "").strip().lower().replace("-", "_")
    if not key:
        return None
    if key == "auto":
        return "Auto"
    if key in _LANGUAGE_BY_KEY:
        return _LANGUAGE_BY_KEY[key]
    return _LANGUAGE_BY_CODE.get(key.split("_", 1)[0])


def _default_language() -> str:
    """QWEN3_DEFAULT_LANGUAGE, validated; German when unset or unsupported.

    ``Auto`` is allowed here on purpose: it is how an operator opts in to the
    model's own language inference instead of a fixed default.
    """
    raw = os.getenv("QWEN3_DEFAULT_LANGUAGE")
    if raw is None or not raw.strip():
        return "German"
    found = _lookup_language(raw)
    if found is None:
        logger.warning(
            "Ignoring unsupported QWEN3_DEFAULT_LANGUAGE=%r; using German. Supported: %s, Auto",
            raw, ", ".join(SUPPORTED_LANGUAGES),
        )
        return "German"
    return found


# Applied when a request says "auto" or names no language at all. The platform is
# German-first, and the previous silent default (English) made every German text
# without an explicit language come out with an English accent.
DEFAULT_LANGUAGE = _default_language()


def _resolve_language(lang: Optional[str]) -> str:
    """The spelling to hand the model for a request's ``lang``.

    "auto" and a missing value mean QWEN3_DEFAULT_LANGUAGE. A language the model
    cannot speak is a 400 that lists what it can: mapping "nl" or "pl" to English
    used to hand the caller English speech for Dutch text with no signal at all.
    """
    if not (lang or "").strip() or lang.strip().lower() == "auto":
        return DEFAULT_LANGUAGE
    found = _lookup_language(lang)
    if found is None:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Language '{lang}' is not supported by Qwen3-TTS. Supported: "
                f"{', '.join(SUPPORTED_LANGUAGES)} (names or ISO codes), or 'auto' for "
                f"the service default ({DEFAULT_LANGUAGE})."
            ),
        )
    return found


# --- Speakers ----------------------------------------------------------------

# The nine CustomVoice speakers documented in the qwen-tts README. Only a labelled
# fallback for /speakers while no model is resident: a loaded model's own
# get_supported_speakers() always wins, and Base/VoiceDesign models have none.
BUILTIN_SPEAKERS = [
    "Vivian", "Serena", "Uncle_Fu", "Dylan", "Eric",
    "Ryan", "Aiden", "Ono_Anna", "Sohee",
]
_SPEAKER_SPELLING = {name.lower(): name for name in BUILTIN_SPEAKERS}


def _model_speakers(model) -> Optional[list]:
    """Speaker names the loaded model reports, or None when it cannot say.

    qwen-tts lower-cases them; restore the documented spelling where we know it.
    An empty list is an answer: a Base model has no built-in speakers.
    """
    try:
        names = model.get_supported_speakers()
    except Exception as e:
        logger.warning("Could not read the model's speaker list: %s", e)
        return None
    if names is None:
        return None
    return [_SPEAKER_SPELLING.get(str(n).lower(), str(n)) for n in names]


def _speaker_catalog() -> tuple:
    """``(speakers, source)`` for /speakers and /status. Never loads a model.

    ``source`` says where the list came from: ``model`` (the loaded weights),
    ``registry`` (no model resident, but the target variant declares no
    custom_voice, so there are none) or ``fallback`` (the documented constant).
    """
    model = tts_model
    if model is not None:
        names = _model_speakers(model)
        if names is not None:
            return names, "model"
    target = current_model_name or _desired_model_name or _configured_model()
    info = AVAILABLE_MODELS.get(target)
    if info is not None and "custom_voice" not in info["capabilities"]:
        return [], "registry"
    return list(BUILTIN_SPEAKERS), "fallback"


def _validate_speaker(model, speaker: str) -> str:
    """400 (naming the valid ones) for a speaker the loaded model does not have."""
    names = _model_speakers(model)
    if names and str(speaker).strip().lower() not in {n.lower() for n in names}:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown speaker '{speaker}'. Available: {', '.join(names)}",
        )
    return speaker


# --- Configuration -----------------------------------------------------------

# STT service URLs for auto-transcription
QWEN3_ASR_URL = os.getenv("QWEN3_ASR_SERVICE_URL", "http://qwen3-asr-service:5002")

# Input bounds. The gateway limits what it forwards, but a direct caller must not
# be able to ask this service for an hour of speech or upload a disk-filling file.
MAX_TEXT_CHARS = _number(os.getenv("MAX_TEXT_CHARS"), "MAX_TEXT_CHARS", 5000, int, minimum=1)
_MIB = 1024 * 1024
MAX_UPLOAD_BYTES = int(
    _number(os.getenv("MAX_UPLOAD_MB"), "MAX_UPLOAD_MB", 20.0, float, minimum=0.001) * _MIB
)
_UPLOAD_CHUNK = _MIB
# multipart boundaries and part headers on top of the file bytes
_MULTIPART_SLACK = _MIB

# Attention kernel. `auto` keeps the historic behaviour (FlashAttention-2 when the
# package imports on CUDA, else SDPA); the image no longer ships flash-attn, so
# that means SDPA unless you built your own image.
ATTN_IMPLEMENTATION = (os.getenv("QWEN3_TTS_ATTN_IMPLEMENTATION") or "auto").strip().lower()

allowed_origins_str = os.getenv("ALLOWED_ORIGINS", "*")
allowed_origins = [origin.strip() for origin in allowed_origins_str.split(",")] if allowed_origins_str else ["*"]
allow_credentials = os.getenv("ALLOW_CREDENTIALS", "false").strip().lower() in {"1", "true", "yes", "on"}
if "*" in allowed_origins and allow_credentials:
    allow_credentials = False


# --- Global model state ------------------------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"
tts_model = None
model_loaded = False
current_model_name = ""
# What the operator asked for via /load_model. Distinct from
# current_model_name (which is "" while a load is in flight): this is what
# an idle-TTL reload must restore, otherwise unloading would silently revert
# the user back to the env default.
_desired_model_name = None
# True while load_model() runs, and the reason the last attempt failed. Both are
# read lock-free by /ready and /health, which must never wait on a load.
_loading = False
_last_load_error = None
# A startup preload was scheduled and has not finished: /ready reports "loading"
# for that window rather than claiming ready before the first load even began.
_preload_pending = False

# One process-global model on one GPU. Without this, requests dispatched onto the
# default thread pool (min(32, cpu+4) workers) all enter the model at once and
# multiply peak VRAM while making every individual request slower.
_GEN_SEM = asyncio.Semaphore(_number(
    os.getenv("TTS_MAX_CONCURRENCY"), "TTS_MAX_CONCURRENCY", 1, int, minimum=1))
# Sentences generated in one forward pass. Peak VRAM scales with this, so it has
# to be bounded by configuration rather than by how long a caller's text is.
TTS_MAX_BATCH = _number(os.getenv("TTS_MAX_BATCH"), "TTS_MAX_BATCH", 8, int, minimum=1)
# Serialises model loading, switching, unloading and the idle reaper against each
# other. Requests take it only to obtain the model, never while generating.
_LOAD_LOCK = asyncio.Lock()
# How long /load_model waits for requests that already hold the current model
# (a /voices/save can sit on the ASR service for a minute) before answering 409.
_SWITCH_DRAIN_TIMEOUT = 180.0


# Idle unloading. This service cannot use the simple ModelSlot pattern because
# the model identity can CHANGE at runtime via /load_model, so residency is
# tracked here instead: a last-used timestamp, gated by an in-flight counter.
# The counter is the safety property — a timestamp alone would happily free the
# weights while a request was still using them.
#   >0 = seconds idle before unloading | 0 = unload as soon as idle | -1 = never
MODEL_TTL = _number(os.getenv("TTS_MODEL_TTL") or os.getenv("MODEL_TTL"), "TTS_MODEL_TTL", 300.0, float)
# Holds on the resident model: one per request that obtained it via
# _acquire_model() (for the request's whole life, not just its generation) plus
# one per generation thread still running, so a cancelled request cannot free
# weights its worker is still reading. Every safety decision (reaper, /unload,
# /load_model) goes by this count.
_inflight = 0
_inflight_lock = threading.Lock()
# Requests currently inside _acquire_model(); what /health reports, because
# _inflight double-counts a request whose generation is running.
_active_requests = 0
_last_used = time.monotonic()


def _touch_model() -> None:
    """Mark the model as used now, restarting its idle countdown."""
    global _last_used
    _last_used = time.monotonic()


def _should_unload(now: Optional[float] = None) -> bool:
    """Whether the idle model may be released right now.

    Split out from the loop so the decision is testable without waiting on a
    real sleep. The in-flight check is the safety property: a timestamp alone
    would happily free weights a request is still using.
    """
    if MODEL_TTL < 0:
        return False
    if tts_model is None:
        return False
    with _inflight_lock:
        if _inflight > 0:
            return False
    elapsed = (now if now is not None else time.monotonic()) - _last_used
    return elapsed >= MODEL_TTL


def _reaper_tick_seconds() -> float:
    """Poll interval: often enough to honour the TTL, rarely enough to be free."""
    if MODEL_TTL <= 0:
        return 5.0
    return max(1.0, min(30.0, MODEL_TTL / 4))


async def _idle_reaper() -> None:
    """Unload the model once it has been idle for MODEL_TTL seconds."""
    if MODEL_TTL < 0:
        logger.info("Qwen3-TTS idle unloading disabled (TTS_MODEL_TTL=-1)")
        return
    while True:
        await asyncio.sleep(_reaper_tick_seconds())
        try:
            if not _should_unload():
                continue
            # _LOAD_LOCK keeps this from racing a /load_model swap or a
            # request obtaining the model. Requests raise _inflight while holding
            # it, so the re-check below sees every request that got the model
            # while we waited for the lock.
            async with _LOAD_LOCK:
                if _should_unload():
                    await asyncio.to_thread(_unload_qwen3_tts)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning("Idle reaper error: %s", e)


def _release_model() -> None:
    """Drop the model and let the caching allocator return VRAM to the driver.

    The order is the point. The wrapper and its modules sit in reference cycles,
    so `empty_cache()` before `gc.collect()` finds the weights still allocated
    and returns nothing: a model switch then peaked at old + new VRAM.
    """
    global tts_model, model_loaded, current_model_name
    tts_model = None
    model_loaded = False
    current_model_name = ""
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()


def _unload_qwen3_tts() -> None:
    """Unload the resident model (idle reaper and POST /unload)."""
    if tts_model is None:
        return
    logger.info("Unloading Qwen3-TTS model '%s'", current_model_name)
    _release_model()


def _release_gen(_fut=None) -> None:
    """A generation thread finished: drop its hold and its concurrency permit."""
    global _inflight
    if _fut is not None and not _fut.cancelled():
        _fut.exception()  # mark retrieved; an abandoned awaiter never will
    with _inflight_lock:
        _inflight -= 1
    _touch_model()
    _GEN_SEM.release()


async def _gen(fn, *args, **kwargs):
    """Run a blocking model call off the event loop, bounded by _GEN_SEM.

    The permit and the in-flight hold are released when the WORKER THREAD ends,
    not when this coroutine does. If the caller is cancelled (client gone,
    shutdown) the thread keeps reading the weights, and releasing either early
    would let the reaper or a model switch pull them out from under it.
    """
    global _inflight
    await _GEN_SEM.acquire()
    with _inflight_lock:
        _inflight += 1
    try:
        fut = asyncio.ensure_future(asyncio.to_thread(fn, *args, **kwargs))
    except BaseException:
        _release_gen()
        raise
    fut.add_done_callback(_release_gen)
    return await asyncio.shield(fut)


def _require_capability(model_name: str, capability: str, switch_to: str, *,
                        status_code: int = 400, what: Optional[str] = None) -> None:
    """Refuse a request the loaded model variant cannot serve.

    Declared capabilities, never ``hasattr``. The variants share one class, so
    the method is present on all of them and raises at call time on the ones
    that cannot do the work — `/tts` documents this for
    ``generate_custom_voice`` and routes on ``AVAILABLE_MODELS`` accordingly.
    ``/voice_design`` still probed with ``hasattr`` and ``/clone-with-ref-text``
    checked nothing at all, so both turned "wrong model loaded" into a raw 500
    instead of the actionable 400 every sibling endpoint returns.

    *what* names the feature in the message when the capability id reads badly.
    """
    model_info = AVAILABLE_MODELS.get(model_name, {})
    if capability in model_info.get("capabilities", []):
        return
    raise HTTPException(
        status_code=status_code,
        detail=(
            f"Current model '{model_info.get('name', model_name or 'unknown')}' does not "
            f"support {what or capability.replace('_', ' ')}. Please switch to {switch_to}."
        ),
    )


def _safe_header(value: str) -> str:
    """Make a string safe to use as an HTTP header value.

    Header values must be latin-1 encodable and free of control characters.
    Without this a non-latin-1 reference filename turned an already-completed
    generation into a 500.
    """
    return re.sub(r"[\x00-\x1f\x7f]", " ", (value or "").encode("latin-1", "replace").decode("latin-1"))


def _attn_implementation() -> str:
    """The `attn_implementation` to load with; see ATTN_IMPLEMENTATION."""
    wanted = ATTN_IMPLEMENTATION
    if wanted not in ("auto", "sdpa", "eager", "flash_attention_2"):
        logger.warning("Ignoring unknown QWEN3_TTS_ATTN_IMPLEMENTATION=%r; using auto", wanted)
        wanted = "auto"
    if wanted in ("sdpa", "eager"):
        return wanted
    if device != "cuda":
        # FlashAttention-2 kernels are CUDA-only and want half precision, which
        # CPU mode (float32) does not use.
        return "sdpa"
    try:
        import flash_attn  # noqa: F401
    except ImportError:
        if wanted == "flash_attention_2":
            logger.warning("QWEN3_TTS_ATTN_IMPLEMENTATION=flash_attention_2 but flash-attn is not installed; using SDPA")
        else:
            logger.info("flash-attn not available, using SDPA attention")
        return "sdpa"
    return "flash_attention_2"


def load_model(model_name=None):
    """Load a Qwen3-TTS model by name.  Unloads the previous model first if switching."""
    global tts_model, model_loaded, current_model_name, _loading, _last_load_error

    if model_name is None:
        model_name = _configured_model()

    # Already loaded
    if tts_model is not None and current_model_name == model_name:
        return tts_model

    _loading = True
    try:
        # Unload previous model
        if tts_model is not None:
            logger.info(f"Unloading current model: {current_model_name}")
            _release_model()

        if model_name not in AVAILABLE_MODELS:
            logger.warning(
                "Model '%s' is not in the registry: every endpoint will refuse it "
                "because its capabilities are unknown", model_name)
        logger.info(f"Loading Qwen3-TTS model: {model_name}")
        from qwen_tts import Qwen3TTSModel

        tts_model = Qwen3TTSModel.from_pretrained(
            model_name,
            device_map=f"{device}:0" if device == "cuda" else "cpu",
            dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            attn_implementation=_attn_implementation(),
        )
        model_loaded = True
        current_model_name = model_name
        _last_load_error = None
        logger.info(f"Qwen3-TTS model '{model_name}' loaded on {device}")
    except Exception as e:
        _last_load_error = f"{type(e).__name__}: {e}"[:300]
        logger.error(f"Failed to load Qwen3-TTS model: {e}", exc_info=True)
        # A half-loaded model can already hold GBs; give them back before the
        # next attempt instead of stacking a second load on top.
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
        raise
    finally:
        _loading = False
    return tts_model


@asynccontextmanager
async def _acquire_model():
    """Hold the model for the whole of a request: ``async with _acquire_model() as (model, name)``.

    The model and its name arrive as one consistent snapshot, and the request
    counts as in flight from here until the ``with`` block ends, however it ends
    (return, error, cancellation). That is what keeps the idle reaper and
    /unload from freeing weights mid-request: /voices/save waits on the ASR
    service for up to a minute *without* a generation running, and the reaper
    used to unload the model in that gap (leaving ``model_used`` blank on the
    saved voice, and two models resident once the request reloaded one).

    Taking ``_LOAD_LOCK`` to obtain the model matters twice:

    - ``load_model`` blanks ``tts_model`` before it starts loading the new
      weights, so an unguarded load landing in that window would kick off a
      *second* concurrent ``from_pretrained`` — two multi-GB models allocating
      at once.
    - ``current_model_name`` is blanked in the same window. Reading it
      separately made capability checks fail with a spurious
      "does not support voice cloning" for the whole duration of a swap.

    Note this reloads ``_desired_model_name``, not the env default: after the
    idle TTL has unloaded a model the user explicitly switched to, reloading the
    env default instead would silently revert their choice.
    """
    global _inflight, _active_requests
    async with _LOAD_LOCK:
        model = tts_model
        if model is None:
            model = await asyncio.to_thread(load_model, _desired_model_name)
        name = current_model_name
        # Raised before the lock is released, so the reaper's re-check under the
        # same lock cannot miss this request.
        with _inflight_lock:
            _inflight += 1
            _active_requests += 1
        _touch_model()
    try:
        yield model, name
    finally:
        with _inflight_lock:
            _inflight -= 1
            _active_requests -= 1
        _touch_model()


async def _drain_requests(timeout: float) -> None:
    """Wait until no request holds the model; 409 if that takes longer than *timeout*.

    Call with ``_LOAD_LOCK`` held: nothing new can obtain the model meanwhile, so
    the count can only fall. Swapping models under a holder would keep the old
    weights alive next to the new ones until it finished.
    """
    deadline = time.monotonic() + timeout
    while True:
        with _inflight_lock:
            busy = _inflight
        if busy <= 0:
            return
        if time.monotonic() >= deadline:
            raise HTTPException(
                status_code=409,
                detail=f"Model is in use by {busy} request(s); retry when idle",
            )
        await asyncio.sleep(0.05)


async def _startup_preload() -> None:
    """Load the model in the background so the port opens at once.

    /health answers while a first-time download is still running, and /ready
    reports 503 "loading" for exactly as long as it takes. The lock is the one
    requests obtain the model under, so one that arrives mid-preload waits for
    this load instead of starting a second copy.
    """
    global _preload_pending
    try:
        async with _LOAD_LOCK:
            if tts_model is None:
                await asyncio.to_thread(load_model, _desired_model_name)
        _touch_model()
    except asyncio.CancelledError:
        raise
    except Exception as e:
        logger.warning(f"Could not preload model: {e}")
    finally:
        _preload_pending = False


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    """Pre-load the model in the background so the first request is fast.

    The idle reaper then releases it if nothing uses it for TTS_MODEL_TTL
    seconds, so an untouched service does not hold multiple GB forever on a
    card it shares with the ASR stack. With TTL 0 the model would be dropped
    at the first tick anyway, so it is not preloaded at all.
    """
    global _preload_pending
    preload = None
    if MODEL_TTL != 0:
        _preload_pending = True
        preload = asyncio.create_task(_startup_preload())
    reaper = asyncio.create_task(_idle_reaper())
    try:
        yield
    finally:
        reaper.cancel()
        if preload is not None:
            preload.cancel()


app = FastAPI(
    title="Qwen3-TTS Service",
    description="Text-to-Speech and Voice Cloning using Qwen3-TTS",
    lifespan=_lifespan,
)

_UPLOAD_PATHS = {"/clone", "/clone-with-ref-text", "/voices/save"}


def _body_limit(path: str) -> int:
    """Largest request body (bytes) accepted at *path*, by declared Content-Length."""
    # Up to three text fields per request (text, ref_text / instruct / description).
    # 12 bytes per character is the worst case, a JSON-escaped astral character.
    text = MAX_TEXT_CHARS * 12 * 3 + 64 * 1024
    if path in _UPLOAD_PATHS:
        return MAX_UPLOAD_BYTES + _MULTIPART_SLACK + text
    return text


class _BodyLimitMiddleware:
    """Answer 413 from the Content-Length header, before anything is buffered.

    FastAPI parses the whole body (multipart parts are spooled to disk, JSON is
    held in memory) before a handler runs, so a size check inside the handler
    comes after the cost was paid. Clients that send no Content-Length (chunked)
    are still bounded per field by the handlers.
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
                    content={"detail": f"Request body is larger than {limit / _MIB:.3g} MB."},
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


def _require_text(text: Optional[str], field: str = "Text") -> str:
    """400 for a missing/blank text field, 413 past MAX_TEXT_CHARS."""
    if not text or not text.strip():
        raise HTTPException(status_code=400, detail=f"{field} not provided")
    if len(text) > MAX_TEXT_CHARS:
        raise HTTPException(
            status_code=413,
            detail=f"{field} is {len(text)} characters; the limit is {MAX_TEXT_CHARS} (MAX_TEXT_CHARS).",
        )
    return text


def _bound_optional_text(text: Optional[str], field: str) -> str:
    """Length-check a field that may be empty."""
    text = text or ""
    if len(text) > MAX_TEXT_CHARS:
        raise HTTPException(
            status_code=413,
            detail=f"{field} is {len(text)} characters; the limit is {MAX_TEXT_CHARS} (MAX_TEXT_CHARS).",
        )
    return text


@contextmanager
def _as_server_error(what: str):
    """Turn an unexpected failure into a logged 500; HTTPExceptions pass through."""
    try:
        yield
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"{what} error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    """Liveness probe.

    `model_resident: false` is NOT an error — the idle reaper released the
    weights to free VRAM and the next request reloads them (restoring the
    switched-to model, not the env default). A non-200 here would make an idle
    container report unhealthy under Docker's `curl -f` check.
    """
    return {
        "status": "ok",
        "model_loaded": model_loaded,
        "model_resident": tts_model is not None,
        "loading": _loading,
        "model_ttl_seconds": MODEL_TTL,
        "active_requests": _active_requests,
        "current_model": current_model_name,
        "desired_model": _desired_model_name,
        "device": device,
    }


@app.get("/ready")
async def ready():
    """Readiness: can this service take a request now?

    200 when the model is resident, or was unloaded (idle TTL, POST /unload) and
    reloads on demand. 503 with a JSON `reason` while a load is running
    (`loading`) and after the last load attempt failed (`load_failed`; the next
    request retries it).

    Like /health this only reads state and never triggers or waits for a load —
    an orchestrator polling it must not be what keeps the model resident.
    """
    resident = tts_model is not None
    body = {
        "ready": True,
        "reason": "resident",
        "model_resident": resident,
        "current_model": current_model_name,
        "desired_model": _desired_model_name or _configured_model(),
        "device": device,
    }
    if _loading or (_preload_pending and not resident):
        body.update(ready=False, reason="loading")
        return JSONResponse(content=body, status_code=503, headers={"Retry-After": "5"})
    if _last_load_error and not resident:
        body.update(ready=False, reason="load_failed", detail=_last_load_error)
        return JSONResponse(content=body, status_code=503)
    if not resident:
        body["reason"] = "unloaded"
    return body


@app.post("/unload")
async def unload():
    """Release the model and its VRAM now, without stopping the container.

    The idle reaper covers the common case; this is the deliberate one — you are
    about to run something else on the same GPU and want the memory back
    immediately. The next request reloads transparently.

    200 when released or already unloaded; **409 while a request holds the
    model** or a load/switch is running, because freeing weights a request is
    still using would crash it. `inflight` counts the holds on the model
    (requests plus generation threads still running). Retry once it is zero.

    Note this also clears the model selected via POST /load_model — the next
    request reloads whichever model is currently desired, not the env default.
    """
    busy = JSONResponse(
        status_code=409,
        content={
            "detail": "Model is in use; retry when idle",
            "unloaded": False,
            "reason": "busy",
            "inflight": _inflight,
        },
    )
    # A load can run for minutes; answer 409 at once rather than queue behind it.
    if _LOAD_LOCK.locked():
        return busy
    async with _LOAD_LOCK:
        # Requests raise _inflight while holding this lock, so once we have it the
        # count cannot rise: what we read is what is using the model.
        with _inflight_lock:
            in_use = _inflight
        if in_use > 0:
            return busy
        if tts_model is None:
            return {
                "unloaded": False, "reason": "not_resident",
                "inflight": 0, "model_resident": False,
            }
        # gc.collect() + empty_cache() synchronise the device; keep them off the
        # event loop, which /health is served from.
        await asyncio.to_thread(_unload_qwen3_tts)

    return {
        "unloaded": True, "reason": "ok",
        "inflight": 0, "model_resident": tts_model is not None,
    }


@app.get("/status")
async def status():
    """Return detailed service status including GPU memory and loaded model."""
    speakers, speakers_source = _speaker_catalog()
    status_info = {
        "status": "ok",
        "device": device,
        "cuda_available": torch.cuda.is_available(),
        "model_loaded": model_loaded,
        "current_model": current_model_name,
        "current_model_info": AVAILABLE_MODELS.get(current_model_name),
        "supported_languages": SUPPORTED_LANGUAGES,
        "default_language": DEFAULT_LANGUAGE,
        "builtin_speakers": speakers,
        "speakers_source": speakers_source,
    }
    if torch.cuda.is_available():
        status_info["gpu_name"] = torch.cuda.get_device_name(0)
        status_info["gpu_memory_allocated"] = torch.cuda.memory_allocated()
        status_info["gpu_memory_total"] = torch.cuda.get_device_properties(0).total_memory
    return status_info


@app.get("/models")
async def list_models():
    """List available Qwen3-TTS model variants."""
    return {
        "models": AVAILABLE_MODELS,
        "current_model": current_model_name,
    }


class LoadModelRequest(BaseModel):
    """Request body for switching the active Qwen3-TTS model."""

    model: str


@app.post("/load_model")
async def switch_model(request: LoadModelRequest):
    """Switch to a different Qwen3-TTS model variant. Downloads if not cached."""
    global _desired_model_name
    model_name = request.model

    if model_name not in AVAILABLE_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model: {model_name}. Available: {list(AVAILABLE_MODELS.keys())}"
        )

    if model_name == current_model_name:
        return {"status": "ok", "message": "Model already loaded", "model": model_name}

    # A switch can mean a multi-GB download, so it runs off the event loop.
    # _LOAD_LOCK excludes _acquire_model(), which is what every handler uses to
    # obtain the model and its name together. Holding it first and then waiting
    # for the requests that already hold the old model to finish (rather than
    # the reverse) cannot deadlock: none of them needs the lock again.
    try:
        async with _LOAD_LOCK:
            if model_name == current_model_name:
                return {"status": "ok", "message": "Model already loaded", "model": model_name}
            await _drain_requests(_SWITCH_DRAIN_TIMEOUT)
            await asyncio.to_thread(load_model, model_name)
            _desired_model_name = model_name
            _touch_model()
        return {
            "status": "ok",
            "message": f"Model switched to {model_name}",
            "model": model_name,
            "model_info": AVAILABLE_MODELS[model_name],
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")


@app.get("/speakers")
async def list_speakers():
    """List the loaded model's built-in speakers.

    A Base or VoiceDesign model has none, so the list is empty and `/tts`
    answers 409 until a CustomVoice model is loaded. `speakers_source` says
    whether the list came from the loaded model, from the registry, or is the
    documented fallback shown while no model is resident.
    """
    speakers, source = _speaker_catalog()
    return {
        "speakers": speakers,
        "speakers_source": source,
        "languages": SUPPORTED_LANGUAGES,
    }


def _cleanup_temp(path):
    """Remove a temp file, never raising.

    Called from `finally` blocks: an OSError escaping here replaced an already
    successful audio response with a 500 and skipped the second temp file.
    """
    if not path:
        return
    try:
        os.unlink(path)
    except OSError:
        pass


class _TooLarge(Exception):
    pass


def _upload_suffix(filename: Optional[str]) -> str:
    """A safe temp-file suffix from a client-supplied name (decoders sniff it)."""
    suffix = re.sub(r"[^A-Za-z0-9.]", "", Path(filename).suffix) if filename else ""
    return suffix[:12] or ".wav"


async def _spool_upload(upload: UploadFile, limit: Optional[int] = None) -> str:
    """Copy *upload* to a temp file in chunks, off the event loop; return its path.

    413 (leaving no file behind) past *limit* (MAX_UPLOAD_MB), 400 for an empty
    upload. The old `await file.read()` pulled the whole upload into memory
    before looking at its size.
    """
    limit = MAX_UPLOAD_BYTES if limit is None else limit
    detail = f"Reference audio is larger than {limit / _MIB:g} MB (MAX_UPLOAD_MB)."
    if upload.size is not None and upload.size > limit:
        raise HTTPException(status_code=413, detail=detail)

    fd, path = tempfile.mkstemp(suffix=_upload_suffix(upload.filename))
    os.close(fd)

    def _copy() -> int:
        total = 0
        with open(path, "wb") as out:
            while True:
                chunk = upload.file.read(_UPLOAD_CHUNK)
                if not chunk:
                    return total
                total += len(chunk)
                if total > limit:
                    raise _TooLarge()
                out.write(chunk)

    try:
        total = await asyncio.to_thread(_copy)
    except _TooLarge:
        _cleanup_temp(path)
        raise HTTPException(status_code=413, detail=detail)
    except BaseException:
        _cleanup_temp(path)
        raise
    if total == 0:
        _cleanup_temp(path)
        raise HTTPException(status_code=400, detail="Reference audio is empty")
    return path


class _AsrUnavailable(Exception):
    """The ASR service could not produce a transcript (down, slow, or an error)."""


async def _auto_transcribe(audio_path: str, filename: str = "audio.wav") -> dict:
    """Auto-transcribe an audio file using the Qwen3-ASR service.

    Returns the full response dict (text, segments, duration). Raises
    ``_AsrUnavailable`` when the service could not answer, so a caller that
    needs a transcript can say so instead of mistaking "ASR is down" for
    "there is no speech in this recording".
    """
    try:
        logger.info(f"Auto-transcribing reference audio via Qwen3-ASR: {filename}")
        # Fail fast on an unreachable service; only the transcription itself is slow.
        async with httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=5.0)) as client:
            with open(audio_path, "rb") as f:
                response = await client.post(
                    f"{QWEN3_ASR_URL}/transcribe",
                    files={"audio": (filename, f)},
                )
    except Exception as e:
        logger.warning(f"Auto-transcription error: {e}")
        raise _AsrUnavailable(f"{type(e).__name__}: {e}")
    if response.status_code != 200:
        logger.warning(f"Auto-transcription failed: HTTP {response.status_code}")
        raise _AsrUnavailable(f"HTTP {response.status_code}")
    data = response.json()
    logger.info(f"Auto-transcription result: '{data.get('text', '').strip()[:100]}...'")
    return data


def _trim_audio_segment(input_path: str, start: float, end: float, output_path: str) -> bool:
    """Use ffmpeg to extract a segment from an audio file."""
    try:
        duration = end - start
        cmd = [
            # -ss before -i: input seeking, which jumps to the position instead
            # of decoding the whole file up to it and discarding the result.
            # Accurate as well as fast on ffmpeg >= 2.1.
            "ffmpeg", "-y",
            "-ss", f"{start:.3f}", "-i", input_path,
            "-t", f"{duration:.3f}",
            "-ar", "24000", "-ac", "1",
            output_path,
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=30)
        return result.returncode == 0 and os.path.exists(output_path)
    except Exception as e:
        logger.warning(f"ffmpeg trim error: {e}")
        return False


def _pick_best_segment(segments: list, min_dur: float = 3.0, max_dur: float = 10.0) -> dict:
    """Pick the best reference segment from ASR timestamps.
    Prefers segments between min_dur and max_dur seconds."""
    if not segments:
        return {}
    # Try to find a single segment in the ideal range
    for seg in segments:
        dur = seg.get("end", 0) - seg.get("start", 0)
        if min_dur <= dur <= max_dur:
            return seg
    # Otherwise merge consecutive segments to reach min_dur
    merged_start = segments[0].get("start", 0)
    merged_end = segments[0].get("end", 0)
    merged_text = segments[0].get("text", "")
    for seg in segments[1:]:
        merged_end = seg.get("end", 0)
        merged_text += " " + seg.get("text", "")
        if (merged_end - merged_start) >= min_dur:
            break
    return {"start": merged_start, "end": min(merged_end, merged_start + max_dur), "text": merged_text.strip()}


# --- Voice Library: persistent speaker prompt cache ---

VOICES_DIR = Path(os.getenv("VOICES_DIR", "/app/voices"))
# Deliberately NOT created here. Importing a module must not require write access
# to the filesystem: this line raised PermissionError on /app for any caller
# without root, which broke the offline unit tests and would break a read-only
# rootfs. Nothing needs it eagerly — _save_voice_prompt() creates it with
# parents=True on write, and _list_voices() returns [] when it is absent.

_MAX_VOICE_ID_LEN = 128
# fullmatch, not match with a `$` anchor: `$` also matches before a trailing
# newline, so "abc%0A" passed as a voice id.
_SAFE_VOICE_RE = re.compile(r"[a-zA-Z0-9_\-]+")
_UNSAFE_VOICE_CHARS_RE = re.compile(r"[^a-zA-Z0-9_\-]+")


def _voice_id_from_name(name: str) -> str:
    """Derive a filesystem-safe voice id from a user-supplied display name."""
    voice_id = _UNSAFE_VOICE_CHARS_RE.sub("_", (name or "").strip().lower()).strip("_")
    if not voice_id:
        raise HTTPException(
            status_code=400,
            detail="Voice name must contain at least one letter, digit, dash, or underscore",
        )
    if len(voice_id) > _MAX_VOICE_ID_LEN:
        raise HTTPException(
            status_code=400,
            detail=f"Voice name is too long (at most {_MAX_VOICE_ID_LEN} characters)",
        )
    return voice_id


def _voice_dir(voice_id: str) -> Path:
    """Return the directory for a voice, validating the id to prevent path traversal."""
    if len(voice_id) > _MAX_VOICE_ID_LEN or not _SAFE_VOICE_RE.fullmatch(voice_id):
        raise HTTPException(status_code=400, detail="Invalid voice id")
    return VOICES_DIR / voice_id


def _load_voice_metadata(voice_id: str) -> dict:
    """Read the metadata.json for a saved voice, or return ``{}``."""
    meta_path = _voice_dir(voice_id) / "metadata.json"
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text())
    except (OSError, ValueError) as e:
        logger.warning("Unreadable metadata for voice '%s': %s", voice_id, e)
        return {}


def _save_voice_prompt(voice_id: str, prompt_item, metadata: dict):
    """Save a VoiceClonePromptItem to disk as tensors + metadata."""
    vdir = _voice_dir(voice_id)
    vdir.mkdir(parents=True, exist_ok=True)
    # Save tensors
    torch.save(prompt_item.ref_spk_embedding, vdir / "ref_spk_embedding.pt")
    code_path = vdir / "ref_code.pt"
    if prompt_item.ref_code is not None:
        torch.save(prompt_item.ref_code, code_path)
    else:
        # Re-saving an in-context voice as x-vector only must not leave its
        # codes behind for a loader that goes by the files it finds.
        _cleanup_temp(code_path)
    # Save metadata
    metadata.update({
        "x_vector_only_mode": prompt_item.x_vector_only_mode,
        "icl_mode": prompt_item.icl_mode,
    })
    # The item's own text is None in x-vector mode. Writing it unconditionally
    # replaced the transcript the caller passed in with null, so every saved voice
    # listed `ref_text: null` and the UI's "Ref:" line showed nothing useful.
    if prompt_item.ref_text:
        metadata["ref_text"] = prompt_item.ref_text
    (vdir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2))


def _load_voice_prompt(voice_id: str):
    """Load a cached VoiceClonePromptItem from disk, or None if there is no such voice.

    Voices saved before in-context mode existed (no ``icl_mode`` in their
    metadata) come back as x-vector only, exactly as they always did.
    """
    from qwen_tts.inference.qwen3_tts_model import VoiceClonePromptItem

    vdir = _voice_dir(voice_id)
    spk_path = vdir / "ref_spk_embedding.pt"
    if not spk_path.exists():
        return None
    ref_spk = torch.load(spk_path, map_location=device, weights_only=True)
    meta = _load_voice_metadata(voice_id)
    ref_text = (meta.get("ref_text") or "").strip()
    code_path = vdir / "ref_code.pt"
    if meta.get("icl_mode") is True and ref_text and code_path.exists():
        return VoiceClonePromptItem(
            ref_code=torch.load(code_path, map_location=device, weights_only=True),
            ref_spk_embedding=ref_spk,
            x_vector_only_mode=False,
            icl_mode=True,
            ref_text=ref_text,
        )
    return VoiceClonePromptItem(
        ref_code=None,
        ref_spk_embedding=ref_spk,
        x_vector_only_mode=True,
        icl_mode=False,
        ref_text=None,
    )


def _list_voices() -> list:
    """List all saved voice profiles with their metadata."""
    voices = []
    if not VOICES_DIR.exists():
        return voices
    for vdir in sorted(VOICES_DIR.iterdir()):
        if vdir.is_dir() and (vdir / "metadata.json").exists():
            try:
                meta = json.loads((vdir / "metadata.json").read_text())
            except (OSError, ValueError) as e:
                # One corrupt profile must not take the whole listing down.
                logger.warning("Skipping voice '%s' with unreadable metadata: %s", vdir.name, e)
                continue
            meta["id"] = vdir.name
            voices.append(meta)
    return voices


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences for chunked generation.

    Keeps sentences together if they're very short (<20 chars) to avoid
    generating tiny audio fragments with bad prosody.
    """
    # Split on sentence-ending punctuation followed by whitespace
    raw = re.split(r'(?<=[.!?;:])\s+', text.strip())
    if not raw:
        return [text]

    # Merge very short fragments with the next sentence
    merged = []
    buf = ""
    for s in raw:
        if buf:
            buf += " " + s
        else:
            buf = s
        if len(buf) >= 20:
            merged.append(buf)
            buf = ""
    if buf:
        if merged:
            merged[-1] += " " + buf
        else:
            merged.append(buf)
    return merged


def _generate_chunks(model, sentences: list[str], language: str, voice_clone_prompt, sample_rate: int = 24000) -> tuple[np.ndarray, int]:
    """Generate audio for sentences in bounded batches, then concatenate with gaps.

    Batch mode is 2x+ faster than sequential, but the batch used to be *every*
    sentence in the request — so peak VRAM scaled with the length of the text a
    caller happened to send, on a card the whole stack shares. A paragraph was
    fine; a chapter was an OOM with no knob to turn. TTS_MAX_BATCH bounds it,
    mirroring ASR_MAX_BATCH on the NeMo services.
    """
    prompt_item = voice_clone_prompt[0] if voice_clone_prompt else None

    all_audio = []
    sr = None
    start = time.time()
    for offset in range(0, len(sentences), TTS_MAX_BATCH):
        batch = sentences[offset:offset + TTS_MAX_BATCH]
        logger.info(
            f"  Generating sentences {offset + 1}-{offset + len(batch)} of "
            f"{len(sentences)} (batch of {len(batch)})..."
        )
        wavs, batch_sr = model.generate_voice_clone(
            text=batch,
            language=[language] * len(batch),
            voice_clone_prompt=[prompt_item] * len(batch),
        )
        sr = batch_sr or sr
        for wav in wavs:
            all_audio.append(np.array(wav))
        del wavs

    logger.info(f"  Generation done in {time.time() - start:.2f}s")

    sr = sr or sample_rate
    # Interleave gaps once, across the whole sequence. Sizing the gap from the
    # rate the model actually returned matters — at any other rate the 24000
    # default produced audibly wrong pauses.
    gap = np.zeros(int(sr * 0.15), dtype=np.float32)  # 150ms
    spaced = []
    for i, chunk in enumerate(all_audio):
        spaced.append(chunk)
        if i < len(all_audio) - 1:
            spaced.append(gap)

    return np.concatenate(spaced), sr


async def _synthesize_with_prompt(model, text: str, language: str, prompt_item) -> tuple:
    """Speak *text* in the voice of *prompt_item*; long texts are chunked."""
    sentences = _split_sentences(text)
    logger.info(f"Voice synthesis: {len(sentences)} chunk(s)")
    if len(sentences) > 1:
        return await _gen(_generate_chunks, model, sentences, language, [prompt_item])
    wavs, sr = await _gen(
        model.generate_voice_clone,
        text=text,
        language=language,
        voice_clone_prompt=[prompt_item],
    )
    return np.array(wavs[0]), sr


def _encode_wav(audio, sr) -> bytes:
    buffer = io.BytesIO()
    sf.write(buffer, audio, sr, format="WAV")
    return buffer.getvalue()


async def _wav_response(audio, sr, headers: dict) -> Response:
    """A finished WAV as one response body.

    Everything is already generated when this runs, so nothing streams from the
    model. `StreamingResponse(BytesIO)` iterated the buffer line by line (split on
    0x0A bytes inside binary audio) and could not send a Content-Length.
    """
    body = await asyncio.to_thread(_encode_wav, np.asarray(audio), sr)
    return Response(content=body, media_type="audio/wav", headers=headers)


@app.get("/voices")
async def list_saved_voices():
    """List all saved voice profiles."""
    return {"voices": await asyncio.to_thread(_list_voices)}


def _clone_mode(mode: Optional[str]) -> str:
    """Normalise /voices/save's `mode`: ``xvector`` (default) or ``icl``."""
    key = (mode or "xvector").strip().lower().replace("-", "_")
    if key in ("xvector", "x_vector", "x_vector_only", "embedding"):
        return "xvector"
    if key in ("icl", "in_context"):
        return "icl"
    raise HTTPException(status_code=400, detail="mode must be 'xvector' or 'icl'")


@app.post("/voices/save")
async def save_voice(
    name: str = Form(...),
    lang: str = Form("auto"),
    ref_text: str = Form(""),
    mode: str = Form("xvector"),
    file: UploadFile = File(...),
):
    """Upload reference audio, extract speaker prompt, and save for reuse.

    `mode=xvector` (default) stores the speaker embedding only. `mode=icl` also
    stores the reference codes and transcript, which conditions generation on the
    actual reference clip and usually clones more faithfully, at the cost of a
    longer prompt per request. `ref_text` supplies the transcript; without it the
    Qwen3-ASR service is asked for one and to find the best 3-10 s segment to trim
    to. In x-vector mode that service is optional: if it is unreachable the voice
    is saved from the untrimmed upload.
    """
    # Validate up front so a bad request fails fast, before the expensive
    # transcription + embedding work.
    voice_id = _voice_id_from_name(name)
    mode = _clone_mode(mode)
    given_text = _bound_optional_text(ref_text, "Reference text").strip()

    tmp_path = None
    trimmed_path = None
    try:
        tmp_path = await _spool_upload(file)
        with _as_server_error("Save voice"):
            # Held for the whole request, including the wait on the ASR service:
            # the reaper must not unload the model in that gap.
            async with _acquire_model() as (model, model_name):
                _require_capability(model_name, "voice_clone", _BASE_MODELS)
                start_time = time.time()

                audio_path = tmp_path
                segment_text = given_text
                text_source = "provided" if given_text else "none"
                if not given_text:
                    asr_result = None
                    try:
                        asr_result = await _auto_transcribe(tmp_path, file.filename or "reference.wav")
                    except _AsrUnavailable as e:
                        if mode == "icl":
                            raise HTTPException(
                                status_code=502,
                                detail=(
                                    f"In-context cloning needs a transcript and the Qwen3-ASR service "
                                    f"failed ({e}). Pass ref_text to skip transcription."
                                ),
                            )
                        logger.warning(f"Saving voice without a transcript or trim: {e}")

                    if asr_result is not None:
                        ref_text_asr = asr_result.get("text", "").strip()
                        if not ref_text_asr:
                            raise HTTPException(
                                status_code=400,
                                detail="Could not transcribe reference audio. Please try a clearer recording.",
                            )
                        segment_text = ref_text_asr
                        text_source = "asr"

                        # Trim to best segment using ffmpeg
                        best = _pick_best_segment(asr_result.get("segments", []))
                        if best and best.get("start") is not None:
                            trimmed_path = tmp_path + "_trimmed.wav"
                            # Off the event loop: subprocess.run here blocks every other request
                            # for as long as ffmpeg takes, up to its own 30 s timeout.
                            trimmed = await asyncio.to_thread(
                                _trim_audio_segment, tmp_path, best["start"], best["end"], trimmed_path)
                            if trimmed:
                                audio_path = trimmed_path
                                segment_text = best.get("text") or ref_text_asr
                                dur = best["end"] - best["start"]
                                logger.info(f"Trimmed reference to {dur:.1f}s segment: '{segment_text[:60]}...'")

                # Extract the speaker prompt; off the event loop
                prompt_kwargs = {"ref_audio": audio_path, "x_vector_only_mode": mode == "xvector"}
                if mode == "icl":
                    prompt_kwargs["ref_text"] = segment_text
                prompt_items = await _gen(model.create_voice_clone_prompt, **prompt_kwargs)
                prompt_item = prompt_items[0]

                # Save to disk
                await asyncio.to_thread(_save_voice_prompt, voice_id, prompt_item, {
                    "name": name,
                    "lang": lang,
                    "original_filename": file.filename,
                    "ref_text": segment_text,
                    "ref_text_source": text_source,
                    "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    # The model this request actually used, not the global: a
                    # concurrent reload or unload blanked it here before.
                    "model_used": model_name,
                })

        elapsed = time.time() - start_time
        logger.info(f"Voice '{name}' saved as '{voice_id}' in {elapsed:.1f}s")

        return {
            "status": "ok",
            "voice_id": voice_id,
            "name": name,
            "ref_text": segment_text,
            "mode": mode,
            "ref_text_source": text_source,
            "processing_time": round(elapsed, 2),
        }

    finally:
        _cleanup_temp(tmp_path)
        _cleanup_temp(trimmed_path)


@app.delete("/voices/{voice_id}")
async def delete_voice(voice_id: str):
    """Delete a saved voice profile."""
    vdir = _voice_dir(voice_id)
    if not vdir.exists():
        raise HTTPException(status_code=404, detail=f"Voice '{voice_id}' not found")
    await asyncio.to_thread(shutil.rmtree, vdir)
    return {"status": "ok", "deleted": voice_id}


@app.post("/voices/{voice_id}/tts")
async def tts_with_saved_voice(
    voice_id: str,
    text: str = Form(...),
    lang: str = Form(""),
):
    """Generate speech using a saved voice profile. Skips audio processing — fast."""
    _require_text(text)
    language = _resolve_language(lang)

    prompt_item = await asyncio.to_thread(_load_voice_prompt, voice_id)
    if prompt_item is None:
        raise HTTPException(status_code=404, detail=f"Voice '{voice_id}' not found")

    with _as_server_error("Saved-voice TTS"):
        # Load the model before the capability check so a cold start
        # (failed preload) doesn't report a misleading capability error.
        async with _acquire_model() as (model, model_name):
            _require_capability(model_name, "voice_clone", _BASE_MODELS)
            start_time = time.time()

            audio, sr = await _synthesize_with_prompt(model, text, language, prompt_item)

        generation_time = time.time() - start_time
        audio_duration = len(audio) / sr
        logger.info(f"Saved-voice TTS done in {generation_time:.2f}s ({audio_duration:.1f}s audio, voice={voice_id})")

        # No empty_cache()/gc.collect() on the response path: they force a device
        # synchronise plus a full gen-2 GC, and discarding the caching allocator
        # makes the *next* generation re-pay cudaMalloc.
        return await _wav_response(audio, sr, {
            "X-Generation-Time": f"{generation_time:.3f}",
            "X-Audio-Duration": f"{audio_duration:.3f}",
            "X-Voice-ID": voice_id,
            "X-Language": language,
        })


class TTSRequest(BaseModel):
    """Request body for built-in-speaker speech synthesis."""

    text: str
    lang: str = ""
    speaker: str = "Vivian"
    instruct: str = ""


@app.post("/tts")
async def text_to_speech(request: TTSRequest):
    """Generate speech using a built-in speaker voice.

    Needs a CustomVoice model: Base and VoiceDesign models have no preset
    speakers, so with one of those loaded this answers 409 naming the fix.
    """
    _require_text(request.text)
    language = _resolve_language(request.lang)
    instruct = _bound_optional_text(request.instruct, "Instruction")

    with _as_server_error("TTS"):
        async with _acquire_model() as (model, model_name):
            # Route on declared capabilities: Base models expose a
            # generate_custom_voice attribute but raise when it is called,
            # so hasattr() alone picks the wrong path. There is no reference-free
            # fallback: create/generate_voice_clone need a reference clip, so the
            # old "try cloning without one" branch could only ever fail.
            _require_capability(
                model_name, "custom_voice", _CUSTOM_VOICE_MODELS,
                status_code=409, what="built-in speakers (preset voices)",
            )
            speaker = _validate_speaker(model, request.speaker)
            start_time = time.time()

            wavs, sr = await _gen(
                model.generate_custom_voice,
                text=request.text,
                language=language,
                speaker=speaker,
                instruct=instruct,
            )

        generation_time = time.time() - start_time
        logger.info(f"TTS generated in {generation_time:.2f}s")

        return await _wav_response(wavs[0], sr, {
            "X-Generation-Time": f"{generation_time:.3f}",
            "X-Speaker": _safe_header(speaker),
            "X-Language": language,
        })


@app.post("/clone")
async def clone_voice(
    text: str = Form(...),
    lang: str = Form(""),
    file: UploadFile = File(...),
):
    """
    Clone a voice from a reference audio file (3+ seconds recommended).
    Provide reference audio as 'file' and the text to synthesize as 'text'.
    """
    _require_text(text)
    language = _resolve_language(lang)

    tmp_path = None
    try:
        tmp_path = await _spool_upload(file)
        with _as_server_error("Voice clone"):
            async with _acquire_model() as (model, model_name):
                _require_capability(model_name, "voice_clone", _BASE_MODELS)
                start_time = time.time()

                logger.info(f"Voice clone request: text='{text[:50]}...', lang={language}, ref={file.filename}")

                # NOTE: this path deliberately does *not* auto-transcribe the reference.
                # It clones via `x_vector_only_mode=True` (speaker embedding only), which
                # never consumes a reference transcript — the previous full Qwen3-ASR-1.7B
                # round trip produced a string that was then never read.

                # Extract speaker embedding first (fast), then use it for chunked generation
                prompt_items = await _gen(
                    model.create_voice_clone_prompt,
                    ref_audio=tmp_path,
                    x_vector_only_mode=True,
                )
                prompt_item = prompt_items[0] if isinstance(prompt_items, list) else prompt_items

                audio, sr = await _synthesize_with_prompt(model, text, language, prompt_item)

            generation_time = time.time() - start_time
            audio_duration = len(audio) / sr
            logger.info(f"Voice clone done in {generation_time:.2f}s ({audio_duration:.1f}s audio)")

            return await _wav_response(audio, sr, {
                "X-Generation-Time": f"{generation_time:.3f}",
                "X-Audio-Duration": f"{audio_duration:.3f}",
                # Header values must be latin-1 encodable; an umlaut in the
                # reference filename used to turn a finished clone into a 500.
                "X-Clone-Source": _safe_header(file.filename or "unknown"),
                "X-Language": language,
            })
    finally:
        _cleanup_temp(tmp_path)


@app.post("/clone-with-ref-text")
async def clone_voice_with_ref_text(
    text: str = Form(...),
    ref_text: str = Form(...),
    lang: str = Form(""),
    file: UploadFile = File(...),
):
    """
    High-quality voice cloning with reference text.
    Provide reference audio + its transcript for best results.
    """
    _require_text(text)
    _require_text(ref_text, "Reference text")
    language = _resolve_language(lang)

    tmp_path = None
    try:
        tmp_path = await _spool_upload(file)
        with _as_server_error("High-quality clone"):
            async with _acquire_model() as (model, model_name):
                # The sibling /clone endpoint checks this; this one did not, so calling
                # it on a CustomVoice or VoiceDesign model produced a raw 500.
                _require_capability(model_name, "voice_clone", _BASE_MODELS)
                start_time = time.time()

                # In-context prompt built once and reused, so a long text is chunked
                # like /clone's instead of going through the model as one sequence.
                prompt_items = await _gen(
                    model.create_voice_clone_prompt,
                    ref_audio=tmp_path,
                    ref_text=ref_text,
                    x_vector_only_mode=False,
                )
                prompt_item = prompt_items[0] if isinstance(prompt_items, list) else prompt_items

                audio, sr = await _synthesize_with_prompt(model, text, language, prompt_item)

            generation_time = time.time() - start_time
            logger.info(f"High-quality voice clone generated in {generation_time:.2f}s")

            return await _wav_response(audio, sr, {
                "X-Generation-Time": f"{generation_time:.3f}",
                "X-Language": language,
            })
    finally:
        _cleanup_temp(tmp_path)


class VoiceDesignRequest(BaseModel):
    """Request body for text-guided voice design synthesis."""

    text: str
    voice_description: str
    lang: str = ""


@app.post("/voice_design")
async def voice_design(request: VoiceDesignRequest):
    """
    Generate speech with a voice designed from a text description.
    Requires the VoiceDesign model to be loaded.
    Example description: "A deep male voice with a warm, calm British accent"
    """
    _require_text(request.text)
    _require_text(request.voice_description, "Voice description")
    language = _resolve_language(request.lang)

    with _as_server_error("Voice design"):
        async with _acquire_model() as (model, model_name):
            _require_capability(model_name, "voice_design", "the VoiceDesign model")
            start_time = time.time()

            wavs, sr = await _gen(
                model.generate_voice_design,
                text=request.text,
                language=language,
                instruct=request.voice_description,
            )

        generation_time = time.time() - start_time
        logger.info(f"Voice design generated in {generation_time:.2f}s")

        return await _wav_response(wavs[0], sr, {
            "X-Generation-Time": f"{generation_time:.3f}",
            "X-Language": language,
            "X-Voice-Description": _safe_header(request.voice_description[:100]),
        })


if __name__ == "__main__":
    # 120 s keep-alive: the gateway pools connections, and uvicorn's 5 s default
    # closes them first, so pooled sockets came back dead.
    uvicorn.run(app, host="0.0.0.0", port=5004, timeout_keep_alive=120)
