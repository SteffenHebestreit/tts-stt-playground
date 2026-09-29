"""Chatterbox-TTS text-to-speech and voice-cloning service.

Wraps Resemble AI's Chatterbox Multilingual (MIT-licensed) — zero-shot TTS
with voice cloning across 23 languages including German. Output audio carries
Resemble's PerTh watermark by design (self-hosted too; v3 does not change that).

Endpoints follow the project's `simple-json-tts-v1` and `voice-clone-tts-v1`
contracts so the frontend gateway can proxy them like the other providers.

Three properties of the library shape most of this file:

- The model object holds the *voice* (`model.conds`). `generate(audio_prompt_path=)`
  overwrites it, so every generation here starts from a private copy of the
  built-in voice and a clone never outlives its request.
- One `generate()` call stops at 1000 speech tokens (~40 s) without an error, so
  text is split into sentence-sized chunks and the audio is joined afterwards.
- `generate()` runs in a worker thread that cannot be cancelled. Whatever the
  thread uses (the model, the generation slot, temp files) is therefore released
  when the THREAD finishes, never when the awaiting coroutine goes away.
"""

import os
import io
import re
import copy
import math
import time
import uuid
import struct
import asyncio
import inspect
import tempfile
import logging
import threading
import functools
from contextlib import asynccontextmanager, suppress
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import torch
import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.background import BackgroundTask
from typing import Optional
from pydantic import BaseModel
import uvicorn

from body_limit import BodyLimitMiddleware
from model_lifecycle import ModelSlot, ttl_from_env
from origin_guard import OriginGuardMiddleware, parse_allowed_origins


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
    if not math.isfinite(value) or (minimum is not None and value < minimum):
        logger.warning("Ignoring %s=%r (not usable); using %s", name, raw, default)
        return default
    return value


def _failure(status_code: int, public: str, detail: object = None, *, exc_info: bool = False) -> HTTPException:
    """An error for the client that says *public* and a request id, nothing else.

    The exception text (library messages carry paths, URLs and shapes) is written to
    the log under the same id, so an operator can find it and a caller cannot read
    the container's internals out of an unauthenticated response. Call with
    ``exc_info=True`` from an ``except`` block.
    """
    request_id = uuid.uuid4().hex[:12]
    logger.error("[request %s] %s: %s", request_id, public, detail, exc_info=exc_info)
    return HTTPException(
        status_code=status_code,
        detail=f"{public} (request id {request_id}).",
        headers={"X-Request-ID": request_id},
    )


class _ExplainedError(RuntimeError):
    """A failure whose message was written for the caller: it names the fix and nothing internal."""


def _server_error(exc: Exception, public: str) -> HTTPException:
    """The 500 for an unexpected exception: generic, unless the message was written for the caller."""
    if isinstance(exc, _ExplainedError):
        logger.warning("%s: %s", public, exc)
        return HTTPException(status_code=500, detail=str(exc))
    return _failure(500, public, exc, exc_info=True)


# --- Configuration -----------------------------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"

# Chatterbox Multilingual language ids, by the names and codes a caller or an operator
# is likely to write. The ids are what generate(language_id=) accepts.
_LANGUAGE_NAMES = {
    "english": "en", "german": "de", "french": "fr", "spanish": "es",
    "italian": "it", "portuguese": "pt", "russian": "ru", "japanese": "ja",
    "korean": "ko", "chinese": "zh", "dutch": "nl", "polish": "pl",
    "turkish": "tr", "swedish": "sv", "danish": "da", "norwegian": "no",
    "finnish": "fi", "greek": "el", "hebrew": "he", "hindi": "hi",
    "arabic": "ar", "malay": "ms", "swahili": "sw",
}
_LANGUAGE_IDS = frozenset(_LANGUAGE_NAMES.values())


def _configured_default_language(raw) -> str:
    """CHATTERBOX_DEFAULT_LANGUAGE as a language id the library accepts; German otherwise.

    Case, surrounding blanks, region tags ("de-DE", "pt_BR") and language names
    ("German") are all fine. What used to be returned as written ("", "de-DE",
    "German", " de ") was handed to every request that asked for "auto", and the
    library rejects all of those. Anything unknown falls back to "de" with a warning.
    """
    text = (raw or "").strip()
    if not text:
        return "de"
    key = text.lower().replace("-", "_")
    language = _LANGUAGE_NAMES.get(key) or _LANGUAGE_NAMES.get(key.split("_")[0])
    if language is None:
        primary = key.split("_")[0]
        language = primary if primary in _LANGUAGE_IDS else None
    if language is None:
        logger.warning(
            "Ignoring unsupported CHATTERBOX_DEFAULT_LANGUAGE=%r; using de. Supported: %s",
            raw, ", ".join(sorted(_LANGUAGE_IDS)),
        )
        return "de"
    return language


DEFAULT_LANGUAGE = _configured_default_language(os.getenv("CHATTERBOX_DEFAULT_LANGUAGE"))

# Which T3 (text-to-speech-token) checkpoint to load. "v3" needs a chatterbox
# build that accepts `t3_model` (GitHub master, not PyPI 0.1.7); on an older
# package the load falls back to the default checkpoint with a warning instead of
# refusing to start. "v2" or "default" select the library default explicitly.
T3_MODEL = (os.getenv("CHATTERBOX_T3_MODEL") or "v3").strip()

# The library pins `revision="main"` and offers no way to change it; a non-default
# value is applied by wrapping its snapshot_download call (see _from_pretrained).
HF_REVISION = (os.getenv("CHATTERBOX_HF_REVISION") or "main").strip() or "main"

# Unset = use the library's own default, which is NOT the same everywhere: 2.0 in
# chatterbox-tts 0.1.7, 1.2 on GitHub master (lowered together with the v3 work).
REPETITION_PENALTY = _number(
    os.getenv("CHATTERBOX_REPETITION_PENALTY"), "CHATTERBOX_REPETITION_PENALTY", None, float, minimum=1.0
)

# Silence inserted between the separately generated chunks of one request.
CHUNK_GAP_MS = _number(os.getenv("CHATTERBOX_CHUNK_GAP_MS"), "CHATTERBOX_CHUNK_GAP_MS", 150, int, minimum=0)

# Input bounds. The gateway limits what it forwards, but a direct caller must not
# be able to ask this service for an hour of speech or upload a disk-filling file.
MAX_TEXT_CHARS = _number(os.getenv("MAX_TEXT_CHARS"), "MAX_TEXT_CHARS", 5000, int, minimum=1)
_MIB = 1024 * 1024
MAX_UPLOAD_BYTES = int(_number(os.getenv("MAX_UPLOAD_MB"), "MAX_UPLOAD_MB", 20.0, float, minimum=0.001) * _MIB)
_UPLOAD_CHUNK = _MIB
# multipart boundaries and part headers on top of the file bytes
_MULTIPART_SLACK = _MIB

# The library resamples the whole reference clip and embeds all of it, although
# only the first 6 s (T3) and 10 s (S3Gen) condition the voice. Decoding is capped
# so a long upload costs seconds, not minutes.
REF_MAX_SECONDS = _number(os.getenv("CHATTERBOX_REF_MAX_SECONDS"), "CHATTERBOX_REF_MAX_SECONDS", 30.0, float, minimum=1.0)
_REF_SAMPLE_RATE = 24000  # chatterbox S3GEN_SR

# Set by the Dockerfile so /status can say which chatterbox build is inside.
CHATTERBOX_REF = os.getenv("CHATTERBOX_REF", "")

tts_model = None
model_loaded = False
# The built-in voice as it was at load time, and what actually got loaded. Kept
# across a TTL unload (the label is informational); the conditionals are dropped
# with the weights.
_default_conds = None
_active_t3_model: Optional[str] = None

# One process-global model on one GPU: overlapping generations thrash VRAM and
# make every request slower than running them back to back. Released by the
# worker thread's completion (see _run_generation), not by request cancellation.
_GEN_CONCURRENCY = _number(os.getenv("TTS_MAX_CONCURRENCY"), "TTS_MAX_CONCURRENCY", 1, int, minimum=1)
_GEN_SEM = asyncio.Semaphore(_GEN_CONCURRENCY)

# Waiting for that permit used to be unbounded: every waiter holds its request (and,
# for /clone, a spooled upload) for as long as generations ahead of it take, so a burst
# of requests was a queue of unbounded length and unbounded wait. A request now waits
# at most TTS_QUEUE_TIMEOUT_S, and when TTS_MAX_QUEUE are already waiting the next
# one is turned away at once; both answer 503 with Retry-After.
QUEUE_TIMEOUT_S = _number(os.getenv("TTS_QUEUE_TIMEOUT_S"), "TTS_QUEUE_TIMEOUT_S", 60.0, float, minimum=0.1)
MAX_QUEUE = _number(os.getenv("TTS_MAX_QUEUE"), "TTS_MAX_QUEUE", 4 * _GEN_CONCURRENCY, int, minimum=0)
_waiting = 0  # requests inside _acquire_generation_slot's wait (the event loop is the only writer)

# `model.conds` and T3's per-call backend are instance state, so two generate()
# calls on one model must never overlap, whatever TTS_MAX_CONCURRENCY says. The
# semaphore normally makes this uncontended; it is the guarantee, not the queue.
_MODEL_LOCK = threading.Lock()

# Request values that are not already a language id ("auto" is the validated default).
LANGUAGE_ALIASES = {"auto": DEFAULT_LANGUAGE, **_LANGUAGE_NAMES}


# --- Model loading -----------------------------------------------------------

def _package_version() -> str:
    try:
        from importlib.metadata import version
        return version("chatterbox-tts")
    except Exception:
        return "unknown"


def _t3_load_args(from_pretrained, requested: str) -> tuple:
    """``(kwargs, label)`` for loading the requested T3 checkpoint.

    ``t3_model`` is passed only if the installed ``from_pretrained`` declares it.
    PyPI chatterbox-tts 0.1.7 predates v3 and would raise TypeError; falling back
    to the default checkpoint keeps the service up, and the WARNING plus the
    ``t3_model`` field of /health say what is really running.
    """
    wanted = (requested or "").strip()
    if wanted.lower() in ("", "default", "none"):
        return {}, "default"
    try:
        accepts = "t3_model" in inspect.signature(from_pretrained).parameters
    except (TypeError, ValueError):
        accepts = False
    if accepts:
        return {"t3_model": wanted}, wanted
    if wanted.lower() in ("v2", "t3_mtl23ls_v2"):
        return {}, "default"  # the library default is v2 anyway
    logger.warning(
        "CHATTERBOX_T3_MODEL=%s requested, but the installed chatterbox-tts %s%s does not accept "
        "t3_model (PyPI 0.1.7 predates v3). Falling back to the default checkpoint; install "
        "chatterbox from GitHub master (Dockerfile build arg CHATTERBOX_REF) to get v3.",
        wanted, _package_version(), f" (ref {CHATTERBOX_REF})" if CHATTERBOX_REF else "",
    )
    return {}, "default"


def _from_pretrained(cls, t3_kwargs: dict):
    """``cls.from_pretrained`` at ``HF_REVISION``.

    The library hard-codes ``revision="main"`` inside from_pretrained, so a pinned
    revision is injected by wrapping the ``snapshot_download`` name that function
    resolves in its own module. Loads run under the ModelSlot lock, so nothing
    else can observe the patch. An explicit pin the library will not honour is a
    configuration error, not something to ignore quietly.
    """
    if HF_REVISION == "main":
        return cls.from_pretrained(device=device, **t3_kwargs)

    import importlib
    module = importlib.import_module(cls.__module__)
    original = getattr(module, "snapshot_download", None)
    if original is None:
        raise RuntimeError(
            f"CHATTERBOX_HF_REVISION={HF_REVISION} cannot be applied: {cls.__module__} no longer "
            "calls snapshot_download. Unset it or pin the chatterbox build instead."
        )

    def pinned(*args, **kwargs):
        kwargs["revision"] = HF_REVISION
        return original(*args, **kwargs)

    module.snapshot_download = pinned
    try:
        return cls.from_pretrained(device=device, **t3_kwargs)
    finally:
        module.snapshot_download = original


def _load_chatterbox():
    """Construct the Chatterbox model (called by the lifecycle slot)."""
    global tts_model, model_loaded, _default_conds, _active_t3_model
    logger.info(
        "Loading Chatterbox Multilingual on %s (requested T3 checkpoint: %s, HF revision: %s)...",
        device, T3_MODEL, HF_REVISION,
    )
    try:
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS

        t3_kwargs, active = _t3_load_args(ChatterboxMultilingualTTS.from_pretrained, T3_MODEL)
        model = _from_pretrained(ChatterboxMultilingualTTS, t3_kwargs)

        # generate(audio_prompt_path=...) replaces model.conds, and the exaggeration
        # update rewrites it in place. Keep a pristine copy of the built-in voice
        # (None if the checkpoint ships none) that every generation starts from.
        conds = getattr(model, "conds", None)
        snapshot = copy.deepcopy(conds) if conds is not None else None

        tts_model = model
        _default_conds = snapshot
        _active_t3_model = active
        model_loaded = True
        logger.info(
            "Chatterbox Multilingual loaded on %s: T3 checkpoint %s, HF revision %s, chatterbox-tts %s%s, "
            "built-in voice %s",
            device, active, HF_REVISION, _package_version(),
            f" (ref {CHATTERBOX_REF})" if CHATTERBOX_REF else "",
            "present" if snapshot is not None else "absent (/tts needs a checkpoint with conds.pt)",
        )
        return model
    except Exception as e:
        logger.error(f"Failed to load Chatterbox model: {e}", exc_info=True)
        raise


def _forget_chatterbox(_model) -> None:
    """Clear the module-level aliases so nothing keeps the weights alive."""
    global tts_model, model_loaded, _default_conds
    tts_model = None
    _default_conds = None
    model_loaded = False


# Idle TTS models are the largest single VRAM saving available when several
# services share one card: on a 12 GB card the default stack sits at ~9.7 GB, so
# this service's ~4 GB is exactly what does not fit alongside it. Reference
# counted, so a generation in flight is never unloaded underneath itself.
#   >0 = seconds idle before unloading, 0 = unload immediately, -1 = never
MODEL_TTL = ttl_from_env(os.getenv, "TTS_MODEL_TTL", "MODEL_TTL", default=300.0)
_model_slot = ModelSlot(
    _load_chatterbox, ttl_seconds=MODEL_TTL, name="Chatterbox Multilingual",
    on_unload=_forget_chatterbox,
)


async def _preload_model() -> None:
    """Load the model in the background so /health answers while it downloads.

    A first start fetches ~3 GB from Hugging Face, which can outlast the
    healthcheck's start period; /ready (503 "loading") is what reports it. The
    lease is released straight away, which arms the idle timer — so with the
    default TTL an untouched service frees its VRAM ~5 minutes after boot rather
    than holding ~4 GB forever. Set TTS_MODEL_TTL=-1 to keep it resident.
    """
    try:
        lease = await _model_slot.acquire_lease()
    except asyncio.CancelledError:
        raise
    except Exception as e:
        logger.warning(f"Could not preload model: {e}")
        return
    await lease.release_async()


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    preload = asyncio.create_task(_preload_model())
    try:
        yield
    finally:
        preload.cancel()
        with suppress(asyncio.CancelledError):
            await preload
        await _model_slot.unload_async()


app = FastAPI(
    title="Chatterbox-TTS Service",
    description="Multilingual TTS and voice cloning using Resemble AI Chatterbox",
    lifespan=_lifespan,
)

_UPLOAD_PATHS = {"/clone", "/clone-with-ref-text"}


def _body_limit(path: str) -> int:
    """Largest request body (bytes) accepted at *path*."""
    # 12 bytes per character is the worst case, a JSON-escaped astral character.
    text = MAX_TEXT_CHARS * 12 * 2 + 64 * 1024
    if path in _UPLOAD_PATHS:
        return MAX_UPLOAD_BYTES + _MULTIPART_SLACK + text
    return text


# Unset or empty ALLOWED_ORIGINS means no CORS headers at all (it used to mean
# "*"); "*" only when it is written down (and logged); otherwise an explicit list.
allowed_origins = parse_allowed_origins(os.getenv("ALLOWED_ORIGINS", ""))
allow_credentials = os.getenv("ALLOW_CREDENTIALS", "false").strip().lower() in {"1", "true", "yes", "on"}
if "*" in allowed_origins and allow_credentials:
    allow_credentials = False

# FastAPI parses the whole body (multipart parts are spooled to disk, JSON is held
# in memory) before a handler runs, so a size check inside the handler comes after
# the cost was paid. body_limit.py checks the declared Content-Length AND counts
# the bytes that arrive, so a chunked upload is cut off at the limit too.
#
# Each add_middleware wraps what was added before it: the body limit is innermost,
# the origin guard (403 for a state-changing request with a foreign Origin header;
# no Origin, e.g. the gateway or curl, is not affected) comes next, and CORS is
# outermost so a 403/413 still carries the CORS headers a listed origin needs in
# order to read it. CORS is only added when origins are configured.
app.add_middleware(BodyLimitMiddleware, limit_for=_body_limit)
app.add_middleware(OriginGuardMiddleware, allowed_origins=allowed_origins)
if allowed_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=allow_credentials,
        allow_methods=["*"],
        allow_headers=["*"],
    )


def _supported_language_ids(model) -> list:
    """Return the language ids the loaded model supports (best effort)."""
    try:
        from chatterbox.mtl_tts import SUPPORTED_LANGUAGES
        return sorted(SUPPORTED_LANGUAGES)
    except Exception:
        supported = getattr(model, "supported_languages", None)
        return sorted(supported) if supported else []


def _resolve_language(language: Optional[str]) -> str:
    """Normalize a request language value to a Chatterbox language id."""
    lang = (language or "").strip().lower().replace("-", "_").split("_")[0]
    if not lang:
        return DEFAULT_LANGUAGE
    return LANGUAGE_ALIASES.get(lang, lang)


def _require_text(text: Optional[str], field: str = "Text") -> str:
    """400 for a missing/blank text field, 413 past MAX_TEXT_CHARS."""
    if not text or not text.strip():
        raise HTTPException(status_code=400, detail=f"{field} not provided")
    if len(text) > MAX_TEXT_CHARS:
        raise HTTPException(
            status_code=413,
            detail=f"{field} is {len(text)} characters; the limit is {MAX_TEXT_CHARS} (raise it with MAX_TEXT_CHARS).",
        )
    return text


def _check_tuning(exaggeration: Optional[float], cfg_weight: Optional[float]) -> None:
    """422 for NaN, infinity or negative values, which only produce garbage audio."""
    for name, value in (("exaggeration", exaggeration), ("cfg_weight", cfg_weight)):
        if value is not None and (not math.isfinite(value) or value < 0):
            raise HTTPException(status_code=422, detail=f"{name} must be a finite number >= 0")


def _safe_header(value: str) -> str:
    """Make a string safe for an HTTP header value.

    Header values must be latin-1 encodable; a non-latin-1 reference filename
    otherwise turned an already-completed generation into a 500.
    """
    return (value or "").encode("latin-1", "replace").decode("latin-1")


def _to_numpy(wav) -> np.ndarray:
    """Model output (tensor or array, any leading dims) as a 1-D float32 array."""
    audio = wav.squeeze().detach().cpu().numpy() if torch.is_tensor(wav) else np.asarray(wav).squeeze()
    return np.atleast_1d(audio).astype(np.float32, copy=False)


def _wav_response(wav, sr: int, extra_headers: Optional[dict] = None) -> StreamingResponse:
    """Convert a model output tensor/array to a streaming WAV response.

    Deliberately does *not* call ``torch.cuda.empty_cache()``/``gc.collect()``:
    that forced a device synchronise plus a full gen-2 GC on the response path
    and destroyed the caching allocator, so the next generation had to re-pay
    ``cudaMalloc``. Fragmentation is handled by PYTORCH_CUDA_ALLOC_CONF instead.
    """
    buffer = io.BytesIO()
    sf.write(buffer, _to_numpy(wav), sr, format="WAV")
    buffer.seek(0)

    headers = {"X-Sample-Rate": str(sr)}
    if extra_headers:
        headers.update({k: _safe_header(v) for k, v in extra_headers.items()})
    return StreamingResponse(buffer, media_type="audio/wav", headers=headers)


# --- Generation --------------------------------------------------------------

class _GenerationCancelled(Exception):
    """The request went away; the remaining chunks were not generated."""


def _fresh_conds(default):
    """A private copy of the built-in voice (None if the checkpoint has none).

    Never hand out the snapshot itself: generate() rewrites `conds.t3` in place
    when the requested exaggeration differs, which would change the voice for
    every later request.
    """
    return copy.deepcopy(default) if default is not None else None


def _generate_text(model, default_conds, chunks, language_id, *, prompt_path=None,
                   exaggeration=None, cfg_weight=None, gap_ms=None, cancel=None):
    """Blocking: synthesise *chunks* in order and join them. Runs in a worker thread.

    Returns ``(float32 array, sample rate)``. Voice handling, all under the model
    lock: a default-voice request sets a fresh copy of the built-in voice before
    every generate() call; a clone lets the FIRST call build its conditionals from
    the reference (the library's own prepare_conditionals) and reuses them for the
    remaining chunks. In both cases the built-in voice is restored afterwards, so
    a clone can never speak in a later /tts or /tts-stream.
    """
    if prompt_path is None and default_conds is None:
        raise _ExplainedError(
            "This checkpoint ships no built-in voice (no conds.pt); send a reference clip to /clone instead."
        )
    gap_ms = CHUNK_GAP_MS if gap_ms is None else gap_ms
    sr = model.sr
    gap = np.zeros(int(sr * gap_ms / 1000), dtype=np.float32)
    pieces = []
    with _MODEL_LOCK:
        try:
            for index, chunk in enumerate(chunks):
                if cancel is not None and cancel.is_set():
                    raise _GenerationCancelled()
                kwargs = {"language_id": language_id}
                if exaggeration is not None:
                    kwargs["exaggeration"] = exaggeration
                if cfg_weight is not None:
                    kwargs["cfg_weight"] = cfg_weight
                if REPETITION_PENALTY is not None:
                    kwargs["repetition_penalty"] = REPETITION_PENALTY
                if prompt_path is None:
                    model.conds = _fresh_conds(default_conds)
                elif index == 0:
                    kwargs["audio_prompt_path"] = prompt_path
                wav = model.generate(chunk, **kwargs)
                if pieces and gap.size:
                    pieces.append(gap)
                pieces.append(_to_numpy(wav))
        finally:
            model.conds = _fresh_conds(default_conds)
    return np.concatenate(pieces), sr


def _unlink(*paths) -> None:
    for path in paths:
        if path:
            with suppress(OSError):
                os.unlink(path)


def _generation_finished(lease, cleanup, fut) -> None:
    """Runs on the event loop when the worker thread is really done."""
    _GEN_SEM.release()
    lease.release_soon()
    _unlink(*cleanup)
    if not fut.cancelled():
        fut.exception()  # mark retrieved: the awaiting request may be long gone


def _busy(message: str) -> HTTPException:
    """503 telling the caller to come back: the generation queue is full or too slow."""
    return HTTPException(
        status_code=503, detail=message,
        headers={"Retry-After": str(max(1, int(QUEUE_TIMEOUT_S // 4)))},
    )


def _shed_if_saturated() -> None:
    """503 at once when a permit is not free and TTS_MAX_QUEUE requests already wait for one.

    Call this before a request costs anything (decoding its upload, taking a model
    reference): `_acquire_generation_slot` applies the same rule again, but by then
    the work that made the request expensive is done.
    """
    if _GEN_SEM.locked() and _waiting >= MAX_QUEUE:
        raise _busy(f"The generation queue is full ({MAX_QUEUE} waiting, TTS_MAX_QUEUE); retry shortly.")


async def _acquire_generation_slot(*, continuation: bool = False) -> None:
    """Take a generation permit: waits at most TTS_QUEUE_TIMEOUT_S, and never more than TTS_MAX_QUEUE deep.

    Nothing is held on any way out of the wait (timeout, cancellation, a full
    queue), so callers have nothing to give back unless this returns.
    *continuation* is a later chunk of a stream that has already started: it may
    join a full queue (a request that began is not cut off half way; the timeout
    still applies), where a new request is turned away.
    """
    global _waiting
    if not _GEN_SEM.locked():
        await _GEN_SEM.acquire()  # a free permit: takes it without waiting
        return
    if not continuation and _waiting >= MAX_QUEUE:
        raise _busy(f"The generation queue is full ({MAX_QUEUE} waiting, TTS_MAX_QUEUE); retry shortly.")
    _waiting += 1
    try:
        await asyncio.wait_for(_GEN_SEM.acquire(), timeout=QUEUE_TIMEOUT_S)
    except asyncio.TimeoutError:
        raise _busy(
            f"No generation slot became free within {QUEUE_TIMEOUT_S:g}s (TTS_QUEUE_TIMEOUT_S); retry shortly.")
    finally:
        _waiting -= 1


async def _run_generation(chunks, language_id, *, prompt_path=None, exaggeration=None,
                          cfg_weight=None, gap_ms=None, cleanup=(), continuation=False):
    """Generate *chunks* on the worker pool; returns ``(audio, sample_rate)``.

    Takes the generation slot and its OWN model reference, and gives both back
    from the worker's completion callback. Cancelling the caller (client
    disconnect, timeout) therefore cannot start a second generation next to a
    running one, and cannot let a TTL or /unload free weights the thread is still
    reading. The thread is told to stop after its current chunk.

    The wait for the slot is bounded (TTS_QUEUE_TIMEOUT_S, and TTS_MAX_QUEUE deep):
    503 with Retry-After, with nothing taken yet, so there is nothing to give back.
    *continuation* marks a later chunk of a stream that has started (see
    `_acquire_generation_slot`).

    *cleanup* (temp files) is owned from the moment of the call.
    """
    cancel = threading.Event()
    try:
        await _acquire_generation_slot(continuation=continuation)
    except BaseException:
        _unlink(*cleanup)
        raise
    try:
        lease = await _model_slot.acquire_lease()
    except BaseException:
        _GEN_SEM.release()
        _unlink(*cleanup)
        raise
    try:
        fut = asyncio.get_running_loop().run_in_executor(
            None,
            functools.partial(
                _generate_text, lease.model, _default_conds, list(chunks), language_id,
                prompt_path=prompt_path, exaggeration=exaggeration, cfg_weight=cfg_weight,
                gap_ms=gap_ms, cancel=cancel,
            ),
        )
    except BaseException:
        lease.release_soon()
        _GEN_SEM.release()
        _unlink(*cleanup)
        raise
    fut.add_done_callback(functools.partial(_generation_finished, lease, tuple(cleanup)))
    try:
        return await asyncio.shield(fut)
    except asyncio.CancelledError:
        cancel.set()
        raise


# --- Text splitting ----------------------------------------------------------

# Closing quotes/brackets may sit between the terminal punctuation and the space:
# German „…“ and »…«, English "…", (…).
_CLOSERS = "\"'\u201c\u201d\u2019\u00ab\u00bb)]"
_OPENERS = "\"'\u201e\u201c\u201d\u00ab\u00bb(["
_TERMINAL = ".!?;:"
_WHITESPACE = re.compile(r"\s+")

# Abbreviations that are (practically) never the last word of a sentence: a period
# after them is not a boundary, whatever follows. Lower case, without the dot.
_ABBREVIATIONS = frozenset("""
    dr prof nr str tel abb tab kap vgl ggf evtl inkl exkl zzgl max min bzw bzgl ca mio mrd
    std sek jh jhd geb gest hrsg dipl ing hr fr st mr mrs ms dt bsp ehem gem sog bes allg zt
""".split())
# Often sentence-final ("Äpfel, Birnen usw. Danach ...") but also mid-sentence
# ("... usw. sind gesund"): a boundary only if a capitalised word follows.
_AMBIGUOUS_ABBREVIATIONS = frozenset({"usw", "etc", "uvm", "usf"})
# Nouns that follow a written-out ordinal number ("im 100. Jahr"). Ordinals of one
# or two digits ("am 3. Mai", "der 21. Jahrgang") need no help: see below.
_ORDINAL_FOLLOWERS = frozenset({
    "jahr", "jahre", "jahrhundert", "jahrestag", "jubilaeum", "jubiläum", "geburtstag", "todestag",
    "auflage", "ausgabe", "wiederholung", "besucher", "kunde", "spiel", "platz", "mal", "folge",
})


def _dot_ends_sentence(text: str, dot: int, after: int) -> bool:
    """Whether the single '.' at *dot* ends a sentence; *after* is where the next word starts.

    German is full of periods that are not boundaries: ordinals ("am 3. Mai"),
    abbreviations ("z. B.", "Dr. Müller", "ca. 5", "Nr. 7"). Splitting there makes
    the TTS speak "am drei" and pause. The bias is deliberate: a missed boundary
    only makes a chunk longer (the length ceiling still applies), a false one
    changes how the text is read.

    Scans outwards from the dot instead of slicing or searching from the start of
    the text: the text is caller-controlled, and a regex like `(\\S+)$` over a long
    unbroken token is quadratic (23 s of event-loop time for 5000 characters).
    """
    begin = dot
    while begin > 0 and not text[begin - 1].isspace():
        begin -= 1
    token = text[begin:dot].lstrip(_OPENERS)
    following = text[after:after + 64].lstrip(_OPENERS)
    word = following.split(None, 1)[0] if following.strip() else ""
    lowered = token.lower()

    if not token:
        return True
    if token.isdigit():
        if len(token) <= 2:
            return False  # "3. Mai", "21. Jahrhundert"
        return not (word[:1].islower() or word.strip(".,;:!?").lower() in _ORDINAL_FOLLOWERS)
    if len(token) == 1 and token.isalpha():
        return False  # initials and the parts of "z. B.", "d. h.", "u. a.", "S. 5"
    if "." in token:
        parts = token.split(".")
        if all(p.isalpha() and len(p) <= 2 for p in parts if p):
            return False  # "z.B.", "d.h.", "i.d.R."
    if lowered in _ABBREVIATIONS:
        return False
    if lowered in _AMBIGUOUS_ABBREVIATIONS:
        return word[:1].isupper()
    return True


def _sentences(line: str) -> list:
    """Split one line at sentence boundaries (German-aware for periods)."""
    out, start = [], 0
    for space in _WHITESPACE.finditer(line):
        # Walk back from the whitespace: optional closing quotes, then the
        # terminal punctuation. Linear overall, unlike one regex for all of it.
        end = space.start()
        closers = end
        while closers > start and line[closers - 1] in _CLOSERS:
            closers -= 1
        first = closers
        while first > start and line[first - 1] in _TERMINAL:
            first -= 1
        if first == closers:
            continue  # no terminal punctuation before this whitespace
        if closers - first == 1 and line[first] == "." and not _dot_ends_sentence(line, first, space.end()):
            continue
        out.append(line[start:end])
        start = space.end()
    out.append(line[start:])
    return [s.strip() for s in out if s.strip()]


def _split_sentences(text: str, first_chunk_chars: int = 40,
                     min_chars: int = 60, max_chars: int = 180) -> list[str]:
    """Split text into chunks for incremental generation.

    Three properties matter for time-to-first-audio:
    - the *first* chunk gets a small floor, because it alone sets TTFA;
    - later chunks get a larger floor, because tiny clips have poor prosody;
    - every chunk gets a ceiling. Without one, text that has no terminal
      punctuation (very common in LLM output) collapses into a single chunk and
      TTFA silently reverts to full-text latency — the exact failure this
      endpoint exists to avoid.
    The same ceiling keeps every chunk of /tts and /clone far below the library's
    1000-token (~40 s) limit per generate() call.
    """
    text = text.strip()
    if not text:
        return []

    raw = [s for line in re.split(r"\n+", text) for s in _sentences(line)]

    # Enforce the ceiling: re-split over-long pieces on commas, then whitespace.
    bounded: list[str] = []
    for part in raw:
        while len(part) > max_chars:
            cut = part.rfind(", ", 0, max_chars)
            # Keep the comma with the chunk it terminates: cutting before it
            # moved the pause to the START of the next chunk, which the model
            # then speaks as a leading ", ...".
            end = cut + 1 if cut >= min_chars else cut
            if cut < min_chars:
                cut = end = part.rfind(" ", 0, max_chars)
            if cut < min_chars:
                cut = end = max_chars  # a single unbroken token; hard-cut it
            bounded.append(part[:end].strip())
            part = part[end:].strip()
        if part:
            bounded.append(part)

    merged: list[str] = []
    buf = ""
    for part in bounded:
        # Flush before overflowing, or the merge step would undo the ceiling
        # the loop above just enforced — and chunk 0 is what sets TTFA.
        if buf and len(buf) + 1 + len(part) > max_chars:
            merged.append(buf)
            buf = ""
        buf = f"{buf} {part}".strip() if buf else part
        floor = first_chunk_chars if not merged else min_chars
        if len(buf) >= floor:
            merged.append(buf)
            buf = ""
    if buf:
        # A trailing fragment is appended only if that keeps us under the
        # ceiling; otherwise it stands alone rather than breaking the bound.
        if merged and len(merged[-1]) + len(buf) + 1 <= max_chars:
            merged[-1] += " " + buf
        else:
            merged.append(buf)
    return merged or [text]


def _split_for_batch(text: str) -> list[str]:
    """Chunks for /tts and /clone: nobody waits for a first chunk, so no small first floor."""
    return _split_sentences(text, first_chunk_chars=60, min_chars=60, max_chars=180)


# --- Status endpoints --------------------------------------------------------

def _effective_repetition_penalty(model) -> Optional[float]:
    """The penalty generate() will use: the env override, else the library default."""
    if REPETITION_PENALTY is not None:
        return REPETITION_PENALTY
    try:
        default = inspect.signature(model.generate).parameters["repetition_penalty"].default
        return None if default is inspect.Parameter.empty else float(default)
    except Exception:
        return None


@app.get("/health")
async def health():
    """Liveness probe.

    `model_resident: false` is NOT an error — the idle TTL unloaded the weights
    to free VRAM and the next request reloads them. Returning a non-200 here
    would make an idle container show as unhealthy under Docker's `curl -f`.
    `t3_model` is the checkpoint the last load actually used (null before the
    first load); `t3_model_requested` is what CHATTERBOX_T3_MODEL asked for.
    """
    return {
        "status": "ok",
        "model_loaded": model_loaded,
        "model_resident": _model_slot.resident,
        "model_ttl_seconds": MODEL_TTL,
        "active_requests": _model_slot.refs,
        "device": device,
        "t3_model": _active_t3_model,
        "t3_model_requested": T3_MODEL,
    }


@app.get("/ready")
async def ready():
    """Readiness probe, separate from /health (see ModelSlot.readiness).

    200 once a load has ever succeeded — a later TTL unload does not make the
    service unready — and also while nothing is known to be wrong; 503 while the
    first load runs (`reason: loading`) or when it failed and never succeeded
    (`reason: load_failed`). Never loads a model itself.

    Unauthenticated, so a failed load is reported as a category only: the
    exception text (download URLs, cache paths) is in the service log, written
    when the load failed.
    """
    state = _model_slot.readiness()
    if state.get("last_error"):
        state = {**state, "last_error": "load_failed",
                 "detail": "The model failed to load; see the service log." if state.get("detail") else None}
    body = {
        "status": "ready" if state["ready"] else "not_ready",
        **state,
        "t3_model": _active_t3_model,
    }
    return JSONResponse(status_code=200 if state["ready"] else 503, content=body)


@app.post("/unload")
async def unload():
    """Release the model and its VRAM now, without stopping the container.

    The idle TTL covers the common case; this is the deliberate one — you are
    about to run something else on the same GPU and want the memory back
    immediately. The next request reloads transparently.

    200 when released or already unloaded; **409 while a request is in flight**
    (including a generation whose client already left), because freeing memory a
    running generation still reads would crash the worker. Retry once
    `active_requests` reaches zero.
    """
    result = await _model_slot.try_unload_async()
    if result["reason"] == "busy":
        return JSONResponse(
            status_code=409,
            content={"detail": "Model is in use; retry when idle", **result},
        )
    return {"model_resident": _model_slot.resident, **result}


@app.get("/status")
async def status():
    """Return detailed service status including GPU memory information."""
    model = tts_model
    status_info = {
        "status": "ok",
        "service": "Chatterbox-TTS",
        "device": device,
        "cuda_available": torch.cuda.is_available(),
        "model_loaded": model_loaded,
        "default_language": DEFAULT_LANGUAGE,
        "supported_languages": _supported_language_ids(model) if model is not None else [],
        "t3_model": _active_t3_model,
        "t3_model_requested": T3_MODEL,
        "hf_revision": HF_REVISION,
        "chatterbox_tts_version": _package_version(),
        "chatterbox_ref": CHATTERBOX_REF or None,
        "repetition_penalty": _effective_repetition_penalty(model) if model is not None else REPETITION_PENALTY,
        "chunk_gap_ms": CHUNK_GAP_MS,
        "max_text_chars": MAX_TEXT_CHARS,
    }
    if torch.cuda.is_available():
        status_info["gpu_name"] = torch.cuda.get_device_name(0)
        status_info["gpu_memory_allocated"] = torch.cuda.memory_allocated()
        status_info["gpu_memory_total"] = torch.cuda.get_device_properties(0).total_memory
    return status_info


@app.get("/languages")
async def languages():
    """List supported language ids."""
    model = tts_model if model_loaded else None
    return {
        "languages": _supported_language_ids(model) if model else sorted(set(LANGUAGE_ALIASES.values())),
        "default": DEFAULT_LANGUAGE,
    }


# --- Synthesis endpoints -----------------------------------------------------

class TTSRequest(BaseModel):
    """Request body for default-voice synthesis (simple-json-tts-v1)."""

    text: str
    language: str = "auto"
    exaggeration: Optional[float] = None
    cfg_weight: Optional[float] = None


@app.post("/tts")
async def text_to_speech(request: TTSRequest):
    """Generate speech with Chatterbox's built-in default voice.

    The text is generated in sentence-sized chunks and joined; one generate()
    call for a long text would silently stop after ~40 s of audio.
    """
    text = _require_text(request.text)
    _check_tuning(request.exaggeration, request.cfg_weight)
    _shed_if_saturated()

    try:
        language_id = _resolve_language(request.language)
        chunks = _split_for_batch(text)
        started = time.monotonic()
        audio, sr = await _run_generation(
            chunks, language_id, exaggeration=request.exaggeration, cfg_weight=request.cfg_weight,
        )
        elapsed = time.monotonic() - started
        logger.info(f"/tts generated {len(text)} chars in {len(chunks)} chunk(s) in {elapsed:.2f}s")
        return _wav_response(audio, sr, {
            "X-Language": language_id,
            "X-Chunk-Count": str(len(chunks)),
            "X-Generation-Time": f"{elapsed:.3f}",
        })
    except HTTPException:
        raise
    except Exception as e:
        raise _server_error(e, "Speech generation failed")


def _streaming_wav_header(sample_rate: int) -> bytes:
    """PCM16 mono WAV header with unknown (maxed) sizes for chunked streaming."""
    byte_rate = sample_rate * 2
    return struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", 0xFFFFFFFF, b"WAVE",
        b"fmt ", 16, 1, 1, sample_rate, byte_rate, 2, 16,
        b"data", 0xFFFFFFFF,
    )


def _to_pcm16(wav) -> bytes:
    """Convert a model output tensor/array to raw PCM16 little-endian bytes."""
    audio = np.clip(_to_numpy(wav), -1.0, 1.0)
    return (audio * 32767.0).astype("<i2").tobytes()


@app.post("/tts-stream")
async def text_to_speech_stream(request: TTSRequest):
    """Sentence-chunked streaming TTS: audio starts after the first sentence.

    Returns a chunked `audio/wav` stream (PCM16 mono, unknown-length header).
    The chatterbox package has no token-level streaming API, so this splits
    the text into sentences and streams each one as soon as it is generated —
    time-to-first-audio drops from total-length to first-sentence latency.
    """
    text = _require_text(request.text)
    _check_tuning(request.exaggeration, request.cfg_weight)
    _shed_if_saturated()
    language_id = _resolve_language(request.language)
    sentences = _split_sentences(text)

    # The generator below runs AFTER this handler returns, so the model is pinned
    # by a lease that outlives the handler. A generator that never starts (client
    # already gone) never runs its `finally`, hence the additional release paths
    # below; the lease makes each of them safe to hit more than once. Every chunk
    # additionally pins the model itself for as long as its worker thread runs
    # (see _run_generation), so closing the stream early cannot free weights that
    # are still being read.
    try:
        lease = await _model_slot.acquire_lease()
    except Exception as e:
        raise _failure(500, "The speech model could not be loaded", e, exc_info=True)
    try:
        sample_rate = lease.model.sr

        # Generate the first chunk up front so a failure here becomes a real HTTP
        # error. Once the StreamingResponse is returned the status line is already
        # on the wire, and the header advertises 0xFFFFFFFF sizes — a client cannot
        # distinguish a truncated stream from a complete one.
        started = time.monotonic()
        first_audio, _sr = await _run_generation(
            sentences[:1], language_id, exaggeration=request.exaggeration, cfg_weight=request.cfg_weight,
        )
        ttfa = time.monotonic() - started
        logger.info(
            f"/tts-stream TTFA {ttfa * 1000:.0f}ms "
            f"(chunk 1/{len(sentences)}, {len(sentences[0])} chars)"
        )
    except BaseException as e:
        # Nothing will consume the generator, so give the reference back here.
        await lease.release_async()
        if isinstance(e, Exception) and not isinstance(e, HTTPException):
            raise _server_error(e, "Speech generation failed")
        raise

    gap_pcm = _to_pcm16(np.zeros(int(sample_rate * CHUNK_GAP_MS / 1000), dtype=np.float32))

    async def generate_chunks():
        try:
            yield _streaming_wav_header(sample_rate)
            yield _to_pcm16(first_audio)
            for index, sentence in enumerate(sentences[1:], start=2):
                try:
                    audio, _sr = await _run_generation(
                        [sentence], language_id,
                        exaggeration=request.exaggeration, cfg_weight=request.cfg_weight,
                        continuation=True,
                    )
                except Exception as e:
                    logger.error(f"Streaming TTS failed at chunk {index}/{len(sentences)}: {e}", exc_info=True)
                    # Raise rather than break: this makes Starlette abort the
                    # chunked body, so the client sees a broken transfer instead of
                    # a clean 200 with silently missing audio.
                    raise
                if gap_pcm:
                    yield gap_pcm
                yield _to_pcm16(audio)
        finally:
            # Completion, error, and a client that disconnects mid-stream (the
            # generator is closed). Never blocks: this can run in the GC.
            lease.release_soon()

    stream = generate_chunks()
    lease.release_on_gc(stream)  # a generator that is dropped without ever starting
    return StreamingResponse(
        stream,
        media_type="audio/wav",
        headers={
            "X-Language": language_id,
            "X-Sample-Rate": str(sample_rate),
            "X-Sentence-Count": str(len(sentences)),
            "X-Time-To-First-Audio": f"{ttfa:.3f}",
        },
        # Runs after the response was handled, including when the client left
        # before the first byte and the generator was never iterated.
        background=BackgroundTask(lease.release_async),
    )


# --- Voice cloning -----------------------------------------------------------

def _upload_suffix(filename: Optional[str]) -> str:
    """A safe temp-file suffix from a client-supplied name (decoders sniff it)."""
    suffix = re.sub(r"[^A-Za-z0-9.]", "", Path(filename).suffix) if filename else ""
    return suffix[:12] or ".wav"


def _decode_reference(path: str, max_seconds: float) -> np.ndarray:
    """First *max_seconds* of *path* as mono float32 at the model's reference rate.

    Loading at 24 kHz is exactly what the library does with the reference; doing
    it here first only stops it from decoding and embedding a whole long file.
    """
    import librosa

    audio, _sr = librosa.load(path, sr=_REF_SAMPLE_RATE, mono=True, duration=max_seconds)
    return audio


def _stage_reference(upload: UploadFile) -> str:
    """Blocking: bounded copy of the upload, re-encoded (capped) as a WAV; returns its path.

    413 past MAX_UPLOAD_MB, 400 for an empty or undecodable file. Only the returned
    WAV survives; the raw copy is removed on every path.
    """
    detail = f"Reference audio is larger than {MAX_UPLOAD_BYTES / _MIB:g} MB (MAX_UPLOAD_MB)."
    raw = out = None
    try:
        fd, raw = tempfile.mkstemp(suffix=_upload_suffix(upload.filename))
        os.close(fd)
        total = 0
        with open(raw, "wb") as fh:
            while True:
                block = upload.file.read(_UPLOAD_CHUNK)
                if not block:
                    break
                total += len(block)
                if total > MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail=detail)
                fh.write(block)
        if total == 0:
            raise HTTPException(status_code=400, detail="Reference audio is empty")
        try:
            audio = _decode_reference(raw, REF_MAX_SECONDS)
        except Exception as e:
            logger.warning("Could not decode reference audio %r: %s", upload.filename, e)
            raise HTTPException(status_code=400, detail="Could not decode the reference audio (unsupported or corrupt file)")
        if audio.size == 0:
            raise HTTPException(status_code=400, detail="Reference audio contains no samples")
        fd, out = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        sf.write(out, audio, _REF_SAMPLE_RATE, format="WAV", subtype="FLOAT")
        staged, out = out, None
        return staged
    finally:
        _unlink(raw, out)


def _discard_staged(fut) -> None:
    if not fut.cancelled() and fut.exception() is None:
        _unlink(fut.result())


async def _stage_reference_async(upload: UploadFile) -> str:
    """`_stage_reference` off the event loop, without leaking its file if we are cancelled."""
    if upload.size is not None and upload.size > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Reference audio is larger than {MAX_UPLOAD_BYTES / _MIB:g} MB (MAX_UPLOAD_MB).",
        )
    fut = asyncio.get_running_loop().run_in_executor(None, _stage_reference, upload)
    try:
        return await asyncio.shield(fut)
    except asyncio.CancelledError:
        fut.add_done_callback(_discard_staged)
        raise


async def _clone(text: str, lang: str, file: UploadFile,
                 exaggeration: Optional[float], cfg_weight: Optional[float]) -> StreamingResponse:
    """Shared implementation for the clone endpoints."""
    text = _require_text(text)
    _check_tuning(exaggeration, cfg_weight)
    _shed_if_saturated()

    try:
        language_id = _resolve_language(lang)
        chunks = _split_for_batch(text)
        prompt_path = await _stage_reference_async(file)

        # _run_generation owns the staged file from here and deletes it when the
        # worker thread is done with it.
        started = time.monotonic()
        audio, sr = await _run_generation(
            chunks, language_id, prompt_path=prompt_path,
            exaggeration=exaggeration, cfg_weight=cfg_weight, cleanup=(prompt_path,),
        )
        logger.info(f"/clone generated {len(text)} chars in {len(chunks)} chunk(s) in {time.monotonic() - started:.2f}s")
        return _wav_response(audio, sr, {
            "X-Language": language_id,
            "X-Chunk-Count": str(len(chunks)),
            "X-Clone-Source": file.filename or "unknown",
        })
    except HTTPException:
        raise
    except Exception as e:
        raise _server_error(e, "Voice cloning failed")


@app.post("/clone")
async def clone_voice(
    text: str = Form(...),
    lang: str = Form("auto"),
    file: UploadFile = File(...),
    exaggeration: Optional[float] = Form(None),
    cfg_weight: Optional[float] = Form(None),
):
    """Zero-shot voice cloning from a reference clip (voice-clone-tts-v1)."""
    return await _clone(text, lang, file, exaggeration, cfg_weight)


@app.post("/clone-with-ref-text")
async def clone_voice_with_ref_text(
    text: str = Form(...),
    ref_text: str = Form(""),
    lang: str = Form("auto"),
    file: UploadFile = File(...),
):
    """Contract-compatible alias: Chatterbox doesn't use a reference transcript,
    so this behaves like /clone and ignores ref_text."""
    return await _clone(text, lang, file, None, None)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5007)
