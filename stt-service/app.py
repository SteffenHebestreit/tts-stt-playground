"""Speech-to-Text service powered by faster-whisper.

Supports CUDA, ROCm, and CPU backends with automatic hardware detection.
Provides batch transcription, streaming SSE, live WebSocket transcription and
language detection endpoints.
"""

from fastapi import FastAPI, UploadFile, HTTPException, File, Form, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse, JSONResponse
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
from faster_whisper import WhisperModel
try:
    from faster_whisper import BatchedInferencePipeline
except ImportError:  # pragma: no cover - only an old or stubbed faster-whisper lacks it
    BatchedInferencePipeline = None
from typing import Any, Union, List
from contextlib import asynccontextmanager
import torch
import os
import re
import logging
import tempfile
import json
import asyncio
import time
import uuid
import gc
import threading
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor

from json_utils import (
    clean_json_inf_nan,
    is_custom_model,
    is_multilingual,
    model_can_translate,
    normalize_language,
    ENGLISH_ONLY_MODELS,
    WHISPER_LANGUAGES,
)
from body_limit import BodyLimitMiddleware
from local_agreement import LocalAgreement, UNSPACED_LANGUAGES, words_from_segments
from origin_guard import OriginGuardMiddleware, parse_allowed_origins
from residency import IdleUnloader, error_category, ttl_from_env

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _startup_preload():
    """Load and warm the model in a worker thread, then start the idle clock.

    Runs off the event loop so the port opens at once: /health answers while a
    first-time download is still running, and /ready reports 503 "loading" for
    exactly as long as it takes instead of the container looking dead.
    """
    global _preload_pending
    try:
        # The same lock acquire_model() loads under, so a request that arrives
        # mid-preload waits for this load instead of starting a second one.
        with _model_ref_lock:
            if whisper_model is None:
                load_model()
                warm_up_model()
    except Exception:  # pragma: no cover - load_model/warm_up_model swallow their own
        logger.exception("Whisper preload failed")
    finally:
        _preload_pending = False
        _preload_done.set()
    if MODEL_TTL > 0:
        # Nothing holds a reference yet, so start the clock: a container that is
        # booted and never used should not sit on the VRAM either.
        _idle_unloader.arm()


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    """Start the model preload on startup; release the executors on shutdown."""
    global _preload_pending
    if MODEL_TTL == 0:
        # "Release the moment nothing is using it" — preloading here would load
        # ~2 GB, warm it, and immediately throw both away. The first request
        # loads it instead, which is what this setting asks for.
        logger.info("Whisper preload skipped: STT_MODEL_TTL=0 unloads on every idle")
    else:
        _preload_pending = True
        _preload_done.clear()
        # Held on the app so the task is not garbage-collected mid-flight.
        _app.state.preload_task = asyncio.create_task(asyncio.to_thread(_startup_preload))

    if MODEL_TTL < 0:
        logger.info("Whisper idle unloading disabled (STT_MODEL_TTL=-1)")
    elif MODEL_TTL > 0:
        logger.info(f"Whisper idle unloading after {MODEL_TTL:.0f}s idle")
    yield
    logger.info("Shutting down thread pool executors...")
    _idle_unloader.cancel()
    rt_executor.shutdown(wait=False)
    executor.shutdown(wait=False)
    _acquire_executor.shutdown(wait=False)


app = FastAPI(
    title="STT Service",
    description="Speech-to-Text Service with Hardware Acceleration",
    lifespan=_lifespan,
)

# CORS (env-configurable). Unset or empty ALLOWED_ORIGINS means no CORS headers at
# all (it used to mean "*"); "*" only when it is written down (and logged);
# otherwise an explicit list. The middleware stack itself is assembled below, next
# to the upload limits it needs.
allowed_origins = parse_allowed_origins(os.getenv("ALLOWED_ORIGINS", ""))
allow_credentials = os.getenv("ALLOW_CREDENTIALS", "false").strip().lower() in {"1", "true", "yes", "on"}
if "*" in allowed_origins and allow_credentials:
    logger.warning("ALLOW_CREDENTIALS=true with ALLOWED_ORIGINS='*' is not permitted by CORS spec; disabling credentials.")
    allow_credentials = False


def _number(raw: str, default, cast, name: str):
    """Parse a numeric env value, falling back to the default rather than crashing at import."""
    raw = (raw or "").strip()
    if not raw:
        return default
    try:
        return cast(raw)
    except ValueError:
        logger.warning(f"Ignoring non-numeric {name}={raw!r}; using {default}")
        return default


def _flag(raw: str) -> bool:
    return (raw or "").strip().lower() in {"1", "true", "yes", "on"}


def _request_id() -> str:
    """Names one request in the log and in the error it returns, so the two can be matched."""
    return uuid.uuid4().hex[:12]


def _internal_error(what: str, request_id: str) -> str:
    """What a client is told about an unexpected failure.

    ``str(exc)`` is written for the operator and can hold temp-file and model
    paths or library internals, so the client gets this and the request id; the
    caller has logged the exception under the same id.
    """
    return f"{what} (internal error). Request id: {request_id}."


def _public_model_name(name):
    """A model name as an unauthenticated response may show it.

    ``WHISPER_MODEL_SIZE`` also takes a local path (a CTranslate2 export on a
    mounted volume); the directory layout is not for callers, so such a value is
    shown as its last component. Sizes and Hugging Face ids (``org/name``) are
    returned unchanged.
    """
    if not name or not isinstance(name, str):
        return name
    if os.path.isabs(name) or name.startswith(("./", "../", "~")):
        return os.path.basename(os.path.normpath(name)) or name
    return name


# Two pools so a long batch upload cannot starve live sessions of a worker.
# CTranslate2 itself serialises on the shared model (see WHISPER_NUM_WORKERS), so
# these bound queueing, not GPU parallelism.
#
# The realtime pool must have a slot per admitted session, plus one. Cancelling
# an asyncio task that is awaiting run_in_executor does NOT stop the worker
# thread — the decode runs to completion regardless — so an abandoned interim
# keeps occupying its slot until it finishes. Sized at 2 with WS_MAX_SESSIONS=4,
# two stopping sessions could freeze the other two.
_MAX_LIVE_SESSIONS = int(os.getenv("WS_MAX_SESSIONS", "4"))
rt_executor = ThreadPoolExecutor(
    max_workers=max(2, _MAX_LIVE_SESSIONS + 1), thread_name_prefix="whisper-rt"
)
# Every batch worker holds a fully decoded copy of its file (16 kHz float32, about
# 230 MB per hour of audio), so the pool size is also the memory ceiling. One
# thread per core let a burst of uploads decode that many files at once.
_BATCH_WORKERS = max(1, _number(
    os.getenv("STT_BATCH_WORKERS", ""), min(4, os.cpu_count() or 4), int, "STT_BATCH_WORKERS"
))
executor = ThreadPoolExecutor(max_workers=_BATCH_WORKERS, thread_name_prefix="whisper-batch")
# The streaming and WebSocket routes take their model reference here (see
# _hold_model_async), not on the loop's default executor: a cold load holds the
# reference lock for seconds to minutes, and every waiter used to park a default
# executor thread on it, which starves everything else that uses that pool (uploads,
# /unload). The acquires serialise on the lock anyway, so one thread is all they need.
_acquire_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="whisper-acquire")

# Upload limits. MAX_UPLOAD_MB is per file, checked while the file is copied out of
# the spooled request; the request body as a whole is bounded by the middleware
# below (body_limit.py), so the wire is bounded too and not only what is decoded.
# <= 0 disables a limit.
MAX_UPLOAD_BYTES = int(_number(os.getenv("MAX_UPLOAD_MB", "200"), 200.0, float, "MAX_UPLOAD_MB") * 1024 * 1024)
# An abuse guard, not a tuning knob: 2 h is far beyond anything the UI produces.
MAX_AUDIO_SECONDS = _number(os.getenv("MAX_AUDIO_SECONDS", "7200"), 7200.0, float, "MAX_AUDIO_SECONDS")

_MIB = 1024 * 1024
# multipart boundaries and part headers, and the small text fields, on top of the file bytes
_MULTIPART_SLACK = _MIB
_FORM_FIELDS_SLACK = 64 * 1024
# /transcribe takes one file (`audio`) or several (`audios`), each bounded on its own
# by MAX_UPLOAD_MB; the request as a whole may carry this many files' worth.
_BATCH_BODY_FILES = 8
_UPLOAD_ROUTES = frozenset({"/transcribe", "/transcribe-stream", "/detect_language"})


def _body_limit(path: str):
    """Largest request body (bytes) accepted at *path*; None means unlimited."""
    if path in _UPLOAD_ROUTES:
        if MAX_UPLOAD_BYTES <= 0:
            return None
        files = _BATCH_BODY_FILES if path == "/transcribe" else 1
        return files * (MAX_UPLOAD_BYTES + _FORM_FIELDS_SLACK) + _MULTIPART_SLACK
    return _MIB  # nothing else takes a body of any size


# Each add_middleware wraps what was added before it: the body limit is innermost,
# the origin guard (403 for a state-changing request or a WebSocket handshake with a
# foreign Origin header; no Origin, e.g. the gateway or curl, is not affected) comes
# next, and CORS is outermost so a 403/413 still carries the CORS headers a listed
# origin needs in order to read it. CORS is only added when origins are configured.
app.add_middleware(
    BodyLimitMiddleware,
    limit_for=_body_limit,
    hint=f"The upload limit is {MAX_UPLOAD_BYTES / _MIB:.3g} MB per file (MAX_UPLOAD_MB).",
)
app.add_middleware(OriginGuardMiddleware, allowed_origins=allowed_origins)
if allowed_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=allow_credentials,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Opt-in: the batched pipeline decodes VAD chunks in parallel, which is much
# faster on long files but drops temperature fallback and previous-text
# conditioning, so it changes output and stays off unless asked for. Never used
# for the WebSocket path.
BATCHED_INFERENCE = _flag(os.getenv("STT_BATCHED_INFERENCE", "false"))
BATCH_SIZE = max(1, _number(os.getenv("STT_BATCH_SIZE", "8"), 8, int, "STT_BATCH_SIZE"))

# Where faster-whisper stores downloaded models. Empty keeps the Hugging Face
# cache (HF_HOME, /root/.cache/huggingface in the image), which is where existing
# installs already have their weights: defaulting to /app/models would orphan
# them and re-download 1.6 GB on upgrade.
WHISPER_MODEL_DIR = os.getenv("WHISPER_MODEL_DIR", "").strip()


def _default_language():
    """STT_DEFAULT_LANGUAGE normalised, or None (auto-detect) when unset or invalid."""
    raw = os.getenv("STT_DEFAULT_LANGUAGE", "")
    try:
        return normalize_language(raw)
    except ValueError:
        logger.warning(f"Ignoring unsupported STT_DEFAULT_LANGUAGE={raw!r}; auto-detecting")
        return None


# Applied when a request does not name a language. An explicit "auto" still
# means auto-detect, so this never overrides a caller's choice.
STT_DEFAULT_LANGUAGE = _default_language()


class SafeJSONResponse(JSONResponse):
    """JSONResponse that sanitises NaN/Infinity before encoding."""
    def render(self, content: Any) -> bytes:
        """Serialise JSON after normalising unsupported float values."""
        return json.dumps(
            clean_json_inf_nan(content),
            ensure_ascii=False,
            allow_nan=False, # This is the default, but being explicit
            indent=None,
            separators=(",", ":"),
        ).encode("utf-8")

def _select_cuda_compute_type() -> str:
    """Pick the CT2 compute type for this GPU.

    Defaults to int8_float16: faster-whisper's own table measures int8 at ~35%
    less VRAM than float16 at equal-or-better speed, which is the difference
    between fitting and not fitting on an 8-12 GB card. Set WHISPER_COMPUTE_TYPE
    to pin it (float16, int8_float16, int8_bfloat16, int8, float32).

    int8_bfloat16 is deliberately NOT auto-selected below compute capability 8.0:
    on Turing, CTranslate2 silently falls back to int8_float32, which costs
    activation memory rather than saving it.
    """
    requested = os.getenv("WHISPER_COMPUTE_TYPE", "").strip().lower()

    supported = set()
    try:
        import ctranslate2
        supported = set(ctranslate2.get_supported_compute_types("cuda"))
    except Exception as e:
        logger.warning(f"Could not probe CT2 compute types ({e}); assuming float16 is safe")

    if requested and requested != "auto":
        if supported and requested not in supported:
            logger.warning(
                f"WHISPER_COMPUTE_TYPE={requested} is not supported by this GPU "
                f"(supported: {sorted(supported)}); falling back to auto-selection"
            )
        else:
            return requested

    # Most memory-efficient first; every entry must be verified as supported.
    preference = ["int8_float16", "int8", "float16", "float32"]
    try:
        if torch.cuda.get_device_capability(0)[0] >= 8:
            preference.insert(0, "int8_bfloat16")
    except Exception:
        pass

    for candidate in preference:
        if not supported or candidate in supported:
            return candidate
    return "float16"


# Hardware detection and optimization (re-check after startup)
def detect_hardware():
    """Auto-detect the best compute device (CUDA > ROCm > CPU) and return ``(device, compute_type)``."""
    force_accel = os.getenv("FORCE_ACCELERATION", "").lower()

    if force_accel == "rocm":
        # ROCm presents as CUDA to PyTorch; ctranslate2 4.x+ required for GPU use.
        # If ctranslate2 was not compiled with HIP support, fall back gracefully to CPU.
        try:
            import ctranslate2
            providers = ctranslate2.get_supported_compute_types("cuda")
            if "float16" in providers:
                device = "cuda"
                compute_type = "float16"
                logger.info(f"ROCm mode: using GPU (ctranslate2 HIP). GPU: {torch.cuda.get_device_name(0)}")
            else:
                raise RuntimeError("ctranslate2 has no float16 CUDA/HIP support")
        except Exception as e:
            logger.warning(f"ROCm requested but ctranslate2 HIP unavailable ({e}); falling back to CPU int8")
            device = "cpu"
            compute_type = "int8"
        return device, compute_type

    # Explicit CPU override
    if os.getenv("USE_CUDA", "").lower() == "false":
        logger.info("Hardware acceleration disabled via USE_CUDA=false")
        return "cpu", "int8"

    if torch.cuda.is_available():
        device = "cuda"
        compute_type = _select_cuda_compute_type()
        logger.info(f"CUDA available with {torch.cuda.device_count()} GPU(s)")
        logger.info(f"GPU: {torch.cuda.get_device_name(0)} using {compute_type}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = "cpu"  # faster-whisper doesn't support MPS
        compute_type = "int8"
        logger.info("Apple Silicon detected, using optimized CPU int8")
    else:
        device = "cpu"
        compute_type = "int8"
        logger.info("Using CPU int8")

    return device, compute_type

# Defer hardware detection until first use.
#
# `device` / `compute_type` are the PREFERRED runtime: what detection picked, or
# what an operator pinned. load_model() only reads them, never writes them. It
# used to overwrite them with whatever it ended up on, so one transient CUDA OOM
# during the first load left them at cpu/int8 for the life of the process and
# every later load — including the one after an idle unload, when the VRAM was
# free again — only ever tried the CPU. What is actually running is tracked
# separately in `active_device` / `active_compute_type`.
device = None
compute_type = None
active_device = None
active_compute_type = None

# large-v3-turbo has 4 decoder layers against large-v3's 32 at comparable accuracy,
# which dominates greedy realtime decoding. FALLBACK_MODEL_SIZE is used if the
# configured name cannot be resolved, so an unknown alias cannot brick startup.
DEFAULT_MODEL_SIZE = "large-v3-turbo"
FALLBACK_MODEL_SIZE = "large-v3"
# Used only when a load fails for resource reasons, so it must be SMALLER than
# the default and still multilingual. `small` is ~0.5 GB against turbo's ~1.6 GB.
# Never use distil-large-v3 here: it is English-only despite having no `.en`
# suffix, so it would silently drop German on exactly the constrained devices
# this ladder exists to serve.
OOM_FALLBACK_MODEL_SIZE = os.getenv("WHISPER_OOM_FALLBACK", "small")

# Approximate weight sizes in MB, used only to keep the fallback ladder
# descending. Multilingual models only — English-only variants (*.en,
# distil-large-v3) are excluded on purpose so they can never be auto-selected
# and silently drop German.
KNOWN_MODEL_SIZES = {
    "tiny": 75,
    "base": 145,
    "small": 490,
    "medium": 1500,
    "large-v1": 2900,
    "large-v2": 2900,
    "large-v3": 3100,
    "large-v3-turbo": 1600,
    "turbo": 1600,
}


def _model_rank(name: str) -> int:
    """Approximate memory cost, for ordering fallbacks.

    A local CTranslate2 export is measured from its weights file. Anything else
    unknown (a Hub repo id, a typo) sorts as very large so it is never treated
    as a safe step down.
    """
    known = KNOWN_MODEL_SIZES.get(name)
    if known is not None:
        return known
    try:
        return max(1, os.path.getsize(os.path.join(name, "model.bin")) // (1024 * 1024))
    except OSError:
        return 10_000


def _is_valid_model_name(name: str) -> bool:
    """Is `name` something faster-whisper can resolve, so a failed load is not a typo?

    Built-in sizes (including the English-only variants) resolve through its own
    table; a Hub repo id or a path is valid by construction. Only a string that
    is none of these is worth retrying as FALLBACK_MODEL_SIZE.
    """
    return (
        name in KNOWN_MODEL_SIZES
        or name in ENGLISH_ONLY_MODELS
        or name == "large"
        or is_custom_model(name)
    )


def configured_model_size() -> str:
    """The model name requested via env, or the project default."""
    return os.getenv("WHISPER_MODEL_SIZE", "").strip() or DEFAULT_MODEL_SIZE


# Initialize Whisper model with error handling
whisper_model = None
model_loaded = False
model_size_loaded = None
model_warmed = False
# Why the last load failed, as a short category (residency.error_category): what
# /health, /ready and the 503s may say to anyone. The exception itself (paths, repo
# ids, URLs) is in the log. None while nothing is known to be wrong.
startup_error = None
# True while load_model() runs; read lock-free by /ready, which must never wait
# on the very lock the loader holds.
_loading = False
# A startup preload has been scheduled and has not finished. /ready reports
# "loading" for this window so an orchestrator does not route to a container
# whose first load has not even begun.
_preload_pending = False
_preload_done = threading.Event()
_preload_done.set()

# Reference counting for the shared model.
#
# Until POST /unload existed the model was loaded once at startup and never
# dropped, so reading the `whisper_model` global at call time was always safe.
# It is not any more: several transcribe paths read the global directly inside a
# worker thread, and an unload landing between the check and the read would give
# them None. Callers therefore take a reference for the duration of their work,
# and unload refuses while any reference is outstanding rather than pulling the
# model out from under a request in flight.
_model_refs = 0
_model_ref_lock = threading.Lock()

# Idle unloading. Whisper is the one always-on GPU consumer in the stack, so
# without this `MODEL_TTL` frees the ~2-4 GB held by qwen3-asr and chatterbox
# and then stalls against 1.6-3 GB of resident CT2 weights that nothing ever
# releases. Reference counted rather than timestamped: cancelling the coroutine
# that awaits a decode does not stop the worker thread running it.
MODEL_TTL = ttl_from_env(os.getenv, "STT_MODEL_TTL", "MODEL_TTL", default=300.0)
_idle_unloader = IdleUnloader(MODEL_TTL, lambda: unload_model(), name="Whisper model")


def acquire_model():
    """Take a reference to the Whisper model, loading it on demand.

    Returns the model object — use the returned value, never the global, or the
    race this exists to close reopens. Loading on demand is what makes idle
    unloading transparent to clients.

    Every call must be paired with exactly one release_model(); prefer
    model_in_use() unless the work outlives the calling frame.
    """
    global _model_refs

    # Outside the reference lock: cancel() takes the unloader's own lock, and
    # _fire() -> unload_model() takes the reference lock, so taking them in the
    # other order here would invert the pair.
    _idle_unloader.cancel()
    with _model_ref_lock:
        model = whisper_model
        if model is None:
            # Released by unload, or never loaded. load_model() sets the global.
            load_model()
            model = whisper_model
            if model is None:
                detail = "Model not available"
                if startup_error:
                    detail += f": {startup_error}"
                raise HTTPException(status_code=503, detail=detail)
        _model_refs += 1
    return model


def release_model():
    """Drop one reference taken by acquire_model(); arm the idle timer at zero."""
    global _model_refs
    with _model_ref_lock:
        if _model_refs > 0:
            _model_refs -= 1
        now_idle = _model_refs == 0
    # Armed outside the lock. With TTL=0 arm() unloads synchronously, and
    # unload_model() takes the same lock — doing this inside the block would
    # self-deadlock on a non-reentrant lock.
    if now_idle:
        _idle_unloader.arm()


@contextmanager
def model_in_use():
    """Hold the Whisper model for the duration of a block."""
    model = acquire_model()
    try:
        yield model
    finally:
        release_model()


class _ModelHold:
    """A model reference that is released at most once.

    acquire_model()/release_model() are a bare counter, so a second release would
    quietly hand back a reference that belongs to another request and let its
    model be unloaded mid-decode. Code that releases from more than one place
    (a stream's generator and its response wrapper) goes through this instead.
    """

    def __init__(self, model):
        self.model = model
        self.adopted = False   # set by the consumer that takes over the release
        self._lock = threading.Lock()
        self._released = False

    def release(self) -> bool:
        """Release the reference; False if it was already released."""
        with self._lock:
            if self._released:
                return False
            self._released = True
        release_model()
        return True


async def _hold_model_async() -> _ModelHold:
    """acquire_model() off the event loop, without leaking the reference on cancel.

    A cold acquire runs load_model() under the reference lock — seconds to minutes
    with a download — so on the loop it froze every live session and /health for
    that long. A worker thread cannot be interrupted, so when this task is
    cancelled while the acquire is still running the thread goes on to take a
    reference nobody will release; hand it back as soon as it lands.
    """
    loop = asyncio.get_running_loop()
    job = _acquire_executor.submit(acquire_model)
    pending = asyncio.wrap_future(job)
    try:
        return _ModelHold(await asyncio.shield(pending))
    except asyncio.CancelledError:
        if job.cancel():
            raise  # still queued behind another acquire: it never ran, no reference exists

        def _give_back(fut):
            if not fut.cancelled() and fut.exception() is None:
                _ModelHold(fut.result()).release()
        pending.add_done_callback(lambda fut: loop.run_in_executor(None, _give_back, fut))
        raise


async def _release_soon(hold, after=None) -> None:
    """Release `hold` off the event loop; safe to await from a `finally` during cancellation.

    release_model() takes the reference lock — held for the whole duration of a
    load — and with STT_MODEL_TTL=0 it also unloads and collects synchronously,
    neither of which belongs on the loop. A cancellation that lands on the await
    here cannot abort the release: it keeps running in its worker thread.

    `after` is a future for a worker thread that is still using the model (a
    stream whose client left mid-decode). The release is deferred until it
    finishes, or a TTL of 0 would unload — and the next request reload — the
    model while that thread is still inside it.
    """
    if hold is None:
        return
    loop = asyncio.get_running_loop()

    def _submit(_fut=None):
        try:
            loop.run_in_executor(None, hold.release)
        except RuntimeError:  # executor already shut down with the loop
            hold.release()

    if after is not None and not after.done():
        after.add_done_callback(_submit)
        return
    try:
        await asyncio.shield(loop.run_in_executor(None, hold.release))
    except asyncio.CancelledError:
        pass


def unload_model() -> dict:
    """Drop the Whisper model and free its memory. Safe to call when unloaded.

    Returns a result dict rather than raising, so the caller decides the status
    code. Refuses while a transcription holds a reference — the alternative is
    freeing memory that a running decode is still reading from.
    """
    global whisper_model, model_loaded, model_warmed, active_device, active_compute_type

    # Before the lock, never inside it: this is the only place both locks are in
    # play at once, and taking them in this order is what keeps the pair from
    # inverting against acquire_model(). Dropping a timer here is safe even when
    # the unload turns out to be refused — release_model() re-arms at zero refs.
    _idle_unloader.cancel()

    with _model_ref_lock:
        if _model_refs > 0:
            return {"unloaded": False, "reason": "busy", "refs": _model_refs}
        if whisper_model is None:
            return {"unloaded": False, "reason": "not_resident", "refs": 0}

        whisper_model = None
        model_loaded = False
        model_warmed = False
        # Nothing is running any more; /health falls back to the preferred runtime,
        # which is what the next load will try first.
        active_device = active_compute_type = None

    # CTranslate2 owns its device memory directly and frees it when the object
    # is finalised, so the collect is what actually returns the VRAM — there is
    # no torch caching allocator in this path for empty_cache() to drain.
    gc.collect()
    logger.info("Whisper model unloaded; memory released")
    return {"unloaded": True, "reason": "ok", "refs": 0}

def ensure_hardware_detected():
    """Run hardware detection once (lazy initialisation)."""
    global device, compute_type
    if device is None:
        device, compute_type = detect_hardware()


def _load_attempts(requested: str, pref_device: str, pref_compute_type: str):
    """The (model_size, device, compute_type) ladder for one load, best first.

    Rebuilt from the PREFERRED runtime on every load, so a fallback taken once
    (a transient OOM, another service briefly holding the VRAM) is not permanent.

    Resource use must never INCREASE down the ladder — responding to a failure by
    asking for more memory cannot help. The two failure modes therefore need
    different responses, told apart by the requested name rather than by the
    exception:
      - a name faster-whisper cannot resolve at all -> retry as a known-good name
        (a size step up is fine, the requested model does not exist);
      - anything else, including a Hub repo id or path -> step DOWN in size, then
        off the GPU. A custom model is valid by construction and is never
        "corrected" to large-v3.
    """
    attempts = [(requested, pref_device, pref_compute_type)]

    if not _is_valid_model_name(requested):
        # Name-resolution safety net: an unknown alias must not brick startup.
        # Only reachable when `requested` is not a real model, so this cannot
        # escalate memory for a working configuration.
        attempts.append((FALLBACK_MODEL_SIZE, pref_device, pref_compute_type))

    if pref_device != "cpu":
        # Smaller multilingual model before abandoning the GPU entirely, but
        # only if it is genuinely smaller than what was asked for.
        # `small` is multilingual — do NOT use distil-large-v3 here, it is
        # English-only and would silently drop German.
        smaller = _model_rank(OOM_FALLBACK_MODEL_SIZE) < _model_rank(requested)
        if smaller:
            attempts.append((OOM_FALLBACK_MODEL_SIZE, pref_device, "int8_float16"))
        attempts.append((requested, "cpu", "int8"))
        if smaller:
            attempts.append((OOM_FALLBACK_MODEL_SIZE, "cpu", "int8"))
    return attempts


def load_model():
    """Load the Whisper model, walking the fallback ladder until one rung loads.

    Callers hold ``_model_ref_lock`` (see acquire_model) so two threads never load
    at once. Sets ``startup_error`` instead of raising when every rung fails.
    """
    global whisper_model, model_loaded, model_size_loaded, startup_error
    global active_device, active_compute_type, _loading

    # Ensure hardware is detected first
    ensure_hardware_detected()

    # CTranslate2 serialises calls per worker; 2 gives live and batch traffic
    # independent slots without doubling activation memory for every request.
    num_workers = max(1, _number(os.getenv("WHISPER_NUM_WORKERS", "2"), 2, int, "WHISPER_NUM_WORKERS"))
    # cpu_threads maps to CT2's intra_threads and num_workers to inter_threads,
    # and CT2 spawns inter_threads model replicas each running intra_threads
    # threads — so os.cpu_count() here would demand num_workers x cpu_count.
    cpu_threads = max(1, (os.cpu_count() or 4) // num_workers)
    requested = configured_model_size()
    attempts = _load_attempts(requested, device, compute_type)

    extra = {"download_root": WHISPER_MODEL_DIR} if WHISPER_MODEL_DIR else {}

    _loading = True
    last_error = None
    try:
        for model_size, dev, ctype in attempts:
            try:
                logger.info(f"Loading {model_size} Whisper model on {dev} with {ctype}...")
                whisper_model = WhisperModel(
                    model_size,
                    device=dev,
                    compute_type=ctype,
                    cpu_threads=cpu_threads if dev == "cpu" else 4,
                    num_workers=num_workers,
                    **extra,
                )
                # What is running now, reported by /health. The preferred
                # `device`/`compute_type` stay untouched so the next load starts
                # from the top of the ladder again.
                active_device, active_compute_type = dev, ctype
                model_size_loaded = model_size
                model_loaded = True
                startup_error = None
                if (model_size, dev, ctype) != attempts[0]:
                    logger.warning(
                        f"Running degraded: loaded {model_size} on {dev}/{ctype} instead of "
                        f"{requested} on {attempts[0][1]}/{attempts[0][2]}; the next load "
                        "tries the preferred configuration again"
                    )
                logger.info(f"Model loaded successfully: {model_size} on {dev}/{ctype}")
                return
            except Exception as e:
                last_error = e
                logger.error(f"Failed to load {model_size} on {dev}/{ctype}: {e}")

        model_loaded = False
        # The category is what a probe may see; the exception (paths, repo ids) and the
        # hint below are for the operator, who has the log.
        startup_error = error_category(last_error)
        hint = ""
        if _flag(os.getenv("HF_HUB_OFFLINE", "")):
            hint = (
                " (HF_HUB_OFFLINE is set: the model must already be in the cache, "
                "or WHISPER_MODEL_SIZE must be a local path)"
            )
        logger.error(
            "All model load attempts failed (%s); service is unhealthy: %s%s",
            startup_error, last_error, hint,
        )
    finally:
        _loading = False


def warm_up_model():
    """Run one throwaway inference so the first real request doesn't pay autotune.

    Uses low-amplitude noise with ``vad_filter=False`` — silence plus VAD short-
    circuits before any kernel runs and would warm nothing. The generator must be
    consumed or no compute happens at all.
    """
    global model_warmed
    model = whisper_model
    if not model_loaded or model is None:
        return
    try:
        t0 = time.monotonic()
        rng = np.random.default_rng(0)
        # cuDNN autotune is keyed on input shape; warm the short and long cases.
        for seconds in (1.0, 5.0):
            noise = (rng.standard_normal(int(seconds * WS_SAMPLE_RATE)) * 1e-3).astype(np.float32)
            segments, _info = model.transcribe(
                noise, language="en", beam_size=1, best_of=1,
                temperature=0.0, without_timestamps=True, vad_filter=False,
            )
            list(segments)
        model_warmed = True
        logger.info(f"Model warm-up complete in {time.monotonic() - t0:.2f}s")
    except Exception as e:
        logger.warning(f"Model warm-up failed (first request will be slower): {e}")


def _validate_task(task: str):
    """400 for a task Whisper does not have; it used to surface as a 500 mid-decode."""
    if task not in ("transcribe", "translate"):
        raise HTTPException(status_code=400, detail=f"Unsupported task {task!r}; use 'transcribe' or 'translate'.")


def _reject_unsupported_translate(task: str):
    """Reject task='translate' on models that cannot translate.

    Whisper's turbo and distil variants were tuned on transcription only; asking
    them to translate silently returns source-language text instead of English.
    A German fine-tune of the turbo model inherits that, so custom ids and paths
    are judged by name too.
    """
    if task != "translate":
        return
    model = model_size_loaded or configured_model_size()
    if not model_can_translate(model):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Model '{model}' does not support translation. "
                "Set WHISPER_MODEL_SIZE to large-v3 (or another multilingual non-turbo model) to use task=translate."
            ),
        )


def _request_language(value):
    """The language a request asks for: the configured default when it names none.

    "auto" is an answer, not an omission, so it still means auto-detect. Locale
    tags ("de-DE") are reduced to the code Whisper accepts, and an unknown one is
    a 400 up front instead of a 500 after the audio was already decoded.
    """
    if value is None or not str(value).strip():
        return STT_DEFAULT_LANGUAGE
    try:
        return normalize_language(value)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"{exc}. Use a Whisper language code such as 'de' or 'en', or 'auto'.")


def _unlink_quiet(path):
    try:
        os.unlink(path)
    except OSError:
        pass


_COPY_CHUNK = 1024 * 1024


async def _spool_upload(upload: UploadFile):
    """Copy an upload to a temp file in chunks; returns ``(path, size)``.

    ``await upload.read()`` with no size pulled the whole file into memory (and
    /transcribe-stream then kept the bytes for the life of the stream). Past
    MAX_UPLOAD_BYTES this answers 413 and leaves nothing behind.
    """
    suffix = os.path.splitext(os.path.basename(upload.filename or "audio.wav"))[1]
    suffix = re.sub(r"[^A-Za-z0-9.]", "", suffix)[:12] or ".wav"
    fd, path = tempfile.mkstemp(suffix=suffix)
    size = 0
    try:
        with os.fdopen(fd, "wb") as out:
            while True:
                chunk = await upload.read(_COPY_CHUNK)
                if not chunk:
                    break
                size += len(chunk)
                if 0 < MAX_UPLOAD_BYTES < size:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Upload is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB (MAX_UPLOAD_MB).",
                    )
                out.write(chunk)
    except BaseException:
        _unlink_quiet(path)
        raise
    return path, size


def _probe_duration_s(path):
    """Container-reported duration in seconds, or None when it cannot be read cheaply."""
    try:
        import av  # a faster-whisper dependency; absent only in stripped test envs
    except ImportError:
        return None
    try:
        with av.open(path) as container:
            if container.duration is not None:
                return container.duration / av.time_base
            for stream in container.streams:
                if stream.type == "audio" and stream.duration and stream.time_base:
                    return float(stream.duration * stream.time_base)
    except Exception:
        pass
    return None


def _check_audio_seconds(seconds):
    """413 when the audio is longer than MAX_AUDIO_SECONDS."""
    if MAX_AUDIO_SECONDS > 0 and seconds is not None and seconds > MAX_AUDIO_SECONDS:
        raise HTTPException(
            status_code=413,
            detail=(
                f"Audio is {seconds / 60:.0f} min long; the limit is "
                f"{MAX_AUDIO_SECONDS / 60:.0f} min (MAX_AUDIO_SECONDS)."
            ),
        )


def _enforce_duration_cap(path):
    """Refuse over-long audio before faster-whisper decodes all of it into memory.

    Header probe only; a file without a usable duration is caught by the check on
    ``info.duration`` after decoding, which is later but still before any segment
    work.
    """
    if MAX_AUDIO_SECONDS > 0:
        _check_audio_seconds(_probe_duration_s(path))


def _transcribe_file(model, path, *, vad_filter, **kwargs):
    """model.transcribe for a file, via the batched pipeline when opted in."""
    # The batched pipeline chunks by VAD; without it, audio past 30 s raises.
    if BATCHED_INFERENCE and BatchedInferencePipeline is not None and vad_filter:
        return BatchedInferencePipeline(model).transcribe(
            path, batch_size=BATCH_SIZE, vad_filter=True, without_timestamps=False, **kwargs
        )
    return model.transcribe(path, vad_filter=vad_filter, **kwargs)


# Add better error handling and logging
@app.post("/transcribe")
async def transcribe_audio(
    # Accept either a single file (legacy) or multiple files (batch)
    audio: UploadFile = File(None),
    audios: List[UploadFile] = File(None),
    task: str = Form("transcribe"),
    # None = the request did not say (STT_DEFAULT_LANGUAGE applies); "auto" = detect.
    language: str = Form(None),
    beam_size: int = Form(5),
    best_of: int = Form(5),
    patience: float = Form(1.0),
    temperature: Union[float, str] = Form("0.0,0.2,0.4,0.6,0.8,1.0"),
    suppress_tokens: str = Form("-1"),
    initial_prompt: str = Form(""),
    condition_on_previous_text: bool = Form(True),
    compression_ratio_threshold: float = Form(2.4),
    no_speech_threshold: float = Form(0.6),
    vad_filter: bool = Form(True),
    vad_threshold: float = Form(0.5),
):
    """Transcribe one or more audio files to text.

    Supports single-file (``audio``) and multi-file (``audios``) uploads.
    Returns per-file results with segments, language, and timing information.
    """
    logger.info(f"Transcription request - task: {task}, language: {language}")

    _validate_task(task)
    _reject_unsupported_translate(task)
    language = _request_language(language)

    # Every upload spooled for this request, removed in the finally below on
    # every path: a 503 from a failed model load used to skip the cleanup and
    # leave the file behind.
    temp_paths: List[str] = []
    try:
        # Normalize file list
        file_list: List[UploadFile] = []
        if audios:
            file_list.extend(audios)
        if audio:
            file_list.append(audio)
        if not file_list:
            raise HTTPException(status_code=400, detail="No audio file(s) provided")

        batch_mode = len(file_list) > 1
        batch_results = []
        total_processing_time = 0.0
        total_audio_duration = 0.0

        # Parse temperature once
        try:
            if isinstance(temperature, str):
                temperature = [float(t) for t in temperature.split(",") if t.strip()]
            else:
                temperature = [float(temperature)]
            if not temperature:
                raise ValueError("empty temperature list")
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail="Invalid temperature value: expected a float or comma-separated floats",
            )

        # Parse suppress_tokens once
        parsed_suppress_tokens = None
        try:
            if isinstance(suppress_tokens, str):
                st = suppress_tokens.strip()
                if st and st != "-1":
                    parsed_suppress_tokens = [int(tok.strip()) for tok in st.split(",") if tok.strip()]
        except ValueError:
            logger.warning(f"Invalid suppress_tokens value '{suppress_tokens}', ignoring.")
            parsed_suppress_tokens = None

        # Process each file sequentially
        for afile in file_list:
            temp_audio_path, saved_bytes = await _spool_upload(afile)
            temp_paths.append(temp_audio_path)
            logger.info(f"Saved audio file: {temp_audio_path}, size: {saved_bytes} bytes")

            logger.info(f"Starting transcription for file {afile.filename} (vad_filter={vad_filter}, vad_threshold={vad_threshold})...")
            start_time = time.time()

            def _do_transcribe(path=temp_audio_path):
                """Run faster-whisper + segment filtering in a worker thread.

                faster-whisper is blocking and the segment generator is consumed
                here, so this whole step runs off the event loop to keep /health
                and concurrent requests responsive during transcription.
                """
                # Before the model is taken (and possibly loaded) for a request
                # that is going to be refused anyway.
                _enforce_duration_cap(path)
                # Hold a reference for the whole call: the segment generator
                # below is lazy, so an unload during iteration would free the
                # model mid-decode.
                with model_in_use() as model:
                    segments, info = _transcribe_file(
                        model,
                        path,
                        task=task,
                        language=language,
                        beam_size=beam_size,
                        best_of=best_of,
                        patience=patience,
                        temperature=temperature,
                        suppress_tokens=parsed_suppress_tokens,
                        initial_prompt=initial_prompt,
                        condition_on_previous_text=condition_on_previous_text,
                        compression_ratio_threshold=compression_ratio_threshold,
                        no_speech_threshold=no_speech_threshold,
                        vad_filter=vad_filter,
                        vad_parameters={
                            "threshold": vad_threshold,
                            "min_speech_duration_ms": 500,
                            "min_silence_duration_ms": 1500,
                            "speech_pad_ms": 300,
                        } if vad_filter else None,
                    )
                    # The container header does not always carry a duration.
                    _check_audio_seconds(info.duration)

                    segs = []
                    full = ""
                    last = ""
                    processed = 0
                    duration = info.duration or 0
                    for segment in segments:
                        processed += 1
                        text = segment.text.strip()
                        seg_duration = segment.end - segment.start

                        if processed % 50 == 0:
                            if duration > 0:
                                progress_pct = min((segment.end / duration) * 100, 100)
                                logger.info(f"Progress: {processed} segments, {segment.end:.1f}s/{duration:.1f}s ({progress_pct:.1f}%)")
                            else:
                                logger.info(f"Progress: {processed} segments, current time: {segment.end:.1f}s")

                        if seg_duration < 0.2 or segment.no_speech_prob > 0.8:
                            continue
                        if text == last or not text:
                            continue
                        segs.append({
                            "start": segment.start,
                            "end": segment.end,
                            "text": text,
                            "avg_logprob": segment.avg_logprob,
                            "no_speech_prob": segment.no_speech_prob,
                        })
                        full += text + " "
                        last = text
                    return segs, full, processed, info

            loop = asyncio.get_running_loop()
            segments_list, full_text, processed_segments, info = await loop.run_in_executor(executor, _do_transcribe)
            total_duration = info.duration or 0

            processing_time = time.time() - start_time
            total_processing_time += processing_time
            total_audio_duration += info.duration or 0
            
            logger.info(f"Transcription completed for {afile.filename}")
            logger.info(f"Final stats: {len(segments_list)} valid segments created from {processed_segments} total segments")
            logger.info(f"Processing time: {processing_time:.2f}s for {total_duration:.1f}s audio")
            if processing_time > 0:
                logger.info(f"Speed: {total_duration/processing_time:.1f}x realtime")

            _unlink_quiet(temp_audio_path)

            batch_results.append({
                "filename": afile.filename,
                "text": full_text.strip(),
                "segments": segments_list,
                "language": info.language,
                "language_probability": info.language_probability,
                "duration": info.duration,
                "task": task,
                "processing_time": processing_time
            })

        # Single file: preserve legacy response shape
        if not batch_mode:
            if not batch_results:
                raise HTTPException(status_code=422, detail="No transcription results (empty or unreadable audio)")
            # avg_logprob can be -inf/NaN on a degenerate segment; the plain
            # JSONResponse rejected it AFTER the whole decode and answered 500.
            return SafeJSONResponse(content=batch_results[0])

        # Batch: new response shape
        if not batch_results:
            raise HTTPException(status_code=422, detail="No transcription results for any provided files")

        batch_response = {
            "batch": True,
            "file_count": len(batch_results),
            "combined_text": " \n".join([r["text"] for r in batch_results]).strip(),
            "results": batch_results,
            "total_processing_time": total_processing_time,
            "total_duration": total_audio_duration,
            "task": task,
            "language": batch_results[0]["language"],
            "language_probability": batch_results[0]["language_probability"],
        }
        return SafeJSONResponse(content=batch_response)

    except HTTPException:
        raise
    except Exception as e:
        req_id = _request_id()
        logger.error("[%s] Transcription error: %s", req_id, e, exc_info=True)
        raise HTTPException(status_code=500, detail=_internal_error("Transcription failed", req_id))
    finally:
        for path in temp_paths:
            _unlink_quiet(path)

class _HeldStreamingResponse(StreamingResponse):
    """StreamingResponse that owns a model reference until its generator adopts it.

    The reference is taken in the handler so a load failure is a real 503 (once an
    SSE body starts the status line is already 200). But an async generator that
    is never started never runs its `finally`: if anything stops Starlette from
    iterating it, the reference would be held for the life of the process, POST
    /unload would answer 409 forever and the idle TTL could never fire again.
    This wrapper is the backstop for exactly that case; a generator that did start
    has taken over and releases for itself.
    """

    def __init__(self, content, hold, tmp_path, **kwargs):
        super().__init__(content, **kwargs)
        self._hold = hold
        self._tmp_path = tmp_path

    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        finally:
            if not self._hold.adopted:
                _unlink_quiet(self._tmp_path)
                await _release_soon(self._hold)
            else:
                # A send that fails (client reset) leaves the generator suspended
                # at a yield with nothing to close it until the garbage collector
                # gets to it. Close it here so its finally, which releases the
                # model, runs now.
                try:
                    await self.body_iterator.aclose()
                except Exception:
                    pass


@app.post("/transcribe-stream")
async def transcribe_audio_stream(
    audio: UploadFile = File(...),
    language: str = Form(None),
    task: str = Form("transcribe"),
    target_language: str = Form("english"),
    beam_size: int = Form(5),
    vad_filter: bool = Form(True),
    vad_threshold: float = Form(0.5),
    no_speech_threshold: float = Form(0.6),
):
    """Stream transcription results via Server-Sent Events as segments are decoded."""
    _validate_task(task)
    _reject_unsupported_translate(task)

    # "auto" (or empty) means auto-detect, as on /transcribe; omitting the field
    # falls back to STT_DEFAULT_LANGUAGE. faster-whisper rejects "auto" itself.
    language = _request_language(language)

    req_id = _request_id()

    # Spool the upload to disk in chunks. The bytes used to be read whole and
    # then stayed referenced by the generator for the entire stream.
    try:
        tmp_file_path, audio_size = await _spool_upload(audio)
        logger.info(f"[{req_id}] Saved temp file for streaming: {tmp_file_path}")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{req_id}] Failed to save temp file: {e}")
        raise HTTPException(status_code=500, detail="Failed to process uploaded file.")

    # The model reference is taken here, and handed to the generator, so a model
    # that cannot load is a 503 — this route used to gate on `model_loaded`, which
    # is False after an idle unload or POST /unload, and so answered 503 for a
    # model that would have reloaded on demand. Probing with acquire+release
    # instead would load the model twice under STT_MODEL_TTL=0 (the release
    # unloads it before the generator asks again). The acquire runs off the event
    # loop: a cold load takes seconds to minutes.
    try:
        await asyncio.to_thread(_enforce_duration_cap, tmp_file_path)
        hold = await _hold_model_async()
    except BaseException:
        _unlink_quiet(tmp_file_path)
        raise

    async def generate_transcription():
        """SSE generator that yields transcription segments as JSON events."""
        # From here on this generator releases the reference, in its finally.
        hold.adopted = True
        effective_target_lang = target_language  # capture outer scope value
        loop = asyncio.get_running_loop()
        # The worker thread currently running for this stream, if any.
        inflight = None

        def _in_worker(fn, *args):
            """Run `fn` on the batch pool, remembering the future so cleanup can wait for it."""
            nonlocal inflight
            fut = loop.run_in_executor(executor, fn, *args)
            # A cancelled await never retrieves the outcome; do it here so a
            # worker error is not reported as "never retrieved" at GC time.
            fut.add_done_callback(lambda f: f.cancelled() or f.exception())
            inflight = fut
            # Shielded: cancelling the task that awaits `fut` would otherwise
            # mark it done while the thread is still running, and the release
            # below would take that for "the worker finished".
            return asyncio.shield(fut)

        try:
            model = hold.model
            logger.info(f"[{req_id}] /transcribe-stream request received: filename={audio.filename}, content_type={audio.content_type}, language={language}, task={task}")
            logger.info(f"[{req_id}] Uploaded audio size: {audio_size} bytes")

            # For translation task, ensure target language is supported
            if task == "translate" and effective_target_lang.lower() not in ["english", "en"]:
                yield f"data: {json.dumps({'warning': f'Translation to {effective_target_lang} not supported, using English'})}\n\n"
                effective_target_lang = "english"

            # Start transcription
            yield f"data: {json.dumps({'status': 'processing', 'task': task})}\n\n"
            logger.info(f"[{req_id}] Starting streaming transcription...")

            # Run transcription in executor to avoid blocking
            def run_transcription():
                """Execute the blocking whisper transcription call inside a thread."""
                return model.transcribe(
                    tmp_file_path,
                    beam_size=beam_size,
                    best_of=beam_size,
                    temperature=(0.0, 0.2, 0.4, 0.6, 0.8),
                    compression_ratio_threshold=2.4,
                    no_speech_threshold=no_speech_threshold,
                    language=language,
                    task=task,
                    vad_filter=vad_filter,
                    vad_parameters={
                        "threshold": vad_threshold,
                        "min_speech_duration_ms": 500,
                        "min_silence_duration_ms": 1500,
                        "speech_pad_ms": 300,
                    } if vad_filter else None,
                )

            segments, info = await _in_worker(run_transcription)
            _check_audio_seconds(info.duration)

            # Send metadata first
            metadata = {
                "language": info.language,
                "language_probability": info.language_probability,
                "duration": info.duration,
                "task": task
            }
            if task == "translate":
                metadata["target_language"] = effective_target_lang

            yield f"data: {json.dumps(clean_json_inf_nan({'metadata': metadata}))}\n\n"

            # Stream segments as they're processed.
            # model.transcribe returns a *lazy* generator: the encoder and
            # decoder run on iteration, not on the call above. Iterating it here
            # would run that work on the event loop and freeze every concurrent
            # live WebSocket session (and /health) for the file's duration, so each
            # step is pulled through the executor instead. The `None` sentinel is
            # required — StopIteration cannot propagate across a Future.
            full_text = ""
            segment_count = 0
            segment_iter = iter(segments)
            i = -1
            while True:
                segment = await _in_worker(next, segment_iter, None)
                if segment is None:
                    break
                i += 1
                segment_count = i + 1
                segment_data = {
                    "segment_id": i,
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text.strip(),
                    "avg_logprob": segment.avg_logprob,
                    "no_speech_prob": segment.no_speech_prob
                }
                full_text += segment.text.strip() + " "

                yield f"data: {json.dumps(clean_json_inf_nan({'segment': segment_data}))}\n\n"

            # Send final result
            final_result = {
                "final_text": full_text.strip(),
                "status": "completed",
                "total_segments": segment_count
            }
            yield f"data: {json.dumps(final_result)}\n\n"

        except Exception as e:
            logger.error("[%s] Streaming transcription error: %s", req_id, e, exc_info=True)
            if isinstance(e, HTTPException):
                message = f"Transcription failed: {e.detail}"  # written for the client
            else:
                message = _internal_error("Transcription failed", req_id)
            error_data = {"error": message, "status": "error"}
            yield f"data: {json.dumps(error_data)}\n\n"
        finally:
            # A client that disconnects mid-stream cancels this generator while a
            # worker thread may still be inside next(), and cancelling the await
            # does not stop that thread. Releasing now would let STT_MODEL_TTL=0
            # unload the model, and the next request load a second copy, under
            # it. So the release waits for the thread. (The temp file can go now:
            # unlinking a file a worker still has open is harmless on POSIX.)
            await _release_soon(hold, after=inflight)
            _unlink_quiet(tmp_file_path)
            logger.info(f"[{req_id}] Cleaned up temp file: {tmp_file_path}")

    return _HeldStreamingResponse(
        generate_transcription(),
        hold,
        tmp_file_path,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )

# --- WebSocket live transcription ---

WS_SAMPLE_RATE = 16000          # required input rate (PCM16 little-endian mono)
# Target length of the audio each interim decode looks at.
#
# Note on cost, because it is easy to get wrong: faster-whisper pads the mel to a
# fixed 3000 frames (30 s) before the encoder, so the ENCODER pass costs the
# same whether this is 8 s or 25 s. What shrinking the window actually saves is
# decoder work — every tick regenerates all the tokens in the window from
# scratch (condition_on_previous_text=False), and that scales with the speech
# inside it. On turbo's 4 decoder layers that is a real but moderate saving,
# not the several-fold win a naive reading suggests.
#
# It is a target, not a fixed tail: the window starts at the last committed
# sentence end (see local_agreement.py) and is cut there once it outgrows this.
# It can stretch to twice this (30 s at most) while no sentence has settled.
#
# 8 s keeps enough left context for punctuation while bounding the token count.
# Raising it back toward 25 s costs less than it appears to and buys context.
WS_WINDOW_S = float(os.getenv("WS_WINDOW_S", "8.0"))
# Floor on how often a partial can be produced. The decode is single-flight, so
# this is a floor and not a schedule — decodes never queue up behind each other.
WS_MIN_NEW_AUDIO_S = float(os.getenv("WS_MIN_NEW_AUDIO_S", "0.5"))
WS_MAX_BUFFER_S = float(os.getenv("WS_MAX_BUFFER_S", "600.0"))
WS_MAX_SESSIONS = _MAX_LIVE_SESSIONS
# Interim hypotheses are greedy and unconditioned, so silence reliably produces
# hallucinated stock phrases. These are the same guards /transcribe already uses.
WS_NO_SPEECH_THRESHOLD = float(os.getenv("WS_NO_SPEECH_THRESHOLD", "0.6"))
WS_MIN_AVG_LOGPROB = float(os.getenv("WS_MIN_AVG_LOGPROB", "-1.0"))
# Interim decode failures are otherwise invisible until the final decode, so the
# client is told after this many in a row (the session itself stays alive).
WS_MAX_INTERIM_ERRORS = max(1, _number(os.getenv("WS_MAX_INTERIM_ERRORS", "3"), 3, int, "WS_MAX_INTERIM_ERRORS"))
# A connected socket that sends nothing still occupies one of WS_MAX_SESSIONS
# slots (and holds the model resident). 0 disables the timeout.
WS_IDLE_TIMEOUT_S = _number(os.getenv("WS_IDLE_TIMEOUT_S", "60"), 60.0, float, "WS_IDLE_TIMEOUT_S")
# Without a fixed language every 0.5 s tick re-runs language detection over the
# window. Once a decode detects one at least this confidently it is kept for the
# rest of the session (unless the client fixed one).
WS_LANGUAGE_LOCK_PROB = _number(os.getenv("WS_LANGUAGE_LOCK_PROB", "0.8"), 0.8, float, "WS_LANGUAGE_LOCK_PROB")
# Close code for the idle timeout: private-use range, mirrors HTTP 408.
WS_CLOSE_IDLE = 4408

_live_sessions = 0


def _error_frame(code: str, message: str) -> dict:
    """A WebSocket error message. `error` duplicates `message` because the shipped
    frontend (and older handshake errors) read that key."""
    return {"type": "error", "code": code, "message": message, "error": message}


def _ws_language(raw):
    """Language for a session config frame: the default when blank, None for "auto".

    Raises ValueError for a code Whisper does not know.
    """
    if raw is None or not str(raw).strip():
        return STT_DEFAULT_LANGUAGE
    return normalize_language(raw)


def _ws_decode(audio: np.ndarray, language, offset_s: float = 0.0):
    """Greedy low-latency decode of a float32 16 kHz buffer (runs in the executor).

    Returns ``(words, info)``: words on the session timeline, ``offset_s`` being
    where this buffer starts on it. Segments failing the no-speech /
    average-logprob guards are dropped: without them a hallucinated phrase repeats
    across consecutive windows and the agreement check *promotes it to confirmed*
    precisely because it repeats.

    Word timestamps are what lets consecutive windows be compared at all (see
    local_agreement.py). Timestamp tokens stay ON: with them off faster-whisper
    seeks to the last word's end and decodes the remainder again, a second full
    encoder pass on every tick.
    """
    with model_in_use() as model:
        segments, info = model.transcribe(
            audio,
            language=language,
            beam_size=1,
            best_of=1,
            # A single temperature cannot escape a repetition loop; one fallback
            # step costs nothing on clean audio because it is only used on failure.
            temperature=(0.0, 0.2),
            condition_on_previous_text=False,
            word_timestamps=True,
            no_speech_threshold=WS_NO_SPEECH_THRESHOLD,
            vad_filter=True,
        )
        # The generator is lazy — it must be consumed inside the block, or the
        # reference is released before any decoding actually happens.
        kept = [
            s for s in segments
            if s.no_speech_prob < WS_NO_SPEECH_THRESHOLD
            and s.avg_logprob > WS_MIN_AVG_LOGPROB
            and s.text.strip()
        ]
    return words_from_segments(kept, offset_s, spaced=info.language not in UNSPACED_LANGUAGES), info


def _ws_final_decode(audio: np.ndarray, language):
    """Accurate decode of the whole session buffer, used once at end of stream.

    The interim path is deliberately greedy; reusing it for the final made the
    live transcript measurably worse than uploading the same audio as a file.
    """
    with model_in_use() as model:
        segments, info = model.transcribe(
            audio,
            language=language,
            beam_size=5,
            best_of=5,
            temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
            condition_on_previous_text=True,
            compression_ratio_threshold=2.4,
            no_speech_threshold=WS_NO_SPEECH_THRESHOLD,
            vad_filter=True,
        )
        parts = [
            s.text.strip() for s in segments
            if s.no_speech_prob < 0.8 and s.text.strip()
        ]
    return " ".join(parts).strip(), info


@app.websocket("/ws/transcribe")
async def websocket_transcribe(websocket: WebSocket):
    """Live transcription over WebSocket.

    Protocol:
    - Optional first text frame: JSON config, e.g. {"language": "de"}
      ("auto" = auto-detect; omitted = STT_DEFAULT_LANGUAGE, else auto-detect;
      locale tags such as "de-DE" are reduced to "de"; an unknown code gets an
      error frame and the previous language stays in force).
    - Binary frames: raw PCM16 little-endian mono audio at 16 kHz.
    - Text frame {"event": "stop"}: finish — the full buffer is decoded once
      more and a {"type": "final", ...} message is sent before closing.
    - A socket that sends nothing for WS_IDLE_TIMEOUT_S is closed (code 4408).

    Server messages: {"type": "partial", "confirmed": str, "pending": str}
    after each interim decode. `confirmed` is the whole session so far, made of
    words two consecutive decodes agreed on (LocalAgreement, on absolute word
    times, so it keeps working after the decode window slides); it only ever
    grows. `pending` is the tentative tail. Then {"type": "final", "text",
    "language", "duration"}. Partials also carry "decode_ms", "lag_ms" and
    "pending_seconds" so latency is observable from the client without a
    profiler. {"type": "error", "code", "message"} reports what would otherwise
    only be logged: a rejected language, or WS_MAX_INTERIM_ERRORS decodes in a
    row failing; the session stays open.

    Ingest and decode are decoupled: the receive loop only ever appends audio,
    and at most one decode is in flight at a time. When a decode is slower than
    realtime the audio that arrived meanwhile is *skipped over* rather than
    queued, so lag stays bounded by one decode instead of growing without limit.
    """
    global _live_sessions

    # Names this session in the log and in the errors it sends (see _internal_error).
    session_id = _request_id()

    # Browsers do not apply the same-origin policy to WebSockets and Starlette's
    # CORSMiddleware does not run on them, so the origin check for this endpoint is
    # OriginGuardMiddleware: a handshake carrying a foreign Origin never reaches this
    # function (it is closed with 1008 before accept()).
    await websocket.accept()

    # Before touching the model: a refused session must not trigger a load. No
    # await between the check and the increment, so concurrent handshakes cannot
    # both pass it.
    if _live_sessions >= WS_MAX_SESSIONS:
        await websocket.send_json(_error_frame(
            "too_many_sessions", f"Too many live sessions (limit {WS_MAX_SESSIONS})"))
        await websocket.close(code=1013)  # try again later
        return
    _live_sessions += 1

    language = STT_DEFAULT_LANGUAGE   # fixed by the client or the default; None = auto-detect
    locked_language = None            # detected once with confidence; used only while `language` is None
    language_epoch = 0                # bumped on a language change so a stale decode is discarded
    consecutive_errors = 0
    chunks: list[np.ndarray] = []
    buffered_samples = 0        # samples currently held in `chunks`
    received_samples = 0        # monotonic; never decremented by the rolling window
    decoded_at_samples = 0
    # A non-positive WS_WINDOW_S would ask every decode for zero seconds of audio.
    agreement = LocalAgreement(max(1.0, WS_WINDOW_S), min(30.0, 2.0 * max(1.0, WS_WINDOW_S)))
    rolled = False
    max_samples = int(WS_MAX_BUFFER_S * WS_SAMPLE_RATE)
    min_new_samples = int(WS_MIN_NEW_AUDIO_S * WS_SAMPLE_RATE)
    idle_timeout = WS_IDLE_TIMEOUT_S if WS_IDLE_TIMEOUT_S > 0 else None
    loop = asyncio.get_running_loop()
    send_lock = asyncio.Lock()  # the decode task and the receive loop both send
    decode_task: asyncio.Task | None = None
    hold = None

    async def _send(payload: dict):
        """Serialise sends — the ingest loop and the decode task share the socket."""
        async with send_lock:
            await websocket.send_json(clean_json_inf_nan(payload))

    def _tail(n_samples: int) -> np.ndarray:
        """Concatenate only the most recent `n_samples`.

        Copying the whole buffer here used to mean a 38 MB memcpy on the event
        loop at the 10-minute cap, ~96% of which the window slice then threw away.
        """
        if n_samples <= 0:
            return np.empty(0, dtype=np.float32)
        collected: list[np.ndarray] = []
        got = 0
        for chunk in reversed(chunks):
            collected.append(chunk)
            got += len(chunk)
            if got >= n_samples:
                break
        if not collected:
            return np.empty(0, dtype=np.float32)
        buf = np.concatenate(list(reversed(collected)))
        return buf[-n_samples:] if len(buf) > n_samples else buf

    async def _interim_decode():
        """One interim decode over the current window; sends a single partial."""
        nonlocal consecutive_errors, locked_language
        started = time.monotonic()
        # Snapshot before the await: audio keeps arriving while the decode runs.
        covered_samples = received_samples
        now_s = covered_samples / WS_SAMPLE_RATE
        window = _tail(int(round((now_s - agreement.window_start(now_s)) * WS_SAMPLE_RATE)))
        if not len(window):
            return
        # The buffer may hold less than the window asks for (rolled or short
        # session), so the offset comes from what was actually cut.
        window_start_s = (covered_samples - len(window)) / WS_SAMPLE_RATE
        epoch = language_epoch
        try:
            words, info = await loop.run_in_executor(
                rt_executor, _ws_decode, window, language or locked_language, window_start_s)
        except asyncio.CancelledError:
            raise
        except Exception as decode_err:
            consecutive_errors += 1
            logger.warning(
                "[%s] WS interim decode failed (%d in a row): %s", session_id, consecutive_errors, decode_err,
                exc_info=not isinstance(decode_err, HTTPException),
            )
            # What the client may see: an HTTPException's detail was written for it,
            # anything else is internal and stays in the log under the session id.
            if isinstance(decode_err, HTTPException):
                detail = decode_err.detail
            else:
                detail = f"internal error. Request id: {session_id}."
            # Once per streak, not per tick: a persistent failure would otherwise
            # flood the client every half second.
            if consecutive_errors == WS_MAX_INTERIM_ERRORS:
                try:
                    await _send(_error_frame(
                        "decode_failed",
                        f"Live decoding failed {consecutive_errors} times in a row: {detail}"))
                except Exception as send_err:
                    logger.debug(f"WS error frame not delivered (client likely gone): {send_err}")
            return
        consecutive_errors = 0
        if epoch != language_epoch:
            return  # the client changed language while this was decoding

        decode_ms = (time.monotonic() - started) * 1000.0
        if not words:
            # Silence or an all-filtered window: keep the last partial on screen
            # rather than blanking the panel the user is reading.
            return

        if (
            language is None
            and locked_language is None
            and info.language
            and (info.language_probability or 0.0) >= WS_LANGUAGE_LOCK_PROB
        ):
            locked_language = info.language

        agreement.update(words, now_s)
        # A send failure here (client vanished mid-decode) would otherwise
        # surface as an unretrieved task exception at GC time rather than
        # ending the session; the receive loop notices the disconnect anyway.
        try:
            await _send({
                "type": "partial",
                "confirmed": agreement.confirmed_text,
                "pending": agreement.pending_text,
                "language": language or locked_language,
                "buffered_seconds": round(received_samples / WS_SAMPLE_RATE, 2),
                # Audio received but not yet reflected in this partial.
                "pending_seconds": round(max(0, received_samples - covered_samples) / WS_SAMPLE_RATE, 2),
                "decode_ms": round(decode_ms, 1),
                "lag_ms": round((time.monotonic() - started) * 1000.0, 1),
            })
        except Exception as send_err:
            logger.debug(f"WS partial send failed (client likely gone): {send_err}")

    try:
        # Load now if an unload released the model, so a dead session fails at
        # the handshake instead of at the first audio frame. The reference is
        # kept for the whole session rather than released after a probe: with
        # STT_MODEL_TTL=0 a probe's release unloads the model at once, and the
        # first decode then loads it a second time. Off the event loop — a cold
        # load runs for seconds to minutes and froze every other session.
        try:
            hold = await _hold_model_async()
        except HTTPException as exc:
            await websocket.send_json(_error_frame("model_unavailable", str(exc.detail)))
            await websocket.close(code=1011)
            return

        while True:
            try:
                message = await asyncio.wait_for(websocket.receive(), timeout=idle_timeout)
            except asyncio.TimeoutError:
                await _send(_error_frame(
                    "idle_timeout", f"No data received for {WS_IDLE_TIMEOUT_S:.0f}s; closing the session"))
                await websocket.close(code=WS_CLOSE_IDLE, reason="idle timeout")
                return
            if message.get("type") == "websocket.disconnect":
                return

            if message.get("bytes") is not None:
                raw = message["bytes"]
                # A truncated frame would make frombuffer raise and kill the
                # whole session; drop the odd trailing byte instead.
                if len(raw) % 2:
                    raw = raw[:-1]
                samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                if len(samples):
                    chunks.append(samples)
                    buffered_samples += len(samples)
                    received_samples += len(samples)
                    # Roll the oldest audio out instead of refusing new audio.
                    # Truncating the newest froze `total_samples`, so the decode
                    # gate could never fire again and the session went silently
                    # dead at 10 minutes while the client kept uploading.
                    while buffered_samples > max_samples and len(chunks) > 1:
                        buffered_samples -= len(chunks.pop(0))
                        if not rolled:
                            rolled = True
                            await _send({
                                "type": "warning",
                                "code": "buffer_rolled",
                                "message": (
                                    f"Session exceeded {WS_MAX_BUFFER_S:.0f}s; "
                                    "the final transcript covers only the most recent audio."
                                ),
                            })
            elif message.get("text") is not None:
                try:
                    control = json.loads(message["text"])
                except json.JSONDecodeError:
                    continue
                if not isinstance(control, dict):
                    continue
                if control.get("event") == "stop":
                    break
                # Only touch the language when the frame actually carries the
                # key. This used to run unconditionally, so ANY control frame
                # that omitted it — a keepalive, a future event type — reset a
                # German session to auto-detect and never told the caller.
                # The browser only ever sends {language} then {event:"stop"}, so
                # it never tripped; an API client sending anything else did.
                if "language" in control:
                    try:
                        requested = _ws_language(control["language"])
                    except ValueError as exc:
                        # Say so, and keep decoding in the previous language: the
                        # alternative, silently auto-detecting, is how a German
                        # session ends up transcribed as English.
                        await _send(_error_frame(
                            "invalid_language",
                            f"{exc}. Use a Whisper language code such as 'de' or 'en', or 'auto'."))
                    else:
                        if requested != language:
                            language = requested
                            locked_language = None
                            language_epoch += 1
                            # The tentative tail was decoded in the old language.
                            agreement.drop_pending()

            # Single-flight interim decode. While one is running the loop keeps
            # draining the socket, so frames are never left queued in the
            # transport; the next decode simply starts from the newest audio.
            if (
                (decode_task is None or decode_task.done())
                and received_samples - decoded_at_samples >= min_new_samples
            ):
                decoded_at_samples = received_samples
                decode_task = asyncio.create_task(_interim_decode())

        # Stop requested. Cancelling only detaches the task so it cannot emit a
        # partial after the final — the worker thread keeps running its decode
        # to completion, because a thread in an executor cannot be interrupted.
        # That is why the final runs on the batch pool below: waiting for the
        # abandoned interim to free a realtime slot would delay every other
        # live session's partials.
        if decode_task is not None and not decode_task.done():
            decode_task.cancel()
            try:
                await decode_task
            except asyncio.CancelledError:
                # Expected: we just cancelled it. Do not let this be mistaken
                # for cancellation of this handler.
                pass
            except Exception:
                pass
            decode_task = None

        if buffered_samples:
            buffer = _tail(buffered_samples)
            # Batch pool: this is a beam-search decode over the whole session
            # buffer and has no business holding a realtime slot. The language
            # detected during the session is reused, so the final cannot come
            # back in a different language than the partials the user watched.
            text, info = await loop.run_in_executor(
                executor, _ws_final_decode, buffer, language or locked_language)
            await _send({
                "type": "final",
                "text": text,
                "language": info.language,
                "language_probability": info.language_probability,
                # Duration of the audio this text actually covers. When the
                # rolling window has dropped older audio the two differ, and
                # reporting the session total here made the transcript look
                # like it was missing content rather than bounded on purpose.
                "duration": round(buffered_samples / WS_SAMPLE_RATE, 2),
                "received_duration": round(received_samples / WS_SAMPLE_RATE, 2),
                "truncated": rolled,
            })
        else:
            await _send({"type": "final", "text": "", "language": None, "duration": 0.0})
        await websocket.close()
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error("[%s] WebSocket transcription error: %s", session_id, e, exc_info=True)
        try:
            await _send(_error_frame("internal_error", _internal_error("The live session failed", session_id)))
            await websocket.close(code=1011)
        except Exception:
            pass
    finally:
        # A client that vanishes mid-decode must not leave the task holding a
        # model slot for the rest of the decode.
        if decode_task is not None and not decode_task.done():
            decode_task.cancel()
        _live_sessions -= 1
        await _release_soon(hold)


@app.get("/health", response_class=SafeJSONResponse)
async def health_check():
    """Liveness for health monitoring. Never loads or waits on the model.

    Distinguishes two states that both mean "no model in memory" but need
    opposite handling:

    - **not resident** — the model was unloaded to free VRAM, or has not been
      loaded yet. The service is fine and the next request will load it, so
      this returns 200. Returning 503 here would make an idle container go
      `unhealthy` under Docker's `curl -f` check and show as down in the UI.
    - **broken** — every load attempt failed. That is a real outage: 503.

    Whether the service can serve *right now* is /ready's question.
    """
    current_model = model_size_loaded or configured_model_size()
    resident = whisper_model is not None
    # The runtime actually in use while a model is resident; otherwise what the
    # next load will try first. `device`/`compute_type` are only ever the
    # preferred pair, so a CPU fallback must be reported from the active one.
    running_device = active_device or device
    running_compute_type = active_compute_type or compute_type
    body = {
        "status": "error" if startup_error else "ok",
        # Kept for API compatibility: existing clients read model_loaded.
        "model_loaded": model_loaded,
        "model_resident": resident,
        "can_load": startup_error is None,
        "model_warmed": model_warmed,
        "device": running_device,
        "compute_type": running_compute_type,
        "preferred_device": device,
        "preferred_compute_type": compute_type,
        # True while the resident model runs on a fallback rung (smaller model or
        # CPU); the next load starts from the preferred configuration again.
        "degraded": bool(resident and active_device is not None and (
            (active_device, active_compute_type) != (device, compute_type)
            or model_size_loaded != configured_model_size()
        )),
        "model_size": _public_model_name(current_model),
        "multilingual": is_multilingual(current_model),
        "live_sessions": _live_sessions,
        # Outstanding references. POST /unload refuses while this is non-zero.
        "model_refs": _model_refs,
        # Seconds idle before the model is released (0 = immediately, -1 = never).
        "model_ttl_seconds": MODEL_TTL,
        "cuda_available": torch.cuda.is_available()
    }
    if startup_error:
        body["startup_error"] = startup_error
        return SafeJSONResponse(content=body, status_code=503)
    return body


@app.get("/ready", response_class=SafeJSONResponse)
async def ready():
    """Readiness: can this service take a transcription request now?

    200 when the model is resident, or has been unloaded (idle TTL, POST /unload)
    and can be reloaded on demand. 503 with a JSON `reason` while a load is
    running (`loading`) and after every load attempt failed (`load_failed`).

    Like /health this only reads state and never triggers or waits for a load —
    an orchestrator polling it must not be what keeps the model resident, and a
    poll during a multi-minute download has to answer at once.
    """
    resident = whisper_model is not None
    body = {
        "ready": True,
        "reason": "resident",
        "model_resident": resident,
        "model_size": _public_model_name(model_size_loaded or configured_model_size()),
        "device": active_device or device,
    }
    if _loading or (_preload_pending and not resident):
        body.update(ready=False, reason="loading")
        return SafeJSONResponse(content=body, status_code=503, headers={"Retry-After": "5"})
    if startup_error:
        body.update(ready=False, reason="load_failed", detail=startup_error)
        return SafeJSONResponse(content=body, status_code=503)
    if not resident:
        body["reason"] = "unloaded"
    return body


@app.get("/info", response_class=SafeJSONResponse)
async def service_info():
    """Return detailed service and GPU information."""
    return {
        "service": "STT Service",
        "device": active_device or device,
        "compute_type": active_compute_type or compute_type,
        "model_loaded": model_loaded,
        "model_size": _public_model_name(model_size_loaded or configured_model_size()),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    }

@app.post("/unload")
async def unload():
    """Release the model and its memory now, without stopping the container.

    The idle TTL handles the common case; this is for the deliberate one — you
    are about to run something else on the same GPU and want the VRAM back
    immediately. The next request reloads transparently, so this is safe to call
    at any time.

    Returns 200 when the model was released or was already unloaded, and **409
    when a transcription is in flight** — freeing memory a running decode is
    still reading would crash the worker, so the caller is told to retry instead.
    """
    # In a worker thread: unload_model() takes the reference lock, which a load
    # in progress holds for its whole duration.
    result = await asyncio.to_thread(unload_model)
    if result["reason"] == "busy":
        return SafeJSONResponse(
            status_code=409,
            content={
                "detail": "Model is in use; retry when idle",
                "model_refs": result["refs"],
                **result,
            },
        )
    return SafeJSONResponse(content={
        "model_resident": whisper_model is not None,
        **result,
    })


@app.get("/models")
async def available_models():
    """List Whisper model variants with size and multilingual capability."""
    return {
        "available_models": [
            {"name": "tiny",            "multilingual": True,  "size_mb": 75},
            {"name": "tiny.en",         "multilingual": False, "size_mb": 75},
            {"name": "base",            "multilingual": True,  "size_mb": 145},
            {"name": "base.en",         "multilingual": False, "size_mb": 145},
            {"name": "small",           "multilingual": True,  "size_mb": 490},
            {"name": "small.en",        "multilingual": False, "size_mb": 490},
            {"name": "medium",          "multilingual": True,  "size_mb": 1500},
            {"name": "medium.en",       "multilingual": False, "size_mb": 1500},
            {"name": "large-v1",        "multilingual": True,  "size_mb": 2900},
            {"name": "large-v2",        "multilingual": True,  "size_mb": 2900},
            {"name": "large-v3",        "multilingual": True,  "size_mb": 3100},
            {"name": "large-v3-turbo",  "multilingual": True,  "size_mb": 1600,
             "note": "Default. 4 decoder layers vs large-v3's 32 — fastest multilingual option; cannot translate"},
            {"name": "distil-large-v3", "multilingual": False, "size_mb": 1500, "note": "English-only, fastest"},
            {"name": "distil-large-v3.5", "multilingual": False, "size_mb": 1500, "note": "English-only; needs faster-whisper >= 1.2"},
        ],
        "current_model": _public_model_name(model_size_loaded or configured_model_size()),
        "supported_languages": list(WHISPER_LANGUAGES),
        # WHISPER_MODEL_SIZE also takes a Hugging Face repo id or a local path
        # of a CTranslate2 export, e.g. a German fine-tune (see README).
        "custom_models": True,
    }

@app.get("/tasks")
async def available_tasks():
    """Describe the supported transcription tasks (transcribe / translate)."""
    return {
        "available_tasks": [
            {
                "task": "transcribe",
                "description": "Transcribe audio to text in the original language",
                "parameters": {
                    "language": "Optional: Source language code (auto-detected if not provided)",
                    "beam_size": "Search beam size (1-5, higher = better quality, slower)",
                    "best_of": "Number of candidates to consider (1-5)"
                }
            },
            {
                "task": "translate",
                "description": "Transcribe and translate audio to English",
                "parameters": {
                    "language": "Optional: Source language code",
                    "target_language": "Target language (currently only 'english' supported)",
                    "beam_size": "Search beam size (1-5)",
                    "best_of": "Number of candidates to consider (1-5)"
                },
                "limitations": [
                    "Translation is only available to English",
                    "Whisper model limitation, not service limitation"
                ]
            }
        ],
        "default_task": "transcribe",
        "default_target_language": "english",
        "streaming_available": True
    }

@app.post("/detect_language")
async def detect_language(
    file: UploadFile = File(...),
    duration_limit: float = 30.0,
):
    """Quickly detect the spoken language and return a sample transcript."""
    # No `model_loaded` pre-check: after an idle unload it is False for a model
    # that reloads on demand. `model_in_use()` inside _detect() does the load
    # under the reference lock and raises 503 only if it genuinely cannot.
    temp_audio_path = None
    try:
        temp_audio_path, _saved_bytes = await _spool_upload(file)

        logger.info(f"Detecting language for: {file.filename}")
        start_time = time.time()
        
        # Use transcribe with short duration for quick language detection.
        # Runs in a worker thread so the event loop stays responsive.
        def _detect():
            """Blocking language-detection transcription, run off the event loop."""
            # faster-whisper decodes the whole file up front, so trim long
            # uploads to duration_limit seconds for fast detection.
            detect_path = temp_audio_path
            trimmed_path = None
            if duration_limit and duration_limit > 0:
                try:
                    import librosa
                    import soundfile as sf_lib
                    audio_data, sr = librosa.load(temp_audio_path, sr=None, mono=True, duration=duration_limit)
                    if len(audio_data) > 0:
                        trimmed_fd, trimmed_path = tempfile.mkstemp(suffix=".wav")
                        os.close(trimmed_fd)
                        sf_lib.write(trimmed_path, audio_data, sr)
                        detect_path = trimmed_path
                except Exception as trim_err:
                    logger.warning(f"Could not trim audio for language detection: {trim_err}")

            if detect_path == temp_audio_path:
                # Trimming failed, so the whole upload is about to be decoded.
                _enforce_duration_cap(detect_path)

            try:
                # Lazy generator again: the reference must outlive the loop that
                # consumes it, not just the transcribe() call.
                with model_in_use() as model:
                    segments, info = model.transcribe(
                        detect_path,
                        task="transcribe",
                        language=None,  # Auto-detect
                        beam_size=1,    # Fastest setting
                        best_of=1,      # Fastest setting
                        vad_filter=True,
                        condition_on_previous_text=False,
                        # Only process first part for speed
                        vad_parameters={
                            "threshold": 0.5,
                            "min_speech_duration_ms": 250,
                            "min_silence_duration_ms": 500,
                        }
                    )
                    sample = ""
                    count = 0
                    for segment in segments:
                        if count >= 3:  # Only need a few segments for detection
                            break
                        if segment.no_speech_prob < 0.8:
                            sample += segment.text.strip() + " "
                            count += 1
                return sample, info
            finally:
                if trimmed_path and os.path.exists(trimmed_path):
                    try:
                        os.unlink(trimmed_path)
                    except OSError:
                        pass

        loop = asyncio.get_running_loop()
        sample_text, info = await loop.run_in_executor(executor, _detect)

        processing_time = time.time() - start_time
        
        logger.info(f"Language detection completed in {processing_time:.2f}s: {info.language} ({info.language_probability:.2f})")
        
        return {
            "detected_language": info.language,
            "language_probability": info.language_probability,
            "sample_text": sample_text.strip(),
            "processing_time": processing_time,
            "audio_duration": info.duration
        }
        
    except HTTPException:
        # A 503 from model_in_use() must stay a 503; the blanket handler below
        # would relabel "no model available" as an internal error.
        raise
    except Exception as e:
        req_id = _request_id()
        logger.error("[%s] Language detection failed: %s", req_id, e, exc_info=True)
        raise HTTPException(status_code=500, detail=_internal_error("Language detection failed", req_id))
    finally:
        if temp_audio_path:
            _unlink_quiet(temp_audio_path)
