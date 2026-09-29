"""Parakeet-TDT speech-to-text service.

Wraps NVIDIA's `parakeet-tdt-0.6b-v3` (FastConformer-TDT) — a fast multilingual
ASR model covering 25 European languages (incl. German) with automatic language
detection. Exposes the project's native `stt-form-v1` contract (`/transcribe`
with segment timestamps) plus an OpenAI-compatible `/v1/audio/transcriptions`
endpoint so external tools can reuse it. Supports CUDA, ROCm, and CPU.
"""

import os
import time
import uuid
import shutil
import asyncio
import logging
from contextlib import AsyncExitStack, asynccontextmanager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import torch
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from body_limit import BodyLimitMiddleware
from model_lifecycle import ModelSlot, RequestGate, ttl_from_env
from nemo_common import (
    MIB, MULTIPART_SLACK, env_flag, env_number, filter_transcribe_kwargs,
    is_cuda_oom, load_nemo_model, matmul_precision_from_env, move_to_cpu, prepared_upload,
    public_model_name, runtime_versions, transcribe_results, tune_for_inference,
)
from origin_guard import OriginGuardMiddleware, parse_allowed_origins
from transcription import parse_hypothesis as _parse_hypothesis

# Upload bounds. The gateway limits what it forwards, but a direct caller must not
# be able to fill the disk, and NeMo encodes a file in ONE pass: past ~24 minutes
# (full attention) parakeet-tdt-0.6b-v3 runs out of GPU memory and the request
# used to die as a generic 500.
MAX_UPLOAD_BYTES = int(env_number("MAX_UPLOAD_MB", 200.0, minimum=0.001) * MIB)
MAX_AUDIO_SECONDS = env_number("NEMO_MAX_AUDIO_S", 1500.0, minimum=0.0)  # 0 = unlimited

# TF32 matmuls are what NVIDIA's transcribe script runs with. bf16 halves the
# weight memory and is faster still, but German WER has not been re-measured
# under it, so it stays opt-in.
MATMUL_PRECISION = matmul_precision_from_env()
USE_BF16 = env_flag("NEMO_BF16")


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    """Pre-load the model in the background so the first request is fast.

    Background, not before ``yield``: a first start downloads ~2.5 GB, and doing
    that before the server accepts connections leaves the port closed (and
    /health unreachable) for minutes. /health answers straight away, /ready is
    503 "loading" until the model is in.
    """
    global _preload_pending
    preload = None
    if MODEL_TTL == 0:
        # "Release it the moment nothing is using it" — preloading ~3 GB only to
        # drop it on the first release is work with no beneficiary.
        logger.info("Preload skipped: ASR_MODEL_TTL=0 unloads on every idle")
    else:
        _preload_pending = True
        preload = asyncio.create_task(_startup_preload())
    try:
        yield
    finally:
        if preload is not None:
            preload.cancel()


async def _startup_preload():
    global _preload_pending
    try:
        await asyncio.to_thread(get_model)
    except Exception as e:
        logger.warning(f"Could not preload model: {e}")
    finally:
        _preload_pending = False


app = FastAPI(
    title="Parakeet-ASR Service",
    description="Fast multilingual Speech-to-Text using NVIDIA Parakeet-TDT",
    lifespan=_lifespan,
)

# Unset or empty means no CORS headers at all (it used to mean "*"); "*" only when
# it is written down (and logged); otherwise an explicit list. origin_guard.py.
allowed_origins = parse_allowed_origins(os.getenv("ALLOWED_ORIGINS", ""))
allow_credentials = os.getenv("ALLOW_CREDENTIALS", "false").strip().lower() in {"1", "true", "yes", "on"}
if "*" in allowed_origins and allow_credentials:
    allow_credentials = False

# Request-body limits (body_limit.py): the declared Content-Length AND the bytes
# that actually arrive are checked, so a chunked upload is cut off at the limit too,
# before Starlette has spooled it to disk. A single-file route carries one upload
# plus the small text fields (64 KiB of slack); the batch route carries up to
# _BATCH_BODY_FILES uploads, each of which is still bounded on its own by
# spool_upload; no other route takes a body of any size.
_SINGLE_UPLOAD_PATHS = frozenset({"/transcribe", "/v1/audio/transcriptions", "/detect_language"})
_BATCH_BODY_FILES = 8


def _body_limit(path: str) -> int:
    """Largest request body (bytes) accepted at *path*."""
    if path in _SINGLE_UPLOAD_PATHS:
        return MAX_UPLOAD_BYTES + MULTIPART_SLACK + 64 * 1024
    if path == "/transcribe-batch":
        return _BATCH_BODY_FILES * (MAX_UPLOAD_BYTES + 64 * 1024) + MULTIPART_SLACK
    return MIB


# Each add_middleware wraps what was added before it: the body limit is innermost,
# the origin guard (403 for a state-changing request with a foreign Origin header;
# no Origin, e.g. the gateway or curl, is not affected) comes next, and CORS is
# outermost so a 403/413 still carries the CORS headers a listed origin needs in
# order to read it. CORS is only added when origins are configured.
app.add_middleware(
    BodyLimitMiddleware,
    limit_for=_body_limit,
    hint=f"The upload limit is {MAX_UPLOAD_BYTES / MIB:.3g} MB per file (MAX_UPLOAD_MB).",
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

# Global model state. ROCm/HIP also reports through torch.cuda in PyTorch.
if torch.cuda.is_available():
    device = "cuda"
elif os.getenv("ROCR_VISIBLE_DEVICES") or os.getenv("HIP_VISIBLE_DEVICES"):
    device = "cuda"
else:
    device = "cpu"

MODEL_NAME = os.getenv("PARAKEET_ASR_MODEL", "nvidia/parakeet-tdt-0.6b-v3")

asr_model = None
model_loaded = False
compute_dtype = "float32"
# A startup preload has been scheduled and has not finished: /ready reports
# "loading" for that window instead of "ready, never loaded".
_preload_pending = False

_FFMPEG = shutil.which("ffmpeg")

# NeMo's transcribe() defaults spin up DataLoader worker processes and print a
# tqdm bar per call. At one-file-per-request granularity that setup costs more
# than the inference; a bounded gate keeps concurrent requests from multiplying
# peak VRAM on the single shared model.
_NEMO_RUNTIME_KWARGS = {"batch_size": 1, "num_workers": 0, "verbose": False}
_NEMO_MAX_BATCH = max(1, env_number("ASR_MAX_BATCH", 8, cast=int, minimum=1))

# Admission (model_lifecycle.RequestGate). ASR_MAX_CONCURRENCY forward passes run at
# once. Every request that got past this point used to have spooled its upload and
# converted the audio, and then waited for the ones ahead of it with no limit on
# either the number or the time, so a burst of uploads piled up temp files and every
# client hung. Now at most ASR_MAX_CONCURRENCY + ASR_MAX_QUEUE requests are admitted
# (the next one is refused at once, before any work), and one that waits longer than
# ASR_QUEUE_TIMEOUT_S for its turn is refused too. Both answer 503 + Retry-After.
_ASR_CONCURRENCY = env_number("ASR_MAX_CONCURRENCY", 1, cast=int, minimum=1)
_ASR_MAX_QUEUE = env_number("ASR_MAX_QUEUE", 4 * _ASR_CONCURRENCY, cast=int, minimum=0)
_ASR_QUEUE_TIMEOUT_S = env_number("ASR_QUEUE_TIMEOUT_S", 60.0, minimum=0.1)
_RETRY_AFTER_S = 5

_OOM_DETAIL = (
    "GPU out of memory while transcribing. Send a shorter file, lower NEMO_MAX_AUDIO_S or "
    "ASR_MAX_BATCH, or free the VRAM another service holds (POST /unload on it)."
)


class _Busy(HTTPException):
    """The 503 the gate answers with. The batch route tells it from a per-file failure."""


def _busy(reason: str) -> _Busy:
    if reason == "queue_timeout":
        detail = (
            f"The service is busy: this request waited {_ASR_QUEUE_TIMEOUT_S:g} s for its turn "
            f"(ASR_QUEUE_TIMEOUT_S). Retry shortly."
        )
    else:
        detail = (
            f"The service is busy: {_gate.max_active + _gate.max_queue} requests are already "
            f"in progress or waiting (ASR_MAX_CONCURRENCY + ASR_MAX_QUEUE). Retry shortly."
        )
    return _Busy(status_code=503, detail=detail, headers={"Retry-After": str(_RETRY_AFTER_S)})


_gate = RequestGate(_ASR_CONCURRENCY, _ASR_MAX_QUEUE, _ASR_QUEUE_TIMEOUT_S, busy=_busy)


def _request_id() -> str:
    """Names one request in the log and in the error it returns, so the two can be matched."""
    return uuid.uuid4().hex[:12]


async def _asr(fn, *args, **kwargs):
    """Run a blocking model call off the event loop: the model pinned, one of the gate's turns held.

    The reference is held across the ``await``, not just around the call that
    takes it: `asyncio.to_thread` runs the NeMo forward pass on a worker thread,
    and releasing early would let the idle reaper free weights the pass is still
    reading. It is taken before the turn on purpose: requests waiting for a turn
    keep the model resident, so with ASR_MODEL_TTL=0 a queue does not unload and
    reload it once per request. A wait that times out leaves the ``async with``
    like any other exit, so the reference goes back.
    """
    async with _model_slot.acquire_async() as model:
        async with _gate.turn():
            return await asyncio.to_thread(fn, model, *args, **kwargs)


def _load_parakeet():
    """Load the Parakeet model onto the active device."""
    global asr_model, model_loaded, compute_dtype
    logger.info(f"Loading Parakeet model '{MODEL_NAME}' on {device}...")
    try:
        import nemo.collections.asr as nemo_asr

        model = load_nemo_model(nemo_asr, MODEL_NAME)
        model.eval()
        if device == "cuda":
            model = model.to("cuda")
        model, compute_dtype = tune_for_inference(
            model, torch, device=device, matmul_precision=MATMUL_PRECISION, bf16=USE_BF16
        )
        asr_model = model
        model_loaded = True
        logger.info(f"Parakeet model '{MODEL_NAME}' loaded on {device} ({compute_dtype})")
        return model
    except Exception as e:
        logger.error(f"Failed to load Parakeet model: {e}", exc_info=True)
        raise


def _release_parakeet(model) -> None:
    """Clear the module aliases and get the weights off the GPU.

    Moving to CPU before dropping the object is deliberate. `empty_cache()` only
    returns blocks that are already unreferenced, so it frees nothing while any
    reference to the module survives — and NeMo registers every instantiated
    model in its own `AppState`, which is not ours to reason about. `.cpu()`
    releases the device allocation regardless of who still points at the object,
    which is the property this endpoint has to guarantee.
    """
    global asr_model, model_loaded
    asr_model = None
    model_loaded = False
    try:
        move_to_cpu(model)
    except Exception as e:
        logger.warning(f"Could not move Parakeet off the GPU before unload: {e}")


# ~3 GB resident. This service is opt-in and bursty — it is selected for a batch
# of files and then sits idle — so holding the card between bursts is the worst
# possible trade on a box where VRAM is the binding constraint.
#   >0 = seconds idle before unloading | 0 = unload immediately | -1 = never
MODEL_TTL = ttl_from_env(os.getenv, "ASR_MODEL_TTL", "MODEL_TTL", default=300.0)
_model_slot = ModelSlot(
    _load_parakeet, ttl_seconds=MODEL_TTL, name="Parakeet-TDT",
    on_unload=_release_parakeet,
)


def get_model():
    """Load or return the cached Parakeet model.

    Only for the startup preload. Request handlers go through ``_asr()``, which
    pins the model for the duration of the call; this does not, so the reference
    is gone by the time it returns.
    """
    with _model_slot.acquire() as model:
        return model


def _prepared_upload(upload: UploadFile):
    """The upload as a 16 kHz mono file NeMo can read; see ``prepared_upload``."""
    return prepared_upload(
        upload, max_bytes=MAX_UPLOAD_BYTES, max_seconds=MAX_AUDIO_SECONDS, ffmpeg=_FFMPEG,
    )


@asynccontextmanager
async def _admitted_upload(upload: UploadFile):
    """``_prepared_upload`` for a request the gate has let in.

    The place is reserved before the upload is copied and converted, so a request
    the service has no room for is refused for free, and it is given back when the
    block ends, whichever way it ends (the temp files go with it).
    """
    with _gate.admit():
        async with _prepared_upload(upload) as prepared:
            yield prepared


def _free_gpu_cache() -> None:
    """After an out-of-memory the failed attempt's blocks are still cached."""
    if device != "cuda":
        return
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass


class _PublicError(Exception):
    """A failure whose message was written for the client; ``_server_error`` passes it through."""


def _server_error(exc: Exception, request_id: str) -> HTTPException:
    """The HTTP failure for an unexpected exception; out-of-memory is not a 500.

    ``str(exc)`` is written for the operator and can hold temp-file and model
    paths or library internals, so a 500 says only that it failed and under which
    request id. The caller has logged the exception itself under the same id.
    """
    if is_cuda_oom(exc):
        _free_gpu_cache()
        return HTTPException(status_code=503, detail=_OOM_DETAIL)
    if isinstance(exc, _PublicError):
        return HTTPException(status_code=500, detail=f"{exc} Request id: {request_id}.")
    return HTTPException(
        status_code=500, detail=f"Transcription failed (internal error). Request id: {request_id}.",
    )


def _transcribe(model, audio_paths: list, **overrides) -> list:
    """``model.transcribe`` with only the arguments this NeMo build takes; one result per file.

    Feature-detected rather than version-checked: 2.7.x and 3.x share the same
    signature and return a list of ``Hypothesis``, and a build that lacks
    ``timestamps`` still answers (text only; the handler then reports a single
    segment covering the file).
    """
    kwargs = filter_transcribe_kwargs(model, {"timestamps": True, **_NEMO_RUNTIME_KWARGS, **overrides})
    return transcribe_results(model.transcribe(audio_paths, **kwargs))


def _run_transcription(model, audio_path: str) -> tuple[str, list]:
    """Run Parakeet on a single prepared audio file; return ``(text, segments)``."""
    results = _transcribe(model, [audio_path])
    return _parse_hypothesis(results[0]) if results else ("", [])


def _run_transcription_batch(model, audio_paths: list) -> list:
    """Run Parakeet on several prepared files in one batched call (NeMo batches internally)."""
    results = _transcribe(model, audio_paths, batch_size=min(len(audio_paths), _NEMO_MAX_BATCH))
    return [_parse_hypothesis(hyp) for hyp in results]


def _run_transcription_files(model, audio_paths: list) -> list:
    """Transcribe *audio_paths*; one item each: ``(text, segments)`` or the ``Exception`` for that file.

    One batched pass first. If that fails as a whole, one unreadable file would
    sink every other file in the request, so the files are retried one at a time
    (which also lowers peak VRAM after an out-of-memory). A batch that comes back
    with fewer results than files is the model's failure for the missing ones,
    reported as such rather than as a silent ``null``.
    """
    try:
        parsed = _run_transcription_batch(model, audio_paths)
    except Exception as e:
        if len(audio_paths) == 1:
            return [e]
        logger.warning("Batched transcription failed (%s); retrying %d files one by one", e, len(audio_paths))
        if is_cuda_oom(e):
            _free_gpu_cache()
        singly: list = []
        for path in audio_paths:
            try:
                singly.append(_run_transcription(model, path))
            except Exception as single:
                singly.append(single)
        return singly
    missing = _PublicError("The model returned no transcription for this file.")
    return [parsed[i] if i < len(parsed) else missing for i in range(len(audio_paths))]


@app.get("/health")
async def health():
    """Liveness probe.

    `model_resident: false` is not an error: the idle TTL released the weights
    and the next request reloads them. Returning non-200 for that would make an
    idle container report unhealthy under Docker's `curl -f`. Whether the service
    can take a request right now is /ready's question.
    """
    return {
        "status": "ok",
        "model_loaded": model_loaded,
        "model_resident": _model_slot.resident,
        "model_ttl_seconds": MODEL_TTL,
        "active_requests": _model_slot.refs,
        "device": device,
    }


@app.get("/ready")
async def ready():
    """Readiness: can this service take a request now?

    200 when the model has loaded at least once (a later idle unload or one
    failed reload does not flip it: the weights are proven loadable and the next
    request retries) and while nothing is known to be wrong. 503 with a JSON
    `reason` while the first load is running (`loading`, with `Retry-After`) and
    when the last load failed and none ever succeeded (`load_failed`; `detail` is a
    short category such as `model_files_unavailable`, never the exception text,
    which can hold paths: that is in the log).

    Like /health this only reads state and never triggers or waits for a load,
    so polling it cannot keep the model resident.
    """
    state = _model_slot.readiness()
    if _preload_pending and not state["ever_loaded"] and state["reason"] == "ok":
        state = {**state, "ready": False, "reason": "loading"}
    body = {
        "ready": state["ready"],
        "reason": state["reason"],
        "model_resident": state["resident"],
        "model_ever_loaded": state["ever_loaded"],
        "current_model": public_model_name(MODEL_NAME),
        "device": device,
    }
    if state["detail"]:
        body["detail"] = state["error_category"] or "load_error"
    if state["ready"]:
        return body
    headers = {"Retry-After": "5"} if state["reason"] == "loading" else None
    return JSONResponse(content=body, status_code=503, headers=headers)


@app.post("/unload")
async def unload():
    """Release the model and its VRAM now, without stopping the container.

    The idle TTL covers the common case; this is the deliberate one — you are
    about to run something else on the same GPU and want the memory back now.
    The next request reloads transparently.

    200 when released or already unloaded; **409 while a request is in flight**,
    since freeing memory a running forward pass still reads would crash the
    worker. Retry once `active_requests` reaches zero.
    """
    # Off the loop: the release hook moves the weights off the GPU under the
    # slot lock, which a synchronous call would make every other request wait for.
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
    status_info = {
        "status": "ok",
        "service": "Parakeet-ASR",
        "device": device,
        "cuda_available": torch.cuda.is_available(),
        "model_loaded": model_loaded,
        "model_resident": _model_slot.resident,
        "model_ttl_seconds": MODEL_TTL,
        "active_requests": _model_slot.refs,
        "current_model": public_model_name(MODEL_NAME),
        "compute_dtype": compute_dtype,
        "matmul_precision": MATMUL_PRECISION,
        "max_upload_mb": MAX_UPLOAD_BYTES / MIB,
        "max_audio_seconds": MAX_AUDIO_SECONDS,
        "queue": _gate.snapshot(),
        "runtime": runtime_versions(torch),
    }
    if torch.cuda.is_available():
        status_info["gpu_name"] = torch.cuda.get_device_name(0)
        status_info["gpu_memory_allocated"] = torch.cuda.memory_allocated()
        status_info["gpu_memory_total"] = torch.cuda.get_device_properties(0).total_memory
    return status_info


@app.post("/transcribe")
async def transcribe_audio(
    audio: UploadFile = File(...),
    language: str = Form("auto"),
):
    """Transcribe an audio file (Parakeet auto-detects the language).

    Returns the project's `stt-form-v1` shape: text, segment timestamps,
    detected language, and duration. 413 for a file over MAX_UPLOAD_MB or longer
    than NEMO_MAX_AUDIO_S; 503 + Retry-After when the service has no room for it.
    """
    req_id = _request_id()
    try:
        start_time = time.time()
        async with _admitted_upload(audio) as prepared:
            logger.info(f"Transcribing: {audio.filename} ({prepared.size / MIB:.1f}MB)")
            duration = prepared.duration
            text, segments = await _asr(_run_transcription, prepared.path)

        if not segments and text:
            segments = [{"start": 0.0, "end": duration, "text": text}]

        processing_time = time.time() - start_time
        logger.info(f"Transcription complete: {len(segments)} segments in {processing_time:.2f}s")

        return JSONResponse(content={
            "text": text,
            "segments": segments,
            # Parakeet auto-detects internally and does not expose a language tag;
            # echo the caller's hint when one was supplied.
            "language": None if (not language or language == "auto") else language,
            "language_probability": None,
            "duration": duration,
            "processing_time": processing_time,
            "task": "transcribe",
            "model": public_model_name(MODEL_NAME),
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.error("[%s] Transcription error: %s", req_id, e, exc_info=True)
        raise _server_error(e, req_id)


@app.post("/v1/audio/transcriptions")
async def openai_transcriptions(
    file: UploadFile = File(...),
    model: str = Form(None),
    language: str = Form(None),
    response_format: str = Form("json"),
):
    """OpenAI-compatible transcription endpoint (`/v1/audio/transcriptions`)."""
    req_id = _request_id()
    try:
        async with _admitted_upload(file) as prepared:
            text, segments = await _asr(_run_transcription, prepared.path)

        if (response_format or "").strip().lower() == "text":
            # Raw text, not a JSON-quoted string: OpenAI clients return the body verbatim.
            return PlainTextResponse(text)
        return JSONResponse(content={"text": text, "segments": segments})

    except HTTPException:
        raise
    except Exception as e:
        logger.error("[%s] OpenAI transcription error: %s", req_id, e, exc_info=True)
        raise _server_error(e, req_id)


@app.post("/detect_language")
async def detect_language(file: UploadFile = File(...)):
    """Best-effort language detection.

    Parakeet detects the language internally but does not surface a probability,
    so this returns a transcript sample for the UI without a confidence score.
    """
    req_id = _request_id()
    try:
        start_time = time.time()
        async with _admitted_upload(file) as prepared:
            duration = prepared.duration
            text, _ = await _asr(_run_transcription, prepared.path)
        return {
            "detected_language": None,
            "language_probability": None,
            "sample_text": text[:200],
            "processing_time": time.time() - start_time,
            "audio_duration": duration,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error("[%s] Language detection error: %s", req_id, e, exc_info=True)
        raise _server_error(e, req_id)


def _file_error(filename, detail, status_code: int) -> dict:
    """One failed file of a batch. *detail* is always a message written for the client."""
    return {"filename": filename, "error": detail if isinstance(detail, str) else str(detail), "status": status_code}


@app.post("/transcribe-batch")
async def transcribe_batch(
    audios: list[UploadFile] = File(...),
    language: str = Form("auto"),
):
    """Batch-transcribe multiple audio files in a single NeMo call.

    Files are prepared (16 kHz mono) first; everything that converts cleanly is
    transcribed in one batched inference pass, which is markedly faster than one
    request per file. Per-file failures (too large, too long, unreadable, no
    result from the model) are reported individually as `{filename, error,
    status}`; the size and duration limits apply to each file. The request as a
    whole holds one place in the queue: 503 + Retry-After when there is none, or
    when it waits longer than ASR_QUEUE_TIMEOUT_S for its turn.
    """
    req_id = _request_id()
    results: list = [None] * len(audios)
    entries: list = []  # (index, filename, PreparedAudio)
    batch_time = None
    try:
        async with AsyncExitStack() as stack:
            stack.enter_context(_gate.admit())
            for idx, audio_file in enumerate(audios):
                try:
                    prepared = await stack.enter_async_context(_prepared_upload(audio_file))
                except HTTPException as e:
                    results[idx] = _file_error(audio_file.filename, e.detail, e.status_code)
                except Exception as e:
                    logger.error(
                        "[%s] Batch item %r could not be prepared: %s", req_id, audio_file.filename, e, exc_info=True)
                    error = _server_error(e, req_id)
                    results[idx] = _file_error(audio_file.filename, error.detail, error.status_code)
                else:
                    entries.append((idx, audio_file.filename, prepared))

            if entries:
                start_time = time.time()
                # Must go through the same guard as the single-file path: this is the
                # widest forward pass in the service, so leaving it outside the
                # semaphore let it run concurrently with /transcribe on the same
                # model — exactly the VRAM spike ASR_MAX_CONCURRENCY=1 exists to stop.
                outputs = await _asr(_run_transcription_files, [entry[2].path for entry in entries])
                batch_time = time.time() - start_time
                for (idx, filename, prepared), output in zip(entries, outputs):
                    if isinstance(output, Exception):
                        logger.error(
                            "[%s] Batch item %r failed: %s", req_id, filename, output, exc_info=output)
                        error = _server_error(output, req_id)
                        results[idx] = _file_error(filename, error.detail, error.status_code)
                        continue
                    text, segments = output
                    results[idx] = {
                        "filename": filename,
                        "text": text,
                        "segments": segments,
                        "duration": prepared.duration,
                    }
    except HTTPException:
        raise
    except Exception as e:
        logger.error("[%s] Batch transcription error: %s", req_id, e, exc_info=True)
        raise _server_error(e, req_id)

    response = {"batch": True, "file_count": len(results), "results": results}
    if batch_time is not None:
        # Reported once for the batch. Dividing it per file was wrong by
        # construction — the files are transcribed in one batched forward
        # pass, so any per-file latency measured through this API was fiction.
        response["batch_processing_time"] = batch_time
    return response


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5005)
