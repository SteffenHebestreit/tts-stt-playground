"""Cohere Transcribe speech-to-text service.

Wraps ``cohere-transcribe-03-2026`` (2B, Apache-2.0, conformer encoder and
transformer decoder; 14 languages: ar de el en es fr it ja ko nl pl pt vi zh) with
transformers' own ``CohereAsrForConditionalGeneration``, no remote code. bf16 on a
CUDA GPU, float32 on the CPU.

* **Weights**: by default from an ungated mirror pinned to one commit and checked
  against the official SHA-256 before loading (cohere_support.py says why).
* **Language**: the model has no language identification; the language is part of
  its decoder prompt. ``auto``/empty means ``COHERE_ASR_DEFAULT_LANGUAGE`` (``de``),
  and a language it does not know is refused with 422.
* **Speech gate and chunks**: the model transcribes silence and noise into words, so
  a voice activity detector decides what it hears (``COHERE_ASR_VAD``), and a long
  recording is cut in its pauses into pieces of at most ``COHERE_ASR_CHUNK_S``
  seconds (transformers_asr.py).
* **Timestamps**: the model has none. Each segment is one chunk, with the chunk's
  start and end, so ``srt``/``vtt``/``verbose_json`` through the gateway are as
  precise as the chunks (a sentence or a few, up to 30 s), not word-level.
"""

import asyncio
import gc
import logging
import os
import shutil
import time
import traceback
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from audio_pipeline import MIB, MULTIPART_SLACK, SAMPLE_RATE, env_number, load_upload
from body_limit import BodyLimitMiddleware
from cohere_support import (
    MAX_CHUNK_S, MIRROR_MODEL, SUPPORTED_LANGUAGES, default_revision, default_sha256,
    expected_size, generated_lengths,
)
from model_lifecycle import ModelSlot, RequestGate, ttl_from_env
from origin_guard import OriginGuardMiddleware, parse_allowed_origins
from transformers_asr import (
    DOWNLOAD_PATTERNS, VAD_MODES, ChunkText, SpeechGate, UnsupportedLanguage, batches,
    build_segments, default_language_from_env, env_choice, expected_sha256, is_auto,
    is_cuda_oom, is_local_model, max_new_tokens_for, plan_speech_chunks, public_model_name,
    resolve_language, speech_seconds, verify_weights,
)

# Upload bounds. The gateway limits what it forwards, but a direct caller must not
# be able to fill the disk or the host's memory with an arbitrarily long file.
MAX_UPLOAD_BYTES = int(env_number("MAX_UPLOAD_MB", 200.0, minimum=0.001) * MIB)
MAX_AUDIO_SECONDS = env_number("MAX_AUDIO_SECONDS", 7200.0, minimum=0.0)  # 0 = unlimited

MODEL_NAME = (os.getenv("COHERE_ASR_MODEL") or "").strip() or MIRROR_MODEL
REVISION = (os.getenv("COHERE_ASR_REVISION") or "").strip() or default_revision(MODEL_NAME)
# Only for the official, gated repository; huggingface_hub never logs it.
_HF_TOKEN = (os.getenv("HF_TOKEN") or "").strip() or None

# A COHERE_ASR_SHA256 that is neither a digest nor "off" stops every load (and says
# why in /ready and the log) instead of quietly loading unchecked weights.
_CONFIG_ERROR: Optional[str] = None
try:
    WEIGHTS_SHA256 = expected_sha256(os.getenv("COHERE_ASR_SHA256"), default_sha256(MODEL_NAME))
except ValueError as exc:
    WEIGHTS_SHA256 = None
    _CONFIG_ERROR = f"COHERE_ASR_SHA256 is {exc}"
    logger.error("%s; the model will not be loaded until it is fixed", _CONFIG_ERROR)

DEFAULT_LANGUAGE = default_language_from_env(
    os.getenv("COHERE_ASR_DEFAULT_LANGUAGE", "de"), supported=SUPPORTED_LANGUAGES, fallback="de")

# Length of the pieces a recording is cut into, in its pauses. 30 s is the most the
# feature extractor takes without splitting a piece itself (cohere_support.MAX_CHUNK_S).
CHUNK_S = env_number("COHERE_ASR_CHUNK_S", MAX_CHUNK_S, minimum=5.0, maximum=MAX_CHUNK_S)
# Pieces per generate() call: what bounds the activations on top of the ~4 GB of weights.
BATCH_SIZE = env_number("COHERE_ASR_BATCH_SIZE", 4, cast=int, minimum=1)
# The decoder's positions end at 1024, ten of them taken by the prompt.
MAX_NEW_TOKENS = env_number(
    "COHERE_ASR_MAX_NEW_TOKENS", max_new_tokens_for(CHUNK_S), cast=int, minimum=32, maximum=1000)
VAD_MODE = env_choice("COHERE_ASR_VAD", "silero", VAD_MODES)
VAD_THRESHOLD = env_number("COHERE_ASR_VAD_THRESHOLD", 0.5, minimum=0.05, maximum=0.95)

_FFMPEG = shutil.which("ffmpeg")
_speech_gate = SpeechGate(VAD_MODE, threshold=VAD_THRESHOLD)

# A startup preload has been scheduled and has not finished: /ready reports
# "loading" for that window instead of "ready, never loaded".
_preload_pending = False


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    """Pre-load the model in the background so the first request is fast.

    Background, not before ``yield``: a first start downloads and hashes ~4 GB, and
    doing that before the server accepts connections leaves the port closed (and
    /health unreachable) for minutes. /health answers straight away; /ready is 503
    "loading" until the model is in.
    """
    global _preload_pending
    preload = None
    if MODEL_TTL == 0:
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
    title="Cohere-ASR Service",
    description="Speech-to-Text with Cohere Transcribe (14 languages, no language detection)",
    lifespan=_lifespan,
)

# Unset or empty means no CORS headers at all; "*" only when it is written down (and
# logged); otherwise an explicit list. origin_guard.py.
allowed_origins = parse_allowed_origins(os.getenv("ALLOWED_ORIGINS", ""))
allow_credentials = os.getenv("ALLOW_CREDENTIALS", "false").strip().lower() in {"1", "true", "yes", "on"}
if "*" in allowed_origins and allow_credentials:
    allow_credentials = False

# Request-body limits (body_limit.py): the declared Content-Length AND the bytes that
# actually arrive are checked, so a chunked upload is cut off at the limit too. The
# one upload route carries one file plus the small text fields (64 KiB of slack); no
# other route takes a body of any size.
_UPLOAD_PATHS = frozenset({"/transcribe"})


def _body_limit(path: str) -> int:
    """Largest request body (bytes) accepted at *path*."""
    if path in _UPLOAD_PATHS:
        return MAX_UPLOAD_BYTES + MULTIPART_SLACK + 64 * 1024
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

# ROCm/HIP also reports through torch.cuda.
if torch.cuda.is_available():
    device = "cuda"
elif os.getenv("ROCR_VISIBLE_DEVICES") or os.getenv("HIP_VISIBLE_DEVICES"):
    device = "cuda"
else:
    device = "cpu"

model_loaded = False
compute_dtype = "float32"
weights_check = "pending" if WEIGHTS_SHA256 else "off"
load_seconds: Optional[float] = None

# Admission (model_lifecycle.RequestGate). ASR_MAX_CONCURRENCY generate() calls run
# at once; at most ASR_MAX_CONCURRENCY + ASR_MAX_QUEUE requests are admitted (the next
# one is refused at once, before any work), and one that waits longer than
# ASR_QUEUE_TIMEOUT_S for its turn is refused too. Both answer 503 + Retry-After.
_ASR_CONCURRENCY = env_number("ASR_MAX_CONCURRENCY", 1, cast=int, minimum=1)
_ASR_MAX_QUEUE = env_number("ASR_MAX_QUEUE", 4 * _ASR_CONCURRENCY, cast=int, minimum=0)
_ASR_QUEUE_TIMEOUT_S = env_number("ASR_QUEUE_TIMEOUT_S", 60.0, minimum=0.1)
_RETRY_AFTER_S = 5

_OOM_DETAIL = (
    "GPU out of memory while transcribing. Lower COHERE_ASR_BATCH_SIZE, or free the VRAM "
    "another service holds (POST /unload on it)."
)


class _Busy(HTTPException):
    """The 503 the gate answers with."""


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
    """Run ``fn(model, *args)`` off the event loop: the model pinned, one of the gate's turns held.

    The slot reference is held across the await, so the idle timer cannot unload the
    weights while a worker thread is using them. It is taken before the turn on
    purpose: requests waiting for a turn keep the model resident, so with
    ASR_MODEL_TTL=0 a queue does not unload and reload it once per request.
    """
    async with _model_slot.acquire_async() as model:
        async with _gate.turn():
            return await asyncio.to_thread(fn, model, *args, **kwargs)


@dataclass
class LoadedModel:
    model: object
    processor: object


def _model_dir() -> str:
    """The checkpoint directory: a mounted path as it is, a Hub id fetched (or found in the cache)."""
    if is_local_model(MODEL_NAME):
        return MODEL_NAME
    from huggingface_hub import snapshot_download

    return snapshot_download(
        MODEL_NAME, revision=REVISION, allow_patterns=list(DOWNLOAD_PATTERNS), token=_HF_TOKEN)


def _free_cuda_memory() -> None:
    """Collect what nothing references any more and hand the cached blocks back to the card."""
    gc.collect()
    if device == "cuda":
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


def _load_cohere() -> LoadedModel:
    """Fetch, verify and load the model (called by the lifecycle slot)."""
    global model_loaded, compute_dtype, weights_check, load_seconds
    if _CONFIG_ERROR:
        raise RuntimeError(_CONFIG_ERROR)
    started = time.time()
    logger.info("Loading Cohere Transcribe '%s' (revision %s) on %s...", public_model_name(MODEL_NAME), REVISION, device)
    from transformers import CohereAsrForConditionalGeneration, CohereAsrProcessor

    model_dir = _model_dir()
    if WEIGHTS_SHA256:
        hashed = time.time()
        try:
            verify_weights(model_dir, WEIGHTS_SHA256, size=expected_size(WEIGHTS_SHA256))
        except Exception:
            weights_check = "failed"
            raise
        weights_check = "ok"
        logger.info("model.safetensors matches SHA-256 %s (%.1f s)", WEIGHTS_SHA256, time.time() - hashed)

    if device == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype = torch.float32
    processor = CohereAsrProcessor.from_pretrained(model_dir)
    model = CohereAsrForConditionalGeneration.from_pretrained(model_dir, dtype=dtype)
    try:
        model.to(device)
    except BaseException as e:
        # A load that fails half-way (CUDA out of memory inside .to()) must give back
        # what it had already moved: the finished frames of the traceback still hold the
        # model, and the allocator keeps freed blocks reserved. On the NAS the process
        # kept 3.3 GB this way while reporting "not resident".
        traceback.clear_frames(e.__traceback__)
        del model
        _free_cuda_memory()
        raise
    model.eval()
    # The VAD's own model, loaded once (it ships in the wheel, nothing is downloaded).
    if VAD_MODE == "silero":
        try:
            _speech_gate.regions(torch.zeros(SAMPLE_RATE).numpy())
        except Exception as e:  # SpeechGate falls back to the energy gate by itself
            logger.warning("VAD warm-up failed: %s", e)

    compute_dtype = str(dtype).replace("torch.", "")
    model_loaded = True
    load_seconds = round(time.time() - started, 1)
    logger.info("Cohere Transcribe loaded on %s (%s) in %.1f s", device, compute_dtype, load_seconds)
    return LoadedModel(model=model, processor=processor)


def _release_cohere(loaded: LoadedModel) -> None:
    """Clear the alias and move the weights off the GPU before the slot drops them."""
    global model_loaded
    model_loaded = False
    try:
        loaded.model.to("cpu")
    except Exception as e:
        logger.warning(f"Could not move Cohere Transcribe off the GPU before unload: {e}")


# ~4 GB of bf16 weights: release them when idle so the card can serve other models.
#   >0 = seconds idle before unloading | 0 = unload immediately | -1 = never
MODEL_TTL = ttl_from_env(os.getenv, "ASR_MODEL_TTL", "MODEL_TTL", default=300.0)
_model_slot = ModelSlot(
    _load_cohere, ttl_seconds=MODEL_TTL, name="Cohere-Transcribe", on_unload=_release_cohere,
)


def get_model():
    """Load or return the cached model. Only for the startup preload: requests go through ``_asr()``."""
    with _model_slot.acquire() as model:
        return model


def _server_error(exc: Exception, request_id: str) -> HTTPException:
    """The HTTP failure for an unexpected exception; out-of-memory is not a 500.

    ``str(exc)`` is written for the operator and can hold temp-file and model paths,
    so a 500 says only that it failed and under which request id.
    """
    if is_cuda_oom(exc):
        if device == "cuda":
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
        return HTTPException(status_code=503, detail=_OOM_DETAIL)
    if not _model_slot.resident and _model_slot.last_error is not None:
        # The load itself failed (no model is resident, and the slot recorded why):
        # the service cannot serve, which is a 503, not a broken request.
        return HTTPException(
            status_code=503,
            detail=f"The model could not be loaded (GET /ready says why). Request id: {request_id}.",
        )
    return HTTPException(
        status_code=500, detail=f"Transcription failed (internal error). Request id: {request_id}.",
    )


# --- inference -----------------------------------------------------------------


def _run_model(loaded: LoadedModel, pieces: list, language: str) -> list:
    """generate() over *pieces* in batches of BATCH_SIZE (blocking; runs on a worker thread)."""
    processor, model = loaded.processor, loaded.model
    tokenizer = processor.tokenizer
    outputs: list[ChunkText] = []
    for batch in batches(pieces, BATCH_SIZE):
        inputs = processor(
            audio=list(batch), language=language, sampling_rate=SAMPLE_RATE,
            return_tensors="pt", punctuation=True,
        )
        inputs.pop("audio_chunk_index", None)   # every piece is short enough not to be split
        inputs = inputs.to(model.device, dtype=model.dtype)
        prompt_length = int(inputs["decoder_input_ids"].shape[-1])
        with torch.inference_mode():
            sequences = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        texts = tokenizer.batch_decode(sequences, skip_special_tokens=True)
        lengths = generated_lengths(
            sequences.tolist(), prompt_length, tokenizer.pad_token_id, tokenizer.eos_token_id)
        if len(texts) != len(batch):
            raise RuntimeError(f"generate() returned {len(texts)} rows for {len(batch)} pieces")
        outputs += [ChunkText(text=t.strip(), capped=n >= MAX_NEW_TOKENS) for t, n in zip(texts, lengths)]
    return outputs


async def _transcribe_upload(upload: UploadFile, language: Optional[str], initial_prompt: Optional[str]) -> dict:
    """Full ``stt-form-v1`` response for one uploaded file."""
    started = time.time()
    try:
        resolved = resolve_language(
            language, default=DEFAULT_LANGUAGE, supported=SUPPORTED_LANGUAGES, model="Cohere Transcribe")
    except UnsupportedLanguage as e:
        raise HTTPException(status_code=422, detail=str(e))
    warnings: list[str] = []
    if is_auto(language) and (language or "").strip():
        warnings.append(
            f"Cohere Transcribe cannot detect the language; transcribed as '{resolved}' "
            f"(COHERE_ASR_DEFAULT_LANGUAGE). Send the language to choose another.")
    if (initial_prompt or "").strip():
        warnings.append("Cohere Transcribe takes no prompt; 'initial_prompt' was ignored.")

    audio = await load_upload(upload, max_bytes=MAX_UPLOAD_BYTES, max_seconds=MAX_AUDIO_SECONDS, ffmpeg=_FFMPEG)
    regions = await asyncio.to_thread(_speech_gate.regions, audio.samples)
    chunks = plan_speech_chunks(audio.samples, regions, max_chunk_s=CHUNK_S)
    logger.info(
        "Transcribing %s (%.1f MB, %.1f s, language=%s): %d chunks, %.1f s of speech",
        upload.filename, audio.size / MIB, audio.duration, resolved, len(chunks), speech_seconds(chunks))

    outputs = []
    if chunks:
        pieces = [audio.samples[c.start:c.end] for c in chunks]
        outputs = await _asr(_run_model, pieces, resolved)
    else:
        warnings.append("No speech was detected in the audio.")
    segments, text, capped = build_segments(chunks, outputs)
    if capped:
        message = (
            f"Generation stopped at the {MAX_NEW_TOKENS}-token limit in the chunk(s) at {', '.join(capped)}; "
            f"that text is probably cut off. Raise COHERE_ASR_MAX_NEW_TOKENS or lower COHERE_ASR_CHUNK_S."
        )
        logger.warning(message)
        warnings.append(message)

    processing_time = time.time() - started
    logger.info("Transcription complete: %d segments in %.2f s", len(segments), processing_time)
    return {
        "text": text,
        "segments": segments,
        "language": resolved,
        "language_probability": None,
        "duration": audio.duration,
        "speech_seconds": round(speech_seconds(chunks), 3),
        "processing_time": processing_time,
        "chunks": len(chunks),
        "truncated": bool(capped),
        "warnings": warnings,
        "task": "transcribe",
        "model": public_model_name(MODEL_NAME),
    }


# --- endpoints -----------------------------------------------------------------


@app.get("/health")
async def health():
    """Liveness probe.

    `model_resident: false` is NOT an error: the idle TTL released the weights to
    free VRAM and the next request reloads them. Whether the service can take a
    request right now is /ready's question.
    """
    return {
        "status": "ok",
        "model_loaded": model_loaded,
        "model_resident": _model_slot.resident,
        "model_ttl_seconds": MODEL_TTL,
        "active_requests": _model_slot.refs,
        "current_model": public_model_name(MODEL_NAME),
        "default_language": DEFAULT_LANGUAGE,
        "device": device,
    }


@app.get("/ready")
async def ready():
    """Readiness: can this service take a request now?

    200 once the model has loaded at least once; 503 with a JSON `reason` while the
    first load is running (`loading`, with `Retry-After`) and when the last load
    failed and none ever succeeded (`load_failed`; `detail` is a short category,
    never the exception text, which can hold paths: that is in the log). A weights
    file that does not match COHERE_ASR_SHA256 is such a failure (`weights_check`
    in /status says `failed`). Like /health this never triggers or waits for a load.
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

    200 when released or already unloaded; **409 while a request is in flight**,
    because freeing memory a running generation still reads would crash the worker.
    The next request reloads transparently.
    """
    result = await _model_slot.try_unload_async()
    if result["reason"] == "busy":
        return JSONResponse(status_code=409, content={"detail": "Model is in use; retry when idle", **result})
    return {"model_resident": _model_slot.resident, **result}


@app.get("/status")
async def status():
    """Detailed service status, including what the gateway's language selector offers."""
    status_info = {
        "status": "ok",
        "service": "Cohere-ASR",
        "device": device,
        "cuda_available": torch.cuda.is_available(),
        "model_loaded": model_loaded,
        "model_resident": _model_slot.resident,
        "model_ttl_seconds": MODEL_TTL,
        "active_requests": _model_slot.refs,
        "current_model": public_model_name(MODEL_NAME),
        "revision": REVISION,
        "weights_sha256": WEIGHTS_SHA256,
        "weights_check": weights_check,
        "load_seconds": load_seconds,
        "supported_languages": sorted(SUPPORTED_LANGUAGES),
        "default_language": DEFAULT_LANGUAGE,
        "language_detection": False,
        "timestamps": "chunk",
        "vad": _speech_gate.describe(),
        "compute_dtype": compute_dtype,
        "chunk_seconds": CHUNK_S,
        "batch_size": BATCH_SIZE,
        "max_new_tokens": MAX_NEW_TOKENS,
        "max_upload_mb": MAX_UPLOAD_BYTES / MIB,
        "max_audio_seconds": MAX_AUDIO_SECONDS,
        "queue": _gate.snapshot(),
    }
    if torch.cuda.is_available():
        status_info["gpu_name"] = torch.cuda.get_device_name(0)
        status_info["gpu_memory_allocated"] = torch.cuda.memory_allocated()
        status_info["gpu_memory_reserved"] = torch.cuda.memory_reserved()
        status_info["gpu_memory_total"] = torch.cuda.get_device_properties(0).total_memory
    return status_info


@app.post("/transcribe")
async def transcribe_audio(
    audio: UploadFile = File(...),
    language: Optional[str] = Form(None),
    initial_prompt: Optional[str] = Form(None),
):
    """Transcribe an audio file in the requested (or default) language.

    Returns the `stt-form-v1` shape: text, one segment per chunk with its start and
    end, language, duration, plus `speech_seconds` (what the VAD passed on),
    `chunks`, `truncated` and `warnings`. 422 for a language the model does not know
    or audio that cannot be decoded, 413 for a file over MAX_UPLOAD_MB or longer
    than MAX_AUDIO_SECONDS, 503 + Retry-After when the service has no room for it.
    """
    req_id = _request_id()
    try:
        with _gate.admit():
            return JSONResponse(content=await _transcribe_upload(audio, language, initial_prompt))
    except HTTPException:
        raise
    except Exception as e:
        logger.error("[%s] Transcription error: %s", req_id, e, exc_info=True)
        raise _server_error(e, req_id)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5009)
