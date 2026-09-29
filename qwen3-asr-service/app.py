"""Qwen3-ASR speech-to-text service.

Uses the Qwen3-ASR-1.7B model for fast multilingual automatic speech
recognition. Supports CUDA, ROCm, and CPU.

Audio is cut into ``QWEN3_ASR_CHUNK_S``-second pieces (at quiet points) before
the model sees it. qwen-asr generates at most ``max_new_tokens`` per piece, so
handing it a long recording whole silently cut the text off after about two
minutes; short pieces cannot hit that cap, and each one yields a segment with a
real start and end. The service does not run the forced aligner, so timestamps
are per piece, not per word, and segments carry no confidence.
"""

import os
import time
import shutil
import asyncio
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import torch
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from audio_pipeline import (
    MIB, MULTIPART_SLACK, SAMPLE_RATE, BodyLimitMiddleware, env_number, join_texts,
    load_upload, model_input, plan_chunks,
)
from model_lifecycle import ModelSlot, ttl_from_env

# Upload bounds. The gateway limits what it forwards, but a direct caller must not
# be able to fill the disk or the host's memory with an arbitrarily long file.
MAX_UPLOAD_BYTES = int(env_number("MAX_UPLOAD_MB", 200.0, minimum=0.001) * MIB)
MAX_AUDIO_SECONDS = env_number("MAX_AUDIO_SECONDS", 7200.0, minimum=0.0)  # 0 = unlimited

# Length of the pieces the audio is cut into. The model accepts clips of up to 20
# minutes, but its output cap is per piece; 30 s keeps every piece far below it and
# bounds the memory one forward pass needs. The 600 s ceiling is so a typo cannot
# ask for a forward pass over an hour of audio.
CHUNK_S = env_number("QWEN3_ASR_CHUNK_S", 30.0, minimum=5.0, maximum=600.0)
# Pieces per forward pass. This is what bounds peak VRAM now that a long file is
# many pieces rather than one: each adds KV cache and encoder activations on top of
# the ~4 GB of weights. 4 is a conservative start for the 12 GB cards the default
# stack shares (not measured on a GPU); raise it where there is headroom, it is faster.
BATCH_SIZE = env_number("QWEN3_ASR_BATCH_SIZE", 4, cast=int, minimum=1)
# Generous next to real speech (a very fast speaker makes ~10 tokens per second)
# so a piece is never cut off by it, and still small enough that a decoder stuck
# in a repetition loop is stopped in seconds. Was a flat 512, which is ~2 minutes.
MAX_NEW_TOKENS = env_number(
    "QWEN3_ASR_MAX_NEW_TOKENS", max(512, int(CHUNK_S * 20)), cast=int, minimum=32,
)
# The decoded text of a piece that stopped on the cap is this close to the cap:
# the "language X<asr_text>" prefix (and the end token) count against it but are
# not in the text, and re-encoding the text can differ by a token or two.
_CAP_MARGIN_TOKENS = 8
# "" leaves the choice to transformers (SDPA). flash_attention_2 needs flash-attn
# in the image, which the stock image does not ship (see the Dockerfile).
ATTN_IMPLEMENTATION = (os.getenv("QWEN3_ASR_ATTN_IMPLEMENTATION") or "").strip().lower()
MODEL_NAME = os.getenv("QWEN3_ASR_MODEL", "Qwen/Qwen3-ASR-1.7B")

_FFMPEG = shutil.which("ffmpeg")

# A startup preload has been scheduled and has not finished: /ready reports
# "loading" for that window instead of "ready, never loaded".
_preload_pending = False


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    """Pre-load the model in the background so the first request is fast.

    Background, not before ``yield``: a first start downloads ~4 GB of weights, and
    doing that before the server accepts connections leaves the port closed (and
    /health unreachable) for minutes. /health answers straight away; /ready is
    503 "loading" until the model is in.
    """
    global _preload_pending
    preload = None
    if MODEL_TTL == 0:
        # "Release it the moment nothing is using it": preloading only to drop
        # it on the first release is work with no beneficiary.
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
    title="Qwen3-ASR Service",
    description="Speech-to-Text using Qwen3-ASR with multilingual support",
    lifespan=_lifespan,
)

allowed_origins_str = os.getenv("ALLOWED_ORIGINS", "*")
allowed_origins = [origin.strip() for origin in allowed_origins_str.split(",")] if allowed_origins_str else ["*"]
allow_credentials = os.getenv("ALLOW_CREDENTIALS", "false").strip().lower() in {"1", "true", "yes", "on"}
if "*" in allowed_origins and allow_credentials:
    allow_credentials = False

# Single-file routes only: a batch legitimately carries many files, each of which
# is bounded on its own. 64 KiB covers the small text fields.
_single_upload_limit = MAX_UPLOAD_BYTES + MULTIPART_SLACK + 64 * 1024
# Added before CORS so CORS is the outer layer and a 413 still carries its headers.
app.add_middleware(
    BodyLimitMiddleware,
    limits={"/transcribe": _single_upload_limit, "/detect_language": _single_upload_limit},
    detail=f"Request body is larger than the upload limit of {MAX_UPLOAD_BYTES / MIB:g} MB (MAX_UPLOAD_MB).",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model state
# Check CUDA first, then ROCm/HIP (also reports as cuda in PyTorch), then CPU
if torch.cuda.is_available():
    # Both CUDA and ROCm expose through torch.cuda — check ROCM_VERSION for ROCm
    device = "cuda"
elif os.getenv("ROCR_VISIBLE_DEVICES") or os.getenv("HIP_VISIBLE_DEVICES"):
    device = "cuda"  # ROCm uses the same CUDA API in PyTorch
else:
    device = "cpu"
asr_model = None
model_loaded = False

# One process-global 1.7B model on one GPU: without a bound, concurrent requests
# dispatched onto the default thread pool all enter the model at once and
# multiply peak VRAM while making each individual request slower.
def _max_concurrency() -> int:
    raw = os.getenv("ASR_MAX_CONCURRENCY", "1")
    try:
        return max(1, int(raw))
    except ValueError:
        logger.warning("Ignoring invalid ASR_MAX_CONCURRENCY=%r; using 1", raw)
        return 1


_ASR_SEM = asyncio.Semaphore(_max_concurrency())

_OOM_DETAIL = (
    "GPU out of memory while transcribing. Send a shorter file, lower QWEN3_ASR_BATCH_SIZE, "
    "or free the VRAM another service holds (POST /unload on it)."
)


async def _asr(fn, *args, **kwargs):
    """Run ``fn(model, *args)`` off the event loop, bounded by _ASR_SEM.

    This function owns the model reference: the slot is acquired here and held
    across the await, so the idle timer cannot unload the weights while a worker
    thread is using them.
    """
    async with _model_slot.acquire_async() as model, _ASR_SEM:
        return await asyncio.to_thread(fn, model, *args, **kwargs)


# qwen_asr expects English language names, not ISO codes ("German", not "de"),
# and refuses any name outside its own list (30 in qwen-asr 0.0.6).
LANGUAGE_NAME_MAP = {
    "zh": "Chinese", "en": "English", "yue": "Cantonese", "ar": "Arabic",
    "de": "German", "fr": "French", "es": "Spanish", "pt": "Portuguese",
    "id": "Indonesian", "it": "Italian", "ko": "Korean", "ru": "Russian",
    "th": "Thai", "vi": "Vietnamese", "ja": "Japanese", "tr": "Turkish",
    "hi": "Hindi", "ms": "Malay", "nl": "Dutch",
    "sv": "Swedish", "da": "Danish", "fi": "Finnish", "pl": "Polish", "cs": "Czech",
    "fil": "Filipino", "tl": "Filipino", "fa": "Persian", "el": "Greek",
    "ro": "Romanian", "hu": "Hungarian", "mk": "Macedonian",
    "cmn": "Chinese",
}
SUPPORTED_LANGUAGE_NAMES = frozenset(LANGUAGE_NAME_MAP.values())
_NAME_LOOKUP = {name.lower(): name for name in SUPPORTED_LANGUAGE_NAMES}


def _is_auto(language) -> bool:
    lang = (language or "").strip()
    return not lang or lang.lower() == "auto"


def _resolve_language(language):
    """Map a request language ('auto', ISO code, locale tag or name) to qwen_asr's format."""
    if _is_auto(language):
        return None
    lowered = language.strip().lower().replace("_", "-")
    mapped = _NAME_LOOKUP.get(lowered) or LANGUAGE_NAME_MAP.get(lowered.split("-")[0])
    if mapped:
        return mapped
    # Never synthesize a language *name* from an unmapped code: "sv" became
    # "Sv", which the model does not recognise. Falling back to auto-detect
    # gives a correct result instead of a confidently wrong one.
    logger.info(f"Language '{language}' has no Qwen3-ASR mapping; using auto-detect")
    return None


def _language_warning(language, resolved) -> Optional[str]:
    """Tell the caller when a language they named was not used."""
    if resolved is None and not _is_auto(language):
        return (
            f"Language '{language.strip()}' is not one Qwen3-ASR supports; the language was detected "
            f"automatically instead."
        )
    return None


def _load_qwen3_asr():
    """Construct the Qwen3-ASR model (called by the lifecycle slot)."""
    global asr_model, model_loaded
    logger.info("Loading Qwen3-ASR model...")
    try:
        from qwen_asr import Qwen3ASRModel

        # Prefer bfloat16 on CUDA; fall back to float16 (ROCm/older GPUs),
        # then float32 on CPU.
        if device == "cuda":
            if torch.cuda.is_bf16_supported():
                dtype = torch.bfloat16
            else:
                logger.warning("bfloat16 not supported on this GPU, using float16")
                dtype = torch.float16
        else:
            dtype = torch.float32

        kwargs = {}
        if ATTN_IMPLEMENTATION in ("sdpa", "eager", "flash_attention_2"):
            kwargs["attn_implementation"] = ATTN_IMPLEMENTATION
        elif ATTN_IMPLEMENTATION:
            logger.warning("Ignoring unknown QWEN3_ASR_ATTN_IMPLEMENTATION=%r", ATTN_IMPLEMENTATION)

        asr_model = Qwen3ASRModel.from_pretrained(
            MODEL_NAME,
            dtype=dtype,
            device_map=f"{device}:0" if device == "cuda" else "cpu",
            max_inference_batch_size=BATCH_SIZE,
            max_new_tokens=MAX_NEW_TOKENS,
            **kwargs,
        )
        model_loaded = True
        logger.info(f"Qwen3-ASR model loaded on {device} ({dtype})")
        return asr_model
    except Exception as e:
        logger.error(f"Failed to load Qwen3-ASR model: {e}", exc_info=True)
        raise


def _forget_qwen3_asr(_model) -> None:
    """Clear the module-level aliases so nothing keeps the weights alive."""
    global asr_model, model_loaded
    asr_model = None
    model_loaded = False


# This model is ~4 GB — the single largest resident cost in the default stack.
# On a 12 GB card that stack already sits at ~9.7 GB, so giving this back when
# idle is what leaves room for a TTS model alongside it. Reference counted, so
# a transcription in flight is never unloaded underneath itself.
#   >0 = seconds idle before unloading | 0 = unload immediately | -1 = never
MODEL_TTL = ttl_from_env(os.getenv, "ASR_MODEL_TTL", "MODEL_TTL", default=300.0)
_model_slot = ModelSlot(
    _load_qwen3_asr, ttl_seconds=MODEL_TTL, name="Qwen3-ASR",
    on_unload=_forget_qwen3_asr,
)


def get_model():
    """Load or return the cached Qwen3-ASR model (lazy singleton).

    Kept for the startup preload. Request handlers go through ``_asr()``, which
    pins the model for the duration of the call.
    """
    with _model_slot.acquire() as model:
        return model


def _server_error(exc: Exception) -> HTTPException:
    """The HTTP failure for an unexpected exception; out-of-memory is not a 500."""
    if type(exc).__name__ == "OutOfMemoryError" or "out of memory" in str(exc).lower():
        return HTTPException(status_code=503, detail=_OOM_DETAIL)
    return HTTPException(status_code=500, detail=str(exc))


# --- inference -----------------------------------------------------------------

@dataclass(frozen=True)
class PieceResult:
    text: str
    language: str
    capped: bool  # generation stopped on the token cap: the text is probably cut off


def _count_tokens(model, text: str) -> Optional[int]:
    """Tokens in *text* under the model's own tokenizer, None if it cannot be reached."""
    tokenizer = getattr(getattr(model, "processor", None), "tokenizer", None)
    if tokenizer is None:
        return None
    try:
        return len(tokenizer.encode(text, add_special_tokens=False))
    except Exception:
        return None


def _run_model(model, pieces, language) -> list:
    """One forward-pass sequence over *pieces* (blocking; runs on a worker thread)."""
    results = model.transcribe(audio=[(p, SAMPLE_RATE) for p in pieces], language=language)
    if len(results) != len(pieces):
        raise RuntimeError(f"Qwen3-ASR returned {len(results)} results for {len(pieces)} audio pieces")
    out = []
    for result in results:
        text = (getattr(result, "text", "") or "").strip()
        tokens = _count_tokens(model, text) if text else 0
        out.append(PieceResult(
            text=text,
            language=(getattr(result, "language", "") or "").strip(),
            capped=tokens is not None and tokens >= MAX_NEW_TOKENS - _CAP_MARGIN_TOKENS,
        ))
    return out


def _build_transcription(chunks, outputs, resolved, warnings: list) -> dict:
    """Assemble the response fields that depend on the model output."""
    segments = []
    seconds_by_language: dict[str, float] = {}
    capped_spans = []
    for chunk, out in zip(chunks, outputs):
        if not out.text:
            continue
        segment = {
            "start": round(chunk.start_s, 3),
            "end": round(chunk.end_s, 3),
            "text": out.text,
        }
        if out.language:
            segment["language"] = out.language
            seconds_by_language[out.language] = seconds_by_language.get(out.language, 0.0) + (chunk.end_s - chunk.start_s)
        if out.capped:
            segment["truncated"] = True
            capped_spans.append(f"{chunk.start_s:.0f}-{chunk.end_s:.0f} s")
        segments.append(segment)

    if capped_spans:
        message = (
            f"Generation stopped at the {MAX_NEW_TOKENS}-token limit in the piece(s) at {', '.join(capped_spans)}; "
            f"that text is probably cut off. Raise QWEN3_ASR_MAX_NEW_TOKENS or lower QWEN3_ASR_CHUNK_S."
        )
        logger.warning(message)
        warnings.append(message)

    # One name for the whole file: the language most of the audio was in. A single
    # noisy piece detected as another language must not relabel a German recording.
    if resolved:
        language = resolved
    elif seconds_by_language:
        language = max(seconds_by_language, key=seconds_by_language.get)
    else:
        language = ""
    return {
        "text": join_texts([s["text"] for s in segments]),
        "segments": segments,
        "language": language,
        "truncated": bool(capped_spans),
    }


async def _transcribe_upload(upload: UploadFile, language: str) -> dict:
    """Full ``stt-form-v1`` response for one uploaded file."""
    started = time.time()
    resolved = _resolve_language(language)
    warnings = []
    note = _language_warning(language, resolved)
    if note:
        warnings.append(note)

    audio = await load_upload(
        upload, max_bytes=MAX_UPLOAD_BYTES, max_seconds=MAX_AUDIO_SECONDS, ffmpeg=_FFMPEG,
    )
    logger.info(
        f"Transcribing: {upload.filename} ({audio.size / MIB:.1f}MB, {audio.duration:.1f}s), "
        f"language={language}"
    )
    chunks = await asyncio.to_thread(plan_chunks, audio.samples, chunk_s=CHUNK_S)
    pieces = [model_input(audio.samples[c.start:c.end]) for c in chunks]
    outputs = await _asr(_run_model, pieces, resolved)

    built = _build_transcription(chunks, outputs, resolved, warnings)
    processing_time = time.time() - started
    logger.info(
        f"Transcription complete: {len(chunks)} pieces, {len(built['segments'])} segments in {processing_time:.2f}s"
    )
    return {
        **built,
        "language_probability": None,
        "duration": audio.duration,
        "processing_time": processing_time,
        "chunks": len(chunks),
        "warnings": warnings,
        "task": "transcribe",
        "model": "qwen3-asr",
    }


# --- endpoints -----------------------------------------------------------------

@app.get("/health")
async def health():
    """Liveness probe.

    `model_resident: false` is NOT an error — the idle TTL released the weights
    to free VRAM and the next request reloads them. A non-200 here would make an
    idle container report unhealthy under Docker's `curl -f` check. Whether the
    service can take a request right now is /ready's question.
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
    when the last load failed and none ever succeeded (`load_failed`, `detail`
    carries the error).

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
        "current_model": MODEL_NAME,
        "device": device,
    }
    if state["detail"]:
        body["detail"] = state["detail"]
    if state["ready"]:
        return body
    headers = {"Retry-After": "5"} if state["reason"] == "loading" else None
    return JSONResponse(content=body, status_code=503, headers=headers)


@app.post("/unload")
async def unload():
    """Release the model and its VRAM now, without stopping the container.

    The idle TTL covers the common case; this is the deliberate one — you are
    about to run something else on the same GPU and want the memory back
    immediately. The next request reloads transparently.

    200 when released or already unloaded; **409 while a request is in flight**,
    because freeing memory a running generation still reads would crash the
    worker. Retry once `active_requests` reaches zero.
    """
    # Off the loop: the release hook moves memory back to the driver under the
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
        "service": "Qwen3-ASR",
        "device": device,
        "cuda_available": torch.cuda.is_available(),
        "model_loaded": model_loaded,
        "model_resident": _model_slot.resident,
        "model_ttl_seconds": MODEL_TTL,
        "active_requests": _model_slot.refs,
        "current_model": MODEL_NAME,
        "chunk_seconds": CHUNK_S,
        "batch_size": BATCH_SIZE,
        "max_new_tokens": MAX_NEW_TOKENS,
        "max_upload_mb": MAX_UPLOAD_BYTES / MIB,
        "max_audio_seconds": MAX_AUDIO_SECONDS,
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
    """Transcribe an audio file to text using Qwen3-ASR.

    Returns text, the language, and one segment per audio piece with its start
    and end. 413 for a file over MAX_UPLOAD_MB or longer than MAX_AUDIO_SECONDS.
    `truncated` and `warnings` say when the model's output limit cut a piece off.
    """
    try:
        return JSONResponse(content=await _transcribe_upload(audio, language))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Transcription error: {e}", exc_info=True)
        raise _server_error(e)


@app.post("/detect_language")
async def detect_language(
    file: UploadFile = File(...),
):
    """Detect the language of an audio file using Qwen3-ASR.

    Only the first piece (QWEN3_ASR_CHUNK_S seconds) is decoded and run: the
    language is decided in the first few seconds, and transcribing a whole
    recording to find it out was minutes of GPU time for nothing.
    """
    try:
        started = time.time()
        audio = await load_upload(
            file, max_bytes=MAX_UPLOAD_BYTES, max_seconds=MAX_AUDIO_SECONDS, ffmpeg=_FFMPEG,
            decode_seconds=CHUNK_S,
        )
        outputs = await _asr(_run_model, [model_input(audio.samples)], None)
        output = outputs[0]

        # The header knows the whole length; without one, a short prefix means the
        # file ended inside it and the prefix IS the file.
        if audio.declared is not None:
            duration = audio.declared
        else:
            duration = audio.duration if audio.duration < CHUNK_S else None

        return {
            "detected_language": output.language,
            "language_probability": None,
            "sample_text": output.text[:200],
            "processing_time": time.time() - started,
            "audio_duration": duration,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Language detection error: {e}", exc_info=True)
        raise _server_error(e)


@app.post("/transcribe-batch")
async def transcribe_batch(
    audios: list[UploadFile] = File(...),
    language: str = Form("auto"),
):
    """Batch-transcribe multiple audio files, returning per-file results.

    Files are processed one after another. A per-file failure (too large, too
    long, unreadable, model error) is reported as `{filename, error, status}`
    without failing the others.
    """
    results = []
    for audio_file in audios:
        try:
            body = await _transcribe_upload(audio_file, language)
            results.append({"filename": audio_file.filename, **body})
        except HTTPException as e:
            results.append({"filename": audio_file.filename, "error": str(e.detail), "status": e.status_code})
        except Exception as e:
            logger.error(f"Batch item {audio_file.filename} failed: {e}", exc_info=True)
            error = _server_error(e)
            results.append({"filename": audio_file.filename, "error": str(error.detail), "status": error.status_code})

    return {
        "batch": True,
        "file_count": len(results),
        "results": results,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5002)
