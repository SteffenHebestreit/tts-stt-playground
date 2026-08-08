"""Qwen3-ASR speech-to-text service.

Uses the Qwen3-ASR-1.7B model for fast multilingual automatic speech
recognition with segment-level timestamps.  Supports CUDA, ROCm, and CPU.
"""

import os
import time
import asyncio
import tempfile
import logging
from contextlib import asynccontextmanager
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import torch
import librosa
import soundfile as sf
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from model_lifecycle import ModelSlot, ttl_from_env


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    """Pre-load the model on startup so the first request is fast."""
    try:
        get_model()
    except Exception as e:
        logger.warning(f"Could not preload model: {e}")
    yield


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
_ASR_SEM = asyncio.Semaphore(max(1, int(os.getenv("ASR_MAX_CONCURRENCY", "1"))))


async def _asr(method: str, *args, **kwargs):
    """Run a model method off the event loop, bounded by _ASR_SEM.

    Takes a method NAME rather than a bound method so this function owns the
    model reference: the slot is acquired here and held across the await, so the
    idle timer cannot unload the weights while a worker thread is using them.
    """
    async with _model_slot.acquire_async() as model, _ASR_SEM:
        return await asyncio.to_thread(getattr(model, method), *args, **kwargs)


# qwen_asr expects English language names, not ISO codes ("German", not "de").
LANGUAGE_NAME_MAP = {
    "zh": "Chinese", "en": "English", "yue": "Cantonese", "ar": "Arabic",
    "de": "German", "fr": "French", "es": "Spanish", "pt": "Portuguese",
    "id": "Indonesian", "it": "Italian", "ko": "Korean", "ru": "Russian",
    "th": "Thai", "vi": "Vietnamese", "ja": "Japanese", "tr": "Turkish",
    "hi": "Hindi", "ms": "Malay", "nl": "Dutch",
}


def _resolve_language(language):
    """Map a request language ('auto', ISO code, or name) to qwen_asr's format."""
    lang = (language or "").strip()
    if not lang or lang.lower() == "auto":
        return None
    mapped = LANGUAGE_NAME_MAP.get(lang.lower().replace("-", "_").split("_")[0])
    if mapped:
        return mapped
    # Never synthesize a language *name* from an unmapped code: "sv" became
    # "Sv", which the model does not recognise. Falling back to auto-detect
    # gives a correct result instead of a confidently wrong one.
    logger.info(f"Language '{language}' has no Qwen3-ASR mapping; using auto-detect")
    return None


def _audio_duration(path: str) -> float:
    """Duration in seconds, read from the header rather than decoding the file.

    librosa.get_duration() decodes the audio; on the event loop that stalled
    every other request for the length of the decode.
    """
    try:
        info = sf.info(path)
        return info.frames / float(info.samplerate) if info.samplerate else 0.0
    except Exception:
        # Formats libsndfile cannot open (some mp3/opus builds) still need librosa.
        return float(librosa.get_duration(path=path))


async def _save_upload(upload: UploadFile) -> tuple[str, bytes]:
    """Save an uploaded file to a temp path and return ``(tmp_path, raw_bytes)``."""
    suffix = Path(upload.filename).suffix if upload.filename else ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await upload.read()
        tmp.write(content)
        return tmp.name, content


def _load_qwen3_asr():
    """Construct the Qwen3-ASR model (called by the lifecycle slot)."""
    global asr_model, model_loaded
    logger.info("Loading Qwen3-ASR model...")
    try:
        from qwen_asr import Qwen3ASRModel

        model_name = os.getenv("QWEN3_ASR_MODEL", "Qwen/Qwen3-ASR-1.7B")

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

        asr_model = Qwen3ASRModel.from_pretrained(
            model_name,
            dtype=dtype,
            device_map=f"{device}:0" if device == "cuda" else "cpu",
            max_inference_batch_size=32,
            max_new_tokens=512,
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

    Kept for startup preload and status endpoints. Request handlers go through
    ``_asr()``, which pins the model for the duration of the call.
    """
    with _model_slot.acquire() as model:
        return model


@app.get("/health")
async def health():
    """Liveness probe.

    `model_resident: false` is NOT an error — the idle TTL released the weights
    to free VRAM and the next request reloads them. A non-200 here would make an
    idle container report unhealthy under Docker's `curl -f` check.
    """
    return {
        "status": "ok",
        "model_loaded": model_loaded,
        "model_resident": _model_slot.resident,
        "model_ttl_seconds": MODEL_TTL,
        "active_requests": _model_slot.refs,
        "device": device,
    }


@app.get("/status")
async def status():
    """Return detailed service status including GPU memory information."""
    status_info = {
        "status": "ok",
        "service": "Qwen3-ASR",
        "device": device,
        "cuda_available": torch.cuda.is_available(),
        "model_loaded": model_loaded,
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
    """
    Transcribe an audio file to text using Qwen3-ASR.

    Returns text, detected language, and segment-level timestamps.
    """
    tmp_path = None
    try:
        start_time = time.time()

        # Save uploaded file
        tmp_path, content = await _save_upload(audio)

        file_size_mb = len(content) / (1024 * 1024)
        logger.info(f"Transcribing: {audio.filename} ({file_size_mb:.1f}MB), language={language}")

        # Get audio duration
        duration = _audio_duration(tmp_path)

        # Transcribe with Qwen3-ASR (off the event loop — model.transcribe is blocking)
        lang_param = _resolve_language(language)
        results = await _asr(
            "transcribe",
            audio=tmp_path,
            language=lang_param,
        )

        processing_time = time.time() - start_time

        if not results:
            raise HTTPException(status_code=422, detail="Transcription produced no result (empty or unreadable audio)")

        # Build response matching the Whisper STT service format for compatibility
        result = results[0]
        detected_language = result.language if hasattr(result, 'language') else language
        text = result.text if hasattr(result, 'text') else str(result)

        # Build segments list
        segments = []
        if hasattr(result, 'time_stamps') and result.time_stamps:
            # If timestamps are available, create segments
            for ts in result.time_stamps:
                segments.append({
                    "start": ts.start_time if hasattr(ts, 'start_time') else 0.0,
                    "end": ts.end_time if hasattr(ts, 'end_time') else 0.0,
                    "text": ts.text if hasattr(ts, 'text') else text,
                    "confidence": ts.confidence if hasattr(ts, 'confidence') else None,
                })
        else:
            # Single segment with full text
            segments.append({
                "start": 0.0,
                "end": duration,
                "text": text,
                "confidence": 1.0,
            })

        logger.info(f"Transcription complete: {len(segments)} segments in {processing_time:.2f}s")

        return JSONResponse(content={
            "text": text,
            "segments": segments,
            "language": detected_language,
            "language_probability": result.language_probability if hasattr(result, 'language_probability') else None,
            "duration": duration,
            "processing_time": processing_time,
            "task": "transcribe",
            "model": "qwen3-asr",
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Transcription error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


@app.post("/detect_language")
async def detect_language(
    file: UploadFile = File(...),
):
    """Detect the language of an audio file using Qwen3-ASR."""
    tmp_path = None
    try:
        start_time = time.time()

        tmp_path, _ = await _save_upload(file)

        # Transcribe to detect language (off the event loop)
        results = await _asr(
            "transcribe",
            audio=tmp_path,
            language=None,  # Auto-detect
        )

        processing_time = time.time() - start_time

        if not results:
            raise HTTPException(status_code=422, detail="Language detection produced no result (empty or unreadable audio)")

        result = results[0]
        detected_language = result.language if hasattr(result, 'language') else "unknown"
        sample_text = result.text[:200] if hasattr(result, 'text') else ""

        duration = _audio_duration(tmp_path)

        return {
            "detected_language": detected_language,
            "language_probability": result.language_probability if hasattr(result, 'language_probability') else None,
            "sample_text": sample_text,
            "processing_time": processing_time,
            "audio_duration": duration,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Language detection error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


@app.post("/transcribe-batch")
async def transcribe_batch(
    audios: list[UploadFile] = File(...),
    language: str = Form("auto"),
):
    """Batch-transcribe multiple audio files, returning per-file results."""
    results = []

    for audio_file in audios:
        tmp_path = None
        try:
            tmp_path, _ = await _save_upload(audio_file)

            start_time = time.time()
            lang_param = _resolve_language(language)
            asr_results = await _asr(
                "transcribe",
                audio=tmp_path,
                language=lang_param,
            )
            processing_time = time.time() - start_time

            if not asr_results:
                raise RuntimeError("Transcription produced no result (empty or unreadable audio)")

            result = asr_results[0]
            duration = _audio_duration(tmp_path)

            results.append({
                "filename": audio_file.filename,
                "text": result.text if hasattr(result, 'text') else str(result),
                "language": result.language if hasattr(result, 'language') else language,
                "duration": duration,
                "processing_time": processing_time,
            })

        except Exception as e:
            results.append({
                "filename": audio_file.filename,
                "error": str(e),
            })
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    return {
        "batch": True,
        "file_count": len(results),
        "results": results,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5002)
