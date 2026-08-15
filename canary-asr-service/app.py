"""Canary-ASR speech-to-text service.

Wraps NVIDIA's `canary-180m-flash` (default) — a small, extremely fast
multilingual ASR model (RTFx >1000) covering English, German, Spanish, and
French with punctuation/capitalisation. Exposes the project's native
`stt-form-v1` contract (`/transcribe` with segment timestamps) plus an
OpenAI-compatible `/v1/audio/transcriptions` endpoint. Supports CUDA and CPU.

Canary has no built-in language identification: the request `language` field
selects the decoder language. Unknown/`auto` falls back to
`CANARY_DEFAULT_LANGUAGE` (default: `de`).
"""

import os
import time
import shutil
import asyncio
import subprocess
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
from transcription import parse_hypothesis as _parse_hypothesis


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    """Pre-load the model on startup so the first request is fast."""
    if MODEL_TTL == 0:
        # "Release it the moment nothing is using it" — preloading only to drop
        # it on the first release is work with no beneficiary.
        logger.info("Preload skipped: ASR_MODEL_TTL=0 unloads on every idle")
    else:
        try:
            get_model()
        except Exception as e:
            logger.warning(f"Could not preload model: {e}")
    yield


app = FastAPI(
    title="Canary-ASR Service",
    description="Fast multilingual Speech-to-Text using NVIDIA Canary",
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

# Global model state. ROCm/HIP also reports through torch.cuda in PyTorch.
if torch.cuda.is_available():
    device = "cuda"
elif os.getenv("ROCR_VISIBLE_DEVICES") or os.getenv("HIP_VISIBLE_DEVICES"):
    device = "cuda"
else:
    device = "cpu"

MODEL_NAME = os.getenv("CANARY_ASR_MODEL", "nvidia/canary-180m-flash")
SUPPORTED_LANGUAGES = {"en", "de", "es", "fr"}
DEFAULT_LANGUAGE = os.getenv("CANARY_DEFAULT_LANGUAGE", "de")
TARGET_SAMPLE_RATE = 16000  # Canary expects 16 kHz mono input

asr_model = None
model_loaded = False

_FFMPEG = shutil.which("ffmpeg")

# NeMo's transcribe() defaults spin up DataLoader worker processes and print a
# tqdm bar per call. At one-file-per-request granularity that setup costs more
# than the inference; a bounded semaphore keeps concurrent requests from
# multiplying peak VRAM on the single shared model.
_NEMO_RUNTIME_KWARGS = {"batch_size": 1, "num_workers": 0, "verbose": False}
_NEMO_MAX_BATCH = max(1, int(os.getenv("ASR_MAX_BATCH", "8")))
_ASR_SEM = asyncio.Semaphore(max(1, int(os.getenv("ASR_MAX_CONCURRENCY", "1"))))


async def _asr(fn, *args, **kwargs):
    """Run a blocking model call off the event loop, bounded by _ASR_SEM.

    The reference is held across the ``await``, not just around the call that
    takes it: `asyncio.to_thread` runs the NeMo forward pass on a worker thread,
    and releasing early would let the idle reaper free weights the pass is still
    reading.
    """
    async with _model_slot.acquire_async() as model, _ASR_SEM:
        return await asyncio.to_thread(fn, model, *args, **kwargs)


def _load_canary():
    """Load the Canary model onto the active device."""
    global asr_model, model_loaded
    logger.info(f"Loading Canary model '{MODEL_NAME}' on {device}...")
    try:
        import nemo.collections.asr as nemo_asr

        model = nemo_asr.models.ASRModel.from_pretrained(model_name=MODEL_NAME)
        model.eval()
        if device == "cuda":
            model = model.to("cuda")
        asr_model = model
        model_loaded = True
        logger.info(f"Canary model '{MODEL_NAME}' loaded on {device}")
        return model
    except Exception as e:
        logger.error(f"Failed to load Canary model: {e}", exc_info=True)
        raise


def _release_canary(model) -> None:
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
        model.cpu()
    except Exception as e:
        logger.warning(f"Could not move Canary off the GPU before unload: {e}")


# ~2 GB resident. Small, but this service exists precisely for the low-VRAM
# German path, where 2 GB held between bursts is the difference between fitting
# a TTS model alongside it and not.
#   >0 = seconds idle before unloading | 0 = unload immediately | -1 = never
MODEL_TTL = ttl_from_env(os.getenv, "ASR_MODEL_TTL", "MODEL_TTL", default=300.0)
_model_slot = ModelSlot(
    _load_canary, ttl_seconds=MODEL_TTL, name="Canary-Flash",
    on_unload=_release_canary,
)


def get_model():
    """Load or return the cached Canary model.

    Only for the startup preload. Request handlers go through ``_asr()``, which
    pins the model for the duration of the call; this does not, so the reference
    is gone by the time it returns.
    """
    with _model_slot.acquire() as model:
        return model


def _resolve_language(language: str) -> str:
    """Map the request language to one Canary supports (no auto-detection)."""
    lang = (language or "").strip().lower()
    if lang in SUPPORTED_LANGUAGES:
        return lang
    fallback = DEFAULT_LANGUAGE if DEFAULT_LANGUAGE in SUPPORTED_LANGUAGES else "en"
    if lang and lang != "auto":
        logger.info(f"Language '{language}' not supported by Canary; using '{fallback}'")
    return fallback


async def _save_upload(upload: UploadFile) -> tuple[str, bytes]:
    """Save an uploaded file to a temp path and return ``(tmp_path, raw_bytes)``."""
    suffix = Path(upload.filename).suffix if upload.filename else ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await upload.read()
        tmp.write(content)
        return tmp.name, content


def _needs_conversion(src_path: str) -> bool:
    """True unless the file is already a 16 kHz mono WAV.

    Probing costs microseconds; the ffmpeg fork/exec it avoids costs tens of
    milliseconds, which is a large fraction of a short-utterance request.
    """
    try:
        info = sf.info(src_path)
    except Exception:
        return True
    return not (info.format == "WAV" and info.samplerate == TARGET_SAMPLE_RATE and info.channels == 1)


def _audio_duration(path: str) -> float:
    """Duration in seconds, read from the header rather than decoding the file."""
    try:
        info = sf.info(path)
        return info.frames / float(info.samplerate) if info.samplerate else 0.0
    except Exception:
        # Formats libsndfile cannot open (some mp3/opus builds) still need librosa.
        return float(librosa.get_duration(path=path))


def _prepare_audio(src_path: str) -> str:
    """Convert any input to 16 kHz mono WAV via ffmpeg; fall back to the original."""
    if not _FFMPEG or not _needs_conversion(src_path):
        return src_path
    out_path = f"{src_path}.16k.wav"
    cmd = [
        _FFMPEG, "-y", "-i", src_path,
        "-ar", str(TARGET_SAMPLE_RATE), "-ac", "1", "-f", "wav",
        out_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=120)
        if result.returncode == 0 and os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            return out_path
        logger.warning(f"ffmpeg conversion failed (rc={result.returncode}); using original file")
    except Exception as e:
        logger.warning(f"ffmpeg conversion error: {e}; using original file")
    return src_path


def _run_transcription(model, audio_path: str, language: str) -> tuple[str, list]:
    """Run Canary on a single prepared audio file; return ``(text, segments)``."""
    kwargs = {
        "source_lang": language,
        "target_lang": language,
        "pnc": "yes",
        # Without these NeMo forks DataLoader workers and prints a tqdm bar for
        # every single-file request — pure overhead at this granularity.
        **_NEMO_RUNTIME_KWARGS,
    }
    try:
        output = model.transcribe([audio_path], timestamps=True, **kwargs)
    except TypeError:
        # Older NeMo builds without timestamp support for this model
        output = model.transcribe([audio_path], **kwargs)
    return _parse_hypothesis(output[0]) if output else ("", [])


@app.get("/health")
async def health():
    """Liveness probe.

    `model_resident: false` is not an error: the idle TTL released the weights
    and the next request reloads them. Returning non-200 for that would make an
    idle container report unhealthy under Docker's `curl -f`.
    """
    return {
        "status": "ok",
        "model_loaded": model_loaded,
        "model_resident": _model_slot.resident,
        "model_ttl_seconds": MODEL_TTL,
        "active_requests": _model_slot.refs,
        "device": device,
    }


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
    result = _model_slot.try_unload()
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
        "service": "Canary-ASR",
        "device": device,
        "cuda_available": torch.cuda.is_available(),
        "model_loaded": model_loaded,
        "model_resident": _model_slot.resident,
        "model_ttl_seconds": MODEL_TTL,
        "active_requests": _model_slot.refs,
        "current_model": MODEL_NAME,
        "supported_languages": sorted(SUPPORTED_LANGUAGES),
        "default_language": DEFAULT_LANGUAGE,
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
    """Transcribe an audio file in the requested (or default) language.

    Returns the project's `stt-form-v1` shape: text, segment timestamps,
    language, and duration.
    """
    tmp_path = None
    prepared_path = None
    try:
        start_time = time.time()
        resolved_language = _resolve_language(language)
        tmp_path, content = await _save_upload(audio)
        prepared_path = await asyncio.to_thread(_prepare_audio, tmp_path)

        file_size_mb = len(content) / (1024 * 1024)
        logger.info(f"Transcribing: {audio.filename} ({file_size_mb:.1f}MB, lang={resolved_language})")

        duration = _audio_duration(prepared_path)
        text, segments = await _asr(_run_transcription, prepared_path, resolved_language)

        if not segments and text:
            segments = [{"start": 0.0, "end": duration, "text": text}]

        processing_time = time.time() - start_time
        logger.info(f"Transcription complete: {len(segments)} segments in {processing_time:.2f}s")

        return JSONResponse(content={
            "text": text,
            "segments": segments,
            "language": resolved_language,
            "language_probability": None,
            "duration": duration,
            "processing_time": processing_time,
            "task": "transcribe",
            "model": MODEL_NAME,
        })

    except Exception as e:
        logger.error(f"Transcription error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        for path in {prepared_path, tmp_path}:
            if path and os.path.exists(path):
                try:
                    os.unlink(path)
                except OSError:
                    pass


@app.post("/v1/audio/transcriptions")
async def openai_transcriptions(
    file: UploadFile = File(...),
    model: str = Form(None),
    language: str = Form(None),
    response_format: str = Form("json"),
):
    """OpenAI-compatible transcription endpoint (`/v1/audio/transcriptions`)."""
    tmp_path = None
    prepared_path = None
    try:
        resolved_language = _resolve_language(language or "auto")
        tmp_path, _ = await _save_upload(file)
        prepared_path = await asyncio.to_thread(_prepare_audio, tmp_path)
        text, segments = await _asr(_run_transcription, prepared_path, resolved_language)

        if response_format == "text":
            return JSONResponse(content=text)
        return JSONResponse(content={"text": text, "segments": segments})

    except Exception as e:
        logger.error(f"OpenAI transcription error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        for path in {prepared_path, tmp_path}:
            if path and os.path.exists(path):
                try:
                    os.unlink(path)
                except OSError:
                    pass


@app.post("/detect_language")
async def detect_language(file: UploadFile = File(...)):
    """Best-effort 'detection': Canary has no LID, so this transcribes with the
    default language and returns a sample without a confidence score."""
    tmp_path = None
    prepared_path = None
    try:
        start_time = time.time()
        tmp_path, _ = await _save_upload(file)
        prepared_path = await asyncio.to_thread(_prepare_audio, tmp_path)
        duration = _audio_duration(prepared_path)
        text, _ = await _asr(_run_transcription, prepared_path, _resolve_language("auto"))
        return {
            "detected_language": None,
            "language_probability": None,
            "sample_text": text[:200],
            "processing_time": time.time() - start_time,
            "audio_duration": duration,
        }
    except Exception as e:
        logger.error(f"Language detection error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        for path in {prepared_path, tmp_path}:
            if path and os.path.exists(path):
                try:
                    os.unlink(path)
                except OSError:
                    pass


@app.post("/transcribe-batch")
async def transcribe_batch(
    audios: list[UploadFile] = File(...),
    language: str = Form("auto"),
):
    """Batch-transcribe multiple audio files, returning per-file results."""
    resolved_language = _resolve_language(language)
    results = []
    for audio_file in audios:
        tmp_path = None
        prepared_path = None
        try:
            tmp_path, _ = await _save_upload(audio_file)
            prepared_path = await asyncio.to_thread(_prepare_audio, tmp_path)
            duration = _audio_duration(prepared_path)

            start_time = time.time()
            text, segments = await _asr(_run_transcription, prepared_path, resolved_language)
            results.append({
                "filename": audio_file.filename,
                "text": text,
                "segments": segments,
                "language": resolved_language,
                "duration": duration,
                "processing_time": time.time() - start_time,
            })
        except Exception as e:
            results.append({"filename": audio_file.filename, "error": str(e)})
        finally:
            for path in {prepared_path, tmp_path}:
                if path and os.path.exists(path):
                    try:
                        os.unlink(path)
                    except OSError:
                        pass

    return {"batch": True, "file_count": len(results), "results": results}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5006)
