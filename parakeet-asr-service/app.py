"""Parakeet-TDT speech-to-text service.

Wraps NVIDIA's `parakeet-tdt-0.6b-v3` (FastConformer-TDT) — a fast multilingual
ASR model covering 25 European languages (incl. German) with automatic language
detection. Exposes the project's native `stt-form-v1` contract (`/transcribe`
with segment timestamps) plus an OpenAI-compatible `/v1/audio/transcriptions`
endpoint so external tools can reuse it. Supports CUDA, ROCm, and CPU.
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

from transcription import parse_hypothesis as _parse_hypothesis


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    """Pre-load the model on startup so the first request is fast."""
    try:
        get_model()
    except Exception as e:
        logger.warning(f"Could not preload model: {e}")
    yield


app = FastAPI(
    title="Parakeet-ASR Service",
    description="Fast multilingual Speech-to-Text using NVIDIA Parakeet-TDT",
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

MODEL_NAME = os.getenv("PARAKEET_ASR_MODEL", "nvidia/parakeet-tdt-0.6b-v3")
TARGET_SAMPLE_RATE = 16000  # Parakeet expects 16 kHz mono input

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
    """Run a blocking model call off the event loop, bounded by _ASR_SEM."""
    async with _ASR_SEM:
        return await asyncio.to_thread(fn, *args, **kwargs)



def get_model():
    """Load or return the cached Parakeet model (lazy singleton)."""
    global asr_model, model_loaded
    if asr_model is None:
        logger.info(f"Loading Parakeet model '{MODEL_NAME}' on {device}...")
        try:
            import nemo.collections.asr as nemo_asr

            asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name=MODEL_NAME)
            asr_model.eval()
            if device == "cuda":
                asr_model = asr_model.to("cuda")
            model_loaded = True
            logger.info(f"Parakeet model '{MODEL_NAME}' loaded on {device}")
        except Exception as e:
            logger.error(f"Failed to load Parakeet model: {e}", exc_info=True)
            raise
    return asr_model


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


def _run_transcription(audio_path: str) -> tuple[str, list]:
    """Run Parakeet on a single prepared audio file; return ``(text, segments)``."""
    output = get_model().transcribe([audio_path], timestamps=True, **_NEMO_RUNTIME_KWARGS)
    return _parse_hypothesis(output[0]) if output else ("", [])


def _run_transcription_batch(audio_paths: list) -> list:
    """Run Parakeet on several prepared files in one batched call (NeMo batches internally)."""
    kwargs = {**_NEMO_RUNTIME_KWARGS, "batch_size": min(len(audio_paths), _NEMO_MAX_BATCH)}
    outputs = get_model().transcribe(audio_paths, timestamps=True, **kwargs)
    return [_parse_hypothesis(hyp) for hyp in (outputs or [])]


@app.get("/health")
async def health():
    """Basic liveness / readiness probe."""
    return {"status": "ok", "model_loaded": model_loaded, "device": device}


@app.get("/status")
async def status():
    """Return detailed service status including GPU memory information."""
    status_info = {
        "status": "ok",
        "service": "Parakeet-ASR",
        "device": device,
        "cuda_available": torch.cuda.is_available(),
        "model_loaded": model_loaded,
        "current_model": MODEL_NAME,
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
    detected language, and duration.
    """
    tmp_path = None
    prepared_path = None
    try:
        start_time = time.time()
        tmp_path, content = await _save_upload(audio)
        prepared_path = await asyncio.to_thread(_prepare_audio, tmp_path)

        file_size_mb = len(content) / (1024 * 1024)
        logger.info(f"Transcribing: {audio.filename} ({file_size_mb:.1f}MB)")

        duration = _audio_duration(prepared_path)
        text, segments = await _asr(_run_transcription, prepared_path)

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
            "model": MODEL_NAME,
        })

    except Exception as e:
        logger.error(f"Transcription error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # prepared_path may equal tmp_path when ffmpeg is unavailable; a set dedupes
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
        tmp_path, _ = await _save_upload(file)
        prepared_path = await asyncio.to_thread(_prepare_audio, tmp_path)
        text, segments = await _asr(_run_transcription, prepared_path)

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
    """Best-effort language detection.

    Parakeet detects the language internally but does not surface a probability,
    so this returns a transcript sample for the UI without a confidence score.
    """
    tmp_path = None
    prepared_path = None
    try:
        start_time = time.time()
        tmp_path, _ = await _save_upload(file)
        prepared_path = await asyncio.to_thread(_prepare_audio, tmp_path)
        duration = _audio_duration(prepared_path)
        text, _ = await _asr(_run_transcription, prepared_path)
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
    """Batch-transcribe multiple audio files in a single NeMo call.

    Files are prepared (16 kHz mono) first; everything that converts cleanly is
    transcribed in one batched inference pass, which is markedly faster than one
    request per file. Per-file failures are reported individually.
    """
    results: list = [None] * len(audios)
    entries: list = []  # (index, filename, prepared_path, duration)
    cleanup: set = set()
    try:
        for idx, audio_file in enumerate(audios):
            try:
                tmp_path, _ = await _save_upload(audio_file)
                cleanup.add(tmp_path)
                prepared_path = await asyncio.to_thread(_prepare_audio, tmp_path)
                cleanup.add(prepared_path)
                duration = _audio_duration(prepared_path)
                entries.append((idx, audio_file.filename, prepared_path, duration))
            except Exception as e:
                results[idx] = {"filename": audio_file.filename, "error": str(e)}

        if entries:
            start_time = time.time()
            # Must go through the same guard as the single-file path: this is the
            # widest forward pass in the service, so leaving it outside the
            # semaphore let it run concurrently with /transcribe on the same
            # model — exactly the VRAM spike ASR_MAX_CONCURRENCY=1 exists to stop.
            batch_out = await _asr(
                _run_transcription_batch, [entry[2] for entry in entries]
            )
            batch_time = time.time() - start_time
            for (idx, filename, _path, duration), parsed in zip(entries, batch_out):
                text, segments = parsed
                results[idx] = {
                    "filename": filename,
                    "text": text,
                    "segments": segments,
                    "duration": duration,
                }

        response = {"batch": True, "file_count": len(results), "results": results}
        if entries:
            # Reported once for the batch. Dividing it per file was wrong by
            # construction — the files are transcribed in one batched forward
            # pass, so any per-file latency measured through this API was fiction.
            response["batch_processing_time"] = batch_time
        return response
    finally:
        for path in cleanup:
            if path and os.path.exists(path):
                try:
                    os.unlink(path)
                except OSError:
                    pass


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5005)
