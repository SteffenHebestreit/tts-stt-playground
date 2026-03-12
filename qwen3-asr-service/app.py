import os
import time
import uuid
import tempfile
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import torch
import numpy as np
import librosa
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI(
    title="Qwen3-ASR Service",
    description="Speech-to-Text using Qwen3-ASR with multilingual support"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
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


async def _save_upload(upload: UploadFile) -> tuple[str, bytes]:
    """Save an uploaded file to a temp path. Returns (tmp_path, content_bytes)."""
    suffix = Path(upload.filename).suffix if upload.filename else ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await upload.read()
        tmp.write(content)
        return tmp.name, content


def get_model():
    """Load or return cached Qwen3-ASR model."""
    global asr_model, model_loaded
    if asr_model is None:
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
        except Exception as e:
            logger.error(f"Failed to load Qwen3-ASR model: {e}", exc_info=True)
            raise
    return asr_model


@app.on_event("startup")
async def startup():
    """Pre-load model on startup."""
    try:
        get_model()
    except Exception as e:
        logger.warning(f"Could not preload model: {e}")


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": model_loaded,
        "device": device,
    }


@app.get("/status")
async def status():
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
        model = get_model()
        start_time = time.time()

        # Save uploaded file
        tmp_path, content = await _save_upload(audio)

        file_size_mb = len(content) / (1024 * 1024)
        logger.info(f"Transcribing: {audio.filename} ({file_size_mb:.1f}MB), language={language}")

        # Get audio duration
        duration = librosa.get_duration(path=tmp_path)

        # Transcribe with Qwen3-ASR
        lang_param = None if language == "auto" else language
        results = model.transcribe(
            audio=tmp_path,
            language=lang_param,
        )

        processing_time = time.time() - start_time

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

    except Exception as e:
        logger.error(f"Transcription error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.post("/detect_language")
async def detect_language(
    file: UploadFile = File(...),
):
    """Detect the language of an audio file using Qwen3-ASR."""
    tmp_path = None
    try:
        model = get_model()
        start_time = time.time()

        tmp_path, _ = await _save_upload(file)

        # Transcribe to detect language
        results = model.transcribe(
            audio=tmp_path,
            language=None,  # Auto-detect
        )

        processing_time = time.time() - start_time
        result = results[0]
        detected_language = result.language if hasattr(result, 'language') else "unknown"
        sample_text = result.text[:200] if hasattr(result, 'text') else ""

        duration = librosa.get_duration(path=tmp_path)

        return {
            "detected_language": detected_language,
            "language_probability": result.language_probability if hasattr(result, 'language_probability') else None,
            "sample_text": sample_text,
            "processing_time": processing_time,
            "audio_duration": duration,
        }

    except Exception as e:
        logger.error(f"Language detection error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.post("/transcribe-batch")
async def transcribe_batch(
    audios: list[UploadFile] = File(...),
    language: str = Form("auto"),
):
    """Batch transcribe multiple audio files."""
    model = get_model()
    results = []

    for audio_file in audios:
        tmp_path = None
        try:
            tmp_path, _ = await _save_upload(audio_file)

            start_time = time.time()
            lang_param = None if language == "auto" else language
            asr_results = model.transcribe(
                audio=tmp_path,
                language=lang_param,
            )
            processing_time = time.time() - start_time

            result = asr_results[0]
            duration = librosa.get_duration(path=tmp_path)

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
                os.unlink(tmp_path)

    return {
        "batch": True,
        "file_count": len(results),
        "results": results,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5002)
