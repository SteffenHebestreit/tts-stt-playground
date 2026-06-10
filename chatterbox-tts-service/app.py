"""Chatterbox-TTS text-to-speech and voice-cloning service.

Wraps Resemble AI's Chatterbox Multilingual (MIT-licensed) — zero-shot TTS
with voice cloning across 23 languages including German. Output audio carries
Resemble's PerTh watermark by design.

Endpoints follow the project's `simple-json-tts-v1` and `voice-clone-tts-v1`
contracts so the frontend gateway can proxy them like the other providers.
"""

import os
import io
import gc
import re
import struct
import asyncio
import tempfile
import logging
from contextlib import asynccontextmanager
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import torch
import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    """Pre-load the model on startup so the first request is fast."""
    try:
        get_model()
    except Exception as e:
        logger.warning(f"Could not preload model: {e}")
    yield


app = FastAPI(
    title="Chatterbox-TTS Service",
    description="Multilingual TTS and voice cloning using Resemble AI Chatterbox",
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

device = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_LANGUAGE = os.getenv("CHATTERBOX_DEFAULT_LANGUAGE", "de")

tts_model = None
model_loaded = False

# Chatterbox Multilingual language ids (subset map for common request values).
LANGUAGE_ALIASES = {
    "auto": DEFAULT_LANGUAGE,
    "english": "en", "german": "de", "french": "fr", "spanish": "es",
    "italian": "it", "portuguese": "pt", "russian": "ru", "japanese": "ja",
    "korean": "ko", "chinese": "zh", "dutch": "nl", "polish": "pl",
    "turkish": "tr", "swedish": "sv", "danish": "da", "norwegian": "no",
    "finnish": "fi", "greek": "el", "hebrew": "he", "hindi": "hi",
    "arabic": "ar", "malay": "ms", "swahili": "sw",
}


def get_model():
    """Load or return the cached Chatterbox model (lazy singleton)."""
    global tts_model, model_loaded
    if tts_model is None:
        logger.info(f"Loading Chatterbox Multilingual on {device}...")
        try:
            from chatterbox.mtl_tts import ChatterboxMultilingualTTS

            tts_model = ChatterboxMultilingualTTS.from_pretrained(device=device)
            model_loaded = True
            logger.info(f"Chatterbox Multilingual loaded on {device}")
        except Exception as e:
            logger.error(f"Failed to load Chatterbox model: {e}", exc_info=True)
            raise
    return tts_model


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


def _wav_response(wav, sr: int, extra_headers: Optional[dict] = None) -> StreamingResponse:
    """Convert a model output tensor/array to a streaming WAV response."""
    audio = wav.squeeze().detach().cpu().numpy() if torch.is_tensor(wav) else np.asarray(wav).squeeze()
    buffer = io.BytesIO()
    sf.write(buffer, audio.astype(np.float32), sr, format="WAV")
    buffer.seek(0)

    if device == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    headers = {"X-Sample-Rate": str(sr)}
    if extra_headers:
        headers.update(extra_headers)
    return StreamingResponse(buffer, media_type="audio/wav", headers=headers)


def _generate(model, text: str, language_id: str, audio_prompt_path: Optional[str] = None,
              exaggeration: Optional[float] = None, cfg_weight: Optional[float] = None):
    """Blocking generation call (run off the event loop)."""
    kwargs = {"language_id": language_id}
    if audio_prompt_path:
        kwargs["audio_prompt_path"] = audio_prompt_path
    if exaggeration is not None:
        kwargs["exaggeration"] = exaggeration
    if cfg_weight is not None:
        kwargs["cfg_weight"] = cfg_weight
    wav = model.generate(text, **kwargs)
    return wav, model.sr


@app.get("/health")
async def health():
    """Basic liveness / readiness probe."""
    return {"status": "ok", "model_loaded": model_loaded, "device": device}


@app.get("/status")
async def status():
    """Return detailed service status including GPU memory information."""
    status_info = {
        "status": "ok",
        "service": "Chatterbox-TTS",
        "device": device,
        "cuda_available": torch.cuda.is_available(),
        "model_loaded": model_loaded,
        "default_language": DEFAULT_LANGUAGE,
        "supported_languages": _supported_language_ids(tts_model) if model_loaded else [],
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


class TTSRequest(BaseModel):
    """Request body for default-voice synthesis (simple-json-tts-v1)."""

    text: str
    language: str = "auto"
    exaggeration: Optional[float] = None
    cfg_weight: Optional[float] = None


@app.post("/tts")
async def text_to_speech(request: TTSRequest):
    """Generate speech with Chatterbox's built-in default voice."""
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text not provided")

    try:
        model = get_model()
        language_id = _resolve_language(request.language)
        wav, sr = await asyncio.to_thread(
            _generate, model, request.text, language_id,
            None, request.exaggeration, request.cfg_weight,
        )
        return _wav_response(wav, sr, {"X-Language": language_id})
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"TTS error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


def _split_sentences(text: str, min_chars: int = 20) -> list[str]:
    """Split text into sentence chunks for incremental generation.

    Short fragments are merged with the following sentence so we don't
    generate tiny clips with poor prosody.
    """
    raw = re.split(r"(?<=[.!?;:])\s+", text.strip())
    merged: list[str] = []
    buf = ""
    for part in raw:
        buf = f"{buf} {part}".strip() if buf else part
        if len(buf) >= min_chars:
            merged.append(buf)
            buf = ""
    if buf:
        if merged:
            merged[-1] += " " + buf
        else:
            merged.append(buf)
    return merged or [text]


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
    audio = wav.squeeze().detach().cpu().numpy() if torch.is_tensor(wav) else np.asarray(wav).squeeze()
    audio = np.clip(audio.astype(np.float32), -1.0, 1.0)
    return (audio * 32767.0).astype("<i2").tobytes()


@app.post("/tts-stream")
async def text_to_speech_stream(request: TTSRequest):
    """Sentence-chunked streaming TTS: audio starts after the first sentence.

    Returns a chunked `audio/wav` stream (PCM16 mono, unknown-length header).
    The chatterbox package has no token-level streaming API, so this splits
    the text into sentences and streams each one as soon as it is generated —
    time-to-first-audio drops from total-length to first-sentence latency.
    """
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text not provided")

    model = get_model()
    language_id = _resolve_language(request.language)
    sentences = _split_sentences(request.text)
    sample_rate = model.sr

    async def generate_chunks():
        yield _streaming_wav_header(sample_rate)
        for index, sentence in enumerate(sentences):
            try:
                wav, _sr = await asyncio.to_thread(
                    _generate, model, sentence, language_id,
                    None, request.exaggeration, request.cfg_weight,
                )
            except Exception as e:
                logger.error(f"Streaming TTS failed at sentence {index + 1}/{len(sentences)}: {e}", exc_info=True)
                break
            yield _to_pcm16(wav)
        if device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    return StreamingResponse(
        generate_chunks(),
        media_type="audio/wav",
        headers={
            "X-Language": language_id,
            "X-Sample-Rate": str(sample_rate),
            "X-Sentence-Count": str(len(sentences)),
        },
    )


async def _clone(text: str, lang: str, file: UploadFile,
                 exaggeration: Optional[float], cfg_weight: Optional[float]) -> StreamingResponse:
    """Shared implementation for the clone endpoints."""
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text not provided")

    tmp_path = None
    try:
        model = get_model()
        language_id = _resolve_language(lang)

        suffix = Path(file.filename).suffix if file.filename else ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        wav, sr = await asyncio.to_thread(
            _generate, model, text, language_id, tmp_path, exaggeration, cfg_weight,
        )
        return _wav_response(wav, sr, {
            "X-Language": language_id,
            "X-Clone-Source": file.filename or "unknown",
        })
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Voice clone error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


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
