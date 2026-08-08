"""Chatterbox-TTS text-to-speech and voice-cloning service.

Wraps Resemble AI's Chatterbox Multilingual (MIT-licensed) — zero-shot TTS
with voice cloning across 23 languages including German. Output audio carries
Resemble's PerTh watermark by design.

Endpoints follow the project's `simple-json-tts-v1` and `voice-clone-tts-v1`
contracts so the frontend gateway can proxy them like the other providers.
"""

import os
import io
import re
import time
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

from model_lifecycle import ModelSlot, ttl_from_env


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    """Pre-load the model on startup so the first request is fast.

    This takes and immediately releases a reference, which arms the idle timer —
    so with the default TTL an untouched service frees its VRAM ~5 minutes after
    boot rather than holding ~4 GB forever. Set TTS_MODEL_TTL=-1 to keep it
    resident (correct on a card with headroom to spare).
    """
    try:
        get_model()
    except Exception as e:
        logger.warning(f"Could not preload model: {e}")
    yield
    _model_slot.unload()


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

# One process-global model on one GPU: overlapping generations thrash VRAM and
# make every request slower than running them back to back. The semaphore must
# also be held *across* a whole clone, because `audio_prompt_path` conditioning
# is stored on the shared model instance — two overlapping clones cross voices.
_GEN_SEM = asyncio.Semaphore(max(1, int(os.getenv("TTS_MAX_CONCURRENCY", "1"))))

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


def _load_chatterbox():
    """Construct the Chatterbox model (called by the lifecycle slot)."""
    global tts_model, model_loaded
    logger.info(f"Loading Chatterbox Multilingual on {device}...")
    try:
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS

        tts_model = ChatterboxMultilingualTTS.from_pretrained(device=device)
        model_loaded = True
        logger.info(f"Chatterbox Multilingual loaded on {device}")
        return tts_model
    except Exception as e:
        logger.error(f"Failed to load Chatterbox model: {e}", exc_info=True)
        raise


def _forget_chatterbox(_model) -> None:
    """Clear the module-level aliases so nothing keeps the weights alive."""
    global tts_model, model_loaded
    tts_model = None
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


def get_model():
    """Load or return the cached Chatterbox model (lazy singleton).

    Kept for the status/language endpoints, which only need a reference and do
    not run inference. Request handlers must use ``_model_slot.acquire()`` so the
    model is pinned for the duration of the call.
    """
    with _model_slot.acquire() as model:
        return model


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


def _safe_header(value: str) -> str:
    """Make a string safe for an HTTP header value.

    Header values must be latin-1 encodable; a non-latin-1 reference filename
    otherwise turned an already-completed generation into a 500.
    """
    return (value or "").encode("latin-1", "replace").decode("latin-1")


def _wav_response(wav, sr: int, extra_headers: Optional[dict] = None) -> StreamingResponse:
    """Convert a model output tensor/array to a streaming WAV response.

    Deliberately does *not* call ``torch.cuda.empty_cache()``/``gc.collect()``:
    that forced a device synchronise plus a full gen-2 GC on the response path
    and destroyed the caching allocator, so the next generation had to re-pay
    ``cudaMalloc``. Fragmentation is handled by PYTORCH_CUDA_ALLOC_CONF instead.
    """
    audio = wav.squeeze().detach().cpu().numpy() if torch.is_tensor(wav) else np.asarray(wav).squeeze()
    buffer = io.BytesIO()
    sf.write(buffer, audio.astype(np.float32), sr, format="WAV")
    buffer.seek(0)

    headers = {"X-Sample-Rate": str(sr)}
    if extra_headers:
        headers.update({k: _safe_header(v) for k, v in extra_headers.items()})
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
    """Liveness probe.

    `model_resident: false` is NOT an error — the idle TTL unloaded the weights
    to free VRAM and the next request reloads them. Returning a non-200 here
    would make an idle container show as unhealthy under Docker's `curl -f`.
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
        language_id = _resolve_language(request.language)
        started = time.monotonic()
        # The model reference is held for the whole generation: releasing before
        # the await would let the idle timer free the weights mid-inference.
        async with _model_slot.acquire_async() as model, _GEN_SEM:
            wav, sr = await asyncio.to_thread(
                _generate, model, request.text, language_id,
                None, request.exaggeration, request.cfg_weight,
            )
        elapsed = time.monotonic() - started
        logger.info(f"/tts generated {len(request.text)} chars in {elapsed:.2f}s")
        return _wav_response(wav, sr, {
            "X-Language": language_id,
            "X-Generation-Time": f"{elapsed:.3f}",
        })
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"TTS error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


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
    """
    text = text.strip()
    if not text:
        return []

    raw = [p for p in re.split(r"(?<=[.!?;:])\s+|\n+", text) if p.strip()]

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

    # The generator below runs AFTER this handler returns, so the model
    # reference is taken explicitly here and handed back in the generator's
    # `finally`. A `with` block cannot span a streaming response's lifetime.
    model = await _model_slot.acquire_ref()
    try:
        language_id = _resolve_language(request.language)
        sentences = _split_sentences(request.text)
        sample_rate = model.sr

        # Generate the first chunk up front so a failure here becomes a real HTTP
        # error. Once the StreamingResponse is returned the status line is already
        # on the wire, and the header advertises 0xFFFFFFFF sizes — a client cannot
        # distinguish a truncated stream from a complete one.
        started = time.monotonic()
        async with _GEN_SEM:
            first_wav, _sr = await asyncio.to_thread(
                _generate, model, sentences[0], language_id,
                None, request.exaggeration, request.cfg_weight,
            )
        ttfa = time.monotonic() - started
        logger.info(
            f"/tts-stream TTFA {ttfa * 1000:.0f}ms "
            f"(chunk 1/{len(sentences)}, {len(sentences[0])} chars)"
        )
    except BaseException:
        # Nothing will consume the generator, so release here or the model stays
        # pinned for the lifetime of the process.
        _model_slot.release_ref()
        raise

    async def generate_chunks():
        try:
            yield _streaming_wav_header(sample_rate)
            yield _to_pcm16(first_wav)
            for index, sentence in enumerate(sentences[1:], start=2):
                try:
                    async with _GEN_SEM:
                        wav, _sr = await asyncio.to_thread(
                            _generate, model, sentence, language_id,
                            None, request.exaggeration, request.cfg_weight,
                        )
                except Exception as e:
                    logger.error(f"Streaming TTS failed at chunk {index}/{len(sentences)}: {e}", exc_info=True)
                    # Raise rather than break: this makes Starlette abort the
                    # chunked body, so the client sees a broken transfer instead of
                    # a clean 200 with silently missing audio.
                    raise
                yield _to_pcm16(wav)
        finally:
            # Runs on completion, on error, and when the client disconnects
            # mid-stream (Starlette closes the generator).
            _model_slot.release_ref()

    return StreamingResponse(
        generate_chunks(),
        media_type="audio/wav",
        headers={
            "X-Language": language_id,
            "X-Sample-Rate": str(sample_rate),
            "X-Sentence-Count": str(len(sentences)),
            "X-Time-To-First-Audio": f"{ttfa:.3f}",
        },
    )


async def _clone(text: str, lang: str, file: UploadFile,
                 exaggeration: Optional[float], cfg_weight: Optional[float]) -> StreamingResponse:
    """Shared implementation for the clone endpoints."""
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text not provided")

    tmp_path = None
    try:
        language_id = _resolve_language(lang)

        suffix = Path(file.filename).suffix if file.filename else ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        # Both guards held across the whole generate call: audio_prompt
        # conditioning lives on the shared model instance, so overlapping clones
        # would cross voices — and the model must not be unloaded mid-inference.
        async with _model_slot.acquire_async() as model, _GEN_SEM:
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
