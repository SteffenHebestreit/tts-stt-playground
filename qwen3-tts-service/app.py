"""Qwen3-TTS text-to-speech and voice-cloning service.

Supports multiple Qwen3-TTS model variants (Base, CustomVoice, VoiceDesign)
with a persistent voice library for saved speaker embeddings.
"""

import os
import io
import gc
import json
import re
import time
import asyncio
import subprocess
import threading
import tempfile
import logging
from contextlib import asynccontextmanager
from pathlib import Path

import torch
import numpy as np
import soundfile as sf
import httpx
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    """Pre-load the model on startup so the first request is fast.

    The idle reaper then releases it if nothing uses it for TTS_MODEL_TTL
    seconds, so an untouched service does not hold multiple GB forever on a
    card it shares with the ASR stack.
    """
    try:
        get_model()
        _touch_model()
    except Exception as e:
        logger.warning(f"Could not preload model: {e}")
    reaper = asyncio.create_task(_idle_reaper())
    try:
        yield
    finally:
        reaper.cancel()


app = FastAPI(
    title="Qwen3-TTS Service",
    description="Text-to-Speech and Voice Cloning using Qwen3-TTS",
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

# STT service URLs for auto-transcription
QWEN3_ASR_URL = os.getenv("QWEN3_ASR_SERVICE_URL", "http://qwen3-asr-service:5002")

# Global model state
device = "cuda" if torch.cuda.is_available() else "cpu"
tts_model = None
model_loaded = False
current_model_name = ""
# What the operator asked for via /load_model. Distinct from
# current_model_name (which is "" while a load is in flight): this is what
# an idle-TTL reload must restore, otherwise unloading would silently revert
# the user back to the env default.
_desired_model_name = None

# One process-global model on one GPU. Without this, requests dispatched onto the
# default thread pool (min(32, cpu+4) workers) all enter the model at once and
# multiply peak VRAM while making every individual request slower.
_GEN_SEM = asyncio.Semaphore(max(1, int(os.getenv("TTS_MAX_CONCURRENCY", "1"))))
# Serialises model switching against itself; generation is kept out by _GEN_SEM.
_LOAD_LOCK = asyncio.Lock()


# Idle unloading. This service cannot use the simple ModelSlot pattern because
# the model identity can CHANGE at runtime via /load_model, so residency is
# tracked here instead: a last-used timestamp, gated by an in-flight counter.
# The counter is the safety property — a timestamp alone would happily free the
# weights while a worker thread was still generating with them.
#   >0 = seconds idle before unloading | 0 = unload as soon as idle | -1 = never
MODEL_TTL = float(os.getenv("TTS_MODEL_TTL", os.getenv("MODEL_TTL", "300")))
_inflight = 0
_inflight_lock = threading.Lock()
_last_used = time.monotonic()


def _touch_model() -> None:
    """Mark the model as used now, restarting its idle countdown."""
    global _last_used
    _last_used = time.monotonic()


def _should_unload(now: Optional[float] = None) -> bool:
    """Whether the idle model may be released right now.

    Split out from the loop so the decision is testable without waiting on a
    real sleep. The in-flight check is the safety property: a timestamp alone
    would happily free weights a worker thread is still generating with.
    """
    if MODEL_TTL < 0:
        return False
    if tts_model is None:
        return False
    with _inflight_lock:
        if _inflight > 0:
            return False
    elapsed = (now if now is not None else time.monotonic()) - _last_used
    return elapsed >= MODEL_TTL


def _reaper_tick_seconds() -> float:
    """Poll interval: often enough to honour the TTL, rarely enough to be free."""
    if MODEL_TTL <= 0:
        return 5.0
    return max(1.0, min(30.0, MODEL_TTL / 4))


async def _idle_reaper() -> None:
    """Unload the model once it has been idle for MODEL_TTL seconds."""
    if MODEL_TTL < 0:
        logger.info("Qwen3-TTS idle unloading disabled (TTS_MODEL_TTL=-1)")
        return
    while True:
        await asyncio.sleep(_reaper_tick_seconds())
        try:
            if not _should_unload():
                continue
            # _LOAD_LOCK keeps this from racing a /load_model swap or a
            # concurrent first-load in _acquire_model(). Re-check after taking
            # it: a request may have arrived while we were waiting for the lock.
            async with _LOAD_LOCK:
                if _should_unload():
                    await asyncio.to_thread(_unload_qwen3_tts)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning("Idle reaper error: %s", e)


def _unload_qwen3_tts() -> None:
    """Drop the model and let the caching allocator return VRAM to the driver."""
    global tts_model, model_loaded, current_model_name
    if tts_model is None:
        return
    logger.info("Unloading Qwen3-TTS model '%s' (idle)", current_model_name)
    tts_model = None
    model_loaded = False
    current_model_name = ""
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()


async def _gen(fn, *args, **kwargs):
    """Run a blocking model call off the event loop, bounded by _GEN_SEM.

    Also marks the model as in use for the duration, so the idle reaper
    cannot free the weights while a worker thread is generating with them.
    """
    global _inflight
    async with _GEN_SEM:
        with _inflight_lock:
            _inflight += 1
        try:
            return await asyncio.to_thread(fn, *args, **kwargs)
        finally:
            with _inflight_lock:
                _inflight -= 1
            _touch_model()


def _safe_header(value: str) -> str:
    """Make a string safe to use as an HTTP header value.

    Header values must be latin-1 encodable. Without this a non-latin-1
    reference filename turned an already-completed generation into a 500.
    """
    return (value or "").encode("latin-1", "replace").decode("latin-1")

# Available Qwen3-TTS model variants
AVAILABLE_MODELS = {
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base": {
        "name": "1.7B Base",
        "description": "General-purpose TTS and voice cloning (4.5GB)",
        "size": "1.7B",
        "capabilities": ["tts", "voice_clone"],
    },
    "Qwen/Qwen3-TTS-12Hz-0.6B-Base": {
        "name": "0.6B Base",
        "description": "Smaller, faster model for TTS and voice cloning (2.5GB)",
        "size": "0.6B",
        "capabilities": ["tts", "voice_clone"],
    },
    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice": {
        "name": "1.7B CustomVoice",
        "description": "Optimized for custom voice synthesis with built-in speakers (4.5GB)",
        "size": "1.7B",
        "capabilities": ["tts", "custom_voice"],
    },
    "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign": {
        "name": "1.7B VoiceDesign",
        "description": "Design voices via text description (4.5GB)",
        "size": "1.7B",
        "capabilities": ["tts", "voice_design"],
    },
}

# Supported languages
SUPPORTED_LANGUAGES = [
    "Chinese", "English", "Japanese", "Korean", "German",
    "French", "Russian", "Portuguese", "Spanish", "Italian"
]

# Built-in speaker names (Qwen3-TTS CustomVoice speakers)
BUILTIN_SPEAKERS = [
    "Vivian", "Ryan", "Ethan", "Olivia", "Aria",
    "Liam", "Nova", "Atlas", "Aurora", "Kai"
]


def load_model(model_name=None):
    """Load a Qwen3-TTS model by name.  Unloads the previous model first if switching."""
    global tts_model, model_loaded, current_model_name

    if model_name is None:
        model_name = os.getenv("QWEN3_TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-Base")

    # Already loaded
    if tts_model is not None and current_model_name == model_name:
        return tts_model

    # Unload previous model
    if tts_model is not None:
        logger.info(f"Unloading current model: {current_model_name}")
        del tts_model
        tts_model = None
        model_loaded = False
        current_model_name = ""
        if device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    logger.info(f"Loading Qwen3-TTS model: {model_name}")
    try:
        from qwen_tts import Qwen3TTSModel

        attn_impl = "flash_attention_2"
        try:
            import flash_attn  # noqa: F401
        except ImportError:
            attn_impl = "sdpa"
            logger.info("flash-attn not available, using SDPA attention")

        tts_model = Qwen3TTSModel.from_pretrained(
            model_name,
            device_map=f"{device}:0" if device == "cuda" else "cpu",
            dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            attn_implementation=attn_impl,
        )
        model_loaded = True
        current_model_name = model_name
        logger.info(f"Qwen3-TTS model '{model_name}' loaded on {device}")
    except Exception as e:
        logger.error(f"Failed to load Qwen3-TTS model: {e}", exc_info=True)
        raise
    return tts_model


def get_model():
    """Return the currently-loaded model, loading the default if needed.

    Prefer ``_acquire_model()`` from request handlers — this variant cannot be
    made safe against a concurrent model swap.
    """
    if tts_model is None:
        return load_model()
    return tts_model


async def _acquire_model():
    """Return ``(model, model_name)`` as a single consistent snapshot.

    Taking ``_LOAD_LOCK`` matters twice:

    - ``load_model`` blanks ``tts_model`` before it starts loading the new
      weights, so an unguarded ``get_model()`` landing in that window would
      kick off a *second* concurrent ``from_pretrained`` — two multi-GB models
      allocating at once, on the event loop.
    - ``current_model_name`` is blanked in the same window. Reading it
      separately made capability checks fail with a spurious
      "does not support voice cloning" for the whole duration of a swap.

    Note this reloads ``_desired_model_name``, not the env default: after the
    idle TTL has unloaded a model the user explicitly switched to, reloading the
    env default instead would silently revert their choice.
    """
    async with _LOAD_LOCK:
        model = tts_model
        if model is None:
            model = await asyncio.to_thread(load_model, _desired_model_name)
        return model, current_model_name


@asynccontextmanager
async def _acquire_all_gen_permits():
    """Acquire every _GEN_SEM permit, so no generation can be in flight.

    ``async with _GEN_SEM`` only takes one permit; with TTS_MAX_CONCURRENCY > 1
    that would leave other generations running against a model being unloaded.
    """
    permits = max(1, int(os.getenv("TTS_MAX_CONCURRENCY", "1")))
    acquired = 0
    try:
        for _ in range(permits):
            await _GEN_SEM.acquire()
            acquired += 1
        yield
    finally:
        for _ in range(acquired):
            _GEN_SEM.release()


@app.get("/health")
async def health():
    """Liveness probe.

    `model_resident: false` is NOT an error — the idle reaper released the
    weights to free VRAM and the next request reloads them (restoring the
    switched-to model, not the env default). A non-200 here would make an idle
    container report unhealthy under Docker's `curl -f` check.
    """
    return {
        "status": "ok",
        "model_loaded": model_loaded,
        "model_resident": tts_model is not None,
        "model_ttl_seconds": MODEL_TTL,
        "active_requests": _inflight,
        "current_model": current_model_name,
        "desired_model": _desired_model_name,
        "device": device,
    }


@app.post("/unload")
async def unload():
    """Release the model and its VRAM now, without stopping the container.

    The idle reaper covers the common case; this is the deliberate one — you are
    about to run something else on the same GPU and want the memory back
    immediately. The next request reloads transparently.

    200 when released or already unloaded; **409 while a generation is in
    flight**, because freeing weights a worker thread is still generating with
    would crash it. Retry once `inflight` reaches zero.

    Note this also clears the model selected via POST /load_model — the next
    request reloads whichever model is currently desired, not the env default.
    """
    with _inflight_lock:
        if _inflight > 0:
            return JSONResponse(
                status_code=409,
                content={
                    "detail": "Model is in use; retry when idle",
                    "unloaded": False,
                    "reason": "busy",
                    "inflight": _inflight,
                },
            )
        if tts_model is None:
            return {
                "unloaded": False, "reason": "not_resident",
                "inflight": 0, "model_resident": False,
            }
        # Unload while still holding the lock: releasing it first would let a
        # request slip in, take the model, and have it freed underneath.
        _unload_qwen3_tts()

    return {
        "unloaded": True, "reason": "ok",
        "inflight": 0, "model_resident": tts_model is not None,
    }


@app.get("/status")
async def status():
    """Return detailed service status including GPU memory and loaded model."""
    status_info = {
        "status": "ok",
        "device": device,
        "cuda_available": torch.cuda.is_available(),
        "model_loaded": model_loaded,
        "current_model": current_model_name,
        "current_model_info": AVAILABLE_MODELS.get(current_model_name),
        "supported_languages": SUPPORTED_LANGUAGES,
        "builtin_speakers": BUILTIN_SPEAKERS,
    }
    if torch.cuda.is_available():
        status_info["gpu_name"] = torch.cuda.get_device_name(0)
        status_info["gpu_memory_allocated"] = torch.cuda.memory_allocated()
        status_info["gpu_memory_total"] = torch.cuda.get_device_properties(0).total_memory
    return status_info


@app.get("/models")
async def list_models():
    """List available Qwen3-TTS model variants."""
    return {
        "models": AVAILABLE_MODELS,
        "current_model": current_model_name,
    }


class LoadModelRequest(BaseModel):
    """Request body for switching the active Qwen3-TTS model."""

    model: str


@app.post("/load_model")
async def switch_model(request: LoadModelRequest):
    """Switch to a different Qwen3-TTS model variant. Downloads if not cached."""
    model_name = request.model

    if model_name not in AVAILABLE_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model: {model_name}. Available: {list(AVAILABLE_MODELS.keys())}"
        )

    if model_name == current_model_name:
        return {"status": "ok", "message": "Model already loaded", "model": model_name}

    # A switch can mean a multi-GB download, so it runs off the event loop.
    # _LOAD_LOCK excludes _acquire_model(), which is what every handler uses to
    # read the model and its name together; draining _GEN_SEM entirely (not just
    # taking one permit) makes sure no generation is still running against the
    # model that is about to be unloaded, even with TTS_MAX_CONCURRENCY > 1.
    try:
        async with _acquire_all_gen_permits(), _LOAD_LOCK:
            if model_name == current_model_name:
                return {"status": "ok", "message": "Model already loaded", "model": model_name}
            await asyncio.to_thread(load_model, model_name)
            global _desired_model_name
            _desired_model_name = model_name
            _touch_model()
        return {
            "status": "ok",
            "message": f"Model switched to {model_name}",
            "model": model_name,
            "model_info": AVAILABLE_MODELS[model_name],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")


@app.get("/speakers")
async def list_speakers():
    """List available built-in speakers."""
    return {
        "speakers": BUILTIN_SPEAKERS,
        "languages": SUPPORTED_LANGUAGES,
    }


def _cleanup_temp(path):
    """Remove a temp file, never raising.

    Called from `finally` blocks: an OSError escaping here replaced an already
    successful audio response with a 500 and skipped the second temp file.
    """
    if not path:
        return
    try:
        os.unlink(path)
    except OSError:
        pass


async def _auto_transcribe(audio_path: str, filename: str = "audio.wav") -> dict:
    """Auto-transcribe an audio file using the Qwen3-ASR service.
    Returns full response dict with text, segments, duration."""
    try:
        logger.info(f"Auto-transcribing reference audio via Qwen3-ASR: {filename}")
        async with httpx.AsyncClient(timeout=60.0) as client:
            with open(audio_path, "rb") as f:
                response = await client.post(
                    f"{QWEN3_ASR_URL}/transcribe",
                    files={"audio": (filename, f)},
                )
            if response.status_code == 200:
                data = response.json()
                text = data.get("text", "").strip()
                logger.info(f"Auto-transcription result: '{text[:100]}...'")
                return data
            else:
                logger.warning(f"Auto-transcription failed: HTTP {response.status_code}")
                return {}
    except Exception as e:
        logger.warning(f"Auto-transcription error: {e}")
        return {}


def _trim_audio_segment(input_path: str, start: float, end: float, output_path: str) -> bool:
    """Use ffmpeg to extract a segment from an audio file."""
    try:
        duration = end - start
        cmd = [
            "ffmpeg", "-y", "-i", input_path,
            "-ss", f"{start:.3f}", "-t", f"{duration:.3f}",
            "-ar", "24000", "-ac", "1",
            output_path,
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=30)
        return result.returncode == 0 and os.path.exists(output_path)
    except Exception as e:
        logger.warning(f"ffmpeg trim error: {e}")
        return False


def _pick_best_segment(segments: list, min_dur: float = 3.0, max_dur: float = 10.0) -> dict:
    """Pick the best reference segment from ASR timestamps.
    Prefers segments between min_dur and max_dur seconds."""
    if not segments:
        return {}
    # Try to find a single segment in the ideal range
    for seg in segments:
        dur = seg.get("end", 0) - seg.get("start", 0)
        if min_dur <= dur <= max_dur:
            return seg
    # Otherwise merge consecutive segments to reach min_dur
    merged_start = segments[0].get("start", 0)
    merged_end = segments[0].get("end", 0)
    merged_text = segments[0].get("text", "")
    for seg in segments[1:]:
        merged_end = seg.get("end", 0)
        merged_text += " " + seg.get("text", "")
        if (merged_end - merged_start) >= min_dur:
            break
    return {"start": merged_start, "end": min(merged_end, merged_start + max_dur), "text": merged_text.strip()}


# --- Voice Library: persistent speaker prompt cache ---

VOICES_DIR = Path(os.getenv("VOICES_DIR", "/app/voices"))
VOICES_DIR.mkdir(parents=True, exist_ok=True)


_SAFE_VOICE_RE = re.compile(r"^[a-zA-Z0-9_\-]+$")
_UNSAFE_VOICE_CHARS_RE = re.compile(r"[^a-zA-Z0-9_\-]+")


def _voice_id_from_name(name: str) -> str:
    """Derive a filesystem-safe voice id from a user-supplied display name."""
    voice_id = _UNSAFE_VOICE_CHARS_RE.sub("_", (name or "").strip().lower()).strip("_")
    if not voice_id:
        raise HTTPException(
            status_code=400,
            detail="Voice name must contain at least one letter, digit, dash, or underscore",
        )
    return voice_id


def _voice_dir(voice_id: str) -> Path:
    """Return the directory for a voice, validating the id to prevent path traversal."""
    if not _SAFE_VOICE_RE.match(voice_id):
        raise HTTPException(status_code=400, detail="Invalid voice id")
    return VOICES_DIR / voice_id


def _load_voice_metadata(voice_id: str) -> dict:
    """Read the metadata.json for a saved voice, or return ``{}``."""
    meta_path = _voice_dir(voice_id) / "metadata.json"
    if meta_path.exists():
        return json.loads(meta_path.read_text())
    return {}


def _save_voice_prompt(voice_id: str, prompt_item, metadata: dict):
    """Save a VoiceClonePromptItem to disk as tensors + metadata."""
    vdir = _voice_dir(voice_id)
    vdir.mkdir(parents=True, exist_ok=True)
    # Save tensors
    torch.save(prompt_item.ref_spk_embedding, vdir / "ref_spk_embedding.pt")
    if prompt_item.ref_code is not None:
        torch.save(prompt_item.ref_code, vdir / "ref_code.pt")
    # Save metadata
    metadata.update({
        "x_vector_only_mode": prompt_item.x_vector_only_mode,
        "icl_mode": prompt_item.icl_mode,
        "ref_text": prompt_item.ref_text,
    })
    (vdir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2))


def _load_voice_prompt(voice_id: str):
    """Load a cached VoiceClonePromptItem from disk (x_vector_only for fast TTS)."""
    from qwen_tts.inference.qwen3_tts_model import VoiceClonePromptItem

    vdir = _voice_dir(voice_id)
    if not vdir.exists():
        return None
    ref_spk = torch.load(vdir / "ref_spk_embedding.pt", map_location=device, weights_only=True)
    return VoiceClonePromptItem(
        ref_code=None,
        ref_spk_embedding=ref_spk,
        x_vector_only_mode=True,
        icl_mode=False,
        ref_text=None,
    )


def _list_voices() -> list:
    """List all saved voice profiles with their metadata."""
    voices = []
    if not VOICES_DIR.exists():
        return voices
    for vdir in sorted(VOICES_DIR.iterdir()):
        if vdir.is_dir() and (vdir / "metadata.json").exists():
            meta = json.loads((vdir / "metadata.json").read_text())
            meta["id"] = vdir.name
            voices.append(meta)
    return voices


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences for chunked generation.

    Keeps sentences together if they're very short (<20 chars) to avoid
    generating tiny audio fragments with bad prosody.
    """
    # Split on sentence-ending punctuation followed by whitespace
    raw = re.split(r'(?<=[.!?;:])\s+', text.strip())
    if not raw:
        return [text]

    # Merge very short fragments with the next sentence
    merged = []
    buf = ""
    for s in raw:
        if buf:
            buf += " " + s
        else:
            buf = s
        if len(buf) >= 20:
            merged.append(buf)
            buf = ""
    if buf:
        if merged:
            merged[-1] += " " + buf
        else:
            merged.append(buf)
    return merged


def _generate_chunks(model, sentences: list[str], language: str, voice_clone_prompt, sample_rate: int = 24000) -> tuple[np.ndarray, int]:
    """Generate audio for sentences using batch mode for speed, then concatenate with gaps."""
    # Batch mode: generate all sentences at once (2x+ faster than sequential)
    prompt_item = voice_clone_prompt[0] if voice_clone_prompt else None
    prompts = [prompt_item] * len(sentences)
    langs = [language] * len(sentences)

    logger.info(f"  Generating {len(sentences)} sentences in batch mode...")
    start = time.time()
    wavs, sr = model.generate_voice_clone(
        text=sentences,
        language=langs,
        voice_clone_prompt=prompts,
    )
    elapsed = time.time() - start
    logger.info(f"  Batch generation done in {elapsed:.2f}s")

    # Concatenate with gaps. The gap is sized from the rate the model actually
    # returned, not the 24000 default — a model at any other rate produced
    # audibly wrong pauses.
    gap = np.zeros(int((sr or sample_rate) * 0.15), dtype=np.float32)  # 150ms
    all_audio = []
    for i, wav in enumerate(wavs):
        all_audio.append(np.array(wav))
        if i < len(wavs) - 1:
            all_audio.append(gap)
    del wavs

    return np.concatenate(all_audio), sr


@app.get("/voices")
async def list_saved_voices():
    """List all saved voice profiles."""
    return {"voices": _list_voices()}


@app.post("/voices/save")
async def save_voice(
    name: str = Form(...),
    lang: str = Form("auto"),
    file: UploadFile = File(...),
):
    """Upload reference audio, extract speaker prompt, and save for reuse.
    Trims audio to best segment via ASR timestamps + ffmpeg."""
    # Validate the name up front so a bad name fails fast, before the
    # expensive transcription + embedding work.
    voice_id = _voice_id_from_name(name)

    tmp_path = None
    trimmed_path = None
    try:
        # Ensure the model is loaded before checking capabilities, otherwise a
        # cold start (failed preload) reports a misleading capability error.
        model, model_name = await _acquire_model()
        model_info = AVAILABLE_MODELS.get(model_name, {})
        if "voice_clone" not in model_info.get("capabilities", []):
            raise HTTPException(status_code=400, detail="Current model does not support voice cloning. Switch to a Base model.")
        start_time = time.time()

        # Save uploaded file
        suffix = Path(file.filename).suffix if file.filename else ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        # Auto-transcribe to get text + segments for trimming
        asr_result = await _auto_transcribe(tmp_path, file.filename or "reference.wav")
        ref_text = asr_result.get("text", "").strip()
        segments = asr_result.get("segments", [])

        if not ref_text:
            raise HTTPException(status_code=400, detail="Could not transcribe reference audio. Please try a clearer recording.")

        # Trim to best segment using ffmpeg
        audio_path = tmp_path
        segment_text = ref_text
        best = _pick_best_segment(segments)
        if best and best.get("start") is not None:
            trimmed_path = tmp_path + "_trimmed.wav"
            if _trim_audio_segment(tmp_path, best["start"], best["end"], trimmed_path):
                audio_path = trimmed_path
                segment_text = best.get("text", ref_text)
                dur = best["end"] - best["start"]
                logger.info(f"Trimmed reference to {dur:.1f}s segment: '{segment_text[:60]}...'")

        # Extract speaker embedding (x_vector_only for fast TTS); off the event loop
        prompt_items = await _gen(
            model.create_voice_clone_prompt,
            ref_audio=audio_path,
            x_vector_only_mode=True,
        )
        prompt_item = prompt_items[0]

        # Save to disk
        _save_voice_prompt(voice_id, prompt_item, {
            "name": name,
            "lang": lang,
            "original_filename": file.filename,
            "ref_text": segment_text,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "model_used": current_model_name,
        })

        elapsed = time.time() - start_time
        logger.info(f"Voice '{name}' saved as '{voice_id}' in {elapsed:.1f}s")

        return {
            "status": "ok",
            "voice_id": voice_id,
            "name": name,
            "ref_text": segment_text,
            "processing_time": round(elapsed, 2),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Save voice error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        _cleanup_temp(tmp_path)
        _cleanup_temp(trimmed_path)


@app.delete("/voices/{voice_id}")
async def delete_voice(voice_id: str):
    """Delete a saved voice profile."""
    import shutil
    vdir = _voice_dir(voice_id)
    if not vdir.exists():
        raise HTTPException(status_code=404, detail=f"Voice '{voice_id}' not found")
    shutil.rmtree(vdir)
    return {"status": "ok", "deleted": voice_id}


@app.post("/voices/{voice_id}/tts")
async def tts_with_saved_voice(
    voice_id: str,
    text: str = Form(...),
    lang: str = Form("English"),
):
    """Generate speech using a saved voice profile. Skips audio processing — fast."""
    prompt_item = _load_voice_prompt(voice_id)
    if prompt_item is None:
        raise HTTPException(status_code=404, detail=f"Voice '{voice_id}' not found")

    if not text:
        raise HTTPException(status_code=400, detail="Text not provided")

    try:
        # Load the model before the capability check so a cold start
        # (failed preload) doesn't report a misleading capability error.
        model, model_name = await _acquire_model()
        model_info = AVAILABLE_MODELS.get(model_name, {})
        if "voice_clone" not in model_info.get("capabilities", []):
            raise HTTPException(status_code=400, detail="Current model does not support voice cloning. Switch to a Base model.")
        start_time = time.time()

        sentences = _split_sentences(text)
        logger.info(f"Saved-voice TTS: {len(sentences)} chunks for voice={voice_id}")

        if len(sentences) > 1:
            audio, sr = await _gen(_generate_chunks, model, sentences, lang, [prompt_item])
        else:
            wavs, sr = await _gen(
                model.generate_voice_clone,
                text=text,
                language=lang,
                voice_clone_prompt=[prompt_item],
            )
            audio = np.array(wavs[0])
            del wavs

        generation_time = time.time() - start_time
        audio_duration = len(audio) / sr
        logger.info(f"Saved-voice TTS done in {generation_time:.2f}s ({audio_duration:.1f}s audio, voice={voice_id})")

        buffer = io.BytesIO()
        sf.write(buffer, audio, sr, format="WAV")
        buffer.seek(0)

        del audio
        # No empty_cache()/gc.collect() here: on a response path they force a
        # device synchronise plus a full gen-2 GC, and discarding the caching
        # allocator makes the *next* generation re-pay cudaMalloc.

        return StreamingResponse(
            buffer,
            media_type="audio/wav",
            headers={
                "X-Generation-Time": f"{generation_time:.3f}",
                "X-Audio-Duration": f"{audio_duration:.3f}",
                "X-Voice-ID": voice_id,
                "X-Language": lang,
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Saved-voice TTS error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


class TTSRequest(BaseModel):
    """Request body for built-in-speaker speech synthesis."""

    text: str
    lang: str = "English"
    speaker: str = "Vivian"
    instruct: str = ""


@app.post("/tts")
async def text_to_speech(request: TTSRequest):
    """Generate speech using a built-in speaker voice."""
    if not request.text:
        raise HTTPException(status_code=400, detail="Text not provided")

    try:
        model, model_name = await _acquire_model()
        start_time = time.time()

        # Route on declared capabilities: Base models expose a
        # generate_custom_voice attribute but raise when it is called,
        # so hasattr() alone picks the wrong path.
        model_info = AVAILABLE_MODELS.get(model_name, {})
        if "custom_voice" in model_info.get("capabilities", []) and hasattr(model, 'generate_custom_voice'):
            wavs, sr = await _gen(
                model.generate_custom_voice,
                text=request.text,
                language=request.lang,
                speaker=request.speaker,
                instruct=request.instruct or "",
            )
        else:
            # Base models have no built-in speakers. Attempt reference-free
            # cloning for library versions that support it, but translate the
            # (likely) failure into actionable guidance instead of a raw 500.
            try:
                wavs, sr = await _gen(
                    model.generate_voice_clone,
                    text=request.text,
                    language=request.lang,
                    ref_audio=None,
                    ref_text="",
                )
            except Exception as fallback_err:
                model_info = AVAILABLE_MODELS.get(model_name, {})
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Current model '{model_info.get('name', current_model_name)}' has no built-in "
                        "speakers. Switch to the CustomVoice model for speaker TTS, or use voice "
                        f"cloning / a saved voice instead. (Underlying error: {fallback_err})"
                    ),
                )

        generation_time = time.time() - start_time
        logger.info(f"TTS generated in {generation_time:.2f}s")

        # Convert to WAV bytes
        buffer = io.BytesIO()
        sf.write(buffer, np.array(wavs[0]), sr, format="WAV")
        buffer.seek(0)

        # Memory cleanup
        del wavs
        # No empty_cache()/gc.collect() here: on a response path they force a
        # device synchronise plus a full gen-2 GC, and discarding the caching
        # allocator makes the *next* generation re-pay cudaMalloc.

        return StreamingResponse(
            buffer,
            media_type="audio/wav",
            headers={
                "X-Generation-Time": f"{generation_time:.3f}",
                "X-Speaker": request.speaker,
                "X-Language": request.lang,
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"TTS error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/clone")
async def clone_voice(
    text: str = Form(...),
    lang: str = Form("English"),
    file: UploadFile = File(...),
):
    """
    Clone a voice from a reference audio file (3+ seconds recommended).
    Provide reference audio as 'file' and the text to synthesize as 'text'.
    """
    if not text:
        raise HTTPException(status_code=400, detail="Text not provided")

    tmp_path = None
    try:
        model, model_name = await _acquire_model()

        model_info = AVAILABLE_MODELS.get(model_name, {})
        capabilities = model_info.get("capabilities", [])
        if "voice_clone" not in capabilities:
            raise HTTPException(
                status_code=400,
                detail=f"Current model '{model_info.get('name', current_model_name)}' does not support voice cloning. "
                       f"Please switch to a Base model (1.7B Base or 0.6B Base)."
            )
        start_time = time.time()

        # Save uploaded reference audio
        suffix = Path(file.filename).suffix if file.filename else ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        logger.info(f"Voice clone request: text='{text[:50]}...', lang={lang}, ref={file.filename}")

        # NOTE: this path deliberately does *not* auto-transcribe the reference.
        # It clones via `x_vector_only_mode=True` (speaker embedding only), which
        # never consumes a reference transcript — the previous full Qwen3-ASR-1.7B
        # round trip produced a string that was then never read.

        # Extract speaker embedding first (fast), then use it for chunked generation
        try:
            prompt_items = await _gen(
                model.create_voice_clone_prompt,
                ref_audio=tmp_path,
                x_vector_only_mode=True,
            )
            prompt_item = prompt_items[0] if isinstance(prompt_items, list) else prompt_items
        except Exception as clone_err:
            err_msg = str(clone_err)
            if "does not support generate_voice_clone" in err_msg:
                raise HTTPException(
                    status_code=400,
                    detail=f"Current model does not support voice cloning. "
                           f"Please switch to a Base model (1.7B Base or 0.6B Base)."
                )
            raise

        # Chunked generation for long texts
        sentences = _split_sentences(text)
        logger.info(f"Voice clone: {len(sentences)} chunks, ref={file.filename}")

        try:
            if len(sentences) > 1:
                audio, sr = await _gen(_generate_chunks, model, sentences, lang, [prompt_item])
            else:
                wavs, sr = await _gen(
                    model.generate_voice_clone,
                    text=text,
                    language=lang,
                    voice_clone_prompt=[prompt_item],
                )
                audio = np.array(wavs[0])
                del wavs
        except Exception as gen_err:
            err_msg = str(gen_err)
            if "ref_text" in err_msg.lower() or "icl mode" in err_msg.lower():
                raise HTTPException(
                    status_code=400,
                    detail="Voice cloning requires reference text but auto-transcription failed. "
                           "Please enable 'Provide reference text' and enter the transcript manually."
                )
            raise

        generation_time = time.time() - start_time
        audio_duration = len(audio) / sr
        logger.info(f"Voice clone done in {generation_time:.2f}s ({audio_duration:.1f}s audio)")

        # Convert to WAV
        buffer = io.BytesIO()
        sf.write(buffer, audio, sr, format="WAV")
        buffer.seek(0)

        # Cleanup
        del audio
        # No empty_cache()/gc.collect() here: on a response path they force a
        # device synchronise plus a full gen-2 GC, and discarding the caching
        # allocator makes the *next* generation re-pay cudaMalloc.

        return StreamingResponse(
            buffer,
            media_type="audio/wav",
            headers={
                "X-Generation-Time": f"{generation_time:.3f}",
                "X-Audio-Duration": f"{audio_duration:.3f}",
                # Header values must be latin-1 encodable; an umlaut in the
                # reference filename used to turn a finished clone into a 500.
                "X-Clone-Source": _safe_header(file.filename or "unknown"),
                "X-Language": lang,
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Voice clone error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        _cleanup_temp(tmp_path)


@app.post("/clone-with-ref-text")
async def clone_voice_with_ref_text(
    text: str = Form(...),
    ref_text: str = Form(...),
    lang: str = Form("English"),
    file: UploadFile = File(...),
):
    """
    High-quality voice cloning with reference text.
    Provide reference audio + its transcript for best results.
    """
    if not text:
        raise HTTPException(status_code=400, detail="Text not provided")
    if not ref_text:
        raise HTTPException(status_code=400, detail="Reference text not provided")

    tmp_path = None
    try:
        model, model_name = await _acquire_model()
        start_time = time.time()

        suffix = Path(file.filename).suffix if file.filename else ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        wavs, sr = await _gen(
            model.generate_voice_clone,
            text=text,
            language=lang,
            ref_audio=tmp_path,
            ref_text=ref_text,
        )

        generation_time = time.time() - start_time
        logger.info(f"High-quality voice clone generated in {generation_time:.2f}s")

        buffer = io.BytesIO()
        sf.write(buffer, np.array(wavs[0]), sr, format="WAV")
        buffer.seek(0)

        del wavs
        # No empty_cache()/gc.collect() here: on a response path they force a
        # device synchronise plus a full gen-2 GC, and discarding the caching
        # allocator makes the *next* generation re-pay cudaMalloc.

        return StreamingResponse(
            buffer,
            media_type="audio/wav",
            headers={
                "X-Generation-Time": f"{generation_time:.3f}",
                "X-Language": lang,
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"High-quality clone error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        _cleanup_temp(tmp_path)


class VoiceDesignRequest(BaseModel):
    """Request body for text-guided voice design synthesis."""

    text: str
    voice_description: str
    lang: str = "English"


@app.post("/voice_design")
async def voice_design(request: VoiceDesignRequest):
    """
    Generate speech with a voice designed from a text description.
    Requires the VoiceDesign model to be loaded.
    Example description: "A deep male voice with a warm, calm British accent"
    """
    if not request.text:
        raise HTTPException(status_code=400, detail="Text not provided")
    if not request.voice_description:
        raise HTTPException(status_code=400, detail="Voice description not provided")

    try:
        model, model_name = await _acquire_model()
        start_time = time.time()

        if hasattr(model, 'generate_voice_design'):
            wavs, sr = await _gen(
                model.generate_voice_design,
                text=request.text,
                language=request.lang,
                instruct=request.voice_description,
            )
        else:
            raise HTTPException(
                status_code=400,
                detail="Current model does not support voice design. Please switch to the VoiceDesign model."
            )

        generation_time = time.time() - start_time
        logger.info(f"Voice design generated in {generation_time:.2f}s")

        buffer = io.BytesIO()
        sf.write(buffer, np.array(wavs[0]), sr, format="WAV")
        buffer.seek(0)

        del wavs
        # No empty_cache()/gc.collect() here: on a response path they force a
        # device synchronise plus a full gen-2 GC, and discarding the caching
        # allocator makes the *next* generation re-pay cudaMalloc.

        return StreamingResponse(
            buffer,
            media_type="audio/wav",
            headers={
                "X-Generation-Time": f"{generation_time:.3f}",
                "X-Language": request.lang,
                "X-Voice-Description": _safe_header(request.voice_description[:100]),
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Voice design error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5004)
