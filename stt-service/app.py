"""Speech-to-Text service powered by faster-whisper.

Supports CUDA, ROCm, and CPU backends with automatic hardware detection.
Provides batch transcription, streaming SSE, and language detection endpoints.
"""

from fastapi import FastAPI, UploadFile, HTTPException, File, Form, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse, JSONResponse
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
from faster_whisper import WhisperModel
from typing import Any, Union, List
from contextlib import asynccontextmanager
import torch
import os
import logging
import tempfile
import io
import json
import asyncio
import time
import uuid
import gc
import threading
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor

from json_utils import clean_json_inf_nan, ENGLISH_ONLY_MODELS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    """Load the Whisper model on startup; release the executors on shutdown."""
    load_model()
    warm_up_model()
    yield
    logger.info("Shutting down thread pool executors...")
    rt_executor.shutdown(wait=False)
    executor.shutdown(wait=False)


app = FastAPI(
    title="STT Service",
    description="Speech-to-Text Service with Hardware Acceleration",
    lifespan=_lifespan,
)

# Add CORS middleware (env-configurable)
allowed_origins_str = os.getenv("ALLOWED_ORIGINS", "*")
allowed_origins = [o.strip() for o in allowed_origins_str.split(",")] if allowed_origins_str else ["*"]
allow_credentials = os.getenv("ALLOW_CREDENTIALS", "false").strip().lower() in {"1", "true", "yes", "on"}
if "*" in allowed_origins and allow_credentials:
    logger.warning("ALLOW_CREDENTIALS=true with ALLOWED_ORIGINS='*' is not permitted by CORS spec; disabling credentials.")
    allow_credentials = False

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,  # Allows configured origins
    allow_credentials=allow_credentials,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Two pools so a long batch upload cannot starve live sessions of a worker.
# CTranslate2 itself serialises on the shared model (see WHISPER_NUM_WORKERS), so
# these bound queueing, not GPU parallelism.
#
# The realtime pool must have a slot per admitted session, plus one. Cancelling
# an asyncio task that is awaiting run_in_executor does NOT stop the worker
# thread — the decode runs to completion regardless — so an abandoned interim
# keeps occupying its slot until it finishes. Sized at 2 with WS_MAX_SESSIONS=4,
# two stopping sessions could freeze the other two.
_MAX_LIVE_SESSIONS = int(os.getenv("WS_MAX_SESSIONS", "4"))
rt_executor = ThreadPoolExecutor(
    max_workers=max(2, _MAX_LIVE_SESSIONS + 1), thread_name_prefix="whisper-rt"
)
executor = ThreadPoolExecutor(max_workers=os.cpu_count(), thread_name_prefix="whisper-batch")

class SafeJSONResponse(JSONResponse):
    """JSONResponse that sanitises NaN/Infinity before encoding."""
    def render(self, content: Any) -> bytes:
        """Serialise JSON after normalising unsupported float values."""
        return json.dumps(
            clean_json_inf_nan(content),
            ensure_ascii=False,
            allow_nan=False, # This is the default, but being explicit
            indent=None,
            separators=(",", ":"),
        ).encode("utf-8")

def _select_cuda_compute_type() -> str:
    """Pick the CT2 compute type for this GPU.

    Defaults to int8_float16: faster-whisper's own table measures int8 at ~35%
    less VRAM than float16 at equal-or-better speed, which is the difference
    between fitting and not fitting on an 8-12 GB card. Set WHISPER_COMPUTE_TYPE
    to pin it (float16, int8_float16, int8_bfloat16, int8, float32).

    int8_bfloat16 is deliberately NOT auto-selected below compute capability 8.0:
    on Turing, CTranslate2 silently falls back to int8_float32, which costs
    activation memory rather than saving it.
    """
    requested = os.getenv("WHISPER_COMPUTE_TYPE", "").strip().lower()

    supported = set()
    try:
        import ctranslate2
        supported = set(ctranslate2.get_supported_compute_types("cuda"))
    except Exception as e:
        logger.warning(f"Could not probe CT2 compute types ({e}); assuming float16 is safe")

    if requested and requested != "auto":
        if supported and requested not in supported:
            logger.warning(
                f"WHISPER_COMPUTE_TYPE={requested} is not supported by this GPU "
                f"(supported: {sorted(supported)}); falling back to auto-selection"
            )
        else:
            return requested

    # Most memory-efficient first; every entry must be verified as supported.
    preference = ["int8_float16", "int8", "float16", "float32"]
    try:
        if torch.cuda.get_device_capability(0)[0] >= 8:
            preference.insert(0, "int8_bfloat16")
    except Exception:
        pass

    for candidate in preference:
        if not supported or candidate in supported:
            return candidate
    return "float16"


# Hardware detection and optimization (re-check after startup)
def detect_hardware():
    """Auto-detect the best compute device (CUDA > ROCm > CPU) and return ``(device, compute_type)``."""
    import torch

    force_accel = os.getenv("FORCE_ACCELERATION", "").lower()

    if force_accel == "rocm":
        # ROCm presents as CUDA to PyTorch; ctranslate2 4.x+ required for GPU use.
        # If ctranslate2 was not compiled with HIP support, fall back gracefully to CPU.
        try:
            import ctranslate2
            providers = ctranslate2.get_supported_compute_types("cuda")
            if "float16" in providers:
                device = "cuda"
                compute_type = "float16"
                logger.info(f"ROCm mode: using GPU (ctranslate2 HIP). GPU: {torch.cuda.get_device_name(0)}")
            else:
                raise RuntimeError("ctranslate2 has no float16 CUDA/HIP support")
        except Exception as e:
            logger.warning(f"ROCm requested but ctranslate2 HIP unavailable ({e}); falling back to CPU int8")
            device = "cpu"
            compute_type = "int8"
        return device, compute_type

    # Explicit CPU override
    if os.getenv("USE_CUDA", "").lower() == "false":
        logger.info("Hardware acceleration disabled via USE_CUDA=false")
        return "cpu", "int8"

    if torch.cuda.is_available():
        device = "cuda"
        compute_type = _select_cuda_compute_type()
        logger.info(f"CUDA available with {torch.cuda.device_count()} GPU(s)")
        logger.info(f"GPU: {torch.cuda.get_device_name(0)} using {compute_type}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = "cpu"  # faster-whisper doesn't support MPS
        compute_type = "int8"
        logger.info("Apple Silicon detected, using optimized CPU int8")
    else:
        device = "cpu"
        compute_type = "int8"
        logger.info("Using CPU int8")

    return device, compute_type

# Defer hardware detection until first use
device = None
compute_type = None

# large-v3-turbo has 4 decoder layers against large-v3's 32 at comparable accuracy,
# which dominates greedy realtime decoding. FALLBACK_MODEL_SIZE is used if the
# configured name cannot be resolved, so an unknown alias cannot brick startup.
DEFAULT_MODEL_SIZE = "large-v3-turbo"
FALLBACK_MODEL_SIZE = "large-v3"
# Used only when a load fails for resource reasons, so it must be SMALLER than
# the default and still multilingual. `small` is ~0.5 GB against turbo's ~1.6 GB.
# Never use distil-large-v3 here: it is English-only despite having no `.en`
# suffix, so it would silently drop German on exactly the constrained devices
# this ladder exists to serve.
OOM_FALLBACK_MODEL_SIZE = os.getenv("WHISPER_OOM_FALLBACK", "small")

# Approximate weight sizes in MB, used only to keep the fallback ladder
# descending. Multilingual models only — English-only variants (*.en,
# distil-large-v3) are excluded on purpose so they can never be auto-selected
# and silently drop German.
KNOWN_MODEL_SIZES = {
    "tiny": 75,
    "base": 145,
    "small": 490,
    "medium": 1500,
    "large-v1": 2900,
    "large-v2": 2900,
    "large-v3": 3100,
    "large-v3-turbo": 1600,
    "turbo": 1600,
}


def _model_rank(name: str) -> int:
    """Approximate memory cost, for ordering fallbacks.

    Unknown names sort as very large so an unrecognised model is never treated
    as a safe step down.
    """
    return KNOWN_MODEL_SIZES.get(name, 10_000)
# Models that only ever emit English and therefore cannot serve task="translate".
TURBO_MODELS = {"large-v3-turbo", "turbo"}


def configured_model_size() -> str:
    """The model name requested via env, or the project default."""
    return os.getenv("WHISPER_MODEL_SIZE", "").strip() or DEFAULT_MODEL_SIZE


# Initialize Whisper model with error handling
whisper_model = None
model_loaded = False
model_size_loaded = None
model_warmed = False
startup_error = None

# Reference counting for the shared model.
#
# Until POST /unload existed the model was loaded once at startup and never
# dropped, so reading the `whisper_model` global at call time was always safe.
# It is not any more: several transcribe paths read the global directly inside a
# worker thread, and an unload landing between the check and the read would give
# them None. Callers therefore take a reference for the duration of their work,
# and unload refuses while any reference is outstanding rather than pulling the
# model out from under a request in flight.
_model_refs = 0
_model_ref_lock = threading.Lock()


def acquire_model():
    """Take a reference to the Whisper model, loading it on demand.

    Returns the model object — use the returned value, never the global, or the
    race this exists to close reopens. Loading on demand is what makes idle
    unloading transparent to clients.

    Every call must be paired with exactly one release_model(); prefer
    model_in_use() unless the work outlives the calling frame.
    """
    global _model_refs

    with _model_ref_lock:
        model = whisper_model
        if model is None:
            # Released by unload, or never loaded. load_model() sets the global.
            load_model()
            model = whisper_model
            if model is None:
                raise HTTPException(status_code=503, detail="Model not available")
        _model_refs += 1
    return model


def release_model():
    """Drop one reference taken by acquire_model()."""
    global _model_refs
    with _model_ref_lock:
        if _model_refs > 0:
            _model_refs -= 1


@contextmanager
def model_in_use():
    """Hold the Whisper model for the duration of a block."""
    model = acquire_model()
    try:
        yield model
    finally:
        release_model()


def unload_model() -> dict:
    """Drop the Whisper model and free its memory. Safe to call when unloaded.

    Returns a result dict rather than raising, so the caller decides the status
    code. Refuses while a transcription holds a reference — the alternative is
    freeing memory that a running decode is still reading from.
    """
    global whisper_model, model_loaded, model_warmed

    with _model_ref_lock:
        if _model_refs > 0:
            return {"unloaded": False, "reason": "busy", "refs": _model_refs}
        if whisper_model is None:
            return {"unloaded": False, "reason": "not_resident", "refs": 0}

        whisper_model = None
        model_loaded = False
        model_warmed = False

    # CTranslate2 owns its device memory directly and frees it when the object
    # is finalised, so the collect is what actually returns the VRAM — there is
    # no torch caching allocator in this path for empty_cache() to drain.
    gc.collect()
    logger.info("Whisper model unloaded; memory released")
    return {"unloaded": True, "reason": "ok", "refs": 0}

def ensure_hardware_detected():
    """Run hardware detection once (lazy initialisation)."""
    global device, compute_type
    if device is None:
        device, compute_type = detect_hardware()

def load_model():
    """Load the Whisper model, falling back to a known model name then to CPU int8."""
    global whisper_model, model_loaded, model_size_loaded, device, compute_type, startup_error

    # Ensure hardware is detected first
    ensure_hardware_detected()

    # CTranslate2 serialises calls per worker; 2 gives live and batch traffic
    # independent slots without doubling activation memory for every request.
    num_workers = max(1, int(os.getenv("WHISPER_NUM_WORKERS", "2")))
    # cpu_threads maps to CT2's intra_threads and num_workers to inter_threads,
    # and CT2 spawns inter_threads model replicas each running intra_threads
    # threads — so os.cpu_count() here would demand num_workers x cpu_count.
    cpu_threads = max(1, (os.cpu_count() or 4) // num_workers)
    requested = configured_model_size()

    # (model_size, device, compute_type). Resource use must never INCREASE down
    # the ladder — responding to a failure by asking for more memory cannot help.
    #
    # The two failure modes need different responses, so they are separated by
    # inspecting the requested name rather than by inspecting the exception:
    #   - unknown name  -> fall back to a known-good name (a size step is fine,
    #                      because the requested model does not exist at all).
    #   - anything else -> step DOWN in size, then off the GPU.
    attempts = [(requested, device, compute_type)]

    if requested not in KNOWN_MODEL_SIZES:
        # Name-resolution safety net: an unknown alias must not brick startup.
        # Only reachable when `requested` is not a real model, so this cannot
        # escalate memory for a working configuration.
        attempts.append((FALLBACK_MODEL_SIZE, device, compute_type))

    if device != "cpu":
        # Smaller multilingual model before abandoning the GPU entirely, but
        # only if it is genuinely smaller than what was asked for.
        # `small` is multilingual — do NOT use distil-large-v3 here, it is
        # English-only and would silently drop German.
        if _model_rank(OOM_FALLBACK_MODEL_SIZE) < _model_rank(requested):
            attempts.append((OOM_FALLBACK_MODEL_SIZE, device, "int8_float16"))
        attempts.append((requested, "cpu", "int8"))
        if _model_rank(OOM_FALLBACK_MODEL_SIZE) < _model_rank(requested):
            attempts.append((OOM_FALLBACK_MODEL_SIZE, "cpu", "int8"))

    last_error = None
    for model_size, dev, ctype in attempts:
        try:
            logger.info(f"Loading {model_size} Whisper model on {dev} with {ctype}...")
            whisper_model = WhisperModel(
                model_size,
                device=dev,
                compute_type=ctype,
                cpu_threads=cpu_threads if dev == "cpu" else 4,
                num_workers=num_workers,
            )
            # Keep the reported runtime honest — /health used to keep advertising
            # cuda/float16 after silently falling back to CPU.
            device, compute_type = dev, ctype
            model_size_loaded = model_size
            model_loaded = True
            startup_error = None
            logger.info(f"Model loaded successfully: {model_size} on {dev}/{ctype}")
            return
        except Exception as e:
            last_error = e
            logger.error(f"Failed to load {model_size} on {dev}/{ctype}: {e}")

    model_loaded = False
    startup_error = str(last_error)
    logger.error(f"All model load attempts failed; service is unhealthy: {last_error}")


def warm_up_model():
    """Run one throwaway inference so the first real request doesn't pay autotune.

    Uses low-amplitude noise with ``vad_filter=False`` — silence plus VAD short-
    circuits before any kernel runs and would warm nothing. The generator must be
    consumed or no compute happens at all.
    """
    global model_warmed
    model = whisper_model
    if not model_loaded or model is None:
        return
    try:
        t0 = time.monotonic()
        rng = np.random.default_rng(0)
        # cuDNN autotune is keyed on input shape; warm the short and long cases.
        for seconds in (1.0, 5.0):
            noise = (rng.standard_normal(int(seconds * WS_SAMPLE_RATE)) * 1e-3).astype(np.float32)
            segments, _info = model.transcribe(
                noise, language="en", beam_size=1, best_of=1,
                temperature=0.0, without_timestamps=True, vad_filter=False,
            )
            list(segments)
        model_warmed = True
        logger.info(f"Model warm-up complete in {time.monotonic() - t0:.2f}s")
    except Exception as e:
        logger.warning(f"Model warm-up failed (first request will be slower): {e}")


def _reject_unsupported_translate(task: str):
    """Reject task='translate' on turbo models, which have no translate capability.

    Whisper's turbo variants were distilled on transcription only; asking them to
    translate silently returns source-language text instead of English.
    """
    if task == "translate" and (model_size_loaded or configured_model_size()) in TURBO_MODELS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Model '{model_size_loaded}' does not support translation. "
                "Set WHISPER_MODEL_SIZE to large-v3 (or another non-turbo model) to use task=translate."
            ),
        )

# Add better error handling and logging
@app.post("/transcribe")
async def transcribe_audio(
    # Accept either a single file (legacy) or multiple files (batch)
    audio: UploadFile = File(None),
    audios: List[UploadFile] = File(None),
    task: str = Form("transcribe"),
    language: str = Form("auto"),
    beam_size: int = Form(5),
    best_of: int = Form(5),
    patience: float = Form(1.0),
    temperature: Union[float, str] = Form("0.0,0.2,0.4,0.6,0.8,1.0"),
    suppress_tokens: str = Form("-1"),
    initial_prompt: str = Form(""),
    condition_on_previous_text: bool = Form(True),
    compression_ratio_threshold: float = Form(2.4),
    no_speech_threshold: float = Form(0.6),
    vad_filter: bool = Form(True),
    vad_threshold: float = Form(0.5),
):
    """Transcribe one or more audio files to text.

    Supports single-file (``audio``) and multi-file (``audios``) uploads.
    Returns per-file results with segments, language, and timing information.
    """
    logger.info(f"Transcription request - task: {task}, language: {language}")

    _reject_unsupported_translate(task)

    try:
        # Normalize file list
        file_list: List[UploadFile] = []
        if audios:
            file_list.extend(audios)
        if audio:
            file_list.append(audio)
        if not file_list:
            raise HTTPException(status_code=400, detail="No audio file(s) provided")

        batch_mode = len(file_list) > 1
        batch_results = []
        combined_text_parts = []
        total_processing_time = 0.0
        total_audio_duration = 0.0

        # Parse temperature once
        try:
            if isinstance(temperature, str):
                temperature = [float(t) for t in temperature.split(",") if t.strip()]
            else:
                temperature = [float(temperature)]
            if not temperature:
                raise ValueError("empty temperature list")
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail="Invalid temperature value: expected a float or comma-separated floats",
            )

        # Parse suppress_tokens once
        parsed_suppress_tokens = None
        try:
            if isinstance(suppress_tokens, str):
                st = suppress_tokens.strip()
                if st and st != "-1":
                    parsed_suppress_tokens = [int(tok.strip()) for tok in st.split(",") if tok.strip()]
        except ValueError:
            logger.warning(f"Invalid suppress_tokens value '{suppress_tokens}', ignoring.")
            parsed_suppress_tokens = None

        # Process each file sequentially
        for afile in file_list:
            safe_filename = os.path.basename(afile.filename or "audio.wav")
            suffix = os.path.splitext(safe_filename)[1] or ".wav"
            tmp_fd, temp_audio_path = tempfile.mkstemp(suffix=suffix)
            with os.fdopen(tmp_fd, "wb") as f:
                content = await afile.read()
                f.write(content)
            logger.info(f"Saved audio file: {temp_audio_path}, size: {os.path.getsize(temp_audio_path)} bytes")

            logger.info(f"Starting transcription for file {afile.filename} (vad_filter={vad_filter}, vad_threshold={vad_threshold})...")
            start_time = time.time()

            def _do_transcribe(path=temp_audio_path):
                """Run faster-whisper + segment filtering in a worker thread.

                faster-whisper is blocking and the segment generator is consumed
                here, so this whole step runs off the event loop to keep /health
                and concurrent requests responsive during transcription.
                """
                # Hold a reference for the whole call: the segment generator
                # below is lazy, so an unload during iteration would free the
                # model mid-decode.
                with model_in_use() as model:
                    segments, info = model.transcribe(
                        path,
                        task=task,
                        language=None if language == "auto" else language,
                        beam_size=beam_size,
                        best_of=best_of,
                        patience=patience,
                        temperature=temperature,
                        suppress_tokens=parsed_suppress_tokens,
                        initial_prompt=initial_prompt,
                        condition_on_previous_text=condition_on_previous_text,
                        compression_ratio_threshold=compression_ratio_threshold,
                        no_speech_threshold=no_speech_threshold,
                        vad_filter=vad_filter,
                        vad_parameters={
                            "threshold": vad_threshold,
                            "min_speech_duration_ms": 500,
                            "min_silence_duration_ms": 1500,
                            "speech_pad_ms": 300,
                        } if vad_filter else None,
                    )

                    segs = []
                    full = ""
                    last = ""
                    processed = 0
                    duration = info.duration or 0
                    for segment in segments:
                        processed += 1
                        text = segment.text.strip()
                        seg_duration = segment.end - segment.start

                        if processed % 50 == 0:
                            if duration > 0:
                                progress_pct = min((segment.end / duration) * 100, 100)
                                logger.info(f"Progress: {processed} segments, {segment.end:.1f}s/{duration:.1f}s ({progress_pct:.1f}%)")
                            else:
                                logger.info(f"Progress: {processed} segments, current time: {segment.end:.1f}s")

                        if seg_duration < 0.2 or segment.no_speech_prob > 0.8:
                            continue
                        if text == last or not text:
                            continue
                        segs.append({
                            "start": segment.start,
                            "end": segment.end,
                            "text": text,
                            "avg_logprob": segment.avg_logprob,
                            "no_speech_prob": segment.no_speech_prob,
                        })
                        full += text + " "
                        last = text
                    return segs, full, processed, info

            loop = asyncio.get_running_loop()
            segments_list, full_text, processed_segments, info = await loop.run_in_executor(executor, _do_transcribe)
            total_duration = info.duration or 0

            processing_time = time.time() - start_time
            total_processing_time += processing_time
            total_audio_duration += info.duration or 0
            
            logger.info(f"Transcription completed for {afile.filename}")
            logger.info(f"Final stats: {len(segments_list)} valid segments created from {processed_segments} total segments")
            logger.info(f"Processing time: {processing_time:.2f}s for {total_duration:.1f}s audio")
            if processing_time > 0:
                logger.info(f"Speed: {total_duration/processing_time:.1f}x realtime")

            try:
                os.unlink(temp_audio_path)
            except Exception:
                pass

            batch_results.append({
                "filename": afile.filename,
                "text": full_text.strip(),
                "segments": segments_list,
                "language": info.language,
                "language_probability": info.language_probability,
                "duration": info.duration,
                "task": task,
                "processing_time": processing_time
            })
            combined_text_parts.append(full_text.strip())

        # Single file: preserve legacy response shape
        if not batch_mode:
            if not batch_results:
                raise HTTPException(status_code=422, detail="No transcription results (empty or unreadable audio)")
            return JSONResponse(content=batch_results[0])

        # Batch: new response shape
        if not batch_results:
            raise HTTPException(status_code=422, detail="No transcription results for any provided files")

        batch_response = {
            "batch": True,
            "file_count": len(batch_results),
            "combined_text": " \n".join([r["text"] for r in batch_results]).strip(),
            "results": batch_results,
            "total_processing_time": total_processing_time,
            "total_duration": total_audio_duration,
            "task": task,
            "language": batch_results[0]["language"],
            "language_probability": batch_results[0]["language_probability"],
        }
        return JSONResponse(content=batch_response)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Transcription error: {str(e)}", exc_info=True)
        # Clean up any temp file left behind
        try:
            if 'temp_audio_path' in locals() and os.path.exists(temp_audio_path):
                os.unlink(temp_audio_path)
        except OSError:
            pass
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

@app.post("/transcribe-stream")
async def transcribe_audio_stream(
    audio: UploadFile = File(...),
    language: str = Form(None),
    task: str = Form("transcribe"),
    target_language: str = Form("english"),
    beam_size: int = Form(5),
    vad_filter: bool = Form(True),
    vad_threshold: float = Form(0.5),
    no_speech_threshold: float = Form(0.6),
):
    """Stream transcription results via Server-Sent Events as segments are decoded."""
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Whisper model not loaded")

    _reject_unsupported_translate(task)

    # Match /transcribe semantics: "auto" (or empty) means auto-detect.
    # faster-whisper rejects "auto" as a language code.
    if not language or language.strip().lower() == "auto":
        language = None

    req_id = uuid.uuid4().hex[:8]
    
    # Save uploaded file to a temporary location first
    try:
        audio_content = await audio.read()
        suffix = os.path.splitext(audio.filename or "audio.wav")[1] or ".wav"
        tmp_fd, tmp_file_path = tempfile.mkstemp(suffix=suffix)
        with os.fdopen(tmp_fd, "wb") as tmp_file:
            tmp_file.write(audio_content)
        logger.info(f"[{req_id}] Saved temp file for streaming: {tmp_file_path}")
    except Exception as e:
        logger.error(f"[{req_id}] Failed to save temp file: {e}")
        raise HTTPException(status_code=500, detail="Failed to process uploaded file.")

    async def generate_transcription():
        """SSE generator that yields transcription segments as JSON events."""
        effective_target_lang = target_language  # capture outer scope value
        # Reference held across the whole stream, not just the transcribe() call:
        # the segment generator is lazy and is pulled one item at a time by the
        # loop below, so the model must stay resident until the last yield.
        # Released in the finally, including when the client disconnects early.
        model = None
        try:
            logger.info(f"[{req_id}] /transcribe-stream request received: filename={audio.filename}, content_type={audio.content_type}, language={language}, task={task}")
            logger.info(f"[{req_id}] Uploaded audio size: {len(audio_content)} bytes")
            
            # For translation task, ensure target language is supported
            if task == "translate" and effective_target_lang.lower() not in ["english", "en"]:
                yield f"data: {json.dumps({'warning': f'Translation to {effective_target_lang} not supported, using English'})}\n\n"
                effective_target_lang = "english"
            
            # Start transcription
            yield f"data: {json.dumps({'status': 'processing', 'task': task})}\n\n"
            logger.info(f"[{req_id}] Starting streaming transcription...")
            
            model = acquire_model()

            # Run transcription in executor to avoid blocking
            def run_transcription():
                """Execute the blocking whisper transcription call inside a thread."""
                return model.transcribe(
                    tmp_file_path,
                    beam_size=beam_size,
                    best_of=beam_size,
                    temperature=(0.0, 0.2, 0.4, 0.6, 0.8),
                    compression_ratio_threshold=2.4,
                    no_speech_threshold=no_speech_threshold,
                    language=language,
                    task=task,
                    vad_filter=vad_filter,
                    vad_parameters={
                        "threshold": vad_threshold,
                        "min_speech_duration_ms": 500,
                        "min_silence_duration_ms": 1500,
                        "speech_pad_ms": 300,
                    } if vad_filter else None,
                )
            
            loop = asyncio.get_running_loop()
            segments, info = await loop.run_in_executor(executor, run_transcription)
            
            # Send metadata first
            metadata = {
                "language": info.language,
                "language_probability": info.language_probability,
                "duration": info.duration,
                "task": task
            }
            if task == "translate":
                metadata["target_language"] = effective_target_lang
            
            yield f"data: {json.dumps(clean_json_inf_nan({'metadata': metadata}))}\n\n"
            
            # Stream segments as they're processed.
            # model.transcribe returns a *lazy* generator: the encoder and
            # decoder run on iteration, not on the call above. Iterating it here
            # would run that work on the event loop and freeze every concurrent
            # live WebSocket session (and /health) for the file's duration, so each
            # step is pulled through the executor instead. The `None` sentinel is
            # required — StopIteration cannot propagate across a Future.
            full_text = ""
            segment_count = 0
            segment_iter = iter(segments)
            i = -1
            while True:
                segment = await loop.run_in_executor(executor, next, segment_iter, None)
                if segment is None:
                    break
                i += 1
                segment_count = i + 1
                segment_data = {
                    "segment_id": i,
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text.strip(),
                    "avg_logprob": segment.avg_logprob,
                    "no_speech_prob": segment.no_speech_prob
                }
                full_text += segment.text.strip() + " "
                
                yield f"data: {json.dumps(clean_json_inf_nan({'segment': segment_data}))}\n\n"
            
            # Send final result
            final_result = {
                "final_text": full_text.strip(),
                "status": "completed",
                "total_segments": segment_count
            }
            yield f"data: {json.dumps(final_result)}\n\n"
                
        except Exception as e:
            logger.error(f"[{req_id}] Streaming transcription error: {e}", exc_info=True)
            error_data = {"error": f"Transcription failed: {str(e)}", "status": "error"}
            yield f"data: {json.dumps(error_data)}\n\n"
        finally:
            if model is not None:
                release_model()
            # Cleanup temporary file
            if os.path.exists(tmp_file_path):
                try:
                    os.unlink(tmp_file_path)
                    logger.info(f"[{req_id}] Cleaned up temp file: {tmp_file_path}")
                except Exception as e:
                    logger.warning(f"[{req_id}] Could not delete temp file {tmp_file_path}: {e}")

    return StreamingResponse(
        generate_transcription(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )

# --- WebSocket live transcription ---

WS_SAMPLE_RATE = 16000          # required input rate (PCM16 little-endian mono)
# How much trailing audio each interim decode looks at.
#
# Note on cost, because it is easy to get wrong: faster-whisper pads the mel to
# a fixed 3000 frames (30 s) before the encoder, so the ENCODER pass costs the
# same whether this is 8 s or 25 s. What shrinking the window actually saves is
# decoder work — every tick regenerates all the tokens in the window from
# scratch (condition_on_previous_text=False), and that scales with the speech
# inside it. On turbo's 4 decoder layers that is a real but moderate saving,
# not the several-fold win a naive reading suggests.
#
# 8 s keeps enough left context for punctuation while bounding the token count.
# Raising it back toward 25 s costs less than it appears to and buys context.
WS_WINDOW_S = float(os.getenv("WS_WINDOW_S", "8.0"))
# Floor on how often a partial can be produced. The decode is single-flight, so
# this is a floor and not a schedule — decodes never queue up behind each other.
WS_MIN_NEW_AUDIO_S = float(os.getenv("WS_MIN_NEW_AUDIO_S", "0.5"))
WS_MAX_BUFFER_S = float(os.getenv("WS_MAX_BUFFER_S", "600.0"))
WS_MAX_SESSIONS = _MAX_LIVE_SESSIONS
# Interim hypotheses are greedy and unconditioned, so silence reliably produces
# hallucinated stock phrases. These are the same guards /transcribe already uses.
WS_NO_SPEECH_THRESHOLD = float(os.getenv("WS_NO_SPEECH_THRESHOLD", "0.6"))
WS_MIN_AVG_LOGPROB = float(os.getenv("WS_MIN_AVG_LOGPROB", "-1.0"))

_live_sessions = 0


def _ws_decode(audio: np.ndarray, language):
    """Greedy low-latency decode of a float32 16 kHz buffer (runs in the executor).

    Segments failing the no-speech / average-logprob guards are dropped: without
    them a hallucinated phrase repeats across consecutive windows and the
    agreement check below *promotes it to confirmed* precisely because it repeats.
    """
    with model_in_use() as model:
        segments, info = model.transcribe(
            audio,
            language=language,
            beam_size=1,
            best_of=1,
            # A single temperature cannot escape a repetition loop; one fallback
            # step costs nothing on clean audio because it is only used on failure.
            temperature=(0.0, 0.2),
            condition_on_previous_text=False,
            without_timestamps=True,
            no_speech_threshold=WS_NO_SPEECH_THRESHOLD,
            vad_filter=True,
        )
        # The generator is lazy — it must be consumed inside the block, or the
        # reference is released before any decoding actually happens.
        parts = [
            s.text.strip() for s in segments
            if s.no_speech_prob < WS_NO_SPEECH_THRESHOLD
            and s.avg_logprob > WS_MIN_AVG_LOGPROB
            and s.text.strip()
        ]
    return " ".join(parts).strip(), info


def _ws_final_decode(audio: np.ndarray, language):
    """Accurate decode of the whole session buffer, used once at end of stream.

    The interim path is deliberately greedy; reusing it for the final made the
    live transcript measurably worse than uploading the same audio as a file.
    """
    with model_in_use() as model:
        segments, info = model.transcribe(
            audio,
            language=language,
            beam_size=5,
            best_of=5,
            temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
            condition_on_previous_text=True,
            compression_ratio_threshold=2.4,
            no_speech_threshold=WS_NO_SPEECH_THRESHOLD,
            vad_filter=True,
        )
        parts = [
            s.text.strip() for s in segments
            if s.no_speech_prob < 0.8 and s.text.strip()
        ]
    return " ".join(parts).strip(), info


@app.websocket("/ws/transcribe")
async def websocket_transcribe(websocket: WebSocket):
    """Live transcription over WebSocket.

    Protocol:
    - Optional first text frame: JSON config, e.g. {"language": "de"}
      ("auto" or omitted = auto-detect).
    - Binary frames: raw PCM16 little-endian mono audio at 16 kHz.
    - Text frame {"event": "stop"}: finish — the full buffer is decoded once
      more and a {"type": "final", ...} message is sent before closing.

    Server messages: {"type": "partial", "confirmed": str, "pending": str}
    after each interim decode (confirmed = word-level prefix stable across the
    last two decodes), then {"type": "final", "text", "language", "duration"}.
    Partials also carry "decode_ms", "lag_ms" and "pending_seconds" so latency is
    observable from the client without a profiler.

    Ingest and decode are decoupled: the receive loop only ever appends audio,
    and at most one decode is in flight at a time. When a decode is slower than
    realtime the audio that arrived meanwhile is *skipped over* rather than
    queued, so lag stays bounded by one decode instead of growing without limit.
    """
    global _live_sessions

    # Starlette's CORSMiddleware does not run on WebSocket routes, so the origin
    # allow-list has to be applied by hand here or this endpoint is the one hole
    # in an otherwise restricted deployment.
    origin = websocket.headers.get("origin")
    if "*" not in allowed_origins and origin and origin not in allowed_origins:
        await websocket.close(code=1008)  # policy violation
        return

    await websocket.accept()

    try:
        # Load now if an unload released the model, so a dead session fails at
        # the handshake instead of at the first audio frame. The reference is
        # dropped immediately — each decode below takes its own.
        acquire_model()
        release_model()
    except HTTPException:
        await websocket.send_json({"type": "error", "error": "Model not available"})
        await websocket.close(code=1011)
        return

    if _live_sessions >= WS_MAX_SESSIONS:
        await websocket.send_json({
            "type": "error",
            "error": f"Too many live sessions (limit {WS_MAX_SESSIONS})",
        })
        await websocket.close(code=1013)  # try again later
        return
    _live_sessions += 1

    language = None
    chunks: list[np.ndarray] = []
    buffered_samples = 0        # samples currently held in `chunks`
    received_samples = 0        # monotonic; never decremented by the rolling window
    decoded_at_samples = 0
    prev_words: list[str] = []
    rolled = False
    max_samples = int(WS_MAX_BUFFER_S * WS_SAMPLE_RATE)
    window_samples = int(WS_WINDOW_S * WS_SAMPLE_RATE)
    min_new_samples = int(WS_MIN_NEW_AUDIO_S * WS_SAMPLE_RATE)
    loop = asyncio.get_running_loop()
    send_lock = asyncio.Lock()  # the decode task and the receive loop both send
    decode_task: asyncio.Task | None = None

    async def _send(payload: dict):
        """Serialise sends — the ingest loop and the decode task share the socket."""
        async with send_lock:
            await websocket.send_json(clean_json_inf_nan(payload))

    def _tail(n_samples: int) -> np.ndarray:
        """Concatenate only the most recent `n_samples`.

        Copying the whole buffer here used to mean a 38 MB memcpy on the event
        loop at the 10-minute cap, ~96% of which the window slice then threw away.
        """
        if n_samples <= 0:
            return np.empty(0, dtype=np.float32)
        collected: list[np.ndarray] = []
        got = 0
        for chunk in reversed(chunks):
            collected.append(chunk)
            got += len(chunk)
            if got >= n_samples:
                break
        if not collected:
            return np.empty(0, dtype=np.float32)
        buf = np.concatenate(list(reversed(collected)))
        return buf[-n_samples:] if len(buf) > n_samples else buf

    async def _interim_decode():
        """One interim decode over the current tail; sends a single partial."""
        nonlocal prev_words
        started = time.monotonic()
        window = _tail(window_samples)
        if not len(window):
            return
        covered_samples = received_samples
        try:
            text, _info = await loop.run_in_executor(rt_executor, _ws_decode, window, language)
        except asyncio.CancelledError:
            raise
        except Exception as decode_err:
            logger.warning(f"WS interim decode failed: {decode_err}")
            return

        decode_ms = (time.monotonic() - started) * 1000.0
        if not text:
            # Silence or an all-filtered window: keep the last partial on screen
            # rather than blanking the panel the user is reading.
            return

        words = text.split()
        agree = 0
        while agree < len(words) and agree < len(prev_words) and words[agree] == prev_words[agree]:
            agree += 1
        prev_words = words
        # A send failure here (client vanished mid-decode) would otherwise
        # surface as an unretrieved task exception at GC time rather than
        # ending the session; the receive loop notices the disconnect anyway.
        try:
            await _send({
                "type": "partial",
                "confirmed": " ".join(words[:agree]),
                "pending": " ".join(words[agree:]),
                "buffered_seconds": round(received_samples / WS_SAMPLE_RATE, 2),
                # Audio received but not yet reflected in this partial.
                "pending_seconds": round(max(0, received_samples - covered_samples) / WS_SAMPLE_RATE, 2),
                "decode_ms": round(decode_ms, 1),
                "lag_ms": round((time.monotonic() - started) * 1000.0, 1),
            })
        except Exception as send_err:
            logger.debug(f"WS partial send failed (client likely gone): {send_err}")

    try:
        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                return

            if message.get("bytes") is not None:
                raw = message["bytes"]
                # A truncated frame would make frombuffer raise and kill the
                # whole session; drop the odd trailing byte instead.
                if len(raw) % 2:
                    raw = raw[:-1]
                samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                if len(samples):
                    chunks.append(samples)
                    buffered_samples += len(samples)
                    received_samples += len(samples)
                    # Roll the oldest audio out instead of refusing new audio.
                    # Truncating the newest froze `total_samples`, so the decode
                    # gate could never fire again and the session went silently
                    # dead at 10 minutes while the client kept uploading.
                    while buffered_samples > max_samples and len(chunks) > 1:
                        buffered_samples -= len(chunks.pop(0))
                        if not rolled:
                            rolled = True
                            await _send({
                                "type": "warning",
                                "code": "buffer_rolled",
                                "message": (
                                    f"Session exceeded {WS_MAX_BUFFER_S:.0f}s; "
                                    "the final transcript covers only the most recent audio."
                                ),
                            })
            elif message.get("text") is not None:
                try:
                    control = json.loads(message["text"])
                except json.JSONDecodeError:
                    continue
                if control.get("event") == "stop":
                    break
                # Only touch the language when the frame actually carries the
                # key. This used to run unconditionally, so ANY control frame
                # that omitted it — a keepalive, a future event type — reset a
                # German session to auto-detect and never told the caller.
                # The browser only ever sends {language} then {event:"stop"}, so
                # it never tripped; an API client sending anything else did.
                if "language" in control:
                    lang = str(control.get("language") or "").strip().lower()
                    language = None if lang in ("", "auto") else lang

            # Single-flight interim decode. While one is running the loop keeps
            # draining the socket, so frames are never left queued in the
            # transport; the next decode simply starts from the newest audio.
            if (
                (decode_task is None or decode_task.done())
                and received_samples - decoded_at_samples >= min_new_samples
            ):
                decoded_at_samples = received_samples
                decode_task = asyncio.create_task(_interim_decode())

        # Stop requested. Cancelling only detaches the task so it cannot emit a
        # partial after the final — the worker thread keeps running its decode
        # to completion, because a thread in an executor cannot be interrupted.
        # That is why the final runs on the batch pool below: waiting for the
        # abandoned interim to free a realtime slot would delay every other
        # live session's partials.
        if decode_task is not None and not decode_task.done():
            decode_task.cancel()
            try:
                await decode_task
            except asyncio.CancelledError:
                # Expected: we just cancelled it. Do not let this be mistaken
                # for cancellation of this handler.
                pass
            except Exception:
                pass
            decode_task = None

        if buffered_samples:
            buffer = _tail(buffered_samples)
            # Batch pool: this is a beam-search decode over the whole session
            # buffer and has no business holding a realtime slot.
            text, info = await loop.run_in_executor(executor, _ws_final_decode, buffer, language)
            await _send({
                "type": "final",
                "text": text,
                "language": info.language,
                "language_probability": info.language_probability,
                # Duration of the audio this text actually covers. When the
                # rolling window has dropped older audio the two differ, and
                # reporting the session total here made the transcript look
                # like it was missing content rather than bounded on purpose.
                "duration": round(buffered_samples / WS_SAMPLE_RATE, 2),
                "received_duration": round(received_samples / WS_SAMPLE_RATE, 2),
                "truncated": rolled,
            })
        else:
            await _send({"type": "final", "text": "", "language": None, "duration": 0.0})
        await websocket.close()
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"WebSocket transcription error: {e}", exc_info=True)
        try:
            await _send({"type": "error", "error": str(e)})
            await websocket.close(code=1011)
        except Exception:
            pass
    finally:
        # A client that vanishes mid-decode must not leave the task holding a
        # model slot for the rest of the decode.
        if decode_task is not None and not decode_task.done():
            decode_task.cancel()
        _live_sessions -= 1


@app.get("/health", response_class=SafeJSONResponse)
async def health_check():
    """Liveness + readiness for health monitoring.

    Distinguishes two states that both mean "no model in memory" but need
    opposite handling:

    - **not resident** — the model was unloaded to free VRAM, or has not been
      loaded yet. The service is fine and the next request will load it, so
      this returns 200. Returning 503 here would make an idle container go
      `unhealthy` under Docker's `curl -f` check and show as down in the UI.
    - **broken** — every load attempt failed. That is a real outage: 503.
    """
    current_model = model_size_loaded or configured_model_size()
    resident = whisper_model is not None
    body = {
        "status": "error" if startup_error else "ok",
        # Kept for API compatibility: existing clients read model_loaded.
        "model_loaded": model_loaded,
        "model_resident": resident,
        "can_load": startup_error is None,
        "model_warmed": model_warmed,
        "device": device,
        "compute_type": compute_type,
        "model_size": current_model,
        "multilingual": current_model not in ENGLISH_ONLY_MODELS,
        "live_sessions": _live_sessions,
        # Outstanding references. POST /unload refuses while this is non-zero.
        "model_refs": _model_refs,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available()
    }
    if startup_error:
        body["startup_error"] = startup_error
        return SafeJSONResponse(content=body, status_code=503)
    return body

@app.get("/info", response_class=SafeJSONResponse)
async def service_info():
    """Return detailed service and GPU information."""
    return {
        "service": "STT Service",
        "device": device,
        "compute_type": compute_type,
        "model_loaded": model_loaded,
        "model_size": model_size_loaded or configured_model_size(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    }

@app.post("/unload")
async def unload():
    """Release the model and its memory now, without stopping the container.

    The idle TTL handles the common case; this is for the deliberate one — you
    are about to run something else on the same GPU and want the VRAM back
    immediately. The next request reloads transparently, so this is safe to call
    at any time.

    Returns 200 when the model was released or was already unloaded, and **409
    when a transcription is in flight** — freeing memory a running decode is
    still reading would crash the worker, so the caller is told to retry instead.
    """
    result = unload_model()
    if result["reason"] == "busy":
        return SafeJSONResponse(
            status_code=409,
            content={
                "detail": "Model is in use; retry when idle",
                "model_refs": result["refs"],
                **result,
            },
        )
    return SafeJSONResponse(content={
        "model_resident": whisper_model is not None,
        **result,
    })


@app.get("/models")
async def available_models():
    """List Whisper model variants with size and multilingual capability."""
    return {
        "available_models": [
            {"name": "tiny",            "multilingual": True,  "size_mb": 75},
            {"name": "tiny.en",         "multilingual": False, "size_mb": 75},
            {"name": "base",            "multilingual": True,  "size_mb": 145},
            {"name": "base.en",         "multilingual": False, "size_mb": 145},
            {"name": "small",           "multilingual": True,  "size_mb": 490},
            {"name": "small.en",        "multilingual": False, "size_mb": 490},
            {"name": "medium",          "multilingual": True,  "size_mb": 1500},
            {"name": "medium.en",       "multilingual": False, "size_mb": 1500},
            {"name": "large-v1",        "multilingual": True,  "size_mb": 2900},
            {"name": "large-v2",        "multilingual": True,  "size_mb": 2900},
            {"name": "large-v3",        "multilingual": True,  "size_mb": 3100},
            {"name": "large-v3-turbo",  "multilingual": True,  "size_mb": 1600,
             "note": "Default. 4 decoder layers vs large-v3's 32 — fastest multilingual option; cannot translate"},
            {"name": "distil-large-v3", "multilingual": False, "size_mb": 1500, "note": "English-only, fastest"},
        ],
        "current_model": model_size_loaded or configured_model_size(),
        "supported_languages": [
            "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr", "pl", "ca", "nl",
            "ar", "sv", "it", "id", "hi", "fi", "vi", "he", "uk", "el", "ms", "cs", "ro",
            "da", "hu", "ta", "no", "th", "ur", "hr", "bg", "lt", "la", "mi", "ml", "cy",
            "sk", "te", "fa", "lv", "bn", "sr", "az", "sl", "kn", "et", "mk", "br", "eu",
            "is", "hy", "ne", "mn", "bs", "kk", "sq", "sw", "gl", "mr", "pa", "si", "km",
            "sn", "yo", "so", "af", "oc", "ka", "be", "tg", "sd", "gu", "am", "yi", "lo",
            "uz", "fo", "ht", "ps", "tk", "nn", "mt", "sa", "lb", "my", "bo", "tl", "mg",
            "as", "tt", "haw", "ln", "ha", "ba", "jw", "su"
        ]
    }

@app.get("/tasks")
async def available_tasks():
    """Describe the supported transcription tasks (transcribe / translate)."""
    return {
        "available_tasks": [
            {
                "task": "transcribe",
                "description": "Transcribe audio to text in the original language",
                "parameters": {
                    "language": "Optional: Source language code (auto-detected if not provided)",
                    "beam_size": "Search beam size (1-5, higher = better quality, slower)",
                    "best_of": "Number of candidates to consider (1-5)"
                }
            },
            {
                "task": "translate",
                "description": "Transcribe and translate audio to English",
                "parameters": {
                    "language": "Optional: Source language code",
                    "target_language": "Target language (currently only 'english' supported)",
                    "beam_size": "Search beam size (1-5)",
                    "best_of": "Number of candidates to consider (1-5)"
                },
                "limitations": [
                    "Translation is only available to English",
                    "Whisper model limitation, not service limitation"
                ]
            }
        ],
        "default_task": "transcribe",
        "default_target_language": "english",
        "streaming_available": True
    }

@app.post("/detect_language")
async def detect_language(
    file: UploadFile = File(...),
    duration_limit: float = 30.0,
):
    """Quickly detect the spoken language and return a sample transcript."""
    if not model_loaded or whisper_model is None:
        logger.error("Model not loaded, attempting to load...")
        load_model()
        if not model_loaded or whisper_model is None:
            raise HTTPException(status_code=503, detail="Model not available")

    safe_filename = os.path.basename(file.filename or "audio.wav")
    suffix = os.path.splitext(safe_filename)[1] or ".wav"
    tmp_fd, temp_audio_path = tempfile.mkstemp(suffix=suffix)
    try:
        with os.fdopen(tmp_fd, "wb") as f:
            content = await file.read()
            f.write(content)

        logger.info(f"Detecting language for: {file.filename}")
        start_time = time.time()
        
        # Use transcribe with short duration for quick language detection.
        # Runs in a worker thread so the event loop stays responsive.
        def _detect():
            """Blocking language-detection transcription, run off the event loop."""
            # faster-whisper decodes the whole file up front, so trim long
            # uploads to duration_limit seconds for fast detection.
            detect_path = temp_audio_path
            trimmed_path = None
            if duration_limit and duration_limit > 0:
                try:
                    import librosa
                    import soundfile as sf_lib
                    audio_data, sr = librosa.load(temp_audio_path, sr=None, mono=True, duration=duration_limit)
                    if len(audio_data) > 0:
                        trimmed_fd, trimmed_path = tempfile.mkstemp(suffix=".wav")
                        os.close(trimmed_fd)
                        sf_lib.write(trimmed_path, audio_data, sr)
                        detect_path = trimmed_path
                except Exception as trim_err:
                    logger.warning(f"Could not trim audio for language detection: {trim_err}")

            try:
                # Lazy generator again: the reference must outlive the loop that
                # consumes it, not just the transcribe() call.
                with model_in_use() as model:
                    segments, info = model.transcribe(
                        detect_path,
                        task="transcribe",
                        language=None,  # Auto-detect
                        beam_size=1,    # Fastest setting
                        best_of=1,      # Fastest setting
                        vad_filter=True,
                        condition_on_previous_text=False,
                        # Only process first part for speed
                        vad_parameters={
                            "threshold": 0.5,
                            "min_speech_duration_ms": 250,
                            "min_silence_duration_ms": 500,
                        }
                    )
                    sample = ""
                    count = 0
                    for segment in segments:
                        if count >= 3:  # Only need a few segments for detection
                            break
                        if segment.no_speech_prob < 0.8:
                            sample += segment.text.strip() + " "
                            count += 1
                return sample, info
            finally:
                if trimmed_path and os.path.exists(trimmed_path):
                    try:
                        os.unlink(trimmed_path)
                    except OSError:
                        pass

        loop = asyncio.get_running_loop()
        sample_text, info = await loop.run_in_executor(executor, _detect)

        processing_time = time.time() - start_time
        
        logger.info(f"Language detection completed in {processing_time:.2f}s: {info.language} ({info.language_probability:.2f})")
        
        return {
            "detected_language": info.language,
            "language_probability": info.language_probability,
            "sample_text": sample_text.strip(),
            "processing_time": processing_time,
            "audio_duration": info.duration
        }
        
    except Exception as e:
        logger.error(f"Language detection failed: {e}")
        raise HTTPException(status_code=500, detail=f"Language detection failed: {str(e)}")
    finally:
        try:
            os.unlink(temp_audio_path)
        except Exception:
            pass
