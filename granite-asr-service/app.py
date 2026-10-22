"""IBM Granite Speech speech-to-text service.

Serves one of two Apache-2.0 checkpoints with transformers' own classes (no remote
code), chosen by ``GRANITE_ASR_MODEL``:

* ``ibm-granite/granite-speech-5.0-2b`` (default): a speech LLM on granite-4.0-1b,
  en fr de es pt ja, punctuated and cased output (German nouns capitalised), told
  the language in its prompt, and biased towards up to 2000 keywords (the
  ``keywords`` form field, or ``initial_prompt``, which is what the gateway sends
  for OpenAI's ``prompt``: both are read as a comma-separated list).
* ``ibm-granite/granite-speech-5.0-470m-turboctc``: 470M parameters, CTC, one greedy
  non-autoregressive pass; en fr de es pt it ja, punctuated and cased. No prompt:
  the language is only checked and reported, keywords cannot be applied.

bf16 on a CUDA GPU, float32 on the CPU. ``auto``/empty means
``GRANITE_ASR_DEFAULT_LANGUAGE`` (``de``); a language the model does not know is
refused with 422. A voice activity detector decides what the model hears and long
recordings are cut in their pauses (``GRANITE_ASR_CHUNK_S``, transformers_asr.py);
neither model has timestamps, so each segment is one chunk with the chunk's start
and end.
"""

import asyncio
import json
import logging
import os
import shutil
import time
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
from granite_support import (
    DEFAULT_MODEL, FAMILY_CTC, FAMILY_LLM, KNOWN_MODELS, KeywordError, build_prompt, default_revision,
    family_for, parse_keywords, supported_languages,
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

MODEL_NAME = (os.getenv("GRANITE_ASR_MODEL") or "").strip() or DEFAULT_MODEL
REVISION = (os.getenv("GRANITE_ASR_REVISION") or "").strip() or default_revision(MODEL_NAME)
# Not needed for the public IBM repositories; passed on for a private fine-tune.
_HF_TOKEN = (os.getenv("HF_TOKEN") or "").strip() or None

# Known before the download for the two IBM checkpoints, read from config.json at the
# first load for anything else. The language list and the default follow it.
FAMILY = KNOWN_MODELS[MODEL_NAME]["family"] if MODEL_NAME in KNOWN_MODELS else None
_LANGUAGES_OVERRIDE = os.getenv("GRANITE_SUPPORTED_LANGUAGES")
_DEFAULT_LANGUAGE_SETTING = os.getenv("GRANITE_ASR_DEFAULT_LANGUAGE", "de")


def _languages_for(family: str) -> frozenset:
    return supported_languages(MODEL_NAME, family, _LANGUAGES_OVERRIDE)


SUPPORTED_LANGUAGES = _languages_for(FAMILY or FAMILY_LLM)
DEFAULT_LANGUAGE = default_language_from_env(_DEFAULT_LANGUAGE_SETTING, supported=SUPPORTED_LANGUAGES, fallback="de")

_CONFIG_ERROR: Optional[str] = None
try:
    WEIGHTS_SHA256 = expected_sha256(os.getenv("GRANITE_ASR_SHA256"), None)
except ValueError as exc:
    WEIGHTS_SHA256 = None
    _CONFIG_ERROR = f"GRANITE_ASR_SHA256 is {exc}"
    logger.error("%s; the model will not be loaded until it is fixed", _CONFIG_ERROR)

# Length of the pieces a recording is cut into, in its pauses. The 2b's model card
# advises at most ~2 minutes per call; shorter pieces keep the decoding of a batch
# short and every segment's timestamps close to its words.
CHUNK_S = env_number("GRANITE_ASR_CHUNK_S", 30.0, minimum=5.0, maximum=120.0)
BATCH_SIZE = env_number("GRANITE_ASR_BATCH_SIZE", 4, cast=int, minimum=1)
MAX_NEW_TOKENS = env_number("GRANITE_ASR_MAX_NEW_TOKENS", max_new_tokens_for(CHUNK_S), cast=int, minimum=32)
VAD_MODE = env_choice("GRANITE_ASR_VAD", "silero", VAD_MODES)
VAD_THRESHOLD = env_number("GRANITE_ASR_VAD_THRESHOLD", 0.5, minimum=0.05, maximum=0.95)

_FFMPEG = shutil.which("ffmpeg")
_speech_gate = SpeechGate(VAD_MODE, threshold=VAD_THRESHOLD)

_preload_pending = False


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    """Pre-load the model in the background so the first request is fast (see cohere-asr-service)."""
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
    title="Granite-ASR Service",
    description="Speech-to-Text with IBM Granite Speech 5.0 (2b speech LLM or 470m TurboCTC)",
    lifespan=_lifespan,
)

allowed_origins = parse_allowed_origins(os.getenv("ALLOWED_ORIGINS", ""))
allow_credentials = os.getenv("ALLOW_CREDENTIALS", "false").strip().lower() in {"1", "true", "yes", "on"}
if "*" in allowed_origins and allow_credentials:
    allow_credentials = False

# Request-body limits (body_limit.py): one upload route, one file plus the small text
# fields (a 2000-keyword list included) in 64 KiB of slack; nothing else takes a body.
_UPLOAD_PATHS = frozenset({"/transcribe"})


def _body_limit(path: str) -> int:
    """Largest request body (bytes) accepted at *path*."""
    if path in _UPLOAD_PATHS:
        return MAX_UPLOAD_BYTES + MULTIPART_SLACK + 256 * 1024
    return MIB


# Body limit innermost, origin guard next, CORS outermost (see cohere-asr-service).
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

_ASR_CONCURRENCY = env_number("ASR_MAX_CONCURRENCY", 1, cast=int, minimum=1)
_ASR_MAX_QUEUE = env_number("ASR_MAX_QUEUE", 4 * _ASR_CONCURRENCY, cast=int, minimum=0)
_ASR_QUEUE_TIMEOUT_S = env_number("ASR_QUEUE_TIMEOUT_S", 60.0, minimum=0.1)
_RETRY_AFTER_S = 5

_OOM_DETAIL = (
    "GPU out of memory while transcribing. Lower GRANITE_ASR_BATCH_SIZE or GRANITE_ASR_CHUNK_S, "
    "or free the VRAM another service holds (POST /unload on it)."
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
    return uuid.uuid4().hex[:12]


async def _asr(fn, *args, **kwargs):
    """Run ``fn(model, *args)`` off the event loop: the model pinned, one of the gate's turns held."""
    async with _model_slot.acquire_async() as model:
        async with _gate.turn():
            return await asyncio.to_thread(fn, model, *args, **kwargs)


@dataclass
class LoadedModel:
    model: object
    processor: object
    family: str


def _model_dir() -> str:
    if is_local_model(MODEL_NAME):
        return MODEL_NAME
    from huggingface_hub import snapshot_download

    return snapshot_download(
        MODEL_NAME, revision=REVISION, allow_patterns=list(DOWNLOAD_PATTERNS), token=_HF_TOKEN)


def _read_family(model_dir: str) -> str:
    with open(os.path.join(model_dir, "config.json"), encoding="utf-8") as fh:
        return family_for(json.load(fh).get("model_type"))


def _load_granite() -> LoadedModel:
    """Fetch, (optionally) verify and load the configured checkpoint."""
    global model_loaded, compute_dtype, weights_check, load_seconds, FAMILY, SUPPORTED_LANGUAGES, DEFAULT_LANGUAGE
    if _CONFIG_ERROR:
        raise RuntimeError(_CONFIG_ERROR)
    started = time.time()
    logger.info("Loading Granite Speech '%s' (revision %s) on %s...", public_model_name(MODEL_NAME), REVISION, device)
    model_dir = _model_dir()
    if WEIGHTS_SHA256:
        try:
            verify_weights(model_dir, WEIGHTS_SHA256)
        except Exception:
            weights_check = "failed"
            raise
        weights_check = "ok"

    family = _read_family(model_dir)
    if FAMILY is None or family != FAMILY:
        FAMILY = family
        SUPPORTED_LANGUAGES = _languages_for(family)
        DEFAULT_LANGUAGE = default_language_from_env(
            _DEFAULT_LANGUAGE_SETTING, supported=SUPPORTED_LANGUAGES, fallback="de")

    if device == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype = torch.float32
    if family == FAMILY_CTC:
        from transformers import GraniteSpeech5ForCTC, GraniteSpeech5Processor

        processor = GraniteSpeech5Processor.from_pretrained(model_dir)
        model = GraniteSpeech5ForCTC.from_pretrained(model_dir, dtype=dtype)
    else:
        from transformers import GraniteSpeechForConditionalGeneration, GraniteSpeechProcessor

        processor = GraniteSpeechProcessor.from_pretrained(model_dir)
        model = GraniteSpeechForConditionalGeneration.from_pretrained(model_dir, dtype=dtype)
    model.to(device)
    model.eval()
    if VAD_MODE == "silero":
        try:
            _speech_gate.regions(torch.zeros(SAMPLE_RATE).numpy())
        except Exception as e:
            logger.warning("VAD warm-up failed: %s", e)

    compute_dtype = str(dtype).replace("torch.", "")
    model_loaded = True
    load_seconds = round(time.time() - started, 1)
    logger.info("Granite Speech (%s) loaded on %s (%s) in %.1f s", family, device, compute_dtype, load_seconds)
    return LoadedModel(model=model, processor=processor, family=family)


def _release_granite(loaded: LoadedModel) -> None:
    global model_loaded
    model_loaded = False
    try:
        loaded.model.to("cpu")
    except Exception as e:
        logger.warning(f"Could not move Granite Speech off the GPU before unload: {e}")


# ~4.6 GB (2b) or ~1 GB (470m) of bf16 weights, released when idle.
#   >0 = seconds idle before unloading | 0 = unload immediately | -1 = never
MODEL_TTL = ttl_from_env(os.getenv, "ASR_MODEL_TTL", "MODEL_TTL", default=300.0)
_model_slot = ModelSlot(
    _load_granite, ttl_seconds=MODEL_TTL, name="Granite-Speech", on_unload=_release_granite,
)


def get_model():
    with _model_slot.acquire() as model:
        return model


def _server_error(exc: Exception, request_id: str) -> HTTPException:
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


def _run_llm(loaded: LoadedModel, pieces: list, language: str, keywords: list) -> list:
    """The speech LLM over *pieces*, BATCH_SIZE at a time, with one prompt for all of them."""
    processor, model = loaded.processor, loaded.model
    tokenizer = processor.tokenizer
    chat = [{"role": "user", "content": build_prompt(language, keywords)}]
    prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    outputs: list[ChunkText] = []
    for batch in batches(pieces, BATCH_SIZE):
        audio = [torch.from_numpy(piece) for piece in batch]
        inputs = processor([prompt] * len(batch), audio, device="cpu", return_tensors="pt")
        inputs = inputs.to(model.device)
        if "input_features" in inputs:
            inputs["input_features"] = inputs["input_features"].to(model.dtype)
        prompt_length = int(inputs["input_ids"].shape[-1])
        with torch.inference_mode():
            sequences = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, num_beams=1)
        new_tokens = sequences[:, prompt_length:]
        texts = tokenizer.batch_decode(new_tokens, add_special_tokens=False, skip_special_tokens=True)
        if len(texts) != len(batch):
            raise RuntimeError(f"generate() returned {len(texts)} rows for {len(batch)} pieces")
        pad, eos = tokenizer.pad_token_id, tokenizer.eos_token_id
        for text, row in zip(texts, new_tokens.tolist()):
            count = 0
            for token in row:
                if token in (pad, eos):
                    break
                count += 1
            outputs.append(ChunkText(text=text.strip(), capped=count >= MAX_NEW_TOKENS))
    return outputs


def _run_ctc(loaded: LoadedModel, pieces: list) -> list:
    """The CTC model over *pieces*: one forward pass and a greedy decode per batch."""
    processor, model = loaded.processor, loaded.model
    outputs: list[ChunkText] = []
    for batch in batches(pieces, BATCH_SIZE):
        inputs = processor(list(batch), sampling_rate=SAMPLE_RATE, return_tensors="pt")
        inputs = inputs.to(model.device, dtype=model.dtype)
        with torch.inference_mode():
            sequences = model.generate(**inputs)
        texts = processor.batch_decode(sequences, skip_special_tokens=True)
        if len(texts) != len(batch):
            raise RuntimeError(f"generate() returned {len(texts)} rows for {len(batch)} pieces")
        outputs += [ChunkText(text=t.strip()) for t in texts]
    return outputs


def _run_model(loaded: LoadedModel, pieces: list, language: str, keywords: list) -> list:
    if loaded.family == FAMILY_CTC:
        return _run_ctc(loaded, pieces)
    return _run_llm(loaded, pieces, language, keywords)


async def _transcribe_upload(upload: UploadFile, language: Optional[str], keywords_field: Optional[str],
                             initial_prompt: Optional[str]) -> dict:
    """Full ``stt-form-v1`` response for one uploaded file."""
    started = time.time()
    model_label = f"Granite Speech ({public_model_name(MODEL_NAME)})"
    try:
        resolved = resolve_language(language, default=DEFAULT_LANGUAGE, supported=SUPPORTED_LANGUAGES, model=model_label)
        keywords = parse_keywords(keywords_field, initial_prompt)
    except (UnsupportedLanguage, KeywordError) as e:
        raise HTTPException(status_code=422, detail=str(e))
    warnings: list[str] = []
    if is_auto(language) and (language or "").strip():
        warnings.append(
            f"Granite Speech is not asked to detect the language; transcribed as '{resolved}' "
            f"(GRANITE_ASR_DEFAULT_LANGUAGE). Send the language to choose another.")
    if keywords and FAMILY == FAMILY_CTC:
        warnings.append(f"{public_model_name(MODEL_NAME)} is a CTC model and takes no keywords; they were ignored.")

    audio = await load_upload(upload, max_bytes=MAX_UPLOAD_BYTES, max_seconds=MAX_AUDIO_SECONDS, ffmpeg=_FFMPEG)
    regions = await asyncio.to_thread(_speech_gate.regions, audio.samples)
    chunks = plan_speech_chunks(audio.samples, regions, max_chunk_s=CHUNK_S)
    logger.info(
        "Transcribing %s (%.1f MB, %.1f s, language=%s, %d keywords): %d chunks, %.1f s of speech",
        upload.filename, audio.size / MIB, audio.duration, resolved, len(keywords), len(chunks),
        speech_seconds(chunks))

    outputs = []
    if chunks:
        pieces = [audio.samples[c.start:c.end] for c in chunks]
        outputs = await _asr(_run_model, pieces, resolved, keywords)
    else:
        warnings.append("No speech was detected in the audio.")
    segments, text, capped = build_segments(chunks, outputs)
    if capped:
        message = (
            f"Generation stopped at the {MAX_NEW_TOKENS}-token limit in the chunk(s) at {', '.join(capped)}; "
            f"that text is probably cut off. Raise GRANITE_ASR_MAX_NEW_TOKENS or lower GRANITE_ASR_CHUNK_S."
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
        "keywords": len(keywords) if FAMILY != FAMILY_CTC else 0,
        "truncated": bool(capped),
        "warnings": warnings,
        "task": "transcribe",
        "model": public_model_name(MODEL_NAME),
    }


# --- endpoints -----------------------------------------------------------------


@app.get("/health")
async def health():
    """Liveness probe; `model_resident: false` is an idle unload, not an error."""
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
    """Readiness, like every backend here: 503 `loading` / `load_failed`, never triggers a load."""
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
    """Release the model and its VRAM now; 409 while a request is in flight."""
    result = await _model_slot.try_unload_async()
    if result["reason"] == "busy":
        return JSONResponse(status_code=409, content={"detail": "Model is in use; retry when idle", **result})
    return {"model_resident": _model_slot.resident, **result}


@app.get("/status")
async def status():
    """Detailed service status, including what the gateway's language selector offers."""
    status_info = {
        "status": "ok",
        "service": "Granite-ASR",
        "device": device,
        "cuda_available": torch.cuda.is_available(),
        "model_loaded": model_loaded,
        "model_resident": _model_slot.resident,
        "model_ttl_seconds": MODEL_TTL,
        "active_requests": _model_slot.refs,
        "current_model": public_model_name(MODEL_NAME),
        "revision": REVISION,
        "model_family": FAMILY,
        "weights_sha256": WEIGHTS_SHA256,
        "weights_check": weights_check,
        "load_seconds": load_seconds,
        "supported_languages": sorted(SUPPORTED_LANGUAGES),
        "default_language": DEFAULT_LANGUAGE,
        "language_detection": False,
        "keywords": FAMILY != FAMILY_CTC,
        "timestamps": "chunk",
        "vad": _speech_gate.describe(),
        "compute_dtype": compute_dtype,
        "chunk_seconds": CHUNK_S,
        "batch_size": BATCH_SIZE,
        "max_new_tokens": MAX_NEW_TOKENS if FAMILY != FAMILY_CTC else None,
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
    keywords: Optional[str] = Form(None),
    initial_prompt: Optional[str] = Form(None),
):
    """Transcribe an audio file in the requested (or default) language.

    `keywords` (and `initial_prompt`) are a comma-, semicolon- or newline-separated
    keyword list for the 2b's keyword biasing (at most 2000). Returns the
    `stt-form-v1` shape plus `speech_seconds`, `chunks`, `keywords` (how many were
    applied), `truncated` and `warnings`. 422 for an unsupported language, a keyword
    list over the limit, or audio that cannot be decoded; 413 for a file over
    MAX_UPLOAD_MB or longer than MAX_AUDIO_SECONDS; 503 + Retry-After when busy.
    """
    req_id = _request_id()
    try:
        with _gate.admit():
            return JSONResponse(content=await _transcribe_upload(audio, language, keywords, initial_prompt))
    except HTTPException:
        raise
    except Exception as e:
        logger.error("[%s] Transcription error: %s", req_id, e, exc_info=True)
        raise _server_error(e, req_id)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5010)
