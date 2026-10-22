"""Speech-to-text through vLLM: the playground's candidate models under continuous batching.

A thin FastAPI adapter speaking the playground's ``stt-form-v1`` contract
(``/transcribe``, ``/health``, ``/ready``, ``/status``, ``/unload``) in front of
``vllm serve <model>``, which it runs as a subprocess on 127.0.0.1 inside the same
container (vllm_engine.py). One model per service, chosen by ``VLLM_ASR_MODEL``:

* ``ibm-granite/granite-speech-5.0-2b`` (default), through vLLM's chat endpoint with
  granite-asr-service's own prompt (language, punctuation, keywords);
* ``evewashere/cohere-transcribe-03-2026-ungated`` at the commit cohere-asr-service
  pins, its weights checked against Cohere's official SHA-256 before the engine
  starts;
* ``Qwen/Qwen3-ASR-1.7B`` (identifies the language; a prompt is its context);
* ``primeline/whisper-large-v3-turbo-german`` (German only; the ``whisper`` provider,
  faster-whisper, stays the quality reference for Whisper).

What makes this service different from the transformers ones is what it is for:
throughput. A recording is cut in its pauses into chunks of at most
``VLLM_ASR_CHUNK_S`` seconds (the same speech gate and planner as cohere/granite),
and **every chunk of every request goes to the engine at once**, so vLLM batches
them with whatever else is running. The chunk texts are joined in order.

The engine's GPU share is fixed at start-up (``VLLM_ASR_GPU_MEMORY_UTILIZATION``). Idle
for ``ASR_MODEL_TTL`` seconds, the engine is put to sleep (weights to host RAM, KV
cache dropped; ``VLLM_ASR_IDLE_MODE=sleep``) or stopped (``stop``); the next request
wakes or restarts it.
"""

import os

# The adapter never computes on the GPU: the VAD runs on the CPU, and the engine
# subprocess owns the card. Hide the GPU from this process before anything can import
# torch, so neither the VAD nor the model slot's cache release (torch.cuda.ipc_collect
# initialises CUDA) opens a CUDA context here (~0.4 GB of VRAM). The engine gets the
# original value back.
_ENGINE_CUDA_VISIBLE_DEVICES = os.environ.get("CUDA_VISIBLE_DEVICES")
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import asyncio
import json
import logging
import shutil
import subprocess
import threading
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import httpx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from audio_pipeline import MIB, MULTIPART_SLACK, env_number, load_upload, model_input
from body_limit import BodyLimitMiddleware
from cohere_support import expected_size
from granite_support import KeywordError, parse_keywords
from model_lifecycle import ModelSlot, RequestGate, ttl_from_env
from origin_guard import OriginGuardMiddleware, parse_allowed_origins
from transformers_asr import (
    DOWNLOAD_PATTERNS, VAD_MODES, ChunkText, UnsupportedLanguage, build_segments,
    default_language_from_env, env_choice, expected_sha256, is_local_model, max_new_tokens_for,
    plan_speech_chunks, public_model_name, resolve_language, speech_seconds, verify_weights,
)
from vad_pool import VadPool
from vllm_engine import Engine, EngineError, STATE_SLEEPING, STATE_STOPPED, signal_process_group
from vllm_support import (
    DEFAULT_CHUNK_S, DEFAULT_MODEL, ENDPOINT_CHAT, FAMILY_COHERE, FAMILY_GRANITE, FAMILY_WHISPER, KNOWN_MODELS,
    MAX_CHUNK_S, OFFERED_MODELS, PROMPT_USE, EngineSettings, chat_body, check_extra_args,
    default_revision, default_sha256, detects_language, endpoint_for, engine_argv, family_for, parse_chat,
    parse_extra_args, parse_transcription, prompt_text, supported_languages, transcription_fields, wav_bytes,
)

# Upload bounds. The gateway limits what it forwards, but a direct caller must not
# be able to fill the disk or the host's memory with an arbitrarily long file.
MAX_UPLOAD_BYTES = int(env_number("MAX_UPLOAD_MB", 200.0, minimum=0.001) * MIB)
MAX_AUDIO_SECONDS = env_number("MAX_AUDIO_SECONDS", 7200.0, minimum=0.0)  # 0 = unlimited

MODEL_NAME = (os.getenv("VLLM_ASR_MODEL") or "").strip() or DEFAULT_MODEL
REVISION = (os.getenv("VLLM_ASR_REVISION") or "").strip() or default_revision(MODEL_NAME)
SERVED_NAME = public_model_name(MODEL_NAME)
# For a gated or private repository only (the official Cohere one); never logged, and
# not passed on to the engine, which reads the downloaded directory.
_HF_TOKEN = (os.getenv("HF_TOKEN") or "").strip() or None

# Known before the download for the measured checkpoints, read from config.json at the
# first start for anything else (until then, Whisper's languages are assumed).
FAMILY: Optional[str] = KNOWN_MODELS[MODEL_NAME].family if MODEL_NAME in KNOWN_MODELS else None
_LANGUAGES_OVERRIDE = os.getenv("VLLM_ASR_SUPPORTED_LANGUAGES")
_DEFAULT_LANGUAGE_SETTING = os.getenv("VLLM_ASR_DEFAULT_LANGUAGE", "de")


def _languages_for(family: str) -> frozenset:
    return supported_languages(MODEL_NAME, family, _LANGUAGES_OVERRIDE)


SUPPORTED_LANGUAGES = _languages_for(FAMILY or FAMILY_WHISPER)
DEFAULT_LANGUAGE = default_language_from_env(_DEFAULT_LANGUAGE_SETTING, supported=SUPPORTED_LANGUAGES, fallback="de")

_CONFIG_ERROR: Optional[str] = None
try:
    WEIGHTS_SHA256 = expected_sha256(os.getenv("VLLM_ASR_SHA256"), default_sha256(MODEL_NAME))
except ValueError as exc:
    WEIGHTS_SHA256 = None
    _CONFIG_ERROR = f"VLLM_ASR_SHA256 is {exc}"
    logger.error("%s; the engine will not be started until it is fixed", _CONFIG_ERROR)

# Pieces a recording is cut into, in its pauses. 28 s by default: under the 30 s
# window every family takes whole, with room for the speech the VAD pads.
CHUNK_S = env_number("VLLM_ASR_CHUNK_S", DEFAULT_CHUNK_S, minimum=5.0, maximum=MAX_CHUNK_S)
MAX_NEW_TOKENS = env_number("VLLM_ASR_MAX_NEW_TOKENS", max_new_tokens_for(CHUNK_S), cast=int, minimum=32)
VAD_MODE = env_choice("VLLM_ASR_VAD", "silero", VAD_MODES)
VAD_THRESHOLD = env_number("VLLM_ASR_VAD_THRESHOLD", 0.5, minimum=0.05, maximum=0.95)
# Worker processes running the Silero VAD (vad_pool.py): one gate in this process
# capped the service at ~200x real time with 32 requests at once. 0 = in this process.
VAD_WORKERS = env_number("VLLM_ASR_VAD_WORKERS", 4, cast=int, minimum=0, maximum=64)

# --- the engine ------------------------------------------------------------------------------

IDLE_MODE = env_choice("VLLM_ASR_IDLE_MODE", "sleep", ("sleep", "stop"))
ENGINE_PORT = env_number("VLLM_ASR_ENGINE_PORT", 8100, cast=int, minimum=1024, maximum=65535)
_EXTRA_ARGS = parse_extra_args(os.getenv("VLLM_ASR_EXTRA_ARGS"))
_COLLIDING = check_extra_args(_EXTRA_ARGS)
if _COLLIDING:
    _CONFIG_ERROR = _CONFIG_ERROR or (
        f"VLLM_ASR_EXTRA_ARGS repeats arguments the service sets itself: {', '.join(_COLLIDING)} "
        f"(use the VLLM_ASR_* settings for them; remote code is never trusted)")
    logger.error("%s; the engine will not be started until it is fixed", _CONFIG_ERROR)
ENGINE_SETTINGS = EngineSettings(
    # The share of the WHOLE card this engine takes at start-up (weights, activations,
    # KV cache, CUDA graphs): 0.45 of 16 GB is ~7.2 GB, room for Whisper (~2.3 GB),
    # Magpie (~3.3 GB) and another small vLLM app on the same card. Less is a smaller
    # KV cache, so fewer chunks decoded at once (/status says how many tokens fit).
    gpu_memory_utilization=env_number("VLLM_ASR_GPU_MEMORY_UTILIZATION", 0.45, minimum=0.05, maximum=0.95),
    max_model_len=env_number("VLLM_ASR_MAX_MODEL_LEN", None, cast=int, minimum=128),
    max_num_seqs=env_number("VLLM_ASR_MAX_NUM_SEQS", None, cast=int, minimum=1),
    max_num_batched_tokens=env_number("VLLM_ASR_MAX_NUM_BATCHED_TOKENS", None, cast=int, minimum=256),
    sleep_mode=IDLE_MODE == "sleep",
    extra_args=_EXTRA_ARGS,
)
START_TIMEOUT_S = env_number("VLLM_ASR_START_TIMEOUT_S", 900.0, minimum=30.0)
REQUEST_TIMEOUT_S = env_number("VLLM_ASR_REQUEST_TIMEOUT_S", 600.0, minimum=5.0)
# Chunk requests in flight to the engine across all requests. vLLM queues what does
# not fit its KV cache itself; this only bounds the adapter's own sockets and buffers.
MAX_PENDING_CHUNKS = env_number("VLLM_ASR_MAX_PENDING_CHUNKS", 256, cast=int, minimum=1)

# Injectable for the tests: the process launcher, how its group is signalled, and the
# transports to the engine's HTTP API.
_popen = subprocess.Popen
_signal_group = signal_process_group
_ENGINE_POLL_S = 1.0  # between /health probes while the engine starts
_SYNC_TRANSPORT: Optional[httpx.BaseTransport] = None
_ASYNC_TRANSPORT: Optional[httpx.AsyncBaseTransport] = None

_FFMPEG = shutil.which("ffmpeg")
_vad = VadPool(VAD_MODE, threshold=VAD_THRESHOLD, workers=VAD_WORKERS)

_preload_pending = False


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    """Start the engine in the background so the first request is fast; stop it on the way out.

    Background, not before ``yield``: a first start downloads the checkpoint and
    compiles the model (minutes), and the port must answer /health meanwhile.
    """
    global _preload_pending
    preload = None
    if MODEL_TTL == 0:
        logger.info("Preload skipped: ASR_MODEL_TTL=0 releases the engine on every idle")
    else:
        _preload_pending = True
        preload = asyncio.create_task(_startup_preload())
    try:
        yield
    finally:
        if preload is not None:
            preload.cancel()
        await asyncio.to_thread(_shutdown_engine)
        _vad.shutdown()
        if _client is not None:
            await _client.aclose()


async def _startup_preload():
    global _preload_pending
    try:
        await asyncio.to_thread(get_model)
    except Exception as e:
        logger.warning(f"Could not start the engine: {e}")
    finally:
        _preload_pending = False


app = FastAPI(
    title="vLLM-ASR Service",
    description="Speech-to-Text through vLLM (continuous batching): Granite Speech, Cohere Transcribe, "
                "Qwen3-ASR or Whisper",
    lifespan=_lifespan,
)

allowed_origins = parse_allowed_origins(os.getenv("ALLOWED_ORIGINS", ""))
allow_credentials = os.getenv("ALLOW_CREDENTIALS", "false").strip().lower() in {"1", "true", "yes", "on"}
if "*" in allowed_origins and allow_credentials:
    allow_credentials = False

# Request-body limits (body_limit.py): one upload route, one file plus the small text
# fields (a 2000-keyword list included) in 256 KiB of slack; nothing else takes a body.
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

# The engine is a CUDA program; this process is not (see the top of the file).
device = "cuda" if (_ENGINE_CUDA_VISIBLE_DEVICES is None or _ENGINE_CUDA_VISIBLE_DEVICES.strip()) else "none"

model_loaded = False
weights_check = "pending" if WEIGHTS_SHA256 else "off"
load_seconds: Optional[float] = None

# Admission (model_lifecycle.RequestGate). Unlike the one-model-call services, a turn
# here is cheap: the engine batches the chunks of every request that holds one, so
# the default lets 32 requests send their chunks at once (ASR_MAX_CONCURRENCY), with
# up to 4 x that many waiting (ASR_MAX_QUEUE).
_ASR_CONCURRENCY = env_number("ASR_MAX_CONCURRENCY", 32, cast=int, minimum=1)
_ASR_MAX_QUEUE = env_number("ASR_MAX_QUEUE", 4 * _ASR_CONCURRENCY, cast=int, minimum=0)
_ASR_QUEUE_TIMEOUT_S = env_number("ASR_QUEUE_TIMEOUT_S", 60.0, minimum=0.1)
_RETRY_AFTER_S = 5


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
_chunk_slots: Optional[asyncio.Semaphore] = None
_client: Optional[httpx.AsyncClient] = None


def _request_id() -> str:
    return uuid.uuid4().hex[:12]


class EngineGone(RuntimeError):
    """The engine process exited while it was supposed to be serving."""


class EngineRefused(RuntimeError):
    """The engine answered a chunk with an error status."""

    def __init__(self, status: int, text: str):
        super().__init__(f"the engine answered {status}: {text[:300]}")
        self.status = status


@dataclass
class LoadedEngine:
    engine: Engine
    family: str
    endpoint: str


# The engine process outlives the slot's "resident" state: asleep, it is still there
# (and woken by the next load). Guarded by this lock between the slot's threads and
# /unload?mode=stop.
_engine: Optional[Engine] = None
_engine_lock = threading.RLock()


def _engine_env() -> dict:
    """The engine's environment: this one, minus the token and with the GPU visible again."""
    env = {k: v for k, v in os.environ.items()
           if k not in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "CUDA_VISIBLE_DEVICES")}
    if _ENGINE_CUDA_VISIBLE_DEVICES is not None:
        env["CUDA_VISIBLE_DEVICES"] = _ENGINE_CUDA_VISIBLE_DEVICES
    # The service downloaded (and checked) the checkpoint; the engine only reads that
    # directory and must not fetch anything else, nor report usage anywhere.
    env["HF_HUB_OFFLINE"] = "1"
    env["VLLM_NO_USAGE_STATS"] = "1"
    env["DO_NOT_TRACK"] = "1"
    if IDLE_MODE == "sleep":
        # /sleep and /wake_up are development endpoints; the engine listens on
        # 127.0.0.1 inside this container only, so nothing outside can reach them.
        env["VLLM_SERVER_DEV_MODE"] = "1"
    return env


def _sync_http(**kwargs) -> httpx.Client:
    return httpx.Client(transport=_SYNC_TRANSPORT, **kwargs)


def _model_dir() -> str:
    if is_local_model(MODEL_NAME):
        return MODEL_NAME
    from huggingface_hub import snapshot_download

    return snapshot_download(
        MODEL_NAME, revision=REVISION, allow_patterns=list(DOWNLOAD_PATTERNS), token=_HF_TOKEN)


def _read_family(model_dir: str) -> str:
    with open(os.path.join(model_dir, "config.json"), encoding="utf-8") as fh:
        return family_for(json.load(fh).get("model_type"))


def _start_new_engine() -> LoadedEngine:
    global weights_check, FAMILY, SUPPORTED_LANGUAGES, DEFAULT_LANGUAGE, _engine
    model_dir = _model_dir()
    if WEIGHTS_SHA256:
        hashed = time.monotonic()
        try:
            verify_weights(model_dir, WEIGHTS_SHA256, size=expected_size(WEIGHTS_SHA256))
        except Exception:
            weights_check = "failed"
            raise
        weights_check = "ok"
        logger.info("model.safetensors matches SHA-256 %s (%.1f s)", WEIGHTS_SHA256, time.monotonic() - hashed)

    family = _read_family(model_dir)
    if FAMILY is None or family != FAMILY:
        FAMILY = family
        SUPPORTED_LANGUAGES = _languages_for(family)
        DEFAULT_LANGUAGE = default_language_from_env(
            _DEFAULT_LANGUAGE_SETTING, supported=SUPPORTED_LANGUAGES, fallback="de")
    known = KNOWN_MODELS.get(MODEL_NAME)
    argv = engine_argv(model_dir, SERVED_NAME, ENGINE_PORT, ENGINE_SETTINGS,
                       default_max_model_len=known.max_model_len if known else None)
    engine = Engine(argv, f"http://127.0.0.1:{ENGINE_PORT}", env=_engine_env(), start_timeout_s=START_TIMEOUT_S,
                    poll_s=_ENGINE_POLL_S,
                    popen=lambda *a, **k: _popen(*a, **k), signal_group=lambda p, sig: _signal_group(p, sig),
                    http=_sync_http)
    engine.start()   # kills what it started before raising, so a failed start keeps no VRAM
    _engine = engine
    return LoadedEngine(engine=engine, family=family, endpoint=endpoint_for(family))


def _load_engine() -> LoadedEngine:
    """Wake the sleeping engine, or start one (called by the model slot, on its own thread)."""
    global model_loaded, load_seconds, _engine
    if _CONFIG_ERROR:
        raise RuntimeError(_CONFIG_ERROR)
    started = time.monotonic()
    with _engine_lock:
        engine = _engine
        if engine is not None and engine.alive() and engine.state == STATE_SLEEPING and FAMILY:
            try:
                seconds = engine.wake()
                logger.info("Engine woken in %.2f s", seconds)
                model_loaded = True
                return LoadedEngine(engine=engine, family=FAMILY, endpoint=endpoint_for(FAMILY))
            except Exception as e:
                logger.warning("Could not wake the engine (%s); starting a new one", e)
        if engine is not None:
            engine.stop()
            _engine = None
        logger.info("Starting vLLM for '%s' (revision %s)...", SERVED_NAME, REVISION)
        loaded = _start_new_engine()
    if VAD_MODE == "silero":
        try:  # the VAD's own model, once per worker (nothing is downloaded)
            _vad.warm()
        except Exception as e:  # SpeechGate falls back to the energy gate by itself
            logger.warning("VAD warm-up failed: %s", e)
    model_loaded = True
    load_seconds = round(time.monotonic() - started, 1)
    logger.info("vLLM serves %s (%s) after %.1f s", SERVED_NAME, loaded.family, load_seconds)
    return loaded


def _release_engine(loaded: LoadedEngine) -> None:
    """The slot's unload hook: put the engine to sleep (VLLM_ASR_IDLE_MODE=sleep) or stop it."""
    global model_loaded, _engine
    model_loaded = False
    with _engine_lock:
        if IDLE_MODE == "sleep" and loaded.engine.alive():
            try:
                seconds = loaded.engine.sleep_engine(1)
                logger.info("Engine asleep after %.2f s (weights in host RAM, KV cache dropped)", seconds)
                return
            except Exception as e:
                logger.warning("Could not put the engine to sleep (%s); stopping it", e)
        loaded.engine.stop()
        if _engine is loaded.engine:
            _engine = None


def _stop_idle_engine() -> bool:
    """Stop an engine that is asleep (no request holds it); True if one was stopped."""
    global _engine
    with _engine_lock:
        if _engine is None or _model_slot.resident or _model_slot.loading:
            return False
        _engine.stop()
        _engine = None
        return True


def _shutdown_engine() -> None:
    global _engine
    with _engine_lock:
        if _engine is not None:
            _engine.stop()
            _engine = None


# The engine is released after this many idle seconds (asleep or stopped).
#   >0 = seconds idle | 0 = as soon as the last request is done | -1 = never
MODEL_TTL = ttl_from_env(os.getenv, "ASR_MODEL_TTL", "MODEL_TTL", default=300.0)
_model_slot = ModelSlot(_load_engine, ttl_seconds=MODEL_TTL, name="vLLM-ASR", on_unload=_release_engine)


def get_model():
    with _model_slot.acquire() as model:
        return model


def _server_error(exc: Exception, request_id: str) -> HTTPException:
    if isinstance(exc, EngineGone):
        return HTTPException(
            status_code=503, detail="The engine stopped while transcribing; it is restarted by the next request. "
                                    f"Request id: {request_id}.",
            headers={"Retry-After": str(_RETRY_AFTER_S)})
    if not _model_slot.resident and _model_slot.last_error is not None:
        return HTTPException(
            status_code=503,
            detail=f"The engine could not be started (GET /ready says why). Request id: {request_id}.",
        )
    return HTTPException(
        status_code=500, detail=f"Transcription failed (internal error). Request id: {request_id}.",
    )


# --- inference -----------------------------------------------------------------


def _engine_client() -> tuple[httpx.AsyncClient, asyncio.Semaphore]:
    """The pooled client to the engine and the in-flight bound, made once inside the event loop."""
    global _client, _chunk_slots
    if _client is None:
        limits = httpx.Limits(max_connections=MAX_PENDING_CHUNKS, max_keepalive_connections=MAX_PENDING_CHUNKS)
        _client = httpx.AsyncClient(base_url=f"http://127.0.0.1:{ENGINE_PORT}", transport=_ASYNC_TRANSPORT,
                                    timeout=httpx.Timeout(REQUEST_TIMEOUT_S, connect=10.0), limits=limits)
        _chunk_slots = asyncio.Semaphore(MAX_PENDING_CHUNKS)
    return _client, _chunk_slots


async def _transcribe_chunk(loaded: LoadedEngine, piece, language: Optional[str], prompt: str,
                            keywords: list) -> ChunkText:
    client, slots = _engine_client()
    wav = wav_bytes(model_input(piece))
    async with slots:
        try:
            if loaded.endpoint == ENDPOINT_CHAT:
                response = await client.post(
                    "/v1/chat/completions", json=chat_body(SERVED_NAME, wav, language, keywords, MAX_NEW_TOKENS))
            else:
                response = await client.post(
                    "/v1/audio/transcriptions", files={"file": ("chunk.wav", wav, "audio/wav")},
                    data=transcription_fields(SERVED_NAME, language, prompt, MAX_NEW_TOKENS))
        except (httpx.ConnectError, httpx.RemoteProtocolError, httpx.ReadError) as exc:
            if not loaded.engine.alive():
                raise EngineGone("the engine exited") from exc
            raise
    if response.status_code != 200:
        raise EngineRefused(response.status_code, response.text)
    payload = response.json()
    out = parse_chat(payload) if loaded.endpoint == ENDPOINT_CHAT else parse_transcription(payload)
    return ChunkText(text=out.text, capped=out.capped)


async def _run_chunks(loaded: LoadedEngine, pieces: list, language: Optional[str], prompt: str,
                      keywords: list) -> list:
    """Every piece to the engine at once (vLLM batches them); the texts in the pieces' order.

    The first failure cancels the rest, so a dead engine does not leave dozens of
    chunk requests to time out one by one.
    """
    tasks = [asyncio.ensure_future(_transcribe_chunk(loaded, piece, language, prompt, keywords)) for piece in pieces]
    try:
        return list(await asyncio.gather(*tasks))
    except BaseException:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise


async def _asr(pieces: list, language: Optional[str], prompt: str, keywords: list) -> list:
    """The chunks through the engine: the engine pinned (awake), one of the gate's turns held."""
    async with _model_slot.acquire_async() as loaded:
        if not loaded.engine.alive():
            raise EngineGone("the engine is not running")
        async with _gate.turn():
            return await _run_chunks(loaded, pieces, language, prompt, keywords)


async def _forget_dead_engine() -> None:
    """After EngineGone: drop the dead engine from the slot (once nothing holds it), so the next request restarts it."""
    if _engine is not None and not _engine.alive():
        await _model_slot.try_unload_async()


def _resolve_request_language(language: Optional[str], label: str) -> tuple[Optional[str], list[str]]:
    """``(language for the engine or None to identify it, warnings)``; UnsupportedLanguage for one it does not know.

    An explicit ``auto`` asks a model that identifies languages (Qwen3-ASR, Whisper) to
    do so; for the others, and for no language at all, ``VLLM_ASR_DEFAULT_LANGUAGE``.
    """
    explicit_auto = (language or "").strip().lower() == "auto"
    if explicit_auto and detects_language(MODEL_NAME, FAMILY or FAMILY_WHISPER):
        return None, []
    resolved = resolve_language(language, default=DEFAULT_LANGUAGE, supported=SUPPORTED_LANGUAGES, model=label)
    warnings = []
    if explicit_auto:
        warnings.append(
            f"{label} does not identify the language; transcribed as '{resolved}' "
            f"(VLLM_ASR_DEFAULT_LANGUAGE). Send the language to choose another.")
    return resolved, warnings


async def _transcribe_upload(upload: UploadFile, language: Optional[str], keywords_field: Optional[str],
                             initial_prompt: Optional[str]) -> dict:
    """Full ``stt-form-v1`` response for one uploaded file."""
    started = time.time()
    label = f"{SERVED_NAME} (vLLM)"
    family = FAMILY or FAMILY_WHISPER
    try:
        resolved, warnings = _resolve_request_language(language, label)
        if family == FAMILY_GRANITE:
            keywords = parse_keywords(keywords_field, initial_prompt)
            prompt = ""
        else:
            keywords = parse_keywords(keywords_field)
            prompt = prompt_text(family, initial_prompt, keywords)
    except (UnsupportedLanguage, KeywordError) as e:
        raise HTTPException(status_code=422, detail=str(e))
    if family == FAMILY_COHERE and ((initial_prompt or "").strip() or keywords):
        warnings.append(f"{SERVED_NAME} takes no prompt; 'initial_prompt' and 'keywords' were ignored.")

    audio = await load_upload(upload, max_bytes=MAX_UPLOAD_BYTES, max_seconds=MAX_AUDIO_SECONDS, ffmpeg=_FFMPEG)
    regions = await _vad.regions(audio.samples)
    chunks = plan_speech_chunks(audio.samples, regions, max_chunk_s=CHUNK_S)
    logger.info(
        "Transcribing %s (%.1f MB, %.1f s, language=%s): %d chunks, %.1f s of speech",
        upload.filename, audio.size / MIB, audio.duration, resolved or "auto", len(chunks), speech_seconds(chunks))

    outputs = []
    if chunks:
        pieces = [audio.samples[c.start:c.end] for c in chunks]
        outputs = await _asr(pieces, resolved, prompt, keywords)
    else:
        warnings.append("No speech was detected in the audio.")
    segments, text, capped = build_segments(chunks, outputs)
    if capped:
        message = (
            f"Generation stopped at the {MAX_NEW_TOKENS}-token limit in the chunk(s) at {', '.join(capped)}; "
            f"that text is probably cut off. Raise VLLM_ASR_MAX_NEW_TOKENS or lower VLLM_ASR_CHUNK_S."
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
        "keywords": len(keywords) if family == FAMILY_GRANITE else 0,
        "prompt_applied": bool(keywords) if family == FAMILY_GRANITE else bool(prompt),
        "truncated": bool(capped),
        "warnings": warnings,
        "task": "transcribe",
        "model": SERVED_NAME,
        "engine": "vllm",
    }


# --- endpoints -----------------------------------------------------------------


def _engine_state() -> str:
    engine = _engine
    if engine is None:
        return STATE_STOPPED
    return engine.snapshot()["state"]


@app.get("/health")
async def health():
    """Liveness probe; `model_resident: false` (engine asleep or stopped) is an idle release, not an error."""
    return {
        "status": "ok",
        "model_loaded": model_loaded,
        "model_resident": _model_slot.resident,
        "model_ttl_seconds": MODEL_TTL,
        "active_requests": _model_slot.refs,
        "current_model": SERVED_NAME,
        "default_language": DEFAULT_LANGUAGE,
        "engine_state": _engine_state(),
        "device": device,
    }


@app.get("/ready")
async def ready():
    """Readiness, like every backend here: 503 `loading` / `load_failed`, never triggers a start."""
    state = _model_slot.readiness()
    if _preload_pending and not state["ever_loaded"] and state["reason"] == "ok":
        state = {**state, "ready": False, "reason": "loading"}
    body = {
        "ready": state["ready"],
        "reason": state["reason"],
        "model_resident": state["resident"],
        "model_ever_loaded": state["ever_loaded"],
        "current_model": SERVED_NAME,
        "device": device,
    }
    if state["detail"]:
        body["detail"] = state["error_category"] or "load_error"
    if state["ready"]:
        return body
    headers = {"Retry-After": "5"} if state["reason"] == "loading" else None
    return JSONResponse(content=body, status_code=503, headers=headers)


@app.post("/unload")
async def unload(mode: Optional[str] = None):
    """Release the engine's VRAM now (``VLLM_ASR_IDLE_MODE``, or ``?mode=stop`` to end the process); 409 while in use."""
    if mode not in (None, "", "sleep", "stop"):
        raise HTTPException(status_code=422, detail="mode is 'sleep' or 'stop'")
    result = await _model_slot.try_unload_async()
    if result["reason"] == "busy":
        return JSONResponse(status_code=409, content={"detail": "Model is in use; retry when idle", **result})
    stopped = False
    if mode == "stop":
        stopped = await asyncio.to_thread(_stop_idle_engine)
    return {"model_resident": _model_slot.resident, "engine_state": _engine_state(), "stopped": stopped, **result}


@app.get("/status")
async def status():
    """Detailed service status, including what the gateway's language selector offers."""
    engine = _engine
    family = FAMILY
    return {
        "status": "ok",
        "service": "vLLM-ASR",
        "device": device,
        "model_loaded": model_loaded,
        "model_resident": _model_slot.resident,
        "model_ttl_seconds": MODEL_TTL,
        "active_requests": _model_slot.refs,
        "current_model": SERVED_NAME,
        "revision": REVISION,
        "model_family": family,
        "offered_models": list(OFFERED_MODELS),
        "weights_sha256": WEIGHTS_SHA256,
        "weights_check": weights_check,
        "load_seconds": load_seconds,
        "supported_languages": sorted(SUPPORTED_LANGUAGES),
        "default_language": DEFAULT_LANGUAGE,
        "language_detection": detects_language(MODEL_NAME, family or FAMILY_WHISPER),
        "prompt": PROMPT_USE.get(family or FAMILY_WHISPER),
        "keywords": family == FAMILY_GRANITE,
        "timestamps": "chunk",
        "vad": _vad.describe(),
        "vad_workers": VAD_WORKERS,
        "chunk_seconds": CHUNK_S,
        "max_new_tokens": MAX_NEW_TOKENS,
        "max_upload_mb": MAX_UPLOAD_BYTES / MIB,
        "max_audio_seconds": MAX_AUDIO_SECONDS,
        "queue": _gate.snapshot(),
        "max_pending_chunks": MAX_PENDING_CHUNKS,
        "engine": {
            **(engine.snapshot() if engine is not None else {"state": STATE_STOPPED}),
            "idle_mode": IDLE_MODE,
            "gpu_memory_utilization": ENGINE_SETTINGS.gpu_memory_utilization,
            "max_model_len": ENGINE_SETTINGS.max_model_len or (
                KNOWN_MODELS[MODEL_NAME].max_model_len if MODEL_NAME in KNOWN_MODELS else None),
            "max_num_seqs": ENGINE_SETTINGS.max_num_seqs,
            "max_num_batched_tokens": ENGINE_SETTINGS.max_num_batched_tokens,
            "extra_args": list(ENGINE_SETTINGS.extra_args),
        },
    }


@app.post("/transcribe")
async def transcribe_audio(
    audio: UploadFile = File(...),
    language: Optional[str] = Form(None),
    keywords: Optional[str] = Form(None),
    initial_prompt: Optional[str] = Form(None),
):
    """Transcribe an audio file in the requested (or default) language.

    `initial_prompt` is Whisper's initial prompt, Qwen3-ASR's context and Granite's
    keyword list (comma separated; `keywords` is read the same way); Cohere takes
    none. Returns the `stt-form-v1` shape plus `speech_seconds`, `chunks`,
    `keywords`, `truncated` and `warnings`. 422 for an unsupported language, a
    keyword list over the limit, or audio that cannot be decoded; 413 for a file over
    MAX_UPLOAD_MB or longer than MAX_AUDIO_SECONDS; 503 + Retry-After when busy.
    """
    req_id = _request_id()
    try:
        with _gate.admit():
            return JSONResponse(content=await _transcribe_upload(audio, language, keywords, initial_prompt))
    except HTTPException:
        raise
    except Exception as e:
        logger.error("[%s] Transcription error: %s", req_id, e, exc_info=not isinstance(e, (EngineGone, EngineError)))
        if isinstance(e, EngineGone):
            await _forget_dead_engine()
        raise _server_error(e, req_id)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5011)
