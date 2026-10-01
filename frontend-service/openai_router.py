"""OpenAI-compatible `/v1` audio surface.

The point of this module: a client must not care which device it is talking to.
On the ARM SBC the backend is whisper-cpp; on the workstation it is
faster-whisper. Same URL, same request, same response shape.

Scope is deliberately the *minimum real clients exercise* — openai-python,
openai-node, curl, Home Assistant and Open WebUI. Field names and defaults below
are taken from the OpenAI OpenAPI spec (`info.version: 2.3.0`); the
non-obvious ones carry a comment saying so, because getting a name wrong defeats
the entire purpose.

Kept out of `app.py` because that module is already long, and wired up through a
factory rather than imports so there is no circular dependency.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import math
import os
import sys
import wave
from array import array
from typing import Any, Callable, Optional

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse, Response

logger = logging.getLogger(__name__)

# The guide states a 25 MB upload limit. Mirrored here so a large upload is
# refused cheaply rather than being buffered into an SBC's memory.
MAX_UPLOAD_BYTES = 25 * 1024 * 1024

# The file is drained from the spooled upload in slices of this size so the
# limit can abort the read instead of first materialising the whole thing.
_READ_CHUNK = 1024 * 1024

# Advertised model ids. `whisper-1` is chosen deliberately: the spec says
# streaming "is not supported for the whisper-1 model and will be ignored", and
# the guide scopes `timestamp_granularities[]` to `whisper-1`. Advertising it
# makes ignoring `stream` and honouring granularities both spec-legal, which is
# exactly this project's capability profile.
STT_MODEL_ID = "whisper-1"
TTS_MODEL_ID = "tts-1"

# AudioResponseFormat enum from the spec. `diarized_json` is listed but scoped
# to a diarizing model we do not have, so it is rejected explicitly rather than
# silently downgraded to something else.
STT_FORMATS = {"json", "text", "srt", "verbose_json", "vtt", "diarized_json"}
STT_FORMATS_SUPPORTED = {"json", "text", "srt", "verbose_json", "vtt"}
# Formats that need per-segment timings rather than just the text.
_SEGMENT_FORMATS = {"srt", "verbose_json", "vtt"}

# `opus`, `aac` and `flac` are still refused: they would need an encoder each
# and no client here has asked for one. `pcm` is the WAV every backend already
# emits without its header, resampled to OpenAI's 24 kHz where it is not already.
SPEECH_FORMATS = {"mp3", "opus", "aac", "flac", "wav", "pcm"}
SPEECH_FORMATS_SUPPORTED = {"mp3", "wav", "pcm"}
SPEECH_MEDIA_TYPES = {"mp3": "audio/mpeg", "wav": "audio/wav", "pcm": "audio/pcm"}

# Spec: CreateSpeechRequest.input has maxLength 4096.
MAX_SPEECH_INPUT = 4096

# Ceiling for one ffmpeg transcode. Well above any realistic synthesis length.
FFMPEG_TIMEOUT_S = 120.0

# OpenAI documents `pcm` as 24 kHz, 16-bit, mono, little-endian, and stock clients
# play it back at exactly that rate. Backends produce 22.05 kHz (Piper medium/high),
# 16 kHz (Piper low) or 24 kHz (Qwen3, Chatterbox), so anything else is resampled.
PCM_SAMPLE_RATE = 24000


def _env_positive_int(name: str, default: int) -> int:
    """A positive integer from the environment, or `default` when unset or unusable."""
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        logger.warning("%s=%r is not an integer; using %d", name, raw, default)
        return default
    if value < 1:
        logger.warning("%s=%r must be at least 1; using %d", name, raw, default)
        return default
    return value


# Every mp3 (the default format) and every resampled pcm response is one ffmpeg
# process. They are cheap one at a time, but a burst of requests used to start a
# process each, with the WAV held in memory beside it. Per worker process; over
# the cap the request is answered 503 + Retry-After, which OpenAI clients retry.
MAX_CONCURRENT_FFMPEG = _env_positive_int("MAX_CONCURRENT_FFMPEG", 4)
FFMPEG_RETRY_AFTER_S = 2


class Slots:
    """A counter of concurrent users of something scarce. Non-blocking: a full house is refused.

    Plain integers rather than an asyncio.Semaphore: nothing ever waits, and a
    semaphore binds to one event loop, which breaks under a test client that
    starts a loop per request. One event loop per worker process, so no lock.
    """

    def __init__(self, limit: int):
        self.limit = limit
        self.active = 0

    def try_acquire(self) -> bool:
        if self.active >= self.limit:
            return False
        self.active += 1
        return True

    def release(self) -> None:
        self.active = max(0, self.active - 1)


_ffmpeg_slots = Slots(MAX_CONCURRENT_FFMPEG)


class FfmpegBusy(Exception):
    """Every ffmpeg slot is taken; the caller should answer 503 and ask for a retry."""

# OpenAI's own voice names. They mean nothing to any backend here, so forwarding
# one makes the provider fail to resolve a voice instead of using its default.
OPENAI_PLACEHOLDER_VOICES = {
    "alloy", "echo", "fable", "onyx", "nova", "shimmer",
    "ash", "ballad", "coral", "sage", "verse", "marin", "cedar",
}


class UpstreamUnavailable(HTTPException):
    """A backend could not be reached at all (refused, DNS, timeout).

    Distinct from a backend that answered with an error: only this one justifies
    trying a different provider, because the answer "I am up and this request is
    wrong" would come back identically from any other.
    """


def openai_error(status: int, message: str, *, param: Optional[str] = None,
                 code: Optional[str] = None) -> JSONResponse:
    """Build the OpenAI error envelope.

    Spec: `Error.required == ['type','message','param','code']` and
    `ErrorResponse.required == ['error']`. openai-python reads `code`, `param`
    and `type` off `body.get("error", body)`, so FastAPI's default
    `{"detail": ...}` leaves all three None and clients cannot branch on them.
    """
    if status in (400, 404, 413, 415, 422):
        err_type = "invalid_request_error"
    elif status == 401:
        err_type = "authentication_error"
    elif status == 403:
        err_type = "permission_error"
    elif status == 429:
        err_type = "rate_limit_error"
    else:
        err_type = "server_error"
    return JSONResponse(
        status_code=status,
        content={"error": {"message": message, "type": err_type, "param": param, "code": code}},
    )


def is_v1_path(path: str) -> bool:
    """True for `/v1` and everything under it, and nothing merely starting `/v1x`."""
    return path == "/v1" or path.startswith("/v1/")


def validation_error_response(exc: Any) -> JSONResponse:
    """Render a FastAPI RequestValidationError in the OpenAI envelope.

    Without this a missing `file` or `temperature=abc` answers with FastAPI's
    `{"detail": [...]}`, which leaves `error.code`/`error.param` unreadable to
    every OpenAI client. Status is 400 (what OpenAI answers for a malformed
    request), not FastAPI's 422.
    """
    errors = exc.errors() if hasattr(exc, "errors") else []
    first = errors[0] if errors else {}
    loc = [str(part) for part in first.get("loc", ()) if part not in ("body", "query", "path")]
    param = loc[-1] if loc else None
    message = first.get("msg") or "Invalid request."
    if param:
        message = f"{message}: '{param}'"
    code = "missing_required_parameter" if first.get("type") == "missing" else "invalid_value"
    return openai_error(400, message, param=param, code=code)


def http_exception_response(request: Request, exc: Any) -> JSONResponse:
    """Render a framework HTTPException (unknown route, wrong method) as an envelope."""
    status = exc.status_code
    detail = exc.detail
    if status == 404:
        message = f"Invalid URL ({request.method} {request.url.path})"
    elif status == 405:
        message = f"Method {request.method} is not allowed for {request.url.path}"
    elif isinstance(detail, str):
        message = detail
    else:
        message = json.dumps(detail)
    response = openai_error(status, message)
    # Keep headers such as `Allow` on a 405.
    for key, value in (getattr(exc, "headers", None) or {}).items():
        response.headers[key] = value
    return response


async def _run_ffmpeg(output_args: list[str], wav_bytes: bytes, label: str) -> Optional[bytes]:
    """Pipe a WAV through ffmpeg and return what it wrote, or None if that did not work.

    None covers ffmpeg not being installed, exiting non-zero, timing out and
    producing nothing; the reason is logged. Raises `FfmpegBusy` before starting
    anything when `MAX_CONCURRENT_FFMPEG` processes are already running.

    Asynchronous on purpose: a blocking `subprocess.run` inside an `async def`
    stalls the whole worker for the length of the encode — health checks and the
    `/ws/stt` relay included — and every mp3 request is one, because mp3 is the
    default format.
    """
    if not _ffmpeg_slots.try_acquire():
        logger.warning("all %d ffmpeg slots are busy; refusing the %s conversion",
                       _ffmpeg_slots.limit, label)
        raise FfmpegBusy()
    try:
        return await _ffmpeg_pipe(output_args, wav_bytes, label)
    finally:
        _ffmpeg_slots.release()


async def _ffmpeg_pipe(output_args: list[str], wav_bytes: bytes, label: str) -> Optional[bytes]:
    try:
        proc = await asyncio.create_subprocess_exec(
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-f", "wav", "-i", "pipe:0", *output_args, "pipe:1",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError:
        logger.warning("ffmpeg not installed; cannot serve response_format=%s", label)
        return None
    except OSError as e:
        logger.warning("ffmpeg could not be started: %s", e)
        return None

    try:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(wav_bytes), timeout=FFMPEG_TIMEOUT_S)
    except asyncio.TimeoutError:
        logger.warning("ffmpeg %s transcode timed out after %gs", label, FFMPEG_TIMEOUT_S)
        return None
    except Exception as e:
        logger.warning("ffmpeg %s transcode error: %s", label, e)
        return None
    finally:
        # A cancelled request (client went away) or a timeout must not leave an
        # encoder running.
        if proc.returncode is None:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            # Bounded: asyncio only reports the exit once every pipe has closed,
            # and a grandchild that inherited them would otherwise hold this up.
            try:
                await asyncio.wait_for(proc.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning("ffmpeg did not exit after being killed")

    if proc.returncode == 0 and stdout:
        return stdout
    logger.warning("ffmpeg %s transcode failed (rc=%s): %s",
                   label, proc.returncode, stderr[:200].decode("utf-8", "replace"))
    return None


async def _wav_to_mp3(wav_bytes: bytes) -> Optional[bytes]:
    """Transcode WAV to MP3 with ffmpeg. Returns None if ffmpeg is unavailable.

    mp3 is the spec's DEFAULT response_format, so a client that sends no format
    at all expects it. Every TTS backend here emits WAV, so the container
    conversion belongs at the gateway rather than in each service.
    """
    return await _run_ffmpeg(["-f", "mp3", "-b:a", "64k"], wav_bytes, "mp3")


def _wav_info(wav_bytes: bytes) -> tuple[int, int, int]:
    """(channels, sample width in bytes, sample rate) from a WAV header. ValueError if it is not one."""
    try:
        with wave.open(io.BytesIO(wav_bytes), "rb") as reader:
            return reader.getnchannels(), reader.getsampwidth(), reader.getframerate()
    except (wave.Error, EOFError) as e:
        raise ValueError(f"not a PCM WAV file ({e})") from e


async def wav_to_pcm_24k(wav_bytes: bytes) -> Optional[bytes]:
    """A 16-bit PCM WAV as raw 24 kHz mono little-endian samples: OpenAI's `pcm` format.

    A WAV that already is 24 kHz only loses its header (and is mixed down to mono
    if it has more channels), with no subprocess. Any other rate goes through
    ffmpeg (`-ar 24000 -ac 1 -f s16le`); None means that was not possible
    (ffmpeg missing or failing). Raises ValueError for audio that is not 16-bit
    PCM and `FfmpegBusy` when no ffmpeg slot is free.
    """
    _channels, width, rate = _wav_info(wav_bytes)
    if width != 2:
        raise ValueError(f"unsupported sample width: {width * 8}-bit (need 16-bit)")
    if rate == PCM_SAMPLE_RATE:
        raw, _ = await asyncio.to_thread(wav_to_pcm, wav_bytes)
        return raw
    return await _run_ffmpeg(
        ["-ar", str(PCM_SAMPLE_RATE), "-ac", "1", "-f", "s16le"], wav_bytes, "pcm")


def wav_to_pcm(wav_bytes: bytes) -> tuple[bytes, int]:
    """Strip a 16-bit PCM WAV down to raw little-endian mono samples.

    Returns ``(pcm, sample_rate)``. The rate is the WAV's own: nothing is
    resampled here (`wav_to_pcm_24k` does that). Raises ValueError for anything
    that is not 16-bit PCM.
    """
    try:
        with wave.open(io.BytesIO(wav_bytes), "rb") as reader:
            channels = reader.getnchannels()
            width = reader.getsampwidth()
            rate = reader.getframerate()
            raw = reader.readframes(reader.getnframes())
    except (wave.Error, EOFError) as e:
        raise ValueError(f"not a PCM WAV file ({e})") from e
    if width != 2:
        raise ValueError(f"unsupported sample width: {width * 8}-bit (need 16-bit)")

    if channels > 1:
        samples = array("h")
        samples.frombytes(raw[: len(raw) - len(raw) % (2 * channels)])
        if sys.byteorder == "big":
            samples.byteswap()
        mono = array("h", (sum(samples[i:i + channels]) // channels
                           for i in range(0, len(samples), channels)))
        if sys.byteorder == "big":
            mono.byteswap()
        raw = mono.tobytes()
    return raw, rate


def backend_detail(exc: Any) -> str:
    """The text of a backend failure as a client may see it.

    `post_form`/`post_json` raise HTTPExceptions whose detail the gateway has
    already reduced to a short sentence and a request id (the backend's own body,
    with its tracebacks and paths, is only in the log). A detail of any other
    shape - a dict, a list - is not something to show a client, so it is not.
    """
    detail = getattr(exc, "detail", None)
    return detail if isinstance(detail, str) and detail.strip() else "The backend could not complete the request."


# A backend refusal that is about what the caller asked for (an unknown speaker, a
# language the model cannot speak, text with nothing to say, a text limit) is the
# caller's to fix. OpenAI answers those 400 `invalid_request_error` (413 for size),
# and its SDKs do not retry them; a 502 made them retry the same request. Everything
# else stays a 502, which they do retry: 5xx, an unreachable backend, and 4xx answers
# about the backend's own state or the internal hop (Piper's 404 "no voice installed
# at all", Qwen3's 409 "another model is loaded", a 401/403 between the services).
_CALLER_FAULT_STATUS = {400: 400, 413: 413, 422: 400}


def _retry_after(exc: Any) -> Optional[str]:
    """A numeric Retry-After carried by an HTTPException, else None."""
    for key, value in (getattr(exc, "headers", None) or {}).items():
        if key.lower() == "retry-after":
            text = str(value).strip()
            return text if text.isascii() and text.isdigit() else None
    return None


def backend_error(what: str, exc: Any) -> JSONResponse:
    """The /v1 answer for a backend that answered with an error (`what`: "Speech", "Transcription").

    The message keeps the backend's sentence, which the gateway has already reduced
    to something a client may see. A busy backend's 503 (its queue is full, the
    request waited its time for a turn, the GPU is out of memory) comes with a
    Retry-After; that stays a 503 `server_busy` with the header, which OpenAI
    clients wait out and retry, like the gateway's own ffmpeg refusal. Its sentence
    says what is wrong itself ("The service is busy: ...", "GPU out of memory ..."),
    so it is only prefixed with the backend it came from.
    """
    status = _CALLER_FAULT_STATUS.get(getattr(exc, "status_code", None))
    if status is not None:
        return openai_error(status, f"{what} backend rejected the request: {backend_detail(exc)}")
    retry_after = _retry_after(exc) if getattr(exc, "status_code", None) == 503 else None
    if retry_after is not None:
        busy = openai_error(503, f"{what} backend: {backend_detail(exc)}", code="server_busy")
        busy.headers["Retry-After"] = retry_after
        return busy
    return openai_error(502, f"{what} backend failed: {backend_detail(exc)}")


# --- a caller that hangs up --------------------------------------------------------


class ClientDisconnected(HTTPException):
    """The caller closed the connection before the backend answered (499, nginx's code for it).

    The answer never goes out and nothing logs its status: uvicorn drops whatever is
    sent after a disconnect, before its access-log line (and the gateway runs with
    --no-access-log). The hang-up is recorded by the INFO line in `unless_client_gone`.
    A class of its own so that a handler turning a backend's HTTPException into a 502
    can let it through.
    """


async def client_gone(request: Request) -> None:
    """Return once the client has closed the connection; never return otherwise.

    Nothing else notices a caller who leaves while a handler waits for a backend:
    uvicorn and Starlette never cancel a handler (only a StreamingResponse listens
    for the disconnect), so a non-streaming request kept the backend working for
    nobody until it finished. Call it only after the body has been read: it takes
    the request's remaining ASGI messages.
    """
    try:
        while True:
            message = await request.receive()
            if message["type"] == "http.disconnect":
                return
    except Exception as exc:     # the connection state cannot be read: wait for the backend alone
        logger.debug("cannot watch the client connection: %s", exc)
        await asyncio.Event().wait()


async def unless_client_gone(request: Request, awaitable: Any) -> Any:
    """Await *awaitable*, unless the caller hangs up first.

    Then the awaitable is cancelled, which makes httpx close the upstream connection
    (a backend that watches for that stops generating), and `ClientDisconnected`
    (499) is raised. Only the wait for the backend's answer is raced: once a
    StreamingResponse exists, Starlette's own disconnect handling takes over.
    """
    work = asyncio.ensure_future(awaitable)
    gone = asyncio.ensure_future(client_gone(request))
    try:
        await asyncio.wait({work, gone}, return_when=asyncio.FIRST_COMPLETED)
    except BaseException:             # this handler itself was cancelled (shutdown)
        work.cancel()
        work.add_done_callback(_retrieved)
        raise
    finally:
        gone.cancel()
    if work.done():
        return work.result()
    work.cancel()
    await asyncio.wait({work})        # let it unwind: that closes the upstream connection
    _retrieved(work)
    logger.info("the caller of %s %s hung up before the backend answered; the backend call was cancelled",
                request.method, request.url.path)
    raise ClientDisconnected(status_code=499, detail="Client closed the request.")


def _retrieved(task: "asyncio.Future") -> None:
    """Mark a finished task's outcome as seen, so an error it ended with is not logged as lost."""
    if not task.cancelled():
        task.exception()


def _number(value: Any, default: float) -> float:
    """A finite float from whatever a backend put in a numeric field."""
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def extract_segments(payload: dict, text: str, duration: Optional[float]) -> list[dict]:
    """Segments from a backend payload, or one spanning the file when it has none.

    whisper.cpp only reports segments for `verbose_json`, and some deployments
    answer with a bare `{"text": ...}` regardless. A single segment is still a
    valid, honest rendering: the whole transcript, start to end of the audio.
    """
    segments: list[dict] = []
    raw = payload.get("segments")
    if isinstance(raw, list):
        for entry in raw:
            if not isinstance(entry, dict):
                continue
            seg_text = str(entry.get("text") or "").strip()
            if not seg_text:
                continue
            start = _number(entry.get("start"), 0.0)
            segments.append({
                "start": start,
                "end": max(_number(entry.get("end"), start), start),
                "text": seg_text,
                "avg_logprob": _number(entry.get("avg_logprob"), 0.0),
                "no_speech_prob": _number(entry.get("no_speech_prob"), 0.0),
            })
    if not segments and text:
        segments.append({
            "start": 0.0, "end": duration or 0.0, "text": text,
            "avg_logprob": 0.0, "no_speech_prob": 0.0,
        })
    return segments


def _timestamp(seconds: float, decimal: str) -> str:
    total_ms = int(round(max(seconds, 0.0) * 1000))
    hours, rest = divmod(total_ms, 3_600_000)
    minutes, rest = divmod(rest, 60_000)
    secs, ms = divmod(rest, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{decimal}{ms:03d}"


def render_srt(segments: list[dict]) -> str:
    """SubRip: 1-based index, `HH:MM:SS,mmm --> HH:MM:SS,mmm`, blank line between cues."""
    cues = [
        f"{index}\n{_timestamp(seg['start'], ',')} --> {_timestamp(seg['end'], ',')}\n{seg['text']}\n"
        for index, seg in enumerate(segments, start=1)
    ]
    return "\n".join(cues)


def render_vtt(segments: list[dict]) -> str:
    """WebVTT: a `WEBVTT` header, then cues with a `.` millisecond separator."""
    cues = [
        f"{_timestamp(seg['start'], '.')} --> {_timestamp(seg['end'], '.')}\n{seg['text']}\n"
        for seg in segments
    ]
    return "WEBVTT\n\n" + "\n".join(cues)


def build_verbose_json(payload: dict, text: str, segments: list[dict],
                       duration: Optional[float], language: str, temperature: float) -> dict:
    """The spec's CreateTranscriptionResponseVerboseJson.

    `TranscriptionSegment` requires `tokens`, `seek`, `compression_ratio` and the
    probabilities, and openai-python validates them. Where a backend cannot
    measure one it is filled with a neutral value (`tokens: []`,
    `compression_ratio: 0`) rather than left out, which would fail the parse.
    """
    if duration is None:
        duration = segments[-1]["end"] if segments else 0.0
    return {
        "task": "transcribe",
        "language": str(payload.get("language") or language or ""),
        "duration": duration,
        "text": text,
        "segments": [
            {
                "id": index,
                "seek": 0,
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"],
                "tokens": [],
                "temperature": temperature,
                "avg_logprob": seg["avg_logprob"],
                "compression_ratio": 0.0,
                "no_speech_prob": seg["no_speech_prob"],
            }
            for index, seg in enumerate(segments)
        ],
    }


async def _read_capped(file: UploadFile, limit: int) -> Optional[bytes]:
    """Read an upload, or return None the moment it exceeds `limit` bytes.

    Checking `len(await file.read())` afterwards, as this used to, allocates the
    whole oversized file first — the very cost the limit exists to avoid.
    """
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await file.read(_READ_CHUNK)
        if not chunk:
            break
        total += len(chunk)
        if total > limit:
            return None
        chunks.append(chunk)
    return b"".join(chunks)


def build_router(
    *,
    get_provider: Callable[..., dict],
    registry: dict,
    post_form: Callable,
    post_json: Callable,
    provider_health: Callable,
    build_tts_payload: Callable,
    mp3_converter: Optional[Callable] = None,
) -> APIRouter:
    """Create the `/v1` router.

    Dependencies are injected rather than imported so this module does not have
    to import `app.py`, which imports nothing from here.
    """
    router = APIRouter(prefix="/v1", tags=["openai"])

    def _default_stt_provider() -> str:
        return (registry.get("ui") or {}).get("default_stt_provider") or "whisper"

    def _default_tts_provider() -> str:
        return (registry.get("ui") or {}).get("default_tts_provider") or "piper"

    async def _healthy_alternative(exclude: str) -> Optional[str]:
        """The one other STT provider that is up, or None if zero or several are.

        More than one candidate is a choice the operator did not make, and
        picking between models silently changes accuracy and language coverage —
        so that case stays an error.
        """
        try:
            report = await provider_health()
        except Exception as e:
            logger.warning("health probe for STT fallback failed: %s", e)
            return None
        health = (report or {}).get("providers") or {}
        candidates = [
            pid for pid, provider in (registry.get("providers") or {}).items()
            if provider.get("kind") == "stt" and pid != exclude
            and (health.get(pid) or {}).get("healthy") is True
        ]
        return candidates[0] if len(candidates) == 1 else None

    # ---------------------------------------------------------------- STT ---

    @router.post("/audio/transcriptions")
    async def create_transcription(
        request: Request,
        file: UploadFile = File(...),
        # Required by the spec, but advisory here: this deployment serves
        # whatever backend it has. Never 400 on an unknown model id — clients
        # hardcode "whisper-1" and would break for no reason.
        model: str = Form(None),
        language: str = Form(None),
        prompt: str = Form(None),
        response_format: str = Form("json"),
        temperature: float = Form(0.0),
        stream: bool = Form(False),
    ):
        """Transcribe audio. Mirrors POST /v1/audio/transcriptions.

        Unknown fields are accepted and ignored rather than rejected: the spec
        has 14 request fields and grows, and a 422 on an unrecognised one breaks
        clients that send newer parameters harmlessly.
        """
        fmt = (response_format or "json").strip().lower()
        if fmt not in STT_FORMATS:
            return openai_error(
                400, f"Invalid value for 'response_format': {response_format}",
                param="response_format", code="invalid_value")
        if fmt not in STT_FORMATS_SUPPORTED:
            # Explicit refusal beats silently returning a different shape.
            return openai_error(
                400,
                f"response_format '{fmt}' is not supported by this deployment. "
                f"Supported: {', '.join(sorted(STT_FORMATS_SUPPORTED))}.",
                param="response_format", code="unsupported_value")

        content = await _read_capped(file, MAX_UPLOAD_BYTES)
        if content is None:
            return openai_error(
                413,
                f"File is too large. Maximum is {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.",
                param="file", code="file_too_large")
        if not content:
            return openai_error(400, "Audio file is empty.", param="file")

        provider_id = _default_stt_provider()
        try:
            provider = get_provider(provider_id, kind="stt")
        except HTTPException as exc:
            return openai_error(503, f"No speech-to-text provider available: {exc.detail}")

        # "auto" must be sent explicitly, not omitted: whisper.cpp defaults to
        # English when the field is absent, which silently mis-transcribes, and
        # the stt-form-v1 services read a missing field as "use
        # STT_DEFAULT_LANGUAGE" but an explicit "auto" as "detect".
        lang = (language or "").strip()
        if lang.lower() == "auto":
            lang = "auto"
        want_segments = fmt in _SEGMENT_FORMATS

        async def transcribe_with(pid: str, prov: dict):
            contract = (prov.get("contracts") or {}).get("transcribe")
            data: dict[str, Any] = {}
            if contract == "openai-audio-transcriptions-v1":
                files = [("file", (file.filename or "audio.wav", content,
                                   file.content_type or "application/octet-stream"))]
                # whisper.cpp only returns timings for verbose_json.
                data["response_format"] = "verbose_json" if want_segments else "json"
                data["language"] = lang or "auto"
                path = prov.get("transcribe_path") or "/v1/audio/transcriptions"
            else:
                files = [("audio", (file.filename or "audio.wav", content,
                                    file.content_type or "application/octet-stream"))]
                # Every stt-form-v1 service accepts the literal "auto" and detects
                # (canary, which cannot, takes its default). Absence is NOT the
                # same thing: it means the operator's STT_DEFAULT_LANGUAGE.
                if lang:
                    data["language"] = lang
                if prompt:
                    data["initial_prompt"] = prompt
                if temperature:
                    data["temperature"] = str(temperature)
                path = "/transcribe"
            return await post_form(pid, path, data=data, files=files, timeout=600.0)

        served_by = provider_id
        try:
            upstream = await transcribe_with(provider_id, provider)
        except UpstreamUnavailable as exc:
            # Only the implicit default may be swapped: this route never named a
            # provider, so "the configured default is down" is the one case where
            # a different backend still answers the question the client asked.
            # (/api/stt names its provider and stays strict.)
            alternative = await _healthy_alternative(provider_id)
            if alternative is None:
                return openai_error(502, f"Transcription backend failed: {backend_detail(exc)}")
            logger.warning(
                "default STT provider '%s' is unreachable (%s); falling back to '%s'",
                provider_id, exc.detail, alternative)
            served_by = alternative
            try:
                upstream = await transcribe_with(alternative, get_provider(alternative, kind="stt"))
            except HTTPException as fallback_exc:
                if getattr(fallback_exc, "status_code", None) in _CALLER_FAULT_STATUS:
                    # The stand-in's own limits (a language it cannot do, a shorter audio
                    # cap), not a verdict on the request: the default was never asked and
                    # may well take it once it is back. A 502, which the SDKs retry, as
                    # when there is no stand-in at all.
                    failed = openai_error(502, (
                        f"Transcription fallback '{alternative}' (the default '{provider_id}' "
                        f"is unreachable) rejected the request: {backend_detail(fallback_exc)}"))
                else:
                    failed = backend_error("Transcription", fallback_exc)
                failed.headers["X-Provider"] = alternative
                failed.headers["X-Provider-Fallback"] = f"{provider_id}->{alternative}"
                return failed
        except HTTPException as exc:
            return backend_error("Transcription", exc)

        try:
            payload = upstream.json()
        except Exception:
            return openai_error(502, "Transcription backend returned a non-JSON response.")
        if not isinstance(payload, dict):
            return openai_error(502, "Transcription backend returned an unexpected response.")

        headers: dict[str, str] = {}
        if served_by != provider_id:
            headers["X-Provider"] = served_by
            headers["X-Provider-Fallback"] = f"{provider_id}->{served_by}"

        text = str(payload.get("text") or payload.get("transcript") or "").strip()
        if fmt == "text":
            # openai-python returns response.text verbatim for this format, so
            # the body must be raw — a JSON-quoted string would reach the caller
            # with literal quote characters.
            return PlainTextResponse(text, media_type="text/plain; charset=utf-8", headers=headers)
        if not want_segments:
            return JSONResponse({"text": text}, headers=headers)

        duration = _number(payload.get("duration"), 0.0) or None
        segments = extract_segments(payload, text, duration)
        if not text:
            text = " ".join(seg["text"] for seg in segments).strip()
        if fmt == "srt":
            return PlainTextResponse(
                render_srt(segments), media_type="text/plain; charset=utf-8", headers=headers)
        if fmt == "vtt":
            return PlainTextResponse(
                render_vtt(segments), media_type="text/plain; charset=utf-8", headers=headers)
        return JSONResponse(
            build_verbose_json(payload, text, segments, duration, lang, temperature),
            headers=headers)

    # ---------------------------------------------------------------- TTS ---

    @router.post("/audio/speech")
    async def create_speech(request: Request):
        """Synthesize speech. Mirrors POST /v1/audio/speech (JSON body, not multipart)."""
        try:
            body = await request.json()
        except Exception:
            return openai_error(400, "Request body must be JSON.")
        if not isinstance(body, dict):
            return openai_error(400, "Request body must be a JSON object.")

        text = body.get("input")
        if not isinstance(text, str) or not text.strip():
            return openai_error(400, "Missing required parameter: 'input'.", param="input")
        if len(text) > MAX_SPEECH_INPUT:
            return openai_error(
                400, f"'input' exceeds the maximum of {MAX_SPEECH_INPUT} characters.",
                param="input", code="string_above_max_length")

        fmt = str(body.get("response_format") or "mp3").strip().lower()
        if fmt not in SPEECH_FORMATS:
            return openai_error(400, f"Invalid value for 'response_format': {fmt}",
                                param="response_format", code="invalid_value")
        if fmt not in SPEECH_FORMATS_SUPPORTED:
            return openai_error(
                400,
                f"response_format '{fmt}' is not supported by this deployment. "
                f"Supported: {', '.join(sorted(SPEECH_FORMATS_SUPPORTED))}.",
                param="response_format", code="unsupported_value")

        speed = body.get("speed", 1.0)
        try:
            speed = float(speed)
        except (TypeError, ValueError):
            return openai_error(400, "'speed' must be a number.", param="speed")
        if not (0.25 <= speed <= 4.0):
            return openai_error(400, "'speed' must be between 0.25 and 4.0.", param="speed")

        provider_id = _default_tts_provider()
        try:
            provider = get_provider(provider_id, kind="tts")
        except HTTPException as exc:
            return openai_error(503, f"No text-to-speech provider available: {exc.detail}")

        # `voice` is not validated against a list here. The spec's own prose and its
        # VoiceIdsShared enum disagree, and the schema accepts any string, so the
        # name goes to the default provider as it is: Piper picks another voice for
        # one it does not have, Chatterbox has no voices to pick from, and Qwen3 and
        # Magpie answer 400 listing theirs (a 400 here too, see backend_error).
        voice = str(body.get("voice") or "").strip()
        if voice.lower() in OPENAI_PLACEHOLDER_VOICES:
            # One of OpenAI's own names, which resolves to nothing here.
            voice = ""

        # Built by the gateway's shared translator, not inline. Each backend
        # names these fields differently (`lang`/`speaker` on qwen3 against
        # `language`/`voice` on piper), and Pydantic drops unknown keys without
        # complaint — so a hand-rolled body here returned the wrong language in
        # the wrong voice with HTTP 200 on any deployment whose default TTS
        # provider was not piper.
        try:
            payload, read_timeout = build_tts_payload(
                provider_id,
                provider,
                text=text,
                voice=voice or None,
                language=str(body.get("language") or "auto"),
                speed=speed,
            )
        except HTTPException as exc:
            return openai_error(
                503, f"Text-to-speech provider '{provider_id}' cannot serve this request: {exc.detail}")

        try:
            # Raced against the caller hanging up: a non-streaming /tts answers only
            # when the whole text is done, which can be many minutes of GPU time.
            upstream = await unless_client_gone(request, post_json(
                provider_id, "/tts", payload, timeout=max(read_timeout, 600.0)))
        except ClientDisconnected:
            raise
        except HTTPException as exc:
            return backend_error("Speech", exc)

        audio = upstream.content
        headers = {"X-Provider": provider_id}
        try:
            if fmt == "mp3":
                # Looked up at call time so a deployment (or a test) can swap the
                # converter without rebuilding the router.
                converted = await (mp3_converter or _wav_to_mp3)(audio)
                if converted is None:
                    return openai_error(
                        501,
                        "response_format 'mp3' requires ffmpeg, which is not available "
                        "in this deployment. Use response_format='wav'.",
                        param="response_format", code="unsupported_value")
                audio = converted
            elif fmt == "pcm":
                try:
                    converted = await wav_to_pcm_24k(audio)
                except ValueError as e:
                    return openai_error(502, f"Speech backend returned audio that cannot be sent as pcm: {e}")
                if converted is None:
                    return openai_error(
                        501,
                        "response_format 'pcm' needs ffmpeg to resample this voice to 24 kHz, "
                        "and it is not available in this deployment. Use response_format='wav'.",
                        param="response_format", code="unsupported_value")
                audio = converted
                # OpenAI's pcm is 24 kHz mono, always: wav_to_pcm_24k resamples
                # whatever the backend produced. Raw samples carry no header, so
                # the rate still travels out of band for clients that ask.
                headers["X-Sample-Rate"] = str(PCM_SAMPLE_RATE)
        except FfmpegBusy:
            busy = openai_error(
                503, "The server is busy converting audio. Try again shortly.",
                code="server_busy")
            busy.headers["Retry-After"] = str(FFMPEG_RETRY_AFTER_S)
            return busy

        return Response(
            content=audio,
            media_type=SPEECH_MEDIA_TYPES[fmt],
            headers=headers,
        )

    # ------------------------------------------------------------- models ---

    def _models_payload() -> list[dict]:
        # `created` is an integer unixtime in the spec. A fixed, plausible value
        # is used rather than "now", so repeated calls are stable.
        return [
            {"id": STT_MODEL_ID, "object": "model", "created": 1677610602, "owned_by": "tts-stt"},
            {"id": TTS_MODEL_ID, "object": "model", "created": 1677610602, "owned_by": "tts-stt"},
        ]

    @router.get("/models")
    async def list_models():
        """Spec: ListModelsResponse.required == ['object','data'] — no `has_more`."""
        return {"object": "list", "data": _models_payload()}

    @router.get("/models/{model_id}")
    async def retrieve_model(model_id: str):
        for entry in _models_payload():
            if entry["id"] == model_id:
                return entry
        return openai_error(404, f"The model '{model_id}' does not exist.",
                            param="model", code="model_not_found")

    return router
