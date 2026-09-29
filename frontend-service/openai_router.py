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
# and no client here has asked for one. `pcm` is a header strip of the WAV every
# backend already emits, so it costs nothing.
SPEECH_FORMATS = {"mp3", "opus", "aac", "flac", "wav", "pcm"}
SPEECH_FORMATS_SUPPORTED = {"mp3", "wav", "pcm"}
SPEECH_MEDIA_TYPES = {"mp3": "audio/mpeg", "wav": "audio/wav", "pcm": "audio/pcm"}

# Spec: CreateSpeechRequest.input has maxLength 4096.
MAX_SPEECH_INPUT = 4096

# Ceiling for one ffmpeg transcode. Well above any realistic synthesis length.
FFMPEG_TIMEOUT_S = 120.0

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


async def _wav_to_mp3(wav_bytes: bytes) -> Optional[bytes]:
    """Transcode WAV to MP3 with ffmpeg. Returns None if ffmpeg is unavailable.

    mp3 is the spec's DEFAULT response_format, so a client that sends no format
    at all expects it. Every TTS backend here emits WAV, so the container
    conversion belongs at the gateway rather than in each service.

    Asynchronous on purpose: a blocking `subprocess.run` inside an `async def`
    stalls the whole worker for the length of the encode — health checks and the
    `/ws/stt` relay included — and every mp3 request is one, because mp3 is the
    default format.
    """
    try:
        proc = await asyncio.create_subprocess_exec(
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-f", "wav", "-i", "pipe:0", "-f", "mp3", "-b:a", "64k", "pipe:1",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError:
        logger.warning("ffmpeg not installed; cannot serve response_format=mp3")
        return None
    except OSError as e:
        logger.warning("ffmpeg could not be started: %s", e)
        return None

    try:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(wav_bytes), timeout=FFMPEG_TIMEOUT_S)
    except asyncio.TimeoutError:
        logger.warning("ffmpeg mp3 transcode timed out after %gs", FFMPEG_TIMEOUT_S)
        return None
    except Exception as e:
        logger.warning("ffmpeg mp3 transcode error: %s", e)
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
    logger.warning("ffmpeg mp3 transcode failed (rc=%s): %s",
                   proc.returncode, stderr[:200].decode("utf-8", "replace"))
    return None


def wav_to_pcm(wav_bytes: bytes) -> tuple[bytes, int]:
    """Strip a 16-bit PCM WAV down to raw little-endian mono samples.

    Returns ``(pcm, sample_rate)``. The rate is the backend's own — nothing is
    resampled, so a caller that needs OpenAI's 24 kHz must read it from the
    response. Raises ValueError for anything that is not 16-bit PCM.
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
        # English when the field is absent, which silently mis-transcribes.
        lang = (language or "").strip()
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
                # faster-whisper rejects the literal "auto"; absence means detect.
                if lang and lang.lower() != "auto":
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
                return openai_error(502, f"Transcription backend failed: {exc.detail}")
            logger.warning(
                "default STT provider '%s' is unreachable (%s); falling back to '%s'",
                provider_id, exc.detail, alternative)
            served_by = alternative
            try:
                upstream = await transcribe_with(alternative, get_provider(alternative, kind="stt"))
            except HTTPException as fallback_exc:
                return openai_error(502, f"Transcription backend failed: {fallback_exc.detail}")
        except HTTPException as exc:
            return openai_error(502, f"Transcription backend failed: {exc.detail}")

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

        # `voice` is never validated against a list. The spec's own prose and its
        # VoiceIdsShared enum disagree, and the schema accepts any string — so an
        # unknown voice falls back to the deployment default rather than 404.
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
            upstream = await post_json(provider_id, "/tts", payload,
                                       timeout=max(read_timeout, 600.0))
        except HTTPException as exc:
            return openai_error(502, f"Speech backend failed: {exc.detail}")

        audio = upstream.content
        headers = {"X-Provider": provider_id}
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
                audio, rate = await asyncio.to_thread(wav_to_pcm, audio)
            except ValueError as e:
                return openai_error(502, f"Speech backend returned audio that cannot be sent as pcm: {e}")
            # Raw samples carry no header, so the rate has to travel out of band.
            headers["X-Sample-Rate"] = str(rate)

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
