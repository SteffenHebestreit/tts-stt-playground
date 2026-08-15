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

import io
import json
import logging
import subprocess
from typing import Any, Callable, Optional

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse, Response

logger = logging.getLogger(__name__)

# The guide states a 25 MB upload limit. Mirrored here so a large upload is
# refused cheaply rather than being buffered into an SBC's memory.
MAX_UPLOAD_BYTES = 25 * 1024 * 1024

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
STT_FORMATS_SUPPORTED = {"json", "text"}

SPEECH_FORMATS = {"mp3", "opus", "aac", "flac", "wav", "pcm"}
SPEECH_FORMATS_SUPPORTED = {"mp3", "wav"}
SPEECH_MEDIA_TYPES = {"mp3": "audio/mpeg", "wav": "audio/wav"}

# Spec: CreateSpeechRequest.input has maxLength 4096.
MAX_SPEECH_INPUT = 4096

# OpenAI's own voice names. They mean nothing to any backend here, so forwarding
# one makes the provider fail to resolve a voice instead of using its default.
OPENAI_PLACEHOLDER_VOICES = {
    "alloy", "echo", "fable", "onyx", "nova", "shimmer",
    "ash", "ballad", "coral", "sage", "verse", "marin", "cedar",
}


def openai_error(status: int, message: str, *, param: Optional[str] = None,
                 code: Optional[str] = None) -> JSONResponse:
    """Build the OpenAI error envelope.

    Spec: `Error.required == ['type','message','param','code']` and
    `ErrorResponse.required == ['error']`. openai-python reads `code`, `param`
    and `type` off `body.get("error", body)`, so FastAPI's default
    `{"detail": ...}` leaves all three None and clients cannot branch on them.
    """
    if status in (400, 404, 415, 422):
        err_type = "invalid_request_error"
    elif status == 401:
        err_type = "authentication_error"
    elif status == 429:
        err_type = "rate_limit_error"
    else:
        err_type = "server_error"
    return JSONResponse(
        status_code=status,
        content={"error": {"message": message, "type": err_type, "param": param, "code": code}},
    )


def _wav_to_mp3(wav_bytes: bytes) -> Optional[bytes]:
    """Transcode WAV to MP3 with ffmpeg. Returns None if ffmpeg is unavailable.

    mp3 is the spec's DEFAULT response_format, so a client that sends no format
    at all expects it. Every TTS backend here emits WAV, so the container
    conversion belongs at the gateway rather than in each service.
    """
    try:
        proc = subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error",
             "-f", "wav", "-i", "pipe:0", "-f", "mp3", "-b:a", "64k", "pipe:1"],
            input=wav_bytes, capture_output=True, timeout=120,
        )
        if proc.returncode == 0 and proc.stdout:
            return proc.stdout
        logger.warning("ffmpeg mp3 transcode failed (rc=%s): %s",
                       proc.returncode, proc.stderr[:200].decode("utf-8", "replace"))
    except FileNotFoundError:
        logger.warning("ffmpeg not installed; cannot serve response_format=mp3")
    except Exception as e:
        logger.warning("ffmpeg mp3 transcode error: %s", e)
    return None


def build_router(
    *,
    get_provider: Callable[..., dict],
    registry: dict,
    post_form: Callable,
    post_json: Callable,
    provider_health: Callable,
    build_tts_payload: Callable,
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

        content = await file.read()
        if len(content) > MAX_UPLOAD_BYTES:
            return openai_error(
                400,
                f"File is too large. Maximum is {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.",
                param="file", code="file_too_large")
        if not content:
            return openai_error(400, "Audio file is empty.", param="file")

        provider_id = _default_stt_provider()
        try:
            provider = get_provider(provider_id, kind="stt")
        except HTTPException as exc:
            return openai_error(503, f"No speech-to-text provider available: {exc.detail}")

        contract = (provider.get("contracts") or {}).get("transcribe")
        data: dict[str, Any] = {}
        # "auto" must be sent explicitly, not omitted: whisper.cpp defaults to
        # English when the field is absent, which silently mis-transcribes.
        lang = (language or "").strip()
        if contract == "openai-audio-transcriptions-v1":
            files = [("file", (file.filename or "audio.wav", content,
                               file.content_type or "application/octet-stream"))]
            data["response_format"] = "json"
            data["language"] = lang or "auto"
            path = provider.get("transcribe_path") or "/v1/audio/transcriptions"
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

        try:
            upstream = await post_form(provider_id, path, data=data, files=files, timeout=600.0)
        except HTTPException as exc:
            return openai_error(502, f"Transcription backend failed: {exc.detail}")

        try:
            payload = upstream.json()
        except Exception:
            return openai_error(502, "Transcription backend returned a non-JSON response.")

        text = (payload.get("text") or "").strip()
        if fmt == "text":
            # openai-python returns response.text verbatim for this format, so
            # the body must be raw — a JSON-quoted string would reach the caller
            # with literal quote characters.
            return PlainTextResponse(text, media_type="text/plain; charset=utf-8")
        return JSONResponse({"text": text})

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
        if fmt == "mp3":
            converted = _wav_to_mp3(audio)
            if converted is None:
                return openai_error(
                    501,
                    "response_format 'mp3' requires ffmpeg, which is not available "
                    "in this deployment. Use response_format='wav'.",
                    param="response_format", code="unsupported_value")
            audio = converted

        return Response(
            content=audio,
            media_type=SPEECH_MEDIA_TYPES[fmt],
            headers={"X-Provider": provider_id},
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
