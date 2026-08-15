"""PiperTTS service for text-to-speech synthesis.

Supports 40+ pre-trained voices across multiple languages and custom
ONNX models trained via the Piper Training service. Default voices use
the Piper binary; custom VITS models use direct ONNX Runtime inference.
"""

import os
import time
import uuid
import tempfile
import json
import asyncio
import logging
import threading
from collections import OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path
import shutil
from typing import Dict, Optional

logger = logging.getLogger(__name__)

import numpy as np
import soundfile as sf

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.background import BackgroundTask
from pydantic import BaseModel, Field
import uvicorn
import aiofiles
import librosa
import io

from naming import (
    sanitize_voice_name,
    language_matches,
    select_best_voice as _select_best_voice,
    prune_old_outputs,
    normalize_phoneme_id_map,
)

@asynccontextmanager
async def _lifespan(_app: FastAPI):
    """Scan the custom-models directory and register any valid voices on startup."""
    await load_custom_voices()
    pruner = asyncio.create_task(_prune_loop())
    try:
        yield
    finally:
        pruner.cancel()


async def _prune_loop():
    """Sweep expired outputs on a timer instead of on the request path.

    The sweep is O(files written in the retention window) and used to run
    synchronously inside /tts — so it got slower the more the service was used,
    and it ran on the event loop.
    """
    while True:
        try:
            await asyncio.to_thread(_prune_old_outputs)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning(f"Output prune failed: {e}")
        await asyncio.sleep(PRUNE_INTERVAL_S)


app = FastAPI(
    title="PiperTTS Service",
    description="Text-to-Speech using Piper with custom and default models",
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

# Where voice models live. compose has always set PIPER_DATA_DIR next to
# PIPER_OUTPUT_DIR, but only the output half was ever read — every model path
# below was the literal "/app/models", so pointing the data dir somewhere else
# silently changed nothing and the service kept reading the old location.
MODELS_DIR = Path(os.getenv("PIPER_DATA_DIR", "/app/models"))
DEFAULT_MODELS_DIR = MODELS_DIR / "default"
CUSTOM_MODELS_DIR = MODELS_DIR / "custom"

# Generated default-voice WAVs land in OUTPUT_DIR and would otherwise accumulate
# indefinitely. Files older than the retention window are pruned best-effort.
OUTPUT_DIR = os.getenv("PIPER_OUTPUT_DIR", "/app/output")
OUTPUT_RETENTION_HOURS = float(os.getenv("OUTPUT_RETENTION_HOURS", "24"))
PRUNE_INTERVAL_S = float(os.getenv("OUTPUT_PRUNE_INTERVAL_S", "3600"))
# When a requested language has no voice, select_best_voice() substitutes an
# English one. Default keeps that (non-breaking) but reports it; set true to
# return 400 instead, which is usually what an API consumer wants.
STRICT_LANGUAGE = os.getenv("PIPER_STRICT_LANGUAGE", "false").strip().lower() in {"1", "true", "yes", "on"}


def _prune_old_outputs() -> None:
    """Best-effort removal of generated WAVs older than the retention window."""
    prune_old_outputs(OUTPUT_DIR, OUTPUT_RETENTION_HOURS)


def _sanitize_voice_name(name: str) -> str:
    """Return *name* if it contains only safe characters, otherwise raise."""
    return sanitize_voice_name(name)

class TTSRequest(BaseModel):
    """Request body for standard Piper text-to-speech synthesis."""

    text: str
    voice: Optional[str] = None
    language: str = "en_US"
    quality: str = "medium"  # x_low, low, medium, high
    gender: Optional[str] = None  # male, female
    # gt=0 guards the `1.0 / speed` length-scale conversion, which turned
    # speed=0 into a ZeroDivisionError -> 500, and passed negatives to the binary.
    speed: float = Field(1.0, gt=0.0, le=4.0)
    output_format: str = "wav"

class VoiceInfo(BaseModel):
    """Metadata describing a built-in or custom voice model."""

    name: str
    language: str
    speaker: str
    quality: str
    sample_rate: int
    gender: Optional[str] = None
    model_type: str = "default"  # "default" or "custom"

class VoiceCloneRequest(BaseModel):
    """Request body for synthesis with a named custom voice."""

    text: str
    voice_name: str
    reference_audio: Optional[str] = None
    # gt=0 guards the `1.0 / speed` length-scale conversion, which turned
    # speed=0 into a ZeroDivisionError -> 500, and passed negatives to the binary.
    speed: float = Field(1.0, gt=0.0, le=4.0)

# Available default voices by language
DEFAULT_VOICES = {
    # English voices
    "en_US-lessac-medium": VoiceInfo(
        name="en_US-lessac-medium",
        language="en_US",
        speaker="lessac",
        quality="medium",
        sample_rate=22050,
        model_type="default"
    ),
    "en_US-amy-medium": VoiceInfo(
        name="en_US-amy-medium",
        language="en_US",
        speaker="amy",
        quality="medium",
        sample_rate=22050,
        model_type="default"
    ),
    "en_GB-alan-medium": VoiceInfo(
        name="en_GB-alan-medium",
        language="en_GB",
        speaker="alan",
        quality="medium",
        sample_rate=22050,
        model_type="default"
    ),
    # German voices
    "de_DE-thorsten-medium": VoiceInfo(
        name="de_DE-thorsten-medium",
        language="de_DE",
        speaker="thorsten",
        quality="medium",
        sample_rate=22050,
        model_type="default"
    ),
    "de_DE-eva_k-x_low": VoiceInfo(
        name="de_DE-eva_k-x_low",
        language="de_DE",
        speaker="eva_k",
        quality="x_low",
        sample_rate=22050,
        model_type="default"
    ),
    # French voices
    "fr_FR-siwis-medium": VoiceInfo(
        name="fr_FR-siwis-medium",
        language="fr_FR",
        speaker="siwis",
        quality="medium",
        sample_rate=22050,
        model_type="default"
    ),
    # Spanish voices
    "es_ES-mls_9972-low": VoiceInfo(
        name="es_ES-mls_9972-low",
        language="es_ES",
        speaker="mls_9972",
        quality="low",
        sample_rate=22050,
        model_type="default"
    ),
    # Italian voices
    "it_IT-riccardo-x_low": VoiceInfo(
        name="it_IT-riccardo-x_low",
        language="it_IT",
        speaker="riccardo",
        quality="x_low",
        sample_rate=22050,
        model_type="default"
    ),
    # Dutch voices
    "nl_NL-mls_5809-low": VoiceInfo(
        name="nl_NL-mls_5809-low",
        language="nl_NL",
        speaker="mls_5809",
        quality="low",
        sample_rate=22050,
        model_type="default"
    )
}

# Custom voices loaded at startup from CUSTOM_MODELS_DIR
CUSTOM_VOICES: Dict[str, VoiceInfo] = {}

# Cache of loaded ONNX inference sessions keyed by model path. Each entry stores
# the file mtime so a re-trained/re-uploaded model is reloaded automatically.
# Bounded and lock-guarded: sessions are multi-hundred-MB and requests build them
# from a thread pool, so an unbounded unsynchronised dict both grew without limit
# and let concurrent first-requests each construct their own session.
_ONNX_SESSION_CACHE: "OrderedDict[str, tuple]" = OrderedDict()
_ONNX_CACHE_SIZE = max(1, int(os.getenv("ONNX_SESSION_CACHE_SIZE", "4")))
_ONNX_CACHE_LOCK = threading.Lock()
# Fewer threads than cores on purpose: these are short sequences where ORT's
# thread ramp-up costs more than the parallelism returns.
_ONNX_THREADS = max(1, int(os.getenv("ONNX_NUM_THREADS", "2")))


def _get_onnx_session(model_path: str):
    """Return a cached ONNX Runtime session for *model_path*, loading it if needed."""
    import onnxruntime as ort

    mtime = os.path.getmtime(model_path)
    with _ONNX_CACHE_LOCK:
        cached = _ONNX_SESSION_CACHE.get(model_path)
        if cached and cached[0] == mtime:
            _ONNX_SESSION_CACHE.move_to_end(model_path)
            return cached[1]

        sess_options = ort.SessionOptions()
        sess_options.inter_op_num_threads = _ONNX_THREADS
        sess_options.intra_op_num_threads = _ONNX_THREADS
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        session = ort.InferenceSession(model_path, sess_options=sess_options)

        _ONNX_SESSION_CACHE[model_path] = (mtime, session)
        _ONNX_SESSION_CACHE.move_to_end(model_path)
        while len(_ONNX_SESSION_CACHE) > _ONNX_CACHE_SIZE:
            _ONNX_SESSION_CACHE.popitem(last=False)
        return session


@app.get("/")
async def root():
    """Return service identity and readiness status."""
    return {"service": "PiperTTS Service", "status": "ready", "version": "1.0.0"}


@app.get("/health")
async def health():
    """Health-check endpoint used by Docker and monitoring."""
    return {"status": "healthy"}


@app.get("/voices")
async def list_voices():
    """List all available voices grouped by language."""
    all_voices = {**DEFAULT_VOICES, **CUSTOM_VOICES}
    
    # Group by language
    voices_by_language = {}
    for voice_name, voice_info in all_voices.items():
        lang = voice_info.language
        if lang not in voices_by_language:
            voices_by_language[lang] = []
        voices_by_language[lang].append(voice_info.model_dump())
    
    return {
        "voices": all_voices,
        "voices_by_language": voices_by_language,
        "total": len(all_voices),
        "default_count": len(DEFAULT_VOICES),
        "custom_count": len(CUSTOM_VOICES),
        "supported_languages": list(voices_by_language.keys())
    }

def select_best_voice(language: str, quality: str, gender: Optional[str] = None) -> str:
    """Pick the best voice across all registered voices (see naming.select_best_voice)."""
    return _select_best_voice({**DEFAULT_VOICES, **CUSTOM_VOICES}, language, quality, gender)

# ffprobe reads a header; it should answer immediately.
FFPROBE_TIMEOUT_S = 30


async def analyze_audio_with_ffmpeg(file_path: str) -> Dict:
    """Run ``ffprobe`` on *file_path* and return codec/duration/quality metadata."""
    try:
        # Get basic audio info
        cmd = [
            "ffprobe", "-v", "quiet", "-print_format", "json", 
            "-show_format", "-show_streams", file_path
        ]
        
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        # Bounded: ffprobe only reads a header, so it should answer at once. A
        # wedged one would otherwise keep the request and the child process
        # alive for as long as the client is willing to wait.
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=FFPROBE_TIMEOUT_S)
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            return {"error": f"ffprobe timed out after {FFPROBE_TIMEOUT_S}s"}

        if process.returncode != 0:
            return {"error": f"FFmpeg analysis failed: {stderr.decode()}"}
        
        ffprobe_data = json.loads(stdout.decode())
        
        # Extract audio stream info
        audio_stream = None
        for stream in ffprobe_data.get("streams", []):
            if stream.get("codec_type") == "audio":
                audio_stream = stream
                break
        
        if not audio_stream:
            return {"error": "No audio stream found"}
        
        analysis = {
            "duration": float(ffprobe_data.get("format", {}).get("duration", 0)),
            "sample_rate": int(audio_stream.get("sample_rate", 0)),
            "channels": int(audio_stream.get("channels", 0)),
            "codec": audio_stream.get("codec_name"),
            "bit_rate": int(audio_stream.get("bit_rate", 0)) if audio_stream.get("bit_rate") else None,
            "format": ffprobe_data.get("format", {}).get("format_name"),
            "size_bytes": int(ffprobe_data.get("format", {}).get("size", 0))
        }
        
        # Add quality assessment
        if analysis["sample_rate"] >= 22050 and analysis["channels"] >= 1:
            analysis["quality_assessment"] = "good"
        elif analysis["sample_rate"] >= 16000:
            analysis["quality_assessment"] = "acceptable"
        else:
            analysis["quality_assessment"] = "poor"
        
        return analysis
        
    except Exception as e:
        return {"error": f"Audio analysis failed: {str(e)}"}

@app.post("/analyze_audio")
async def analyze_audio(audio_file: UploadFile = File(...)):
    """Upload an audio file and receive codec, duration, and quality metadata."""
    temp_path = None
    try:
        # Save temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            content = await audio_file.read()
            temp_file.write(content)
            temp_path = temp_file.name

        # Analyze with ffmpeg
        analysis = await analyze_audio_with_ffmpeg(temp_path)

        # Add librosa analysis for more details
        try:
            audio_data, sr = librosa.load(temp_path, sr=None)
            analysis["librosa"] = {
                "duration": len(audio_data) / sr,
                "sample_rate": sr,
                "rms_energy": float(librosa.feature.rms(y=audio_data)[0].mean()),
                "zero_crossing_rate": float(librosa.feature.zero_crossing_rate(audio_data)[0].mean()),
                "spectral_centroid": float(librosa.feature.spectral_centroid(y=audio_data, sr=sr)[0].mean())
            }
        except Exception as e:
            analysis["librosa_error"] = str(e)

        return analysis

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Audio analysis failed: {str(e)}")
    finally:
        # Always clean up the temp file, even on failure
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except OSError:
                pass

def _custom_onnx_infer(model_path: str, text: str, voice_name: str, speed: float = 1.0) -> bytes:
    """Run direct ONNX inference for custom-trained VITS models.

    Custom models were trained with a character-level IPA phoneme vocab
    via espeak, not Piper's espeak phoneme_id_map format — so the Piper
    binary can't be used. We phonemize here using the same settings as
    training (phonemizer + espeak backend), then look up IDs.

    The exported graph has no duration/length-scale input, so *speed* is
    applied as a pitch-preserving time-stretch on the generated audio.
    """
    from phonemizer import phonemize

    # Load config (piper-tts scans for {voice}.json, not .onnx.json)
    config_path = Path(model_path).parent / f"{Path(model_path).stem}.json"
    if not config_path.exists():
        config_path = Path(model_path).with_suffix('.onnx.json')

    phoneme_to_id: dict = {}
    sample_rate = 22050
    phonemizer_lang = "de"  # sensible default; overridden from config

    if config_path.exists():
        cfg = json.loads(config_path.read_text())
        phoneme_to_id = cfg.get("phoneme_id_map", {})
        sample_rate = cfg.get("audio", {}).get("sample_rate", 22050)
        phonemizer_lang = cfg.get("phonemizer_language", phonemizer_lang)

    # Fallback: phoneme_vocab.json saved alongside the model
    if not phoneme_to_id:
        vocab_path = CUSTOM_MODELS_DIR / voice_name / "phoneme_vocab.json"
        if vocab_path.exists():
            phoneme_to_id = json.loads(vocab_path.read_text())

    if not phoneme_to_id:
        raise RuntimeError(
            f"No phoneme_id_map found for voice '{voice_name}'. "
            "Cannot perform inference without the phoneme vocabulary."
        )

    # Piper-format configs map each phoneme to a *list* of ids ("_": [0]);
    # training vocabs map to a plain int. Normalize to ints so ID lookup
    # below never feeds lists into the int64 tensor.
    phoneme_to_id = normalize_phoneme_id_map(phoneme_to_id)
    if not phoneme_to_id:
        raise RuntimeError(
            f"phoneme_id_map for voice '{voice_name}' contains no usable integer ids."
        )

    # Phonemize exactly as during training:
    #   phonemize(text, language=lang, backend='espeak', strip=True)
    ipa_text = phonemize(
        text,
        language=phonemizer_lang,
        backend='espeak',
        strip=True,
    )

    # Map each IPA character to its training ID
    pad_id   = phoneme_to_id.get("<pad>", 0)
    start_id = phoneme_to_id.get("<start>", pad_id)
    end_id   = phoneme_to_id.get("<end>", pad_id)
    unk_id   = phoneme_to_id.get("<unk>", pad_id)

    ids = [start_id]
    for ch in ipa_text:
        ids.append(phoneme_to_id.get(ch, unk_id))
    ids.append(end_id)

    text_tensor   = np.array([ids], dtype=np.int64)
    length_tensor = np.array([len(ids)], dtype=np.int64)

    # Run ONNX inference (session is cached across requests)
    session = _get_onnx_session(model_path)

    outputs = session.run(
        None,
        {"text": text_tensor, "text_lengths": length_tensor},
    )
    audio = outputs[0]  # shape: (batch, time) or (time,)
    if audio.ndim > 1:
        audio = audio[0]
    audio = audio.astype(np.float32)

    # Apply playback speed (clamped to the UI slider range). Best-effort:
    # fall back to the unstretched audio if the clip is too short to stretch.
    if speed and abs(speed - 1.0) > 1e-3 and audio.size > 0:
        try:
            rate = float(np.clip(speed, 0.5, 2.0))
            audio = librosa.effects.time_stretch(audio, rate=rate).astype(np.float32)
        except Exception as stretch_err:
            logger.warning(f"Speed adjustment failed for voice '{voice_name}': {stretch_err}")

    # Write to WAV buffer
    buf = io.BytesIO()
    sf.write(buf, audio, sample_rate, format="WAV")
    buf.seek(0)
    return buf.read()


@app.post("/tts")
async def text_to_speech(request: TTSRequest):
    """Generate speech audio from text.

    Selects the best matching voice if none is specified.  Custom VITS models
    are served via ONNX Runtime; default Piper voices use the Piper CLI.
    """
    try:
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="Text must not be empty")

        # Piper (and the custom ONNX path) only produce WAV; reject other
        # formats instead of returning mislabeled audio.
        if request.output_format.lower() != "wav":
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported output_format '{request.output_format}': only 'wav' is supported",
            )

        # Select voice if not specified
        if not request.voice:
            request.voice = select_best_voice(request.language, request.quality, request.gender)
        
        # Check if voice exists
        all_voices = {**DEFAULT_VOICES, **CUSTOM_VOICES}
        if request.voice not in all_voices:
            # Try to find alternative
            request.voice = select_best_voice(request.language, request.quality, request.gender)
            if request.voice not in all_voices:
                raise HTTPException(status_code=404, detail=f"No suitable voice found for language '{request.language}' and quality '{request.quality}'")
        
        voice_info = all_voices[request.voice]

        # select_best_voice() degrades to English (and ultimately to a hardcoded
        # en_US voice) when nothing serves the requested language. That keeps the
        # service working, but on its own it means a client asking for German TTS
        # on a deployment with no German voice gets ENGLISH AUDIO for German text
        # — wrong output, HTTP 200, no signal. Make the substitution explicit.
        lang_fallback = not language_matches(voice_info.language, request.language)
        if lang_fallback:
            logger.warning(
                "No voice for language '%s'; falling back to '%s' (%s). "
                "Set PIPER_STRICT_LANGUAGE=true to fail instead.",
                request.language, request.voice, voice_info.language,
            )
            if STRICT_LANGUAGE:
                available = sorted({v.language for v in all_voices.values()})
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"No voice available for language '{request.language}'. "
                        f"Available languages: {', '.join(available)}"
                    ),
                )

        # Determine model path
        if voice_info.model_type == "default":
            model_path = str(DEFAULT_MODELS_DIR / f"{request.voice}.onnx")
        else:
            model_path = str(CUSTOM_MODELS_DIR / request.voice / f"{request.voice}.onnx")
        
        if not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail=f"Model file not found for voice '{request.voice}'")

        # Custom VITS models use a character-level IPA vocab — route to direct ONNX inference
        if voice_info.model_type == "custom":
            wav_bytes = await asyncio.to_thread(
                _custom_onnx_infer, model_path, request.text, request.voice, request.speed
            )
            return StreamingResponse(
                io.BytesIO(wav_bytes),
                media_type="audio/wav",
                headers={
                    "X-Voice-Used": request.voice,
                    "X-Language": voice_info.language,
                    "X-Language-Requested": request.language or "auto",
                    "X-Language-Fallback": "true" if lang_fallback else "false",
                    "X-Quality": voice_info.quality,
                },
            )

        # Standard Piper binary for default models. Output pruning runs on a
        # background timer (see _prune_loop), not here.
        output_filename = f"{uuid.uuid4()}.{request.output_format}"
        output_path = os.path.join(OUTPUT_DIR, output_filename)

        cmd = [
            "piper",
            "--model", model_path,
            "--output_file", output_path
        ]

        if request.speed != 1.0:
            cmd.extend(["--length_scale", str(1.0 / request.speed)])

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        # Collapse newlines: the piper CLI treats each input line as a separate
        # utterance and writes them over the same --output_file, so multi-line
        # text would return only the last line's audio.
        piper_input = " ".join(request.text.split())
        stdout, stderr = await process.communicate(input=piper_input.encode())

        if process.returncode != 0:
            error_msg = stderr.decode() if stderr else "Unknown error"
            raise HTTPException(status_code=500, detail=f"TTS generation failed: {error_msg}")

        if not os.path.exists(output_path):
            raise HTTPException(status_code=500, detail="TTS output file was not created")

        return FileResponse(
            path=output_path,
            filename=output_filename,
            media_type="audio/wav",
            headers={
                "X-Voice-Used": request.voice,
                "X-Language": voice_info.language,
                "X-Language-Requested": request.language or "auto",
                "X-Language-Fallback": "true" if lang_fallback else "false",
                "X-Quality": voice_info.quality
            },
            # Nothing else serves OUTPUT_DIR (there is no StaticFiles mount), so
            # the file is dead the moment it has been streamed.
            background=BackgroundTask(os.unlink, output_path),
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/synthesize")
async def synthesize_with_custom_voice(request: VoiceCloneRequest):
    """Synthesize speech with a named custom voice (uses ONNX Runtime)."""
    voice_name = _sanitize_voice_name(request.voice_name)

    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text must not be empty")

    if voice_name not in CUSTOM_VOICES:
        raise HTTPException(status_code=404, detail=f"Custom voice '{voice_name}' not found")

    model_path = str(CUSTOM_MODELS_DIR / voice_name / f"{voice_name}.onnx")
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail=f"Custom model file not found for '{voice_name}'")

    try:
        wav_bytes = await asyncio.to_thread(
            _custom_onnx_infer, model_path, request.text, voice_name, request.speed
        )
        return StreamingResponse(
            io.BytesIO(wav_bytes),
            media_type="audio/wav",
            headers={"X-Custom-Voice": voice_name},
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload_model")
async def upload_custom_model(
    model_file: UploadFile = File(...),
    config_file: Optional[UploadFile] = File(None),
    voice_name: str = Form(...),
    model_name: str = Form(None),
):
    """Upload a custom-trained ONNX model and optional JSON config."""
    try:
        final_voice_name = _sanitize_voice_name(model_name or voice_name)

        voice_dir = CUSTOM_MODELS_DIR / final_voice_name
        voice_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model file
        model_path = voice_dir / f"{final_voice_name}.onnx"
        async with aiofiles.open(model_path, 'wb') as f:
            content = await model_file.read()
            await f.write(content)
        
        # Handle config file
        config_path = voice_dir / f"{final_voice_name}.json"
        if config_file:
            # Save provided config file
            async with aiofiles.open(config_path, 'wb') as f:
                content = await config_file.read()
                await f.write(content)
        else:
            # Generate basic config if none provided
            basic_config = {
                "audio": {
                    "sample_rate": 22050,
                    "quality": "medium"
                },
                "espeak": {
                    "voice": "en-us"
                },
                "inference": {
                    "noise_scale": 0.667,
                    "length_scale": 1,
                    "noise_w": 0.8
                },
                "phoneme_type": "espeak",
                "phoneme_map": {},
                "phoneme_id_map": {
                    "_": [0], "^": [1], "$": [2], " ": [3]
                },
                "model_card": {
                    "language": "en",
                    "speaker": final_voice_name,
                    "dataset": f"Custom training - {final_voice_name}",
                    "license": "Custom"
                }
            }
            
            async with aiofiles.open(config_path, 'w') as f:
                await f.write(json.dumps(basic_config, indent=2))
        
        # Load config to create voice info
        async with aiofiles.open(config_path, 'r') as f:
            config_content = await f.read()
            config = json.loads(config_content)
        
        # Add to custom voices
        CUSTOM_VOICES[final_voice_name] = VoiceInfo(
            name=final_voice_name,
            language=config.get("model_card", {}).get("language", "en"),
            speaker=config.get("model_card", {}).get("speaker", final_voice_name),
            quality=config.get("audio", {}).get("quality", "medium"),
            sample_rate=config.get("audio", {}).get("sample_rate", 22050),
            model_type="custom"
        )
        
        return {
            "status": "success",
            "message": f"Custom voice '{final_voice_name}' uploaded successfully",
            "voice_info": CUSTOM_VOICES[final_voice_name]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model upload failed: {str(e)}")

@app.delete("/voice/{voice_name}")
async def delete_custom_voice(voice_name: str):
    """Delete a custom voice and its model files."""
    try:
        voice_name = _sanitize_voice_name(voice_name)

        if voice_name not in CUSTOM_VOICES:
            raise HTTPException(status_code=404, detail=f"Custom voice '{voice_name}' not found")

        voice_dir = CUSTOM_MODELS_DIR / voice_name
        if voice_dir.exists():
            shutil.rmtree(voice_dir)

        # Drop any cached ONNX session for this voice's model
        _ONNX_SESSION_CACHE.pop(str(voice_dir / f"{voice_name}.onnx"), None)

        # Remove from custom voices
        del CUSTOM_VOICES[voice_name]
        
        return {"status": "success", "message": f"Custom voice '{voice_name}' deleted"}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/voice/{voice_name}")
async def get_voice_info(voice_name: str):
    """Return metadata for a single voice by name."""
    all_voices = {**DEFAULT_VOICES, **CUSTOM_VOICES}
    if voice_name not in all_voices:
        raise HTTPException(status_code=404, detail=f"Voice '{voice_name}' not found")
    
    return all_voices[voice_name]

async def load_custom_voices():
    """Scan ``CUSTOM_MODELS_DIR`` and register each valid voice into ``CUSTOM_VOICES``."""
    custom_models_dir = CUSTOM_MODELS_DIR
    if not custom_models_dir.exists():
        return
    
    for voice_dir in custom_models_dir.iterdir():
        if voice_dir.is_dir():
            voice_name = voice_dir.name
            config_path = voice_dir / f"{voice_name}.json"
            model_path = voice_dir / f"{voice_name}.onnx"
            
            if config_path.exists() and model_path.exists():
                try:
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                    
                    CUSTOM_VOICES[voice_name] = VoiceInfo(
                        name=voice_name,
                        language=config.get("model_card", {}).get("language", "en"),
                        speaker=config.get("model_card", {}).get("speaker", voice_name),
                        quality=config.get("audio", {}).get("quality", "medium"),
                        sample_rate=config.get("audio", {}).get("sample_rate", 22050),
                        model_type="custom"
                    )
                    
                    logger.info(f"Loaded custom voice: {voice_name}")

                except Exception as e:
                    logger.warning(f"Failed to load custom voice {voice_name}: {e}")

@app.post("/refresh_voices")
async def refresh_voices():
    """Re-scan the custom models directory and update the voice registry."""
    CUSTOM_VOICES.clear()
    # Drop cached sessions too — a removed voice would otherwise keep serving
    # from a session whose model file no longer exists.
    with _ONNX_CACHE_LOCK:
        _ONNX_SESSION_CACHE.clear()
    await load_custom_voices()
    
    return {
        "status": "success",
        "message": "Voice list refreshed",
        "custom_voices": len(CUSTOM_VOICES)
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)
