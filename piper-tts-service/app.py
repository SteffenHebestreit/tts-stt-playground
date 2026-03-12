import os
import uuid
import subprocess
import tempfile
import json
import asyncio
import logging
from pathlib import Path
import shutil
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

import numpy as np
import soundfile as sf

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import aiofiles
import librosa
import io

app = FastAPI(title="PiperTTS Service", description="Text-to-Speech using Piper with custom and default models")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TTSRequest(BaseModel):
    text: str
    voice: Optional[str] = None
    language: str = "en_US"
    quality: str = "medium"  # x_low, low, medium, high
    gender: Optional[str] = None  # male, female
    speed: float = 1.0
    output_format: str = "wav"

class VoiceInfo(BaseModel):
    name: str
    language: str
    speaker: str
    quality: str
    sample_rate: int
    gender: Optional[str] = None
    model_type: str = "default"  # "default" or "custom"

class VoiceCloneRequest(BaseModel):
    text: str
    voice_name: str
    reference_audio: Optional[str] = None

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

# Custom voices (loaded from trained models)
CUSTOM_VOICES = {}

@app.on_event("startup")
async def startup():
    """Load custom voices on startup"""
    await load_custom_voices()

@app.get("/")
async def root():
    """Return service identity and status information."""
    return {"service": "PiperTTS Service", "status": "ready", "version": "1.0.0"}

@app.get("/health")
async def health():
    """Return service health check status."""
    return {"status": "healthy"}

@app.get("/voices")
async def list_voices():
    """List all available voices (default + custom)"""
    all_voices = {**DEFAULT_VOICES, **CUSTOM_VOICES}
    
    # Group by language
    voices_by_language = {}
    for voice_name, voice_info in all_voices.items():
        lang = voice_info.language
        if lang not in voices_by_language:
            voices_by_language[lang] = []
        voices_by_language[lang].append(voice_info.dict())
    
    return {
        "voices": all_voices,
        "voices_by_language": voices_by_language,
        "total": len(all_voices),
        "default_count": len(DEFAULT_VOICES),
        "custom_count": len(CUSTOM_VOICES),
        "supported_languages": list(voices_by_language.keys())
    }

def select_best_voice(language: str, quality: str, gender: Optional[str] = None) -> str:
    """Select the best matching voice based on criteria"""
    all_voices = {**DEFAULT_VOICES, **CUSTOM_VOICES}
    
    # Filter by language first
    matching_voices = []
    for voice_name, voice_info in all_voices.items():
        if voice_info.language.startswith(language.split('_')[0]):  # Match base language
            matching_voices.append((voice_name, voice_info))
    
    if not matching_voices:
        # Fallback to English if no matching language
        for voice_name, voice_info in all_voices.items():
            if voice_info.language.startswith('en'):
                matching_voices.append((voice_name, voice_info))
    
    if not matching_voices:
        # Ultimate fallback
        return "en_US-lessac-medium"
    
    # Filter by quality
    quality_order = ["x_low", "low", "medium", "high"]
    preferred_voices = []
    for voice_name, voice_info in matching_voices:
        if voice_info.quality == quality:
            preferred_voices.append((voice_name, voice_info))
    
    if not preferred_voices:
        preferred_voices = matching_voices
    
    # Filter by gender if specified
    if gender:
        gender_voices = []
        for voice_name, voice_info in preferred_voices:
            if hasattr(voice_info, 'gender') and voice_info.gender == gender:
                gender_voices.append((voice_name, voice_info))
        if gender_voices:
            preferred_voices = gender_voices
    
    # Return the first match
    return preferred_voices[0][0]

async def analyze_audio_with_ffmpeg(file_path: str) -> Dict:
    """Analyze audio file using ffmpeg to extract metadata"""
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
        
        stdout, stderr = await process.communicate()
        
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
    """Analyze uploaded audio file"""
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
        
        # Clean up
        os.unlink(temp_path)
        
        return analysis
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Audio analysis failed: {str(e)}")

def _custom_onnx_infer(model_path: str, text: str, voice_name: str) -> bytes:
    """Run direct ONNX inference for custom-trained VITS models.

    Custom models were trained with a character-level IPA phoneme vocab
    via espeak, not Piper's espeak phoneme_id_map format — so the Piper
    binary can't be used. We phonemize here using the same settings as
    training (phonemizer + espeak backend), then look up IDs.
    """
    import onnxruntime as ort
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
        vocab_path = Path(f"/app/models/custom/{voice_name}/phoneme_vocab.json")
        if vocab_path.exists():
            phoneme_to_id = json.loads(vocab_path.read_text())

    if not phoneme_to_id:
        raise RuntimeError(
            f"No phoneme_id_map found for voice '{voice_name}'. "
            "Cannot perform inference without the phoneme vocabulary."
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

    # Run ONNX inference
    sess_options = ort.SessionOptions()
    sess_options.inter_op_num_threads = 2
    sess_options.intra_op_num_threads = 2
    session = ort.InferenceSession(model_path, sess_options=sess_options)

    outputs = session.run(
        None,
        {"text": text_tensor, "text_lengths": length_tensor},
    )
    audio = outputs[0]  # shape: (batch, time) or (time,)
    if audio.ndim > 1:
        audio = audio[0]
    audio = audio.astype(np.float32)

    # Write to WAV buffer
    buf = io.BytesIO()
    sf.write(buf, audio, sample_rate, format="WAV")
    buf.seek(0)
    return buf.read()


@app.post("/tts")
async def text_to_speech(request: TTSRequest):
    """Generate speech using Piper TTS"""
    try:
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
        
        # Determine model path
        if voice_info.model_type == "default":
            model_path = f"/app/models/default/{request.voice}.onnx"
        else:
            model_path = f"/app/models/custom/{request.voice}/{request.voice}.onnx"
        
        if not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail=f"Model file not found for voice '{request.voice}'")

        # Custom VITS models use a character-level IPA vocab — route to direct ONNX inference
        if voice_info.model_type == "custom":
            wav_bytes = await asyncio.to_thread(
                _custom_onnx_infer, model_path, request.text, request.voice
            )
            return StreamingResponse(
                io.BytesIO(wav_bytes),
                media_type="audio/wav",
                headers={
                    "X-Voice-Used": request.voice,
                    "X-Language": voice_info.language,
                    "X-Quality": voice_info.quality,
                },
            )

        # Standard Piper binary for default models
        output_filename = f"{uuid.uuid4()}.{request.output_format}"
        output_path = f"/app/output/{output_filename}"

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

        stdout, stderr = await process.communicate(input=request.text.encode())

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
                "X-Quality": voice_info.quality
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/synthesize")
async def synthesize_with_custom_voice(request: VoiceCloneRequest):
    """Synthesize speech using custom trained voice"""
    try:
        # Check if custom voice exists
        if request.voice_name not in CUSTOM_VOICES:
            raise HTTPException(status_code=404, detail=f"Custom voice '{request.voice_name}' not found")
        
        voice_info = CUSTOM_VOICES[request.voice_name]
        model_path = f"/app/models/custom/{request.voice_name}/{request.voice_name}.onnx"
        
        if not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail=f"Custom model file not found for '{request.voice_name}'")
        
        # Generate unique filename
        output_filename = f"{uuid.uuid4()}.wav"
        output_path = f"/app/output/{output_filename}"
        
        # Run Piper TTS with custom model
        cmd = [
            "piper",
            "--model", model_path,
            "--output_file", output_path
        ]
        
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate(input=request.text.encode())
        
        if process.returncode != 0:
            error_msg = stderr.decode() if stderr else "Unknown error"
            raise HTTPException(status_code=500, detail=f"Custom TTS generation failed: {error_msg}")
        
        return FileResponse(
            path=output_path,
            filename=output_filename,
            media_type="audio/wav",
            headers={"X-Custom-Voice": request.voice_name}
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload_model")
async def upload_custom_model(
    model_file: UploadFile = File(...),
    config_file: Optional[UploadFile] = File(None),
    voice_name: str = Form(...),
    model_name: str = Form(None)  # Allow model_name as alternative to voice_name
):
    """Upload a custom trained model"""
    try:
        # Use model_name if provided, otherwise use voice_name
        final_voice_name = model_name or voice_name
        
        # Create directory for custom voice
        voice_dir = Path(f"/app/models/custom/{final_voice_name}")
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
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model upload failed: {str(e)}")

@app.delete("/voice/{voice_name}")
async def delete_custom_voice(voice_name: str):
    """Delete a custom voice"""
    try:
        if voice_name not in CUSTOM_VOICES:
            raise HTTPException(status_code=404, detail=f"Custom voice '{voice_name}' not found")
        
        # Remove files
        voice_dir = Path(f"/app/models/custom/{voice_name}")
        if voice_dir.exists():
            shutil.rmtree(voice_dir)
        
        # Remove from custom voices
        del CUSTOM_VOICES[voice_name]
        
        return {"status": "success", "message": f"Custom voice '{voice_name}' deleted"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/voice/{voice_name}")
async def get_voice_info(voice_name: str):
    """Get information about a specific voice"""
    all_voices = {**DEFAULT_VOICES, **CUSTOM_VOICES}
    if voice_name not in all_voices:
        raise HTTPException(status_code=404, detail=f"Voice '{voice_name}' not found")
    
    return all_voices[voice_name]

async def load_custom_voices():
    """Load custom voices from the models directory"""
    custom_models_dir = Path("/app/models/custom")
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
    """Refresh the list of custom voices"""
    CUSTOM_VOICES.clear()
    await load_custom_voices()
    
    return {
        "status": "success",
        "message": "Voice list refreshed",
        "custom_voices": len(CUSTOM_VOICES)
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)
