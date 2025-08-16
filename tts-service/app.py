from fastapi import FastAPI, UploadFile, HTTPException, Response, Form
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import subprocess
import io
import os
import sys
import logging
import tempfile
import json
import base64
import asyncio
import zipfile
import tarfile
import requests
import httpx
from pathlib import Path
import shutil
import numpy as np
import soundfile as sf

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Piper TTS Service", description="Fast Neural Text-to-Speech Service using Piper")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Piper configuration
PIPER_EXECUTABLE = "piper"
MODELS_DIR = Path("/app/models")  # Docker path for models
LOCAL_MODELS_DIR = Path("models")  # Local development path

# Voice cloning service configuration
VOICE_CLONING_SERVICE_URL = "http://voice-cloning-service:5001"

# Language to model mapping for Piper (high-quality models prioritized for German)
LANGUAGE_MODELS = {
    "en": {"name": "en_US-lessac-medium", "path": "en/en_US/lessac/medium"},
    "en-us": {"name": "en_US-lessac-medium", "path": "en/en_US/lessac/medium"}, 
    "en-gb": {"name": "en_GB-cori-medium", "path": "en/en_GB/cori/medium"},
    "de": {"name": "de_DE-thorsten-medium", "path": "de/de_DE/thorsten/medium"},  # High-quality German model
    "fr": {"name": "fr_FR-siwis-medium", "path": "fr/fr_FR/siwis/medium"}, 
    "es": {"name": "es_ES-sharvard-medium", "path": "es/es_ES/sharvard/medium"},
    "it": {"name": "it_IT-riccardo-x_low", "path": "it/it_IT/riccardo/x_low"},
    "pt": {"name": "pt_BR-faber-medium", "path": "pt/pt_BR/faber/medium"},
    "ru": {"name": "ru_RU-ruslan-medium", "path": "ru/ru_RU/ruslan/medium"},
    "pl": {"name": "pl_PL-mc_speech-medium", "path": "pl/pl_PL/mc_speech/medium"},
    "nl": {"name": "nl_NL-mls_5809-low", "path": "nl/nl_NL/mls_5809/low"},
    "cs": {"name": "cs_CZ-jirka-medium", "path": "cs/cs_CZ/jirka/medium"},
    "ar": {"name": "ar_JO-kareem-medium", "path": "ar/ar_JO/kareem/medium"},
    "zh": {"name": "zh_CN-huayan-medium", "path": "zh/zh_CN/huayan/medium"},
    "ja": {"name": "ja_JP-kaho-medium", "path": "ja/ja_JP/kaho/medium"},
    "ko": {"name": "ko_KR-kss-medium", "path": "ko/ko_KR/kss/medium"},
    "hi": {"name": "hi_IN-male-medium", "path": "hi/hi_IN/male/medium"},
    "tr": {"name": "tr_TR-dfki-medium", "path": "tr/tr_TR/dfki/medium"},
    "hu": {"name": "hu_HU-anna-medium", "path": "hu/hu_HU/anna/medium"}
}

# Model download URLs (Hugging Face model repository)
MODEL_BASE_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0"

# Default models to download on startup (prioritizing German and common languages)
DEFAULT_MODELS = [
    "en_US-lessac-medium",  # English (US) - high quality 
    "de_DE-thorsten-medium",  # German - high quality for German users
]

# Global variables
models_downloaded = set()
piper_available = False

def check_piper_installation():
    """Check if Piper is installed and available"""
    global piper_available
    try:
        result = subprocess.run([PIPER_EXECUTABLE, "--help"], 
                              capture_output=True, text=True, timeout=10)
        piper_available = result.returncode == 0
        if piper_available:
            logger.info("✅ Piper TTS executable found and working")
        else:
            logger.error("❌ Piper TTS executable not working properly")
        return piper_available
    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError) as e:
        logger.error(f"❌ Piper TTS not found: {e}")
        piper_available = False
        return False

def get_models_dir():
    """Get the appropriate models directory"""
    if MODELS_DIR.exists():
        return MODELS_DIR
    else:
        LOCAL_MODELS_DIR.mkdir(exist_ok=True)
        return LOCAL_MODELS_DIR

async def download_model(model_name: str) -> bool:
    """Download a Piper model if not already present"""
    models_dir = get_models_dir()
    model_path = models_dir / f"{model_name}.onnx"
    config_path = models_dir / f"{model_name}.onnx.json"
    
    if model_path.exists() and config_path.exists():
        logger.info(f"Model {model_name} already exists")
        models_downloaded.add(model_name)
        return True
    
    # Find the model config
    model_config = None
    for lang_models in LANGUAGE_MODELS.values():
        if isinstance(lang_models, dict) and lang_models["name"] == model_name:
            model_config = lang_models
            break
    
    if not model_config:
        logger.error(f"Model configuration not found for {model_name}")
        return False
    
    try:
        logger.info(f"📥 Downloading Piper model: {model_name}")
        
        # Build URLs using the correct path structure
        model_url = f"{MODEL_BASE_URL}/{model_config['path']}/{model_name}.onnx"
        config_url = f"{MODEL_BASE_URL}/{model_config['path']}/{model_name}.onnx.json"
        
        logger.info(f"Model URL: {model_url}")
        logger.info(f"Config URL: {config_url}")
        
        # Download model
        model_response = requests.get(model_url, timeout=120)
        model_response.raise_for_status()
        
        with open(model_path, 'wb') as f:
            f.write(model_response.content)
        
        # Download config
        config_response = requests.get(config_url, timeout=30)
        config_response.raise_for_status()
        
        with open(config_path, 'wb') as f:
            f.write(config_response.content)
        
        models_downloaded.add(model_name)
        logger.info(f"✅ Downloaded model: {model_name}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to download model {model_name}: {e}")
        # Clean up partial downloads
        for path in [model_path, config_path]:
            if path.exists():
                path.unlink()
        return False

async def ensure_model(language: str) -> str:
    """Ensure the model for a language is available"""
    model_config = LANGUAGE_MODELS.get(language.lower())
    if not model_config:
        # Try to find a close match
        lang_code = language.split('-')[0].lower()
        model_config = LANGUAGE_MODELS.get(lang_code)
        if not model_config:
            logger.warning(f"Language '{language}' not supported, using English")
            model_config = LANGUAGE_MODELS["en"]
    
    model_name = model_config["name"]
    
    if model_name not in models_downloaded:
        await download_model(model_name)
    
    return model_name

def synthesize_with_piper(text: str, model_name: str, speed: float = 1.0) -> bytes:
    """Synthesize speech using Piper"""
    models_dir = get_models_dir()
    model_path = models_dir / f"{model_name}.onnx"
    
    if not model_path.exists():
        raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
    
    try:
        # Create temporary output file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as output_file:
            output_path = output_file.name
        
        # Prepare Piper command
        cmd = [
            PIPER_EXECUTABLE,
            "--model", str(model_path),
            "--output_file", output_path
        ]
        
        # Add speed control if supported
        if speed != 1.0:
            cmd.extend(["--length_scale", str(1.0 / speed)])  # Piper uses length_scale (inverse of speed)
        
        # Run Piper
        result = subprocess.run(
            cmd,
            input=text,
            text=True,
            capture_output=True,
            timeout=30
        )
        
        if result.returncode != 0:
            logger.error(f"Piper synthesis failed: {result.stderr}")
            raise HTTPException(status_code=500, detail=f"Synthesis failed: {result.stderr}")
        
        # Read the generated audio
        with open(output_path, 'rb') as f:
            audio_data = f.read()
        
        # Clean up
        os.unlink(output_path)
        
        return audio_data
        
    except subprocess.TimeoutExpired:
        logger.error("Piper synthesis timed out")
        raise HTTPException(status_code=500, detail="Synthesis timed out")
    except Exception as e:
        logger.error(f"Synthesis error: {e}")
        raise HTTPException(status_code=500, detail=f"Synthesis failed: {str(e)}")

@app.on_event("startup")
async def startup_event():
    """Initialize Piper and download default models"""
    logger.info("🚀 Starting Piper TTS Service...")
    
    # Check Piper installation
    if not check_piper_installation():
        logger.error("❌ Piper TTS not available - service may not work properly")
        return
    
    # Create models directory
    models_dir = get_models_dir()
    logger.info(f"📁 Models directory: {models_dir}")
    
    # Download default models
    for model_name in DEFAULT_MODELS:
        await download_model(model_name)
    
    logger.info("✅ Piper TTS Service ready!")

@app.post("/synthesize")
async def synthesize_text(
    text: str = Form(...), 
    speaker_wav: UploadFile = None,  # Ignored for Piper (doesn't support voice cloning)
    language: str = Form("en"),
    speed: float = Form(1.0),
    emotion: str = Form("neutral")  # Ignored for Piper
):
    """Synthesize speech from text using Piper TTS"""
    if not piper_available:
        raise HTTPException(status_code=503, detail="Piper TTS not available")
    
    try:
        # Ensure model for language is available
        model_name = await ensure_model(language)
        
        # Synthesize
        audio_data = synthesize_with_piper(text, model_name, speed)
        
        return StreamingResponse(
            io.BytesIO(audio_data),
            media_type="audio/wav",
            headers={
                "Content-Disposition": f"attachment; filename=piper_tts_{language}_{len(text)}_chars.wav",
                "X-Language": language,
                "X-Model": model_name,
                "X-Text-Length": str(len(text)),
                "X-Speed": str(speed)
            }
        )
        
    except Exception as e:
        logger.error(f"Synthesis error: {e}")
        raise HTTPException(status_code=500, detail=f"Synthesis failed: {str(e)}")

@app.post("/synthesize-stream")
async def synthesize_text_stream(
    text: str = Form(...),
    speaker_wav: UploadFile = None,  # Ignored for Piper
    language: str = Form("en"),
    speed: float = Form(1.0),
    chunk_size: int = Form(200)  # Characters per chunk
):
    """Stream TTS synthesis for long texts by processing in chunks"""
    if not piper_available:
        raise HTTPException(status_code=503, detail="Piper TTS not available")
    
    async def generate_audio_chunks():
        """Generator function for streaming TTS synthesis"""
        try:
            # Ensure model for language is available
            model_name = await ensure_model(language)
            
            # Split text into chunks for streaming
            words = text.split()
            chunks = []
            current_chunk = []
            current_length = 0
            
            for word in words:
                if current_length + len(word) + 1 <= chunk_size:
                    current_chunk.append(word)
                    current_length += len(word) + 1
                else:
                    if current_chunk:
                        chunks.append(" ".join(current_chunk))
                    current_chunk = [word]
                    current_length = len(word)
            
            if current_chunk:
                chunks.append(" ".join(current_chunk))
            
            # Send metadata
            metadata = {
                "total_chunks": len(chunks),
                "language": language,
                "model": model_name,
                "speed": speed,
                "text_length": len(text)
            }
            yield f"data: {json.dumps({'metadata': metadata})}\n\n"
            
            # Process each chunk
            for i, chunk in enumerate(chunks):
                try:
                    # Generate speech for chunk
                    audio_data = synthesize_with_piper(chunk, model_name, speed)
                    
                    # Encode audio as base64 for transmission
                    audio_base64 = base64.b64encode(audio_data).decode('utf-8')
                    
                    # Send chunk info and audio data
                    chunk_info = {
                        "chunk_id": i,
                        "text": chunk,
                        "audio_size": len(audio_data),
                        "audio_data": audio_base64,
                        "audio_format": "wav",
                        "status": "chunk_ready"
                    }
                    yield f"data: {json.dumps(chunk_info)}\n\n"
                    
                except Exception as chunk_error:
                    logger.error(f"Error processing chunk {i}: {chunk_error}")
                    error_chunk = {
                        "chunk_id": i,
                        "text": chunk,
                        "error": str(chunk_error),
                        "status": "chunk_error"
                    }
                    yield f"data: {json.dumps(error_chunk)}\n\n"
            
            # Send completion
            completion = {"status": "completed", "total_chunks_processed": len(chunks)}
            yield f"data: {json.dumps(completion)}\n\n"
                
        except Exception as e:
            logger.error(f"Streaming synthesis error: {e}")
            error_data = {"error": f"Synthesis failed: {str(e)}", "status": "error"}
            yield f"data: {json.dumps(error_data)}\n\n"
    
    return StreamingResponse(
        generate_audio_chunks(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
            "Access-Control-Allow-Headers": "*"
        }
    )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "ok",
        "piper_available": piper_available,
        "models_downloaded": len(models_downloaded),
        "available_models": list(models_downloaded)
    }

@app.get("/info")
async def service_info():
    """Service information endpoint"""
    return {
        "service": "Piper TTS Service",
        "engine": "Piper (ONNX Neural TTS)",
        "piper_available": piper_available,
        "models_downloaded": len(models_downloaded),
        "available_models": list(models_downloaded),
        "default_language": "en",
        "models_dir": str(get_models_dir()),
        "features": {
            "fast_synthesis": True,
            "local_processing": True,
            "high_quality_german": True,
            "streaming": True,
            "voice_cloning": False  # Piper doesn't support voice cloning
        }
    }

@app.get("/languages")
async def supported_languages():
    """Get supported languages for Piper TTS"""
    return {
        "supported_languages": [
            {"code": "en", "name": "English (US)", "model": "en_US-lessac-medium"},
            {"code": "en-gb", "name": "English (UK)", "model": "en_GB-cori-medium"},
            {"code": "de", "name": "German", "model": "de_DE-thorsten-medium", "note": "High quality"},
            {"code": "fr", "name": "French", "model": "fr_FR-siwis-medium"},
            {"code": "es", "name": "Spanish", "model": "es_ES-sharvard-medium"},
            {"code": "it", "name": "Italian", "model": "it_IT-riccardo-x_low"},
            {"code": "pt", "name": "Portuguese (Brazil)", "model": "pt_BR-faber-medium"},
            {"code": "ru", "name": "Russian", "model": "ru_RU-ruslan-medium"},
            {"code": "pl", "name": "Polish", "model": "pl_PL-mc_speech-medium"},
            {"code": "nl", "name": "Dutch", "model": "nl_NL-mls_5809-low"},
            {"code": "cs", "name": "Czech", "model": "cs_CZ-jirka-medium"},
            {"code": "ar", "name": "Arabic", "model": "ar_JO-kareem-medium"},
            {"code": "zh", "name": "Chinese", "model": "zh_CN-huayan-medium"},
            {"code": "ja", "name": "Japanese", "model": "ja_JP-kaho-medium"},
            {"code": "ko", "name": "Korean", "model": "ko_KR-kss-medium"},
            {"code": "hi", "name": "Hindi", "model": "hi_IN-male-medium"},
            {"code": "tr", "name": "Turkish", "model": "tr_TR-dfki-medium"},
            {"code": "hu", "name": "Hungarian", "model": "hu_HU-anna-medium"}
        ],
        "default_language": "en",
        "note": "All models are neural ONNX models optimized for quality and speed"
    }

@app.get("/models")
async def model_info():
    """Get information about available TTS models"""
    return {
        "engine": "Piper TTS",
        "model_type": "Neural ONNX",
        "downloaded_models": list(models_downloaded),
        "available_models": [config["name"] for config in LANGUAGE_MODELS.values()],
        "capabilities": [
            "Fast neural synthesis",
            "High-quality German speech",
            "Local processing (no internet required)",
            "Multiple language support",
            "Speed control",
            "Low latency streaming"
        ],
        "supported_features": {
            "voice_cloning": False,
            "multi_language": True,
            "speed_control": True,
            "streaming": True,
            "local_processing": True,
            "high_quality_german": True
        },
        "note": "Piper provides significantly better German speech quality compared to XTTS v2"
    }

@app.post("/download-model")
async def download_model_endpoint(language: str = Form(...)):
    """Download a model for a specific language"""
    if not piper_available:
        raise HTTPException(status_code=503, detail="Piper TTS not available")
    
    model_config = LANGUAGE_MODELS.get(language.lower())
    if not model_config:
        raise HTTPException(status_code=400, detail=f"Language '{language}' not supported")
    
    model_name = model_config["name"]
    success = await download_model(model_name)
    if success:
        return {
            "status": "success",
            "language": language,
            "model": model_name,
            "message": f"Model for {language} downloaded successfully"
        }
    else:
        raise HTTPException(status_code=500, detail=f"Failed to download model for {language}")

# Voice Cloning Integration Endpoints

@app.post("/clone-voice")
async def clone_voice(
    voice_id: str = Form(...),
    name: str = Form(""),
    description: str = Form(""),
    audio_file: UploadFile = None
):
    """Clone a voice using the voice cloning service"""
    try:
        async with httpx.AsyncClient() as client:
            # Prepare form data for voice cloning service
            form_data = {
                "voice_id": voice_id,
                "name": name,
                "description": description
            }
            
            files = {}
            if audio_file:
                files["audio_file"] = (audio_file.filename, await audio_file.read(), audio_file.content_type)
            
            # Call voice cloning service
            response = await client.post(
                f"{VOICE_CLONING_SERVICE_URL}/clone",
                data=form_data,
                files=files,
                timeout=60.0
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Voice cloning failed: {response.text}"
                )
                
    except httpx.RequestError as e:
        logger.error(f"Voice cloning service unavailable: {e}")
        raise HTTPException(
            status_code=503,
            detail="Voice cloning service unavailable"
        )
    except Exception as e:
        logger.error(f"Voice cloning error: {e}")
        raise HTTPException(status_code=500, detail=f"Voice cloning failed: {str(e)}")

@app.post("/synthesize-cloned")
async def synthesize_with_cloned_voice(
    text: str = Form(...),
    voice_id: str = Form(...),
    language: str = Form("en")
):
    """Synthesize speech using a cloned voice"""
    try:
        async with httpx.AsyncClient() as client:
            # Call voice cloning service for synthesis
            response = await client.post(
                f"{VOICE_CLONING_SERVICE_URL}/synthesize",
                json={
                    "voice_id": voice_id,
                    "text": text,
                    "language": language
                },
                timeout=60.0
            )
            
            if response.status_code == 200:
                return StreamingResponse(
                    io.BytesIO(response.content),
                    media_type="audio/wav",
                    headers={
                        "Content-Disposition": f"attachment; filename=cloned_voice_{voice_id}_{len(text)}_chars.wav",
                        "X-Voice-ID": voice_id,
                        "X-Language": language,
                        "X-Text-Length": str(len(text)),
                        "X-Engine": "OpenVoice"
                    }
                )
            else:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Cloned voice synthesis failed: {response.text}"
                )
                
    except httpx.RequestError as e:
        logger.error(f"Voice cloning service unavailable: {e}")
        raise HTTPException(
            status_code=503,
            detail="Voice cloning service unavailable"
        )
    except Exception as e:
        logger.error(f"Cloned voice synthesis error: {e}")
        raise HTTPException(status_code=500, detail=f"Synthesis failed: {str(e)}")

@app.get("/cloned-voices")
async def list_cloned_voices():
    """List all available cloned voices"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{VOICE_CLONING_SERVICE_URL}/voices",
                timeout=30.0
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Failed to fetch voices: {response.text}"
                )
                
    except httpx.RequestError as e:
        logger.error(f"Voice cloning service unavailable: {e}")
        raise HTTPException(
            status_code=503,
            detail="Voice cloning service unavailable"
        )
    except Exception as e:
        logger.error(f"Failed to list voices: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list voices: {str(e)}")

@app.delete("/cloned-voices/{voice_id}")
async def delete_cloned_voice(voice_id: str):
    """Delete a cloned voice"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.delete(
                f"{VOICE_CLONING_SERVICE_URL}/voices/{voice_id}",
                timeout=30.0
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Failed to delete voice: {response.text}"
                )
                
    except httpx.RequestError as e:
        logger.error(f"Voice cloning service unavailable: {e}")
        raise HTTPException(
            status_code=503,
            detail="Voice cloning service unavailable"
        )
    except Exception as e:
        logger.error(f"Failed to delete voice: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete voice: {str(e)}")

@app.get("/cloned-voices/{voice_id}")
async def get_cloned_voice_info(voice_id: str):
    """Get information about a specific cloned voice"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{VOICE_CLONING_SERVICE_URL}/voices/{voice_id}",
                timeout=30.0
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Voice not found: {response.text}"
                )
                
    except httpx.RequestError as e:
        logger.error(f"Voice cloning service unavailable: {e}")
        raise HTTPException(
            status_code=503,
            detail="Voice cloning service unavailable"
        )
    except Exception as e:
        logger.error(f"Failed to get voice info: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get voice info: {str(e)}")

@app.get("/cloned-voices/{voice_id}/sample")
async def get_cloned_voice_sample(voice_id: str):
    """Get the reference audio sample for a cloned voice"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{VOICE_CLONING_SERVICE_URL}/voices/{voice_id}/sample",
                timeout=30.0
            )
            
            if response.status_code == 200:
                return StreamingResponse(
                    io.BytesIO(response.content),
                    media_type="audio/wav",
                    headers={
                        "Content-Disposition": f"attachment; filename=sample_{voice_id}.wav"
                    }
                )
            else:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Sample not found: {response.text}"
                )
                
    except httpx.RequestError as e:
        logger.error(f"Voice cloning service unavailable: {e}")
        raise HTTPException(
            status_code=503,
            detail="Voice cloning service unavailable"
        )
    except Exception as e:
        logger.error(f"Failed to get voice sample: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get voice sample: {str(e)}")

@app.get("/voice-cloning-status")
async def get_voice_cloning_status():
    """Check if voice cloning service is available"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{VOICE_CLONING_SERVICE_URL}/health",
                timeout=10.0
            )
            
            if response.status_code == 200:
                cloning_status = response.json()
                return {
                    "voice_cloning_available": True,
                    "service_status": cloning_status
                }
            else:
                return {
                    "voice_cloning_available": False,
                    "error": "Service responded with error"
                }
                
    except httpx.RequestError as e:
        return {
            "voice_cloning_available": False,
            "error": f"Service unavailable: {str(e)}"
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
