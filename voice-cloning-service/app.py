import os
import logging
import tempfile
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import uvicorn
import torch
import torchaudio
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Voice Cloning Service", version="1.0.0")

# Configuration
VOICES_DIR = Path("/app/cloned_voices")
MODELS_DIR = Path("/app/models")
TEMP_DIR = Path("/app/temp")

# Ensure directories exist
VOICES_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR.mkdir(parents=True, exist_ok=True)

class VoiceCloneRequest(BaseModel):
    voice_id: str
    text: str
    language: str = "en"

class VoiceInfo(BaseModel):
    voice_id: str
    name: str
    description: str
    created_at: str
    file_path: str
    sample_duration: float

class VoiceCloneService:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        # Initialize OpenVoice models/metadata
        self.base_speaker_tts = None
        self.tone_color_converter = None
        self.voices_metadata = {}
        self.openvoice_available = False

        self._initialize_openvoice()
        self._load_voices_metadata()
    
    def _initialize_openvoice(self):
        """Initialize OpenVoice models"""
        try:
            # Try to import OpenVoice components
            from openvoice.api import BaseSpeakerTTS, ToneColorConverter
            import os
            
            # Store references for later use
            self.BaseSpeakerTTS = BaseSpeakerTTS
            self.ToneColorConverter = ToneColorConverter
            self.openvoice_available = True
            
            logger.info("Initializing OpenVoice models...")
            
            # Prefer V1 checkpoints for stability; fallback to V2 if available
            ckpt_root_v1 = MODELS_DIR / 'checkpoints'
            ckpt_root_v2 = MODELS_DIR / 'checkpoints_v2'

            config_v1 = ckpt_root_v1 / 'base_speakers' / 'EN' / 'config.json'
            ckpt_v1 = ckpt_root_v1 / 'base_speakers' / 'EN' / 'checkpoint.pth'
            conv_cfg_v1 = ckpt_root_v1 / 'converter' / 'config.json'
            conv_ckpt_v1 = ckpt_root_v1 / 'converter' / 'checkpoint.pth'

            config_v2 = ckpt_root_v2 / 'base_speakers' / 'EN' / 'config.json'
            ckpt_v2 = ckpt_root_v2 / 'base_speakers' / 'EN' / 'checkpoint.pth'
            conv_cfg_v2 = ckpt_root_v2 / 'converter' / 'config.json'
            conv_ckpt_v2 = ckpt_root_v2 / 'converter' / 'checkpoint.pth'

            if config_v1.exists() and ckpt_v1.exists() and conv_cfg_v1.exists() and conv_ckpt_v1.exists():
                logger.info("Loading OpenVoice V1 checkpoints")
                self.base_speaker_tts = self.BaseSpeakerTTS(str(config_v1), device=self.device)
                self.base_speaker_tts.load_ckpt(str(ckpt_v1))
                self.tone_color_converter = self.ToneColorConverter(str(conv_cfg_v1), device=self.device)
                self.tone_color_converter.load_ckpt(str(conv_ckpt_v1))
            elif config_v2.exists() and ckpt_v2.exists() and conv_cfg_v2.exists() and conv_ckpt_v2.exists():
                logger.info("Loading OpenVoice V2 checkpoints")
                self.base_speaker_tts = self.BaseSpeakerTTS(str(config_v2), device=self.device)
                self.base_speaker_tts.load_ckpt(str(ckpt_v2))
                self.tone_color_converter = self.ToneColorConverter(str(conv_cfg_v2), device=self.device)
                self.tone_color_converter.load_ckpt(str(conv_ckpt_v2))
            else:
                logger.warning("OpenVoice checkpoints not found. Run download_models.py or mount models.")
                self.openvoice_available = False
                return
            
            logger.info("OpenVoice models initialized successfully")
            
        except ImportError as e:
            logger.warning(f"OpenVoice not available: {e}")
            logger.info("Voice cloning service will run in fallback mode")
            self.openvoice_available = False
        except Exception as e:
            logger.error(f"Failed to initialize OpenVoice: {e}")
            logger.error("Voice cloning functionality will be limited")
            self.openvoice_available = False
    
    def _load_voices_metadata(self):
        """Load metadata for all cloned voices"""
        metadata_file = VOICES_DIR / "voices_metadata.json"
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    self.voices_metadata = json.load(f)
                logger.info(f"Loaded metadata for {len(self.voices_metadata)} voices")
            except Exception as e:
                logger.error(f"Failed to load voices metadata: {e}")
                self.voices_metadata = {}
        else:
            self.voices_metadata = {}
    
    def _save_voices_metadata(self):
        """Save metadata for all cloned voices"""
        metadata_file = VOICES_DIR / "voices_metadata.json"
        try:
            with open(metadata_file, 'w') as f:
                json.dump(self.voices_metadata, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save voices metadata: {e}")
    
    def clone_voice(self, voice_id: str, audio_file: bytes, name: str = "", description: str = "") -> Dict:
        """Clone a voice from audio sample"""
        try:
            # Save uploaded audio temporarily
            temp_audio_path = TEMP_DIR / f"temp_{voice_id}.wav"
            with open(temp_audio_path, 'wb') as f:
                f.write(audio_file)
            
            # Load and validate audio
            audio, sr = torchaudio.load(temp_audio_path)
            
            # Ensure mono audio
            if audio.shape[0] > 1:
                audio = torch.mean(audio, dim=0, keepdim=True)
            
            # Resample to 16kHz if needed
            if sr != 16000:
                resampler = torchaudio.transforms.Resample(sr, 16000)
                audio = resampler(audio)
                sr = 16000
            
            # Calculate duration
            duration = audio.shape[1] / sr
            
            # Save voice dir and reference audio first
            voice_dir = VOICES_DIR / voice_id
            voice_dir.mkdir(exist_ok=True)
            voice_audio_path = voice_dir / "reference.wav"
            torchaudio.save(voice_audio_path, audio, sr)

            has_embedding = False
            # Extract speaker embedding if OpenVoice models are available
            if self.tone_color_converter is not None:
                try:
                    se_path = voice_dir / "speaker_embedding.pth"
                    gs = self.tone_color_converter.extract_se(str(voice_audio_path), se_save_path=str(se_path))
                    has_embedding = True
                except Exception as ee:
                    logger.warning(f"Failed to extract speaker embedding: {ee}")
            else:
                # Fallback: just save the audio for future processing
                logger.warning("OpenVoice not available, saving audio for future processing")
            
            # Save metadata
            metadata = {
                "voice_id": voice_id,
                "name": name or voice_id,
                "description": description,
                "created_at": datetime.now().isoformat(),
                "file_path": str(voice_audio_path),
                "sample_duration": duration,
                "has_embedding": has_embedding
            }
            
            self.voices_metadata[voice_id] = metadata
            self._save_voices_metadata()
            
            # Clean up temp file
            temp_audio_path.unlink(missing_ok=True)
            
            logger.info(f"Voice {voice_id} cloned successfully")
            return metadata
            
        except Exception as e:
            logger.error(f"Failed to clone voice {voice_id}: {e}")
            raise HTTPException(status_code=500, detail=f"Voice cloning failed: {str(e)}")
    
    def synthesize_with_cloned_voice(self, voice_id: str, text: str, language: str = "en") -> bytes:
        """Synthesize speech using a cloned voice"""
        try:
            if voice_id not in self.voices_metadata:
                raise HTTPException(status_code=404, detail=f"Voice {voice_id} not found")
            
            voice_dir = VOICES_DIR / voice_id
            se_path = voice_dir / "speaker_embedding.pth"
            
            if not se_path.exists() or self.base_speaker_tts is None or self.tone_color_converter is None:
                raise HTTPException(
                    status_code=400, 
                    detail="Voice cloning models not available or voice not properly processed"
                )
            
            # Load speaker embedding
            target_se = torch.load(se_path, map_location=self.device)
            
            # Generate speech with base model
            temp_base_path = TEMP_DIR / f"base_{voice_id}.wav"
            self.base_speaker_tts.tts(
                text, 
                str(temp_base_path), 
                speaker='default', 
                language=language,
                speed=1.0
            )
            
            # Apply tone color conversion
            temp_output_path = TEMP_DIR / f"output_{voice_id}.wav"
            
            # Load base audio
            # For conversion, use source speaker embedding by extracting from base TTS output or use a default if provided
            default_se_v1 = MODELS_DIR / 'checkpoints' / 'base_speakers' / 'EN' / 'en_default_se.pth'
            default_se_v2 = MODELS_DIR / 'checkpoints_v2' / 'base_speakers' / 'EN' / 'en_default_se.pth'
            if default_se_v1.exists():
                source_se = torch.load(default_se_v1, map_location=self.device)
            elif default_se_v2.exists():
                source_se = torch.load(default_se_v2, map_location=self.device)
            else:
                # As a fallback, extract se from the generated base audio
                try:
                    source_se = self.tone_color_converter.extract_se(str(temp_base_path))
                except Exception as ee:
                    logger.error(f"Failed to get source speaker embedding: {ee}")
                    raise
            
            self.tone_color_converter.convert(
                audio_src_path=str(temp_base_path),
                src_se=source_se,
                tgt_se=target_se,
                output_path=str(temp_output_path),
                message="Converting tone color..."
            )
            
            # Read the output audio
            with open(temp_output_path, 'rb') as f:
                audio_data = f.read()
            
            # Clean up temp files
            temp_base_path.unlink(missing_ok=True)
            temp_output_path.unlink(missing_ok=True)
            
            return audio_data
            
        except Exception as e:
            logger.error(f"Failed to synthesize with voice {voice_id}: {e}")
            raise HTTPException(status_code=500, detail=f"Synthesis failed: {str(e)}")
    
    def list_voices(self) -> List[VoiceInfo]:
        """List all available cloned voices"""
        voices = []
        for voice_id, metadata in self.voices_metadata.items():
            voices.append(VoiceInfo(**metadata))
        return voices
    
    def delete_voice(self, voice_id: str) -> bool:
        """Delete a cloned voice"""
        try:
            if voice_id not in self.voices_metadata:
                raise HTTPException(status_code=404, detail=f"Voice {voice_id} not found")
            
            # Remove voice directory
            voice_dir = VOICES_DIR / voice_id
            if voice_dir.exists():
                import shutil
                shutil.rmtree(voice_dir)
            
            # Remove from metadata
            del self.voices_metadata[voice_id]
            self._save_voices_metadata()
            
            logger.info(f"Voice {voice_id} deleted successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete voice {voice_id}: {e}")
            raise HTTPException(status_code=500, detail=f"Deletion failed: {str(e)}")

# Initialize service
voice_service = VoiceCloneService()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "voice-cloning-service",
        "device": voice_service.device,
        "openvoice_available": voice_service.openvoice_available and (voice_service.tone_color_converter is not None)
    }

@app.post("/clone")
async def clone_voice(
    voice_id: str = Form(...),
    name: str = Form(""),
    description: str = Form(""),
    audio_file: UploadFile = File(...)
):
    """Clone a voice from an audio sample"""
    if not audio_file.content_type.startswith('audio/'):
        raise HTTPException(status_code=400, detail="File must be an audio file")
    
    audio_data = await audio_file.read()
    result = voice_service.clone_voice(voice_id, audio_data, name, description)
    
    return {
        "message": f"Voice {voice_id} cloned successfully",
        "voice_info": result
    }

@app.post("/synthesize")
async def synthesize_voice(request: VoiceCloneRequest):
    """Synthesize speech using a cloned voice"""
    audio_data = voice_service.synthesize_with_cloned_voice(
        request.voice_id, 
        request.text, 
        request.language
    )
    
    # Save to temporary file and return
    temp_file = TEMP_DIR / f"synthesis_{request.voice_id}.wav"
    with open(temp_file, 'wb') as f:
        f.write(audio_data)
    
    return FileResponse(
        temp_file,
        media_type="audio/wav",
        filename=f"speech_{request.voice_id}.wav"
    )

@app.get("/voices")
async def list_voices():
    """List all available cloned voices"""
    voices = voice_service.list_voices()
    return {"voices": voices}

@app.delete("/voices/{voice_id}")
async def delete_voice(voice_id: str):
    """Delete a cloned voice"""
    success = voice_service.delete_voice(voice_id)
    if success:
        return {"message": f"Voice {voice_id} deleted successfully"}
    else:
        raise HTTPException(status_code=500, detail="Failed to delete voice")

@app.get("/voices/{voice_id}")
async def get_voice_info(voice_id: str):
    """Get information about a specific voice"""
    if voice_id not in voice_service.voices_metadata:
        raise HTTPException(status_code=404, detail=f"Voice {voice_id} not found")
    
    return voice_service.voices_metadata[voice_id]

@app.get("/voices/{voice_id}/sample")
async def get_voice_sample(voice_id: str):
    """Get the reference audio sample for a voice"""
    if voice_id not in voice_service.voices_metadata:
        raise HTTPException(status_code=404, detail=f"Voice {voice_id} not found")
    
    voice_dir = VOICES_DIR / voice_id
    sample_path = voice_dir / "reference.wav"
    
    if not sample_path.exists():
        raise HTTPException(status_code=404, detail="Voice sample not found")
    
    return FileResponse(
        sample_path,
        media_type="audio/wav",
        filename=f"sample_{voice_id}.wav"
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5001)
