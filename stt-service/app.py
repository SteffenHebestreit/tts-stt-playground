from fastapi import FastAPI, UploadFile, HTTPException, File, Form
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from faster_whisper import WhisperModel
import torch
import os
import logging
import tempfile
import io
import json
import asyncio

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="STT Service", description="Speech-to-Text Service with Hardware Acceleration")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Hardware detection and optimization
def detect_hardware():
    """Detect and configure hardware acceleration"""
    if torch.cuda.is_available():
        device = "cuda"
        compute_type = "float16"
        logger.info(f"🚀 CUDA available with {torch.cuda.device_count()} GPU(s)")
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = "cpu"  # Faster-whisper doesn't support MPS yet
        compute_type = "int8"
        logger.info("🚀 Apple Silicon detected, using optimized CPU")
    else:
        device = "cpu"
        compute_type = "int8"
        logger.info("💻 Using CPU")
    
    # Check environment override
    env_device = os.getenv("USE_CUDA", "").lower()
    if env_device == "false":
        device = "cpu"
        compute_type = "int8"
        logger.info("🔧 Hardware acceleration disabled via environment")
    
    return device, compute_type

# Initialize hardware
device, compute_type = detect_hardware()

# Initialize Whisper model with error handling
whisper_model = None
model_loaded = False

def load_model():
    """Load Whisper model with fallback"""
    global whisper_model, model_loaded
    try:
        model_size = os.getenv("WHISPER_MODEL_SIZE", "tiny")
        logger.info(f"Loading {model_size} Whisper model on {device} with {compute_type}...")
        
        whisper_model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
            cpu_threads=os.cpu_count() if device == "cpu" else 4,
            num_workers=1  # Reduce memory usage
        )
        model_loaded = True
        logger.info("✅ Model loaded successfully")
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        # Try CPU fallback with int8
        if device != "cpu" or compute_type != "int8":
            logger.info("🔄 Trying CPU int8 fallback...")
            try:
                whisper_model = WhisperModel(
                    os.getenv("WHISPER_MODEL_SIZE", "tiny"),
                    device="cpu",
                    compute_type="int8",
                    cpu_threads=os.cpu_count(),
                    num_workers=1
                )
                model_loaded = True
                logger.info("✅ Model loaded on CPU with int8")
            except Exception as cpu_error:
                logger.error(f"❌ CPU fallback failed: {cpu_error}")
                model_loaded = False

# Load model on startup
@app.on_event("startup")
async def startup_event():
    load_model()

@app.post("/transcribe")
async def transcribe_audio(
    audio: UploadFile = File(...),
    language: str = Form(None),
    task: str = Form("transcribe"),  # transcribe or translate
    target_language: str = Form("english"),  # target language for translation task
    beam_size: int = Form(5),
    best_of: int = Form(5)
):
    """Transcribe audio to text"""
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Whisper model not loaded")
    
    try:
        # Read audio file
        audio_content = await audio.read()
        
        # Save to temporary file for processing
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(audio_content)
            tmp_file_path = tmp_file.name
        
        try:
            # For translation task, ensure target language is supported
            if task == "translate":
                # Whisper can only translate to English currently
                if target_language.lower() not in ["english", "en"]:
                    logger.warning(f"Translation to {target_language} not supported, defaulting to English")
                    target_language = "english"
                
                logger.info(f"Performing translation to {target_language}")
            
            # Transcribe with optimized parameters
            segments, info = whisper_model.transcribe(
                tmp_file_path,
                beam_size=min(beam_size, 5),  # Limit beam size for performance
                best_of=min(best_of, 5),      # Limit best_of for performance
                temperature=0.0,              # Deterministic output
                language=language,
                task=task,
                vad_filter=True,              # Voice activity detection
                vad_parameters=dict(
                    min_silence_duration_ms=500,
                    threshold=0.5
                )
            )
            
            # Extract text and metadata
            text = " ".join([segment.text.strip() for segment in segments])
            
            # Gather segment details
            segment_details = []
            for segment in segments:
                segment_details.append({
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text.strip(),
                    "avg_logprob": segment.avg_logprob,
                    "no_speech_prob": segment.no_speech_prob
                })
            
            result = {
                "text": text,
                "language": info.language,
                "language_probability": info.language_probability,
                "duration": info.duration,
                "segments": segment_details,
                "task": task
            }
            
            # Add target language info for translation tasks
            if task == "translate":
                result["target_language"] = target_language
                result["note"] = "Translation performed to English (Whisper limitation)"
            
            return result
            
        finally:
            # Cleanup temporary file
            os.unlink(tmp_file_path)
            
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

@app.post("/transcribe-stream")
async def transcribe_audio_stream(
    audio: UploadFile = File(...),
    language: str = Form(None),
    task: str = Form("transcribe"),
    target_language: str = Form("english")
):
    """Stream transcription results as they become available"""
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Whisper model not loaded")
    
    async def generate_transcription():
        """Generator function for streaming transcription"""
        try:
            # Read audio file
            audio_content = await audio.read()
            
            # Save to temporary file for processing
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                tmp_file.write(audio_content)
                tmp_file_path = tmp_file.name
            
            try:
                # For translation task, ensure target language is supported
                if task == "translate" and target_language.lower() not in ["english", "en"]:
                    yield f"data: {json.dumps({'warning': f'Translation to {target_language} not supported, using English'})}\n\n"
                    target_language = "english"
                
                # Start transcription
                yield f"data: {json.dumps({'status': 'processing', 'task': task})}\n\n"
                
                # Transcribe with streaming-friendly parameters
                segments, info = whisper_model.transcribe(
                    tmp_file_path,
                    beam_size=3,  # Reduced for faster streaming
                    best_of=3,    # Reduced for faster streaming
                    temperature=0.0,
                    language=language,
                    task=task,
                    vad_filter=True,
                    vad_parameters=dict(
                        min_silence_duration_ms=300,  # More responsive
                        threshold=0.5
                    )
                )
                
                # Send metadata first
                metadata = {
                    "language": info.language,
                    "language_probability": info.language_probability,
                    "duration": info.duration,
                    "task": task
                }
                if task == "translate":
                    metadata["target_language"] = target_language
                
                yield f"data: {json.dumps({'metadata': metadata})}\n\n"
                
                # Stream segments as they're processed
                full_text = ""
                for i, segment in enumerate(segments):
                    segment_data = {
                        "segment_id": i,
                        "start": segment.start,
                        "end": segment.end,
                        "text": segment.text.strip(),
                        "avg_logprob": segment.avg_logprob,
                        "no_speech_prob": segment.no_speech_prob
                    }
                    full_text += segment.text.strip() + " "
                    
                    yield f"data: {json.dumps({'segment': segment_data})}\n\n"
                
                # Send final result
                final_result = {
                    "final_text": full_text.strip(),
                    "status": "completed",
                    "total_segments": i + 1 if 'i' in locals() else 0
                }
                yield f"data: {json.dumps(final_result)}\n\n"
                
            finally:
                # Cleanup temporary file
                os.unlink(tmp_file_path)
                
        except Exception as e:
            logger.error(f"Streaming transcription error: {e}")
            error_data = {"error": f"Transcription failed: {str(e)}", "status": "error"}
            yield f"data: {json.dumps(error_data)}\n\n"
    
    return StreamingResponse(
        generate_transcription(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream"
        }
    )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "ok",
        "model_loaded": model_loaded,
        "device": device,
        "compute_type": compute_type,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available()
    }

@app.get("/info")
async def service_info():
    """Service information endpoint"""
    return {
        "service": "STT Service",
        "device": device,
        "compute_type": compute_type,
        "model_loaded": model_loaded,
        "model_size": os.getenv("WHISPER_MODEL_SIZE", "tiny"),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    }

@app.get("/models")
async def available_models():
    """List available Whisper models"""
    return {
        "available_models": [
            "tiny", "tiny.en", "base", "base.en", 
            "small", "small.en", "medium", "medium.en",
            "large-v1", "large-v2", "large-v3"
        ],
        "current_model": os.getenv("WHISPER_MODEL_SIZE", "tiny"),
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
    """Get information about available transcription tasks"""
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
