from fastapi import FastAPI, UploadFile, HTTPException, File, Form
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from faster_whisper import WhisperModel
from typing import Any, Union, List
import torch
import os
import logging
import tempfile
import io
import json
import asyncio
import time
import uuid
import math
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="STT Service", description="Speech-to-Text Service with Hardware Acceleration")

# Add CORS middleware (env-configurable)
allowed_origins_str = os.getenv("ALLOWED_ORIGINS", "*")
allowed_origins = [o.strip() for o in allowed_origins_str.split(",")] if allowed_origins_str else ["*"]
allow_credentials = os.getenv("ALLOW_CREDENTIALS", "false").strip().lower() in {"1", "true", "yes", "on"}
if "*" in allowed_origins and allow_credentials:
    logger.warning("ALLOW_CREDENTIALS=true with ALLOWED_ORIGINS='*' is not permitted by CORS spec; disabling credentials.")
    allow_credentials = False

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,  # Allows configured origins
    allow_credentials=allow_credentials,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Create a thread pool for running blocking IO
executor = ThreadPoolExecutor(max_workers=os.cpu_count())

def clean_json_inf_nan(data):
    """Recursively clean a data structure, replacing float inf/NaN with None."""
    if isinstance(data, dict):
        return {k: clean_json_inf_nan(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [clean_json_inf_nan(i) for i in data]
    elif isinstance(data, float):
        if math.isinf(data) or math.isnan(data):
            return None  # Replace with null in JSON
        return data
    return data

class SafeJSONResponse(JSONResponse):
    """JSONResponse subclass that handles NaN/Infinity values."""
    def render(self, content: Any) -> bytes:
        return json.dumps(
            clean_json_inf_nan(content),
            ensure_ascii=False,
            allow_nan=False, # This is the default, but being explicit
            indent=None,
            separators=(",", ":"),
        ).encode("utf-8")

# Hardware detection and optimization (re-check after startup)
def detect_hardware():
    """Detect and configure hardware acceleration"""
    import torch

    force_accel = os.getenv("FORCE_ACCELERATION", "").lower()

    if force_accel == "rocm":
        # ROCm presents as CUDA to PyTorch; ctranslate2 4.x+ required for GPU use.
        # If ctranslate2 was not compiled with HIP support, fall back gracefully to CPU.
        try:
            import ctranslate2
            providers = ctranslate2.get_supported_compute_types("cuda")
            if "float16" in providers:
                device = "cuda"
                compute_type = "float16"
                logger.info(f"ROCm mode: using GPU (ctranslate2 HIP). GPU: {torch.cuda.get_device_name(0)}")
            else:
                raise RuntimeError("ctranslate2 has no float16 CUDA/HIP support")
        except Exception as e:
            logger.warning(f"ROCm requested but ctranslate2 HIP unavailable ({e}); falling back to CPU int8")
            device = "cpu"
            compute_type = "int8"
        return device, compute_type

    # Explicit CPU override
    if os.getenv("USE_CUDA", "").lower() == "false":
        logger.info("Hardware acceleration disabled via USE_CUDA=false")
        return "cpu", "int8"

    if torch.cuda.is_available():
        device = "cuda"
        compute_type = "float16"
        logger.info(f"CUDA available with {torch.cuda.device_count()} GPU(s)")
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = "cpu"  # faster-whisper doesn't support MPS
        compute_type = "int8"
        logger.info("Apple Silicon detected, using optimized CPU int8")
    else:
        device = "cpu"
        compute_type = "int8"
        logger.info("Using CPU int8")

    return device, compute_type

# Defer hardware detection until first use
device = None
compute_type = None

# Initialize Whisper model with error handling
whisper_model = None
model_loaded = False
model_size_loaded = None

def ensure_hardware_detected():
    """Ensure hardware detection has been run"""
    global device, compute_type
    if device is None:
        device, compute_type = detect_hardware()

def load_model():
    """Load Whisper model with fallback"""
    global whisper_model, model_loaded, model_size_loaded
    
    # Ensure hardware is detected first
    ensure_hardware_detected()
    
    try:
        model_size = os.getenv("WHISPER_MODEL_SIZE", "large-v3")
        model_size_loaded = model_size
        logger.info(f"Loading {model_size} Whisper model on {device} with {compute_type}...")
        
        whisper_model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
            cpu_threads=os.cpu_count() if device == "cpu" else 4,
            num_workers=1  # Reduce memory usage
        )
        model_loaded = True
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        # Try CPU fallback with int8
        if device != "cpu" or compute_type != "int8":
            logger.info("Trying CPU int8 fallback...")
            try:
                whisper_model = WhisperModel(
                    os.getenv("WHISPER_MODEL_SIZE", "large-v3"),
                    device="cpu",
                    compute_type="int8",
                    cpu_threads=os.cpu_count(),
                    num_workers=1
                )
                model_loaded = True
                model_size_loaded = os.getenv("WHISPER_MODEL_SIZE", "large-v3")
                logger.info("Model loaded on CPU with int8")
            except Exception as cpu_error:
                logger.error(f"CPU fallback failed: {cpu_error}")
                model_loaded = False

# Load model on startup
@app.on_event("startup")
async def startup_event():
    load_model()

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down thread pool executor...")
    executor.shutdown(wait=False)

# Add better error handling and logging
@app.post("/transcribe")
async def transcribe_audio(
    # Accept either a single file (legacy) or multiple files (batch)
    audio: UploadFile = File(None),
    audios: List[UploadFile] = File(None),  # client can send multiple 'audios' fields or 'audio'
    task: str = Form("transcribe"),
    language: str = Form("auto"),
    beam_size: int = Form(5),
    best_of: int = Form(5),
    patience: float = Form(1.0),
    temperature: Union[float, str] = Form("0.0,0.2,0.4,0.6,0.8,1.0"),
    suppress_tokens: str = Form("-1"),
    initial_prompt: str = Form(""),
    condition_on_previous_text: bool = Form(True),
    compression_ratio_threshold: float = Form(2.4),
    no_speech_threshold: float = Form(0.6),
    vad_filter: bool = Form(True),
    vad_threshold: float = Form(0.5),
):
    """Transcribe audio file to text"""
    logger.info(f"Transcription request - task: {task}, language: {language}")
    
    try:
        # Normalize file list
        file_list: List[UploadFile] = []
        if audios:
            file_list.extend(audios)
        if audio:
            file_list.append(audio)
        if not file_list:
            raise HTTPException(status_code=400, detail="No audio file(s) provided")

        batch_mode = len(file_list) > 1
        batch_results = []
        combined_text_parts = []
        total_processing_time = 0.0
        total_audio_duration = 0.0

        # Ensure model is loaded (once)
        global whisper_model
        if whisper_model is None:
            logger.error("Model not loaded, attempting to load...")
            load_model()  # Sets global whisper_model internally
            if whisper_model is None:
                raise HTTPException(status_code=503, detail="Model not available")

        # Parse temperature once
        if isinstance(temperature, str):
            temperature = [float(t) for t in temperature.split(",")]
        else:
            temperature = [float(temperature)]

        # Parse suppress_tokens once
        parsed_suppress_tokens = None
        try:
            if isinstance(suppress_tokens, str):
                st = suppress_tokens.strip()
                if st and st != "-1":
                    parsed_suppress_tokens = [int(tok.strip()) for tok in st.split(",") if tok.strip()]
        except ValueError:
            logger.warning(f"Invalid suppress_tokens value '{suppress_tokens}', ignoring.")
            parsed_suppress_tokens = None

        # Process each file sequentially
        for afile in file_list:
            safe_filename = os.path.basename(afile.filename or "audio.wav")
            suffix = os.path.splitext(safe_filename)[1] or ".wav"
            tmp_fd, temp_audio_path = tempfile.mkstemp(suffix=suffix)
            with os.fdopen(tmp_fd, "wb") as f:
                content = await afile.read()
                f.write(content)
            logger.info(f"Saved audio file: {temp_audio_path}, size: {os.path.getsize(temp_audio_path)} bytes")

            logger.info(f"Starting transcription for file {afile.filename} (vad_filter={vad_filter}, vad_threshold={vad_threshold})...")
            start_time = time.time()
            segments, info = whisper_model.transcribe(
                temp_audio_path,
                task=task,
                language=None if language == "auto" else language,
                beam_size=beam_size,
                best_of=best_of,
                patience=patience,
                temperature=temperature,
                suppress_tokens=parsed_suppress_tokens,
                initial_prompt=initial_prompt,
                condition_on_previous_text=condition_on_previous_text,
                compression_ratio_threshold=compression_ratio_threshold,
                no_speech_threshold=no_speech_threshold,
                vad_filter=vad_filter,
                vad_parameters={
                    "threshold": vad_threshold,
                    "min_speech_duration_ms": 500,
                    "min_silence_duration_ms": 1500,
                    "speech_pad_ms": 300,
                } if vad_filter else None,
            )

            segments_list = []
            full_text = ""
            last_text = ""
            processed_segments = 0
            total_duration = info.duration or 0
            
            logger.info(f"Processing segments for {afile.filename} (duration: {total_duration:.1f}s)")
            
            for segment in segments:
                processed_segments += 1
                text = segment.text.strip()
                seg_duration = segment.end - segment.start
                
                # Log progress every 50 segments or at significant timestamps
                if processed_segments % 50 == 0 or segment.end > 0:
                    if total_duration > 0:
                        progress_pct = min((segment.end / total_duration) * 100, 100)
                        logger.info(f"Progress: {processed_segments} segments processed, {segment.end:.1f}s/{total_duration:.1f}s ({progress_pct:.1f}%)")
                    else:
                        logger.info(f"Progress: {processed_segments} segments processed, current time: {segment.end:.1f}s")
                
                if seg_duration < 0.2 or segment.no_speech_prob > 0.8:
                    continue
                if text == last_text or not text:
                    continue
                segment_dict = {
                    "start": segment.start,
                    "end": segment.end,
                    "text": text,
                    "avg_logprob": segment.avg_logprob,
                    "no_speech_prob": segment.no_speech_prob
                }
                segments_list.append(segment_dict)
                full_text += text + " "
                last_text = text

            processing_time = time.time() - start_time
            total_processing_time += processing_time
            total_audio_duration += info.duration or 0
            
            logger.info(f"Transcription completed for {afile.filename}")
            logger.info(f"Final stats: {len(segments_list)} valid segments created from {processed_segments} total segments")
            logger.info(f"Processing time: {processing_time:.2f}s for {total_duration:.1f}s audio")
            logger.info(f"Speed: {total_duration/processing_time:.1f}x realtime")

            try:
                os.unlink(temp_audio_path)
            except Exception:
                pass

            batch_results.append({
                "filename": afile.filename,
                "text": full_text.strip(),
                "segments": segments_list,
                "language": info.language,
                "language_probability": info.language_probability,
                "duration": info.duration,
                "task": task,
                "processing_time": processing_time
            })
            combined_text_parts.append(full_text.strip())

        # Single file: preserve legacy response shape
        if not batch_mode:
            if not batch_results:
                raise HTTPException(status_code=422, detail="No transcription results (empty or unreadable audio)")
            return JSONResponse(content=batch_results[0])

        # Batch: new response shape
        if not batch_results:
            raise HTTPException(status_code=422, detail="No transcription results for any provided files")

        batch_response = {
            "batch": True,
            "file_count": len(batch_results),
            "combined_text": " \n".join([r["text"] for r in batch_results]).strip(),
            "results": batch_results,
            "total_processing_time": total_processing_time,
            "total_duration": total_audio_duration,
            "task": task,
            "language": batch_results[0]["language"],
            "language_probability": batch_results[0]["language_probability"],
        }
        return JSONResponse(content=batch_response)
        
    except Exception as e:
        logger.error(f"Transcription error: {str(e)}", exc_info=True)
        # Clean up temp file if it exists
        if 'temp_audio_path' in locals() and os.path.exists(temp_audio_path):
            os.unlink(temp_audio_path)
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

@app.post("/transcribe-stream")
async def transcribe_audio_stream(
    audio: UploadFile = File(...),
    language: str = Form(None),
    task: str = Form("transcribe"),
    target_language: str = Form("english"),
    beam_size: int = Form(5),
    vad_filter: bool = Form(True),
    vad_threshold: float = Form(0.5),
    no_speech_threshold: float = Form(0.6),
):
    """Stream transcription results as they become available"""
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Whisper model not loaded")
    
    req_id = uuid.uuid4().hex[:8]
    
    # Save uploaded file to a temporary location first
    try:
        audio_content = await audio.read()
        suffix = os.path.splitext(audio.filename or "audio.wav")[1] or ".wav"
        tmp_fd, tmp_file_path = tempfile.mkstemp(suffix=suffix)
        with os.fdopen(tmp_fd, "wb") as tmp_file:
            tmp_file.write(audio_content)
        logger.info(f"[{req_id}] Saved temp file for streaming: {tmp_file_path}")
    except Exception as e:
        logger.error(f"[{req_id}] Failed to save temp file: {e}")
        raise HTTPException(status_code=500, detail="Failed to process uploaded file.")

    async def generate_transcription():
        """Generator function for streaming transcription"""
        try:
            logger.info(f"[{req_id}] /transcribe-stream request received: filename={audio.filename}, content_type={audio.content_type}, language={language}, task={task}")
            logger.info(f"[{req_id}] Uploaded audio size: {len(audio_content)} bytes")
            
            # For translation task, ensure target language is supported
            if task == "translate" and target_language.lower() not in ["english", "en"]:
                yield f"data: {json.dumps({'warning': f'Translation to {target_language} not supported, using English'})}\n\n"
                target_language = "english"
            
            # Start transcription
            yield f"data: {json.dumps({'status': 'processing', 'task': task})}\n\n"
            logger.info(f"[{req_id}] Starting streaming transcription...")
            
            # Run transcription in executor to avoid blocking
            def run_transcription():
                return whisper_model.transcribe(
                    tmp_file_path,
                    beam_size=beam_size,
                    best_of=beam_size,
                    temperature=(0.0, 0.2, 0.4, 0.6, 0.8),
                    compression_ratio_threshold=2.4,
                    no_speech_threshold=no_speech_threshold,
                    language=language,
                    task=task,
                    vad_filter=vad_filter,
                    vad_parameters={
                        "threshold": vad_threshold,
                        "min_speech_duration_ms": 500,
                        "min_silence_duration_ms": 1500,
                        "speech_pad_ms": 300,
                    } if vad_filter else None,
                )
            
            loop = asyncio.get_running_loop()
            segments, info = await loop.run_in_executor(executor, run_transcription)
            
            # Send metadata first
            metadata = {
                "language": info.language,
                "language_probability": info.language_probability,
                "duration": info.duration,
                "task": task
            }
            if task == "translate":
                metadata["target_language"] = target_language
            
            yield f"data: {json.dumps(clean_json_inf_nan({'metadata': metadata}))}\n\n"
            
            # Stream segments as they're processed
            full_text = ""
            segment_count = 0
            for i, segment in enumerate(segments):
                segment_count = i + 1
                segment_data = {
                    "segment_id": i,
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text.strip(),
                    "avg_logprob": segment.avg_logprob,
                    "no_speech_prob": segment.no_speech_prob
                }
                full_text += segment.text.strip() + " "
                
                yield f"data: {json.dumps(clean_json_inf_nan({'segment': segment_data}))}\n\n"
            
            # Send final result
            final_result = {
                "final_text": full_text.strip(),
                "status": "completed",
                "total_segments": segment_count
            }
            yield f"data: {json.dumps(final_result)}\n\n"
                
        except Exception as e:
            logger.error(f"[{req_id}] Streaming transcription error: {e}", exc_info=True)
            error_data = {"error": f"Transcription failed: {str(e)}", "status": "error"}
            yield f"data: {json.dumps(error_data)}\n\n"
        finally:
            # Cleanup temporary file
            if os.path.exists(tmp_file_path):
                try:
                    os.unlink(tmp_file_path)
                    logger.info(f"[{req_id}] Cleaned up temp file: {tmp_file_path}")
                except Exception as e:
                    logger.warning(f"[{req_id}] Could not delete temp file {tmp_file_path}: {e}")

    return StreamingResponse(
        generate_transcription(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )

ENGLISH_ONLY_MODELS = {"tiny.en", "base.en", "small.en", "medium.en", "distil-large-v2", "distil-large-v3", "distil-medium.en", "distil-small.en"}

@app.get("/health", response_class=SafeJSONResponse)
async def health_check():
    """Health check endpoint"""
    current_model = model_size_loaded or os.getenv("WHISPER_MODEL_SIZE", "large-v3")
    return {
        "status": "ok",
        "model_loaded": model_loaded,
        "device": device,
        "compute_type": compute_type,
        "model_size": current_model,
        "multilingual": current_model not in ENGLISH_ONLY_MODELS,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available()
    }

@app.get("/info", response_class=SafeJSONResponse)
async def service_info():
    """Service information endpoint"""
    return {
        "service": "STT Service",
        "device": device,
        "compute_type": compute_type,
        "model_loaded": model_loaded,
        "model_size": os.getenv("WHISPER_MODEL_SIZE", "large-v3"),
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
            {"name": "tiny",            "multilingual": True,  "size_mb": 75},
            {"name": "tiny.en",         "multilingual": False, "size_mb": 75},
            {"name": "base",            "multilingual": True,  "size_mb": 145},
            {"name": "base.en",         "multilingual": False, "size_mb": 145},
            {"name": "small",           "multilingual": True,  "size_mb": 490},
            {"name": "small.en",        "multilingual": False, "size_mb": 490},
            {"name": "medium",          "multilingual": True,  "size_mb": 1500},
            {"name": "medium.en",       "multilingual": False, "size_mb": 1500},
            {"name": "large-v1",        "multilingual": True,  "size_mb": 2900},
            {"name": "large-v2",        "multilingual": True,  "size_mb": 2900},
            {"name": "large-v3",        "multilingual": True,  "size_mb": 3100},
            {"name": "distil-large-v3", "multilingual": False, "size_mb": 1500, "note": "English-only, fastest"},
        ],
        "current_model": os.getenv("WHISPER_MODEL_SIZE", "large-v3"),
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

@app.post("/detect_language")
async def detect_language(
    file: UploadFile = File(...),
    duration_limit: float = 30.0  # Only analyze first 30 seconds for speed
):
    """Detect the language of an audio file quickly"""
    if not model_loaded or whisper_model is None:
        logger.error("Model not loaded, attempting to load...")
        load_model()
        if not model_loaded or whisper_model is None:
            raise HTTPException(status_code=503, detail="Model not available")

    safe_filename = os.path.basename(file.filename or "audio.wav")
    suffix = os.path.splitext(safe_filename)[1] or ".wav"
    tmp_fd, temp_audio_path = tempfile.mkstemp(suffix=suffix)
    try:
        with os.fdopen(tmp_fd, "wb") as f:
            content = await file.read()
            f.write(content)

        logger.info(f"Detecting language for: {file.filename}")
        start_time = time.time()
        
        # Use transcribe with short duration for quick language detection
        segments, info = whisper_model.transcribe(
            temp_audio_path,
            task="transcribe",
            language=None,  # Auto-detect
            beam_size=1,    # Fastest setting
            best_of=1,      # Fastest setting
            vad_filter=True,
            condition_on_previous_text=False,
            # Only process first part for speed
            vad_parameters={
                "threshold": 0.5,
                "min_speech_duration_ms": 250,
                "min_silence_duration_ms": 500,
            }
        )
        
        processing_time = time.time() - start_time
        
        # Get a sample of text for confidence
        sample_text = ""
        segment_count = 0
        for segment in segments:
            if segment_count >= 3:  # Only need a few segments for detection
                break
            if segment.no_speech_prob < 0.8:
                sample_text += segment.text.strip() + " "
                segment_count += 1
        
        logger.info(f"Language detection completed in {processing_time:.2f}s: {info.language} ({info.language_probability:.2f})")
        
        return {
            "detected_language": info.language,
            "language_probability": info.language_probability,
            "sample_text": sample_text.strip(),
            "processing_time": processing_time,
            "audio_duration": info.duration
        }
        
    except Exception as e:
        logger.error(f"Language detection failed: {e}")
        raise HTTPException(status_code=500, detail=f"Language detection failed: {str(e)}")
    finally:
        try:
            os.unlink(temp_audio_path)
        except Exception:
            pass
