"""Frontend service for the TTS-STT platform.

Serves the web UI, static assets, and API documentation pages.
Acts as a gateway that provides browser-facing URLs for all backend services.
"""

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.background import BackgroundTask

from openai_router import build_router as build_openai_router
from pydantic import BaseModel
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import asyncio
import hashlib
import httpx
import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def _asset_version() -> str:
    """Cache-busting token derived from the static assets themselves.

    It has to satisfy two things at once:
    - identical across uvicorn workers, or they hand out different asset URLs
      for the same files and thrash the browser cache;
    - different whenever the assets actually change, or browsers keep serving a
      stale app.js after an update.

    A restart timestamp fails the first, and a pinned image tag fails the second
    (deployments on `:latest` would never bust). Hashing the files satisfies both.
    """
    explicit = os.getenv("APP_VERSION")
    if explicit:
        return explicit

    digest = hashlib.sha256()
    static_dir = Path(__file__).parent / "static"
    try:
        for path in sorted(static_dir.rglob("*")):
            if path.is_file():
                digest.update(path.name.encode())
                digest.update(str(path.stat().st_mtime_ns).encode())
                digest.update(str(path.stat().st_size).encode())
    except OSError:
        # Fall back to a per-process token rather than failing to start.
        return str(int(time.time()))
    return digest.hexdigest()[:12]


APP_VERSION = _asset_version()


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    """Close the shared pooled HTTP client when the service stops."""
    yield
    global _http_client
    client, _http_client = _http_client, None
    aclose = getattr(client, "aclose", None) if client is not None else None
    if aclose is not None:
        try:
            await aclose()
        except Exception:
            pass


app = FastAPI(title="TTS-STT Frontend Service", version="2.0.0", lifespan=_lifespan)

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

BASE_DIR = Path(__file__).resolve().parent

app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# Internal Docker-network URLs (container-to-container communication)
TTS_SERVICE_URL = os.getenv("TTS_SERVICE_URL", "http://piper-tts-service:5000")
STT_SERVICE_URL = os.getenv("STT_SERVICE_URL", "http://stt-service:8000")
VOICE_TRAINING_URL = os.getenv("VOICE_TRAINING_URL", "http://piper-training-service:8080")
QWEN3_TTS_SERVICE_URL = os.getenv("QWEN3_TTS_SERVICE_URL", "http://qwen3-tts-service:5004")
QWEN3_ASR_SERVICE_URL = os.getenv("QWEN3_ASR_SERVICE_URL", "http://qwen3-asr-service:5002")
PARAKEET_ASR_SERVICE_URL = os.getenv("PARAKEET_ASR_SERVICE_URL", "http://parakeet-asr-service:5005")
CANARY_ASR_SERVICE_URL = os.getenv("CANARY_ASR_SERVICE_URL", "http://canary-asr-service:5006")
CHATTERBOX_TTS_SERVICE_URL = os.getenv("CHATTERBOX_TTS_SERVICE_URL", "http://chatterbox-tts-service:5007")
WHISPER_CPP_SERVICE_URL = os.getenv("WHISPER_CPP_SERVICE_URL", "http://whisper-cpp:8080")

# Browser-facing URLs (host ports, used by client-side JavaScript)
ENABLE_WHISPER_CPP = os.getenv("ENABLE_WHISPER_CPP", "false").strip().lower() in {"1", "true", "yes", "on"}
ENABLE_PARAKEET_ASR = os.getenv("ENABLE_PARAKEET_ASR", "false").strip().lower() in {"1", "true", "yes", "on"}
ENABLE_CANARY_ASR = os.getenv("ENABLE_CANARY_ASR", "false").strip().lower() in {"1", "true", "yes", "on"}
ENABLE_CHATTERBOX_TTS = os.getenv("ENABLE_CHATTERBOX_TTS", "false").strip().lower() in {"1", "true", "yes", "on"}


def _build_basic_tts_messages() -> dict:
    """Return shared metadata-driven messages for the generic TTS flow."""
    return {
        "validation_text": "Please enter some text to synthesize",
        "start": "Generating speech with {provider}...",
        "success": "Speech generated successfully!",
        "error": "Generation failed: {error}",
        "voice_auto_option": "Auto-Select Best Voice",
    }


def _build_stt_messages() -> dict:
    """Return shared metadata-driven messages for the generic STT flow."""
    return {
        "validation_file": "Please select an audio file",
        "start": "Processing audio with {provider}...",
        "success": "Audio processed successfully!",
        "error": "Processing failed: {error}",
        "segmented_heading": "Transcription with Segmentation",
        "result_heading": "Transcription Result",
        "copy_action": "Copy Text",
        "copy_success": "Transcription copied to clipboard",
        "language_label": "Language",
        "duration_label": "Duration",
        "segments_label": "Segments",
        "unknown": "Unknown",
        "not_available": "N/A",
    }


def _build_provider_registry() -> dict:
    """Build the browser-facing provider registry for the frontend UI."""
    providers = {
        "piper": {
            "kind": "tts",
            "display_name": "PiperTTS (Local Training)",
            "short_name": "PiperTTS",
            "internal_url": TTS_SERVICE_URL,
            "health_endpoint": "/health",
            "capabilities": ["tts", "voice_catalog", "custom_models", "training_target"],
            "contracts": {
                "tts": "simple-json-tts-v1",
                "voice_catalog": "voice-catalog-v1",
                "managed_voices": "custom-voice-library-v1",
            },
            "settings": {
                "defaults": {
                    "language": "auto",
                    "quality": "medium",
                    "gender": "any",
                    "speed": 1.0,
                },
                "languages": [
                    {"value": "auto", "label": "Auto-Detect"},
                    {"value": "en", "label": "English"},
                    {"value": "de", "label": "German"},
                    {"value": "fr", "label": "French"},
                    {"value": "es", "label": "Spanish"},
                    {"value": "it", "label": "Italian"},
                    {"value": "nl", "label": "Dutch"},
                ],
                "qualities": [
                    {"value": "medium", "label": "Medium Quality"},
                    {"value": "high", "label": "High Quality"},
                    {"value": "low", "label": "Low Quality (Faster)"},
                    {"value": "x_low", "label": "Ultra Low (Fastest)"},
                ],
                "genders": [
                    {"value": "any", "label": "Any Gender"},
                    {"value": "male", "label": "Male Voice"},
                    {"value": "female", "label": "Female Voice"},
                ],
                "speed": {
                    "min": 0.5,
                    "max": 2.0,
                    "step": 0.1,
                    "default": 1.0,
                },
            },
            "ui": {
                "family": "piper",
                "selectable_as_engine": True,
                "show_status": True,
                "tab_label": "Text-to-Speech",
                "messages": {
                    "tts_generation": _build_basic_tts_messages(),
                    "custom_voice_library": {
                        "loading": "Loading voices...",
                        "empty_invalid": "No voices available.",
                        "empty": "No custom trained voices found. Train a voice model and it will appear here.",
                        "unavailable": "Failed to load voices. Is the PiperTTS service running?",
                        "action_test": "Test",
                        "action_delete": "Delete",
                        "test_start": "Generating test audio...",
                        "test_error": "Test failed: {error}",
                        "delete_confirm": "Delete custom voice \"{voice_id}\"? This cannot be undone.",
                        "delete_success": "Voice \"{voice_id}\" deleted",
                        "delete_error": "Failed to delete voice: {error}",
                    },
                },
                "sections": {
                    "tts": {
                        "title": "Text-to-Speech (PiperTTS)",
                        "description": "Generate high-quality speech using PiperTTS with intelligent voice selection based on language, quality, and gender preferences.",
                        "text_placeholder": "Enter the text you want to convert to speech...",
                        "text_sample": "Hello! This is a test of the PiperTTS neural text-to-speech system.",
                        "custom_voices_title": "Custom Voices",
                        "custom_voices_description": "Manage custom trained voices loaded in PiperTTS. Delete voices you no longer need.",
                    }
                },
            },
        },
        "qwen3": {
            "kind": "tts",
            "display_name": "Qwen3-TTS (Voice Cloning)",
            "short_name": "Qwen3-TTS",
            "internal_url": QWEN3_TTS_SERVICE_URL,
            "health_endpoint": "/health",
            "capabilities": ["tts", "voice_clone", "saved_voices", "model_switching", "model_unload"],
            "contracts": {
                "tts": "simple-json-tts-v1",
                "voice_catalog": "speaker-catalog-v1",
                "model_catalog": "model-catalog-v1",
                "model_selection": "model-selection-v1",
                "runtime_status": "runtime-status-v1",
                "saved_voices": "saved-voice-library-v1",
                "voice_clone": "voice-clone-tts-v1",
                "voice_design": "voice-design-tts-v1",
            },
            "settings": {
                "defaults": {
                    "language": "English",
                    "speaker": "Vivian",
                },
                "languages": [
                    {"value": "English", "label": "English"},
                    {"value": "German", "label": "German"},
                    {"value": "French", "label": "French"},
                    {"value": "Spanish", "label": "Spanish"},
                    {"value": "Italian", "label": "Italian"},
                    {"value": "Portuguese", "label": "Portuguese"},
                    {"value": "Russian", "label": "Russian"},
                    {"value": "Japanese", "label": "Japanese"},
                    {"value": "Korean", "label": "Korean"},
                    {"value": "Chinese", "label": "Chinese"},
                ],
            },
            "ui": {
                "family": "qwen3",
                "selectable_as_engine": True,
                "show_status": True,
                "tab_label": "Qwen3 TTS",
                "clone_tab_label": "Voice Cloning",
                "messages": {
                    "tts_generation": _build_basic_tts_messages(),
                    "model_switching": {
                        "start": "Switching model... This may take a while if the model needs to download.",
                        "success": "Model switched to {model}",
                        "error": "Failed to switch model: {error}",
                    },
                    "model_catalog": {
                        "current_description": "Current: {model} | Capabilities: {capabilities}",
                        "unavailable_option": "Service unavailable",
                    },
                    "voice_library": {
                        "save_start": "Saving voice \"{name}\" (transcribing + extracting embedding)...",
                        "save_success": "Voice \"{name}\" saved! Use it from \"Saved Voices\" for fast TTS.",
                    },
                    "saved_voice_library": {
                        "empty_option": "No saved voices - upload a sample first",
                        "error_option": "Error loading voices",
                        "info_ref": "Ref: \"{ref_text}\"",
                        "info_saved": "Saved: {created_at}",
                        "no_selection_delete": "No voice selected to delete.",
                        "delete_confirm": "Delete saved voice \"{voice_name}\"?",
                        "delete_success": "Voice \"{voice_name}\" deleted.",
                        "delete_error": "Failed to delete voice: {error}",
                    },
                    "builtin_tts": {
                        "validation_text": "Please enter some text to synthesize",
                        "start": "Generating speech with {speaker}...",
                        "success": "Speech generated in {duration}s (Speaker: {speaker})",
                        "error": "Generation failed: {error}",
                    },
                    "saved_voice_tts": {
                        "validation_text": "Please enter some text.",
                        "validation_voice": "No saved voice selected. Upload a sample first.",
                        "start": "Generating speech with saved voice...",
                        "progress": "Generating speech... {elapsed}s",
                        "success": "Speech generated in {duration}s",
                        "success_with_audio": "Speech generated in {duration}s ({audio_duration}s audio)",
                        "error": "Generation failed: {error}",
                        "action_busy": "Generating...",
                    },
                    "voice_clone": {
                        "validation_text": "Please enter some text.",
                        "validation_voice_file": "Please select a voice sample file.",
                        "validation_ref_text": "Please enter the reference audio transcript or uncheck the option.",
                        "start_auto_transcribe": "Auto-transcribing reference audio via Qwen3-ASR, then cloning...",
                        "start_manual_ref": "Cloning and generating speech...",
                        "progress_auto_transcribe": "Auto-transcribing + cloning... {elapsed}s",
                        "progress_generate": "Generating voice clone... {elapsed}s",
                        "success": "Voice cloning completed in {duration}s",
                        "success_with_save": "Voice cloning completed in {duration}s (voice \"{name}\" saved for fast reuse)",
                        "error": "Generation failed: {error}",
                        "action_busy": "Processing...",
                    },
                    "voice_design": {
                        "validation_text": "Please enter some text to synthesize.",
                        "validation_description": "Please describe the voice you want.",
                        "start": "Designing voice and generating speech...",
                        "progress": "Designing voice... {elapsed}s",
                        "success": "Voice design completed in {duration}s",
                        "error": "Generation failed: {error}",
                        "action_busy": "Generating...",
                    },
                    "runtime_status": {
                        "unavailable": "Qwen3-TTS Service Unavailable",
                        "online": "Qwen3-TTS Service Online",
                        "device_label": "Device",
                        "model_label": "Model",
                        "not_loaded": "Not Loaded",
                        "unknown": "Unknown",
                        "gpu_suffix": "GPU",
                        "cpu_suffix": "CPU",
                        "gpu_memory_label": "GPU Memory",
                        "speakers_label": "Speakers",
                    },
                },
                "forms": {
                    "builtin_tts": {
                        "fields": {
                            "text": {
                                "label": "Text to synthesize:",
                                "placeholder": "Enter the text you want to convert to speech...",
                                "sample": "Hello! This is a demonstration of Qwen3-TTS neural text-to-speech.",
                            },
                            "language": {"label": "Language:"},
                            "speaker": {"label": "Speaker Voice:"},
                            "instruction": {
                                "label": "Voice Instruction (Optional):",
                                "placeholder": "e.g., Speak slowly and calmly, with a warm tone",
                                "hint": "Describe how the voice should sound. Leave empty for default style.",
                            },
                        },
                        "actions": {
                            "generate": "Generate Speech",
                        },
                    },
                    "voice_clone": {
                        "fields": {
                            "text": {
                                "label": "Text to synthesize:",
                                "placeholder": "Enter the text you want to synthesize...",
                                "sample": "Hello! This is a demonstration of Qwen3-TTS voice cloning.",
                            },
                            "language": {"label": "Output Language:"},
                            "model": {"label": "Model:"},
                            "voice_source": {"label": "Voice Source:"},
                            "saved_voice": {"label": "Select Saved Voice:"},
                            "voice_file": {
                                "label": "Voice Sample Audio:",
                                "drop_text": "Drop audio file here or click to browse",
                                "hint": "Supports: MP3, WAV, M4A, FLAC (recommended: 3-10 seconds, clear speech)",
                            },
                            "save_voice_name": {
                                "label": "Save this voice as (optional):",
                                "placeholder": "e.g., My Voice, John, Customer Support",
                                "hint": "Name this voice to save it for fast reuse. Leave empty to clone without saving.",
                            },
                            "ref_text_toggle": {
                                "label": "Provide reference text manually (auto-transcribed via Qwen3-ASR if unchecked)",
                            },
                            "ref_text": {
                                "label": "Reference Audio Transcript:",
                                "placeholder": "Type the exact words spoken in the voice sample audio...",
                                "hint": "Providing the transcript of the reference audio improves voice cloning quality.",
                            },
                            "voice_description": {
                                "label": "Voice Description:",
                                "placeholder": "Describe the voice you want, e.g.: A deep male voice with a warm, calm British accent and slow speaking pace",
                                "hint": "Describe the characteristics of the voice: gender, pitch, accent, tempo, tone, emotion, etc.",
                            },
                        },
                        "actions": {
                            "refresh_saved": "Refresh",
                            "delete_saved": "Delete",
                            "generate": "Generate Speech",
                            "generate_design": "Design & Generate",
                        },
                    },
                },
                "sections": {
                    "builtin_tts": {
                        "title": "Text-to-Speech (Qwen3-TTS)",
                        "description": "Generate speech using Qwen3-TTS with built-in neural voices. Supports multiple languages with high-quality synthesis.",
                        "text_placeholder": "Enter the text you want to convert to speech...",
                        "text_sample": "Hello! This is a demonstration of Qwen3-TTS neural text-to-speech.",
                        "instruction_placeholder": "e.g., Speak slowly and calmly, with a warm tone",
                        "instruction_hint": "Describe how the voice should sound. Leave empty for default style.",
                    },
                    "cloning": {
                        "title": "Voice Cloning (Qwen3-TTS)",
                        "description": "Upload a voice sample and generate speech in that voice across multiple languages using Qwen3-TTS.",
                        "text_placeholder": "Enter the text you want to synthesize...",
                        "text_sample": "Hello! This is a demonstration of Qwen3-TTS voice cloning.",
                        "saved_source_label": "Saved Voices (fast)",
                        "upload_source_label": "Upload New Sample",
                        "save_voice_placeholder": "e.g., My Voice, John, Customer Support",
                        "ref_text_placeholder": "Type the exact words spoken in the voice sample audio...",
                        "ref_text_hint": "Providing the transcript of the reference audio improves voice cloning quality.",
                        "voice_description_placeholder": "Describe the voice you want, e.g.: A deep male voice with a warm, calm British accent and slow speaking pace",
                        "voice_description_hint": "Describe the characteristics of the voice: gender, pitch, accent, tempo, tone, emotion, etc.",
                        "modes": {
                            "design": {
                                "title": "Voice Design (Qwen3-TTS)",
                                "description": "Describe the voice you want using text and generate speech with that designed voice.",
                                "button_label": "Design & Generate"
                            },
                            "unsupported": {
                                "title": "Voice Cloning (Qwen3-TTS)",
                                "description": "The CustomVoice model uses built-in speakers only and does not support voice cloning or voice design. Switch to a Base model (1.7B or 0.6B) for cloning, or use VoiceDesign for text-described voices.",
                                "button_label": "Generate Speech"
                            },
                            "saved": {
                                "title": "Voice Cloning (Qwen3-TTS)",
                                "description": "Use a saved voice for fast TTS, or upload a new sample to clone.",
                                "button_label": "Generate Speech"
                            }
                        }
                    }
                },
            },
        },
        "whisper": {
            "kind": "stt",
            "display_name": "Whisper (faster-whisper)",
            "short_name": "Whisper STT",
            "internal_url": STT_SERVICE_URL,
            "health_endpoint": "/health",
            # "streaming" is the SSE route /transcribe-stream (a whole file,
            # segments pushed as they decode). "live_transcribe" is the
            # WebSocket route /ws/transcribe (a microphone, partials as you
            # speak). They are different things and only this provider has
            # either — see the gate in /ws/stt.
            "capabilities": [
                "transcribe", "segments", "detect_language",
                "streaming", "live_transcribe", "model_unload",
            ],
            # `language_detect` is the machine-readable truth for API clients.
            # Some backends expose a /detect_language route that always returns
            # null, so route-exists is not the same as capability-exists.
            "language_detect": True,
            "contracts": {
                "transcribe": "stt-form-v1",
                "detect_language": "stt-detect-language-v1",
            },
            "settings": {
                "defaults": {
                    "language": "auto",
                    "enable_segmentation": True,
                },
                "languages": [
                    {"value": "auto", "label": "Auto-Detect"},
                    {"value": "en", "label": "English"},
                    {"value": "de", "label": "German"},
                    {"value": "fr", "label": "French"},
                    {"value": "es", "label": "Spanish"},
                    {"value": "it", "label": "Italian"},
                    {"value": "nl", "label": "Dutch"},
                ],
            },
            "ui": {
                "selectable_as_stt": True,
                "show_status": True,
                "messages": {
                    "transcription": _build_stt_messages(),
                },
            },
        },
        "qwen3-asr": {
            "kind": "stt",
            "display_name": "Qwen3-ASR (multilingual)",
            "short_name": "Qwen3-ASR",
            "internal_url": QWEN3_ASR_SERVICE_URL,
            "health_endpoint": "/health",
            "capabilities": ["transcribe", "segments", "detect_language", "model_unload"],
            "language_detect": True,
            "contracts": {
                "transcribe": "stt-form-v1",
                "detect_language": "stt-detect-language-v1",
            },
            "settings": {
                "defaults": {
                    "language": "auto",
                    "enable_segmentation": True,
                },
                "languages": [
                    {"value": "auto", "label": "Auto-Detect"},
                    {"value": "en", "label": "English"},
                    {"value": "de", "label": "German"},
                    {"value": "fr", "label": "French"},
                    {"value": "es", "label": "Spanish"},
                    {"value": "it", "label": "Italian"},
                    {"value": "nl", "label": "Dutch"},
                ],
            },
            "ui": {
                "selectable_as_stt": True,
                "show_status": True,
                "messages": {
                    "transcription": _build_stt_messages(),
                },
            },
        },
        "piper-training": {
            "kind": "training",
            "display_name": "Piper Training",
            "short_name": "Voice Training",
            "internal_url": VOICE_TRAINING_URL,
            "health_endpoint": "/health",
            "capabilities": ["dataset_preparation", "voice_training", "model_export"],
            "contracts": {
                "training": "voice-training-job-v1",
            },
            "settings": {
                "defaults": {
                    "language": "en",
                    "gender": "female",
                    "epochs": 1000,
                    "batch_size": "32",
                },
                "languages": [
                    {"value": "en", "label": "English"},
                    {"value": "de", "label": "German"},
                    {"value": "fr", "label": "French"},
                    {"value": "es", "label": "Spanish"},
                    {"value": "it", "label": "Italian"},
                    {"value": "nl", "label": "Dutch"},
                ],
                "genders": [
                    {"value": "female", "label": "Female"},
                    {"value": "male", "label": "Male"},
                    {"value": "neutral", "label": "Neutral"},
                ],
                "batch_sizes": [
                    {"value": "16", "label": "16 (Lower memory)"},
                    {"value": "32", "label": "32 (Recommended)"},
                    {"value": "64", "label": "64 (Higher memory)"},
                ],
                "epochs": {
                    "min": 100,
                    "max": 5000,
                    "step": 100,
                },
            },
            "ui": {
                "family": "piper",
                "show_status": True,
                "tab_label": "Voice Training",
                "messages": {
                    "start_training": {
                        "validation_name": "Please enter a voice model name",
                        "validation_files": "Please select training audio files",
                        "start": "Starting VITS training pipeline for {deployment_target}...",
                        "success": "Training started successfully!",
                        "error": "Training failed: {error}",
                        "completed": "Training completed. Deployment target: {deployment_target}.",
                        "failed": "Training failed. Check training jobs for details.",
                        "progress": "Progress: {progress}% (Epoch {current_epoch}/{total_epochs})",
                    },
                    "train_from_dataset": {
                        "validation_name": "Please enter a voice model name",
                        "confirm": "Start training \"{voice_name}\" from the existing prepared dataset (train.json / val.json)?",
                        "start": "Starting training for \"{voice_name}\" from existing dataset...",
                        "success_status": "Training started! Job ID: {job_id}",
                        "success_notification": "Training started for \"{voice_name}\" with target {deployment_target}",
                        "error": "Failed: {error}",
                    },
                    "resume_training": {
                        "validation_name": "Please enter a voice model name",
                        "confirm": "Resume training for voice \"{voice_name}\" from the last checkpoint?",
                        "start_notification": "Resuming training for \"{voice_name}\"...",
                        "success_notification": "Training resumed for \"{voice_name}\" with target {deployment_target}",
                        "success_status": "Resumed training for \"{voice_name}\" — monitoring progress...",
                        "error_status": "Resume failed: {error}",
                        "error_notification": "Resume failed: {error}",
                    },
                    "model_management": {
                        "deploy_start": "Deploying \"{model_name}\" to {deployment_target}...",
                        "deploy_success": "Model \"{model_name}\" deployment status: {status} on {deployment_target}.",
                        "deploy_error": "Export failed: {error}",
                        "download_success": "Model download started",
                        "download_error": "Failed to download model",
                        "delete_confirm": "Delete model \"{job_id}\" and all training data? This cannot be undone.",
                        "delete_success": "Model deleted successfully",
                        "delete_error": "Failed to delete model",
                        "cancel_confirm": "Cancel this training job?",
                        "cancel_success": "Training job cancelled",
                        "cancel_error": "Failed to cancel job",
                    },
                    "model_list": {
                        "loading": "Loading trained models...",
                        "empty": "No trained models found. Start training to create your first model!",
                        "error": "Failed to load models. Is the training service running?",
                        "deploy_action": "Deploy",
                        "download_action": "Download",
                        "delete_action": "Delete",
                    },
                    "job_list": {
                        "loading": "Loading training jobs...",
                        "empty": "No training jobs found.",
                        "error": "Failed to load training jobs.",
                        "details_action": "Details",
                        "resume_action": "Resume",
                        "cancel_action": "Cancel",
                    },
                    "job_details": {
                        "fetch_error": "Failed to fetch job details",
                        "job_label": "Job",
                        "status_label": "Status",
                        "deployment_target_label": "Deployment Target",
                        "progress_label": "Progress",
                        "current_epoch_label": "Current Epoch",
                        "configuration_heading": "Configuration",
                        "epochs_label": "Epochs",
                        "batch_size_label": "Batch Size",
                        "learning_rate_label": "Learning Rate",
                        "best_loss_label": "Best Loss",
                        "recent_logs_heading": "Recent Logs",
                        "na": "N/A",
                    },
                },
                "forms": {
                    "start_training": {
                        "fields": {
                            "voice_name": {
                                "label": "Voice Model Name:",
                                "placeholder": "Enter a unique name for this voice model",
                            },
                            "language": {"label": "Target Language:"},
                            "gender": {"label": "Voice Gender:"},
                            "files": {
                                "label": "Training Audio Files:",
                                "drop_text": "Drop multiple audio files here or click to browse",
                                "hint": "Recommended: 10+ minutes of high-quality speech data",
                            },
                            "epochs": {"label": "Training Epochs:"},
                            "batch_size": {"label": "Batch Size:"},
                            "deployment_target": {"label": "Post-Export Deployment Target:"},
                        },
                        "actions": {
                            "submit": "Start Training",
                        },
                    },
                    "continue_training": {
                        "title": "Continue Existing Training",
                        "description": "If a training job was interrupted (e.g. after a restart) and a checkpoint exists, use <strong>Resume from Checkpoint</strong>. If no checkpoint was saved yet but the dataset (train.json / val.json) is already prepared, use <strong>Train from Dataset</strong>.",
                        "fields": {
                            "voice_name": {
                                "label": "Voice Model Name:",
                                "placeholder": "e.g. luna",
                            },
                            "epochs": {"label": "Epochs:"},
                            "deployment_target": {"label": "Deployment Target:"},
                        },
                        "actions": {
                            "resume": "Resume from Checkpoint",
                            "train_from_dataset": "Train from Dataset",
                        },
                    },
                    "model_management": {
                        "fields": {
                            "deployment_target": {"label": "Manual Deployment Target:"},
                        },
                    },
                },
                "sections": {
                    "training": {
                        "title": "Voice Training",
                        "description": "Train custom voice models using VITS neural networks. Upload multiple audio files with transcripts for high-quality voice cloning.",
                        "voice_name_placeholder": "Enter a unique name for this voice model",
                        "continue_voice_name_placeholder": "e.g. luna",
                        "models_description": "Manage your trained voice models. Export to TTS to use them, download, or delete.",
                    }
                },
            },
        },
    }

    if ENABLE_PARAKEET_ASR:
        providers["parakeet"] = {
            "kind": "stt",
            "display_name": "Parakeet-TDT (realtime, 25 EU langs)",
            "short_name": "Parakeet ASR",
            "internal_url": PARAKEET_ASR_SERVICE_URL,
            "health_endpoint": "/health",
            # /detect_language exists but is a stub that always returns null,
            # so the capability is NOT declared. Parakeet auto-detects
            # internally during transcription but does not report it.
            "capabilities": ["transcribe", "segments", "model_unload"],
            "language_detect": False,
            "contracts": {
                "transcribe": "stt-form-v1",
            },
            "settings": {
                "defaults": {
                    "language": "auto",
                    "enable_segmentation": True,
                },
                "languages": [
                    {"value": "auto", "label": "Auto-Detect"},
                    {"value": "en", "label": "English"},
                    {"value": "de", "label": "German"},
                    {"value": "fr", "label": "French"},
                    {"value": "es", "label": "Spanish"},
                    {"value": "it", "label": "Italian"},
                    {"value": "nl", "label": "Dutch"},
                ],
            },
            "ui": {
                "selectable_as_stt": True,
                "show_status": True,
                "messages": {
                    "transcription": _build_stt_messages(),
                },
            },
        }

    if ENABLE_CANARY_ASR:
        providers["canary"] = {
            "kind": "stt",
            "display_name": "Canary-180M (realtime, en/de/es/fr)",
            "short_name": "Canary ASR",
            "internal_url": CANARY_ASR_SERVICE_URL,
            "health_endpoint": "/health",
            "capabilities": ["transcribe", "segments", "model_unload"],
            # Canary has no language identification at all — its /detect_language
            # route transcribes with the default language and returns null. The
            # contract is therefore NOT declared: a client must not be told it
            # can detect language here.
            "language_detect": False,
            "contracts": {
                "transcribe": "stt-form-v1",
            },
            "settings": {
                "defaults": {
                    "language": "de",
                    "enable_segmentation": True,
                },
                # Canary has no auto-detection — the language picks the decoder
                "languages": [
                    {"value": "de", "label": "German"},
                    {"value": "en", "label": "English"},
                    {"value": "fr", "label": "French"},
                    {"value": "es", "label": "Spanish"},
                ],
            },
            "ui": {
                "selectable_as_stt": True,
                "show_status": True,
                "messages": {
                    "transcription": _build_stt_messages(),
                },
            },
        }

    if ENABLE_CHATTERBOX_TTS:
        providers["chatterbox"] = {
            "kind": "tts",
            "display_name": "Chatterbox (Multilingual, MIT)",
            "short_name": "Chatterbox",
            "internal_url": CHATTERBOX_TTS_SERVICE_URL,
            "health_endpoint": "/health",
            "capabilities": ["tts", "voice_clone", "tts_stream", "model_unload"],
            "contracts": {
                "tts": "simple-json-tts-v1",
                "voice_clone": "voice-clone-tts-v1",
                # Sentence-chunked streaming: audio starts after the first
                # chunk instead of after the whole text. The gateway routes
                # here automatically when the contract key is present.
                "tts_stream": "chunked-wav-stream-v1",
            },
            "settings": {
                "defaults": {
                    "language": "de",
                },
                "languages": [
                    {"value": "de", "label": "German"},
                    {"value": "en", "label": "English"},
                    {"value": "fr", "label": "French"},
                    {"value": "es", "label": "Spanish"},
                    {"value": "it", "label": "Italian"},
                    {"value": "nl", "label": "Dutch"},
                    {"value": "pt", "label": "Portuguese"},
                    {"value": "pl", "label": "Polish"},
                ],
            },
            "ui": {
                # Uses the generic TTS panel (same family as Piper)
                "family": "piper",
                "selectable_as_engine": True,
                "show_status": True,
                "tab_label": "Text-to-Speech",
                "messages": {
                    "tts_generation": _build_basic_tts_messages(),
                },
                "sections": {
                    "tts": {
                        "title": "Text-to-Speech (Chatterbox)",
                        "description": "Generate multilingual speech with Resemble AI Chatterbox. Output is watermarked. Voice cloning is available via the API (/clone).",
                        "text_placeholder": "Enter the text you want to convert to speech...",
                        "text_sample": "Hallo! Dies ist ein Test von Chatterbox Multilingual.",
                    }
                },
            },
        }

    if ENABLE_WHISPER_CPP:
        providers["whisper-cpp"] = {
            "kind": "stt",
            "display_name": "whisper.cpp (OpenAI-compatible)",
            "short_name": "whisper.cpp",
            "internal_url": WHISPER_CPP_SERVICE_URL,
            "health_endpoint": "/",
            "capabilities": ["transcribe", "openai_compatible"],
            # whisper.cpp auto-detects when language=auto is sent (which the
            # gateway now always does), but exposes no separate LID route.
            "language_detect": True,
            "contracts": {
                "transcribe": "openai-audio-transcriptions-v1",
            },
            # whisper-server (current whisper.cpp) only serves its native
            # /inference route; it accepts the same form fields as the OpenAI
            # endpoint and returns {"text": ...}.
            "transcribe_path": "/inference",
            "settings": {
                "defaults": {
                    "language": "auto",
                    "enable_segmentation": False,
                },
                "languages": [
                    {"value": "auto", "label": "Auto-Detect"},
                    {"value": "en", "label": "English"},
                    {"value": "de", "label": "German"},
                    {"value": "fr", "label": "French"},
                    {"value": "es", "label": "Spanish"},
                    {"value": "it", "label": "Italian"},
                    {"value": "nl", "label": "Dutch"},
                ],
            },
            "ui": {
                "selectable_as_stt": True,
                "show_status": True,
                "messages": {
                    "transcription": _build_stt_messages(),
                },
            },
        }

    registry = {
        "providers": providers,
        "ui": {
            "default_tts_provider": os.getenv("DEFAULT_TTS_PROVIDER", "piper"),
            "default_stt_provider": os.getenv("DEFAULT_STT_PROVIDER", "whisper"),
            "training_provider": os.getenv("TRAINING_PROVIDER", "piper-training"),
            "enable_whisper_cpp": ENABLE_WHISPER_CPP,
            "enable_parakeet_asr": ENABLE_PARAKEET_ASR,
            "enable_canary_asr": ENABLE_CANARY_ASR,
            "enable_chatterbox_tts": ENABLE_CHATTERBOX_TTS,
            "copy": {
                "app_subtitle": "Neural Text-to-Speech with Voice Training & Cloning + Speech-to-Text",
                "stt_tab_label": "Speech-to-Text",
                "stt_title": "Speech-to-Text",
                "stt_description": "Convert speech to text. Supports transcription with optional audio segmentation for training data preparation.",
            },
        },
    }

    override = os.getenv("PROVIDER_REGISTRY_JSON", "").strip()
    if override:
        parsed = json.loads(override)
        registry["providers"].update(parsed.get("providers", {}))
        registry["ui"].update(parsed.get("ui", {}))

    return registry


PROVIDER_REGISTRY = _build_provider_registry()


def _template_provider_lists() -> tuple[list[tuple[str, dict]], list[tuple[str, dict]], list[tuple[str, dict]]]:
    """Return provider lists used to render the UI template."""
    providers = PROVIDER_REGISTRY["providers"]
    tts_providers = [
        (provider_id, provider)
        for provider_id, provider in providers.items()
        if provider.get("kind") == "tts" and provider.get("ui", {}).get("selectable_as_engine")
    ]
    stt_providers = [
        (provider_id, provider)
        for provider_id, provider in providers.items()
        if provider.get("kind") == "stt" and provider.get("ui", {}).get("selectable_as_stt")
    ]
    status_providers = [
        (provider_id, provider)
        for provider_id, provider in providers.items()
        if provider.get("ui", {}).get("show_status")
    ]
    return tts_providers, stt_providers, status_providers


def _get_provider(provider_id: str, kind: Optional[str] = None) -> dict:
    """Return a registered provider and validate its kind when requested."""
    provider = PROVIDER_REGISTRY["providers"].get(provider_id)
    if not provider:
        raise HTTPException(status_code=404, detail=f"Unknown provider: {provider_id}")
    if kind and provider.get("kind") != kind:
        raise HTTPException(status_code=400, detail=f"Provider {provider_id} is not a {kind} provider")
    return provider


def _normalize_qwen3_language(language: str) -> str:
    """Map common language codes to the English labels expected by Qwen3-TTS."""
    language_map = {
        "auto": "English",
        "en": "English",
        "en_us": "English",
        "en_gb": "English",
        "de": "German",
        "de_de": "German",
        "fr": "French",
        "fr_fr": "French",
        "es": "Spanish",
        "es_es": "Spanish",
        "it": "Italian",
        "it_it": "Italian",
        "pt": "Portuguese",
        "ru": "Russian",
        "ja": "Japanese",
        "ko": "Korean",
        "zh": "Chinese",
        "zh_cn": "Chinese",
        # Qwen3-TTS has no Dutch support; fall back to English rather than erroring
        "nl": "English",
    }
    normalized = (language or "English").strip().lower().replace("-", "_")
    return language_map.get(normalized, language if language and language[:1].isupper() else "English")


def _normalize_piper_voice_catalog(payload: dict) -> list[dict]:
    """Convert the Piper /voices response into a normalized voice list."""
    voices = payload.get("voices", {})
    normalized = []
    for voice_id, voice in voices.items():
        normalized.append({
            "id": voice_id,
            "name": voice.get("name") or voice.get("speaker") or voice_id,
            "language": voice.get("language"),
            "description": voice.get("quality"),
            "kind": voice.get("model_type", "default"),
            "raw": voice,
        })
    return normalized


def _normalize_qwen3_voice_catalog(payload: dict) -> list[dict]:
    """Convert the Qwen3 speaker list into a normalized voice list."""
    speakers = payload.get("speakers", [])
    languages = payload.get("languages", [])
    normalized = []
    for speaker in speakers:
        normalized.append({
            "id": speaker,
            "name": speaker,
            "language": "multilingual",
            "description": f"Built-in speaker. Languages: {', '.join(languages[:4])}{'...' if len(languages) > 4 else ''}",
            "kind": "builtin",
            "raw": {"speaker": speaker, "languages": languages},
        })
    return normalized


def _normalize_qwen3_model_catalog(payload: dict) -> list[dict]:
    """Convert the Qwen3 model list into a normalized model catalog."""
    models = payload.get("models", {})
    current_model = payload.get("current_model")
    normalized = []
    for model_id, info in models.items():
        capabilities = info.get("capabilities", [])
        normalized.append({
            "id": model_id,
            "name": info.get("name", model_id),
            "description": info.get("description", ""),
            "capabilities": capabilities,
            "capabilities_text": ", ".join(capabilities),
            "is_current": model_id == current_model,
            "raw": info,
        })
    return normalized


def _normalize_qwen3_runtime_status(payload: dict, provider_id: str) -> dict:
    """Convert provider-native Qwen3 runtime data into a stable frontend status shape."""
    model_info = payload.get("current_model_info") or {}
    gpu_memory_allocated = payload.get("gpu_memory_allocated")
    gpu_memory_gb = None
    if gpu_memory_allocated not in (None, ""):
        try:
            gpu_memory_gb = round(float(gpu_memory_allocated) / (1024 ** 3), 1)
        except (TypeError, ValueError):
            gpu_memory_gb = None

    speakers = payload.get("builtin_speakers")
    if not isinstance(speakers, list):
        speakers = []

    return {
        "provider": provider_id,
        "device_name": str(payload.get("device") or "").upper() or None,
        "device_type": "gpu" if payload.get("cuda_available") else "cpu",
        "model_loaded": bool(payload.get("model_loaded")),
        "model_name": model_info.get("name") or payload.get("current_model") or None,
        "gpu_memory_gb": gpu_memory_gb,
        "speakers": speakers,
    }


def _truncate_text(value: Optional[str], limit: int = 80) -> Optional[str]:
    """Return a compact preview string for longer free-form text fields."""
    text = str(value or "").strip()
    if not text:
        return None
    if len(text) <= limit:
        return text
    return f"{text[:limit].rstrip()}..."


def _format_frontend_timestamp(value: Optional[str]) -> Optional[str]:
    """Convert upstream timestamps into a stable, human-readable label."""
    if value in (None, ""):
        return None

    text = str(value).strip()
    if not text:
        return None

    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return text

    suffix = ""
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(timezone.utc)
        suffix = " UTC"

    return f"{parsed.strftime('%Y-%m-%d %H:%M')}{suffix}"


def _normalize_training_target_label(target_id: Optional[str]) -> str:
    """Map deployment target ids to stable browser-facing labels."""
    labels = {
        "none": "Manual download only",
        "piper-volume": "Piper shared volume",
        "piper-http": "Piper upload API",
    }
    if not target_id:
        return "Default"
    return labels.get(target_id, str(target_id).replace("-", " ").title())


def _normalize_qwen3_saved_voice_library(payload: dict, provider_id: str) -> dict:
    """Normalize saved-voice library entries for browser consumers."""
    voices_payload = payload.get("voices") if isinstance(payload, dict) else []
    normalized_voices = []

    if isinstance(voices_payload, list):
        for voice in voices_payload:
            if not isinstance(voice, dict):
                continue

            ref_text = voice.get("ref_text") or voice.get("reference_text")
            created_at = voice.get("created_at")
            normalized_voice = dict(voice)
            normalized_voice.update({
                "id": voice.get("id") or voice.get("voice_id") or voice.get("name"),
                "name": voice.get("name") or voice.get("id") or voice.get("voice_id"),
                "language": voice.get("language") or voice.get("lang"),
                "reference_text": ref_text or None,
                "reference_preview": _truncate_text(ref_text, 80),
                "created_at": created_at or None,
                "created_at_display": _format_frontend_timestamp(created_at),
            })
            normalized_voices.append(normalized_voice)

    return {
        "provider": provider_id,
        "voices": normalized_voices,
    }


def _normalize_training_logs(logs_payload) -> list[dict]:
    """Normalize training log entries into a stable text-oriented structure."""
    normalized_logs = []
    if not isinstance(logs_payload, list):
        return normalized_logs

    for entry in logs_payload[-5:]:
        if isinstance(entry, dict):
            timestamp = entry.get("timestamp") or entry.get("time")
            message = entry.get("message") or entry.get("text") or json.dumps(entry)
        else:
            timestamp = None
            message = str(entry)

        timestamp_display = _format_frontend_timestamp(timestamp)
        display = f"{timestamp_display}: {message}" if timestamp_display else message
        normalized_logs.append({
            "timestamp": timestamp or None,
            "timestamp_display": timestamp_display,
            "message": message,
            "display": display,
        })

    return normalized_logs


def _normalize_training_job(payload: dict) -> dict:
    """Normalize training job payloads for job tables and detail views."""
    if not isinstance(payload, dict):
        return {}

    config = payload.get("config") if isinstance(payload.get("config"), dict) else {}
    created_at = payload.get("created_at") or config.get("created_at")
    voice_name = (
        payload.get("voice_name")
        or payload.get("model_name")
        or config.get("voice_name")
        or config.get("speaker_name")
        or payload.get("job_id")
    )
    deployment_target = payload.get("deployment_target")
    best_loss = payload.get("best_loss")
    if best_loss in (None, ""):
        best_loss = payload.get("loss")

    try:
        normalized_best_loss = float(best_loss) if best_loss not in (None, "") else None
    except (TypeError, ValueError):
        normalized_best_loss = None

    config_summary = {
        "epochs": config.get("epochs") or payload.get("total_epochs"),
        "batch_size": config.get("batch_size"),
        "learning_rate": config.get("learning_rate"),
    }

    normalized_job = dict(payload)
    normalized_job.update({
        "voice_name": voice_name,
        "model_name": payload.get("model_name") or voice_name,
        "deployment_target_label": _normalize_training_target_label(deployment_target),
        "created_at": created_at or None,
        "created_at_display": _format_frontend_timestamp(created_at) or "N/A",
        "config_summary": config_summary,
        "best_loss": normalized_best_loss,
        "best_loss_display": f"{normalized_best_loss:.4f}" if normalized_best_loss is not None else None,
        "recent_logs": _normalize_training_logs(payload.get("logs")),
    })
    return normalized_job


def _normalize_training_jobs_payload(payload):
    """Normalize training job list responses while preserving top-level shape."""
    if isinstance(payload, list):
        return [_normalize_training_job(job) for job in payload if isinstance(job, dict)]

    if isinstance(payload, dict) and isinstance(payload.get("jobs"), list):
        normalized_payload = dict(payload)
        normalized_payload["jobs"] = [
            _normalize_training_job(job)
            for job in payload.get("jobs", [])
            if isinstance(job, dict)
        ]
        return normalized_payload

    return payload


def _normalize_training_export_response(payload: dict) -> dict:
    """Add stable deployment labels to training export responses."""
    if not isinstance(payload, dict):
        return payload

    normalized_payload = dict(payload)
    deployment = normalized_payload.get("deployment")
    if isinstance(deployment, dict):
        target = deployment.get("target")
        normalized_payload["deployment"] = {
            **deployment,
            "target_label": _normalize_training_target_label(target),
        }
    return normalized_payload


def _build_error_from_response(response: httpx.Response) -> HTTPException:
    """Convert an upstream HTTP error into a frontend HTTPException."""
    detail = response.text
    try:
        payload = response.json()
        detail = payload.get("detail") or payload
    except Exception:
        pass
    return HTTPException(status_code=response.status_code, detail=detail)


def _build_upstream_request_error(service_name: str, exc: httpx.RequestError) -> HTTPException:
    """Convert an upstream transport failure into a 503 frontend HTTPException."""
    request_url = getattr(getattr(exc, "request", None), "url", None)
    detail = f"{service_name} is unavailable"
    if request_url:
        detail = f"{service_name} is unavailable: {request_url}"
    return HTTPException(status_code=503, detail=detail)


def _passthrough_headers(response: httpx.Response) -> dict:
    """Return a filtered set of upstream headers safe to forward.

    Content-Length is intentionally NOT forwarded: adapters often re-serialize
    the body (e.g. normalized STT JSON), and a stale upstream length makes
    uvicorn fail with "Response content longer than Content-Length". Starlette
    recomputes the correct length from the actual body.
    """
    keep = {"content-type", "content-disposition"}
    return {
        key: value
        for key, value in response.headers.items()
        if key.lower() in keep or key.lower().startswith("x-")
    }


async def _extract_form_payload(request: Request) -> tuple[dict, Optional[list[tuple[str, tuple[str, bytes, str]]]]]:
    """Normalize a Starlette form request into httpx-compatible data/files payloads.

    Text fields are returned as a dict (repeated keys become lists): httpx
    multipart encoding only accepts dict-shaped ``data`` — passing a list of
    tuples makes AsyncClient raise "Attempted to send an sync request".
    """
    form = await request.form()

    data: dict = {}
    files: list[tuple[str, tuple[str, bytes, str]]] = []
    for key, value in form.multi_items():
        if hasattr(value, "filename"):
            content = await value.read()
            files.append((
                key,
                (value.filename or "upload.bin", content, value.content_type or "application/octet-stream"),
            ))
        else:
            text = str(value)
            if key in data:
                existing = data[key]
                if isinstance(existing, list):
                    existing.append(text)
                else:
                    data[key] = [existing, text]
            else:
                data[key] = text

    return data, files or None


# Shared, connection-pooled HTTP client. Reused across requests so we don't pay
# pool/connection setup on every proxied call. Recreated transparently if
# httpx.AsyncClient is swapped out (e.g. patched in unit tests).
_http_client: Optional[httpx.AsyncClient] = None
_http_client_factory = None

# Upstream uvicorn workers are started with --timeout-keep-alive 120; the client
# expiry must stay <= that, or the pool hands out sockets the server has already
# closed and every such request pays a silent retry.
_KEEPALIVE_EXPIRY_S = float(os.getenv("UPSTREAM_KEEPALIVE_EXPIRY", "115"))
_POOL_LIMITS = httpx.Limits(
    max_connections=int(os.getenv("UPSTREAM_MAX_CONNECTIONS", "100")),
    max_keepalive_connections=int(os.getenv("UPSTREAM_MAX_KEEPALIVE", "20")),
    keepalive_expiry=_KEEPALIVE_EXPIRY_S,
)


# Health probes must stay well under the UI's refresh cadence; a slow backend
# should show as unhealthy quickly rather than stalling the status row.
PROVIDER_HEALTH_TIMEOUT_S = float(os.getenv("PROVIDER_HEALTH_TIMEOUT", "6"))


def _timeout(read: float) -> httpx.Timeout:
    """Per-phase timeouts.

    A bare float applies the *whole* budget to each phase including connect, so a
    600 s synthesis budget also meant a 600 s wait for a dead upstream. Connect
    should fail fast; only the read phase needs the long budget.

    Every upstream call in this module goes through here. Passing a bare float
    to httpx directly is the bug this function exists to prevent, and it was
    still doing exactly that on nine call sites — including voice cloning and
    saved-voice TTS at 600 s, where stopping the qwen3 container left the
    browser's spinner running for ten minutes before the "unavailable" arrived.
    """
    return httpx.Timeout(connect=3.0, read=read, write=120.0, pool=10.0)


def _get_http_client() -> httpx.AsyncClient:
    """Return the process-wide pooled AsyncClient (per-call timeouts are passed explicitly)."""
    global _http_client, _http_client_factory
    if _http_client is None or _http_client_factory is not httpx.AsyncClient:
        _http_client = httpx.AsyncClient(limits=_POOL_LIMITS)
        _http_client_factory = httpx.AsyncClient
    return _http_client


async def _stream_upstream(
    method: str,
    url: str,
    *,
    display_name: str,
    extra_headers: Optional[dict] = None,
    read_timeout: float = 600.0,
    **request_kwargs,
) -> StreamingResponse:
    """Proxy an upstream response body through without buffering it.

    Reading `response.content` here would defeat the whole point of the
    providers' sentence-streaming endpoints: the backend would stream, and the
    gateway would sit on the bytes until the last one arrived. Instead the
    upstream response is opened in streaming mode and its raw chunks are handed
    straight to the client as they land.
    """
    client = _get_http_client()
    req = client.build_request(method, url, timeout=_timeout(read_timeout), **request_kwargs)
    try:
        upstream = await client.send(req, stream=True)
    except httpx.RequestError as exc:
        raise _build_upstream_request_error(display_name, exc) from exc

    if upstream.status_code >= 400:
        # Error bodies are small: read it, close the connection, and report.
        # The read itself can fail mid-body (upstream died while sending its
        # error); that must still surface as a 502 with the provider named,
        # not as an unhandled 500 from a raw httpx exception.
        try:
            await upstream.aread()
        except httpx.RequestError as exc:
            await upstream.aclose()
            raise _build_upstream_request_error(display_name, exc) from exc
        finally:
            await upstream.aclose()
        raise _build_error_from_response(upstream)

    headers = {k: v for k, v in upstream.headers.items() if k.lower().startswith("x-")}
    if extra_headers:
        headers.update(extra_headers)
    # Tell any intermediary not to buffer this, which would re-introduce the
    # exact latency the streaming path exists to remove.
    headers["Cache-Control"] = "no-cache"
    headers["X-Accel-Buffering"] = "no"

    return StreamingResponse(
        upstream.aiter_raw(),
        status_code=upstream.status_code,
        media_type=upstream.headers.get("content-type", "audio/wav"),
        headers=headers,
        # Mandatory: the client is process-wide, so without this every streamed
        # request leaks its pooled connection.
        background=BackgroundTask(upstream.aclose),
    )


async def _provider_get(provider_id: str, path: str, timeout: float = 30.0) -> httpx.Response:
    """Run a GET against a registered provider's internal URL."""
    provider = _get_provider(provider_id)
    client = _get_http_client()
    try:
        response = await client.get(f"{provider['internal_url']}{path}", timeout=_timeout(timeout))
    except httpx.RequestError as exc:
        raise _build_upstream_request_error(provider.get("display_name", provider_id), exc) from exc
    if response.status_code >= 400:
        raise _build_error_from_response(response)
    return response


async def _provider_delete(provider_id: str, path: str, timeout: float = 30.0) -> httpx.Response:
    """Run a DELETE against a registered provider's internal URL."""
    provider = _get_provider(provider_id)
    client = _get_http_client()
    try:
        response = await client.delete(f"{provider['internal_url']}{path}", timeout=_timeout(timeout))
    except httpx.RequestError as exc:
        raise _build_upstream_request_error(provider.get("display_name", provider_id), exc) from exc
    if response.status_code >= 400:
        raise _build_error_from_response(response)
    return response


async def _provider_json_post(provider_id: str, path: str, payload: dict, timeout: float = 120.0) -> httpx.Response:
    """Run a JSON POST against a registered provider's internal URL."""
    provider = _get_provider(provider_id)
    client = _get_http_client()
    try:
        response = await client.post(f"{provider['internal_url']}{path}", json=payload, timeout=_timeout(timeout))
    except httpx.RequestError as exc:
        raise _build_upstream_request_error(provider.get("display_name", provider_id), exc) from exc
    if response.status_code >= 400:
        raise _build_error_from_response(response)
    return response


async def _provider_form_post_raw(provider_id: str, path: str, *, data: dict,
                                  files: list, timeout: float = 300.0) -> httpx.Response:
    """Multipart POST with an already-built payload.

    `_provider_form_post` re-parses the incoming Request, which suits the
    `/api/*` adapters that forward a browser form verbatim. The `/v1` router has
    already translated the request into OpenAI-shaped fields, so it needs to
    hand the payload over directly instead.
    """
    provider = _get_provider(provider_id)
    client = _get_http_client()
    try:
        request_kwargs = {"data": data, "timeout": _timeout(timeout)}
        if files:
            request_kwargs["files"] = files
        response = await client.post(f"{provider['internal_url']}{path}", **request_kwargs)
    except httpx.RequestError as exc:
        raise _build_upstream_request_error(provider.get("display_name", provider_id), exc) from exc
    if response.status_code >= 400:
        raise _build_error_from_response(response)
    return response


async def _provider_form_post(provider_id: str, path: str, request: Request, timeout: float = 300.0) -> httpx.Response:
    """Run a multipart form POST against a registered provider's internal URL."""
    provider = _get_provider(provider_id)
    data, files = await _extract_form_payload(request)
    client = _get_http_client()

    try:
        request_kwargs = {"data": data, "timeout": _timeout(timeout)}
        if files:
            request_kwargs["files"] = files
        response = await client.post(f"{provider['internal_url']}{path}", **request_kwargs)
    except httpx.RequestError as exc:
        raise _build_upstream_request_error(provider.get("display_name", provider_id), exc) from exc
    if response.status_code >= 400:
        raise _build_error_from_response(response)
    return response


async def _proxy_training_get(path: str, timeout: float = 30.0):
    """Proxy a GET request to the training service."""
    provider = _get_provider("piper-training", kind="training")
    client = _get_http_client()
    try:
        response = await client.get(f"{provider['internal_url']}{path}", timeout=_timeout(timeout))
    except httpx.RequestError as exc:
        raise _build_upstream_request_error(provider.get("display_name", "training service"), exc) from exc
    if response.status_code >= 400:
        raise _build_error_from_response(response)
    return response


async def _proxy_training_delete(path: str, timeout: float = 30.0):
    """Proxy a DELETE request to the training service."""
    provider = _get_provider("piper-training", kind="training")
    client = _get_http_client()
    try:
        response = await client.delete(f"{provider['internal_url']}{path}", timeout=_timeout(timeout))
    except httpx.RequestError as exc:
        raise _build_upstream_request_error(provider.get("display_name", "training service"), exc) from exc
    if response.status_code >= 400:
        raise _build_error_from_response(response)
    return response


async def _proxy_training_form_post(path: str, request: Request, timeout: float = 300.0):
    """Proxy a multipart form POST request to the training service."""
    provider = _get_provider("piper-training", kind="training")
    data, files = await _extract_form_payload(request)
    client = _get_http_client()

    try:
        request_kwargs = {"data": data, "timeout": _timeout(timeout)}
        if files:
            request_kwargs["files"] = files
        response = await client.post(f"{provider['internal_url']}{path}", **request_kwargs)
    except httpx.RequestError as exc:
        raise _build_upstream_request_error(provider.get("display_name", "training service"), exc) from exc
    if response.status_code >= 400:
        raise _build_error_from_response(response)
    return response


class FrontendTTSRequest(BaseModel):
    """Normalized text-to-speech request accepted by the frontend adapter."""

    provider: str
    text: str
    voice: Optional[str] = None
    language: str = "auto"
    quality: Optional[str] = None
    gender: Optional[str] = None
    speed: Optional[float] = None
    instructions: Optional[str] = None
    output_format: str = "wav"


def _build_tts_payload(
    provider_id: str,
    provider: dict,
    *,
    text: str,
    voice: Optional[str] = None,
    language: str = "auto",
    quality: Optional[str] = None,
    gender: Optional[str] = None,
    speed: Optional[float] = None,
    instructions: Optional[str] = None,
    output_format: str = "wav",
) -> tuple[dict, float]:
    """Translate a normalized TTS request into one provider's native body.

    Shared by `/api/tts` and the OpenAI-compatible `/v1/audio/speech`, and that
    sharing is the point: `/v1` used to build Piper's body unconditionally, so
    on a deployment with DEFAULT_TTS_PROVIDER=qwen3 it sent `language` and
    `voice` to a service whose fields are `lang` and `speaker`. Pydantic ignores
    unknown keys, so both were dropped in silence and every request came back as
    the default speaker reading German text in English — HTTP 200, wrong audio,
    no signal. The same failure mode the whisper.cpp language default had.

    Returns ``(payload, read_timeout_seconds)``.
    """
    contract = provider.get("contracts", {}).get("tts")
    if contract != "simple-json-tts-v1":
        raise HTTPException(
            status_code=400, detail=f"Unsupported TTS contract for provider {provider_id}"
        )

    if provider_id == "piper":
        payload = {
            "text": text,
            "output_format": output_format,
            "speed": speed if speed is not None else 1.0,
        }
        if voice:
            payload["voice"] = voice
        if language and language != "auto":
            payload["language"] = language
        if quality:
            payload["quality"] = quality
        if gender:
            payload["gender"] = gender
        return payload, 120.0

    if provider_id == "chatterbox":
        # No speed control in the chatterbox API; "auto" resolves to
        # CHATTERBOX_DEFAULT_LANGUAGE service-side.
        return {"text": text, "language": language or "auto"}, 300.0

    if provider_id == "qwen3":
        return {
            "text": text,
            "lang": _normalize_qwen3_language(language),
            "speaker": voice or "Vivian",
            "instruct": instructions or "",
        }, 120.0

    raise HTTPException(
        status_code=400, detail=f"Unsupported TTS contract for provider {provider_id}"
    )


class ProviderModelSelectionRequest(BaseModel):
    """Request body for selecting a provider model variant."""

    model: str


class ProviderVoiceDesignRequest(BaseModel):
    """Request body for provider-scoped voice design."""

    text: str
    voice_description: str
    lang: str = "English"


async def _build_frontend_stt_payload(provider_id: str, form, contract: str) -> tuple[str, dict, list[tuple[str, tuple[str, bytes, str]]]]:
    """Translate normalized frontend STT form data into a provider-specific request.

    ``data`` is dict-shaped because httpx multipart encoding rejects sequence
    payloads when files are present.
    """
    audio = form.get("audio")
    if not hasattr(audio, "filename"):
        raise HTTPException(status_code=400, detail="Audio file not provided")

    filename = audio.filename or "audio.bin"
    content_type = audio.content_type or "application/octet-stream"
    content = await audio.read()
    language = str(form.get("language", "auto")).strip()

    data: dict = {}
    if contract == "stt-form-v1":
        files = [("audio", (filename, content, content_type))]
        if language and language != "auto":
            data["language"] = language
        return "/transcribe", data, files

    if contract == "openai-audio-transcriptions-v1":
        files = [("file", (filename, content, content_type))]
        data["response_format"] = "json"
        # Send "auto" EXPLICITLY rather than omitting the field. whisper.cpp's
        # server defaults to `std::string language = "en"` and only overrides it
        # when the form field is present, so omitting it here meant "auto"
        # silently transcribed German audio as English on every whisper-cpp
        # deployment. "auto" is an explicitly supported value there
        # (`-l LANG ... 'auto' for auto-detect`, and it is special-cased in the
        # server's language validation), so this restores real auto-detection.
        data["language"] = language if language else "auto"
        backend_path = _get_provider(provider_id).get("transcribe_path") or "/v1/audio/transcriptions"
        return backend_path, data, files

    raise HTTPException(status_code=400, detail=f"Unsupported STT contract for provider {provider_id}")


def _normalize_frontend_stt_response(payload: dict, contract: str) -> dict:
    """Normalize provider transcription payloads into the shared browser-facing STT shape."""
    segments_payload = payload.get("segments") if isinstance(payload, dict) else None
    segments = []
    if isinstance(segments_payload, list):
        for segment in segments_payload:
            if not isinstance(segment, dict):
                continue
            segments.append({
                "start": float(segment.get("start", 0) or 0),
                "end": float(segment.get("end", 0) or 0),
                "text": segment.get("text", "") or "",
            })

    if contract == "openai-audio-transcriptions-v1":
        text = payload.get("text") or payload.get("transcript") or ""
    else:
        text = payload.get("text") or ""

    if not text and segments:
        text = " ".join(segment["text"] for segment in segments).strip()

    language = payload.get("language")
    duration = payload.get("duration")
    normalized_duration = float(duration) if duration not in (None, "") else None

    return {
        "text": text,
        "segments": segments,
        "language": language or None,
        "duration": normalized_duration,
    }


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Render the main web UI page with service URLs injected into the template."""
    tts_providers, stt_providers, status_providers = _template_provider_lists()
    # Request-first signature. The legacy TemplateResponse(name, context) form
    # is not merely deprecated in Starlette 1.x, it is gone: the name slot takes
    # the request, so the context dict lands where the template name belongs and
    # the loader raises "unhashable type: 'dict'". Supported since Starlette
    # 0.29, so this works on the pinned 0.115.8 and on anything newer.
    return templates.TemplateResponse(request, "index.html", {
        "provider_registry_json": json.dumps(PROVIDER_REGISTRY),
        "tts_provider_options": tts_providers,
        "stt_provider_options": stt_providers,
        "status_provider_options": status_providers,
        "default_tts_provider": PROVIDER_REGISTRY["ui"]["default_tts_provider"],
        "default_stt_provider": PROVIDER_REGISTRY["ui"]["default_stt_provider"],
        "app_version": APP_VERSION,
    })


@app.get("/api-docs", response_class=HTMLResponse)
async def api_docs():
    """Serve the interactive API documentation page."""
    with open(BASE_DIR / "static" / "api_docs.html", "r", encoding="utf-8") as f:
        content = f.read()
    return HTMLResponse(content=content)


@app.get("/health")
async def health():
    """Return service health status and configured backend URLs.

    Kept lightweight: Docker probes this every 30s, so it returns only the
    small service map. The full provider registry is available at /providers.
    """
    return {
        "status": "healthy",
        "services": {
            provider_id: provider["internal_url"]
            for provider_id, provider in PROVIDER_REGISTRY["providers"].items()
        },
    }


@app.get("/providers")
async def providers():
    """Return the UI provider registry and provider contracts."""
    return PROVIDER_REGISTRY


@app.get("/api/health")
async def provider_health():
    """Probe every provider's health endpoint concurrently.

    The browser used to do this itself, one cross-origin request per provider,
    sequentially — so a single unreachable backend stalled the whole status row
    for its full timeout, and every backend port had to be reachable from the
    browser just to render an indicator.

    Doing it here means the browser makes one same-origin call, the probes run in
    parallel, and the stack no longer needs its backend ports published.
    """
    client = _get_http_client()
    providers_map = PROVIDER_REGISTRY.get("providers", {})

    async def probe(provider_id: str, provider: dict) -> tuple[str, dict]:
        url = f"{provider['internal_url']}{provider.get('health_endpoint', '/health')}"
        started = time.monotonic()
        try:
            response = await client.get(url, timeout=_timeout(PROVIDER_HEALTH_TIMEOUT_S))
            body = {}
            try:
                body = response.json()
            except Exception:
                pass
            return provider_id, {
                "healthy": response.status_code < 400,
                "status_code": response.status_code,
                "latency_ms": round((time.monotonic() - started) * 1000, 1),
                # Surfacing these lets the UI show *why* a service is not ready
                # instead of a bare red dot.
                "model_loaded": body.get("model_loaded"),
                # A service can be healthy with no model in memory once idle
                # unloading is on. "idle" is not "down" — the next request
                # loads it. Absent on providers that do not report residency.
                "model_resident": body.get("model_resident"),
                "model_size": body.get("model_size") or body.get("current_model"),
                "device": body.get("device"),
            }
        except Exception as exc:
            return provider_id, {
                "healthy": False,
                "status_code": None,
                "latency_ms": round((time.monotonic() - started) * 1000, 1),
                "error": type(exc).__name__,
            }

    results = await asyncio.gather(
        *(probe(pid, p) for pid, p in providers_map.items() if p.get("internal_url"))
    )
    return {"providers": dict(results)}


@app.get("/api/providers/{provider_id}/voices")
async def provider_voices(provider_id: str):
    """Return a normalized voice catalog for a TTS provider."""
    provider = _get_provider(provider_id, kind="tts")
    contract = provider.get("contracts", {}).get("voice_catalog")

    if contract == "voice-catalog-v1":
        response = await _provider_get(provider_id, "/voices", timeout=15.0)
        return {
            "provider": provider_id,
            "contract": contract,
            "voices": _normalize_piper_voice_catalog(response.json()),
        }

    if contract == "speaker-catalog-v1":
        response = await _provider_get(provider_id, "/speakers", timeout=15.0)
        return {
            "provider": provider_id,
            "contract": contract,
            "voices": _normalize_qwen3_voice_catalog(response.json()),
        }

    raise HTTPException(status_code=400, detail=f"Provider {provider_id} does not expose a normalized voice catalog")


@app.get("/api/providers/{provider_id}/models")
async def provider_models(provider_id: str):
    """Return a normalized model catalog for providers that support model variants."""
    provider = _get_provider(provider_id)
    contract = provider.get("contracts", {}).get("model_catalog")
    if contract != "model-catalog-v1":
        raise HTTPException(status_code=400, detail=f"Provider {provider_id} does not expose a model catalog")

    response = await _provider_get(provider_id, "/models")
    models = _normalize_qwen3_model_catalog(response.json())
    return {
        "provider": provider_id,
        "models": models,
        "current_model": next((model for model in models if model.get("is_current")), None),
    }


@app.post("/api/providers/{provider_id}/models/select")
async def provider_select_model(provider_id: str, request: ProviderModelSelectionRequest):
    """Switch the active model for a provider that supports model variants."""
    provider = _get_provider(provider_id)
    contract = provider.get("contracts", {}).get("model_selection")
    if contract != "model-selection-v1":
        raise HTTPException(status_code=400, detail=f"Provider {provider_id} does not support model switching")

    response = await _provider_json_post(provider_id, "/load_model", {"model": request.model}, timeout=300.0)
    payload = response.json()
    return {
        "provider": provider_id,
        "message": payload.get("message", ""),
        "model": {
            "id": request.model,
            "name": payload.get("model_info", {}).get("name") or request.model,
        },
    }


@app.get("/api/providers/{provider_id}/status")
async def provider_status(provider_id: str):
    """Return provider status for providers that expose runtime metadata."""
    provider = _get_provider(provider_id)
    contract = provider.get("contracts", {}).get("runtime_status")
    if contract != "runtime-status-v1":
        raise HTTPException(status_code=400, detail=f"Provider {provider_id} does not expose a status endpoint")

    response = await _provider_get(provider_id, "/status")
    return _normalize_qwen3_runtime_status(response.json(), provider_id)


@app.post("/api/providers/{provider_id}/unload")
async def provider_unload(provider_id: str):
    """Ask a provider to release its model and free the memory now.

    Passes the upstream status and body through verbatim rather than going via
    `_provider_json_post`, which raises on any >=400 and would flatten the
    upstream body to its `detail`. A 409 here is a normal, actionable answer —
    the model is busy — and the caller needs the reference count that comes with
    it to know when retrying is worthwhile.
    """
    provider = _get_provider(provider_id)
    if not provider.get("capabilities") or "model_unload" not in provider["capabilities"]:
        raise HTTPException(
            status_code=400,
            detail=f"Provider {provider_id} does not support unloading",
        )

    client = _get_http_client()
    try:
        response = await client.post(f"{provider['internal_url']}/unload", timeout=_timeout(30.0))
    except httpx.RequestError as exc:
        raise _build_upstream_request_error(provider.get("display_name", provider_id), exc) from exc

    try:
        body = response.json()
    except Exception:
        body = {"detail": response.text}
    return JSONResponse(status_code=response.status_code, content={"provider": provider_id, **body})


@app.get("/api/providers/{provider_id}/saved-voices")
async def provider_saved_voices(provider_id: str):
    """List saved voices for providers that support a voice library."""
    provider = _get_provider(provider_id)
    contract = provider.get("contracts", {}).get("saved_voices")
    if contract != "saved-voice-library-v1":
        raise HTTPException(status_code=400, detail=f"Provider {provider_id} does not support saved voices")

    response = await _provider_get(provider_id, "/voices")
    return _normalize_qwen3_saved_voice_library(response.json(), provider_id)


@app.get("/api/providers/{provider_id}/custom-voices")
async def provider_custom_voices(provider_id: str):
    """List managed custom voices for providers that support custom model management."""
    provider = _get_provider(provider_id)
    contract = provider.get("contracts", {}).get("managed_voices")
    if contract != "custom-voice-library-v1":
        raise HTTPException(status_code=400, detail=f"Provider {provider_id} does not support custom voice management")

    response = await _provider_get(provider_id, "/voices")
    normalized_voices = _normalize_piper_voice_catalog(response.json())
    custom_voices = [voice for voice in normalized_voices if voice.get("kind") == "custom"]
    return {
        "provider": provider_id,
        "contract": contract,
        "voices": custom_voices,
    }


@app.post("/api/providers/{provider_id}/saved-voices")
async def provider_save_voice(provider_id: str, request: Request):
    """Create a saved voice entry for providers that support voice libraries."""
    provider = _get_provider(provider_id)
    contract = provider.get("contracts", {}).get("saved_voices")
    if contract != "saved-voice-library-v1":
        raise HTTPException(status_code=400, detail=f"Provider {provider_id} does not support saved voices")

    response = await _provider_form_post(provider_id, "/voices/save", request, timeout=300.0)
    return response.json()


@app.delete("/api/providers/{provider_id}/saved-voices/{voice_id}")
async def provider_delete_saved_voice(provider_id: str, voice_id: str):
    """Delete a saved voice entry for providers that support voice libraries."""
    provider = _get_provider(provider_id)
    contract = provider.get("contracts", {}).get("saved_voices")
    if contract != "saved-voice-library-v1":
        raise HTTPException(status_code=400, detail=f"Provider {provider_id} does not support saved voices")

    response = await _provider_delete(provider_id, f"/voices/{voice_id}")
    return response.json()


@app.delete("/api/providers/{provider_id}/custom-voices/{voice_id}")
async def provider_delete_custom_voice(provider_id: str, voice_id: str):
    """Delete a managed custom voice for providers that support custom model management."""
    provider = _get_provider(provider_id)
    contract = provider.get("contracts", {}).get("managed_voices")
    if contract != "custom-voice-library-v1":
        raise HTTPException(status_code=400, detail=f"Provider {provider_id} does not support custom voice management")

    response = await _provider_delete(provider_id, f"/voice/{voice_id}")
    return response.json()


@app.post("/api/providers/{provider_id}/saved-voices/{voice_id}/tts")
async def provider_saved_voice_tts(provider_id: str, voice_id: str, request: Request):
    """Synthesize speech with a saved voice profile through the frontend adapter."""
    provider = _get_provider(provider_id)
    contract = provider.get("contracts", {}).get("saved_voices")
    if contract != "saved-voice-library-v1":
        raise HTTPException(status_code=400, detail=f"Provider {provider_id} does not support saved voices")

    response = await _provider_form_post(provider_id, f"/voices/{voice_id}/tts", request, timeout=600.0)
    headers = _passthrough_headers(response)
    headers["X-Provider"] = provider_id
    return Response(content=response.content, media_type=response.headers.get("content-type", "audio/wav"), headers=headers)


@app.post("/api/providers/{provider_id}/voice-clone")
async def provider_voice_clone(provider_id: str, request: Request):
    """Run provider-scoped voice cloning through a frontend adapter."""
    provider = _get_provider(provider_id)
    contract = provider.get("contracts", {}).get("voice_clone")
    if contract != "voice-clone-tts-v1":
        raise HTTPException(status_code=400, detail=f"Provider {provider_id} does not support voice cloning")

    form = await request.form()
    use_ref_text = bool(str(form.get("ref_text", "")).strip())
    backend_path = "/clone-with-ref-text" if use_ref_text else "/clone"

    response = await _provider_form_post(provider_id, backend_path, request, timeout=600.0)
    headers = _passthrough_headers(response)
    headers["X-Provider"] = provider_id
    return Response(content=response.content, media_type=response.headers.get("content-type", "audio/wav"), headers=headers)


@app.post("/api/providers/{provider_id}/voice-design")
async def provider_voice_design(provider_id: str, request: ProviderVoiceDesignRequest):
    """Run provider-scoped voice design through a frontend adapter."""
    provider = _get_provider(provider_id)
    contract = provider.get("contracts", {}).get("voice_design")
    if contract != "voice-design-tts-v1":
        raise HTTPException(status_code=400, detail=f"Provider {provider_id} does not support voice design")

    response = await _provider_json_post(provider_id, "/voice_design", request.model_dump(), timeout=600.0)
    headers = _passthrough_headers(response)
    headers["X-Provider"] = provider_id
    return Response(content=response.content, media_type=response.headers.get("content-type", "audio/wav"), headers=headers)


@app.websocket("/ws/stt")
async def frontend_ws_stt(websocket: WebSocket):
    """Relay the live-transcription WebSocket to the STT provider.

    Without this the browser has to dial the STT container's published port
    directly. getUserMedia requires a secure context, so any real deployment is
    HTTPS — and there a `ws://localhost:5001` handshake is hard-blocked as mixed
    content. Behind single-port ingress that port is not routable at all.

    Query string is forwarded, so ?provider=whisper selects the upstream.
    """
    import websockets

    # Starlette's CORSMiddleware does not run on WebSocket routes, so the
    # allow-list has to be enforced here or this endpoint is the one hole in an
    # otherwise restricted deployment.
    origin = websocket.headers.get("origin")
    if "*" not in allowed_origins and origin and origin not in allowed_origins:
        await websocket.close(code=1008)  # policy violation
        return

    provider_id = websocket.query_params.get("provider", "whisper")
    try:
        provider = _get_provider(provider_id, kind="stt")
    except HTTPException as exc:
        await websocket.close(code=1008, reason=str(exc.detail)[:120])
        return

    # Only stt-service implements /ws/transcribe. The other STT backends are
    # request/response only — qwen3-asr, parakeet and canary expose no
    # WebSocket at all, and whisper-cpp is the upstream whisper-server binary
    # with no Python layer. Without this check the relay happily dialled
    # ws://<backend>/ws/transcribe, and the caller got an opaque connection
    # failure instead of being told the provider cannot do this.
    if "live_transcribe" not in (provider.get("capabilities") or []):
        await websocket.close(
            code=1008,
            reason=f"provider '{provider_id}' does not support live transcription"[:120],
        )
        return

    upstream_url = provider["internal_url"].replace("http://", "ws://").replace("https://", "wss://") + "/ws/transcribe"

    await websocket.accept()
    try:
        async with websockets.connect(
            upstream_url,
            max_size=None,        # audio frames are not size-bounded by the protocol
            ping_interval=20,
            open_timeout=10,
            # Forward the browser's origin so the STT service can apply its own
            # allow-list; without it the upstream only ever sees no origin.
            additional_headers={"Origin": origin} if origin else None,
        ) as upstream:
            async def client_to_upstream():
                while True:
                    message = await websocket.receive()
                    if message.get("type") == "websocket.disconnect":
                        return
                    if message.get("bytes") is not None:
                        await upstream.send(message["bytes"])
                    elif message.get("text") is not None:
                        await upstream.send(message["text"])

            async def upstream_to_client():
                async for message in upstream:
                    if isinstance(message, bytes):
                        await websocket.send_bytes(message)
                    else:
                        await websocket.send_text(message)

            done, pending = await asyncio.wait(
                [asyncio.create_task(client_to_upstream()),
                 asyncio.create_task(upstream_to_client())],
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in pending:
                task.cancel()
            for task in done:
                # Surface a relay-side failure rather than closing silently.
                exc = task.exception()
                if exc:
                    raise exc
    except WebSocketDisconnect:
        pass
    except Exception as e:
        # The handshake with the browser already succeeded, so a plain close
        # here would reach the page as a normal 1000 — no `error` event fires
        # and the UI would keep claiming it is listening. Send an explicit
        # error frame, which the client already knows how to render.
        logger.warning(f"WebSocket relay to {provider_id} failed: {e}")
        try:
            await websocket.send_json({
                "type": "error",
                "error": f"Live transcription unavailable: {provider_id} could not be reached.",
            })
        except Exception:
            pass
        try:
            await websocket.close(code=1011)
        except Exception:
            pass
        return
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


@app.post("/api/stt")
async def frontend_stt(request: Request):
    """Transcribe audio through a normalized frontend STT adapter."""
    form = await request.form()
    provider_id = str(form.get("provider", "")).strip()
    if not provider_id:
        raise HTTPException(status_code=400, detail="Provider not provided")

    provider = _get_provider(provider_id, kind="stt")
    contract = provider.get("contracts", {}).get("transcribe") or "stt-form-v1"
    backend_path, data, files = await _build_frontend_stt_payload(provider_id, form, contract)

    client = _get_http_client()
    try:
        response = await client.post(f"{provider['internal_url']}{backend_path}", data=data, files=files, timeout=_timeout(300.0))
    except httpx.RequestError as exc:
        raise _build_upstream_request_error(provider.get("display_name", provider_id), exc) from exc

    if response.status_code >= 400:
        raise _build_error_from_response(response)

    headers = _passthrough_headers(response)
    headers["X-Provider"] = provider_id
    return JSONResponse(
        content=_normalize_frontend_stt_response(response.json(), contract),
        headers=headers,
    )


@app.post("/api/tts")
async def frontend_tts(request: FrontendTTSRequest):
    """Synthesize speech through a normalized frontend TTS adapter."""
    provider = _get_provider(request.provider, kind="tts")

    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text not provided")

    payload, read_timeout = _build_tts_payload(
        request.provider,
        provider,
        text=request.text,
        voice=request.voice,
        language=request.language,
        quality=request.quality,
        gender=request.gender,
        speed=request.speed,
        instructions=request.instructions,
        output_format=request.output_format,
    )

    # Prefer the provider's chunked streaming endpoint when it declares one:
    # time-to-first-audio then tracks the first sentence instead of the whole
    # text. The body is proxied without buffering (see _stream_upstream) —
    # buffering here would throw the entire benefit away.
    path = "/tts-stream" if "tts_stream" in provider.get("contracts", {}) else "/tts"

    return await _stream_upstream(
        "POST",
        f"{provider['internal_url']}{path}",
        display_name=provider.get("display_name", request.provider),
        json=payload,
        read_timeout=read_timeout,
        extra_headers={"X-Provider": request.provider},
    )


@app.get("/api/training/deployment-targets")
async def frontend_training_deployment_targets():
    """Return training deployment targets through the frontend adapter."""
    response = await _proxy_training_get("/deployment-targets")
    return response.json()


@app.post("/api/training/train")
async def frontend_training_start(request: Request):
    """Start a training job through the frontend adapter."""
    response = await _proxy_training_form_post("/train", request, timeout=300.0)
    return response.json()


@app.post("/api/training/train-from-dataset")
async def frontend_training_from_dataset(request: Request):
    """Start a dataset-backed training job through the frontend adapter."""
    response = await _proxy_training_form_post("/train-from-dataset", request, timeout=120.0)
    return response.json()


@app.post("/api/training/resume")
async def frontend_training_resume(request: Request):
    """Resume a training job through the frontend adapter."""
    response = await _proxy_training_form_post("/resume-training", request, timeout=120.0)
    return response.json()


@app.get("/api/training/jobs")
async def frontend_training_jobs():
    """List training jobs through the frontend adapter."""
    response = await _proxy_training_get("/jobs")
    return _normalize_training_jobs_payload(response.json())


@app.get("/api/training/status/{job_id}")
async def frontend_training_status(job_id: str):
    """Get a single training job status through the frontend adapter."""
    response = await _proxy_training_get(f"/status/{job_id}")
    return _normalize_training_job(response.json())


@app.post("/api/training/export/{job_id}")
async def frontend_training_export(job_id: str, request: Request):
    """Export and optionally deploy a model bundle through the frontend adapter."""
    response = await _proxy_training_form_post(f"/export/{job_id}", request, timeout=120.0)
    return _normalize_training_export_response(response.json())


@app.get("/api/training/download/{job_id}")
async def frontend_training_download(job_id: str):
    """Download an exported model bundle through the frontend adapter."""
    response = await _proxy_training_get(f"/download/{job_id}", timeout=120.0)
    return Response(content=response.content, media_type=response.headers.get("content-type", "application/octet-stream"), headers=_passthrough_headers(response))


@app.delete("/api/training/model/{job_id}")
async def frontend_training_delete_model(job_id: str):
    """Delete a trained model through the frontend adapter."""
    response = await _proxy_training_delete(f"/model/{job_id}")
    return response.json()


@app.delete("/api/training/job/{job_id}")
async def frontend_training_cancel_job(job_id: str):
    """Cancel a training job through the frontend adapter."""
    response = await _proxy_training_delete(f"/job/{job_id}")
    return response.json()


# --- OpenAI-compatible /v1 surface ------------------------------------------
#
# Mounted last so every helper it depends on is defined. Dependencies are passed
# in rather than imported, so openai_router.py never imports this module and
# there is no circular dependency.
#
# /api/* is unchanged and remains the browser contract; /v1/* is the surface
# that lets an OpenAI client talk to any of the four devices identically.
app.include_router(
    build_openai_router(
        get_provider=_get_provider,
        registry=PROVIDER_REGISTRY,
        post_form=_provider_form_post_raw,
        post_json=_provider_json_post,
        provider_health=provider_health,
        build_tts_payload=_build_tts_payload,
    )
)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3000)
