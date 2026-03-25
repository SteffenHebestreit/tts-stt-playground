"""Frontend service for the TTS-STT platform.

Serves the web UI, static assets, and API documentation pages.
Acts as a gateway that provides browser-facing URLs for all backend services.
"""

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import time

# Cache-busting version: bumped on every restart so browsers fetch fresh assets
APP_VERSION = str(int(time.time()))

app = FastAPI(title="TTS-STT Frontend Service", version="2.0.0")

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

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Internal Docker-network URLs (container-to-container communication)
TTS_SERVICE_URL = os.getenv("TTS_SERVICE_URL", "http://piper-tts-service:5000")
STT_SERVICE_URL = os.getenv("STT_SERVICE_URL", "http://stt-service:8000")
VOICE_TRAINING_URL = os.getenv("VOICE_TRAINING_URL", "http://piper-training-service:8080")
QWEN3_TTS_SERVICE_URL = os.getenv("QWEN3_TTS_SERVICE_URL", "http://qwen3-tts-service:5004")
QWEN3_ASR_SERVICE_URL = os.getenv("QWEN3_ASR_SERVICE_URL", "http://qwen3-asr-service:5002")

# Browser-facing URLs (host ports, used by client-side JavaScript)
BROWSER_TTS_SERVICE_URL = os.getenv("BROWSER_TTS_URL", "http://localhost:5000")
BROWSER_STT_SERVICE_URL = os.getenv("BROWSER_STT_URL", "http://localhost:5001")
BROWSER_VOICE_TRAINING_URL = os.getenv("BROWSER_TRAINING_URL", "http://localhost:8080")
BROWSER_QWEN3_TTS_SERVICE_URL = os.getenv("BROWSER_QWEN3_TTS_URL", "http://localhost:5004")
BROWSER_QWEN3_ASR_SERVICE_URL = os.getenv("BROWSER_QWEN3_ASR_URL", "http://localhost:5002")


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Render the main web UI page with service URLs injected into the template."""
    return templates.TemplateResponse("index.html", {
        "request": request,
        "tts_service_url": BROWSER_TTS_SERVICE_URL,
        "stt_service_url": BROWSER_STT_SERVICE_URL,
        "voice_training_url": BROWSER_VOICE_TRAINING_URL,
        "qwen3_tts_service_url": BROWSER_QWEN3_TTS_SERVICE_URL,
        "qwen3_asr_service_url": BROWSER_QWEN3_ASR_SERVICE_URL,
        "app_version": APP_VERSION,
    })


@app.get("/api-docs", response_class=HTMLResponse)
async def api_docs():
    """Serve the interactive API documentation page."""
    with open("static/api_docs.html", "r", encoding="utf-8") as f:
        content = f.read()
    return HTMLResponse(content=content)


@app.get("/health")
async def health():
    """Return service health status and configured backend URLs."""
    return {
        "status": "healthy",
        "services": {
            "tts": TTS_SERVICE_URL,
            "stt": STT_SERVICE_URL,
            "voice_training": VOICE_TRAINING_URL,
            "qwen3_tts": QWEN3_TTS_SERVICE_URL,
            "qwen3_asr": QWEN3_ASR_SERVICE_URL,
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3000)
