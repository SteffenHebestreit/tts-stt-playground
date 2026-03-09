# Frontend Service

Web interface and API documentation hub for the TTS-STT platform. Built with FastAPI and Jinja2.

## Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Main web interface |
| GET | `/api-docs` | Interactive API documentation |
| GET | `/health` | Health check |

## Features

- Tab-based UI: Speech-to-Text, Voice Training, Text-to-Speech, Voice Cloning
- TTS engine selector: PiperTTS (local training) or Qwen3-TTS (voice cloning)
- Service health indicators for all backend services
- File drag-and-drop upload for audio files
- Real-time training progress monitoring
- Audio playback for generated speech

## Configuration

The frontend connects to backend services via browser-accessible localhost URLs:

| Service | Browser URL |
|---------|------------|
| PiperTTS | http://localhost:5000 |
| STT | http://localhost:5001 |
| Training | http://localhost:8080 |
| Qwen3-TTS | http://localhost:5004 |
| Qwen3-ASR | http://localhost:5002 |

Access the frontend at **http://localhost:3000**.
