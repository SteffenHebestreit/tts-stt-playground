# Frontend Service Dockerfile
# 3.10 reaches end of life in 2026-10; 3.12 is what the test suite also runs on.
FROM python:3.12-slim

WORKDIR /app

# Install system dependencies.
# ffmpeg is required by /v1/audio/speech: mp3 is the OpenAI default
# response_format, every TTS backend here emits WAV, and doing the container
# conversion once at the gateway beats teaching each backend to transcode.
# Without it that endpoint returns a documented 501 pointing at wav.
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY app.py .
COPY openai_router.py .
COPY entrypoint.sh .
COPY templates/ templates/
COPY static/ static/

# Create directories
RUN mkdir -p /app/static/css /app/static/js /app/templates

# Expose port
EXPOSE 3000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:3000/health || exit 1

# Start command. Worker count comes from FRONTEND_WORKERS (default 2); see
# entrypoint.sh for why this is a script rather than an inline CMD.
CMD ["sh", "/app/entrypoint.sh"]
