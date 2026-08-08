# Frontend Service Dockerfile
FROM python:3.10-slim

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
COPY templates/ templates/
COPY static/ static/

# Create directories
RUN mkdir -p /app/static/css /app/static/js /app/templates

# Expose port
EXPOSE 3000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:3000/health || exit 1

# Start command.
#   --workers 2       the gateway is I/O-bound with no per-process model state.
#                     Asset cache-busting is derived from a hash of static/, so
#                     every worker computes the same value without pinning it.
#   --timeout-keep-alive  >= the httpx keepalive_expiry used for upstream calls,
#                     so the server does not FIN first and hand back dead sockets.
CMD ["python", "-m", "uvicorn", "app:app", \
     "--host", "0.0.0.0", "--port", "3000", \
     "--workers", "2", \
     "--proxy-headers", \
     "--timeout-keep-alive", "120", \
     "--no-access-log"]
