FROM python:3.11-slim

WORKDIR /app

# Install system dependencies including Piper TTS
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    unzip \
    build-essential \
    libsndfile1-dev \
    ffmpeg \
    espeak-ng \
    && rm -rf /var/lib/apt/lists/*

# Install Piper TTS binary
RUN wget -O piper.tar.gz "https://github.com/rhasspy/piper/releases/download/v1.2.0/piper_amd64.tar.gz" \
    && tar -xzf piper.tar.gz \
    && cp piper/piper /usr/local/bin/ \
    && chmod +x /usr/local/bin/piper \
    && rm -rf piper.tar.gz piper

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip wheel setuptools
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY app.py .

# Create models directory
RUN mkdir -p /app/models && chmod 755 /app/models

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PIPER_MODELS_DIR=/app/models

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
