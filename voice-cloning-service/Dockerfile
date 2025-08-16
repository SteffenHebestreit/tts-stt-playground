FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    git \
    wget \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip wheel setuptools \
 && pip install --no-cache-dir -r requirements.txt

# Install OpenVoice from GitHub (editable install)
RUN git clone https://github.com/myshell-ai/OpenVoice.git /tmp/OpenVoice \
 && cd /tmp/OpenVoice \
 && git checkout main \
 && pip install -e . \
 && cd /app \
 && rm -rf /tmp/OpenVoice

# Create necessary directories
RUN mkdir -p /app/models/checkpoints /app/models/checkpoints_v2 /app/cloned_voices /app/temp

# Create model directory structure
RUN python -c "\
from pathlib import Path; \
base_dir = Path('/app/models/checkpoints'); \
base_dir.mkdir(parents=True, exist_ok=True); \
(base_dir / 'base_speakers' / 'EN').mkdir(parents=True, exist_ok=True); \
(base_dir / 'converter').mkdir(parents=True, exist_ok=True); \
base_dir2 = Path('/app/models/checkpoints_v2'); \
base_dir2.mkdir(parents=True, exist_ok=True); \
(base_dir2 / 'base_speakers' / 'EN').mkdir(parents=True, exist_ok=True); \
(base_dir2 / 'converter').mkdir(parents=True, exist_ok=True); \
print('Model directories prepared.'); \
"

# Copy model downloader script
COPY download_models.py .

# Copy application code
COPY app.py .

# Expose port
EXPOSE 5001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5001/health || exit 1

# Run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5001"]
