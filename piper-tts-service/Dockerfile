FROM python:3.10-slim

# Install system dependencies.
# alsa-utils and pulseaudio were removed deliberately: this service only ever
# writes WAV files, it never opens a sound card.
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    espeak-ng \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install Piper TTS.
# TARGETARCH is supplied by BuildKit and is exactly "amd64" or "arm64", which
# matches the release asset names. Hardcoding piper_amd64 made this image
# unbuildable for ARM SBCs (RK3588), where Piper is the primary TTS engine.
ARG TARGETARCH
RUN wget -O /tmp/piper.tar.gz \
      "https://github.com/rhasspy/piper/releases/download/v1.2.0/piper_${TARGETARCH}.tar.gz" \
    && tar -xzf /tmp/piper.tar.gz -C /opt/ \
    && rm /tmp/piper.tar.gz \
    && ln -s /opt/piper/piper /usr/local/bin/piper

WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create directories
RUN mkdir -p /app/models /app/output /app/voices

# Download default models for multiple languages
RUN mkdir -p /app/models/default && \
    cd /app/models/default && \
    # English voices
    wget -q https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx && \
    wget -q https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json && \
    wget -q https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/amy/medium/en_US-amy-medium.onnx && \
    wget -q https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/amy/medium/en_US-amy-medium.onnx.json && \
    wget -q https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_GB/alan/medium/en_GB-alan-medium.onnx && \
    wget -q https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_GB/alan/medium/en_GB-alan-medium.onnx.json && \
    # German voices
    wget -q https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/de/de_DE/thorsten/medium/de_DE-thorsten-medium.onnx && \
    wget -q https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/de/de_DE/thorsten/medium/de_DE-thorsten-medium.onnx.json && \
    wget -q https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/de/de_DE/eva_k/x_low/de_DE-eva_k-x_low.onnx && \
    wget -q https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/de/de_DE/eva_k/x_low/de_DE-eva_k-x_low.onnx.json && \
    # French voices
    wget -q https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/fr/fr_FR/siwis/medium/fr_FR-siwis-medium.onnx && \
    wget -q https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/fr/fr_FR/siwis/medium/fr_FR-siwis-medium.onnx.json && \
    # Spanish voices
    wget -q https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/es/es_ES/mls_9972/low/es_ES-mls_9972-low.onnx && \
    wget -q https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/es/es_ES/mls_9972/low/es_ES-mls_9972-low.onnx.json && \
    # Italian voices
    wget -q https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/it/it_IT/riccardo/x_low/it_IT-riccardo-x_low.onnx && \
    wget -q https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/it/it_IT/riccardo/x_low/it_IT-riccardo-x_low.onnx.json && \
    # Dutch voices
    wget -q https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/nl/nl_NL/mls_5809/low/nl_NL-mls_5809-low.onnx && \
    wget -q https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/nl/nl_NL/mls_5809/low/nl_NL-mls_5809-low.onnx.json

# Copy application code
COPY . .

EXPOSE 5000

CMD ["python", "app.py"]
