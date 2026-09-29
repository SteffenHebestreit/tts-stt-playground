# 3.10 reaches end of life in 2026-10; requirements.txt is resolved and tested on 3.12.
FROM python:3.12-slim

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

# Install Piper TTS. rhasspy/piper is archived upstream (its successor,
# OHF-Voice/piper1-gpl, is a Python package with a different CLI); v1.2.0 stays
# until that migration is done.
# TARGETARCH is supplied by BuildKit and is exactly "amd64" or "arm64", which
# matches the release asset names. Hardcoding piper_amd64 made this image
# unbuildable for ARM SBCs (RK3588), where Piper is the primary TTS engine.
#
# A builder that is not BuildKit (the classic `docker build`, podman) does not set
# TARGETARCH, and the empty value asked for piper_.tar.gz: a 404. Without it the
# host's own architecture is used, amd64 unless the host says aarch64: the classic
# builder never cross-builds, so the host IS the target. (A fixed amd64 default
# would build an x86 binary into an image for an ARM board without any error.)
ARG TARGETARCH
RUN arch="${TARGETARCH:-}" \
    && if [ -z "$arch" ]; then \
         case "$(uname -m)" in aarch64|arm64) arch=arm64 ;; *) arch=amd64 ;; esac; \
       fi \
    && wget -O /tmp/piper.tar.gz \
      "https://github.com/rhasspy/piper/releases/download/v1.2.0/piper_${arch}.tar.gz" \
    && tar -xzf /tmp/piper.tar.gz -C /opt/ \
    && rm /tmp/piper.tar.gz \
    && ln -s /opt/piper/piper /usr/local/bin/piper

WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create directories
RUN mkdir -p /app/models /app/output /app/voices

# Voices shipped in the image, as Piper voice ids (<lang>-<speaker>-<quality>).
# This is the only list: the service does not need it, it registers whatever
# <id>.onnx + <id>.onnx.json pairs it finds in /app/models/default when it starts
# (and on POST /refresh_voices). A bind-mounted models volume hides these files
# anyway, so voices can also simply be copied into that volume.
# Change the set at build time: --build-arg PIPER_VOICES="de_DE-thorsten-medium ..."
ARG PIPER_VOICES="en_US-lessac-medium en_US-amy-medium en_GB-alan-medium de_DE-thorsten-medium de_DE-eva_k-x_low fr_FR-siwis-medium es_ES-mls_9972-low it_IT-riccardo-x_low nl_NL-mls_5809-low"
ARG PIPER_VOICES_REVISION=v1.0.0
RUN set -eu; \
    mkdir -p /app/models/default; \
    cd /app/models/default; \
    for voice in ${PIPER_VOICES}; do \
      lang="${voice%%-*}"; rest="${voice#*-}"; speaker="${rest%-*}"; quality="${rest##*-}"; \
      base="https://huggingface.co/rhasspy/piper-voices/resolve/${PIPER_VOICES_REVISION}/${lang%%_*}/${lang}/${speaker}/${quality}/${voice}"; \
      wget -q "${base}.onnx" "${base}.onnx.json"; \
    done

# Copy application code
COPY . .

EXPOSE 5000

# /health is liveness only; /ready also fails while no voice model is installed.
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# 120 s keep-alive: the gateway pools connections, and uvicorn's 5 s default closes
# them between requests, racing the next one.
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5000", "--timeout-keep-alive", "120"]
