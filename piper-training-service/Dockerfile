# CUDA 12.8 + cu128 wheels, matching every other GPU service in this repo.
# This was the last service still on 11.8/cu118, which caps kernel support at
# ~sm_90 and therefore fails on Blackwell (RTX 50-series, sm_120) with
# cudaErrorNoKernelImageForDevice. Nothing here pins an old torch — the 11.8
# base was simply never bumped with its siblings.
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

# Install system dependencies including AMD ROCm support detection tools
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    wget \
    curl \
    build-essential \
    libsndfile1 \
    sox \
    ffmpeg \
    espeak-ng \
    pciutils \
    lshw \
    && rm -rf /var/lib/apt/lists/*

# Optional: Add AMD ROCm repository (commented out by default)
# RUN wget -q -O - https://repo.radeon.com/rocm/rocm.gpg.key | apt-key add - \
#     && echo 'deb [arch=amd64] https://repo.radeon.com/rocm/apt/debian/ ubuntu main' > /etc/apt/sources.list.d/rocm.list \
#     && apt-get update && apt-get install -y rocm-dev rocm-libs \
#     && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install system dependencies for building packages
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install basic build tools
RUN pip3 install --upgrade pip setuptools wheel

# Install Python dependencies in stages to avoid conflicts
COPY requirements.txt .

# Install core dependencies first with CUDA support
RUN pip3 install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cu128

# Install the rest of the requirements
RUN pip3 install --no-cache-dir -r requirements.txt

# Install monotonic_align manually (optional, our code has fallback)
# RUN pip3 install "monotonic_align @ git+git://github.com/rhasspy/monotonic_align.git"

# Copy application code
COPY . .

# Make the startup script executable
RUN chmod +x /app/start.sh

EXPOSE 8080

# Use the device detection startup script
CMD ["/app/start.sh"]
