# whisper.cpp STT server — CPU build
# For GPU acceleration use docker-compose.vulkan.yml (Vulkan) or docker-compose.rocm.yml (ROCm)

FROM ubuntu:22.04 AS builder

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    git cmake build-essential \
    && rm -rf /var/lib/apt/lists/*

# Clone and build whisper.cpp (server only, static binary)
RUN git clone https://github.com/ggerganov/whisper.cpp.git /src && \
    cd /src && \
    cmake -B build \
        -DCMAKE_BUILD_TYPE=Release \
        -DWHISPER_BUILD_SERVER=ON \
        -DBUILD_SHARED_LIBS=OFF && \
    cmake --build build -j$(nproc) --target whisper-server

# ── Runtime ──────────────────────────────────────────────────────────────────
FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /src/build/bin/whisper-server /usr/local/bin/whisper-server

RUN mkdir -p /models

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 8080

ENTRYPOINT ["/entrypoint.sh"]
