# whisper.cpp STT server — CPU build
# For GPU acceleration use docker-compose.vulkan.yml (Vulkan) or docker-compose.rocm.yml (ROCm)

FROM ubuntu:22.04 AS builder

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    git cmake build-essential ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Clone and build whisper.cpp (server only, static binary).
#
# GGML_NATIVE=OFF is required, not an optimisation. It defaults ON when not
# cross-compiling, and ggml then probes the BUILD host's CPU and appends
# whatever it finds — including SVE/SME. A native arm64 CI runner is Neoverse
# (Armv9, SVE2), while the RK3588's Cortex-A76 is Armv8.2-A with no SVE, no SME
# and no i8mm, so a runner-tuned binary SIGILLs on the target board.
# Do NOT also pass GGML_CPU_ALL_VARIANTS — ggml hard-errors on the combination.
#
# The clone is pinned: without it the amd64 and arm64 halves of the same
# published tag can be built from different upstream commits. The org moved
# from ggerganov to ggml-org.
ARG TARGETARCH
ARG WHISPER_CPP_REF=v1.7.6
RUN git clone --depth 1 --branch "${WHISPER_CPP_REF}" \
      https://github.com/ggml-org/whisper.cpp.git /src && \
    cd /src && \
    if [ "$TARGETARCH" = "arm64" ]; then \
      ARCH_FLAGS="-DGGML_CPU_ARM_ARCH=armv8.2-a+dotprod+fp16"; \
    else \
      ARCH_FLAGS=""; \
    fi && \
    cmake -B build \
        -DCMAKE_BUILD_TYPE=Release \
        -DWHISPER_BUILD_SERVER=ON \
        -DBUILD_SHARED_LIBS=OFF \
        -DGGML_NATIVE=OFF \
        ${ARCH_FLAGS} && \
    cmake --build build -j$(nproc) --target whisper-server

# ── Runtime ──────────────────────────────────────────────────────────────────
FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /src/build/bin/whisper-server /usr/local/bin/whisper-server

RUN mkdir -p /models

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 8080

ENTRYPOINT ["/entrypoint.sh"]
