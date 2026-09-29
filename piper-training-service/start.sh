#!/bin/bash
# Device Detection and Setup Script for TTS Training Service
# Supports NVIDIA CUDA, AMD ROCm, Apple Metal, and CPU fallback

echo "🔍 Detecting compute devices..."

# Function to check for NVIDIA CUDA
check_nvidia() {
    if command -v nvidia-smi &> /dev/null; then
        echo "✅ NVIDIA GPU detected:"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits
        # Only a default. `export CUDA_VISIBLE_DEVICES=0` here overrode the
        # value compose (and .env) pass in, so choosing another GPU, or setting
        # it empty to hide them all, did nothing. `=` rather than `:=` keeps an
        # explicitly empty value, which is how CUDA is told to see no device.
        : "${CUDA_VISIBLE_DEVICES=0}"
        export CUDA_VISIBLE_DEVICES
        return 0
    else
        echo "❌ No NVIDIA GPU detected"
        return 1
    fi
}

# Function to check for AMD ROCm
check_amd() {
    if command -v rocm-smi &> /dev/null; then
        echo "✅ AMD GPU (ROCm) detected:"
        rocm-smi --showproductname --showmeminfo
        # Only set these if the container environment did not. A bare `export`
        # here silently overrode whatever docker-compose.rocm.yml configured,
        # making HSA_OVERRIDE_GFX_VERSION unreachable for this service — and
        # forcing gfx1100 kernels is actively wrong once the base image moves to
        # a wheel index that has native gfx1151 support.
        : "${HIP_VISIBLE_DEVICES:=0}"
        : "${ROC_ENABLE_PRE_VEGA:=1}"
        : "${ROCM_PATH:=/opt/rocm}"
        export HIP_VISIBLE_DEVICES ROC_ENABLE_PRE_VEGA ROCM_PATH
        # HSA_OVERRIDE_GFX_VERSION is deliberately NOT defaulted here: it belongs
        # to the compose overlay, which knows which wheel index the image uses.
        if [ -n "${HSA_OVERRIDE_GFX_VERSION:-}" ]; then
            export HSA_OVERRIDE_GFX_VERSION
            echo "   using HSA_OVERRIDE_GFX_VERSION=${HSA_OVERRIDE_GFX_VERSION}"
        fi
        return 0
    elif command -v lspci &> /dev/null && lspci | grep -i amd | grep -i vga &> /dev/null; then
        echo "✅ AMD GPU detected (no ROCm, using CPU fallback)"
        echo "💡 For optimal performance, install ROCm: https://docs.amd.com/bundle/ROCm-Installation-Guide-v5.4.3/page/How_to_Install_ROCm.html"
        return 1
    else
        echo "❌ No AMD GPU detected"
        return 1
    fi
}

# Function to check for Apple Metal
check_apple() {
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo "✅ Apple Silicon detected, Metal Performance Shaders available"
        system_profiler SPHardwareDataType | grep "Chip:"
        return 0
    else
        echo "❌ Not running on macOS"
        return 1
    fi
}

# Main detection logic
DEVICE_TYPE="cpu"
if check_nvidia; then
    DEVICE_TYPE="cuda"
    echo "🚀 Using NVIDIA CUDA acceleration"
elif check_amd; then
    DEVICE_TYPE="hip"
    echo "🚀 Using AMD ROCm acceleration"
elif check_apple; then
    DEVICE_TYPE="mps"
    echo "🚀 Using Apple Metal acceleration"
else
    echo "⚠️ No GPU acceleration available, using CPU"
    echo "💡 Training will be slower but still functional"
    # Optimize CPU usage
    export OMP_NUM_THREADS=$(nproc)
    export MKL_NUM_THREADS=$(nproc)
fi

# Set device type for the training service
export TRAINING_DEVICE_TYPE=$DEVICE_TYPE

# Memory optimization based on available RAM
TOTAL_RAM=$(free -g 2>/dev/null | awk '/^Mem:/{print $2}')
TOTAL_RAM=${TOTAL_RAM:-0}
echo "💾 Total system RAM: ${TOTAL_RAM}GB"

# PYTORCH_CUDA_ALLOC_CONF is only defaulted from the RAM size: compose sets it
# (TRAINING_PYTORCH_CUDA_ALLOC_CONF) and an explicit setting must win.
if [ "$TOTAL_RAM" -lt 16 ]; then
    echo "⚠️ Low RAM detected, enabling aggressive memory optimization"
    : "${PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512}"
    export PYTORCH_CUDA_ALLOC_CONF
    export TRAINING_MEMORY_OPTIMIZATION=aggressive
elif [ "$TOTAL_RAM" -lt 32 ]; then
    echo "✅ Moderate RAM, enabling standard memory optimization"
    : "${PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:1024}"
    export PYTORCH_CUDA_ALLOC_CONF
    export TRAINING_MEMORY_OPTIMIZATION=standard
else
    echo "✅ High RAM, using performance optimization"
    export TRAINING_MEMORY_OPTIMIZATION=performance
fi

# Start the training service.
#
# exec: python becomes this process (PID 1 in the container). Without it bash
# stays PID 1, `docker stop` delivers SIGTERM to bash, which does not forward it,
# and the service never learns it is being stopped: no chance to ask the running
# job to stop, so an update or restart always ended in SIGKILL mid-epoch.
echo "🚀 Starting TTS Training Service with $DEVICE_TYPE acceleration..."
exec python3 app.py
