#!/bin/bash
# Device Detection and Setup Script for TTS Training Service
# Supports NVIDIA CUDA, AMD ROCm, Apple Metal, and CPU fallback

echo "🔍 Detecting compute devices..."

# Function to check for NVIDIA CUDA
check_nvidia() {
    if command -v nvidia-smi &> /dev/null; then
        echo "✅ NVIDIA GPU detected:"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits
        export CUDA_VISIBLE_DEVICES=0
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
        export HIP_VISIBLE_DEVICES=0
        export ROC_ENABLE_PRE_VEGA=1
        # Optimizations for Strix Halo AI Max 395
        export HSA_OVERRIDE_GFX_VERSION=11.0.0
        export ROCM_PATH=/opt/rocm
        return 0
    elif lspci | grep -i amd | grep -i vga &> /dev/null; then
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
TOTAL_RAM=$(free -g | awk '/^Mem:/{print $2}')
echo "💾 Total system RAM: ${TOTAL_RAM}GB"

if [ "$TOTAL_RAM" -lt 16 ]; then
    echo "⚠️ Low RAM detected, enabling aggressive memory optimization"
    export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
    export TRAINING_MEMORY_OPTIMIZATION=aggressive
elif [ "$TOTAL_RAM" -lt 32 ]; then
    echo "✅ Moderate RAM, enabling standard memory optimization"
    export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:1024
    export TRAINING_MEMORY_OPTIMIZATION=standard
else
    echo "✅ High RAM, using performance optimization"
    export TRAINING_MEMORY_OPTIMIZATION=performance
fi

# Start the training service
echo "🚀 Starting TTS Training Service with $DEVICE_TYPE acceleration..."
python3 app.py