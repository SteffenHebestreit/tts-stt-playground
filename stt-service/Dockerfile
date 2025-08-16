FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    build-essential \
    ffmpeg \
    libffi-dev \
    libssl-dev \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Create hardware detection and PyTorch installation script
RUN cat > /app/install_torch.py << 'EOF'
#!/usr/bin/env python3
import subprocess
import sys
import os

def run_command(cmd):
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

def detect_hardware():
    """Detect available hardware acceleration"""
    acceleration = "cpu"
    
    # Check for NVIDIA GPU
    success, stdout, _ = run_command("nvidia-smi")
    if success and "NVIDIA" in stdout:
        print("🚀 NVIDIA GPU detected, installing CUDA support...")
        acceleration = "cuda"
        return acceleration
    
    # Check for AMD GPU (ROCm)
    success, stdout, _ = run_command("rocm-smi")
    if success:
        print("🚀 AMD GPU detected, installing ROCm support...")
        acceleration = "rocm"
        return acceleration
    
    # Check environment variable override
    env_accel = os.getenv("FORCE_ACCELERATION", "").lower()
    if env_accel in ["cuda", "rocm", "cpu"]:
        print(f"🔧 Hardware acceleration forced via environment: {env_accel}")
        return env_accel
    
    print("💻 No GPU detected, using CPU...")
    return acceleration

def install_pytorch(acceleration):
    """Install PyTorch based on detected hardware"""
    if acceleration == "cuda":
        # Install CUDA version (CUDA 12.1 compatible)
        cmd = "pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121"
    elif acceleration == "rocm":
        # Install ROCm version
        cmd = "pip install torch torchaudio --index-url https://download.pytorch.org/whl/rocm5.6"
    else:
        # Install CPU version
        cmd = "pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu"
    
    print(f"Installing PyTorch: {cmd}")
    success, stdout, stderr = run_command(cmd)
    if not success:
        print(f"❌ Failed to install PyTorch: {stderr}")
        # Fallback to CPU if GPU installation fails
        if acceleration != "cpu":
            print("🔄 Falling back to CPU version...")
            return install_pytorch("cpu")
        sys.exit(1)
    print("✅ PyTorch installed successfully")

if __name__ == "__main__":
    acceleration = detect_hardware()
    install_pytorch(acceleration)
EOF

# Copy requirements and install base dependencies
COPY requirements.txt .
RUN pip install --upgrade pip wheel setuptools
RUN pip install --no-cache-dir -r requirements.txt

# Install PyTorch based on hardware
RUN python3 /app/install_torch.py

# Copy application
COPY app.py .

# Create models directory
RUN mkdir -p /app/models && chmod 755 /app/models

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
