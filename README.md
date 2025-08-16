# TTS-STT Services

A production-ready Text-to-Speech (TTS), Speech-to-Text (STT), and Voice Cloning service with intelligent hardware acceleration support for CUDA, ROCm, and CPU.

## Features

🚀 **Hardware Acceleration**
- Automatic detection of NVIDIA CUDA, AMD ROCm, and Apple Silicon
- Intelligent fallback to CPU if GPU acceleration fails
- Configurable hardware override options

🎯 **TTS Service (Text-to-Speech)**
- Piper TTS with high-quality voices
- Fast synthesis with low latency
- Streaming audio responses
- Multiple language support

🎭 **Voice Cloning Service**
- OpenVoice integration for voice cloning
- Custom voice creation from audio samples
- Voice synthesis with cloned voices
- Persistent voice storage

🎤 **STT Service (Speech-to-Text)**
- Faster-Whisper for optimized inference
- Multiple model sizes (tiny to large-v3)
- Voice Activity Detection (VAD)
- Transcription and translation support
- 99+ language support

📊 **Production Ready**
- Health checks and monitoring endpoints
- Error handling with graceful fallbacks
- Docker containerization
- Resource optimization
- Comprehensive logging

## Quick Start

### Prerequisites
- Docker and Docker Compose
- (Optional) NVIDIA GPU with CUDA drivers for GPU acceleration
- (Optional) AMD GPU with ROCm drivers for AMD acceleration

### 1. Clone and Setup

```bash
git clone <your-repo>
cd tts-stt
```

### 2. Configure Hardware Acceleration

Edit `.env` file to configure hardware:

```bash
# Automatic detection (recommended)
USE_CUDA=auto

# Force specific acceleration
# USE_CUDA=true         # Force CUDA
# FORCE_ACCELERATION=rocm  # Force AMD ROCm
# FORCE_ACCELERATION=cpu   # Force CPU only
```

### 3. Choose Model Sizes

Configure model sizes based on your needs:

```bash
# TTS: High-quality Piper models 
TTS_LANGUAGE=en

# STT: Choose model size vs accuracy trade-off
WHISPER_MODEL_SIZE=tiny    # Fast, basic accuracy
# WHISPER_MODEL_SIZE=base   # Balanced
# WHISPER_MODEL_SIZE=small  # Good accuracy
# WHISPER_MODEL_SIZE=medium # Better accuracy
# WHISPER_MODEL_SIZE=large-v3 # Best accuracy, slower
```

### 4. Launch Services

```bash
# Build and start services
docker-compose up --build

# Or run in background
docker-compose up --build -d
```

### 5. Verify Services

Check service status:
```bash
# TTS Service health
curl http://localhost:5000/health

# STT Service health  
curl http://localhost:5001/health

# Voice Cloning Service health
curl http://localhost:5002/health

# Detailed service info
curl http://localhost:5000/info
curl http://localhost:5001/info
```

## API Usage

### Text-to-Speech (TTS)

**Endpoint:** `POST http://localhost:5000/synthesize`

**Basic Text Synthesis:**
```bash
curl -X POST "http://localhost:5000/synthesize" \
  -F "text=Hello, this is a test of the text to speech system!" \
  -F "language=en" \
  --output synthesized_audio.wav
```

**Voice Cloning with Speaker Audio:**
```bash
curl -X POST "http://localhost:5000/synthesize" \
  -F "text=Hello, this will sound like the speaker sample!" \
  -F "speaker_wav=@speaker_sample.wav" \
  -F "language=en" \
  --output cloned_voice.wav
```

**Supported Languages:**
- English (en), Spanish (es), French (fr), German (de), Italian (it)
- Portuguese (pt), Polish (pl), Turkish (tr), Russian (ru), Dutch (nl)
- Czech (cs), Arabic (ar), Chinese (zh-cn), Japanese (ja), Hungarian (hu)
- Korean (ko), Hindi (hi) and many more...

### Speech-to-Text (STT)

**Endpoint:** `POST http://localhost:5001/transcribe`

**Basic Transcription:**
```bash
curl -X POST "http://localhost:5001/transcribe" \
  -F "audio=@audio_file.wav" \
  -F "language=en"
```

**Advanced Transcription with Parameters:**
```bash
curl -X POST "http://localhost:5001/transcribe" \
  -F "audio=@audio_file.wav" \
  -F "language=auto" \
  -F "task=transcribe" \
  -F "beam_size=5" \
  -F "best_of=5"
```

**Translation to English:**
```bash
curl -X POST "http://localhost:5001/transcribe" \
  -F "audio=@foreign_audio.wav" \
  -F "task=translate"
```

**Response Format:**
```json
{
  "text": "Transcribed text here",
  "language": "en",
  "language_probability": 0.99,
  "duration": 5.2,
  "segments": [
    {
      "start": 0.0,
      "end": 2.5,
      "text": "First segment",
      "avg_logprob": -0.3,
      "no_speech_prob": 0.1
    }
  ],
  "task": "transcribe"
}
```

## Hardware Acceleration Setup

### NVIDIA CUDA

1. Install NVIDIA drivers and CUDA toolkit
2. Install nvidia-docker2:
```bash
# Ubuntu/Debian
sudo apt-get install nvidia-docker2
sudo systemctl restart docker
```

3. Update docker-compose.yml to enable GPU:
```yaml
services:
  tts-service:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

4. Set environment:
```bash
USE_CUDA=true
FORCE_ACCELERATION=cuda
```

### AMD ROCm

1. Install ROCm drivers
2. Set environment:
```bash
USE_CUDA=false
FORCE_ACCELERATION=rocm
```

### CPU Optimization

For CPU-only deployment:
```bash
USE_CUDA=false
FORCE_ACCELERATION=cpu
```

## Model Information

### TTS Models
- **XTTS v2**: Multilingual model supporting voice cloning
- **Languages**: 50+ languages supported
- **Quality**: High-quality neural synthesis

### STT Models
- **tiny**: ~39 MB, ~32x speed, basic accuracy
- **base**: ~74 MB, ~16x speed, good accuracy  
- **small**: ~244 MB, ~6x speed, better accuracy
- **medium**: ~769 MB, ~2x speed, high accuracy
- **large-v3**: ~1550 MB, ~1x speed, best accuracy

## Performance Tuning

### Memory Optimization
```bash
# Use smaller models for lower memory usage
WHISPER_MODEL_SIZE=tiny

# Limit workers
WORKERS=1
```

### GPU Memory
```bash
# Monitor GPU usage
nvidia-smi

# For multiple GPUs, models will use GPU 0 by default
```

### CPU Optimization
```bash
# Services automatically detect CPU count
# Uses optimized compute types (int8 for CPU)
```

## Monitoring and Debugging

### Health Checks
```bash
# Service status
curl http://localhost:5000/health
curl http://localhost:5001/health

# Detailed info
curl http://localhost:5000/info
curl http://localhost:5001/info

# Available STT models
curl http://localhost:5001/models
```

### Logs
```bash
# View service logs
docker-compose logs tts-service
docker-compose logs stt-service

# Follow logs
docker-compose logs -f
```

### Common Issues

**Model Loading Errors:**
- Check available memory (models require 2-8GB)
- Try smaller model sizes
- Check GPU drivers if using acceleration

**Audio Format Issues:**
- Supported formats: WAV, MP3, M4A, FLAC
- Recommended: WAV, 16kHz, 16-bit

**Performance Issues:**
- Use appropriate model sizes for your hardware
- Monitor GPU/CPU usage
- Consider batch processing for multiple files

## API Documentation

### TTS Endpoints
- `POST /synthesize` - Generate speech from text
- `GET /health` - Service health check
- `GET /info` - Service information

### STT Endpoints  
- `POST /transcribe` - Transcribe audio to text
- `GET /health` - Service health check
- `GET /info` - Service information
- `GET /models` - Available models and languages

## Security Notes

- Services run on localhost by default
- For production deployment, add authentication
- Use reverse proxy for SSL termination
- Validate file uploads and sizes

## Contributing

1. Fork the repository
2. Create feature branch
3. Add tests for new features
4. Submit pull request

## License

MIT License - see LICENSE file for details
