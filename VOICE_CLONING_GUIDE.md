# Voice Cloning Implementation Guide

## Overview

This implementation adds instant voice cloning capabilities to your TTS-STT service using OpenVoice technology. You can now clone voices from short audio samples (3-30 seconds) and use them for high-quality speech synthesis.

## Architecture

### Hybrid Approach
- **PiperTTS**: High-quality baseline synthesis (German/English)
- **OpenVoice**: Voice cloning and tone color conversion
- **Separation**: Independent services for specialized functionality

### Services
1. **TTS Service** (Port 5000): PiperTTS for high-quality baseline synthesis
2. **Voice Cloning Service** (Port 5002): Dedicated OpenVoice service for voice cloning
3. **STT Service** (Port 5001): Speech-to-Text (unchanged)

## Features

### Voice Cloning
- ✅ **Instant Voice Cloning**: Clone voices from 3-30 second audio samples
- ✅ **Cross-lingual Support**: Use cloned voices for different languages
- ✅ **Persistent Storage**: Cloned voices saved and reusable
- ✅ **Voice Management**: List, play samples, delete voices
- ✅ **Metadata Storage**: Names, descriptions, creation dates

### Quality Benefits
- 🎯 **High German Quality**: PiperTTS provides excellent German synthesis
- 🎭 **Voice Flexibility**: OpenVoice enables any voice characteristics
- ⚡ **Fast Processing**: Optimized for real-time synthesis
- 🌍 **Multi-language**: Works across supported languages

## Installation & Setup

### Prerequisites
- Docker and Docker Compose
- NVIDIA GPU (recommended for voice cloning)
- At least 4GB free disk space for models

### Quick Start

1. **Build and start services**:
```bash
docker compose build
docker compose up -d
```

2. **Check service status**:
```bash
# Check all services
curl http://localhost:5000/health
curl http://localhost:5002/health

# Check voice cloning availability
curl http://localhost:5000/voice-cloning-status
```

3. **Access the demo**:
   Open `demo.html` in your browser

### Model Downloads
- **PiperTTS models**: Downloaded automatically on first use
- **OpenVoice models**: Downloaded during container startup
- **Storage**: Models cached in `./voice_models/` directory

## API Endpoints

### Voice Cloning

#### Clone a Voice
```bash
POST /clone-voice
Content-Type: multipart/form-data

voice_id: string (required)
name: string (optional)
description: string (optional)  
audio_file: file (required, 3-30 seconds)
```

#### Synthesize with Cloned Voice
```bash
POST /synthesize-cloned
Content-Type: multipart/form-data

text: string (required)
voice_id: string (required)
language: string (default: "en")
```

#### List Cloned Voices
```bash
GET /cloned-voices
```

#### Get Voice Information
```bash
GET /cloned-voices/{voice_id}
```

#### Play Voice Sample
```bash
GET /cloned-voices/{voice_id}/sample
```

#### Delete Voice
```bash
DELETE /cloned-voices/{voice_id}
```

### Service Status
```bash
GET /voice-cloning-status
```

## Usage Examples

### 1. Clone a Voice
```javascript
const formData = new FormData();
formData.append('voice_id', 'my_voice');
formData.append('name', 'My Custom Voice');
formData.append('description', 'High-quality voice clone');
formData.append('audio_file', audioFile);

const response = await fetch('http://localhost:5000/clone-voice', {
    method: 'POST',
    body: formData
});
```

### 2. Synthesize with Cloned Voice
```javascript
const formData = new FormData();
formData.append('text', 'Hello, this is my cloned voice!');
formData.append('voice_id', 'my_voice');
formData.append('language', 'en');

const response = await fetch('http://localhost:5000/synthesize-cloned', {
    method: 'POST',
    body: formData
});

const audioBlob = await response.blob();
```

### 3. List Available Voices
```javascript
const response = await fetch('http://localhost:5000/cloned-voices');
const data = await response.json();
console.log(data.voices);
```

## Best Practices

### Audio Samples for Cloning
- **Duration**: 3-30 seconds (optimal: 10-15 seconds)
- **Quality**: Clear, noise-free recordings
- **Format**: WAV, MP3, or M4A
- **Content**: Natural speech, avoid music/effects
- **Language**: Match the language you plan to synthesize

### Voice Management
- Use descriptive voice IDs
- Add meaningful names and descriptions
- Test voices before important use
- Keep reference samples organized

### Performance Tips
- GPU acceleration significantly improves speed
- Cache frequently used voices
- Use appropriate chunk sizes for long texts
- Monitor disk space for voice storage

## Troubleshooting

### Common Issues

1. **"Voice cloning service unavailable"**
   - Check if voice-cloning-service container is running
   - Verify port 5002 is accessible
   - Check container logs: `docker logs tts-stt-voice-cloning-service-1`

2. **"OpenVoice models not available"**
   - Models downloading on first startup (can take 10+ minutes)
   - Check available disk space
   - Restart container if download failed

3. **Poor cloning quality**
   - Use longer, clearer audio samples
   - Ensure sample matches target language
   - Check for background noise in sample

4. **GPU not detected**
   - Uncomment GPU sections in docker-compose.yml
   - Install NVIDIA Container Toolkit
   - Verify GPU compatibility

### Debug Commands
```bash
# Check service logs
docker logs tts-stt-voice-cloning-service-1
docker logs tts-stt-tts-service-1

# Check service health
curl http://localhost:5000/health
curl http://localhost:5002/health

# List cloned voices
curl http://localhost:5000/cloned-voices

# Test voice cloning status
curl http://localhost:5000/voice-cloning-status
```

## File Structure

```
tts-stt/
├── voice-cloning-service/
│   ├── app.py                 # OpenVoice service
│   ├── requirements.txt       # Python dependencies
│   ├── Dockerfile            # Service container
│   └── download_models.py    # Model downloader
├── tts-service/
│   ├── app.py                # Enhanced TTS with cloning integration
│   └── requirements.txt      # Updated dependencies
├── docker-compose.yml        # Updated with voice cloning service
├── demo.html                 # Updated demo with voice cloning UI
└── VOICE_CLONING_GUIDE.md    # This guide
```

## Performance Metrics

### Typical Performance
- **Voice Cloning**: 30-60 seconds per voice
- **Synthesis Speed**: 2-5x real-time with GPU
- **Model Loading**: 30-120 seconds on startup
- **Memory Usage**: 2-4GB GPU memory, 1-2GB system RAM

### Quality Comparison
- **PiperTTS**: Excellent for German/English baseline
- **OpenVoice**: High-quality voice characteristics transfer
- **Combined**: Best of both worlds

## Future Enhancements

### Planned Features
- [ ] Real-time voice conversion
- [ ] Voice mixing and blending
- [ ] Emotion control for cloned voices
- [ ] Batch voice cloning
- [ ] Voice quality assessment

### Integration Options
- [ ] REST API improvements
- [ ] WebSocket streaming
- [ ] Python SDK
- [ ] Voice analytics dashboard

## Support

### Getting Help
1. Check this guide for common solutions
2. Review container logs for error details
3. Test with provided demo interface
4. Verify service health endpoints

### Resources
- [OpenVoice Documentation](https://github.com/myshell-ai/OpenVoice)
- [PiperTTS Documentation](https://github.com/rhasspy/piper)
- [Docker Compose Reference](https://docs.docker.com/compose/)

---

**Note**: This implementation prioritizes quality and reliability. The hybrid approach ensures you have both excellent baseline quality (PiperTTS) and flexible voice cloning (OpenVoice) capabilities.
