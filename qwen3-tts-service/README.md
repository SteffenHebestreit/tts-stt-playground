# Qwen3-TTS Service

Text-to-speech, voice cloning, and saved-voice playback using [Qwen3-TTS](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base). This service replaces the older XTTS path and supports both direct cloning and a persistent voice library.

## Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/tts` | Generate speech with a built-in speaker |
| POST | `/clone` | Clone voice from a reference audio file |
| POST | `/clone-with-ref-text` | Clone with explicit transcript for higher quality |
| POST | `/voice_design` | Generate speech from a text-described voice |
| GET | `/voices` | List saved voice profiles |
| POST | `/voices/save` | Save a reusable voice profile |
| POST | `/voices/{voice_id}/tts` | Speak using a saved voice profile |
| DELETE | `/voices/{voice_id}` | Delete a saved voice profile |
| GET | `/models` | List available Qwen3-TTS model variants |
| POST | `/load_model` | Switch to a different model variant |
| GET | `/speakers` | List built-in speakers |
| GET | `/status` | Detailed device and model information |
| GET | `/health` | Health check |

## Model Variants

| Model | Use Case | Approx. VRAM |
|-------|----------|--------------|
| `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | Best general TTS + cloning quality | ~4.5 to 5 GB |
| `Qwen/Qwen3-TTS-12Hz-0.6B-Base` | Lower-VRAM TrueNAS / shared-GPU deployments | ~2.5 to 3 GB |
| `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` | Built-in custom-voice features | ~4.5 to 5 GB |
| `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` | Text-guided voice design | ~4.5 to 5 GB |

## Usage

### Built-in speaker TTS

```bash
curl -X POST "http://localhost:5004/tts" \
	-H "Content-Type: application/json" \
	-d '{"text":"Hello world","lang":"English","speaker":"Vivian"}' \
	--output speech.wav
```

### Voice cloning

```bash
curl -X POST "http://localhost:5004/clone" \
	-F "text=Hello world" \
	-F "lang=English" \
	-F "file=@reference.wav" \
	--output clone.wav
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `QWEN3_TTS_MODEL` | `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | Initial model variant |
| `QWEN3_ASR_SERVICE_URL` | `http://qwen3-asr-service:5002` | ASR service used for auto-transcribing reference audio |
| `VOICES_DIR` | `/app/voices` | Persistent directory for saved voice profiles |
| `ALLOWED_ORIGINS` | `*` | Comma-separated CORS origins |
| `ALLOW_CREDENTIALS` | `false` | Enables CORS credentials when origins are explicit |

## Requirements

- CUDA or ROCm GPU strongly recommended
- CPU mode works, but is not suitable for real-time use
- The service downloads model weights automatically on first start
- Saved voices are persisted in the mounted `/app/voices` volume
