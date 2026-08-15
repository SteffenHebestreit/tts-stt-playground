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
| `TTS_MODEL_TTL` / `MODEL_TTL` | `300` | Seconds idle before the weights are released. `-1` pins them resident. A reload restores whichever model was last selected via `/load_model`, not the env default. |
| `TTS_MAX_CONCURRENCY` | `1` | Concurrent generations against the shared model |
| `TTS_MAX_BATCH` | `8` | Sentences generated in one forward pass when a long text is chunked. Peak VRAM scales with it, so it bounds the cost of a long request rather than letting the caller's text length decide. |
| `ALLOWED_ORIGINS` | `*` | Comma-separated CORS origins |
| `ALLOW_CREDENTIALS` | `false` | Enables CORS credentials when origins are explicit |

## Model variants and capabilities

The four variants share one class, so every generation method exists on all of
them and only raises when called on a variant that cannot do the work. The
service therefore routes on the declared capability table rather than probing
with `hasattr`, and answers **400 naming the model to switch to** when the
loaded variant cannot serve a request:

| Endpoint | Requires | Served by |
|---|---|---|
| `/tts` | `custom_voice` | CustomVoice |
| `/clone`, `/clone-with-ref-text`, `/voices/save`, `/voices/{id}/tts` | `voice_clone` | 1.7B Base, 0.6B Base |
| `/voice_design` | `voice_design` | VoiceDesign |

Switch with `POST /load_model`. The choice survives an idle unload.

## Requirements

- CUDA or ROCm GPU strongly recommended
- CPU mode works, but is not suitable for real-time use
- The service downloads model weights automatically on first start
- Saved voices are persisted in the mounted `/app/voices` volume
