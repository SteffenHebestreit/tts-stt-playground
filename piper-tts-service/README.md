# PiperTTS Service

Text-to-speech synthesis using [Piper TTS](https://github.com/rhasspy/piper). The service exposes built-in Piper voices plus custom ONNX models exported by the training service.

## Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/tts` | Generate speech with automatic or explicit voice selection |
| POST | `/synthesize` | Generate speech with a specific custom voice |
| GET | `/voices` | List built-in and custom voices |
| GET | `/voice/{voice_name}` | Get metadata for a specific voice |
| POST | `/upload_model` | Upload a custom ONNX model and optional config |
| DELETE | `/voice/{voice_name}` | Delete a custom voice |
| POST | `/refresh_voices` | Re-scan the custom models directory |
| POST | `/analyze_audio` | Inspect an uploaded audio file via ffprobe |
| GET | `/health` | Health check |

## Usage

### Automatic voice selection

```bash
curl -X POST "http://localhost:5000/tts" \
  -H "Content-Type: application/json" \
  -d '{"text":"Hello world","language":"en_US","quality":"medium"}' \
  --output speech.wav
```

### Specific custom voice

```bash
curl -X POST "http://localhost:5000/synthesize" \
  -H "Content-Type: application/json" \
  -d '{"text":"Hello world","voice_name":"my_voice"}' \
  --output speech.wav
```

### Upload a trained model

```bash
curl -X POST "http://localhost:5000/upload_model" \
  -F "model_file=@my_voice.onnx" \
  -F "config_file=@my_voice.json" \
  -F "model_name=my_voice"
```

Custom voice names are restricted to letters, numbers, `_`, and `-` to avoid path traversal and inconsistent model paths.

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `PIPER_DATA_DIR` | `/app/models` | Directory containing default and custom models |
| `PIPER_OUTPUT_DIR` | `/app/output` | Directory for generated files |
| `ALLOWED_ORIGINS` | `*` | Comma-separated CORS origins |
| `ALLOW_CREDENTIALS` | `false` | Enables CORS credentials when origins are explicit |

Built-in voice models live under `models/default/`; custom voices are stored under `models/custom/{voice_name}/`.
