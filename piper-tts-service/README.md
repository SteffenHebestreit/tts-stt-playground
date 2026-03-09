# PiperTTS Service

Text-to-speech synthesis using [Piper TTS](https://github.com/rhasspy/piper). Provides 40+ pre-trained voices across multiple languages and supports custom ONNX models trained with the Piper Training Service.

## Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/tts` | Generate speech with automatic voice selection |
| POST | `/synthesize` | Generate speech with a specific voice by name |
| GET | `/voices` | List all available voices (built-in and custom) |
| POST | `/upload_model` | Upload a custom ONNX model and config |
| DELETE | `/voice/{voice_name}` | Remove a custom voice |
| GET | `/voice/{voice_name}` | Get details for a specific voice |
| POST | `/refresh_voices` | Rescan and reload the voice list |
| POST | `/analyze_audio` | Analyze audio file properties via ffmpeg |
| GET | `/health` | Health check |

## Usage

### Generate speech

```bash
curl -X POST "http://localhost:5000/tts" \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "language": "en", "quality": "medium"}' \
  --output speech.wav
```

Request fields: `text`, `language` (ISO code), `quality` (low/medium/high), `gender` (male/female), `speed` (0.5–2.0), `output_format` (wav).

### Use a specific voice

```bash
curl -X POST "http://localhost:5000/synthesize" \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "voice": "en_US-lessac-medium", "speed": 1.0}' \
  --output speech.wav
```

### Upload a custom trained model

```bash
curl -X POST "http://localhost:5000/upload_model" \
  -F "model_file=@my_voice.onnx" \
  -F "config_file=@my_voice.json" \
  -F "model_name=my_voice"
```

## Available Languages

Built-in voices cover English (en), German (de), French (fr), Spanish (es), Italian (it), Dutch (nl), and others. Run `GET /voices` for the full list.

## Configuration

Built-in voice models are downloaded automatically from the Piper releases. Custom models are stored in the bind-mounted `models/` directory.

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `PIPER_DATA_DIR` | `/app/models` | Directory for voice model files |
| `PIPER_OUTPUT_DIR` | `/app/output` | Directory for generated audio output |
