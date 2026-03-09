# Piper Voice Training Service

VITS neural network training pipeline for creating custom Piper TTS voice models. Automatically transcribes uploaded audio via the STT service, prepares training datasets, trains a VITS model, and exports to ONNX format for use with PiperTTS.

## Endpoints

### Training

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/train` | Upload audio files and start a training job |
| POST | `/train-from-dataset` | Start training from an already-prepared dataset |
| POST | `/resume-training` | Resume a training job from a checkpoint |
| POST | `/retrain-from-segments` | Retrain using existing segments |
| GET | `/status/{job_id}` | Training progress and status |
| GET | `/jobs` | List all training jobs |
| DELETE | `/job/{job_id}` | Cancel a running training job |

### Data Processing

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/process-audio` | Transcribe and segment audio via STT service |
| POST | `/prepare-dataset` | Build training dataset from segments |
| POST | `/generate-missing-mels` | Generate mel spectrograms for dataset entries |
| POST | `/restore-backup` | Restore a backed-up dataset |

### Model Export

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/export/{job_id}` | Export trained model to ONNX format |
| GET | `/download/{job_id}` | Download the exported ONNX model |
| DELETE | `/model/{job_id}` | Delete trained model files |

### Utility

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |

## Training Workflow

```bash
# 1. Upload audio and start training (multipart form)
curl -X POST "http://localhost:8080/train" \
  -F "model_name=my_voice" \
  -F "language=de" \
  -F "epochs=1000" \
  -F "audio_files=@recording.wav"

# 2. Poll for progress
curl "http://localhost:8080/status/{job_id}"

# 3. Export to ONNX when training completes
curl -X POST "http://localhost:8080/export/{job_id}"

# 4. Download the model
curl "http://localhost:8080/download/{job_id}" --output model.zip
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `CUDA_VISIBLE_DEVICES` | `0` | GPU device index |
| `STT_SERVICE_URL` | `http://stt-service:8000` | Internal STT service URL |

## Architecture

- **VITS model**: Variational Inference with adversarial learning for end-to-end TTS
- **Training precision**: FP32 (FP16 disabled — VITS normalizing flow overflows FP16 max)
- **Batch size**: Capped at 4 with automatic reduction on GPU OOM
- **Checkpoint interval**: Every 5 epochs
- **Dataset split**: 90% train / 10% validation

## Requirements

- NVIDIA GPU with CUDA support
- STT service running for audio transcription
- Sufficient audio data (10+ minutes of clean speech recommended)
