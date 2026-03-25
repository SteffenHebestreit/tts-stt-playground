# Piper Voice Training Service

VITS training pipeline for creating custom Piper-compatible ONNX voices. The service can upload raw recordings, segment them via STT, prepare mel/phoneme datasets, train a model, resume interrupted jobs, and export directly into the shared Piper model volume.

## Endpoints

### Training

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/train` | Upload audio files and start a new training job |
| POST | `/resume-training` | Resume an interrupted job from its latest checkpoint |
| POST | `/train-from-dataset` | Train directly from an existing prepared dataset |
| POST | `/retrain-from-segments` | Re-transcribe existing clips and retrain |
| GET | `/status/{job_id}` | Retrieve progress and current status |
| GET | `/jobs` | List known jobs |
| DELETE | `/job/{job_id}` | Cancel a running job |

### Data Preparation

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/process-audio` | Segment and transcribe a long recording via STT |
| POST | `/prepare-dataset` | Build a dataset from STT-derived segments |
| POST | `/generate-missing-mels` | Regenerate missing mel spectrograms |
| POST | `/restore-backup` | Restore a local backup dataset |

### Export and Cleanup

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/export/{job_id}` | Export a completed checkpoint to ONNX |
| GET | `/download/{job_id}` | Download the exported model |
| DELETE | `/model/{job_id}` | Remove a trained model and its associated data |
| GET | `/health` | Health check |

## Workflow

1. Upload recordings with `/train`, or prepare data first with `/process-audio` and `/prepare-dataset`.
2. Poll `/status/{job_id}` or `/jobs` while training runs.
3. Resume with `/resume-training` if a container restart interrupts work.
4. Export with `/export/{job_id}` or let the automatic post-training export complete.
5. Use the resulting model from the PiperTTS service once it is copied into the shared models volume.

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `CUDA_VISIBLE_DEVICES` | `0` | GPU device index |
| `STT_SERVICE_URL` | `http://stt-service:8000` | STT backend used for segmentation and transcription |
| `ALLOWED_ORIGINS` | `*` | Comma-separated CORS origins |
| `ALLOW_CREDENTIALS` | `false` | Enables CORS credentials when origins are explicit |

## Implementation Notes

- Uses FP32 training because VITS normalizing-flow layers are unstable in FP16
- Automatically lowers the effective batch size when GPU memory is tight
- Saves checkpoints every 5 epochs and restores job state on startup
- Splits datasets into 90% training and 10% validation by default

## Requirements

- CUDA or ROCm GPU recommended for practical training times
- STT backend must be reachable for upload-driven training flows
- Clean, single-speaker audio yields the best results; 10+ minutes is a practical minimum
