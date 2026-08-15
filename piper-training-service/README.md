# Piper Voice Training Service

VITS training pipeline for creating ONNX voice bundles. The service can upload raw recordings, segment them via STT, prepare mel/phoneme datasets, train a model, resume interrupted jobs, export a runtime bundle, and deploy that bundle to a configured target.

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
| POST | `/prepare-dataset` | Build a dataset from STT-derived segments |
| POST | `/generate-missing-mels` | Regenerate missing mel spectrograms |
| POST | `/restore-backup` | Restore a local backup dataset |

### Export and Cleanup

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/export/{job_id}` | Export a completed checkpoint to ONNX |
| GET | `/deployment-targets` | List configured deployment targets and the default target |
| GET | `/download/{job_id}` | Download the exported model |
| DELETE | `/model/{job_id}` | Remove a trained model and its associated data |
| GET | `/health` | Health check |

## Workflow

1. Upload recordings with `/train`, or prepare data first with `/prepare-dataset`.
2. Poll `/status/{job_id}` or `/jobs` while training runs.
3. Resume with `/resume-training` if a container restart interrupts work.
4. Export with `/export/{job_id}` or let the automatic post-training export complete.
5. Deploy automatically to the configured target, or choose manual download only.

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `CUDA_VISIBLE_DEVICES` | `0` | GPU device index |
| `STT_SERVICE_URL` | `http://stt-service:8000` | STT backend used for segmentation and transcription. The only source for that address — `/train` and `/retrain-from-segments` accept an `stt_service_url` field but reject any value that does not match this one. |
| `ALLOW_CLIENT_STT_URL` | `false` | Honour a per-request `stt_service_url` instead. Leave off: it lets any caller redirect the service's uploads to a host of their choosing and read the resulting error out of the job status. |
| `PIPER_TTS_SERVICE_URL` | `http://piper-tts-service:5000` | Piper runtime URL used by Piper deployment targets |
| `SHARED_MODELS_DIR` | `/app/shared_models` | Shared model directory used by the `piper-volume` target |
| `DEFAULT_DEPLOYMENT_TARGET` | `piper-volume` | Default deployment target used after export |
| `DEPLOYMENT_TARGETS_JSON` | unset | Optional JSON override for deployment targets and default target |
| `ALLOWED_ORIGINS` | `*` | Comma-separated CORS origins |
| `ALLOW_CREDENTIALS` | `false` | Enables CORS credentials when origins are explicit |

## Deployment Targets

The service now treats deployment as a target contract rather than assuming Piper is the only destination.

- `none`: export the ONNX bundle only and leave it available for download
- `piper-volume`: copy the bundle into the shared Piper models volume and refresh voices
- `piper-http`: upload the bundle through Piper's HTTP API and refresh voices

Training endpoints that eventually export a model accept an optional `deployment_target` form field. If omitted, the service uses `DEFAULT_DEPLOYMENT_TARGET`.

## Implementation Notes

- Uses FP32 training because VITS normalizing-flow layers are unstable in FP16
- Automatically lowers the effective batch size when GPU memory is tight
- Saves checkpoints every 5 epochs and restores job state on startup
- Splits datasets into 90% training and 10% validation by default

## Requirements

- CUDA or ROCm GPU recommended for practical training times
- STT backend must be reachable for upload-driven training flows
- Clean, single-speaker audio yields the best results; 10+ minutes is a practical minimum
