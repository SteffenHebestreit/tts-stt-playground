# Piper Voice Training Service

Pipeline for creating ONNX voice bundles: upload raw recordings, segment them via STT, prepare mel/phoneme datasets, train a model, resume interrupted jobs, export a runtime bundle, and deploy that bundle to a configured target.

> **Experimental trainer: exported voices are NOT production quality.**
> `/health`, every job status and the startup log carry `trainer_kind` and `trainer_caveat` (from `training_utils.TRAINER_KIND` / `TRAINER_CAVEAT`) so nobody mistakes the result for a finished Piper voice. As it stands, the text encoder is only trained through a duration loss with synthetic targets (a uniform split of the mel length over the tokens); there is no alignment search and no adversarial or prior loss. The bundle turns text into audio whose length follows the text, but the audio is not expected to be intelligible speech. For a usable voice, fine-tune an existing Piper checkpoint instead. Do not claim otherwise in a UI or a report built on this service.

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
| DELETE | `/job/{job_id}` | Cancel a running job. Takes effect at the next epoch boundary. |

### Data Preparation

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/prepare-dataset` | Build a dataset from STT-derived segments (body: `model_name`, `segments`, optional `language`) |
| POST | `/generate-missing-mels` | Regenerate missing mel spectrograms |
| POST | `/restore-backup` | Restore a local backup dataset |

### Export and Cleanup

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/export/{job_id}` | Export a completed checkpoint to ONNX |
| GET | `/deployment-targets` | List configured deployment targets and the default target |
| GET | `/download/{job_id}` | Download the exported model |
| DELETE | `/model/{job_id}` | Remove a trained model and its associated data |
| GET | `/health` | Liveness: cheap, independent of GPU, storage and STT |
| GET | `/ready` | Readiness: 200 when a job can be accepted, 503 with the reasons when the environment is broken |

## Workflow

1. Upload recordings with `/train`, or prepare data first with `/prepare-dataset`.
2. Poll `/status/{job_id}` or `/jobs` while training runs.
3. Resume with `/resume-training` if a container restart interrupts work.
4. Export with `/export/{job_id}` or let the automatic post-training export complete.
5. Deploy automatically to the configured target, or choose manual download only.

### One job at a time

The trainer takes the whole GPU, so the service runs `TRAINING_MAX_CONCURRENT` jobs at once (default `1`). Every way of starting a job (`/train`, `/train-from-dataset`, `/retrain-from-segments`, `/resume-training`) answers **409** while the slot is taken, and the message names the job that holds it. Two jobs never work on the same model name, whatever the limit, because they would write the same dataset and voice. `/generate-missing-mels`, `/prepare-dataset` and `/restore-backup` are refused with 409 for a model a job is training.

The slot is held from the moment a job is accepted until its background task has really finished. That is later than `DELETE /job/{id}` returns: a cancelled job keeps the slot until the training loop reaches its next epoch boundary.

### Job statuses

`initializing` -> `processing` / `transcribing` -> `training` -> `exporting` -> `completed`, or one of `failed`, `cancelled`, `interrupted` (found in `checkpoints/` after a restart).

- `completed` means there is an exported ONNX. If training finished but the **export** failed, the job is `failed` and the message says to retry with `POST /export/{job_id}`; the checkpoint is intact. A failed **deployment** (Piper unreachable) still ends `completed`, because the ONNX exists and can be downloaded.
- The trainer's own "training completed" callback is not shown as `completed`: the runner still exports and deploys.

### Resuming

`POST /resume-training` only resumes a job that is not running. A job that is running (or still stopping after a cancel) is refused with 409, so two threads can never write the same checkpoints. A job that already `completed` is exported, not resumed. When `job_id` is given it must belong to `model_name` (400 otherwise): resuming job A "as" model B used to deploy A's weights under B's name.

### Cancelling

`DELETE /job/{job_id}` is checked once per epoch, so it takes effect at the next
epoch boundary rather than immediately. A cancelled job:

- does **not** produce `final_model.pt`, and is **not** exported or deployed —
  the point of cancelling is that the voice does not go live;
- keeps every periodic checkpoint, so `/resume-training` continues from where it
  stopped;
- ends in status `cancelled`, which is distinct from `failed`, and stays that way:
  no later phase overwrites it. A cancel during STT transcription or dataset
  preparation stops the job at the next phase change.

A job that is not running (for example `interrupted` after a restart) answers 400: there is nothing to stop.

### Deleting

`DELETE /model/{job_id}` always removes that job's checkpoints and exported
model. The dataset under `data/<model_name>` and the deployed voice are keyed on
the model *name*, which several jobs share whenever a voice is retrained — so
those are removed only when no other job trained the same name. The response
lists any job ids that kept them alive in `retained_for_jobs`.

A job that is still running is refused with **409**: cancel it, wait until it has stopped, then delete it.

Names and ids are checked twice. A `model_name` (and any job id in a path) must be 1-64
characters from `[A-Za-z0-9_-]`, the alphabet the Piper runtime and the gateway accept;
anything else is answered with 400 (surrounding whitespace is trimmed). The same rule is
applied again to every name read back from `checkpoints/<job>/job_state.json` before it is
turned into a path, and a directory that would resolve outside `data/`, `checkpoints/` or
`models/` is never removed. A state file with a name outside the alphabet is restored
without a model name, so deleting that job removes its own files but no dataset.

### Where `/prepare-dataset` may read audio from

`audio_path` is client input, so it is restricted:

- a local path must resolve (symlinks followed) to somewhere inside `data/` or a directory listed in `TRAINING_ALLOWED_AUDIO_DIRS`; `..` is rejected; anything else is a 400 before any file is opened;
- an `http(s)` URL is refused unless its host is listed in `TRAINING_ALLOWED_URL_HOSTS` (empty by default). Allowed downloads have a 60 s timeout, are capped at `MAX_UPLOAD_MB`, and do not follow redirects;
- any other scheme (`file:`, `ftp:`, ...) is refused.

Uploaded files (`/train`) land in `data/<model>/temp_uploads/`, which is inside `data/`. Segment text is limited to `MAX_TEXT_CHARS`. Only the requested `start_time`..`end_time` slice of each recording is decoded. The response reports `num_samples` (written), `num_segments_received` and `num_skipped`.

### Language

The dataset language decides the espeak-ng voice used for phonemes, and is passed to the STT service so a short clip is not language-detected on its own. `/train`, `/train-from-dataset`, `/retrain-from-segments` and `/prepare-dataset` (`language` in the body) take it; when a request names none, `TRAINING_DEFAULT_LANGUAGE` (default `de`) applies. Supported: `de en es fr it nl pt ru` (regional tags such as `de-AT` or `en-GB` are accepted). An unsupported language is a 400; it used to be phonemised as English without a word.

A sample whose text cannot be phonemised is dropped and counted, never kept with its raw text as "phonemes". If more than `PHONEMIZER_MAX_FAILURE_FRACTION` of the samples fail, or phonemizer/espeak-ng is missing, the dataset is not built and the job fails with the cause.

### Transcript quality filter

stt-service reports `avg_logprob` per segment, not a confidence. The service uses `exp(avg_logprob)` (the geometric-mean token probability, 0..1) and keeps segments with at least `STT_MIN_CONFIDENCE` (default `0.6`, i.e. `avg_logprob >= -0.51`; Whisper itself treats `-1.0`, or 0.37, as a failed decode). `0` keeps everything. A segment with `avg_logprob: null` (stt-service's answer for -inf/NaN) scores 0. If the STT backend reports neither value, segments are kept and a warning says the filter cannot reject anything.

An unreachable or rejecting STT service fails the job with the reason (`STT service at ... unavailable after 3 attempts`); it no longer ends as "no valid segments". 502/503/504 answers are retried, other error statuses are not.

### Shutdown and updates

Training runs as a task of its own, not as part of a request, so on `SIGTERM` (`docker stop`, an image update, a TrueNAS app upgrade) the service asks every running job to stop and waits up to `TRAINING_SHUTDOWN_GRACE_S` (default 30). The job notices at its next epoch boundary, records `cancelled` in its state and stays resumable from the last periodic checkpoint (every 5 epochs). `start.sh` `exec`s Python so the signal reaches it. Set the container's stop grace period (`stop_grace_period` in compose) above `TRAINING_SHUTDOWN_GRACE_S`; if an epoch is longer than that, the process is killed and the job shows up as `interrupted` after the restart, as it always did.

After a restart, jobs found in `checkpoints/` reappear as `interrupted` (state still `training`), `cancelled` or `failed` (with the recorded error). `POST /resume-training` continues them.

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `CUDA_VISIBLE_DEVICES` | `0` | GPU device index. `start.sh` only supplies `0` when the variable is unset; an explicit value (including an empty one, which hides all GPUs) is kept. |
| `PYTORCH_CUDA_ALLOC_CONF` | by RAM | Allocator settings. `start.sh` only guesses `max_split_size_mb` from the RAM size when unset. |
| `TRAINING_MAX_CONCURRENT` | `1` | Jobs that may run at once; more get a 409 naming the running job. |
| `TRAINING_DEFAULT_LANGUAGE` | `de` | Dataset language when a request names none. A value the service cannot phonemise stops the start. |
| `MAX_UPLOAD_MB` | `500` | Most one `/train` request may upload (all files together), and the most one URL download may serve. Enforced on `Content-Length` and on the running byte count, before the upload is stored. Other request bodies are capped at 16 MB. |
| `MAX_TEXT_CHARS` | `1000` | Longest transcript accepted per segment in `/prepare-dataset`. |
| `TRAINING_ALLOWED_AUDIO_DIRS` | unset | Comma-separated extra directories (inside the container) `/prepare-dataset` may read audio from, besides `data/`. |
| `TRAINING_ALLOWED_URL_HOSTS` | unset | Comma-separated hostnames `/prepare-dataset` may download audio from. Empty: URLs are refused. |
| `STT_MIN_CONFIDENCE` | `0.6` | Lowest `exp(avg_logprob)` a transcript segment needs to be trained on (0..1). |
| `PHONEMIZER_MAX_FAILURE_FRACTION` | `0.05` | Share of samples that may fail phonemisation before the dataset build is abandoned. |
| `TRAINING_SHUTDOWN_GRACE_S` | `30` | How long shutdown waits for running jobs to stop. Keep the container's stop grace period above it. |
| `UVICORN_TIMEOUT_KEEP_ALIVE` | `120` | Idle keep-alive of the HTTP server, so the frontend's pooled connections are not closed under it. Keep it at or above the gateway's `UPSTREAM_KEEPALIVE_EXPIRY` (115). |
| `TRAINING_DEVICE_TYPE` | set by `start.sh` | `cuda`, `hip`, `mps` or `cpu`, as detected at startup. `/ready` answers 503 if it says a GPU but PyTorch runs on the CPU. |
| `STT_SERVICE_URL` | `http://stt-service:8000` | STT backend used for segmentation and transcription. The only source for that address — `/train` and `/retrain-from-segments` accept an `stt_service_url` field but reject any value that does not match this one. |
| `ALLOW_CLIENT_STT_URL` | `false` | Honour a per-request `stt_service_url` instead. Leave off: it lets any caller redirect the service's uploads to a host of their choosing and read the resulting error out of the job status. |
| `PIPER_TTS_SERVICE_URL` | `http://piper-tts-service:5000` | Piper runtime URL used by Piper deployment targets |
| `SHARED_MODELS_DIR` | `/app/shared_models` | Shared model directory used by the `piper-volume` target |
| `DEFAULT_DEPLOYMENT_TARGET` | `piper-volume` | Default deployment target used after export |
| `DEPLOYMENT_TARGETS_JSON` | unset | Optional JSON override for deployment targets and default target |
| `ALLOWED_ORIGINS` | *(empty)* | Comma-separated origins a browser page may call this service from. **Unset or empty: no CORS headers at all.** `*` opens it to every page and must be written down (it logs a warning). Independently of this, a state-changing request (anything but `GET`/`HEAD`/`OPTIONS`) that carries an `Origin` header for another host than the one it was sent to is refused with `403` unless the origin is listed: a multipart `POST` (`/train-from-dataset`, `/resume-training`, `/export/{id}`) needs no CORS preflight, so CORS alone never protected these. Requests without an `Origin` (the gateway, curl) are not affected |
| `ALLOW_CREDENTIALS` | `false` | Enables CORS credentials when origins are explicit |

## Health and readiness

Use `/health` for the container health check: it never touches the GPU, storage or STT. Use `/ready` to find out whether a job can start: it writes a probe file into `data/`, `checkpoints/` and `models/` (an unwritable bind mount is the usual first-run failure on NAS setups), checks that the accelerator `start.sh` found is the one PyTorch is using (a wheel without kernels for the card, for example, leaves it on the CPU), and reports free disk space. A running job does not make it `not_ready`; `accepting_jobs` says whether the slot is free.

## Deployment Targets

The service now treats deployment as a target contract rather than assuming Piper is the only destination.

- `none`: export the ONNX bundle only and leave it available for download
- `piper-volume`: copy the bundle into the shared Piper models volume and refresh voices (copied under a temporary name and renamed, so the runtime never sees a half-written model)
- `piper-http`: upload the bundle through Piper's HTTP API and refresh voices

Training endpoints that eventually export a model accept an optional `deployment_target` form field. If omitted, the service uses `DEFAULT_DEPLOYMENT_TARGET`.

## Implementation Notes

- Uses FP32 training because VITS normalizing-flow layers are unstable in FP16
- Automatically lowers the effective batch size when GPU memory is tight
- Saves checkpoints every 5 epochs and restores job state on startup
- Splits datasets into 90% training and 10% validation by default
- Only `sample_rate=22050` is accepted: the mel features and the model are fixed at that rate
- Blocking work (mel generation, dataset preparation, metadata, ONNX export, deleting large directories) runs in worker threads, so `/health` and `/status` keep answering while it runs

## Requirements

- CUDA or ROCm GPU recommended for practical training times
- STT backend must be reachable for upload-driven training flows
- Clean, single-speaker audio yields the best results; 10+ minutes is a practical minimum
